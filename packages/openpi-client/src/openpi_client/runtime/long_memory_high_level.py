"""Current-four-view Sol inference with separate completed memory and active task."""
from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import math
import logging
import os
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from openpi.serving.high_level_archive import HighLevelArchive
from openpi.shared.high_level_index_protocol import INDEX_MEMORY_SCHEMA, decode_index_prediction
from openpi.shared.completed_step_memory import CURRENT_TASKS
from openpi.shared.long_memory import LongMemoryConfig, MemoryState, decode_memory, encode_memory
from openpi.shared.sol_eight_task_prompt import PROMPT_VERSION, SYSTEM_PROMPT

CAMERAS = ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist')
MEMORY_SCHEMA = INDEX_MEMORY_SCHEMA


@dataclass(frozen=True)
class MemoryUpdate:
    state: MemoryState
    archive_directory: str | None = None


def parse_update(payload, *, max_chars=1200):
    state = decode_index_prediction(payload)
    return MemoryUpdate(decode_memory(encode_memory(state), max_chars=max_chars))


def image_uint8(image):
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError('Expected a finite RGB frame')
    if np.issubdtype(a.dtype, np.floating) and a.size and a.max() <= 1 and a.min() >= 0:
        a = a * 255
    return np.clip(a, 0, 255).astype(np.uint8)


def png_url(image):
    buffer = io.BytesIO()
    Image.fromarray(image_uint8(image)).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


class GptMemoryBackend:
    """Archive exact request/images before dispatch; no hidden retries or paid startup test."""
    def __init__(self, *, model='gpt-6.2-sol', api_key=None, base_url='https://api.openai.com/v1',
                 timeout=45, reasoning_effort='low', archive_root=None, cost_from_usage=None, status_publisher=None):
        self.model, self.key = model, api_key or os.environ.get('OPENAI_API_KEY')
        if not self.key:
            raise ValueError('OPENAI_API_KEY is not configured')
        root = archive_root or os.environ.get('OPENPI_HIGH_LEVEL_DATASET_DIR')
        if not root:
            raise ValueError('OPENPI_HIGH_LEVEL_DATASET_DIR is required for durable request retention')
        self.archive = HighLevelArchive(root)
        self.url = base_url.rstrip('/') + '/responses'
        self.timeout, self.reasoning_effort = timeout, reasoning_effort
        self.cost_from_usage = cost_from_usage
        self.status_publisher = status_publisher
        self.sequence, self.sequence_lock = 0, threading.Lock()

    def notify(self, kind, record, directory=None):
        if self.status_publisher is not None:
            try:self.status_publisher.event(kind, record, directory)
            except Exception:logging.warning('High-level display publication failed')

    def reset_status(self, epoch):
        self.notify('reset', {'epoch': epoch})

    def __call__(self, *, task, memory, frames, now_s=None, epoch=0, seq=1,
                 camera_timestamps=None, camera_timestamp_domain=None):
        if not frames:
            raise ValueError('Four current camera images are required')
        captured_s, images = frames[-1]
        if any(c not in images for c in CAMERAS):
            raise ValueError('All four current camera views are required')
        if not isinstance(camera_timestamps, dict) or any(c not in camera_timestamps for c in CAMERAS):
            raise ValueError('Actual timestamps for all four cameras are required')
        timestamps = {c: float(camera_timestamps[c]) for c in CAMERAS}
        if (camera_timestamp_domain not in ('ros_header_seconds', 'unix_seconds')
                or any(not math.isfinite(t) or t <= 0 for t in timestamps.values())
                or max(timestamps.values()) - min(timestamps.values()) > .25):
            raise ValueError('Camera timestamp domain/alignment is invalid')
        content = [{'type': 'input_text' , 'text': json.dumps({
            'overall_goal': task, 'previous_state': json.loads(encode_memory(memory))}, ensure_ascii=False)}]
        for camera in CAMERAS:
            content.extend([{'type': 'input_text', 'text': camera},
                            {'type': 'input_image', 'image_url': png_url(images[camera]), 'detail': 'high'}])
        body = {'model': self.model, 'store': False, 'reasoning': {'effort': self.reasoning_effort},
                'max_output_tokens': 1000,
                'input': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}],
                'text': {'format': {'type': 'json_schema', 'name': 'robot_summary_memory',
                                    'strict': True, 'schema': MEMORY_SCHEMA}}}
        wire = json.dumps(body).encode()
        with self.sequence_lock:
            self.sequence += 1
            request_sequence = self.sequence
        record = {'epoch': epoch, 'seq': request_sequence, 'supervisor_seq': seq, 'requested_unix': time.time(), 'model': self.model,
                  'prompt_version': PROMPT_VERSION, 'previous_state': json.loads(encode_memory(memory)),
                  'camera_timestamps': timestamps, 'timestamp_domain': camera_timestamp_domain,
                  'policy_capture_elapsed_seconds': captured_s,
                  'image_count': 4, 'image_history_seconds': 0,
                  'known_cost_usd': None, 'cost_status': 'not_configured'}
        directory = self.archive.prepare(body, record)  # Must succeed before urlopen.
        self.notify('submitted', record, directory)
        raw = None
        started = time.monotonic()
        try:
            request = urllib.request.Request(self.url, data=wire, headers={
                'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                raw = error.read()
                record['http_status'] = error.code
                raise
            result = json.loads(raw)
            self.last_usage = result.get('usage', {})
            self.last_response_id, self.last_model = result.get('id'), result.get('model', self.model)
            record.update(response_id=self.last_response_id, returned_model=self.last_model,
                          usage=self.last_usage, returned_service_tier=result.get('service_tier'))
            if self.cost_from_usage is not None:
                record.update(known_cost_usd=self.cost_from_usage(self.last_usage), cost_status='computed')
            if result.get('status') != 'completed':
                raise RuntimeError('Incomplete high-level reply')
            text = ''.join(b['text'] for item in result.get('output', [])
                           for b in item.get('content', []) if b.get('type') == 'output_text')
            payload = json.loads(text)
            update = parse_update(payload)
            record.update(status='completed', prediction=payload)
        except Exception as error:
            record.update(status='error', error_type=type(error).__name__)
            self.archive.outcome(directory, {'applied': False, 'reason': 'request_or_parse_failed',
                                             'decided_unix': time.time()})
            raise
        finally:
            record.update(returned_unix=time.time(), latency_seconds=time.monotonic() - started)
            self.archive.finish(directory, record, raw)
            self.notify('finished', record, directory)
        return MemoryUpdate(update.state, directory)

    def record_outcome(self, update, *, applied, reason):
        if update.archive_directory:
            self.archive.outcome(update.archive_directory, {'applied': applied, 'reason': reason,
                                                           'decided_unix': time.time()})
            self.notify('outcome', {'applied': applied, 'reason': reason}, update.archive_directory)


class MemorySupervisor:
    """GPT returns the complete next state. Never append, infer or clear bottle memory locally."""
    def __init__(self, backend, *, settings=LongMemoryConfig(), cameras=CAMERAS):
        self.backend, self.settings, self.cameras = backend, settings, cameras
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-summary')
        self.future, self.seq, self.closed = None, 0, False
        self.archive_error = None
        self.reset()

    def _outcome(self, update, applied, reason):
        if hasattr(self.backend, 'record_outcome'):
            self.backend.record_outcome(update, applied=applied, reason=reason)

    def reset(self, *, reason='episode_reset'):
        with self.lock:
            self.generation = getattr(self, 'generation', 0) + 1
            previous = self.future
            self.future = None
            if previous is not None:
                previous.cancel()
                def discard(future):
                    try:
                        _, _, update, _ = future.result()
                    except Exception:
                        # Real backend failures already persist their discarded/error outcome.
                        return
                    try:
                        self._outcome(update, False, reason)
                    except Exception as error:
                        self.archive_error = type(error).__name__
                previous.add_done_callback(discard)
            self.memory, self.frames = MemoryState(), deque()
            self.frame_timestamps, self.frame_timestamp_domain = None, None
            self.last_capture = self.last_request = self.last_observed = -math.inf
            self.error = None
            if hasattr(self.backend, 'reset_status'):self.backend.reset_status(self.generation)

    def _poll(self):
        with self.lock:
            if self.future is None or not self.future.done():
                return
            future, self.future = self.future, None
            try:
                generation, observed, update, elapsed = future.result()
                if not isinstance(update, MemoryUpdate):
                    raise ValueError('Backend must return a MemoryUpdate')
                # Validate before journaling an application or changing any state.
                index = CURRENT_TASKS.index(update.state.current_task) + 1
                state = parse_update({'bottle_description': update.state.bottle_description,
                                      'memory': update.state.memory, 'current_task_index': index},
                                     max_chars=self.settings.max_memory_chars).state
                apply = generation == self.generation and not self.closed
                self._outcome(update, apply, 'applied' if apply else 'episode_reset_or_closed')
                if apply:
                    # Index already decoded to the exact training text, fields still separate.
                    self.memory = state
                    self.last_observed, self.latency_seconds, self.error = observed, elapsed, None
            except Exception as error:
                self.error = type(error).__name__

    def observe(self, images, *, now_s, task, camera_timestamps=None, camera_timestamp_domain=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('High-level supervisor is closed')
            if self.archive_error:
                raise RuntimeError('High-level archive failure: ' + self.archive_error)
            if not math.isfinite(now_s) or now_s < 0 or now_s < self.last_capture:
                raise ValueError('Reset at episode start')
            self._poll()
            if now_s - self.last_capture >= 1 / self.settings.video_hz - 1e-6:
                self.frames.clear()  # The reviewed prompt consumes one current four-view observation.
                self.frames.append((now_s, {c: image_uint8(images[c]).copy() for c in self.cameras}))
                self.last_capture = now_s
                self.frame_timestamps = dict(camera_timestamps) if camera_timestamps is not None else None
                self.frame_timestamp_domain = camera_timestamp_domain
            if self.future is None and now_s - self.last_request >= self.settings.update_seconds:
                self.seq += 1
                generation, previous, seq = self.generation, self.memory, self.seq
                selected, submitted = list(self.frames), time.monotonic()
                timestamps, timestamp_domain = self.frame_timestamps, self.frame_timestamp_domain
                def update():
                    if generation != self.generation or self.closed:
                        raise CancelledError()
                    value = self.backend(task=task, memory=previous, frames=selected,
                                         now_s=now_s, epoch=generation, seq=seq,
                                         camera_timestamps=timestamps, camera_timestamp_domain=timestamp_domain)
                    return generation, now_s, value, time.monotonic() - submitted
                self.last_request = now_s
                self.future = self.executor.submit(update)
            return encode_memory(self.memory)

    def ready_memory(self, *, now_s):
        self._poll()
        if self.error or self.archive_error:
            raise RuntimeError("High-level result failed: " + str(self.error or self.archive_error))
        if not self.memory.current_task:
            raise RuntimeError('High-level result unavailable; no action generated')
        return encode_memory(self.memory)

    def close(self):
        with self.lock:
            self.closed = True
            self.reset(reason='supervisor_closed')
        self.executor.shutdown(wait=False, cancel_futures=True)


class LongMemoryPolicy:
    def __init__(self, low_level_policy, supervisor, *, first_result_timeout=45.):
        self.low_level_policy, self.supervisor = low_level_policy, supervisor
        self.first_result_timeout = first_result_timeout
        self.started = time.monotonic()
        self.task = None

    def reset(self):
        self.supervisor.reset()
        self.started, self.task = time.monotonic(), None
        if hasattr(self.low_level_policy, 'reset'):
            self.low_level_policy.reset()

    def infer(self, observation, *args, **kwargs):
        task = observation.get('task', observation.get('prompt'))
        if not isinstance(task, str):
            raise ValueError('An overall task is required')
        if self.task is not None and task != self.task:
            self.reset()
        self.task = task
        self.supervisor.observe(observation['images'], now_s=time.monotonic() - self.started, task=task,
                                camera_timestamps=observation.get('camera_timestamps'),
                                camera_timestamp_domain=observation.get('camera_timestamp_domain'))
        deadline = time.monotonic() + self.first_result_timeout
        waiting_published = False
        while True:
            self.supervisor._poll()
            error = self.supervisor.archive_error or self.supervisor.error
            notify = getattr(self.supervisor.backend, 'notify', lambda *args: None)
            if error:
                notify('error', {'error': str(error)})
                raise RuntimeError('High-level request failed: ' + str(error))
            now_s = time.monotonic() - self.started
            if self.supervisor.memory.current_task:
                memory = self.supervisor.ready_memory(now_s=now_s)
                break
            # Wait only for the first result after an explicit reset. A completed
            # current-episode state remains usable while the next request runs.
            if time.monotonic() >= deadline:
                notify('error', {'error': 'Initial high-level result timed out; no action generated'})
                raise RuntimeError('Initial high-level result timed out; no action generated')
            if not waiting_published:
                notify('waiting', {})
                waiting_published = True
            time.sleep(.005)
        # Camera timing is high-level archive provenance, never a model feature.
        # Preserve the original observation for the supervisor and its archive.
        low_level_observation = {key: value for key, value in observation.items()
                                 if key not in ('camera_timestamps', 'camera_timestamp_domain', 'origin_observation')}
        low_level_observation['long_memory'] = memory
        return self.low_level_policy.infer(low_level_observation, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.low_level_policy, name)

"""Run the live Sol observer on an independent clock during a robot episode."""
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import json
import logging
import math
import threading
import time
import urllib.request

import numpy as np
from PIL import Image
from openpi_client.runtime.long_memory_high_level import CAMERAS, MemoryUpdate
from openpi.shared.long_memory import MemoryState, decode_memory, encode_memory

DEADLINE = 4.0
OBSERVATION_LEASE = 8.0
CAMERA_URL = 'http://127.0.0.1:8011/api/high-level/observation'


class ArchiveWriteError(RuntimeError):
    pass


class StrictArchive:
    def __init__(self, archive):
        self.archive = archive

    def __getattr__(self, name):
        method = getattr(self.archive, name)
        if name not in ('prepare', 'finish', 'outcome'):
            return method

        def write(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as error:
                raise ArchiveWriteError(name) from error
        return write


def camera_snapshot():
    with urllib.request.urlopen(CAMERA_URL, timeout=.7) as response:
        value = json.load(response)
    stamps = {name: float(value['camera_timestamps'][name]) for name in CAMERAS}
    now = time.time()
    if any(not math.isfinite(t) or not -.1 <= now - t <= 1. for t in stamps.values()):
        raise ValueError('Camera frames are stale')
    if max(stamps.values()) - min(stamps.values()) > .25:
        raise ValueError('Camera frames are out of sync')
    pixels = {}
    for name in CAMERAS:
        with Image.open(io.BytesIO(base64.b64decode(value['camera_png_b64'][name], validate=True))) as frame:
            pixels[name] = np.asarray(frame.convert('RGB'))
    return pixels, stamps


class PeriodicSolSupervisor:
    """Keep the last accepted state and journal late or superseded replies."""

    def __init__(self, backend, *, settings, interval_seconds, snapshot=camera_snapshot):
        self.backend, self.settings, self.snapshot = backend, settings, snapshot
        self.interval_seconds = interval_seconds
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='sol-request')
        self.inflight = set()
        self.seq = self.generation = 0
        self.closed = False
        self.archive_error = self.error = None
        self.reset()
        self.scheduler = threading.Thread(target=self._schedule, name='sol-tick', daemon=True)
        self.scheduler.start()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.episode_started = time.monotonic()
            self.memory = MemoryState()
            self.task = None
            self.lease_until = self.next_tick = 0.
            self.applied_seq = 0
            self.last_frame = -math.inf
            self.error = None
            self.backend.reset_status(self.generation)
            self.wake.set()

    def observe(self, images, *, now_s, task, **provenance):
        with self.lock:
            if self.closed or self.archive_error:
                raise RuntimeError(f'Sol supervisor unavailable: {self.archive_error}')
            now = time.monotonic()
            if self.task is None or now >= self.lease_until:
                self.next_tick = now
                self.wake.set()
            self.task = task
            self.lease_until = now + OBSERVATION_LEASE

    def _poll(self):
        # Worker completion is handled independently of low-level inference RPCs.
        pass

    def ready_memory(self, *, now_s):
        with self.lock:
            if self.archive_error or not self.memory.current_task:
                raise RuntimeError('High-level result unavailable; no action generated')
            return encode_memory(self.memory)

    def _schedule(self):
        while True:
            self.wake.clear()
            delay = None
            with self.lock:
                if self.closed:
                    return
                now = time.monotonic()
                if self.task is not None and not self.archive_error:
                    if now >= self.lease_until:
                        self.reset()
                    elif now >= self.next_tick:
                        tick = self.next_tick
                        interval = self.interval_seconds
                        self.next_tick = tick + (math.floor((now - tick) / interval) + 1) * interval
                        if len(self.inflight) < 8:
                            self.seq += 1
                            generation, seq = self.generation, self.seq
                            job = self.executor.submit(self._request, generation, seq, self.task, now)
                            self.inflight.add(job)
                            job.add_done_callback(self._finished)
                    if self.task is not None:
                        delay = max(0., min(self.lease_until, self.next_tick) - time.monotonic())
            self.wake.wait(delay)

    def _request(self, generation, seq, task, started):
        try:
            frames, timestamps = self.snapshot()
        except Exception:
            logging.warning('Skipped Sol tick: no fresh synchronized camera snapshot', exc_info=True)
            return
        with self.lock:
            now = time.monotonic()
            newest = min(timestamps.values())
            if (self.closed or self.archive_error or generation != self.generation
                    or now >= self.lease_until or now - started > DEADLINE or newest <= self.last_frame):
                return
            self.last_frame = newest
            memory = self.memory
            episode_elapsed = started - self.episode_started
        result = self.backend(task=task, memory=memory, frames=[(episode_elapsed, frames)],
                              epoch=generation, seq=seq, camera_timestamps=timestamps,
                              camera_timestamp_domain='unix_seconds')
        return generation, seq, result, time.monotonic() - started

    def _finished(self, future):
        with self.lock:
            self.inflight.discard(future)
            if future.cancelled():
                return
            try:
                reply = future.result()
            except Exception as error:
                # The production backend journals HTTP and parse failures itself.
                if isinstance(error, ArchiveWriteError):
                    self.archive_error = type(error).__name__
                    logging.exception('Sol archive or local I/O failed')
                else:
                    logging.warning('Sol request failed: %s', type(error).__name__)
                return
            if reply is None:
                return
            generation, seq, update, elapsed = reply
            if not isinstance(update, MemoryUpdate):
                self.archive_error = 'InvalidMemoryUpdate'
                return
            if elapsed > DEADLINE:
                reason = 'reply_over_4_seconds'
            elif generation != self.generation or self.closed or self.task is None or time.monotonic() >= self.lease_until:
                reason = 'episode_reset_or_closed'
            elif seq <= self.applied_seq:
                reason = 'superseded_by_newer_reply'
            else:
                reason = 'applied'
            try:
                state = decode_memory(encode_memory(update.state), max_chars=self.settings.max_memory_chars)
                self.backend.record_outcome(update, applied=reason == 'applied', reason=reason)
            except Exception as error:
                self.archive_error = type(error).__name__
                logging.exception('Could not commit Sol application outcome')
                return
            if reason == 'applied':
                self.memory = state
                self.applied_seq = seq

    def close(self):
        with self.lock:
            self.closed = True
            self.reset()
        self.executor.shutdown(wait=False, cancel_futures=True)

"""Read-only timing probes for the running deployment; no control changes."""
from collections import defaultdict
from functools import wraps
import json
import logging
import threading
import time


def install():
    from openpi.robot.aloha_real import real_env, runtime, chunked_policy, robot_utils
    from openpi.robot.aloha_real.policy_recording import PolicyRecording
    from openpi.robot.aloha_real.video_memory_preview import VideoMemoryPreview

    local = threading.local()
    rpc_lock = threading.Lock()
    rpc_samples = []
    window = defaultdict(list)
    last_start = None
    next_report = time.perf_counter() + 2.0

    def timed(obj, name, label):
        original = getattr(obj, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                measurements = getattr(local, 'measurements', None)
                if measurements is not None:
                    key = label(*args, **kwargs) if callable(label) else label
                    measurements[key] += (time.perf_counter() - started) * 1000

        setattr(obj, name, wrapped)

    timed(robot_utils.ImageRecorder, '_select_history_frames',
          lambda self, cam_name: 'history_stack_' + cam_name)
    timed(robot_utils.ImageRecorder, 'get_latest_images', 'raw_images_total')
    from openpi.robot.aloha_real import mosaic_runtime
    timed(mosaic_runtime, '_pack_prepared', 'camera_mosaic')
    timed(robot_utils.ImageRecorder, 'get_images', 'images_total')
    timed(real_env.AlohaRealEnvironment, 'get_observation', 'observation_copy')
    timed(real_env.AlohaRealEnvironment, 'apply_action', 'apply_action_including_next_observation')
    timed(real_env.RealEnv, 'get_observation', 'next_observation')
    for name in ('get_qpos', 'get_qvel', 'get_effort'):
        timed(real_env.RealEnv, name, name)
    timed(chunked_policy.ChunkedPolicy, 'infer', 'policy_action_including_handoff_wait')
    timed(VideoMemoryPreview, 'submit', 'preview_submit')
    timed(PolicyRecording, 'append', 'recording_append')
    timed(runtime.Runtime, '_publish_runtime_state', 'publish_state')
    timed(runtime.Runtime, '_apply_pending_inference_settings', 'pending_settings')

    original_init = runtime.Runtime.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        client = self._policy._policy
        original_infer = client.infer

        @wraps(original_infer)
        def infer(*a, **kw):
            started = time.perf_counter()
            try:
                return original_infer(*a, **kw)
            finally:
                with rpc_lock:
                    rpc_samples.append((time.perf_counter() - started) * 1000)

        client.infer = infer

    runtime.Runtime.__init__ = init
    original_step = runtime.Runtime._step

    def summary(values):
        values = sorted(values)
        return {'n': len(values), 'avg': round(sum(values) / len(values), 3),
                'p95': round(values[int(.95 * (len(values) - 1))], 3),
                'max': round(values[-1], 3)}

    @wraps(original_step)
    def step(self):
        nonlocal last_start, next_report
        started = time.perf_counter()
        local.measurements = defaultdict(float)
        try:
            return original_step(self)
        finally:
            done = time.perf_counter()
            measurements = local.measurements
            local.measurements = None
            measurements['step_total'] = (done - started) * 1000
            stacked = sum(value for key, value in measurements.items() if key.startswith('history_stack_'))
            measurements['camera_wait_and_read'] = max(0, measurements.get('raw_images_total', 0) - stacked)
            measurements['action_send_and_other'] = max(
                0, measurements.get('apply_action_including_next_observation', 0)
                - measurements.get('next_observation', 0))
            if last_start is not None:
                measurements['loop_interval'] = (started - last_start) * 1000
            last_start = started
            for key, value in measurements.items():
                window[key].append(value)
            if done >= next_report:
                with rpc_lock:
                    rpc = list(rpc_samples)
                    rpc_samples.clear()
                report = {key: summary(values) for key, values in window.items()}
                if rpc:
                    report['model_rpc_async'] = summary(rpc)
                logging.info('LATENCY_MS %s', json.dumps(report, separators=(',', ':')))
                window.clear()
                next_report = time.perf_counter() + 2.0

    runtime.Runtime._step = step
    logging.info('LATENCY_DIAGNOSTICS installed; summaries every 2s; nested timings must not be added together')

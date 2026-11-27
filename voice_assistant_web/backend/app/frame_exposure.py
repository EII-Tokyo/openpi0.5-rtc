"""Bounded latest frame-metadata cache. Never opens devices or touches the SDK."""
import json
import math
import threading
import time

CAMERAS = ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist')


class FrameExposureCache:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.latest = {}

    def update(self, payload):
        try:
            if len(payload) > 16384:
                return
            data = json.loads(payload)
            if data.get('version') != 1 or not isinstance(data.get('cameras'), dict):
                return
            received = self.clock()
            samples = {}
            for name in CAMERAS:
                value = data['cameras'].get(name)
                if not isinstance(value, dict) or value.get('source') != 'frame_metadata':
                    continue
                stamp = value.get('received_unix')
                if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp > received + 1:
                    continue
                exposure = value.get('actual_exposure_ms')
                if exposure is not None and (isinstance(exposure, bool) or not isinstance(exposure, (int, float)) or not math.isfinite(exposure) or not 0 < exposure <= 10000):
                    exposure = None
                frame = value.get('frame_number')
                if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                    frame = None
                samples[name] = {'actual_exposure_ms': exposure, 'frame_number': frame,
                                 'received_unix': stamp, 'source': 'frame_metadata',
                                 'auto_exposure': value.get('auto_exposure') if isinstance(value.get('auto_exposure'), bool) else None,
                                 'error': str(value.get('error') or '')[:160] or None}
            with self.lock:
                for name, value in samples.items():
                    if value['received_unix'] >= self.latest.get(name, {}).get('received_unix', 0):
                        self.latest[name] = value
        except (ValueError, TypeError, AttributeError):
            return

    def snapshot(self):
        now = self.clock()
        with self.lock:
            cameras = {name: dict(value) for name, value in self.latest.items()}
        for value in cameras.values():
            value['age_seconds'] = max(0, now - value['received_unix'])
            value['fresh'] = value['age_seconds'] <= 2
        return {'cameras': cameras, 'server_unix': now, 'source': 'frame_metadata',
                'stale_after_seconds': 2}

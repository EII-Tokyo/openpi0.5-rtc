"""Prepare camera tiles on callback threads using the checkpoint input contract."""
from collections import deque
import time
import numpy as np
from openpi.data.temporal_mosaic import HighCameraMosaicConfig, resize_pad
from openpi.robot.aloha_real import robot_utils

_original_init = robot_utils.ImageRecorder.__init__
_original_callback = robot_utils.ImageRecorder.image_cb
_video_input = None


def _init(self, *args, **kwargs):
    self._mosaic_settings = HighCameraMosaicConfig(**_video_input["mosaic"])
    self._video_camera_keys = tuple(_video_input["camera_keys"])
    self._video_resolution = tuple(_video_input["image_resolution"])
    self._prepared_history = {name: deque() for name in self._mosaic_settings.cameras}
    self._prepared_latest = {}
    self._prepared_received = {}
    _original_init(self, *args, **kwargs)


def _callback(self, name, message):
    _original_callback(self, name, message)
    if name not in self._video_camera_keys:
        return
    with self._image_lock:
        raw = getattr(self, f"{name}_rgb_image")
        stamp = getattr(self, f"{name}_timestamp")
    full = resize_pad(raw, self._video_resolution[0])
    tile = resize_pad(raw, self._mosaic_settings.tile_size) if name in self._prepared_history else None
    with self._image_lock:
        self._prepared_latest[name] = full
        self._prepared_received[name] = time.monotonic()
        if tile is not None:
            history = self._prepared_history[name]
            history.append((stamp, tile, full))
            while history and stamp - history[0][0] > self.history_window_seconds:
                history.popleft()


def _pack_prepared(history, settings):
    if not history:
        raise RuntimeError("No prepared camera history")
    stamps = np.asarray([item[0] for item in history])
    targets = stamps[-1] + settings.offsets_seconds
    right = np.clip(np.searchsorted(stamps, targets), 0, len(stamps) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(stamps[left] - targets) <= np.abs(stamps[right] - targets), left, right)
    tiles = [history[int(i)][1] for i in nearest]
    mosaics = []
    for start in range(0, len(tiles), settings.sample_hz):
        block = tiles[start:start + settings.sample_hz]
        mosaics.append(np.concatenate((np.concatenate(block[:2], axis=1),
                                        np.concatenate(block[2:], axis=1)), axis=0))
    mosaics.append(history[-1][2])
    return np.stack(mosaics)


def mosaic_images(self):
    self.get_latest_images()
    with self._image_lock:
        now = time.monotonic()
        for name in self._video_camera_keys:
            if now - self._prepared_received.get(name, 0.0) > 0.5:
                raise RuntimeError(f"Prepared camera {name} is missing or stale")
        histories = {name: list(history) for name, history in self._prepared_history.items()}
        result = dict(self._prepared_latest)
    for name, history in histories.items():
        result[name] = _pack_prepared(history, self._mosaic_settings)
    for name in self.camera_names:
        result[f"{name}_depth"] = None
    return result


def install(video_input):
    global _video_input
    if getattr(robot_utils.ImageRecorder, '_prepared_mosaic_installed', False):
        if _video_input != video_input:
            raise RuntimeError("Cannot change checkpoint video settings on an existing recorder; restart runtime")
        return
    _video_input = dict(video_input)
    robot_utils.ImageRecorder.__init__ = _init
    robot_utils.ImageRecorder.image_cb = _callback
    robot_utils.ImageRecorder.get_images = mosaic_images
    robot_utils.ImageRecorder._prepared_mosaic_installed = True

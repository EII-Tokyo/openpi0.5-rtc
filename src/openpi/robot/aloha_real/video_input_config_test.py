"""Offline contract/pixel checks; no ROS subscriptions or robot commands."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from openpi.data.temporal_mosaic import HighCameraMosaicConfig, pack_high_history, resize_pad
from openpi.robot.aloha_real.video_input_config import configure_video_input

CAMERAS = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']


def contract(tile_size=112, seconds=5, cameras=('cam_high',)):
    settings = HighCameraMosaicConfig(history_seconds=seconds, tile_size=tile_size, cameras=cameras)
    return dict(config_name='test', checkpoint_step='14999', camera_keys=CAMERAS,
                image_resolution=[2*tile_size]*2, num_frames=seconds+1, stride_seconds=1.0,
                mosaic=dataclasses.asdict(settings))


class VideoInputTest(unittest.TestCase):
    def test_legacy_high_camera_snapshot_remains_serializable(self):
        legacy = HighCameraMosaicConfig(camera='cam_high')
        self.assertEqual(legacy.cameras, ('cam_high',))
        self.assertNotIn('camera', dataclasses.asdict(legacy))
        explicit = HighCameraMosaicConfig(cameras=tuple(CAMERAS), history_seconds=1)
        self.assertEqual(dataclasses.asdict(explicit)['cameras'], tuple(CAMERAS))
        with self.assertRaises(ValueError):
            HighCameraMosaicConfig(camera='cam_low')

    def setUp(self):
        fake = types.ModuleType('openpi.robot.aloha_real.robot_utils')
        class Recorder:
            def __init__(self, *args, **kwargs):
                self.camera_names = CAMERAS
            def image_cb(self, *args):
                pass
        fake.ImageRecorder = Recorder
        name = 'openpi.robot.aloha_real.mosaic_runtime'
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('mosaic_runtime.py'))
        self.mosaic = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'openpi.robot.aloha_real.robot_utils': fake}):
            spec.loader.exec_module(self.mosaic)
        self.fake = fake

    def test_pixels_match_training_across_models(self):
        rng = np.random.default_rng(42)
        for tile, seconds in ((112, 5), (224, 5), (224, 1)):
            settings = HighCameraMosaicConfig(cameras=('cam_high',), tile_size=tile, history_seconds=seconds)
            for count in (1, 7, 311):
                raw = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(count)]
                prepared = [(i/50, resize_pad(im, tile), resize_pad(im, tile*2)) for i, im in enumerate(raw)]
                stamps = np.array([p[0] for p in prepared])
                # Training dataset's timestamp-based nearest-frame rule.
                targets = stamps[-1] + settings.offsets_seconds
                right = np.clip(np.searchsorted(stamps, targets), 0, count-1)
                left = np.maximum(right-1, 0)
                idx = np.where(abs(stamps[left]-targets) <= abs(stamps[right]-targets), left, right)
                expected = pack_high_history([raw[int(i)] for i in idx], raw[-1], settings)
                actual = self.mosaic._pack_prepared(prepared, settings)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (seconds+1, tile*2, tile*2, 3))

    def test_all_camera_mosaic_and_single_frame(self):
        value = contract(224, 1, tuple(CAMERAS))
        parsed = configure_video_input({'runtime': {'video_input': value}}, '/missing/14999')
        self.mosaic.install(parsed)
        recorder = self.fake.ImageRecorder()
        self.assertEqual(set(recorder._prepared_history), set(CAMERAS))
        self.assertEqual(recorder._video_resolution, (448, 448))
        self.mosaic.install(parsed)
        with self.assertRaises(RuntimeError):
            self.mosaic.install(contract())
        value.update(mosaic=None, num_frames=1)
        self.assertIsNone(configure_video_input({'runtime': {'video_input': value}}, '/missing/14999')['mosaic'])

    def test_rejects_missing_or_incompatible_contract(self):
        with self.assertRaises(ValueError):
            configure_video_input({}, '/missing/14999')
        value = contract()
        with self.assertRaises(ValueError):
            configure_video_input({'runtime': {'video_input': value}}, '/missing/19000')
        for change in ({'image_resolution': [448,448]}, {'num_frames': 2}, {'camera_keys': ['cam_low']}):
            with self.assertRaises(ValueError):
                configure_video_input({'runtime': {'video_input': {**value, **change}}}, '/missing/14999')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'14999'
            root.mkdir()
            (root/'manifest.json').write_text(json.dumps({'config': 'other'}))
            with self.assertRaises(ValueError):
                configure_video_input({'runtime': {'video_input': value}}, str(root))


if __name__ == '__main__':
    unittest.main()

import types

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.data.datasets import HighCameraMosaicDataset, TemporalFrameStackDataset
from openpi.data.temporal_mosaic import HighCameraMosaicConfig, pack_high_history

CAMERAS = ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist')


class FakeDataset:
    def __init__(self):
        self.reads = []

    def __len__(self):
        return 330

    def __getitem__(self, index):
        self.reads.append(index)
        start = 0 if index < 310 else 310
        result = {'episode_index': int(start > 0), 'frame_index': index - start,
                  'timestamp': (index - start) / 50, 'action': np.full((50, 14), index),
                  'observation.state': np.full(14, index)}
        for camera in CAMERAS:
            result['observation.images.' + camera] = np.full((8, 8, 3), (index + CAMERAS.index(camera)) % 255, np.uint8)
        return result


def test_quadrants_oldest_first_and_current_full_size():
    settings = HighCameraMosaicConfig(tile_size=4)
    history = [np.full((8, 8, 3), index, np.uint8) for index in range(20)]
    result = pack_high_history(history, np.full((8, 8, 3), 99, np.uint8), settings)
    assert result.shape == (6, 8, 8, 3)
    for group in range(5):
        for tile, (y, x) in enumerate(((0, 0), (0, 4), (4, 0), (4, 4))):
            assert np.all(result[group, y:y + 4, x:x + 4] == group * 4 + tile)
    assert np.all(result[-1] == 99)


def test_real_tile_resolution_and_chw_float():
    settings = HighCameraMosaicConfig()
    history = [np.full((3, 32, 48), i / 255, np.float32) for i in range(20)]
    result = pack_high_history(history, np.full((3, 448, 448), .5, np.float32), settings)
    assert result.dtype == np.uint8 and result.shape == (6, 448, 448, 3)
    assert np.all(result[-1] == 128)


def test_individual_timestamp_rounding_does_not_drift():
    ds = HighCameraMosaicDataset(FakeDataset(), fps=50, settings=HighCameraMosaicConfig(tile_size=4))
    indices = ds.history_indices(300)
    targets = 6 + ds.settings.offsets_seconds
    assert np.max(np.abs(ds._timestamps[indices] - targets)) <= .010001
    assert np.all(indices <= 300)
    # 50/4=12.5: repeated rounded stride would accumulate ~0.2s drift.
    assert indices[-1] - indices[0] in (237, 238)


@pytest.mark.parametrize('index', [0, 1, 300, 310, 329])
def test_boundaries_actions_and_three_current_views(index):
    base = FakeDataset()
    settings = HighCameraMosaicConfig(tile_size=4)
    ds = HighCameraMosaicDataset(base, fps=50, settings=settings)
    current = base[index]
    sample = ds[index]
    np.testing.assert_array_equal(sample['action'], current['action'])
    np.testing.assert_array_equal(sample['observation.state'], current['observation.state'])
    for camera in CAMERAS[1:]:
        np.testing.assert_array_equal(sample['observation.images.' + camera], current['observation.images.' + camera])
    indices = ds.history_indices(index)
    assert np.all(ds._episode_indices[indices] == ds._episode_indices[index])
    assert sample['observation.images.cam_high'].shape == (6, 8, 8, 3)
    np.testing.assert_array_equal(sample['observation.images.cam_high'][-1], current['observation.images.cam_high'])


def test_trainable_redirect_keeps_current_actions_consistent():
    base = FakeDataset()
    mask = np.zeros(len(base), bool)
    mask[300] = True
    ds = HighCameraMosaicDataset(base, fps=50, settings=HighCameraMosaicConfig(tile_size=4), trainable_mask=mask)
    assert np.all(ds[0]['action'] == 300)


def test_old_temporal_wrapper_still_stacks_all_cameras():
    ds = TemporalFrameStackDataset(FakeDataset(), fps=50, num_frames=3, stride_seconds=1)
    sample = ds[300]
    assert all(sample['observation.images.' + camera].shape == (3, 8, 8, 3) for camera in CAMERAS)


def test_fast_video_path_queries_only_high_once():
    class VideoFake(LeRobotDataset):
        def __init__(self):
            self.fake = FakeDataset()
            self.hf_dataset = {column: [self.fake[i][column] for i in range(len(self.fake))]
                               for column in ('episode_index', 'frame_index', 'timestamp')}
            self.meta = types.SimpleNamespace(video_keys=['observation.images.' + camera for camera in CAMERAS])
            self.image_transforms = None
            self.queries = []

        @property
        def num_frames(self):
            return len(self.fake)

        def __getitem__(self, index):
            return self.fake[index]

        def _query_videos(self, timestamps, episode):
            self.queries.append(timestamps)
            assert set(timestamps) == {'observation.images.cam_high'}
            frames = [torch.full((3, 8, 8), round(t * 50) % 255 / 255) for t in timestamps['observation.images.cam_high']]
            return {'observation.images.cam_high': torch.stack(frames)}

    base = VideoFake()
    ds = HighCameraMosaicDataset(base, fps=50, settings=HighCameraMosaicConfig(tile_size=4))
    base.fake.reads.clear()
    sample = ds[300]
    assert base.fake.reads == [300] and len(base.queries) == 1
    assert sample['observation.images.cam_high'].shape == (6, 8, 8, 3)


@pytest.mark.parametrize('kwargs', [{'sample_hz':3}, {'history_seconds':0}, {'camera':'cam_low'}])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        HighCameraMosaicConfig(**kwargs)

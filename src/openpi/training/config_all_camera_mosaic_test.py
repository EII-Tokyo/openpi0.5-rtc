"""Check the requested input geometry and preservation of existing sampling weights."""
from collections import Counter

import numpy as np
from openpi.training import config

BASE = 'eii_with_rinse_cam4_fullft_all_69repo_h100_448_high_mem5s4hz_mosaic2x2_pool256'
NAME = 'eii_with_rinse_cam4_fullft_all_98repo_h200x2_448_allcam_mem1s4hz_mosaic2x2_pool256'
CAMERAS = ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist')


def test_new_layout_and_original_layout_are_independent():
    """All four views use one four-frame historical mosaic plus their current frame."""
    cfg, original = config.get_config(NAME), config.get_config(BASE)
    pipeline = cfg.data.transform_pipeline
    assert cfg.model.image_resolution == pipeline.image_resolution == (448, 448)
    assert cfg.model.image_keys == CAMERAS
    assert pipeline.video_memory_num_frames == 2
    assert pipeline.high_camera_mosaic.cameras == CAMERAS
    np.testing.assert_array_equal(pipeline.high_camera_mosaic.offsets_seconds, [-1, -.75, -.5, -.25])
    assert pipeline.high_camera_mosaic.tile_size == 224
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 256 and cfg.fsdp_devices == 2
    assert original.data.transform_pipeline.video_memory_num_frames == 6
    assert original.data.transform_pipeline.high_camera_mosaic.cameras == ('cam_high',)


def test_existing_repo_weights_are_preserved():
    """The original duplicates retain their weight; newly collected exports enter once."""
    cfg, original = config.get_config(NAME), config.get_config(BASE)
    old, new = Counter(original.data.repo_ids), Counter(cfg.data.repo_ids)
    assert len(new) == 98 and sum(new.values()) == 112
    assert all(new[repo] == weight for repo, weight in old.items())
    added = set(new) - set(old)
    assert len(added) == 29 and all(new[repo] == 1 for repo in added)

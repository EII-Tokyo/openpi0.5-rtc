import json

import h5py
import numpy as np
import pytest

from openpi.training import rlt_timeline_replay


def _write_timeline_hdf5(path, *, frames: int = 8) -> None:
    with h5py.File(path, "w") as root:
        root.attrs["key_region_id"] = "demo"
        root.attrs["reward"] = 1
        root.attrs["replay_state_grain"] = "raw_frame_timeline"
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=np.arange(frames * 4, dtype=np.float32).reshape(frames, 4))
        obs.create_dataset("qvel", data=np.zeros((frames, 4), dtype=np.float32))
        obs.create_dataset("effort", data=np.zeros((frames, 4), dtype=np.float32))
        root.create_dataset("action", data=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        root.create_dataset("reference_action", data=100 + np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        timeline = root.create_group("rlt_timeline")
        timeline.attrs["state_grain"] = "raw_frame_timeline"
        timeline.attrs["z_rl_source"] = "vla_same_forward"
        timeline.attrs["rl_token_checkpoint_path"] = "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        timeline.create_dataset("z_rl", data=1000 + np.arange(frames * 3, dtype=np.float32).reshape(frames, 3))
        timeline.create_dataset("proprio", data=2000 + np.arange(frames * 5, dtype=np.float32).reshape(frames, 5))
        timeline.create_dataset("step_index", data=np.arange(frames, dtype=np.int64))
        timeline.create_dataset("valid", data=np.ones((frames,), dtype=np.bool_))
        root.create_dataset("timestamps", data=np.arange(frames, dtype=np.float64))


def test_build_paper_replay_arrays_from_frame_timeline(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_timeline_hdf5(h5_path, frames=8)

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=2,
        chunk_stride=2,
    )

    assert arrays["action"].shape == (3, 2, 2)
    assert arrays["reference_action"].shape == (3, 2, 2)
    assert arrays["next_reference_action"].shape == (3, 2, 2)
    np.testing.assert_allclose(arrays["z_rl"][:, 0], [1000, 1006, 1012])
    np.testing.assert_allclose(arrays["next_z_rl"][:, 0], [1006, 1012, 1018])
    np.testing.assert_allclose(arrays["action"][1, :, 0], [4, 6])
    np.testing.assert_allclose(arrays["next_reference_action"][1, :, 0], [108, 110])
    assert arrays["done"].tolist() == [False, False, True]
    assert arrays["reward_seq"].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    assert manifest["replay_state_grain"] == "paper_subsampled_anchor"
    assert manifest["source_format"] == "rlt_timeline_hdf5"
    assert manifest["current_frames"] == [0, 2, 4]
    assert manifest["next_frames"] == [2, 4, 6]


def test_build_paper_replay_rejects_runtime_cache_block_z(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_timeline_hdf5(h5_path, frames=8)
    with h5py.File(h5_path, "a") as root:
        root["rlt_timeline"].attrs["z_rl_source"] = "runtime_action_cache_block"

    with pytest.raises(ValueError, match="z_rl_source"):
        rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
            h5_path,
            train_horizon=2,
            chunk_stride=2,
        )


def test_write_paper_replay_shard_from_timeline_hdf5(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    out = tmp_path / "shards" / "key_region_demo.npz"
    _write_timeline_hdf5(h5_path, frames=8)

    rlt_timeline_replay.write_paper_replay_shard_from_timeline_hdf5(
        h5_path,
        out,
        train_horizon=2,
        chunk_stride=2,
        overwrite=False,
    )

    with np.load(out, allow_pickle=False) as data:
        assert data["z_rl"].shape == (3, 3)
        assert data["action"].shape == (3, 2, 2)
        manifest = json.loads(str(data["manifest"].item()))
    assert manifest["shard_path"] == str(out.resolve())
    assert manifest["formal_replay_ready"] is True

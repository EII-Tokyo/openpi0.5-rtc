from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.data.hdf5_reader import inspect_episode
from aloha_isaac_replay.data.hdf5_reader import is_success_episode


def _write_minimal_episode(path: Path, *, reward: float = 1.0) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["reward"] = reward
        h5.attrs["fps"] = 50.0
        h5.attrs["camera_names"] = np.array(["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"], dtype="S")
        observations = h5.create_group("observations")
        qpos = np.stack([np.arange(14), np.arange(14) + 0.1, np.arange(14) + 0.2]).astype(np.float64)
        observations.create_dataset("qpos", data=qpos)
        h5.create_dataset("action", data=np.vstack([qpos[1:], qpos[-1:]]))
        h5.create_dataset("reference_action", data=qpos)
        h5.create_dataset("timestamps", data=np.array([0.0, 0.02, 0.04], dtype=np.float64))


def test_inspect_episode_reads_numeric_metadata_without_images(tmp_path: Path) -> None:
    episode = tmp_path / "episode.hdf5"
    _write_minimal_episode(episode)
    before_mtime = episode.stat().st_mtime_ns

    inspection = inspect_episode(episode)

    assert episode.stat().st_mtime_ns == before_mtime
    assert inspection.complete_for_replay is True
    assert inspection.episode_length == 3
    assert inspection.qpos.shape == (3, 14)
    assert inspection.action_semantics.label == "absolute_joint_target_next_qpos"
    assert is_success_episode(inspection)


def test_inspect_episode_rejects_incomplete_camera_metadata(tmp_path: Path) -> None:
    episode = tmp_path / "episode.hdf5"
    _write_minimal_episode(episode)
    with h5py.File(episode, "a") as h5:
        h5.attrs["camera_names"] = np.array(["cam_high"], dtype="S")

    try:
        inspect_episode(episode)
    except ValueError as exc:
        assert "camera:cam_low" in str(exc)
    else:
        raise AssertionError("inspect_episode should reject missing camera metadata")


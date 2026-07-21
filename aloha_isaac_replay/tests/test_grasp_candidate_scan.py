from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.data.grasp_candidate_scan import inspect_grasp_candidate
from aloha_isaac_replay.scripts.scan_isaac_hdf5_tabletop_grasp_candidates import (
    _classify_tabletop_candidate,
    _open_then_close_frame_indices,
)


def _write_episode(path: Path, qpos: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations/qpos", data=qpos)
        h5.create_dataset("action", data=qpos)
        h5.create_dataset("timestamps", data=np.arange(len(qpos), dtype=np.float64) / 50.0)
        h5.attrs["fps"] = 50


def test_pickup_candidate_scores_open_close_and_post_close_motion(tmp_path: Path) -> None:
    qpos = np.zeros((160, 14), dtype=np.float64)
    qpos[:, 6] = np.r_[np.ones(60) * 0.9, np.linspace(0.9, 0.2, 30), np.ones(70) * 0.2]
    qpos[:, 0] = np.linspace(0.0, 0.4, len(qpos))
    _write_episode(tmp_path / "episode.hdf5", qpos)

    row = inspect_grasp_candidate(tmp_path / "episode.hdf5")

    assert row.close_frame is not None
    assert row.post_close_frames >= 20
    assert row.gripper_close_delta > 0.2
    assert row.score > 4.0


def test_short_already_closed_key_region_is_rejected(tmp_path: Path) -> None:
    qpos = np.zeros((18, 14), dtype=np.float64)
    qpos[:, 6] = 0.25
    qpos[:, 0] = np.linspace(0.0, 0.03, len(qpos))
    _write_episode(tmp_path / "episode.hdf5", qpos)

    row = inspect_grasp_candidate(tmp_path / "episode.hdf5")

    assert row.likely_full_pickup is False
    assert "too_short_for_pickup_scan" in row.reasons
    assert "left_gripper_not_open_at_start" in row.reasons


def test_high_level_episode_without_timestamps_uses_50hz_fallback(tmp_path: Path) -> None:
    qpos = np.zeros((100, 14), dtype=np.float64)
    qpos[:, 6] = np.r_[np.ones(50) * 0.8, np.ones(50) * 0.2]
    with h5py.File(tmp_path / "episode.hdf5", "w") as h5:
        h5.create_dataset("observations/qpos", data=qpos)
        h5.create_dataset("action", data=qpos)

    row = inspect_grasp_candidate(tmp_path / "episode.hdf5")

    assert row.fps == 50.0
    assert row.duration_s == 1.98
    assert row.close_frame is not None


def test_open_then_close_frame_indices_select_pregrasp_not_release() -> None:
    gripper = np.r_[
        np.ones(30) * 0.1,
        np.linspace(0.1, 0.95, 20),
        np.ones(30) * 0.95,
        np.linspace(0.95, 0.2, 20),
        np.ones(30) * 0.2,
    ]

    frames = _open_then_close_frame_indices(gripper, open_threshold=0.65, close_threshold=0.35, lookahead=80)

    assert frames
    assert all(gripper[index] >= 0.65 for index in frames)
    assert all(np.min(gripper[index + 1 : index + 81]) <= 0.35 for index in frames)
    assert all(index >= 40 for index in frames)


def test_tabletop_candidate_classification_keeps_bad_frames_with_reasons() -> None:
    row = {
        "bbox_valid": True,
        "raw_gripper": 0.2,
        "midpoint_tabletop_height_error_m": 0.15,
        "closing_dot_object_x_abs": 0.7,
    }

    result = _classify_tabletop_candidate(row, open_threshold=0.65, close_threshold=0.35)

    assert result["candidate_class"] == "NOT_TABLETOP_GRASP_CANDIDATE"
    assert result["candidate_reasons"] == [
        "already_closed",
        "finger_midpoint_far_from_tabletop_bottle_height",
        "closing_axis_not_perpendicular_to_bottle_axis",
    ]


def test_tabletop_candidate_classification_accepts_open_aligned_tabletop_frame() -> None:
    row = {
        "bbox_valid": True,
        "raw_gripper": 0.9,
        "midpoint_tabletop_height_error_m": 0.01,
        "closing_dot_object_x_abs": 0.1,
    }

    result = _classify_tabletop_candidate(row, open_threshold=0.65, close_threshold=0.35)

    assert result["candidate_class"] == "TABLETOP_GRASP_CANDIDATE"
    assert result["candidate_reasons"] == []

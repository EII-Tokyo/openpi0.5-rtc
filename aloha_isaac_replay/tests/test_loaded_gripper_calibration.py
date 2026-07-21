from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aloha_isaac_replay.scripts.analyze_hdf5_loaded_gripper_calibration import (
    analyze_loaded_gripper_calibration,
)


def _write_episode(path: Path, *, qpos: np.ndarray, action: np.ndarray, effort: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        obs = h5.create_group("observations")
        obs.create_dataset("qpos", data=qpos.astype(np.float32))
        obs.create_dataset("effort", data=effort.astype(np.float32))
        h5.create_dataset("action", data=action.astype(np.float32))


def test_loaded_gripper_calibration_detects_close_intent_effort_plateau(tmp_path: Path) -> None:
    qpos = np.ones((20, 14), dtype=np.float64) * 0.9
    action = np.ones((20, 14), dtype=np.float64) * 0.9
    effort = np.zeros((20, 14), dtype=np.float64)
    # Frames 10-15 mimic real loaded soft-bottle contact: command closes hard,
    # observed qpos stays on a loaded plateau, and effort is high.
    qpos[10:16, 6] = 0.57
    action[10:16, 6] = 0.04
    effort[10:16, 6] = -800.0
    _write_episode(tmp_path / "episode.hdf5", qpos=qpos, action=action, effort=effort)

    report = analyze_loaded_gripper_calibration(
        hdf5_path=tmp_path / "episode.hdf5",
        output_dir=tmp_path / "out",
        side="left",
        start_frame=8,
        end_frame=18,
        close_action_threshold=0.12,
        qpos_action_gap_threshold=0.25,
        effort_abs_threshold=100.0,
        qpos_plateau_delta_threshold=0.01,
    )

    assert report["interpretation"]["status"] == "LOADED_CLOSE_PLATEAU_DETECTED"
    assert report["interpretation"]["qpos_is_not_direct_loaded_pad_gap_sensor"] is True
    assert report["loaded_close_plateau_frame_count"] == 5
    assert report["longest_loaded_close_plateau_cluster"]["start_hdf5_frame"] == 11
    assert report["longest_loaded_close_plateau_cluster"]["end_hdf5_frame"] == 15
    assert report["longest_loaded_close_plateau_cluster"]["qpos_action_gap_mean"] == pytest.approx(0.53)
    assert Path(report["json"]).exists()
    assert Path(report["markdown"]).exists()


def test_loaded_gripper_calibration_reports_no_plateau_when_effort_is_low(tmp_path: Path) -> None:
    qpos = np.ones((20, 14), dtype=np.float64) * 0.57
    action = np.ones((20, 14), dtype=np.float64) * 0.04
    effort = np.zeros((20, 14), dtype=np.float64)
    _write_episode(tmp_path / "episode.hdf5", qpos=qpos, action=action, effort=effort)

    report = analyze_loaded_gripper_calibration(
        hdf5_path=tmp_path / "episode.hdf5",
        output_dir=tmp_path / "out",
        side="left",
        start_frame=0,
        end_frame=20,
        effort_abs_threshold=100.0,
    )

    assert report["interpretation"]["status"] == "NO_LOADED_CLOSE_PLATEAU_DETECTED"
    assert report["loaded_close_plateau_frame_count"] == 0

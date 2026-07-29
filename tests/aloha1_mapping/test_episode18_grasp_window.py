from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.aloha1_mapping.episode18_grasp_window import build_frame_records
from tools.aloha1_mapping.episode18_grasp_window import detect_gripper_phases
from tools.aloha1_mapping.episode18_grasp_window import load_episode_window
from tools.aloha1_mapping.episode18_grasp_window import robust_change_threshold
from tools.aloha1_mapping.episode18_grasp_window import write_episode_reports


def _write_episode(
    path: Path,
    *,
    action: np.ndarray | None = None,
    qpos: np.ndarray | None = None,
) -> str:
    action_data = (
        np.zeros((300, 14), dtype=np.float64) if action is None else action
    )
    qpos_data = np.zeros((300, 14), dtype=np.float64) if qpos is None else qpos
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action_data)
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=qpos_data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_episode_window_keeps_action_and_qpos_separate(tmp_path: Path) -> None:
    path = tmp_path / "episode.hdf5"
    action = np.zeros((300, 14), dtype=np.float64)
    qpos = np.zeros((300, 14), dtype=np.float64)
    action[:, 0] = np.arange(300)
    qpos[:, 0] = -np.arange(300)
    expected_hash = _write_episode(path, action=action, qpos=qpos)

    window = load_episode_window(
        path,
        208,
        244,
        expected_sha256=expected_hash,
    )

    assert window.frames.tolist() == list(range(208, 245))
    assert window.action.shape == (37, 14)
    assert window.qpos.shape == (37, 14)
    assert window.action[0, 0] == pytest.approx(208.0)
    assert window.qpos[0, 0] == pytest.approx(-208.0)
    assert window.source_sha256 == expected_hash


@pytest.mark.parametrize(
    "mutation",
    ["missing_action", "missing_qpos", "wrong_action_shape", "wrong_qpos_shape"],
)
def test_load_episode_window_rejects_invalid_datasets(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"{mutation}.hdf5"
    with h5py.File(path, "w") as handle:
        if mutation != "missing_action":
            shape = (300, 13) if mutation == "wrong_action_shape" else (300, 14)
            handle.create_dataset("action", data=np.zeros(shape))
        observations = handle.create_group("observations")
        if mutation != "missing_qpos":
            shape = (300, 15) if mutation == "wrong_qpos_shape" else (300, 14)
            observations.create_dataset("qpos", data=np.zeros(shape))

    with pytest.raises(ValueError, match="action|qpos|14"):
        load_episode_window(path, 208, 244)


def test_load_episode_window_rejects_bad_range_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "episode.hdf5"
    _write_episode(path)

    with pytest.raises(ValueError, match="range"):
        load_episode_window(path, 244, 208)
    with pytest.raises(ValueError, match="range"):
        load_episode_window(path, 290, 310)
    with pytest.raises(ValueError, match="SHA-256"):
        load_episode_window(path, 208, 244, expected_sha256="0" * 64)


def test_detect_gripper_phases_separates_command_and_readback() -> None:
    frames = np.arange(208, 245)
    action_gripper = np.ones(frames.size, dtype=np.float64)
    qpos_gripper = np.ones(frames.size, dtype=np.float64)
    action_gripper[frames >= 225] -= 0.05 * (frames[frames >= 225] - 224)
    qpos_gripper[frames >= 229] -= 0.04 * (frames[frames >= 229] - 228)

    phases = detect_gripper_phases(
        action_gripper,
        qpos_gripper,
        first_frame=208,
    )

    assert phases.close_command_start_frame == 225
    assert phases.readback_response_start_frame == 229
    assert (
        phases.close_command_start_frame
        != phases.readback_response_start_frame
    )
    assert phases.command_direction == "decreasing"
    assert phases.readback_direction == "decreasing"


def test_robust_threshold_uses_scaled_epsilon_when_mad_is_zero() -> None:
    signal = np.full(20, 1000.0, dtype=np.float64)
    threshold = robust_change_threshold(signal)

    assert threshold > 0.0
    assert threshold <= np.finfo(np.float64).eps * 1001.0


def test_report_records_keep_command_and_readback_fields_distinct(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.hdf5"
    action = np.zeros((300, 14), dtype=np.float64)
    qpos = np.zeros((300, 14), dtype=np.float64)
    action[225:, 6] = -0.2
    qpos[229:, 6] = -0.1
    expected_hash = _write_episode(path, action=action, qpos=qpos)
    window = load_episode_window(
        path,
        208,
        244,
        expected_sha256=expected_hash,
    )
    phases = detect_gripper_phases(
        window.action[:, 6],
        window.qpos[:, 6],
        first_frame=208,
    )

    records = build_frame_records(window, phases)

    assert len(records) == 37
    required = {
        "frame",
        "action_left_arm_6d",
        "qpos_left_arm_6d",
        "action_left_gripper",
        "qpos_left_gripper",
        "action_step_norm",
        "qpos_step_norm",
        "phase_labels",
    }
    assert required <= records[0].keys()
    assert records[17]["action_left_gripper"] == pytest.approx(-0.2)
    assert records[17]["qpos_left_gripper"] == pytest.approx(0.0)

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    report = write_episode_reports(
        window,
        phases,
        json_output=json_path,
        csv_output=csv_path,
    )

    assert report["frame_rate_status"] == "NOT_EMITTED_UNTIL_SOURCE_PROVEN"
    assert report["frame_count"] == 37
    assert json.loads(json_path.read_text())["frames"] == records
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 37
    assert rows[0]["frame"] == "208"

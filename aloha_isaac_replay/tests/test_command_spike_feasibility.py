from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aloha_isaac_replay.scripts.analyze_hdf5_command_spike_feasibility import analyze_command_spikes
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _command_delta_distribution
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _contact_quality_summary
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _object_lift_gate


def _write_hdf5(path: Path, qpos: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations/qpos", data=qpos.astype(np.float32))


def _write_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "dof_mapping:",
                "  - canonical_name: left_waist",
                "    dataset_index: 0",
                "    sign: 1.0",
                "    offset: 0.0",
                "    scale: 1.0",
                "    unit: rad",
                "  - canonical_name: left_shoulder",
                "    dataset_index: 1",
                "    sign: 1.0",
                "    offset: 0.0",
                "    scale: 1.0",
                "    unit: rad",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_metrics(path: Path, *, end_frame: int = 80, steps: int = 79) -> None:
    path.write_text(
        json.dumps(
            {
                "inputs": {"hdf5_gripper_start_frame": 0, "hdf5_gripper_end_frame": end_frame},
                "hdf5_gripper_replay_steps": steps,
                "hdf5_replay_rate_hz": 50.0,
                "physics_dt": 0.004,
                "hdf5_replay_target_hold_steps": 5,
                "hdf5_replay_substep_mode": "zero_order_hold",
                "tracking_summary": {
                    "groups": {
                        "controlled": {
                            "max_abs_error_dof_name": "left_shoulder",
                            "max_abs_error_step": 10,
                            "max_abs_error": 0.04,
                        }
                    }
                },
                "tracking_spike_packet": {
                    "pre_step_qpos": 0.1,
                    "post_step_qpos": 0.13,
                    "actual_delta_during_hold": 0.03,
                    "estimated_actual_velocity_during_hold": 1.5,
                    "tracking_ratio": 0.5,
                    "max_abs_error_signed": 0.04,
                    "contact_categories_at_step": [],
                },
                "drive_authority_audit": {"estimated_effort_clipped": False},
                "physical_grasp_gate": {"status": "PASS_PHYSICAL_GRASP_SEMANTICS"},
                "target_limit_gate_ok": True,
                "controller_replay_fidelity_gate": {"status": "FAIL_POST_STEP_TRACKING_EXCEEDS_THRESHOLD"},
            }
        ),
        encoding="utf-8",
    )


def test_command_spike_feasibility_classifies_repeated_cluster(tmp_path: Path) -> None:
    qpos = np.zeros((80, 14), dtype=np.float64)
    # Step 10 corresponds to HDF5 frame 11.  A 0.08 rad target jump at 50 Hz is 4 rad/s.
    qpos[9, 1] = 0.00
    qpos[10, 1] = 0.04
    qpos[11, 1] = 0.12
    qpos[12, 1] = 0.20
    qpos[13, 1] = 0.24
    _write_hdf5(tmp_path / "episode.hdf5", qpos)
    _write_mapping(tmp_path / "mapping.yaml")
    _write_metrics(tmp_path / "metrics.json")

    report = analyze_command_spikes(
        hdf5_path=tmp_path / "episode.hdf5",
        mapping_path=tmp_path / "mapping.yaml",
        metrics_path=tmp_path / "metrics.json",
        output_dir=tmp_path / "out",
        spike_threshold_rad_s=2.0,
        cluster_gap_steps=1,
    )

    assert report["failure_classification"] == "REPEATED_SPIKE_CLUSTER"
    failure = report["failure_step"]
    assert failure["joint_name"] == "left_shoulder"
    assert failure["failure_step"] == 10
    assert failure["hdf5_frame_index"] == 11
    assert failure["target_velocity_prev_to_current"] == pytest.approx(4.0)
    assert failure["is_spike"] is True
    clusters = report["per_joint"]["left_shoulder"]["clusters"]
    assert clusters
    assert clusters[0]["cluster_start_step"] <= 10 <= clusters[0]["cluster_end_step"]
    assert Path(report["json"]).exists()
    assert Path(report["markdown"]).exists()


def test_command_spike_feasibility_rejects_metrics_hdf5_window_mismatch(tmp_path: Path) -> None:
    _write_hdf5(tmp_path / "episode.hdf5", np.zeros((20, 14), dtype=np.float64))
    _write_mapping(tmp_path / "mapping.yaml")
    _write_metrics(tmp_path / "metrics.json", end_frame=20, steps=18)

    try:
        analyze_command_spikes(
            hdf5_path=tmp_path / "episode.hdf5",
            mapping_path=tmp_path / "mapping.yaml",
            metrics_path=tmp_path / "metrics.json",
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "metrics/HDF5 timing mismatch" in str(exc)
    else:
        raise AssertionError("expected metrics/HDF5 timing mismatch")


def test_command_delta_distribution_fails_when_target_velocity_exceeds_threshold() -> None:
    report = _command_delta_distribution(
        tracking_rows=[
            {
                "phase": "close",
                "step": 12,
                "target": np.array([0.10]),
                "previous_target": np.array([0.00]),
                "pre_qpos": np.array([0.01]),
                "post_qpos": np.array([0.03]),
                "qvel": np.array([1.0]),
            }
        ],
        groups={"controlled": [0]},
        dof_names=["left_shoulder"],
        effective_target_dt=0.02,
        max_abs_target_velocity=2.0,
    )

    assert report["pass"] is False
    assert report["status"] == "FAIL_COMMAND_TARGET_VELOCITY_EXCEEDS_THRESHOLD"
    assert report["classification"] == "SINGLE_SPIKE_RESIDUAL"
    assert report["recommendation"] == "BLOCK_CCD_FIX_COMMAND_CONTINUITY_FIRST"
    assert report["formal_replay_targets_modified"] is False
    assert report["deleted_frames"] == 0
    assert report["smoothed_frames"] == 0
    assert report["interpolated_frames"] == 0
    assert report["spike_count"] == 1
    assert report["spike_steps"] == [12]
    spike = report["top_target_velocity_spikes"][0]
    assert spike["dof_name"] == "left_shoulder"
    assert spike["target_velocity"] == pytest.approx(5.0)


def test_command_delta_distribution_without_threshold_is_diagnostic_only() -> None:
    report = _command_delta_distribution(
        tracking_rows=[
            {
                "phase": "close",
                "step": 12,
                "target": np.array([0.10]),
                "previous_target": np.array([0.00]),
                "pre_qpos": np.array([0.01]),
                "post_qpos": np.array([0.03]),
            }
        ],
        groups={"controlled": [0]},
        dof_names=["left_shoulder"],
        effective_target_dt=0.02,
        max_abs_target_velocity=None,
    )

    assert report["pass"] is True
    assert report["status"] == "DIAGNOSTIC_ONLY_NO_THRESHOLD"
    assert report["classification"] == "DIAGNOSTIC_ONLY_NO_THRESHOLD"


def test_command_delta_distribution_reports_repeated_spike_cluster() -> None:
    report = _command_delta_distribution(
        tracking_rows=[
            {
                "phase": "close",
                "step": 20,
                "target": np.array([0.10]),
                "previous_target": np.array([0.00]),
                "pre_qpos": np.array([0.00]),
                "post_qpos": np.array([0.02]),
            },
            {
                "phase": "close",
                "step": 21,
                "target": np.array([0.18]),
                "previous_target": np.array([0.10]),
                "pre_qpos": np.array([0.02]),
                "post_qpos": np.array([0.06]),
            },
        ],
        groups={"controlled": [0]},
        dof_names=["left_shoulder"],
        effective_target_dt=0.02,
        max_abs_target_velocity=2.0,
    )

    assert report["pass"] is False
    assert report["classification"] == "REPEATED_SPIKE_CLUSTER"
    assert report["spike_count"] == 2
    assert report["spike_clusters"][0]["cluster_start_step"] == 20
    assert report["spike_clusters"][0]["cluster_end_step"] == 21
    assert report["largest_cluster_length_seconds"] == pytest.approx(0.04)


def test_object_lift_gate_is_skipped_for_contact_only_threshold() -> None:
    report = _object_lift_gate(object_lift=-0.0012, min_object_lift=0.0)

    assert report["pass"] is True
    assert report["required"] is False
    assert report["status"] == "SKIPPED_CONTACT_ONLY_GATE"


def test_object_lift_gate_requires_positive_threshold() -> None:
    report = _object_lift_gate(object_lift=0.0012, min_object_lift=0.005)

    assert report["pass"] is False
    assert report["required"] is True
    assert report["status"] == "FAIL_OBJECT_LIFT_BELOW_THRESHOLD"


def test_contact_quality_summary_uses_all_contact_samples() -> None:
    report = _contact_quality_summary(
        [
            {
                "step": 10,
                "contact_data_sample": [
                    {"separation": 0.002, "impulse": [0.0, 0.0, 0.0]},
                    {"separation": -0.001, "impulse": [0.0, 3.0, 4.0]},
                ]
            },
            {
                "step": 12,
                "contact_data_sample": [
                    {"separation": 0.001, "impulse": [1.0, 0.0, 0.0]},
                ]
            },
        ]
    )

    assert report["row_count"] == 2
    assert report["contact_step_count"] == 2
    assert report["first_step"] == 10
    assert report["last_step"] == 12
    assert report["contact_data_sample_count"] == 3
    assert report["separation_min"] == pytest.approx(-0.001)
    assert report["separation_max"] == pytest.approx(0.002)
    assert report["max_impulse_norm"] == pytest.approx(5.0)
    assert report["nonzero_impulse_count"] == 2
    assert report["nonzero_impulse_step_count"] == 2

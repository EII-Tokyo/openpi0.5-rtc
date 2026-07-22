from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aloha_isaac_replay.scripts.analyze_hdf5_command_spike_feasibility import analyze_command_spikes
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _command_delta_distribution
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _apply_diagnostic_loaded_clamp_squeeze
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _contact_quality_summary
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _fixed_reference_grasp_geometry_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _grasp_band_proxy_axis_length_stage_units
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _lift_transport_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _lift_contact_wrench_patch_audit
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _object_lift_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _prelift_static_grasp_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _table_load_bearing_contact_gate


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


def test_diagnostic_loaded_clamp_squeeze_uses_guarded_reference_target() -> None:
    target = np.array([0.10, 0.20, 0.050, 0.051], dtype=np.float64)
    raw_close_target = np.array([0.10, 0.20, 0.025, 0.026], dtype=np.float64)

    squeezed, row = _apply_diagnostic_loaded_clamp_squeeze(
        enabled=True,
        target=target,
        dof_names=["joint0", "joint1", "left_left_finger", "left_right_finger"],
        finger_dof_names={"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
        runtime_limits=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 0.1], [0.0, 0.1]], dtype=np.float64),
        limit_margin=0.0,
        squeeze_depth=0.002,
        phase="post_close_hold",
        moving_fingers="both",
        reference_target=target,
        reference_target_source="width_guarded_step_target",
    )

    assert squeezed[2] == pytest.approx(0.049)
    assert squeezed[3] == pytest.approx(0.050)
    assert squeezed[2] > raw_close_target[2]
    assert row["reference_target_source"] == "width_guarded_step_target"
    assert row["finger_targets"]["left_finger"]["reference_target"] == pytest.approx(0.050)


def test_prelift_static_grasp_gate_passes_stable_hold() -> None:
    hold_rows = [
        {
            "phase": "post_close_hold",
            "step": step,
            "object_center_x": 0.10 + 1e-6 * step,
            "object_center_y": -0.20,
            "object_center_z": 0.034 + 1e-6 * step,
        }
        for step in range(30)
    ]
    gate = _prelift_static_grasp_gate(
        hold_rows=hold_rows,
        hold_bilateral_gate={
            "pass": True,
            "status": "PASS_BILATERAL_GRASP_FORMATION",
            "bilateral_contact_steps": list(range(14, 30)),
            "bilateral_contact_step_count": 16,
            "finger_rows": [
                {"finger_path": "left", "nonzero_impulse_step_count": 13},
                {"finger_path": "right", "nonzero_impulse_step_count": 14},
            ],
        },
        hold_tracking_gate={"pass": True, "status": "PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD"},
        min_hold_steps=30,
        min_tail_bilateral_steps=10,
        min_each_finger_nonzero_impulse_steps=3,
        max_object_xy_sweep=0.0015,
        max_object_z_delta=0.001,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_PRELIFT_STATIC_GRASP_DIAGNOSTIC"
    assert gate["tail_bilateral_contact_steps"] == 16
    assert gate["formal_close_replay_success"] is False


def test_prelift_static_grasp_gate_rejects_late_lift_only_contact() -> None:
    hold_rows = [
        {
            "phase": "post_close_hold",
            "step": step,
            "object_center_x": 0.10,
            "object_center_y": -0.20,
            "object_center_z": 0.034,
        }
        for step in range(30)
    ]
    gate = _prelift_static_grasp_gate(
        hold_rows=hold_rows,
        hold_bilateral_gate={
            "pass": True,
            "status": "PASS_BILATERAL_GRASP_FORMATION",
            "bilateral_contact_steps": [20, 21, 22],
            "bilateral_contact_step_count": 3,
            "finger_rows": [
                {"finger_path": "left", "nonzero_impulse_step_count": 3},
                {"finger_path": "right", "nonzero_impulse_step_count": 3},
            ],
        },
        hold_tracking_gate={"pass": True, "status": "PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD"},
        min_hold_steps=30,
        min_tail_bilateral_steps=10,
        min_each_finger_nonzero_impulse_steps=3,
        max_object_xy_sweep=0.0015,
        max_object_z_delta=0.001,
    )

    assert gate["pass"] is False
    assert "tail_bilateral_contact_not_sustained" in gate["failed_checks"]


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


def test_grasp_band_proxy_axis_length_is_short_local_contact_coupon() -> None:
    # The grasp-band collider is a local rear-quarter contact coupon. It must
    # not use the full 20.6 cm bottle length for projection or contact gates.
    assert _grasp_band_proxy_axis_length_stage_units(0.052, 4.0) == pytest.approx(0.0468)


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


def test_table_load_bearing_gate_accepts_near_contact_with_weak_impulse() -> None:
    contact_summary = {
        "object_contact_categories": {
            "target_finger": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 40,
                        "impulse_norm": {"p50": 0.055, "mean": 0.06, "max": 0.12},
                        "separation": {"p50": -0.0003, "p95": 0.00002},
                    }
                }
            },
            "workcell_or_environment": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 1,
                        "impulse_norm": {"p50": 0.0, "mean": 0.004, "max": 0.10},
                        "separation": {"p50": 0.0008, "p95": 0.006},
                    }
                }
            },
        }
    }

    gate = _table_load_bearing_contact_gate(contact_summary=contact_summary, eval_phase="post_close_lift")

    assert gate["pass"] is True
    assert gate["status"] == "PASS_TABLE_NEAR_CONTACT_NON_LOAD_BEARING"
    assert gate["classification_inputs"]["median_zero_and_separated"] is True
    assert gate["classification_inputs"]["weak_relative_impulse"] is True


def test_table_load_bearing_gate_rejects_sustained_table_support() -> None:
    contact_summary = {
        "object_contact_categories": {
            "target_finger": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 40,
                        "impulse_norm": {"p50": 0.055, "mean": 0.06, "max": 0.12},
                        "separation": {"p50": -0.0003, "p95": 0.00002},
                    }
                }
            },
            "workcell_or_environment": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 25,
                        "impulse_norm": {"p50": 0.03, "mean": 0.04, "max": 0.10},
                        "separation": {"p50": -0.0002, "p95": 0.0001},
                    }
                }
            },
        }
    }

    gate = _table_load_bearing_contact_gate(contact_summary=contact_summary, eval_phase="post_close_lift")

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_TABLE_LOAD_BEARING_CONTACT"
    assert gate["table_to_finger_impulse_mean_ratio"] == pytest.approx(0.04 / 0.06)


def test_lift_transport_gate_distinguishes_follow_from_strict_clearance() -> None:
    rows = [
        {
            "phase": "post_close_lift",
            "step": 0,
            "object_center_z": 0.034,
            "finger_mid_center_z": 0.050,
            "target_contact_pair_found": True,
        },
        {
            "phase": "post_close_lift",
            "step": 39,
            "object_center_z": 0.0378,
            "finger_mid_center_z": 0.0571,
            "target_contact_pair_found": True,
        },
    ]
    contact_summary = {
        "target_contact_steps": [{"phase": "post_close_lift", "step": step} for step in range(40)],
        "object_contact_categories": {
            "target_finger": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 40,
                        "impulse_norm": {"p50": 0.055, "mean": 0.06, "max": 0.12},
                        "separation": {"p50": -0.0003, "p95": 0.00002},
                    }
                }
            },
            "workcell_or_environment": {
                "phase_quality": {
                    "post_close_lift": {
                        "contact_step_count": 40,
                        "nonzero_impulse_step_count": 1,
                        "impulse_norm": {"p50": 0.0, "mean": 0.004, "max": 0.10},
                        "separation": {"p50": 0.0008, "p95": 0.006},
                    }
                }
            },
        },
    }
    object_lift_gate = _object_lift_gate(object_lift=0.0038, min_object_lift=0.005)

    gate = _lift_transport_gate(
        rows=rows,
        object_lift_gate=object_lift_gate,
        contact_summary=contact_summary,
        min_object_lift=0.005,
        diagnostic_held_object_mode="none",
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_STRICT_OBJECT_LIFT_CLEARANCE"
    assert gate["transport_follow_gate"]["pass"] is True
    assert gate["strict_lift_clearance_gate"]["pass"] is False
    assert gate["table_load_bearing_contact_gate"]["pass"] is True


def test_lift_contact_wrench_patch_audit_summarizes_finger_impulse_and_patch() -> None:
    rows = [
        {
            "phase": "post_close_lift",
            "step": 0,
            "object_center_x": 0.0,
            "object_center_y": 0.0,
            "object_center_z": 0.05,
        },
        {
            "phase": "post_close_lift",
            "step": 1,
            "object_center_x": 0.0,
            "object_center_y": -0.004,
            "object_center_z": 0.0505,
        },
    ]
    contact_pair_rows = [
        {
            "phase": "post_close_lift",
            "step": 0,
            "collider0": "/World/Bottle500/Collisions/body",
            "collider1": "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy",
            "sorted_pair": [
                "/World/Bottle500/Collisions/body",
                "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy",
            ],
            "contact_data_sample": [
                {
                    "position": [0.0, -0.015, 0.052],
                    "normal": [0.0, 1.0, 0.0],
                    "impulse": [0.0, -0.10, 0.01],
                    "separation": -0.0004,
                }
            ],
        },
        {
            "phase": "post_close_lift",
            "step": 1,
            "collider0": "/World/Bottle500/Collisions/body",
            "collider1": "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy",
            "sorted_pair": [
                "/World/Bottle500/Collisions/body",
                "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy",
            ],
            "contact_data_sample": [
                {
                    "position": [0.0, 0.013, 0.052],
                    "normal": [0.0, -1.0, 0.0],
                    "impulse": [0.0, -0.08, 0.01],
                    "separation": -0.0002,
                }
            ],
        },
    ]

    audit = _lift_contact_wrench_patch_audit(
        rows=rows,
        contact_pair_rows=contact_pair_rows,
        object_path="/World/Bottle500",
        expected_finger_paths=[
            "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy",
            "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy",
        ],
    )

    assert audit["diagnostic_only"] is True
    assert audit["status"] == "DIAGNOSTIC_LATERAL_DRIFT_WITH_LOW_LIFT"
    assert audit["usable_vector_sample_count"] == 2
    assert audit["object_delta_world"] == pytest.approx([0.0, -0.004, 0.0005])
    assert audit["net_finger_impulse_world"] == pytest.approx([0.0, -0.18, 0.02])
    assert audit["net_lateral_impulse_y"] == pytest.approx(-0.18)
    assert audit["net_vertical_impulse_z"] == pytest.approx(0.02)
    assert audit["by_finger"]["left_finger"]["sample_count"] == 1
    assert audit["by_finger"]["right_finger"]["sample_count"] == 1
    assert audit["by_finger"]["left_finger"]["mean_contact_position_relative_to_object_center"] == pytest.approx(
        [0.0, -0.015, 0.002]
    )


def test_fixed_reference_grasp_geometry_gate_rejects_table_penetration_and_lateral_impulse() -> None:
    gate = _fixed_reference_grasp_geometry_gate(
        tabletop_reference_contract={"table_top_z_m": 0.0},
        object_contact_reset_box={"min": [0.0, 0.0, 0.001]},
        object_final_contact_box={"min": [0.0, 0.0, -0.004]},
        start_alignment={
            "reference_contact_center": {
                "cross_closing_axis_offset_norm_m": 0.0001,
                "correction_to_midplane_norm_m": 0.0001,
            }
        },
        final_alignment={
            "reference_contact_center": {
                "cross_closing_axis_offset_norm_m": 0.007,
                "correction_to_midplane_norm_m": 0.0035,
            }
        },
        lift_contact_wrench_patch_audit={
            "net_vertical_impulse_z": 2.0,
            "net_lateral_impulse_xy_norm": 0.7,
        },
        fixed_reference_required=True,
        lift_required=True,
    )

    assert gate["required"] is True
    assert gate["pass"] is False
    assert gate["status"] == "FAIL_FIXED_REFERENCE_CONTACT_PROXY_TABLE_PENETRATION"
    assert gate["final_contact_table_penetration_m"] == pytest.approx(0.004)
    assert gate["net_lateral_to_vertical_impulse_ratio"] == pytest.approx(0.35)
    assert "final_contact_not_deeply_inside_table" in gate["failed_checks"]
    assert "net_lateral_impulse_not_dominant" in gate["failed_checks"]


def test_fixed_reference_grasp_geometry_gate_skips_contact_only_runs() -> None:
    gate = _fixed_reference_grasp_geometry_gate(
        tabletop_reference_contract={"table_top_z_m": 0.0},
        object_contact_reset_box={},
        object_final_contact_box={},
        start_alignment={},
        final_alignment={},
        lift_contact_wrench_patch_audit={},
        fixed_reference_required=True,
        lift_required=False,
    )

    assert gate["required"] is False
    assert gate["pass"] is True
    assert gate["status"] == "SKIPPED_NOT_FIXED_REFERENCE_LIFT"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.supplier_cad_one_joint import build_bidirectional_targets
from tools.aloha1_mapping.supplier_cad_one_joint import evaluate_one_joint_run
from tools.aloha1_mapping.supplier_cad_one_joint import summarize_robots

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/validate_aloha_viper_supplier_cad_one_joint.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)


def test_bidirectional_targets_stay_inside_runtime_limits() -> None:
    targets = build_bidirectional_targets(
        start=[0.9, 0.0],
        joint_index=0,
        lower=-1.0,
        upper=1.0,
        requested_delta=0.2,
    )
    assert targets == [[0.7, 0.0], [0.98, 0.0]]
    assert all(-1.0 <= target[0] <= 1.0 for target in targets)
    assert all(target[1] == 0.0 for target in targets)


def test_one_joint_gate_checks_direction_range_and_non_target_motion() -> None:
    result = evaluate_one_joint_run(
        dof_names=["waist", "shoulder", "elbow"],
        commanded_indices=[1],
        commanded_delta=[0.05],
        start=[0.0, -0.5, 0.2],
        end=[0.00001, -0.451, 0.19999],
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
        readback_minimum=0.001,
        target_tolerance=0.01,
        unexpected_tolerance=0.001,
    )
    assert result["status"] == "PASS"
    assert result["direction_ok"] is True
    assert result["range_ok"] is True
    assert result["max_unexpected_delta"] == pytest.approx(0.00001)


def test_one_joint_gate_rejects_motion_in_another_dof() -> None:
    result = evaluate_one_joint_run(
        dof_names=["waist", "shoulder"],
        commanded_indices=[0],
        commanded_delta=[0.05],
        start=[0.0, 0.0],
        end=[0.05, 0.02],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        readback_minimum=0.001,
        target_tolerance=0.01,
        unexpected_tolerance=0.001,
    )
    assert result["status"] == "FAIL"
    assert result["max_unexpected_delta"] == 0.02


def test_explicit_symmetric_fingers_are_one_control_group() -> None:
    result = evaluate_one_joint_run(
        dof_names=["left_finger", "right_finger"],
        commanded_indices=[0, 1],
        commanded_delta=[-0.01, 0.01],
        start=[0.05, -0.05],
        end=[0.0401, -0.0401],
        lower=[0.021, -0.057],
        upper=[0.057, -0.021],
        readback_minimum=0.001,
        target_tolerance=0.001,
        unexpected_tolerance=0.001,
        symmetric_pair=(0, 1),
        symmetric_tolerance=0.001,
    )
    assert result["status"] == "PASS"
    assert result["symmetric_residual"] == 0.0


def test_missing_right_stage_keeps_dual_robot_summary_partial() -> None:
    summary = summarize_robots(
        {
            "follower_left": {"status": "PASS"},
            "follower_right": {
                "status": "NOT_RUN",
                "blocker": (
                    "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_"
                    "FOLLOWER_RIGHT_STAGE"
                ),
            },
        }
    )
    assert summary["status"] == "PARTIAL"
    assert summary["pass_count"] == 1
    assert summary["not_run_count"] == 1


def test_runtime_script_freezes_stage_and_right_arm_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SimulationApp" in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert "aloha_viperx_supplier_cad_arm_max_force_over_combined.usda" in source
    assert "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE" in (
        source
    )
    assert "SurfaceGripper" not in source
    assert "192.168.1.103" not in source


def test_current_runtime_report_replaces_historical_right_not_run() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["scope"] == (
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
    )
    assert report["stage"]["immutable"] is True
    assert report["gripper_validation"]["mimic_parent"] == "left_finger"
    assert report["gripper_validation"]["mimic_multiplier"] == -1.0
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
    ]
    assert report["task8"] == "NOT_RUN"

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/validate_aloha_viper_follower_right_one_joint.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_validation.json"
)
STRUCTURE = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_structure_validation.json"
)
CURVES = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_one_joint_curves.csv"
)
EXPECTED_DOF_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
]


def test_gripper_summary_accepts_four_ordered_aperture_states() -> None:
    from tools.validate_aloha_viper_follower_right_one_joint import _gripper_summary

    cases = [
        {
            "state": state,
            "repeat": 0,
            "status": "PASS",
            "readback_left_m": left,
            "readback_right_m": -left,
            "aperture_m": 2.0 * left,
            "mimic_residual_m": 0.0,
            "legal_range": True,
        }
        for state, left in (
            ("closed", 0.021),
            ("partially_closed", 0.039),
            ("open", 0.052),
            ("maximum_legal_aperture", 0.057),
        )
    ]

    summary = _gripper_summary(cases)

    assert summary["status"] == "PASS"
    assert summary["aperture_monotonicity"] == "PASS"


def test_gripper_readback_is_sampled_after_terminal_target_settle() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["protocol"]["terminal_target_settle_steps"] == 30
    assert all(
        item["terminal_target_settle_steps"] == 30
        for item in report["gripper_validation"]["all_repeats"]
    )


def test_right_runtime_script_is_robot_local_and_simulation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "supplier_cad_follower_right.usda" in source
    assert "/follower_right/vx300s_right/root_joint" in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT" in source
    assert "SurfaceGripper" not in source
    assert "fixed_joint" not in source.lower()
    assert "192.168.1.103" not in source


def test_right_arm_one_joint_cases_all_pass() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["scope"] == (
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
    )
    assert report["dof_order"] == EXPECTED_DOF_ORDER
    arm_cases = report["arm_one_joint_cases"]
    assert len(arm_cases) == 24
    assert all(item["status"] == "PASS" for item in arm_cases)
    assert {
        (item["joint_name"], item["direction"], item["repeat"])
        for item in arm_cases
    } == {
        (name, direction, repeat)
        for name in EXPECTED_DOF_ORDER[:6]
        for direction in ("negative", "positive")
        for repeat in (0, 1)
    }


def test_right_gripper_direction_and_aperture_pass_but_mimic_fails() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gripper = report["gripper_validation"]
    assert gripper["status"] == "FAIL"
    assert gripper["control_mode"] == "SOURCE_URDF_MIMIC_LEFT_DRIVEN"
    assert gripper["mimic_parent"] == "left_finger"
    assert gripper["mimic_multiplier"] == -1.0
    assert gripper["mimic_offset"] == 0.0
    assert gripper["maximum_mimic_residual_m"] > 0.001
    assert gripper["aperture_monotonicity"] == "PASS"
    assert gripper["motion_direction"] == "PASS"
    assert gripper["legal_range"] == "PASS"
    assert gripper["states"]["closed"]["aperture_m"] < (
        gripper["states"]["partially_closed"]["aperture_m"]
    )
    assert gripper["states"]["partially_closed"]["aperture_m"] < (
        gripper["states"]["open"]["aperture_m"]
    )
    assert gripper["states"]["open"]["aperture_m"] < (
        gripper["states"]["maximum_legal_aperture"]["aperture_m"]
    )


def test_right_static_pose_first_frame_and_determinism_pass() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["first_frame_jump"]["status"] == "PASS"
    assert report["static_pose_hold"]["status"] == "PASS"
    assert report["determinism"]["status"] == "PASS"
    assert report["determinism"]["unique_signature_count"] == 1
    assert report["stage"]["immutable"] is True
    assert report["workcell_placement_verified"] is False
    assert report["hard_blockers"] == [
        "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
    ]
    assert report["task8"] == "NOT_RUN"


def test_right_curves_are_machine_readable_and_cover_all_cases() -> None:
    with CURVES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert set(EXPECTED_DOF_ORDER).issubset(rows[0])
    assert {int(row["repeat"]) for row in rows} == {0, 1}
    assert len({row["test"] for row in rows}) >= 16


def test_right_structure_report_keeps_overlap_evidence_boundary() -> None:
    report = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["articulation_count"] == 1
    assert report["dof_order"] == EXPECTED_DOF_ORDER
    assert report["supplier_finger_identity"] == "PASS"
    assert report["generic_finger_deactivated"] is True
    assert report["initial_overlap"]["status"] == "PASS"
    assert report["initial_overlap"]["evidence_method"] == (
        "VERIFIED_ROBOT_LOCAL_GEOMETRY_EQUIVALENCE_TO_FOLLOWER_LEFT"
    )
    assert report["initial_overlap"]["unexpected_overlap"] is False
    assert report["initial_overlap"]["attachment_common_volume_retained"] is True
    assert report["classification"] == (
        "RIGHT_ROBOT_LOCAL_STRUCTURE_PARTIAL_MIMIC_ACCURACY_FAILED"
    )

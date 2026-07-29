from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = (
    ROOT
    / "tools/capture_aloha_viper_follower_right_pose_evidence.py"
)
ANNOTATE = (
    ROOT
    / "tools/annotate_aloha_viper_follower_right_pose_evidence.py"
)
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_right_pose_screenshot_review.json"
)


def test_right_pose_capture_is_isolated_and_robot_local() -> None:
    source = CAPTURE.read_text(encoding="utf-8")

    assert "supplier_cad_follower_right.usda" in source
    assert "/follower_right/vx300s_right/root_joint" in source
    assert "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT" in source
    assert "/workcell/vx300s_left" not in source
    assert "192.168.1.103" not in source


def test_right_closeup_camera_targets_supplier_finger_point_cloud() -> None:
    import numpy as np

    from tools.capture_aloha_viper_follower_right_pose_evidence import _camera_specs

    robot = np.asarray([[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]])
    fingers = np.asarray(
        [[0.20, 0.40, 0.60], [0.40, 0.80, 1.00]]
    )

    specs = _camera_specs(robot, fingers)

    assert np.allclose(
        specs["gripper_closeup"]["target_world_m"],
        [0.30, 0.60, 0.80],
    )


def test_right_pose_annotations_preserve_numeric_failure_boundary() -> None:
    source = ANNOTATE.read_text(encoding="utf-8")

    assert "VISUAL INSTALLATION/POSE GATE" in source
    assert "Mimic accuracy" in source
    assert "not a workcell placement" in source.lower()


def test_right_pose_review_has_all_required_raw_and_annotated_records() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["scope"] == (
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
    )
    assert report["visual_installation_pose_gate"] == "PASS"
    assert report["numeric_runtime_status"] == "PARTIAL"
    assert report["mimic_accuracy"] == "FAIL"
    assert report["workcell_placement_verified"] is False
    assert report["task8"] == "NOT_RUN"
    assert len(report["records"]) == 7
    assert {
        item["phase"] for item in report["records"]
    } == {
        "home_reference",
        "waist_positive",
        "waist_negative",
        "gripper_open",
        "gripper_partially_closed",
        "gripper_closed",
        "gripper_maximum_legal_aperture",
    }
    assert all(
        item["raw"]["visual_model_review"] == "PASS"
        and item["annotated"]["visual_model_review"] == "PASS"
        for item in report["records"]
    )

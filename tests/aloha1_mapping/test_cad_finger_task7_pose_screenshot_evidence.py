from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = (
    ROOT
    / "tools/capture_aloha_viper_cad_finger_task7_pose_evidence.py"
)
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_pose_screenshot_review.json"
)


def test_capture_does_not_fabricate_follower_right() -> None:
    source = CAPTURE.read_text(encoding="utf-8")
    assert "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT" in source
    assert "mirror" not in source.lower()
    assert "follower_right" not in source


def test_pose_review_preserves_scope_and_certified_states() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert report["task7"] == "PARTIAL"
    assert report["task8"] == "NOT_RUN"
    assert report["right_arm"]["status"] == "NOT_RUN"
    assert report["right_arm"]["blocker"] == (
        "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
    )
    assert report["left_arm"]["status"] == "PASS"
    assert {
        (item["view"], item["phase"])
        for item in report["left_arm"]["records"]
    } == {
        ("full_arm_oblique", "open_maximum_legal_aperture"),
        ("full_arm_oblique", "partially_closed"),
        ("full_arm_oblique", "closed"),
        ("gripper_closeup", "open_maximum_legal_aperture"),
        ("gripper_closeup", "partially_closed"),
        ("gripper_closeup", "closed"),
    }
    assert all(
        item["raw"]["visual_model_review"] == "PASS"
        and item["annotated"]["visual_model_review"] == "PASS"
        for item in report["left_arm"]["records"]
    )
    assert report["left_arm"]["fixed_camera_within_each_view"] is True
    assert report["left_arm"]["states_visually_distinct"] is True
    assert report["source_runtime_evidence"]["numeric_structure"] == "PASS"
    assert report["source_runtime_evidence"]["bottle_static_hold"] == "PASS"
    assert report["screenshot_role"] == (
        "AUXILIARY_EVIDENCE_NOT_PHYSICS_ACCEPTANCE"
    )

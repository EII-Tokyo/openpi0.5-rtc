from __future__ import annotations

from tools.finalize_aloha1_cad_derived_attempt7_review import classify_review_status
from tools.finalize_aloha1_cad_derived_attempt7_review import validate_visual_decision


def test_review_stays_partial_until_exact_videos_are_user_confirmed() -> None:
    status = classify_review_status(
        machine_status="PASS",
        visual_status="PASS",
        user_confirmation="NOT_RUN",
    )

    assert status == "PARTIAL"


def test_visual_decision_requires_all_critical_checks() -> None:
    decision = {
        "full_arm_visible": True,
        "gripper_and_bottle_visible": True,
        "initial_pose_distinct": True,
        "bottle_direction_visible": True,
        "gripper_points_downward": True,
        "phases_visibly_distinct": True,
        "vertical_lift_visible": True,
        "hold_end_visible": True,
        "world_z_visually_upright": True,
        "critical_occlusion": False,
    }

    assert validate_visual_decision(decision) == "PASS"
    decision["critical_occlusion"] = True
    assert validate_visual_decision(decision) == "FAIL"

from __future__ import annotations

from tools.aloha1_mapping.task7_failure_screenshot_gate import classify_review


def test_legible_failure_image_without_finger_geometry_gate_is_not_top_level_pass() -> None:
    result = classify_review(
        requested_visual_status="PASS",
        capture={
            "visual_review_checks": {
                "whole_arm_or_failure_region_visible": "PASS",
                "collision_overlay_visible": "PASS",
                "labels_do_not_hide_failure_region": "PASS",
                "failure_reason_marked": "PASS",
                "raw_and_annotated_are_distinct": "PASS",
            }
        },
    )

    assert result["status"] == "PARTIAL"
    assert result["visual_model_review"] == "PASS_LEGIBILITY_ONLY"
    assert result["visual_evidence_legibility"] == "PASS"
    assert result["finger_installation_and_collision_gate"] == "NOT_RUN"
    assert result["reason"] == "FINGER_GEOMETRY_AND_RUNTIME_STATE_NOT_EVALUATED"


def test_verified_legal_finger_state_can_close_the_screenshot_gate() -> None:
    result = classify_review(
        requested_visual_status="PASS",
        capture={
            "visual_review_checks": {
                "whole_arm_or_failure_region_visible": "PASS",
                "collision_overlay_visible": "PASS",
                "labels_do_not_hide_failure_region": "PASS",
                "failure_reason_marked": "PASS",
                "raw_and_annotated_are_distinct": "PASS",
            },
            "finger_geometry_gate": {
                "runtime_state_source": "fresh_physics_readback",
                "joint_positions_within_authored_limits": True,
                "left_and_right_colliders_have_distinct_rigid_body_owners": True,
                "cad_handedness_mapping_verified": True,
                "inward_surfaces_opposed": True,
                "finger_pair_overlap": False,
            },
        },
    )

    assert result["status"] == "PASS"
    assert result["visual_model_review"] == "PASS"
    assert result["visual_evidence_legibility"] == "PASS"
    assert result["finger_installation_and_collision_gate"] == "PASS"
    assert result["reason"] == "VISUAL_AND_FINGER_GEOMETRY_GATES_VERIFIED"

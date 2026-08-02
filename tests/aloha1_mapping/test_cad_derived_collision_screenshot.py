from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "tools/capture_aloha1_cad_derived_collision_evidence.py"
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_cad_derived_collision_screenshot_review.json"
)


def test_capture_contract_covers_both_arms_and_collision_overlay() -> None:
    source = CAPTURE.read_text(encoding="utf-8")
    assert "/persistent/physics/visualizationDisplayColliders" in source
    assert "follower_left" in source
    assert "follower_right" in source
    assert "physics_collider_overlay" in source
    assert "AUTHORED_COLLIDER_GEOMETRY_NOT_COOKED_HULL_READBACK" in source
    assert "Task 8" not in source


def test_final_review_requires_every_raw_and_annotated_capture() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["task8"] == "NOT_RUN"
    assert report["capture_count"] == 12
    assert report["all_raw_visual_reviews_pass"] is True
    assert report["all_annotated_visual_reviews_pass"] is True
    assert report["fixed_camera_within_view"] is True
    assert report["normal_overlay_same_pose"] is True
    assert {
        (record["state_id"], record["view"], record["mode"])
        for record in report["captures"]
    } == {
        (state, view, mode)
        for state in ("home_reference", "replacement_sample02", "replacement_sample05")
        for view in ("whole_workcell_oblique", "true_top")
        for mode in ("normal", "physics_collider_overlay")
    }
    for record in report["captures"]:
        assert Path(record["raw"]["absolute_path"]).is_file()
        assert Path(record["annotated"]["absolute_path"]).is_file()
        assert record["raw"]["visual_model_review"] == "PASS"
        assert record["annotated"]["visual_model_review"] == "PASS"
        assert record["simulation"]["robots"] == [
            "follower_left",
            "follower_right",
        ]
        assert record["simulation"]["task8"] == "NOT_RUN"

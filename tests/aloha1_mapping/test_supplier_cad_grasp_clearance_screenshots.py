from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_clearance_screenshot_review.json"
)


def test_clearance_screenshot_review_contains_distinct_top_and_side_pairs() -> None:
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["capture_count"] == 4
    assert payload["task8"] == "NOT_RUN"
    assert payload["all_individually_visually_reviewed"] is True
    expected = {
        ("rejected_run13", "true_world_top"),
        ("rejected_run13", "world_side"),
        ("corrected_cad", "true_world_top"),
        ("corrected_cad", "world_side"),
    }
    actual = {
        (record["state"], record["view"])
        for record in payload["captures"]
    }
    assert actual == expected

    for record in payload["captures"]:
        assert record["visual_review"] == "PASS"
        assert record["raw_sha256"] != record["annotated_sha256"]
        assert Path(record["raw_absolute_path"]).is_file()
        assert Path(record["annotated_absolute_path"]).is_file()
        assert record["width_px"] >= 1200
        assert record["height_px"] >= 800
        assert record["camera_projection"] == "ORTHOGRAPHIC"
        assert record["complete_gripper_visible"] is True
        assert record["left_right_fingers_visible"] is True
        assert record["bottle_visible"] is True
        assert record["coordinate_axes_visible"] is True
        if record["view"] == "world_side":
            assert record["ee_and_grasp_frames_distinguished"] is True
            assert (
                record["ee_grasp_frame_projection_status"]
                == "VISIBLE_IN_WORLD_SIDE"
            )
        else:
            assert record["ee_and_grasp_frames_distinguished"] is False
            assert (
                record["ee_grasp_frame_projection_status"]
                == "NOT_OBSERVABLE_DUE_TO_PROJECTION_ALONG_GRIPPER_X"
            )

    by_key = {
        (record["state"], record["view"]): record
        for record in payload["captures"]
    }
    for view in ("true_world_top", "world_side"):
        rejected = by_key[("rejected_run13", view)]
        corrected = by_key[("corrected_cad", view)]
        assert rejected["camera_matrix"] == corrected["camera_matrix"]
        assert rejected["geometry_signature"] != corrected[
            "geometry_signature"
        ]

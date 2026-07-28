from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle_screenshot_review.json"
)


def test_final_bottle_screenshot_review_passes_every_raw_and_annotation() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["raw_capture_count"] == 4
    assert report["annotated_capture_count"] == 4
    assert report["all_raw_vision_reviews_pass"] is True
    assert report["all_annotated_vision_reviews_pass"] is True
    assert report["fixed_camera_across_phases"] is True
    assert report["runtime_report_status"] == "PASS"
    assert report["runtime_repeat_summary"]["pass_count"] == 20
    assert report["runtime_repeat_summary"]["deterministic"] is True
    assert report["screenshot_is_auxiliary"] is True
    assert report["task8"] == "NOT_RUN"
    assert {
        record["phase"] for record in report["records"]
    } == {"open", "bilateral_contact", "release", "hold_end"}
    for record in report["records"]:
        assert record["raw"]["visual_model_review"] == "PASS"
        assert record["annotated"]["visual_model_review"] == "PASS"
        assert Path(record["raw"]["absolute_path"]).is_file()
        assert Path(record["annotated"]["absolute_path"]).is_file()


def test_review_preserves_retake_and_contact_projection_history() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    statuses = {
        item["attempt"]: item["status"]
        for item in report["retake_history"]
    }
    assert statuses["attempt3"] == "REJECTED_OCCLUSION_AND_UNCOUNTED_STEP_RISK"
    assert statuses["v3_annotation_first"] == (
        "REJECTED_PHASE_CONTEXT_MISLABEL"
    )
    assert statuses["v3_annotation_attempt2"] == "PASS"
    by_phase = {record["phase"]: record for record in report["records"]}
    assert by_phase["open"]["contact_projection_count"] == 0
    for phase in ("bilateral_contact", "release", "hold_end"):
        assert by_phase[phase]["contact_projection_count"] == 2

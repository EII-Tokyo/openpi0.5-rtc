from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_visual_review_entrypoint_requires_manual_decisions() -> None:
    source = (
        PROJECT_ROOT / "tools/review_aloha1_correct_finger_screenshots.py"
    ).read_text(encoding="utf-8")

    assert "PENDING_VISUAL_MODEL_REVIEW" in source
    assert "--review-decisions" in source
    assert "annotated_absolute_path" in source
    assert "stage_comparisons" in source


def test_visual_review_report_has_one_reviewed_pair_per_capture() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_visual_screenshot_review.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_all_screenshot_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert report["review_method"] == "VISUAL_MODEL_MANUAL_SELF_REVIEW"
    assert report["capture_pair_count"] == manifest["observed_capture_count"]
    assert report["all_runtime_conclusions_require_numeric_evidence"] is True
    for item in report["captures"]:
        assert item["visual_review"]["status"] == "PASS"
        assert item["visual_review"]["reviewed_by"] == "Codex visual model"
        assert item["visual_review"]["objects_visible"]
        assert item["visual_review"]["view_exposes_test_target"] is True
        assert Path(item["original_absolute_path"]).is_file()
        assert Path(item["annotated_absolute_path"]).is_file()
        assert len(item["original_sha256"]) == 64
        assert len(item["annotated_sha256"]) == 64
        assert item["detection_target"]
        assert item["acceptance_criteria"]
        assert "retake_history" in item


def test_open_contact_release_hold_comparisons_are_explicit() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_visual_screenshot_review.json"
        ).read_text(encoding="utf-8")
    )

    comparisons = report["stage_comparisons"]
    assert len(comparisons) == 4
    for item in comparisons:
        assert item["status"] == "PASS"
        assert item["same_camera_anchor"] is True
        assert item["open_vs_contact_visually_distinct"] is True
        assert item["release_vs_hold_runtime_state_distinct"] is True
        assert item["runtime_drop_m"] < item["drop_gate_m"]

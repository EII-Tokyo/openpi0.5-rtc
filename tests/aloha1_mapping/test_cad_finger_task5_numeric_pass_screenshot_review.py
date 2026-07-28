from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tools/finalize_aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.py"
)


def test_finalizer_records_visual_review_and_auxiliary_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY" in source
    assert "open_maximum_legal_aperture" in source
    assert "partially_closed" in source
    assert "closed" in source
    assert "not_same_frame_physics_evidence" in source
    assert '"bottle_contact_grasp": "NOT_RUN"' in source
    assert '"task7": "NOT_RUN"' in source
    assert '"task8": "NOT_RUN"' in source
    assert "REJECTED_STALE_CAMERA_TARGET" in source


def test_generated_review_is_machine_readable_when_present() -> None:
    report_path = (
        ROOT
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json"
    )
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["capture_count"] == 3
    assert report["gates"]["fixed_camera_exact"] is True
    assert report["gates"]["all_raw_visual_reviews_pass"] is True
    assert report["gates"]["all_annotated_visual_reviews_pass"] is True
    assert report["scope"]["bottle_contact_grasp"] == "NOT_RUN"

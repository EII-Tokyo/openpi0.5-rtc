from __future__ import annotations

from tools.validate_aloha1_grasp_initialization_negative_controls import _camera_clipping_range
from tools.validate_aloha1_grasp_initialization_negative_controls import _camera_visual_scope
from tools.validate_aloha1_grasp_initialization_negative_controls import _closeup_camera_distance
from tools.validate_aloha1_grasp_initialization_negative_controls import aggregate_controls


def _record(scenario: str, *, observed: list[str], status: str) -> dict[str, object]:
    return {
        "scenario": scenario,
        "status": status,
        "observed_failure_codes": observed,
        "fresh_process": True,
        "stage_immutable": True,
        "raw_screenshot": "/tmp/raw.png",
        "annotated_screenshot": "/tmp/annotated.png",
        "visual_model_review": "PASS",
    }


def test_four_expected_negative_controls_form_a_pass() -> None:
    report = aggregate_controls(
        [
            _record(
                "STATIC_LOAD_WITHOUT_RESET",
                observed=["FAIL_INITIALIZATION_CONTRACT"],
                status="EXPECTED_FAIL_OBSERVED",
            ),
            _record(
                "ILLEGAL_Q_ZERO",
                observed=["FINGER_PAIR_OVERLAP"],
                status="EXPECTED_FAIL_OBSERVED",
            ),
            _record(
                "LEGAL_OPEN_CLOSE_SWEEP",
                observed=[],
                status="PASS",
            ),
            _record(
                "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
                observed=["FINGER_LIMIT_VIOLATION"],
                status="EXPECTED_FAIL_OBSERVED",
            ),
        ]
    )

    assert report["status"] == "PASS"
    assert report["control_count"] == 4
    assert report["all_fresh_processes"] is True
    assert report["source_or_final_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_missing_expected_failure_is_rejected() -> None:
    records = [
        _record(
            "STATIC_LOAD_WITHOUT_RESET",
            observed=[],
            status="FAIL",
        ),
        _record(
            "ILLEGAL_Q_ZERO",
            observed=["FINGER_PAIR_OVERLAP"],
            status="EXPECTED_FAIL_OBSERVED",
        ),
        _record("LEGAL_OPEN_CLOSE_SWEEP", observed=[], status="PASS"),
        _record(
            "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
            observed=["ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION"],
            status="EXPECTED_FAIL_OBSERVED",
        ),
    ]

    report = aggregate_controls(records)

    assert report["status"] == "FAIL"
    assert report["failed_controls"] == [
        "STATIC_LOAD_WITHOUT_RESET",
        "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
    ]


def test_closeup_camera_near_clip_does_not_remove_fingers() -> None:
    near_m, far_m = _camera_clipping_range("closeup")

    assert 0.0 < near_m < 0.42
    assert far_m > 0.42


def test_closeup_camera_uses_isolated_collider_scope() -> None:
    assert _camera_visual_scope("overview") == "FULL_STAGE_WITH_COLLIDER_OVERLAY"
    assert _camera_visual_scope("closeup") == "FINGER_COLLIDERS_ONLY"


def test_closeup_camera_distance_scales_with_open_finger_span() -> None:
    assert _closeup_camera_distance(0.10) == 0.75
    assert _closeup_camera_distance(0.25) == 1.0


def test_pending_visual_review_is_rejected_when_required() -> None:
    records = [
        _record(
            "STATIC_LOAD_WITHOUT_RESET",
            observed=["FAIL_INITIALIZATION_CONTRACT"],
            status="EXPECTED_FAIL_OBSERVED",
        ),
        _record(
            "ILLEGAL_Q_ZERO",
            observed=["FINGER_PAIR_OVERLAP"],
            status="EXPECTED_FAIL_OBSERVED",
        ),
        _record("LEGAL_OPEN_CLOSE_SWEEP", observed=[], status="PASS"),
        _record(
            "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
            observed=["FINGER_LIMIT_VIOLATION"],
            status="EXPECTED_FAIL_OBSERVED",
        ),
    ]
    records[2]["visual_model_review"] = "PENDING"

    report = aggregate_controls(records, require_visual_review=True)

    assert report["status"] == "FAIL"
    assert report["gates"]["LEGAL_OPEN_CLOSE_SWEEP"][
        "visual_model_review_pass"
    ] is False

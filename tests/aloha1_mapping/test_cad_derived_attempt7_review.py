from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from tools import finalize_aloha1_cad_derived_attempt7_review as review
from tools.finalize_aloha1_cad_derived_attempt7_review import classify_review_status
from tools.finalize_aloha1_cad_derived_attempt7_review import validate_visual_decision


def test_review_stays_partial_until_exact_videos_are_user_confirmed() -> None:
    status = classify_review_status(
        machine_status="PASS",
        visual_status="PASS",
        user_confirmation="NOT_RUN",
    )

    assert status == "PARTIAL"


def test_confirmation_validator_is_available() -> None:
    assert hasattr(review, "validate_user_confirmation")


def test_build_report_requires_confirmation_manifest() -> None:
    assert "confirmation" in inspect.signature(review.build_report).parameters


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirmation_fixture(tmp_path: Path) -> tuple[dict, list[dict], dict]:
    stage_path = tmp_path / "stage.usda"
    stage_path.write_text("#usda 1.0\n", encoding="utf-8")
    stage = {
        "absolute_path": str(stage_path.resolve()),
        "sha256_before": _hash(stage_path),
        "sha256_after": _hash(stage_path),
    }
    samples = []
    confirmations = []
    for index in range(1, 6):
        sample_id = f"sample_{index:02d}"
        video_path = tmp_path / f"{sample_id}.mp4"
        video_path.write_bytes(sample_id.encode("utf-8"))
        video = {
            "absolute_path": str(video_path.resolve()),
            "sha256": _hash(video_path),
            "frame_count": 100 + index,
            "fps": 60,
            "probe": {"resolution": [1440, 680]},
        }
        samples.append({"sample_id": sample_id, "videos": {"annotated": video}})
        confirmations.append(
            {
                "sample_id": sample_id,
                "annotated_video_absolute_path": video["absolute_path"],
                "annotated_video_sha256": video["sha256"],
                "frame_count": video["frame_count"],
                "fps": video["fps"],
                "resolution": video["probe"]["resolution"],
                "status": "PASS",
            }
        )
    confirmation = {
        "schema_version": 1,
        "attempt_id": "Z_UP_ATTEMPT7",
        "status": "PASS",
        "confirmation_source": "USER_EXPLICIT_CONFIRMATION",
        "confirmation_text": "全部确认",
        "confirmation_date": "2026-08-02",
        "stage": {
            "absolute_path": stage["absolute_path"],
            "sha256": stage["sha256_after"],
        },
        "samples": confirmations,
    }
    return confirmation, samples, stage


def test_confirmation_accepts_five_hash_bound_videos(tmp_path: Path) -> None:
    confirmation, samples, stage = _confirmation_fixture(tmp_path)

    result = review.validate_user_confirmation(
        confirmation,
        sample_reports=samples,
        stage=stage,
    )

    assert result["status"] == "PASS"
    assert result["sample_count"] == 5


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["samples"].pop(), "five unique samples"),
        (
            lambda value: value["samples"].__setitem__(
                4, dict(value["samples"][0])
            ),
            "five unique samples",
        ),
        (
            lambda value: value["samples"][0].__setitem__(
                "annotated_video_sha256", "0" * 64
            ),
            "video hash",
        ),
        (
            lambda value: value["stage"].__setitem__("sha256", "0" * 64),
            "Stage hash",
        ),
        (
            lambda value: value.__setitem__("attempt_id", "attempt_6"),
            "attempt",
        ),
        (
            lambda value: value.__setitem__("status", "NOT_RUN"),
            "explicit PASS",
        ),
    ],
)
def test_confirmation_rejects_unbound_or_incomplete_evidence(
    tmp_path: Path, mutation: object, match: str
) -> None:
    confirmation, samples, stage = _confirmation_fixture(tmp_path)
    mutation(confirmation)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        review.validate_user_confirmation(
            confirmation,
            sample_reports=samples,
            stage=stage,
        )


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

from __future__ import annotations

import hashlib
from itertools import pairwise
import json
from pathlib import Path

import pytest

from tools.build_aloha1_task7b2_horizontal_video import REQUIRED_PHASES
from tools.build_aloha1_task7b2_horizontal_video import select_review_frames
from tools.build_aloha1_task7b2_horizontal_video import validate_frame_manifest
from tools.finalize_aloha1_task7b2_horizontal_video_review import finalize_video_review


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_manifest(tmp_path: Path) -> dict[str, object]:
    frames = []
    phases = [
        "release_dynamic",
        "support_settle",
        "open_pregrasp",
        "vertical_descent",
        "bilateral_contact",
        "closing_preload",
        "vertical_lift",
        "support_clear",
        "hold_end",
    ]
    for frame, phase in enumerate(phases):
        views = {}
        for view in ("overview", "gripper_closeup"):
            image = tmp_path / view / f"{frame:06d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{view}:{frame}".encode())
            views[view] = {
                "absolute_path": str(image.resolve()),
                "sha256": _sha256(image),
                "resolution": [960, 540],
            }
        frames.append(
            {
                "physics_frame": frame,
                "time_s": frame / 60.0,
                "phase": phase,
                "views": views,
            }
        )
    return {
        "schema_version": 1,
        "runtime_trial_signature": "abc123",
        "frames": frames,
    }


def test_complete_two_view_manifest_is_required(tmp_path: Path) -> None:
    manifest = _frame_manifest(tmp_path)
    result = validate_frame_manifest(manifest)

    assert result["first_physics_frame"] == 0
    assert result["last_physics_frame"] == 8
    assert result["missing_physics_frames"] == []
    assert result["views"] == ["overview", "gripper_closeup"]
    assert set(result["phase_frame_ranges"]) == set(REQUIRED_PHASES)

    manifest["frames"][4]["views"].pop("gripper_closeup")
    with pytest.raises(ValueError, match="gripper_closeup"):
        validate_frame_manifest(manifest)


def test_review_samples_cover_boundaries_and_half_second_intervals() -> None:
    ranges = {
        "release_dynamic": [1, 1],
        "support_settle": [2, 121],
        "open_pregrasp": [122, 131],
        "vertical_descent": [132, 150],
        "bilateral_contact": [151, 165],
        "closing_preload": [152, 164],
        "vertical_lift": [166, 166],
        "support_clear": [167, 167],
        "hold_end": [168, 287],
    }
    samples = select_review_frames(
        first_frame=0,
        last_frame=287,
        phase_frame_ranges=ranges,
        max_interval_frames=30,
    )

    assert samples[0] == 0
    assert samples[-1] == 287
    assert max(b - a for a, b in pairwise(samples)) <= 30
    for start, end in ranges.values():
        assert start in samples
        assert end in samples


def test_review_cannot_promote_before_both_views_pass(tmp_path: Path) -> None:
    candidates = []
    for view in ("overview", "gripper_closeup"):
        raw = tmp_path / f"{view}_raw.mp4"
        annotated = tmp_path / f"{view}_annotated.mp4"
        raw.write_bytes(f"{view}:raw".encode())
        annotated.write_bytes(f"{view}:annotated".encode())
        candidates.append(
            {
                "view_name": view,
                "runtime_trial_signature": "abc123",
                "raw_candidate_absolute_path": str(raw.resolve()),
                "raw_candidate_sha256": _sha256(raw),
                "annotated_candidate_absolute_path": str(annotated.resolve()),
                "annotated_candidate_sha256": _sha256(annotated),
                "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
                "promotion_status": "NOT_REVIEWED",
            }
        )
    manifest = {
        "attempt_id": "attempt_test",
        "physical_trial_status": "FAIL",
        "machine_conclusion": "HORIZONTAL_PICKUP_NOT_VERIFIED",
        "videos": candidates,
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(manifest), encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_test",
                "reviewed_by": "Codex visual model",
                "views": {
                    "overview": {
                        "status": "PASS",
                        "reviewed_sample_frames": [0, 1],
                        "retake_reason": None,
                    },
                    "gripper_closeup": {
                        "status": "FAIL",
                        "reviewed_sample_frames": [0, 1],
                        "retake_reason": "finger inner surface occluded",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
    )

    assert report["status"] == "FAIL"
    assert report["promotion_status"] == "REJECTED_VISUAL_REVIEW"
    assert not (tmp_path / "verified").exists()


def test_visual_pass_preserves_physical_fail_label(tmp_path: Path) -> None:
    candidates = []
    for view in ("overview", "gripper_closeup"):
        raw = tmp_path / f"{view}_raw.mp4"
        annotated = tmp_path / f"{view}_annotated.mp4"
        raw.write_bytes(f"{view}:raw".encode())
        annotated.write_bytes(f"{view}:annotated".encode())
        candidates.append(
            {
                "view_name": view,
                "runtime_trial_signature": "same-signature",
                "raw_candidate_absolute_path": str(raw.resolve()),
                "raw_candidate_sha256": _sha256(raw),
                "annotated_candidate_absolute_path": str(annotated.resolve()),
                "annotated_candidate_sha256": _sha256(annotated),
                "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
                "promotion_status": "NOT_REVIEWED",
            }
        )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_test",
                "physical_trial_status": "FAIL",
                "machine_conclusion": "HORIZONTAL_PICKUP_NOT_VERIFIED",
                "videos": candidates,
            }
        ),
        encoding="utf-8",
    )
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_test",
                "reviewed_by": "Codex visual model",
                "views": {
                    view: {
                        "status": "PASS",
                        "reviewed_sample_frames": [0, 1],
                        "retake_reason": None,
                    }
                    for view in ("overview", "gripper_closeup")
                },
            }
        ),
        encoding="utf-8",
    )

    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
    )

    assert report["status"] == "PASS"
    assert report["physical_trial_status"] == "FAIL"
    assert report["machine_conclusion"] == "HORIZONTAL_PICKUP_NOT_VERIFIED"
    assert report["promotion_status"] == ("PROMOTED_VISUAL_EVIDENCE_PHYSICAL_FAIL")
    assert all(
        record["verified_raw_sha256"] == _sha256(Path(record["verified_raw_absolute_path"]))
        for record in report["videos"]
    )

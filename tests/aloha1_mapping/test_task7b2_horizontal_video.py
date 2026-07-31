from __future__ import annotations

import hashlib
from itertools import pairwise
import json
from pathlib import Path

from PIL import Image
import pytest

from tools import build_aloha1_task7b2_horizontal_video as video_builder
from tools.build_aloha1_task7b2_horizontal_video import REQUIRED_PHASES
from tools.build_aloha1_task7b2_horizontal_video import select_review_frames
from tools.build_aloha1_task7b2_horizontal_video import validate_frame_manifest
from tools.finalize_aloha1_task7b2_horizontal_video_review import finalize_video_review

SOURCE_VIEWS = ("overview", "gripper_closeup")
COMPOSITE_VIEW = "full_arm_composite"
LAYOUT = "FULL_ARM_WITH_SYNCHRONIZED_GRIPPER_INSET"
NUMERIC_EVIDENCE_SCOPE = "WORLD_AABB_CAMERA_FRUSTUM_AND_IMAGE_BOUNDS_ONLY"
OCCLUSION_EVALUATION_STATUS = "NOT_EVALUATED_REQUIRES_VISUAL_REVIEW"
REQUIRED_FULL_ARM_PRIMS = (
    "/World/ALOHA1/follower_base",
    "/World/Bottle500",
    "/World/Table",
)
REQUIRED_FULL_ARM_LINKS = ("shoulder", "elbow", "forearm", "wrist", "gripper")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_manifest(
    tmp_path: Path,
    *,
    valid_images: bool = False,
) -> dict[str, object]:
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
        for view in SOURCE_VIEWS:
            image = tmp_path / view / f"{frame:06d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            if valid_images:
                color = (210, 30, 30) if view == "overview" else (20, 40, 220)
                Image.new("RGB", (960, 540), color).save(image)
            else:
                image.write_bytes(f"{view}:{frame}".encode())
            views[view] = {
                "absolute_path": str(image.resolve()),
                "sha256": _sha256(image),
                "resolution": [960, 540],
                "physics_frame": frame,
                "time_s": frame / 60.0,
                "runtime_trial_signature": "abc123",
            }
        views["overview"]["framing_evidence"] = {
            "projected_in_frame_prims": list(REQUIRED_FULL_ARM_PRIMS),
            "projected_in_frame_links": list(REQUIRED_FULL_ARM_LINKS),
            "numeric_evidence_scope": NUMERIC_EVIDENCE_SCOPE,
            "occlusion_evaluation_status": OCCLUSION_EVALUATION_STATUS,
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
        "required_full_arm_prims": list(REQUIRED_FULL_ARM_PRIMS),
        "required_full_arm_links": list(REQUIRED_FULL_ARM_LINKS),
        "frames": frames,
    }


def _candidate_manifest(
    tmp_path: Path,
    *,
    physical_trial_status: str,
) -> tuple[Path, str]:
    candidates = []
    confirmed_hash = ""
    for view in (*SOURCE_VIEWS, COMPOSITE_VIEW):
        raw = tmp_path / f"{view}_raw.mp4"
        annotated = tmp_path / f"{view}_annotated.mp4"
        raw.write_bytes(f"{view}:raw".encode())
        annotated.write_bytes(f"{view}:annotated".encode())
        record = {
            "view_name": view,
            "runtime_trial_signature": "same-signature",
            "frame_count": 9,
            "first_physics_frame": 0,
            "last_physics_frame": 8,
            "raw_candidate_absolute_path": str(raw.resolve()),
            "raw_candidate_sha256": _sha256(raw),
            "annotated_candidate_absolute_path": str(annotated.resolve()),
            "annotated_candidate_sha256": _sha256(annotated),
            "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
            "promotion_status": "NOT_REVIEWED",
        }
        if view == COMPOSITE_VIEW:
            record.update(
                {
                    "evidence_role": "PRIMARY_FULL_ARM_EVIDENCE",
                    "layout": LAYOUT,
                    "source_views": list(SOURCE_VIEWS),
                    "layout_regions": {
                        "full_arm": {
                            "source_view": "overview",
                            "width_fraction": 2 / 3,
                        },
                        "gripper_inset": {
                            "source_view": "gripper_closeup",
                            "width_fraction": 1 / 3,
                        },
                    },
                    "framing_evidence_input": {
                        "validated_for_every_physics_frame": True,
                        "numeric_evidence_scope": NUMERIC_EVIDENCE_SCOPE,
                        "occlusion_evaluation_status": (
                            OCCLUSION_EVALUATION_STATUS
                        ),
                    },
                }
            )
            confirmed_hash = record["annotated_candidate_sha256"]
        else:
            record["evidence_role"] = "SYNCHRONIZED_SOURCE"
        candidates.append(record)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_test",
                "runtime_trial_signature": "same-signature",
                "physical_trial_status": physical_trial_status,
                "machine_conclusion": (
                    "HORIZONTAL_PICKUP_VERIFIED"
                    if physical_trial_status == "PASS"
                    else "HORIZONTAL_PICKUP_NOT_VERIFIED"
                ),
                "videos": candidates,
            }
        ),
        encoding="utf-8",
    )
    return candidate_path, confirmed_hash


def _decisions(tmp_path: Path, *, status: str) -> Path:
    decisions_path = tmp_path / "decisions.json"
    reviewed_video = tmp_path / f"{COMPOSITE_VIEW}_annotated.mp4"
    decisions_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_test",
                "reviewed_by": "Codex visual model",
                "views": {
                    COMPOSITE_VIEW: {
                        "status": status,
                        "reviewed_sample_frames": [0, 1],
                        "reviewed_entire_video": True,
                        "reviewed_annotated_video_sha256": _sha256(reviewed_video),
                        "complete_full_arm_confirmed": status == "PASS",
                        "no_full_arm_occlusion_confirmed": status == "PASS",
                        "retake_reason": (None if status == "PASS" else "full arm leaves the frame"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return decisions_path


def test_complete_synchronized_sources_and_full_arm_framing_are_required(
    tmp_path: Path,
) -> None:
    manifest = _frame_manifest(tmp_path)
    result = validate_frame_manifest(manifest)

    assert result["first_physics_frame"] == 0
    assert result["last_physics_frame"] == 8
    assert result["missing_physics_frames"] == []
    assert result["views"] == list(SOURCE_VIEWS)
    assert result["required_full_arm_prims"] == list(REQUIRED_FULL_ARM_PRIMS)
    assert result["required_full_arm_links"] == list(REQUIRED_FULL_ARM_LINKS)
    assert result["numeric_evidence_scope"] == NUMERIC_EVIDENCE_SCOPE
    assert result["occlusion_evaluation_status"] == OCCLUSION_EVALUATION_STATUS
    assert set(result["phase_frame_ranges"]) == set(REQUIRED_PHASES)

    manifest["frames"][4]["views"].pop("gripper_closeup")
    with pytest.raises(ValueError, match="gripper_closeup"):
        validate_frame_manifest(manifest)


def test_manifest_fails_closed_on_missing_framing_or_source_sync(
    tmp_path: Path,
) -> None:
    manifest = _frame_manifest(tmp_path / "missing")
    manifest["frames"][4]["views"]["overview"]["framing_evidence"][
        "projected_in_frame_links"
    ].remove("elbow")
    with pytest.raises(ValueError, match="elbow"):
        validate_frame_manifest(manifest)

    manifest = _frame_manifest(tmp_path / "desynchronized")
    manifest["frames"][4]["views"]["gripper_closeup"]["time_s"] += 0.25
    with pytest.raises(ValueError, match="time_s"):
        validate_frame_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("numeric_evidence_scope", "PIXEL_VISIBILITY"),
        ("occlusion_evaluation_status", "VISIBLE"),
    ],
)
def test_manifest_requires_projection_only_scope_and_unevaluated_occlusion(
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    manifest = _frame_manifest(tmp_path)
    manifest["frames"][4]["views"]["overview"]["framing_evidence"][field] = (
        invalid_value
    )

    with pytest.raises(ValueError, match="framing evidence"):
        validate_frame_manifest(manifest)


def test_manifest_rejects_legacy_visible_fields(tmp_path: Path) -> None:
    manifest = _frame_manifest(tmp_path)
    framing = manifest["frames"][4]["views"]["overview"]["framing_evidence"]
    framing["visible_prims"] = framing.pop("projected_in_frame_prims")
    framing["visible_links"] = framing.pop("projected_in_frame_links")

    with pytest.raises(ValueError, match="visible_prims.*visible_links"):
        validate_frame_manifest(manifest)


def test_composite_uses_two_thirds_full_arm_and_synchronized_inset(
    tmp_path: Path,
) -> None:
    manifest = _frame_manifest(tmp_path, valid_images=True)
    records = video_builder.compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_trial_signature=manifest["runtime_trial_signature"],
        output_dir=tmp_path / "composite",
    )

    assert len(records) == len(manifest["frames"])
    first = records[0]
    composite = first["views"][COMPOSITE_VIEW]
    assert composite["layout"] == LAYOUT
    assert composite["source_views"] == list(SOURCE_VIEWS)
    assert composite["physics_frame"] == first["physics_frame"] == 0
    assert composite["time_s"] == first["time_s"] == 0.0
    assert composite["runtime_trial_signature"] == "abc123"
    with Image.open(composite["absolute_path"]) as image:
        assert image.size == (1440, 540)
        assert image.getpixel((100, 100)) == (210, 30, 30)
        assert image.getpixel((1200, 270)) == (20, 40, 220)


@pytest.mark.parametrize(
    ("view", "field", "invalid_value", "match"),
    [
        ("overview", "physics_frame", 1, "physics_frame"),
        ("gripper_closeup", "time_s", 0.25, "time_s"),
        (
            "gripper_closeup",
            "runtime_trial_signature",
            "different",
            "runtime_trial_signature",
        ),
    ],
)
def test_composite_rejects_desynchronized_source_views(
    tmp_path: Path,
    view: str,
    field: str,
    invalid_value: object,
    match: str,
) -> None:
    manifest = _frame_manifest(tmp_path, valid_images=True)
    manifest["frames"][0]["views"][view][field] = invalid_value

    with pytest.raises(ValueError, match=match):
        video_builder.compose_synchronized_frames(
            source_records=manifest["frames"],
            runtime_trial_signature=manifest["runtime_trial_signature"],
            output_dir=tmp_path / "composite",
        )


def test_encoded_composite_pair_requires_60fps_and_identical_frame_count() -> None:
    probe = {
        "r_frame_rate": "60/1",
        "frame_count": 9,
    }
    video_builder.validate_encoded_video_pair(
        view=COMPOSITE_VIEW,
        expected_frame_count=9,
        raw_probe=probe,
        annotated_probe=probe,
    )

    with pytest.raises(RuntimeError, match="frame count"):
        video_builder.validate_encoded_video_pair(
            view=COMPOSITE_VIEW,
            expected_frame_count=9,
            raw_probe=probe,
            annotated_probe={**probe, "frame_count": 8},
        )
    with pytest.raises(RuntimeError, match="60 fps"):
        video_builder.validate_encoded_video_pair(
            view=COMPOSITE_VIEW,
            expected_frame_count=9,
            raw_probe={**probe, "r_frame_rate": "30/1"},
            annotated_probe=probe,
        )


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


def test_annotation_banner_distinguishes_trial_result_from_current_phase() -> None:
    banner = video_builder.annotation_status_banner(
        physical_status="PASS",
        failure_mode="stable_hold",
        phase="support_settle",
        frame=60,
        last_frame=321,
        drive_classification="DIAGNOSTIC_ONLY_FORCE_DRIVE_UNCALIBRATED",
    )

    assert banner["result"] == "TRIAL MACHINE RESULT PASS: stable_hold"
    assert banner["phase"] == "CURRENT PHASE support_settle"
    assert banner["drive"] == "DIAGNOSTIC FORCE DRIVE: UNCALIBRATED"
    assert not banner["result"].startswith("PHYSICAL PASS")

    hold = video_builder.annotation_status_banner(
        physical_status="PASS",
        failure_mode="stable_hold",
        phase="hold_end",
        frame=240,
        last_frame=321,
        drive_classification="DIAGNOSTIC_ONLY_FORCE_DRIVE_UNCALIBRATED",
    )
    final = video_builder.annotation_status_banner(
        physical_status="PASS",
        failure_mode="stable_hold",
        phase="hold_end",
        frame=321,
        last_frame=321,
        drive_classification="DIAGNOSTIC_ONLY_FORCE_DRIVE_UNCALIBRATED",
    )

    assert hold["phase"] == "CURRENT PHASE hold_interval"
    assert final["phase"] == "CURRENT PHASE hold_end"


def test_visual_failure_remains_rejected_visual_review(tmp_path: Path) -> None:
    candidate_path, _ = _candidate_manifest(
        tmp_path,
        physical_trial_status="FAIL",
    )
    decisions_path = _decisions(tmp_path, status="FAIL")

    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
    )

    assert report["status"] == "FAIL"
    assert report["visual_review_status"] == "FAIL"
    assert report["promotion_status"] == "REJECTED_VISUAL_REVIEW"
    assert report["user_video_confirmation"] == "BLOCKED_BY_VISUAL_REVIEW"
    assert not (tmp_path / "verified").exists()


@pytest.mark.parametrize(
    ("field", "invalid_value", "match"),
    [
        ("reviewed_entire_video", False, "entire video"),
        ("complete_full_arm_confirmed", False, "complete full arm"),
        (
            "no_full_arm_occlusion_confirmed",
            False,
            "full-arm occlusion",
        ),
        (
            "reviewed_annotated_video_sha256",
            "0" * 64,
            "reviewed video hash",
        ),
    ],
)
def test_visual_pass_requires_hash_bound_entire_video_full_arm_review(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    match: str,
) -> None:
    candidate_path, _ = _candidate_manifest(
        tmp_path,
        physical_trial_status="PASS",
    )
    decisions_path = _decisions(tmp_path, status="PASS")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions["views"][COMPOSITE_VIEW][field] = invalid_value
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        finalize_video_review(
            candidate_manifest_path=candidate_path,
            decisions_path=decisions_path,
            verified_root=tmp_path / "verified",
            promote=True,
        )


@pytest.mark.parametrize(
    ("path", "invalid_value", "match"),
    [
        (
            ("layout_regions", "full_arm", "width_fraction"),
            0.5,
            "full-arm width fraction",
        ),
        (
            ("layout_regions", "gripper_inset", "width_fraction"),
            0.5,
            "gripper inset width fraction",
        ),
        (
            ("framing_evidence_input", "numeric_evidence_scope"),
            "PIXEL_VISIBILITY",
            "numeric evidence scope",
        ),
    ],
)
def test_finalizer_rejects_invalid_primary_composite_contract(
    tmp_path: Path,
    path: tuple[str, ...],
    invalid_value: object,
    match: str,
) -> None:
    candidate_path, _ = _candidate_manifest(
        tmp_path,
        physical_trial_status="PASS",
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    primary = next(
        item
        for item in candidate["videos"]
        if item["view_name"] == COMPOSITE_VIEW
    )
    target = primary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decisions_path = _decisions(tmp_path, status="PASS")

    with pytest.raises(ValueError, match=match):
        finalize_video_review(
            candidate_manifest_path=candidate_path,
            decisions_path=decisions_path,
            verified_root=tmp_path / "verified",
            promote=True,
        )


@pytest.mark.parametrize("physical_trial_status", ["FAIL", "PASS"])
def test_visual_pass_awaits_explicit_user_confirmation(
    tmp_path: Path,
    physical_trial_status: str,
) -> None:
    candidate_path, confirmed_hash = _candidate_manifest(
        tmp_path,
        physical_trial_status=physical_trial_status,
    )
    decisions_path = _decisions(tmp_path, status="PASS")

    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
    )

    assert report["status"] == "PARTIAL"
    assert report["visual_review_status"] == "PASS"
    assert report["physical_trial_status"] == physical_trial_status
    assert report["promotion_status"] == "AWAITING_USER_VIDEO_CONFIRMATION"
    assert report["user_video_confirmation"] == "PENDING"
    assert report["user_confirmation_target"] == {
        "view_name": COMPOSITE_VIEW,
        "kind": "annotated",
        "sha256": confirmed_hash,
    }
    assert not (tmp_path / "verified").exists()


def test_explicit_user_confirmation_is_bound_to_composite_video_hash(
    tmp_path: Path,
) -> None:
    candidate_path, confirmed_hash = _candidate_manifest(
        tmp_path,
        physical_trial_status="PASS",
    )
    decisions_path = _decisions(tmp_path, status="PASS")

    with pytest.raises(ValueError, match="confirmed video hash"):
        finalize_video_review(
            candidate_manifest_path=candidate_path,
            decisions_path=decisions_path,
            verified_root=tmp_path / "mismatch",
            promote=True,
            user_confirmed_video_sha256="0" * 64,
        )

    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
        user_confirmed_video_sha256=confirmed_hash,
    )

    assert report["status"] == "PASS"
    assert report["user_video_confirmation"] == "CONFIRMED"
    assert report["user_confirmed_video_sha256"] == confirmed_hash
    assert report["promotion_status"] == "PROMOTED_VISUAL_EVIDENCE_PHYSICAL_PASS"
    assert all(
        record["verified_raw_sha256"] == _sha256(Path(record["verified_raw_absolute_path"]))
        for record in report["videos"]
    )


def test_user_confirmation_target_requires_current_composite_video_hash(
    tmp_path: Path,
) -> None:
    candidate_path, confirmed_hash = _candidate_manifest(
        tmp_path,
        physical_trial_status="PASS",
    )
    decisions_path = _decisions(tmp_path, status="PASS")
    composite = tmp_path / f"{COMPOSITE_VIEW}_annotated.mp4"
    composite.write_bytes(b"tampered-after-manifest")

    with pytest.raises(ValueError, match="annotated candidate hash mismatch"):
        finalize_video_review(
            candidate_manifest_path=candidate_path,
            decisions_path=decisions_path,
            verified_root=tmp_path / "verified",
            promote=False,
            user_confirmed_video_sha256=confirmed_hash,
        )

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
import pytest

from tools.build_aloha1_grasp_20cm_video import REQUIRED_PHASES
from tools.build_aloha1_grasp_20cm_video import annotate_collision_evidence
from tools.build_aloha1_grasp_20cm_video import annotate_composite_frames
from tools.build_aloha1_grasp_20cm_video import build_candidate_manifest
from tools.build_aloha1_grasp_20cm_video import build_review_contact_sheets
from tools.build_aloha1_grasp_20cm_video import compose_synchronized_frames
from tools.build_aloha1_grasp_20cm_video import encode_frame_sequence
from tools.build_aloha1_grasp_20cm_video import required_phases_for_report
from tools.build_aloha1_grasp_20cm_video import validate_encoded_video_pair
from tools.build_aloha1_grasp_20cm_video import validate_frame_manifest
from tools.finalize_aloha1_grasp_20cm_visual_review import _markdown
from tools.finalize_aloha1_grasp_20cm_visual_review import apply_user_confirmation
from tools.finalize_aloha1_grasp_20cm_visual_review import complete_sheet_frame_coverage
from tools.finalize_aloha1_grasp_20cm_visual_review import normalized_rejected_attempts

VIEWS = ("overview", "gripper_closeup")
FULL_ARM_LINKS = (
    "base",
    "shoulder",
    "elbow",
    "forearm",
    "wrist",
    "gripper",
)


def test_visual_review_sheet_coverage_supports_full_runtime_length() -> None:
    sheets = [
        {"frame_numbers": list(range(1, 501))},
        {"frame_numbers": list(range(501, 913))},
    ]

    assert complete_sheet_frame_coverage(
        sheets,
        expected_frame_count=912,
    )
    assert not complete_sheet_frame_coverage(
        sheets,
        expected_frame_count=913,
    )


def test_visual_review_rejected_attempts_must_be_explicit() -> None:
    assert normalized_rejected_attempts(None) == []
    assert normalized_rejected_attempts(
        [{"run": "repeat_v5", "status": "REJECTED_OPEN_FRAME_TOO_EARLY"}]
    ) == [
        {
            "run": "repeat_v5",
            "status": "REJECTED_OPEN_FRAME_TOO_EARLY",
        }
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> dict[str, object]:
    phases = (
        "RELEASE_DYNAMIC",
        "SETTLE",
        "OPEN_PREGRASP",
        "VERTICAL_DESCENT",
        "BILATERAL_CONTACT",
        "CLOSE_PRELOAD",
        "VERTICAL_LIFT",
        "HEIGHT_REACHED",
        "HOLD",
    )
    frames = []
    for frame, phase in enumerate(phases, start=1):
        views = {}
        for view in VIEWS:
            destination = tmp_path / view / f"{frame:06d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            color = (190, 40, 40) if view == "overview" else (40, 60, 190)
            Image.new("RGB", (960, 540), color).save(destination)
            views[view] = {
                "absolute_path": str(destination.resolve()),
                "sha256": _sha256(destination),
                "resolution": [960, 540],
                "physics_frame": frame,
                "time_s": frame / 60.0,
                "runtime_signature": "signature",
            }
        views["overview"]["framing_evidence"] = {
            "required_full_arm_links_in_frame": list(FULL_ARM_LINKS),
            "occlusion_status": "PENDING_VISUAL_MODEL_REVIEW",
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
        "runtime_signature": "signature",
        "required_full_arm_links": list(FULL_ARM_LINKS),
        "frames": frames,
    }


def _passing_report() -> dict[str, object]:
    return {
        "status": "PASS",
        "reason": "stable_20cm_hold",
        "deterministic_signature": "signature",
        "classification": "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING",
        "metrics": {
            "maximum_clearance_m": 0.2003,
            "hold_drop_m": 0.0045,
            "hold_duration_s": 2.0,
        },
    }


def _telemetry(manifest: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "physics_frame": record["physics_frame"],
            "time_s": record["time_s"],
            "phase": record["phase"],
            "clearance_m": 0.025 * index,
            "left_geometric_contact": index >= 4,
            "right_geometric_contact": index >= 4,
            "left_solver_active_contact": index >= 4,
            "right_solver_active_contact": index >= 4,
            "hold_drop_m": 0.0,
            "bottle_vertical_velocity_m_s": 0.0,
            "bottle_angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "ik": {"status": "PASS"},
        }
        for index, record in enumerate(manifest["frames"], start=1)
    ]


def test_manifest_requires_every_physics_frame_and_full_arm_scope(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    result = validate_frame_manifest(manifest)

    assert result["missing_physics_frames"] == []
    assert result["views"] == list(VIEWS)
    assert result["required_full_arm_links"] == list(FULL_ARM_LINKS)
    assert result["frame_count"] == 9

    manifest["frames"][4]["views"].pop("gripper_closeup")
    with pytest.raises(ValueError, match="gripper_closeup"):
        validate_frame_manifest(manifest)


def test_failed_run_video_accepts_only_a_contiguous_phase_prefix(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["frames"] = manifest["frames"][:6]
    failed_report = {
        **_passing_report(),
        "status": "FAIL",
        "reason": "close_preload_timeout",
    }

    required = required_phases_for_report(
        report=failed_report,
        manifest=manifest,
    )
    result = validate_frame_manifest(manifest, required_phases=required)

    assert required == list(REQUIRED_PHASES[:6])
    assert result["evidence_scope"] == "TERMINAL_FAILURE_PHASE_PREFIX"

    manifest["frames"] = [
        record
        for record in manifest["frames"]
        if record["phase"] != "BILATERAL_CONTACT"
    ]
    with pytest.raises(ValueError, match="non-contiguous phase prefix"):
        required_phases_for_report(report=failed_report, manifest=manifest)


def test_passing_run_video_still_requires_every_success_phase(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["frames"] = manifest["frames"][:6]

    with pytest.raises(ValueError, match="PASS report lacks required phases"):
        required_phases_for_report(
            report=_passing_report(),
            manifest=manifest,
        )


def test_failed_video_manifest_is_evidence_only_not_promotable(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    validation = validate_frame_manifest(manifest)
    failed_report = {
        **_passing_report(),
        "status": "FAIL",
        "reason": "close_preload_timeout",
    }
    fake_video = tmp_path / "failure.mp4"
    fake_video.write_bytes(b"failure-evidence")
    candidate = build_candidate_manifest(
        report=failed_report,
        frame_validation=validation,
        encoded_videos=[
            {
                "absolute_path": str(fake_video),
                "sha256": _sha256(fake_video),
                "frame_count": validation["frame_count"],
                "fps": 60,
            }
        ],
    )

    assert candidate["promotion_status"] == "MACHINE_FAIL_EVIDENCE_ONLY"


def test_overview_must_claim_every_required_full_arm_link(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["frames"][3]["views"]["overview"]["framing_evidence"][
        "required_full_arm_links_in_frame"
    ].remove("elbow")

    with pytest.raises(ValueError, match="elbow"):
        validate_frame_manifest(manifest)


def test_composite_uses_full_arm_and_synchronized_closeup(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    records = compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_signature="signature",
        output_dir=tmp_path / "composite",
    )
    first = Path(records[0]["absolute_path"])
    with Image.open(first) as image:
        assert image.size == (1440, 540)
        assert image.getpixel((100, 100)) == (190, 40, 40)
        assert image.getpixel((1200, 100)) == (0, 0, 0)
        assert image.getpixel((1200, 270)) == (40, 60, 190)


def test_annotation_uses_non_overlapping_panel_and_machine_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    composites = compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_signature="signature",
        output_dir=tmp_path / "composite",
    )
    records = annotate_composite_frames(
        composite_records=composites,
        telemetry=_telemetry(manifest),
        report=_passing_report(),
        output_dir=tmp_path / "annotated",
    )

    assert len(records) == 9
    with Image.open(records[-1]["absolute_path"]) as image:
        assert image.size == (1440, 680)
        # The source pixels are untouched; all annotation lives below them.
        assert image.getpixel((100, 100)) == (190, 40, 40)
        assert image.getpixel((1200, 270)) == (40, 60, 190)
        assert image.getpixel((20, 560)) != (0, 0, 0)
    assert records[-1]["machine_status"] == "PASS"
    assert records[-1]["phase"] == "HOLD"


def test_h264_pair_is_60fps_complete_and_yuv420p(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    composites = compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_signature="signature",
        output_dir=tmp_path / "composite",
    )
    annotated = annotate_composite_frames(
        composite_records=composites,
        telemetry=_telemetry(manifest),
        report=_passing_report(),
        output_dir=tmp_path / "annotated",
    )
    raw = encode_frame_sequence(
        frames_dir=tmp_path / "composite",
        first_frame=1,
        frame_count=9,
        destination=tmp_path / "raw.mp4",
        log_path=tmp_path / "raw.log",
    )
    marked = encode_frame_sequence(
        frames_dir=tmp_path / "annotated",
        first_frame=1,
        frame_count=9,
        destination=tmp_path / "annotated.mp4",
        log_path=tmp_path / "annotated.log",
    )

    validate_encoded_video_pair(
        expected_frame_count=len(composites),
        raw_probe=raw["probe"],
        annotated_probe=marked["probe"],
    )
    assert raw["probe"]["fps"] == 60
    assert raw["probe"]["pixel_format"] == "yuv420p"
    assert raw["probe"]["frame_count"] == len(composites)
    assert marked["probe"]["resolution"] == [1440, 680]
    assert len(annotated) == 9


def test_review_contact_sheets_cover_every_frame_exactly_once(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    composites = compose_synchronized_frames(
        source_records=manifest["frames"],
        runtime_signature="signature",
        output_dir=tmp_path / "composite",
    )

    sheets = build_review_contact_sheets(
        frame_records=composites,
        output_dir=tmp_path / "sheets",
    )

    assert [
        frame for sheet in sheets for frame in sheet["frame_numbers"]
    ] == list(range(1, 10))
    assert sheets[0]["resolution"] == [1440, 675]
    assert sheets[0]["visual_model_review"] == "NOT_RUN"


def test_candidate_waits_for_visual_model_and_user_review(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    videos = []
    for label in ("raw", "annotated"):
        path = tmp_path / f"{label}.mp4"
        path.write_bytes(label.encode())
        videos.append(
            {
                "kind": label,
                "absolute_path": str(path.resolve()),
                "sha256": _sha256(path),
                "frame_count": 9,
                "fps": 60,
            }
        )

    candidate = build_candidate_manifest(
        report=_passing_report(),
        frame_validation=validate_frame_manifest(manifest),
        encoded_videos=videos,
    )

    assert candidate["promotion_status"] == "AWAITING_VISUAL_MODEL_REVIEW"
    assert candidate["visual_model_review"] == "NOT_RUN"
    assert candidate["user_confirmation"] == "NOT_RUN"
    assert candidate["task8"] == "NOT_RUN"


def test_user_confirmation_is_bound_to_exact_annotated_hash() -> None:
    review = {
        "status": "PARTIAL",
        "machine_status": "PASS",
        "visual_model_review": "PASS",
        "user_confirmation": "NOT_RUN",
        "promotion_status": "AWAITING_USER_VIDEO_CONFIRMATION",
        "primary_action_video": {
            "annotated": {"sha256": "a" * 64},
        },
        "task8": "NOT_RUN",
    }

    confirmed = apply_user_confirmation(
        review,
        confirmed_annotated_sha256="a" * 64,
    )

    assert confirmed["status"] == "PASS"
    assert confirmed["user_confirmation"] == {
        "status": "PASS",
        "confirmed_annotated_sha256": "a" * 64,
        "source": "USER_CONFIRMED_VIDEO_IN_CONVERSATION",
    }
    assert confirmed["promotion_status"] == "USER_CONFIRMED_PASS"
    assert confirmed["task8"] == "NOT_RUN"


def test_user_confirmation_rejects_a_different_video_hash() -> None:
    review = {
        "status": "PARTIAL",
        "machine_status": "PASS",
        "visual_model_review": "PASS",
        "user_confirmation": "NOT_RUN",
        "promotion_status": "AWAITING_USER_VIDEO_CONFIRMATION",
        "primary_action_video": {
            "annotated": {"sha256": "a" * 64},
        },
        "task8": "NOT_RUN",
    }

    with pytest.raises(ValueError, match="annotated video SHA-256"):
        apply_user_confirmation(
            review,
            confirmed_annotated_sha256="b" * 64,
        )


def test_confirmed_markdown_names_exact_user_confirmed_hash() -> None:
    digest = "c" * 64
    run_report = {
        "status": "PASS",
        "user_confirmation": {
            "status": "PASS",
            "confirmed_annotated_sha256": digest,
        },
        "promotion_status": "USER_CONFIRMED_PASS",
        "deterministic_signature": "d" * 64,
        "metrics": {
            "maximum_clearance_m": 0.2,
            "hold_drop_m": 0.004,
        },
    }
    review_report = {
        "primary_action_video": {
            "raw": {"absolute_path": "/raw.mp4", "sha256": "a" * 64},
            "annotated": {
                "absolute_path": "/annotated.mp4",
                "sha256": digest,
            },
        },
        "collision_screenshot_evidence": {
            "run_root": "/collision",
        },
    }

    rendered = _markdown(run_report, review_report)

    assert "User video confirmation: `PASS`" in rendered
    assert digest in rendered
    assert "single-position baseline is `PASS`" in rendered
    assert "Five-position acceptance is not yet complete" in rendered


def test_clean_primary_video_marks_collider_capture_not_run(
    tmp_path: Path,
) -> None:
    result = annotate_collision_evidence(
        manifest={
            "collision_evidence": {
                "enabled": False,
                "purpose": "PRIMARY_CLEAN_VIDEO",
                "records": [],
            },
            "frames": [],
        },
        report={"status": "PASS", "reason": "stable_20cm_hold"},
        output_dir=tmp_path,
    )

    assert result == {
        "status": "NOT_RUN_PRIMARY_CLEAN_VIDEO",
        "enabled": False,
        "purpose": "PRIMARY_CLEAN_VIDEO",
        "records": [],
    }

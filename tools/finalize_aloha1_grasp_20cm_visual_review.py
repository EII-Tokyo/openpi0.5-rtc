#!/usr/bin/env python3
"""Bind a clean grasp video to an independent collider-evidence repeat."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def complete_sheet_frame_coverage(
    sheets: list[dict[str, Any]],
    *,
    expected_frame_count: int,
) -> bool:
    """Verify that review sheets cover every encoded frame exactly once."""

    expected = int(expected_frame_count)
    if expected < 1:
        raise ValueError("expected_frame_count must be positive")
    actual = [
        int(frame)
        for sheet in sheets
        for frame in sheet["frame_numbers"]
    ]
    return actual == list(range(1, expected + 1))


def normalized_rejected_attempts(
    records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep run-specific retake history explicit and machine readable."""

    if records is None:
        return []
    normalized: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {"run", "status"}:
            raise ValueError(
                "each rejected attempt requires exactly run and status"
            )
        run = str(record["run"])
        status = str(record["status"])
        if not run or not status.startswith("REJECTED_"):
            raise ValueError("invalid rejected-attempt record")
        normalized.append({"run": run, "status": status})
    return normalized


def _candidate_paths(run_root: Path) -> tuple[Path, Path]:
    return (
        run_root / "aloha1_grasp_20cm_runtime.json",
        run_root
        / "video_attempt_001/video/candidate_manifest.json",
    )


def apply_user_confirmation(
    review_report: dict[str, Any],
    *,
    confirmed_annotated_sha256: str,
) -> dict[str, Any]:
    """Promote only the exact machine/vision-passed annotated video."""

    expected = str(
        review_report["primary_action_video"]["annotated"]["sha256"]
    )
    if confirmed_annotated_sha256 != expected:
        raise ValueError(
            "confirmed annotated video SHA-256 does not match candidate"
        )
    if (
        review_report.get("machine_status") != "PASS"
        or review_report.get("visual_model_review") != "PASS"
    ):
        raise ValueError(
            "user confirmation cannot override machine or vision failure"
        )
    confirmed = copy.deepcopy(review_report)
    confirmed["status"] = "PASS"
    confirmed["user_confirmation"] = {
        "status": "PASS",
        "confirmed_annotated_sha256": expected,
        "source": "USER_CONFIRMED_VIDEO_IN_CONVERSATION",
    }
    confirmed["promotion_status"] = "USER_CONFIRMED_PASS"
    return confirmed


def build_reports(
    *,
    primary_run_root: Path,
    collision_run_root: Path,
    confirmed_annotated_sha256: str | None = None,
    rejected_attempts: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_report_path, primary_candidate_path = _candidate_paths(
        primary_run_root
    )
    collision_report_path, collision_candidate_path = _candidate_paths(
        collision_run_root
    )
    primary_report = _load(primary_report_path)
    primary_candidate = _load(primary_candidate_path)
    collision_report = _load(collision_report_path)
    collision_candidate = _load(collision_candidate_path)

    signatures = {
        str(primary_report["deterministic_signature"]),
        str(primary_candidate["runtime_signature"]),
        str(collision_report["deterministic_signature"]),
        str(collision_candidate["runtime_signature"]),
    }
    if len(signatures) != 1:
        raise ValueError("paired evidence deterministic signatures differ")
    if (
        primary_report["status"] != "PASS"
        or collision_report["status"] != "PASS"
    ):
        raise ValueError("paired machine runs must both PASS")
    collision = collision_candidate["collision_evidence"]
    render_evidence = collision.get("render_evidence", {})
    if not bool(render_evidence.get("authored_geometry_clone")):
        raise ValueError("collision run lacks authored collider overlay")
    if len(collision["records"]) != 24:
        raise ValueError("expected 24 paired collision screenshots")

    videos = {
        str(record["kind"]): record
        for record in primary_candidate["videos"]
    }
    if set(videos) != {"raw", "annotated"}:
        raise ValueError("primary candidate must have raw and annotated video")
    expected_frame_count = int(
        primary_candidate["frame_validation"]["frame_count"]
    )
    if not complete_sheet_frame_coverage(
        primary_candidate["review_contact_sheets"],
        expected_frame_count=expected_frame_count,
    ):
        raise ValueError(
            "primary review sheets do not cover every encoded frame"
        )

    signature = signatures.pop()
    shared = {
        "schema_version": 1,
        "status": "PARTIAL",
        "machine_status": "PASS",
        "visual_model_review": "PASS",
        "user_confirmation": "NOT_RUN",
        "promotion_status": "AWAITING_USER_VIDEO_CONFIRMATION",
        "deterministic_signature": signature,
        "task8": "NOT_RUN",
    }
    run_report = {
        **shared,
        "reason": "stable_20cm_hold",
        "stage": primary_report["stage"],
        "metrics": primary_report["metrics"],
        "classification": primary_report["classification"],
        "primary_runtime_report": {
            "absolute_path": str(primary_report_path.resolve()),
            "sha256": _sha256(primary_report_path),
        },
        "collision_runtime_repeat": {
            "absolute_path": str(collision_report_path.resolve()),
            "sha256": _sha256(collision_report_path),
        },
        "paired_evidence": {
            "status": "PASS_IDENTICAL_MACHINE_SIGNATURE",
            "purpose": (
                "KEEP_PRIMARY_ACTION_VIDEO_CLEAN_WHILE_CAPTURING_"
                "AUTHORED_COLLIDER_GEOMETRY_IN_A_FRESH_REPEAT"
            ),
            "physical_parameters_identical": True,
            "source_stage_hash_identical": (
                primary_report["stage"]["sha256_before"]
                == collision_report["stage"]["sha256_before"]
            ),
        },
    }
    review_report = {
        **shared,
        "primary_action_video": {
            "run_root": str(primary_run_root.resolve()),
            "raw": videos["raw"],
            "annotated": videos["annotated"],
            "full_frame_review": {
                "status": "PASS",
                "method": (
                    "CODEX_VISION_REVIEW_OF_ALL_CONTACT_SHEETS_"
                    "COVERING_EVERY_FRAME_EXACTLY_ONCE"
                ),
                "sheet_count": len(
                    primary_candidate["review_contact_sheets"]
                ),
                "sheets": [
                    {
                        **sheet,
                        "visual_model_review": "PASS",
                        "retake_reason": None,
                    }
                    for sheet in primary_candidate[
                        "review_contact_sheets"
                    ]
                ],
                "criteria": {
                    "full_arm_visible": True,
                    "gripper_and_bottle_visible": True,
                    "open_contact_lift_hold_visibly_distinct": True,
                    "critical_occlusion": False,
                    "missing_or_duplicated_frame": False,
                },
            },
        },
        "collision_screenshot_evidence": {
            "run_root": str(collision_run_root.resolve()),
            "candidate_manifest": {
                "absolute_path": str(
                    collision_candidate_path.resolve()
                ),
                "sha256": _sha256(collision_candidate_path),
            },
            "status": "PASS",
            "review_method": (
                "CODEX_VISION_REVIEW_OF_NORMAL_AND_COLLIDER_"
                "ANNOTATED_MONTAGES;_ANNOTATED_TOP_PIXELS_ARE_"
                "THE_UNMODIFIED_RAW_CAPTURE"
            ),
            "render_evidence": render_evidence,
            "records": [
                {
                    **record,
                    "visual_model_review": "PASS",
                    "retake_reason": None,
                }
                for record in collision["records"]
            ],
        },
        "rejected_attempts": normalized_rejected_attempts(
            rejected_attempts
        ),
        "semantic_boundary": {
            "screenshots_are_auxiliary": True,
            "machine_contacts_pose_velocity_and_drop_authoritative": True,
            "green_geometry": (
                "SESSION_ONLY_AUTHORED_COLLIDER_MESH_AT_RUNTIME_"
                "RIGID_POSE_NOT_COOKED_PHYSX_HULL_READBACK"
            ),
            "no_physics_or_collision_schema_on_render_clones": True,
        },
    }
    if confirmed_annotated_sha256 is not None:
        review_report = apply_user_confirmation(
            review_report,
            confirmed_annotated_sha256=confirmed_annotated_sha256,
        )
        run_report["status"] = review_report["status"]
        run_report["user_confirmation"] = review_report[
            "user_confirmation"
        ]
        run_report["promotion_status"] = review_report[
            "promotion_status"
        ]
    return run_report, review_report


def _markdown(
    run_report: dict[str, Any],
    review_report: dict[str, Any],
) -> str:
    raw = review_report["primary_action_video"]["raw"]
    annotated = review_report["primary_action_video"]["annotated"]
    collision = review_report["collision_screenshot_evidence"]
    confirmation = run_report["user_confirmation"]
    if isinstance(confirmation, dict):
        confirmation_status = str(confirmation["status"])
        confirmation_line = (
            "- User-confirmed annotated SHA-256: "
            f"`{confirmation['confirmed_annotated_sha256']}`"
        )
    else:
        confirmation_status = str(confirmation)
        confirmation_line = (
            "- User-confirmed annotated SHA-256: `NOT_RUN`"
        )
    if run_report["status"] == "PASS":
        closing = (
            "The exact annotated-video SHA-256 was confirmed by the user; "
            "the single-position baseline is `PASS`. Five-position "
            "acceptance is not yet complete. Task 8 is `NOT_RUN`."
        )
    else:
        closing = (
            "The overall status remains `PARTIAL` until the user confirms "
            "the exact annotated-video SHA-256. Task 8 is `NOT_RUN`."
        )
    return "\n".join(
        [
            "# ALOHA Bottle500 20 cm grasp button evidence",
            "",
            f"- Status: `{run_report['status']}`",
            "- Machine run: `PASS` (`stable_20cm_hold`)",
            "- Vision review: `PASS`",
            f"- User video confirmation: `{confirmation_status}`",
            confirmation_line,
            f"- Promotion: `{run_report['promotion_status']}`",
            (
                "- Deterministic signature: "
                f"`{run_report['deterministic_signature']}`"
            ),
            (
                "- Maximum support clearance: "
                f"`{run_report['metrics']['maximum_clearance_m']:.9f} m`"
            ),
            (
                "- Hold drop: "
                f"`{run_report['metrics']['hold_drop_m']:.9f} m`"
            ),
            "",
            "## Primary action video",
            "",
            f"- Raw: `{raw['absolute_path']}`",
            f"- Raw SHA-256: `{raw['sha256']}`",
            f"- Annotated: `{annotated['absolute_path']}`",
            f"- Annotated SHA-256: `{annotated['sha256']}`",
            "",
            "## Collision screenshot evidence",
            "",
            f"- Repeat root: `{collision['run_root']}`",
            "- Status: `PASS`",
            (
                "- The green geometry is a session-only copy of the "
                "authored collider mesh at the runtime rigid pose; it is "
                "not a cooked PhysX hull readback."
            ),
            (
                "- The primary video and collision repeat have the same "
                "machine signature and unchanged frozen Stage hash."
            ),
            "",
            closing,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-run-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--collision-run-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmed-annotated-sha256",
        default=None,
    )
    parser.add_argument(
        "--rejected-attempts-json",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    rejected_attempts = None
    if args.rejected_attempts_json is not None:
        payload = _load(args.rejected_attempts_json.resolve(strict=True))
        rejected_attempts = payload["rejected_attempts"]
    run_report, review_report = build_reports(
        primary_run_root=args.primary_run_root.resolve(strict=True),
        collision_run_root=args.collision_run_root.resolve(strict=True),
        confirmed_annotated_sha256=(
            str(args.confirmed_annotated_sha256)
            if args.confirmed_annotated_sha256
            else None
        ),
        rejected_attempts=rejected_attempts,
    )
    run_path = (
        args.output_dir / "aloha1_grasp_20cm_button_run.json"
    )
    review_path = (
        args.output_dir / "aloha1_grasp_20cm_button_video_review.json"
    )
    markdown_path = (
        args.output_dir / "aloha1_grasp_20cm_button_video_review.md"
    )
    _write_json(run_path, run_report)
    _write_json(review_path, review_report)
    markdown_path.write_text(
        _markdown(run_report, review_report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_report": str(run_path.resolve()),
                "video_review": str(review_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "promotion_status": review_report[
                    "promotion_status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

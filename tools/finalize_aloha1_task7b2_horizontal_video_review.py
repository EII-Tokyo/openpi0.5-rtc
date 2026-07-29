#!/usr/bin/env python3
"""Finalize visual-model review of horizontal Bottle500 videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

SOURCE_VIEWS = ("overview", "gripper_closeup")
COMPOSITE_VIEW = "full_arm_composite"
VIEWS = (*SOURCE_VIEWS, COMPOSITE_VIEW)
LAYOUT = "FULL_ARM_WITH_SYNCHRONIZED_GRIPPER_INSET"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_name(attempt_id: str, view: str, kind: str) -> str:
    return f"{attempt_id}_{view}_{kind}_visual_evidence.mp4"


def finalize_video_review(
    *,
    candidate_manifest_path: Path,
    decisions_path: Path,
    verified_root: Path,
    promote: bool,
    user_confirmed_video_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate visual review and require hash-bound user confirmation to close."""
    candidate = _load(candidate_manifest_path)
    decisions = _load(decisions_path)
    if candidate["attempt_id"] != decisions["attempt_id"]:
        raise ValueError("attempt identifiers differ")
    records = {item["view_name"]: item for item in candidate["videos"]}
    if set(records) != set(VIEWS):
        raise ValueError("candidate must contain synchronized sources and composite")
    primary = records[COMPOSITE_VIEW]
    if primary.get("evidence_role") != "PRIMARY_FULL_ARM_EVIDENCE":
        raise ValueError("composite is not marked as primary full-arm evidence")
    if primary.get("layout") != LAYOUT:
        raise ValueError("composite layout mismatch")
    if list(primary.get("source_views", [])) != list(SOURCE_VIEWS):
        raise ValueError("composite synchronized source views mismatch")
    decision_views = decisions.get("views", {})
    if set(decision_views) != {COMPOSITE_VIEW}:
        raise ValueError("review must contain exactly the primary composite view")
    signatures = {str(records[view]["runtime_trial_signature"]) for view in VIEWS}
    if len(signatures) != 1:
        raise ValueError("view runtime signatures differ")

    videos = []
    for view in VIEWS:
        source = dict(records[view])
        if view == COMPOSITE_VIEW:
            decision = decision_views[view]
            status = str(decision["status"])
            if status not in {"PASS", "FAIL"}:
                raise ValueError(f"invalid review status for {view}: {status}")
            reviewed_frames = [int(value) for value in decision["reviewed_sample_frames"]]
            if not reviewed_frames:
                raise ValueError(f"{view} has no reviewed samples")
            source["vision_review_status"] = status
            source["reviewed_sample_frames"] = reviewed_frames
            source["retake_reason"] = decision.get("retake_reason")
            source["visual_review"] = {
                key: value
                for key, value in decision.items()
                if key not in {"status", "reviewed_sample_frames", "retake_reason"}
            }
        else:
            source["vision_review_status"] = "SYNCHRONIZED_SOURCE_NOT_PRIMARY"
            source["reviewed_sample_frames"] = []
            source["retake_reason"] = None
        videos.append(source)

    visual_pass = str(decision_views[COMPOSITE_VIEW]["status"]) == "PASS"
    confirmation_target = {
        "view_name": COMPOSITE_VIEW,
        "kind": "annotated",
        "sha256": str(primary["annotated_candidate_sha256"]),
    }
    if not visual_pass:
        user_video_confirmation = "BLOCKED_BY_VISUAL_REVIEW"
    elif user_confirmed_video_sha256 is None:
        user_video_confirmation = "PENDING"
    elif user_confirmed_video_sha256 != confirmation_target["sha256"]:
        raise ValueError("confirmed video hash does not match composite candidate")
    else:
        user_video_confirmation = "CONFIRMED"

    if not visual_pass:
        promotion_status = "REJECTED_VISUAL_REVIEW"
        report_status = "FAIL"
    elif user_video_confirmation == "PENDING":
        promotion_status = "AWAITING_USER_VIDEO_CONFIRMATION"
        report_status = "PARTIAL"
    elif candidate["physical_trial_status"] == "PASS":
        promotion_status = "PROMOTED_VISUAL_EVIDENCE_PHYSICAL_PASS"
        report_status = "PASS"
    else:
        promotion_status = "PROMOTED_VISUAL_EVIDENCE_PHYSICAL_FAIL"
        report_status = "FAIL"

    if promote and visual_pass and user_video_confirmation == "CONFIRMED":
        verified_root.mkdir(parents=True, exist_ok=True)
        for record in videos:
            view = record["view_name"]
            for kind in ("raw", "annotated"):
                source_path = Path(record[f"{kind}_candidate_absolute_path"]).resolve(strict=True)
                if _sha256(source_path) != record[f"{kind}_candidate_sha256"]:
                    raise ValueError(f"{view}/{kind} candidate hash mismatch")
                destination = (verified_root / _verified_name(candidate["attempt_id"], view, kind)).resolve()
                shutil.copy2(source_path, destination)
                record[f"verified_{kind}_absolute_path"] = str(destination)
                record[f"verified_{kind}_sha256"] = _sha256(destination)
                record["promotion_status"] = promotion_status
    else:
        for record in videos:
            record["promotion_status"] = promotion_status

    return {
        "schema_version": 1,
        "status": report_status,
        "visual_review_status": "PASS" if visual_pass else "FAIL",
        "attempt_id": candidate["attempt_id"],
        "reviewed_by": decisions["reviewed_by"],
        "review_method": (
            "Codex vision model inspected phase boundaries and uniform "
            "samples at intervals no greater than 0.5 seconds."
        ),
        "physical_trial_status": candidate["physical_trial_status"],
        "machine_conclusion": candidate["machine_conclusion"],
        "promotion_status": promotion_status,
        "user_video_confirmation": user_video_confirmation,
        "user_confirmation_target": confirmation_target,
        "user_confirmed_video_sha256": (
            user_confirmed_video_sha256 if user_video_confirmation == "CONFIRMED" else None
        ),
        "scope": ("VISUAL_CAPTURE_VALIDATION_ONLY; machine physics result unchanged"),
        "attempt_history": decisions.get("attempt_history", []),
        "reviewed_contact_sheets": decisions.get("reviewed_contact_sheets", []),
        "candidate_manifest_absolute_path": str(candidate_manifest_path.resolve()),
        "candidate_manifest_sha256": _sha256(candidate_manifest_path),
        "decisions_absolute_path": str(decisions_path.resolve()),
        "decisions_sha256": _sha256(decisions_path),
        "videos": videos,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Horizontal Bottle Grasp Video Review",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Visual review: `{report['visual_review_status']}`",
        f"- Physical trial: `{report['physical_trial_status']}`",
        f"- Machine conclusion: `{report['machine_conclusion']}`",
        f"- Promotion: `{report['promotion_status']}`",
        f"- User video confirmation: `{report['user_video_confirmation']}`",
        "",
        (
            "Visual PASS certifies only that the primary composite exposes "
            "the complete recorded trial. The report remains PARTIAL until "
            "the user confirms the exact annotated composite video hash."
        ),
        "",
        "| View | visual review | raw | annotated |",
        "|---|---|---|---|",
    ]
    for item in report["videos"]:
        raw = item.get("verified_raw_absolute_path") or item["raw_candidate_absolute_path"]
        annotated = item.get("verified_annotated_absolute_path") or item["annotated_candidate_absolute_path"]
        lines.append(f"| {item['view_name']} | {item['vision_review_status']} | `{raw}` | `{annotated}` |")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--verified-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--user-confirmed-video-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = finalize_video_review(
        candidate_manifest_path=args.candidate_manifest.resolve(strict=True),
        decisions_path=args.decisions.resolve(strict=True),
        verified_root=args.verified_root.resolve(),
        promote=args.promote,
        user_confirmed_video_sha256=args.user_confirmed_video_sha256,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "physical_trial_status": report["physical_trial_status"],
                "promotion_status": report["promotion_status"],
                "report": str(args.report_json.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

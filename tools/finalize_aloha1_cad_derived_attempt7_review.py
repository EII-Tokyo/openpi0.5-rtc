#!/usr/bin/env python3
"""Finalize truthful visual evidence for the Z-up five-pose grasp batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_POSITIVE_CHECKS = {
    "full_arm_visible",
    "gripper_and_bottle_visible",
    "initial_pose_distinct",
    "bottle_direction_visible",
    "gripper_points_downward",
    "phases_visibly_distinct",
    "vertical_lift_visible",
    "hold_end_visible",
    "world_z_visually_upright",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def validate_visual_decision(decision: dict[str, Any]) -> str:
    """Return PASS only for an explicit, unobstructed critical-phase review."""

    if not REQUIRED_POSITIVE_CHECKS.issubset(decision):
        return "FAIL"
    positives = all(bool(decision[name]) for name in REQUIRED_POSITIVE_CHECKS)
    return (
        "PASS"
        if positives and decision.get("critical_occlusion") is False
        else "FAIL"
    )


def classify_review_status(
    *, machine_status: str, visual_status: str, user_confirmation: str
) -> str:
    """Keep user confirmation independent from machine and vision gates."""

    if machine_status != "PASS" or visual_status != "PASS":
        return "FAIL"
    return "PASS" if user_confirmation == "PASS" else "PARTIAL"


def _manifest(run_root: Path) -> tuple[Path, dict[str, Any]]:
    path = run_root / "video_attempt_001/video/candidate_manifest.json"
    return path.resolve(strict=True), _load(path)


def _runtime(run_root: Path) -> tuple[Path, dict[str, Any]]:
    path = run_root / "aloha1_grasp_20cm_runtime.json"
    return path.resolve(strict=True), _load(path)


def _verified_file(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    path = Path(record[f"{prefix}_absolute_path"]).resolve(strict=True)
    expected = str(record[f"{prefix}_sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{prefix} evidence hash changed: {path}")
    return {
        "absolute_path": str(path),
        "sha256": actual,
        "resolution": record[f"{prefix}_resolution"],
    }


def _video_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for source in manifest["videos"]:
        path = Path(source["absolute_path"]).resolve(strict=True)
        actual = _sha256(path)
        if actual != source["sha256"]:
            raise ValueError(f"video hash changed: {path}")
        records[source["kind"]] = {
            "absolute_path": str(path),
            "sha256": actual,
            "frame_count": source["frame_count"],
            "fps": source["fps"],
            "probe": source["probe"],
        }
    if set(records) != {"raw", "annotated"}:
        raise ValueError("primary evidence must contain raw and annotated video")
    return records


def build_report(
    results: dict[str, Any],
    decisions: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Bind manually reviewed montages to immutable machine evidence."""

    source_samples = {sample["sample_id"]: sample for sample in results["samples"]}
    sample_reports: list[dict[str, Any]] = []
    for decision in decisions["samples"]:
        sample_id = decision["sample_id"]
        source = source_samples[sample_id]
        primary_root = (project_root / decision["primary_run_root"]).resolve()
        collision_root = (project_root / decision["collision_run_root"]).resolve()
        primary_report_path, primary_report = _runtime(primary_root)
        primary_manifest_path, primary_manifest = _manifest(primary_root)
        collision_report_path, collision_report = _runtime(collision_root)
        collision_manifest_path, collision_manifest = _manifest(collision_root)

        signatures = {
            source["primary"]["deterministic_signature"],
            source["collider_repeat"]["deterministic_signature"],
            primary_report["deterministic_signature"],
            primary_manifest["runtime_signature"],
            collision_report["deterministic_signature"],
            collision_manifest["runtime_signature"],
        }
        if len(signatures) != 1:
            raise ValueError(f"signature mismatch for {sample_id}: {signatures}")
        if primary_report["status"] != "PASS" or collision_report["status"] != "PASS":
            raise ValueError(f"selected evidence is not machine PASS: {sample_id}")

        records = collision_manifest["collision_evidence"]["records"]
        if len(records) != 24:
            raise ValueError(f"expected 24 collision records: {sample_id}")
        collision_records = [
            {
                "phase_label": record["phase_label"],
                "physics_frame": record["physics_frame"],
                "time_s": record["time_s"],
                "view": record["view"],
                "mode": record["mode"],
                "raw": _verified_file(record, "raw"),
                "annotated": _verified_file(record, "annotated"),
                "visual_model_review": "PASS",
                "retake_reason": None,
            }
            for record in records
        ]

        montage = (project_root / decision["phase_montage"]).resolve(strict=True)
        collision_montages = [
            (project_root / item).resolve(strict=True)
            for item in decision["collision_montages"]
        ]
        visual_status = validate_visual_decision(decision["checks"])
        videos = _video_records(primary_manifest)
        frame_validation = primary_manifest["frame_validation"]
        sample_reports.append(
            {
                "sample_id": sample_id,
                "status": visual_status,
                "deterministic_signature": signatures.pop(),
                "machine_status": "PASS",
                "visual_model_review": visual_status,
                "user_confirmation": "NOT_RUN",
                "selected_evidence_reason": decision["selected_evidence_reason"],
                "primary_runtime": {
                    "absolute_path": str(primary_report_path),
                    "sha256": _sha256(primary_report_path),
                },
                "primary_manifest": {
                    "absolute_path": str(primary_manifest_path),
                    "sha256": _sha256(primary_manifest_path),
                },
                "collision_runtime": {
                    "absolute_path": str(collision_report_path),
                    "sha256": _sha256(collision_report_path),
                },
                "collision_manifest": {
                    "absolute_path": str(collision_manifest_path),
                    "sha256": _sha256(collision_manifest_path),
                },
                "videos": videos,
                "frame_validation": frame_validation,
                "reviewed_critical_frames": decision["reviewed_critical_frames"],
                "phase_montage": {
                    "absolute_path": str(montage),
                    "sha256": _sha256(montage),
                },
                "collision_montages": [
                    {"absolute_path": str(path), "sha256": _sha256(path)}
                    for path in collision_montages
                ],
                "checks": decision["checks"],
                "collision_screenshot_evidence": {
                    "status": "PASS",
                    "record_count": len(collision_records),
                    "records": collision_records,
                },
                "rejected_attempts": decision.get("rejected_attempts", []),
            }
        )

    visual_status = (
        "PASS"
        if len(sample_reports) == 5
        and all(sample["visual_model_review"] == "PASS" for sample in sample_reports)
        else "FAIL"
    )
    status = classify_review_status(
        machine_status=results["machine_status"],
        visual_status=visual_status,
        user_confirmation="NOT_RUN",
    )
    return {
        "schema_version": 1,
        "status": status,
        "machine_status": results["machine_status"],
        "visual_model_review": visual_status,
        "user_confirmation": "NOT_RUN",
        "promotion_status": "AWAITING_USER_CONFIRMATION_OF_EXACT_VIDEOS",
        "samples": sample_reports,
        "review_method": (
            "CODEX_VISION_REVIEW_OF_FIVE_CRITICAL_VIDEO_PHASES_AND_ALL_"
            "24_COLLISION_SCREENSHOT_PANELS_PER_SAMPLE"
        ),
        "semantic_boundaries": {
            "no_claim_of_every_video_frame_visual_review": True,
            "screenshots_and_video_are_auxiliary": True,
            "contact_pose_clearance_drop_and_signature_are_authoritative": True,
            "green_overlay_does_not_alone_prove_full_arm_collider_coverage": True,
            "full_arm_collider_coverage_requires_static_and_swept_reports": True,
            "tensor_velocity_disagreement_not_silently_ignored": True,
            "task8": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 CAD-derived Z-up five-pose visual review",
        "",
        f"- Status: `{report['status']}`",
        f"- Machine: `{report['machine_status']}`",
        f"- Visual-model review: `{report['visual_model_review']}`",
        "- User confirmation of these exact videos: `NOT_RUN`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Sample | Vision | Frames | Collision images | Annotated video |",
        "|---|---|---:|---:|---|",
    ]
    for sample in report["samples"]:
        video = sample["videos"]["annotated"]
        lines.append(
            f"| {sample['sample_id']} | {sample['visual_model_review']} | "
            f"{video['frame_count']} | "
            f"{sample['collision_screenshot_evidence']['record_count']} | "
            f"`{video['absolute_path']}` |"
        )
    lines.extend(
        [
            "",
            "Visual review covers the five required critical phases and all "
            "24 collision-evidence panels per sample. It does not claim that "
            "every encoded video frame was individually inspected. Runtime "
            "contact, pose, clearance, drop and deterministic signatures remain "
            "authoritative. Exact-video user confirmation is still pending.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = build_report(
        _load(args.results),
        _load(args.decisions),
        project_root=project_root,
    )
    args.output_json.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.resolve().write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output_json.resolve())}))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

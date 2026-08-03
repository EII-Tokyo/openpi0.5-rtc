#!/usr/bin/env python3
"""Finalize visual review for ALOHA Home/Sleep digital failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
DEFAULT_ORIGINAL = REPORT_ROOT / "aloha1_home_sleep_digital_video_review.json"
DEFAULT_RETAKE = REPORT_ROOT / "aloha1_home_sleep_collision_video_retake.json"
DEFAULT_OUTPUT = REPORT_ROOT / "aloha1_home_sleep_digital_evidence_review.json"
DEFAULT_MARKDOWN = REPORT_ROOT / "aloha1_home_sleep_digital_evidence_review.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path_value: str, expected_hash: str) -> None:
    path = Path(path_value).resolve(strict=True)
    actual = _sha256(path)
    if actual != expected_hash:
        raise ValueError(f"evidence SHA-256 mismatch: {path}: {actual} != {expected_hash}")


def _retained_screenshots(
    report: dict[str, Any], mode: str, *, review_note: str
) -> list[dict[str, Any]]:
    retained = []
    for source in report["screenshots"]:
        if source["mode"] != mode:
            continue
        item = dict(source)
        item["visual_review"] = {
            "status": "PASS",
            "reviewed_by": "Codex visual model",
            "whole_follower_left_visible": True,
            "home_sleep_states_visibly_distinct": True,
            "annotation_readable_and_non_occluding": True,
            "collision_overlay_distinct": mode == "collision_overlay",
            "note": review_note,
        }
        retained.append(item)
    return retained


def build_visual_review(
    original: dict[str, Any],
    collision_retake: dict[str, Any],
    *,
    normal_review_status: str,
    collision_retake_review_status: str,
    rejected_collision_reason: str,
) -> dict[str, Any]:
    """Select the accepted normal evidence and the high-contrast collision retake."""

    if original["stage"]["sha256_before"] != collision_retake["stage"]["sha256_before"]:
        raise ValueError("normal/collision Stage hash mismatch")
    if original["manifest"]["sha256"] != collision_retake["manifest"]["sha256"]:
        raise ValueError("normal/collision manifest hash mismatch")
    if (
        original["manifest"]["command_signature"]
        != collision_retake["manifest"]["command_signature"]
    ):
        raise ValueError("normal/collision command signature mismatch")
    normal_pass = normal_review_status == "PASS"
    collision_pass = collision_retake_review_status == "PASS"
    normal_screenshots = _retained_screenshots(
        original,
        "normal",
        review_note=(
            "Whole arm visible; Home and Sleep poses differ; labels do not obscure geometry."
        ),
    )
    collision_screenshots = _retained_screenshots(
        collision_retake,
        "collision_overlay",
        review_note=(
            "Bright-red full-body collider overlay is visually distinct from source materials."
        ),
    )
    required_labels = {
        "before_limit_exceedance",
        "first_limit_exceedance",
        "first_sleep_hold_end",
        "final_home_recovery",
    }
    labels_by_mode = {
        "normal": {item["label"] for item in normal_screenshots},
        "collision_overlay": {item["label"] for item in collision_screenshots},
    }
    complete = all(labels == required_labels for labels in labels_by_mode.values())
    status = (
        "PASS_FAILURE_EVIDENCE"
        if normal_pass and collision_pass and complete
        else "FAIL_VISUAL_EVIDENCE"
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": "DIGITAL_SLEEP_LIMIT_FAILURE_VISUAL_REVIEW",
        "visual_review_is_auxiliary": True,
        "machine_telemetry_remains_primary": True,
        "stage_sha256": original["stage"]["sha256_before"],
        "manifest_sha256": original["manifest"]["sha256"],
        "command_signature": original["manifest"]["command_signature"],
        "retained_videos": {
            "normal": dict(original["videos"]["normal"]),
            "collision_overlay": dict(
                collision_retake["videos"]["collision_overlay"]
            ),
        },
        "retained_screenshots": normal_screenshots + collision_screenshots,
        "gates": {
            "normal_visual_review": normal_pass,
            "collision_retake_visual_review": collision_pass,
            "all_required_stages_present_per_mode": complete,
            "whole_arm_visible": normal_pass and collision_pass,
            "states_visibly_distinct": normal_pass and collision_pass,
            "annotations_non_occluding": normal_pass and collision_pass,
        },
        "rejected_attempts": [
            {
                "source_report": original.get("report_path", "original capture report"),
                "mode": "collision_overlay",
                "reason": rejected_collision_reason,
                "retained_as_history": True,
            }
        ],
        "review_notes": {
            "normal": "PASS: fixed overview contains the complete follower_left arm.",
            "collision_overlay": (
                "PASS after retake: red colliders are distinguishable from cyan/white visuals."
            ),
            "boundary": (
                "PASS applies to evidence quality only; the digital motion gate remains FAIL."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Home/Sleep digital evidence review",
        "",
        f"- Status: `{report['status']}`",
        f"- Stage SHA-256: `{report['stage_sha256']}`",
        f"- Command signature: `{report['command_signature']}`",
        "- Evidence role: auxiliary visual evidence; numeric telemetry remains authoritative.",
        "- Digital motion gate: `FAIL` (this visual PASS does not authorize real motion).",
        "",
        "## Retake history",
        "",
        "- Attempt 1 normal: `PASS`.",
        "- Attempt 1 collision overlay: `REJECTED_COLLIDER_OVERLAY_NOT_DISTINCT`.",
        "- Attempt 2 collision overlay: `PASS`; bright-red overlay is distinguishable.",
        "",
        "## Retained videos",
        "",
    ]
    for mode, item in report["retained_videos"].items():
        lines.append(
            f"- `{mode}`: `{item['absolute_path']}` ({item['frame_count']} frames, "
            f"SHA-256 `{item['sha256']}`)"
        )
    lines.extend(["", "## Retained screenshots", ""])
    lines.extend(
        (
            f"- `{item['mode']}/{item['label']}`: `{item['annotated_absolute_path']}`"
        )
        for item in report["retained_screenshots"]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--collision-retake", type=Path, default=DEFAULT_RETAKE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    original = json.loads(args.original.read_text(encoding="utf-8"))
    retake = json.loads(args.collision_retake.read_text(encoding="utf-8"))
    for report, modes in ((original, ("normal",)), (retake, ("collision_overlay",))):
        for mode in modes:
            video = report["videos"][mode]
            _verify_file(video["absolute_path"], video["sha256"])
        for screenshot in report["screenshots"]:
            if screenshot["mode"] not in modes:
                continue
            _verify_file(screenshot["raw_absolute_path"], screenshot["raw_sha256"])
            _verify_file(
                screenshot["annotated_absolute_path"], screenshot["annotated_sha256"]
            )
    original["report_path"] = str(args.original.resolve())
    report = build_visual_review(
        original,
        retake,
        normal_review_status="PASS",
        collision_retake_review_status="PASS",
        rejected_collision_reason="REJECTED_COLLIDER_OVERLAY_NOT_DISTINCT",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] == "PASS_FAILURE_EVIDENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

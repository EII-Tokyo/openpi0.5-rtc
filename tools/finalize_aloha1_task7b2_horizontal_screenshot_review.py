#!/usr/bin/env python3
"""Finalize vision-model review of horizontal-grasp screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(
    *,
    candidate_path: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decision_map = {(item["view_name"], item["phase"]): item for item in decisions["captures"]}
    captures = []
    statuses = []
    for capture in candidate["captures"]:
        key = (capture["view_name"], capture["phase"])
        if key not in decision_map:
            raise ValueError(f"missing visual decision: {key}")
        decision = decision_map[key]
        merged = dict(capture)
        merged["vision_review_status"] = decision["status"]
        merged["retake_reason"] = decision.get("retake_reason")
        merged["vision_review"] = {
            key: value
            for key, value in decision.items()
            if key not in {"view_name", "phase", "status", "retake_reason"}
        }
        statuses.append(decision["status"])
        captures.append(merged)
    status = (
        "PASS"
        if all(value == "PASS" for value in statuses)
        else "FAIL"
        if any(value == "FAIL" for value in statuses)
        else "PARTIAL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "physical_trial_status": candidate["physical_trial_status"],
        "machine_conclusion": candidate["machine_conclusion"],
        "runtime_trial_signature": candidate["runtime_trial_signature"],
        "reviewed_by": decisions["reviewed_by"],
        "review_method": decisions["review_method"],
        "attempt_history": decisions["attempt_history"],
        "scope": ("SCREENSHOT_VISUAL_EVIDENCE_ONLY; runtime physics result unchanged"),
        "candidate_manifest_absolute_path": str(candidate_path.resolve()),
        "candidate_manifest_sha256": _sha256(candidate_path),
        "decisions_absolute_path": str(decisions_path.resolve()),
        "decisions_sha256": _sha256(decisions_path),
        "capture_count": len(captures),
        "captures": captures,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Horizontal Bottle Grasp Screenshot Review",
        "",
        f"- Screenshot gate: `{report['status']}`",
        f"- Physical trial: `{report['physical_trial_status']}`",
        f"- Machine conclusion: `{report['machine_conclusion']}`",
        "",
        (
            "The true-top images are PARTIAL because the actual wrist/gripper "
            "pose occludes the finger inner surfaces. Runtime-projected L/R "
            "collider origins are auxiliary markers, not a substitute for "
            "visible contact geometry. The side-oblique images pass."
        ),
        "",
        "| phase | view | visual | raw | annotated |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        (
            f"| {item['phase']} | {item['view_name']} | "
            f"{item['vision_review_status']} | "
            f"`{item['raw_absolute_path']}` | "
            f"`{item['annotated_absolute_path']}` |"
        )
        for item in report["captures"]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = finalize(
        candidate_path=args.candidate.resolve(strict=True),
        decisions_path=args.decisions.resolve(strict=True),
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_md.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "physical_trial_status": report["physical_trial_status"],
                "report": str(args.report_json.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

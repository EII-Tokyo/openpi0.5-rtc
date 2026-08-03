#!/usr/bin/env python3
"""Aggregate two fresh ALOHA Home/Sleep Isaac runs into a fail-closed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _limit_conflicts(run: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    lower, upper = run["preflight"]["limits"]["follower_left"]
    conflicts = []
    for index, (name, target) in enumerate(
        zip(manifest["joint_order"], manifest["sleep_rad"], strict=True)
    ):
        if target < lower[index] or target > upper[index]:
            conflicts.append(
                {
                    "joint_name": name,
                    "joint_index": index,
                    "official_sleep_target_rad": float(target),
                    "frozen_usd_lower_rad": float(lower[index]),
                    "frozen_usd_upper_rad": float(upper[index]),
                    "violation_rad": max(
                        float(lower[index]) - float(target),
                        float(target) - float(upper[index]),
                    ),
                }
            )
    return conflicts


def build_digital_report(
    run_1: dict[str, Any],
    run_2: dict[str, Any],
    visual_review: dict[str, Any],
    manifest: dict[str, Any],
    source_boundary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the digital gate without weakening endpoint or limit requirements."""

    signatures = [
        run_1["summary"]["normalized_numeric_signature"],
        run_2["summary"]["normalized_numeric_signature"],
    ]
    repeatable = signatures[0] == signatures[1]
    hashes_frozen = all(
        run["stage"]["sha256_before"] == run["stage"]["sha256_after"]
        and run["manifest"]["sha256_before"] == run["manifest"]["sha256_after"]
        and run["source_or_final_asset_modified"] is False
        for run in (run_1, run_2)
    )
    command_identity = all(
        run["manifest"]["command_signature"] == manifest["command_signature"]
        for run in (run_1, run_2)
    )
    visual_pass = str(visual_review["status"]).startswith("PASS")
    runs_pass = run_1["status"] == run_2["status"] == "PASS"
    conflicts = _limit_conflicts(run_1, manifest)
    if runs_pass and repeatable and hashes_frozen and command_identity and visual_pass:
        status = "PASS"
        classification = "DIGITAL_HOME_SLEEP_VERIFIED"
    elif conflicts and all(
        run["summary"]["gates"].get("endpoints") is False
        and run["summary"]["gates"].get("legal_limits") is False
        for run in (run_1, run_2)
    ):
        status = "FAIL"
        classification = "OFFICIAL_SLEEP_TARGET_OUTSIDE_FROZEN_JOINT_LIMITS"
    else:
        status = "FAIL"
        classification = "DIGITAL_HOME_SLEEP_GATE_FAILED"
    report = {
        "schema_version": 1,
        "status": status,
        "classification": classification,
        "runtime": dict(run_1["runtime"]),
        "stage": dict(run_1["stage"]),
        "manifest": {
            "sha256": run_1["manifest"]["sha256_before"],
            "command_signature": manifest["command_signature"],
        },
        "fresh_process_run_count": 2,
        "numeric_repeatability": "PASS" if repeatable else "FAIL",
        "normalized_numeric_signatures": signatures,
        "limit_conflicts": conflicts,
        "gates": {
            "both_numeric_runs_pass": runs_pass,
            "numeric_signatures_match": repeatable,
            "hashes_remained_frozen": hashes_frozen,
            "command_identity": command_identity,
            "visual_evidence_review": visual_pass,
            "all_endpoints_reached": all(
                run["summary"]["gates"].get("endpoints") is True
                for run in (run_1, run_2)
            ),
            "all_targets_legal": all(
                run["summary"]["gates"].get("legal_limits") is True
                for run in (run_1, run_2)
            ),
            "three_cycles_completed": all(
                run["summary"]["gates"].get("three_cycles_complete") is True
                for run in (run_1, run_2)
            ),
            "final_home_reached": all(
                run["summary"]["gates"].get("final_home") is True
                for run in (run_1, run_2)
            ),
            "follower_right_stationary": all(
                run["summary"]["gates"].get("follower_right_stationary") is True
                for run in (run_1, run_2)
            ),
            "grippers_stationary": all(
                run["summary"]["gates"].get("grippers_stationary") is True
                for run in (run_1, run_2)
            ),
            "no_impulse_carrying_contact": all(
                run["summary"]["gates"].get("no_impulse_carrying_contact") is True
                for run in (run_1, run_2)
            ),
        },
        "runs": [
            {
                "repeat_index": run.get("repeat_index"),
                "status": run["status"],
                "summary": run["summary"],
                "telemetry": run.get("telemetry"),
            }
            for run in (run_1, run_2)
        ],
        "visual_evidence": visual_review,
        "real_execution_authorized": False,
        "real_preflight_status": "NOT_RUN_DIGITAL_GATE_FAILED"
        if status != "PASS"
        else "NOT_RUN_AUTHORIZATION_REQUIRED",
        "real_execution_status": "NOT_RUN_DIGITAL_GATE_FAILED"
        if status != "PASS"
        else "NOT_RUN_AUTHORIZATION_REQUIRED",
        "source_or_final_asset_modified": False,
        "task8_status": "COMPLETE_WITH_NO_PROMOTION",
    }
    if source_boundary is not None:
        report["source_boundary"] = source_boundary
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Home/Sleep digital validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Numeric repeatability: `{report['numeric_repeatability']}`",
        f"- Fresh Isaac processes: `{report['fresh_process_run_count']}`",
        f"- Real preflight: `{report['real_preflight_status']}`",
        f"- Real execution: `{report['real_execution_status']}`",
        "- Real execution authorized: `false`",
        "",
        "## Limit conflicts",
        "",
        "| Joint | Official Sleep | Frozen lower | Frozen upper | Violation |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            "| `{joint_name}` | {official_sleep_target_rad:.6f} | "
            "{frozen_usd_lower_rad:.6f} | {frozen_usd_upper_rad:.6f} | "
            "{violation_rad:.6f} |".format(**item)
        )
        for item in report["limit_conflicts"]
    )
    lines.extend(
        [
            "",
            "The official ALOHA Sleep command is preserved exactly. The frozen USD/URDF "
            "limits are also preserved exactly. PhysX clamps the three out-of-range targets, "
            "so direction, repeatability, stationary bodies, contact absence, and final Home "
            "pass, but the Sleep endpoint and legal-target gates fail.",
            "",
            "No real-robot command was sent and this report does not authorize one.",
        ]
    )
    if "source_boundary" in report:
        boundary = report["source_boundary"]
        lines.extend(
            [
                "",
                "## Source boundary",
                "",
                f"- Pinned exact-model official Sleep: `{boundary['official_sleep_rad']}` rad.",
                f"- Local third-party mirror Sleep: `{boundary['local_mirror_sleep_rad']}` rad.",
                "- The local mirror differs and is explicitly not treated as official authority.",
                "- A historical read-only robot report also differs; it is retained as project "
                "evidence, not used to authorize or generate motion in this run.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-1", type=Path, default=REPORT_ROOT / "aloha1_home_sleep_digital_run_01.json"
    )
    parser.add_argument(
        "--run-2", type=Path, default=REPORT_ROOT / "aloha1_home_sleep_digital_run_02.json"
    )
    parser.add_argument(
        "--visual",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_evidence_review.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_command_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_validation.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_validation.md",
    )
    parser.add_argument(
        "--parameter-source-audit",
        type=Path,
        default=REPORT_ROOT / "aloha1_official_parameter_source_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inputs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.run_1, args.run_2, args.visual, args.manifest)
    ]
    parameter_audit = json.loads(args.parameter_source_audit.read_text(encoding="utf-8"))
    mirror = next(
        item
        for item in parameter_audit["local_mirror_observations"]
        if item["id"] == "local_aloha_sleep_positions_differ_from_pinned_upstream"
    )
    report = build_digital_report(
        *inputs,
        source_boundary={
            "official_sleep_rad": mirror["pinned_official_sleep_positions"][:6],
            "official_source_class": "OFFICIAL_PINNED_EXACT_MODEL_SOURCE",
            "local_mirror_sleep_rad": mirror["local_mirror_sleep_positions"][:6],
            "local_mirror_source_class": "THIRD_PARTY_AGGREGATE_LOCAL_MIRROR",
            "local_mirror_used_as_command_authority": False,
            "historical_robot_report": str(
                ROOT
                / "docs/aloha1_isaac_adaptation/09_phase4_real_aloha1_joint_signal_probe_2026-07-17.md"
            ),
            "historical_robot_report_used_as_command_authority": False,
        },
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

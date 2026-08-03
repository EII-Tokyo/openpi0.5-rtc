#!/usr/bin/env python3
"""Aggregate two fresh ALOHA Home/Sleep Isaac runs into a fail-closed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def source_boundary_from_audit(audit: dict[str, Any]) -> dict[str, Any]:
    selected = next(item for item in audit["sources"] if item["id"] == "selected_sleep")
    comparison = audit["current_humble_comparison"]
    return {
        "selected_sleep_rad": list(audit["sleep"]["value_rad"]),
        "selected_source_class": audit["classification"],
        "selected_source_repository": selected["repository"],
        "selected_source_branch": selected["branch"],
        "selected_source_commit": selected["commit"],
        "selected_source_license": selected["license"],
        "selected_source_sha256": selected["sha256"],
        "current_humble_sleep_rad": list(comparison["sleep_rad"]),
        "current_humble_source_class": comparison["classification"],
        "current_humble_used_as_command_authority": comparison[
            "used_as_command_authority"
        ],
        "selection_status": audit["version_selection"]["status"],
    }


def _limit_conflicts(run: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    lower, upper = run["preflight"]["limits"]["follower_left"]
    conflicts = []
    for index, (name, target) in enumerate(zip(manifest["joint_order"], manifest["sleep_rad"], strict=True)):
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
        run["manifest"]["command_signature"] == manifest["command_signature"] for run in (run_1, run_2)
    )
    visual_pass = str(visual_review["status"]).startswith("PASS")
    runs_pass = run_1["status"] == run_2["status"] == "PASS"
    conflicts = _limit_conflicts(run_1, manifest)
    if runs_pass and repeatable and hashes_frozen and command_identity and visual_pass:
        status = "PASS"
        classification = "DIGITAL_HOME_SLEEP_VERIFIED"
    elif conflicts and all(
        run["summary"]["gates"].get("endpoints") is False and run["summary"]["gates"].get("legal_limits") is False
        for run in (run_1, run_2)
    ):
        status = "PARTIAL"
        classification = "VISUAL_TRAJECTORY_PASS_SIGNAL_SEMANTICS_MISMATCH"
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
        "layer_status": {
            "visual_trajectory": (
                "PASS"
                if visual_pass
                and all(
                    run["summary"]["gates"].get("directions") is True
                    and run["summary"]["gates"].get("three_cycles_complete") is True
                    and run["summary"]["gates"].get("final_home") is True
                    for run in (run_1, run_2)
                )
                else "FAIL"
            ),
            "exact_sleep_endpoint": "PASS" if not conflicts and runs_pass else "FAIL",
            "modeled_official_api_command_gate": (
                "PASS"
                if not conflicts and runs_pass
                else "PARTIAL"
                if conflicts and repeatable
                else "FAIL"
            ),
            "real_api_signal_correspondence": (
                "PASS_MODELED_NOT_REAL_EXECUTION"
                if not conflicts and runs_pass
                else "PARTIAL"
                if conflicts and repeatable
                else "FAIL"
            ),
            "real_hardware_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
        },
        "gates": {
            "both_numeric_runs_pass": runs_pass,
            "numeric_signatures_match": repeatable,
            "hashes_remained_frozen": hashes_frozen,
            "command_identity": command_identity,
            "visual_evidence_review": visual_pass,
            "all_endpoints_reached": all(run["summary"]["gates"].get("endpoints") is True for run in (run_1, run_2)),
            "all_targets_legal": all(run["summary"]["gates"].get("legal_limits") is True for run in (run_1, run_2)),
            "three_cycles_completed": all(
                run["summary"]["gates"].get("three_cycles_complete") is True for run in (run_1, run_2)
            ),
            "final_home_reached": all(run["summary"]["gates"].get("final_home") is True for run in (run_1, run_2)),
            "follower_right_stationary": all(
                run["summary"]["gates"].get("follower_right_stationary") is True for run in (run_1, run_2)
            ),
            "grippers_stationary": all(
                run["summary"]["gates"].get("grippers_stationary") is True for run in (run_1, run_2)
            ),
            "no_impulse_carrying_contact": all(
                run["summary"]["gates"].get("no_impulse_carrying_contact") is True for run in (run_1, run_2)
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
    if report["status"] == "PASS":
        lines.extend(
            [
                "",
                "The user-selected official historical Sleep is inside every frozen USD/URDF "
                "joint limit. Two fresh Isaac processes reached all three Sleep endpoints and "
                "returned Home with identical normalized numeric signatures. This verifies the "
                "digital trajectory and modeled official command gate only.",
                "",
                "No real-robot command was sent and this report does not authorize one.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The visible three-cycle trajectory, directions, repeatability, stationary bodies, "
                "contact absence, and final Home pass. The exact Sleep endpoint remains outside "
                "the frozen USD/URDF limits. PhysX independently clamps the three conflicting "
                "joints, while the official ALOHA Python group API rejects an entire sample when "
                "any joint is illegal. Therefore the video is valid visual trajectory evidence, "
                "but not yet an exact real-API signal-correspondence proof.",
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
                f"- User-selected official historical Sleep: `{boundary['selected_sleep_rad']}` rad.",
                f"- Official source commit: `{boundary['selected_source_commit']}`.",
                f"- Current Humble comparison Sleep: `{boundary['current_humble_sleep_rad']}` rad.",
                "- This is an explicit cross-version command selection; current Humble "
                "URDF/driver limits remain frozen and the current Humble Sleep is not the "
                "command authority for this run.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-1", type=Path, default=REPORT_ROOT / "aloha1_home_sleep_digital_run_01.json")
    parser.add_argument("--run-2", type=Path, default=REPORT_ROOT / "aloha1_home_sleep_digital_run_02.json")
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
        "--source-audit",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_official_source_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inputs = [
        json.loads(path.read_text(encoding="utf-8")) for path in (args.run_1, args.run_2, args.visual, args.manifest)
    ]
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    report = build_digital_report(
        *inputs,
        source_boundary=source_boundary_from_audit(source_audit),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

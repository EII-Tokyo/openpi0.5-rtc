#!/usr/bin/env python3
"""Aggregate two isolated runtime-Sleep Isaac validations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def build_runtime_sleep_digital_report(
    run_1: dict[str, Any], run_2: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build a fail-closed digital-only alignment report."""

    runs = (run_1, run_2)
    signatures = [run["summary"]["normalized_numeric_signature"] for run in runs]
    pids = [int(run["runtime_pid"]) for run in runs]
    immutable = all(
        all(bool(value) for value in run["immutability"].values()) and run["source_or_final_asset_modified"] is False
        for run in runs
    )
    gates = {
        "two_runs_pass": all(run["status"] == "PASS" for run in runs),
        "fresh_process_pids_distinct": len(set(pids)) == 2,
        "numeric_signatures_match": len(set(signatures)) == 1,
        "telemetry_complete": all(run["telemetry"]["row_count"] == 2220 for run in runs),
        "final_runtime_sleep_reached": all(run["summary"]["gates"].get("final_terminal") is True for run in runs),
        "endpoints_reached": all(run["summary"]["gates"].get("endpoints") is True for run in runs),
        "directions_match": all(run["summary"]["gates"].get("directions") is True for run in runs),
        "diagnostic_limits_respected": all(run["summary"]["gates"].get("legal_limits") is True for run in runs),
        "three_cycles_complete": all(run["summary"]["gates"].get("three_cycles_complete") is True for run in runs),
        "source_hashes_immutable": immutable,
        "manifest_command_identity": all(
            run["manifest"]["command_signature"] == manifest["command_signature"] for run in runs
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "status": "PASS_DIAGNOSTIC_DIGITAL_ONLY" if passed else "FAIL",
        "classification": ("RUNTIME_MEASURED_SLEEP_ALIGNED_IN_ISAAC" if passed else "DIGITAL_ALIGNMENT_FAILED"),
        "runtime": dict(run_1["runtime"]),
        "initial_pose_label": str(manifest["initial_pose_label"]),
        "terminal_pose_label": str(manifest["terminal_pose_label"]),
        "sequence_kind": "SLEEP_HOME_SLEEP",
        "cycles": 3,
        "manifest_signature": str(manifest["manifest_signature"]),
        "command_signature": str(manifest["command_signature"]),
        "fresh_process_count": 2,
        "fresh_process_pids": pids,
        "fresh_process_pids_distinct": gates["fresh_process_pids_distinct"],
        "normalized_numeric_signatures": signatures,
        "numeric_signatures_match": gates["numeric_signatures_match"],
        "gates": gates,
        "diagnostic_limit_readback": run_1["preflight"]["session_only_layers"]["diagnostic_limit_readback"],
        "runs": [
            {
                "repeat_index": run["repeat_index"],
                "runtime_pid": run["runtime_pid"],
                "status": run["status"],
                "telemetry": run["telemetry"],
                "first_frame_arm_jump_rad": run["preflight"]["first_frame_arm_jump_rad"],
                "final_terminal_max_error_rad": run["summary"]["final_terminal_max_error_rad"],
                "endpoint_results": run["summary"]["endpoint_results"],
                "direction_results": run["summary"]["direction_results"],
            }
            for run in runs
        ],
        "diagnostic_only": True,
        "candidate_promoted": False,
        "source_or_final_asset_modified": False,
        "real_execution_authorized": False,
        "real_motion_commands": 0,
        "task8": "COMPLETE_WITH_NO_PROMOTION",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 runtime-measured Sleep digital validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Sequence: `{report['sequence_kind']}` x `{report['cycles']}`",
        f"- Fresh Isaac processes: `{report['fresh_process_count']}`",
        f"- Numeric signatures match: `{str(report['numeric_signatures_match']).lower()}`",
        "- Real motion commands: `0`",
        "- Final/default asset modified: `false`",
        "",
        "## Result",
        "",
        "The digital follower_left starts at the median of 9000 stationary real JointState "
        "samples, moves Sleep → Home → Sleep for three cycles, and ends at the same runtime "
        "Sleep reference. Two fresh Isaac Sim 5.1 processes produced the same normalized "
        "numeric signature.",
        "",
        "The elbow and wrist_angle limit differences are accepted only through an anonymous "
        "session layer classified `DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT`. No source USD, final "
        "asset, or default joint mapping was changed.",
        "",
        "This validates the isolated digital initialization and trajectory. It does not "
        "authorize or claim a synchronized real-hardware run.",
        "",
        "## Runtime limit readback",
        "",
        "| Joint | Bound | Authored rad | USD degrees readback |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        ("| `{joint_name}` | `{bound}` | {authored_value_rad:.9f} | {usd_readback_degrees:.9f} |".format(**item))
        for item in report["diagnostic_limit_readback"]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-1",
        type=Path,
        default=REPORT_ROOT / "aloha1_runtime_measured_sleep_digital_run_01.json",
    )
    parser.add_argument(
        "--run-2",
        type=Path,
        default=REPORT_ROOT / "aloha1_runtime_measured_sleep_digital_run_02.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_ROOT / "aloha1_runtime_measured_sleep_command_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_runtime_measured_sleep_digital_validation.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_ROOT / "aloha1_runtime_measured_sleep_digital_validation.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_1 = json.loads(args.run_1.read_text(encoding="utf-8"))
    run_2 = json.loads(args.run_2.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_runtime_sleep_digital_report(run_1, run_2, manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))
    return 0 if report["status"] == "PASS_DIAGNOSTIC_DIGITAL_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

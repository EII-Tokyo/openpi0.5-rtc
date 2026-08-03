#!/usr/bin/env python3
"""Fail-closed ALOHA Home/Sleep real runner; default invocation is DRY_RUN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.home_sleep_real_safety import build_dry_run_plan
from tools.aloha1_mapping.home_sleep_real_safety import validate_real_execution_gate

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_runner_report(
    *,
    execute_real: bool,
    robot: str,
    digital_status: str,
    preflight_status: str,
    manifest_sha_matches: bool,
    preflight_manifest_sha_matches: bool,
    authorization: dict[str, Any],
    manifest_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    """Classify execution permission before any live transport may be imported."""

    if not execute_real:
        return build_dry_run_plan(
            manifest_sha256=manifest_sha256,
            digital_status=digital_status,
            sample_count=sample_count,
        )
    gates = {
        "execute_real": execute_real,
        "robot": robot,
        "manifest_sha_matches": manifest_sha_matches,
        "digital_report_status": digital_status,
        "preflight_report_status": preflight_status,
        "preflight_manifest_sha_matches": preflight_manifest_sha_matches,
        "real_motion_authorized": authorization.get("real_motion_authorized"),
        "operator_workspace_clear": authorization.get("operator_workspace_clear"),
        "stop_control_ready": authorization.get("stop_control_ready"),
    }
    gate = validate_real_execution_gate(gates)
    return {
        "schema_version": 1,
        "mode": "EXECUTE_REAL_REQUESTED",
        "status": "READY_FOR_LIVE_ADAPTER" if gate["status"] == "PASS" else "BLOCKED",
        "gate": gate,
        "manifest_sha256": manifest_sha256,
        "planned_samples": sample_count,
        "network_access_performed": False,
        "ros_transport_instantiated": False,
        "ssh_connection_opened": False,
        "serial_device_opened": False,
        "torque_changed": False,
        "commands_published": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--robot", default="follower_left")
    parser.add_argument(
        "--digital-report",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_validation.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_real_preflight.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_command_manifest.json",
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_real_run.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    digital = json.loads(args.digital_report.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_hash = _sha256(args.manifest)
    authorization = (
        json.loads(args.authorization.read_text(encoding="utf-8"))
        if args.authorization is not None
        else {}
    )
    report = build_runner_report(
        execute_real=args.execute_real,
        robot=args.robot,
        digital_status=str(digital["status"]),
        preflight_status=str(preflight["status"]),
        manifest_sha_matches=(digital["manifest"]["sha256"] == manifest_hash),
        preflight_manifest_sha_matches=(preflight["manifest"]["sha256"] == manifest_hash),
        authorization=authorization,
        manifest_sha256=manifest_hash,
        sample_count=int(manifest["sample_count"]),
    )
    if report["status"] == "READY_FOR_LIVE_ADAPTER":
        report["status"] = "BLOCKED_LIVE_ADAPTER_NOT_CONFIGURED"
        report["gate"]["failed_gates"] = ["live_publisher_adapter"]
        report["gate"]["transport_may_be_instantiated"] = False
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "commands_published": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

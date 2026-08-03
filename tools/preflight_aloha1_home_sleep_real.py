#!/usr/bin/env python3
"""Build an offline fail-closed preflight for future ALOHA real motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.home_sleep_real_safety import build_dry_run_plan

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preflight_report(
    *,
    digital_report: dict[str, Any],
    manifest: dict[str, Any],
    digital_report_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Build only the offline portion; never contact the robot from this function."""

    dry_run = build_dry_run_plan(
        manifest_sha256=manifest_sha256,
        digital_status=str(digital_report["status"]),
        sample_count=int(manifest["sample_count"]),
    )
    return {
        "schema_version": 1,
        "status": dry_run["status"],
        "classification": "ALOHA_REAL_HOME_SLEEP_OFFLINE_PREFLIGHT",
        "digital_report": {
            "sha256": digital_report_sha256,
            "status": digital_report["status"],
            "classification": digital_report.get("classification"),
        },
        "manifest": {
            "sha256": manifest_sha256,
            "command_signature": manifest["command_signature"],
            "sample_count": manifest["sample_count"],
        },
        "robot": "follower_left",
        "read_only_remote_checks_performed": False,
        "network_access_performed": False,
        "ssh_connection_opened": False,
        "ros_transport_instantiated": False,
        "serial_device_opened": False,
        "torque_changed": False,
        "commands_published": 0,
        "real_execution_authorized": False,
        "remaining_live_checks": [
            "exact robot identity and arm namespace",
            "current joint readback near Home",
            "motor torque/error/temperature status",
            "operator workspace clear",
            "operator stop control ready",
        ],
        "boundary": (
            "No access to 192.168.1.103 is permitted while the digital gate is not PASS."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 Home/Sleep real preflight",
            "",
            f"- Status: `{report['status']}`",
            f"- Digital gate: `{report['digital_report']['status']}`",
            "- Remote access performed: `false`",
            "- ROS/SSH/serial transport instantiated: `false`",
            "- Torque changed: `false`",
            "- Commands published: `0`",
            "",
            report["boundary"],
            "",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digital-report",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_validation.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_command_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_real_preflight.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_real_preflight.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    digital = json.loads(args.digital_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_preflight_report(
        digital_report=digital,
        manifest=manifest,
        digital_report_sha256=_sha256(args.digital_report),
        manifest_sha256=_sha256(args.manifest),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "network_access_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

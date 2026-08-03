#!/usr/bin/env python3
"""Build the machine-readable offline gate for supervised real/sim replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/aloha1_mapping"
DEFAULT_OUTPUT = REPORTS / "aloha1_home_sleep_sync_offline_preflight.json"
DEFAULT_MARKDOWN = REPORTS / "aloha1_home_sleep_sync_offline_preflight.md"
DEFAULT_INPUTS = {
    "fake_coordinator": REPORTS / "aloha1_home_sleep_sync_fake_run.json",
    "real_worker_dry_run": REPORTS
    / "aloha1_home_sleep_sync_real_worker_dry_run.json",
    "ros1_source_preflight": REPORTS
    / "aloha1_home_sleep_sync_real_preflight.json",
    "isaac_api_audit": REPORTS / "aloha1_home_sleep_sync_isaac_api_audit.json",
    "isaac_fresh_01": REPORTS / "aloha1_home_sleep_sync_isaac_fresh_01.json",
    "isaac_fresh_02": REPORTS / "aloha1_home_sleep_sync_isaac_fresh_02.json",
    "isaac_paced": REPORTS / "aloha1_home_sleep_sync_isaac_paced.json",
}

REMAINING_LIVE_GATES = (
    "real_access_authorized",
    "read_only_103_preflight_pass",
    "deployed_joint_order_verified",
    "deployed_position_mode_verified",
    "stop_path_verified",
    "cam_high_stream_verified",
    "operator_workspace_clear",
    "real_motion_authorized",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_offline_readiness(
    *,
    fake_status: str,
    isaac_statuses: list[str],
    isaac_signatures: list[str],
    ros_source_audit_status: str,
    prohibited_side_effects_detected: bool,
) -> dict[str, Any]:
    """Classify offline evidence without implying that real motion occurred."""

    gates = {
        "fake_coordinator": fake_status == "PASS_FAKE_TRANSPORT",
        "isaac_process_statuses": bool(isaac_statuses)
        and all(status == "PASS" for status in isaac_statuses),
        "isaac_deterministic_signature": bool(isaac_signatures)
        and len(set(isaac_signatures)) == 1,
        "ros1_official_source_audit": ros_source_audit_status
        in {"NOT_RUN_AUTHORIZATION_REQUIRED", "PASS"},
        "prohibited_side_effects_absent": not prohibited_side_effects_detected,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "status": (
            "READY_FOR_SUPERVISED_REAL_EXECUTION"
            if not failed
            else "BLOCKED_OFFLINE_PREFLIGHT"
        ),
        "offline_gates": gates,
        "failed_gates": failed,
        "remaining_live_gates": list(REMAINING_LIVE_GATES),
        "real_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
        "real_digital_correspondence": "NOT_RUN_REAL_EVIDENCE_MISSING",
        "ready_is_not_real_validation": True,
    }


def _side_effect_detected(report: dict[str, Any]) -> bool:
    boolean_fields = (
        "network_access_performed",
        "ssh_connection_opened",
        "ros_transport_instantiated",
        "serial_device_opened",
        "torque_changed",
        "publisher_constructed",
    )
    if any(report.get(field) is True for field in boolean_fields):
        return True
    numeric_fields = (
        "commands_published_to_real_hardware",
        "commands_published",
    )
    return any(int(report.get(field, 0)) != 0 for field in numeric_fields)


def build_report(input_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Load and bind every offline evidence report by absolute path and hash."""

    paths = input_paths or DEFAULT_INPUTS
    payloads = {
        name: json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    isaac_names = ("isaac_fresh_01", "isaac_fresh_02", "isaac_paced")
    signatures = [
        str(payloads[name]["summary"]["normalized_numeric_signature"])
        for name in isaac_names
    ]
    classification = classify_offline_readiness(
        fake_status=str(payloads["fake_coordinator"]["status"]),
        isaac_statuses=[str(payloads[name]["status"]) for name in isaac_names],
        isaac_signatures=signatures,
        ros_source_audit_status=str(payloads["ros1_source_preflight"]["status"]),
        prohibited_side_effects_detected=any(
            _side_effect_detected(payloads[name])
            for name in (
                "fake_coordinator",
                "real_worker_dry_run",
                "ros1_source_preflight",
            )
        ),
    )
    return {
        "schema_version": 1,
        **classification,
        "scope": "ALOHA1_FOLLOWER_LEFT_SYNCHRONIZED_HOME_SLEEP_OFFLINE_GATE",
        "frozen_command": {
            "sample_count": 1850,
            "command_rate_hz": 50,
            "command_signature": payloads["fake_coordinator"]["identity"][
                "command_signature"
            ],
            "manifest_sha256": payloads["fake_coordinator"]["identity"][
                "manifest_sha256"
            ],
        },
        "isaac": {
            "runtime": payloads["isaac_paced"]["runtime"],
            "numeric_signature": signatures[0],
            "fresh_process_count": len(isaac_names),
            "signatures_identical": len(set(signatures)) == 1,
            "paced_scheduler": payloads["isaac_paced"]["scheduler"],
            "source_or_final_asset_modified": any(
                bool(payloads[name].get("source_or_final_asset_modified"))
                for name in isaac_names
            ),
        },
        "real_transport": {
            "ros_adapter_status": payloads["ros1_source_preflight"]["status"],
            "worker_status": payloads["real_worker_dry_run"]["status"],
            "publisher_constructed": False,
            "commands_published": 0,
            "torque_changed": False,
            "network_access_performed": False,
        },
        "inputs": {
            name: {
                "absolute_path": str(path.resolve(strict=True)),
                "sha256": _sha256(path),
                "status": payloads[name].get("status"),
            }
            for name, path in paths.items()
        },
        "task8": "COMPLETE_WITH_NO_PROMOTION",
    }


def _markdown(report: dict[str, Any]) -> str:
    gates = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in report["offline_gates"].items()
    )
    remaining = "\n".join(
        f"- `{name}`" for name in report["remaining_live_gates"]
    )
    return f"""# ALOHA1 synchronized Home/Sleep offline gate

Status: **{report['status']}**

This status means the offline protocol and Isaac worker are ready for a
separately authorized, supervised real-hardware run. It does **not** mean that
the real robot was accessed or that real/digital correspondence already passed.

## Offline gates

{gates}

## Isaac evidence

- Runtime: Isaac Sim `{report['isaac']['runtime']['isaac_sim']}`, Kit
  `{report['isaac']['runtime']['kit']}`, PhysX `{report['isaac']['runtime']['physx']}`
- Fresh processes: `{report['isaac']['fresh_process_count']}`
- Identical signature: `{report['isaac']['numeric_signature']}`
- Paced start skew: `{report['isaac']['paced_scheduler']['start_skew_ns']} ns`
- Paced maximum lateness: `{report['isaac']['paced_scheduler']['maximum_lateness_ns']} ns`
- Burst catch-up: `{str(report['isaac']['paced_scheduler']['burst_catchup_used']).lower()}`

## Remaining live gates

{remaining}

Real execution remains **{report['real_execution']}**. No ROS publisher,
network connection, motor command, or torque change was made by this gate.
Task 8 remains `{report['task8']}`.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report()
    output = args.output.resolve()
    markdown = args.markdown.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "real_execution": report["real_execution"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "READY_FOR_SUPERVISED_REAL_EXECUTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())

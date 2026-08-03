#!/usr/bin/env python3
"""Generate the offline/read-only gate report for real ALOHA replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_sync_real_preflight.json"
)
CORE_SOURCE_ROOT = (
    ROOT
    / ".codex/artifacts/20260803-aloha1-synchronized-real-sim/sources/interbotix_ros_core_noetic"
)
MANIPULATORS_SOURCE_ROOT = (
    ROOT / ".codex/artifacts/20260803-aloha-home-sleep-root-cause/history_probe"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_offline_preflight_report() -> dict[str, Any]:
    """Describe verified local source evidence without touching ROS or 103."""

    message_source = (
        CORE_SOURCE_ROOT
        / "interbotix_ros_xseries/interbotix_xs_msgs/msg/JointGroupCommand.msg"
    )
    driver_source = (
        CORE_SOURCE_ROOT
        / "interbotix_ros_xseries/interbotix_xs_sdk/src/xs_sdk_obj.cpp"
    )
    example_source = (
        MANIPULATORS_SOURCE_ROOT
        / "interbotix_ros_xsarms/examples/interbotix_xsarm_pid/src/xsarm_pid.cpp"
    )
    core_license_source = CORE_SOURCE_ROOT / "LICENSE"
    manipulators_license_source = MANIPULATORS_SOURCE_ROOT / "LICENSE"
    sources = [
        message_source,
        driver_source,
        example_source,
        core_license_source,
        manipulators_license_source,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned official sources: {missing}")
    return {
        "schema_version": 1,
        "status": "NOT_RUN_AUTHORIZATION_REQUIRED",
        "scope": "OFFLINE_SOURCE_AUDIT_ONLY",
        "real_access_authorized": False,
        "real_motion_authorized": False,
        "network_access_performed": False,
        "ros_transport_instantiated": False,
        "publisher_constructed": False,
        "commands_published": 0,
        "torque_changed": False,
        "live_blockers": [
            "real_access_authorized",
            "real_motion_authorized",
            "operator_workspace_clear",
            "stop_path_verified",
            "joint_order_verified",
            "camera_ready",
            "manifest_hash_match",
            "digital_gate_pass",
        ],
        "official_source_evidence": {
            "interbotix_ros_core": {
                "repository": "https://github.com/Interbotix/interbotix_ros_core.git",
                "branch": "noetic",
                "commit": "172841ffa93f7556fff9c7455ad0e77688fa156e",
                "license": "BSD-3-Clause",
                "local_path": str(CORE_SOURCE_ROOT.resolve()),
            },
            "interbotix_ros_manipulators": {
                "repository": "https://github.com/Interbotix/interbotix_ros_manipulators.git",
                "branch": "noetic",
                "commit": "0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9",
                "license": "BSD-3-Clause",
                "local_path": str(MANIPULATORS_SOURCE_ROOT.resolve()),
            },
            "files": [
                {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}
                for path in sources
            ],
            "verified_semantics": {
                "message_fields": ["string name", "float32[] cmd"],
                "arm_group_name": "arm",
                "relative_command_topic": "commands/joint_group",
                "command_order_source": "motor_config.yaml groups entry",
                "position_mode_units": "radian",
            },
            "unverified_runtime_semantics": {
                "deployed_joint_order": "READ_ONLY_103_PREFLIGHT_REQUIRED",
                "deployed_operating_mode": "READ_ONLY_103_PREFLIGHT_REQUIRED",
                "stop_hold_path": "READ_ONLY_103_PREFLIGHT_REQUIRED",
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_offline_preflight_report()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "publisher_constructed": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

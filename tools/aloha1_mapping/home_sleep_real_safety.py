"""Pure fail-closed safety gates for future ALOHA Home/Sleep real execution."""

from __future__ import annotations

from typing import Any

REQUIRED_BOOLEAN_GATES = (
    "execute_real",
    "manifest_sha_matches",
    "preflight_manifest_sha_matches",
    "real_motion_authorized",
    "operator_workspace_clear",
    "stop_control_ready",
)


def validate_real_execution_gate(gates: dict[str, Any]) -> dict[str, Any]:
    """Reject execution unless every explicit hardware gate is satisfied."""

    failed = [name for name in REQUIRED_BOOLEAN_GATES if gates.get(name) is not True]
    if gates.get("robot") != "follower_left":
        failed.append("robot")
    if gates.get("digital_report_status") != "PASS":
        failed.append("digital_report_status")
    if gates.get("preflight_report_status") != "PASS":
        failed.append("preflight_report_status")
    return {
        "status": "PASS" if not failed else "BLOCKED",
        "failed_gates": failed,
        "transport_may_be_instantiated": not failed,
    }


def build_dry_run_plan(
    *, manifest_sha256: str, digital_status: str, sample_count: int
) -> dict[str, Any]:
    """Return an offline plan whose construction has no network or device effects."""

    status = (
        "NOT_RUN_DIGITAL_GATE_FAILED"
        if digital_status != "PASS"
        else "NOT_RUN_AUTHORIZATION_REQUIRED"
    )
    return {
        "schema_version": 1,
        "mode": "DRY_RUN",
        "status": status,
        "manifest_sha256": manifest_sha256,
        "digital_report_status": digital_status,
        "planned_samples": int(sample_count),
        "network_access_performed": False,
        "ros_transport_instantiated": False,
        "ssh_connection_opened": False,
        "serial_device_opened": False,
        "torque_changed": False,
        "commands_published": 0,
        "real_motion_authorized": False,
    }

"""Fail-closed ROS1 adapter helpers for synchronized ALOHA replay.

This module intentionally has no module-level ROS imports.  A live ROS adapter
can only be loaded after every explicit live gate passes.  Read-only discovery
uses an injected factory and never asks that factory to create a publisher.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import math
from typing import Any, Protocol

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER

LIVE_GATES = (
    "real_access_authorized",
    "real_motion_authorized",
    "operator_workspace_clear",
    "stop_path_verified",
    "joint_order_verified",
    "camera_ready",
    "manifest_hash_match",
    "digital_gate_pass",
)

EXPECTED_READ_ONLY_SNAPSHOT = {
    "joint_states_topic": "/puppet_left/joint_states",
    "joint_states_type": "sensor_msgs/JointState",
    "command_topic": "/puppet_left/commands/joint_group",
    "command_type": "interbotix_xs_msgs/JointGroupCommand",
    "camera_topic": "/cam_high",
    "camera_type": "aloha.msg/RGBGrayscaleImage",
    "operating_mode": "position",
    "group_name": "arm",
}


class ReadOnlyRosFactory(Protocol):
    def read_only_snapshot(self) -> Mapping[str, object]:
        """Return discovered topic, message, group, mode, and safety facts."""


def serialize_joint_group_command(
    message_type: Callable[[], Any],
    group_name: str,
    q_rad: Sequence[float],
) -> Any:
    """Create the exact official ROS1 JointGroupCommand for the six arm DOFs."""

    if group_name != "arm":
        raise ValueError("follower_left arm replay requires group_name='arm'")
    commands = [float(value) for value in q_rad]
    if len(commands) != len(ARM_JOINT_ORDER):
        raise ValueError(f"expected {len(ARM_JOINT_ORDER)} arm commands")
    if not all(math.isfinite(value) for value in commands):
        raise ValueError("all arm commands must be finite")
    message = message_type()
    message.name = group_name
    message.cmd = commands
    return message


def live_adapter_gate(gates: Mapping[str, object]) -> dict[str, object]:
    """Evaluate every gate needed before a ROS command publisher may exist."""

    failed = [name for name in LIVE_GATES if gates.get(name) is not True]
    return {
        "status": "PASS" if not failed else "BLOCKED",
        "required_gates": list(LIVE_GATES),
        "failed_gates": failed,
        "publisher_permitted": not failed,
    }


def build_ros_adapter(
    authorization: Mapping[str, object],
    *,
    module_importer: Callable[[str], object] = importlib.import_module,
) -> dict[str, object]:
    """Load ROS modules only after live authorization and all safety gates pass.

    Loading modules does not create a publisher.  Publisher construction belongs
    to the separately authorized live execution entry point.
    """

    gate = live_adapter_gate(authorization)
    if gate["status"] != "PASS":
        authorization_missing = any(
            authorization.get(name) is not True
            for name in ("real_access_authorized", "real_motion_authorized")
        )
        return {
            **gate,
            "status": (
                "NOT_RUN_AUTHORIZATION_REQUIRED"
                if authorization_missing
                else "BLOCKED"
            ),
            "ros_modules_imported": False,
            "publisher_constructed": False,
        }

    rospy_module = module_importer("rospy")
    message_module = module_importer("interbotix_xs_msgs.msg")
    if not hasattr(message_module, "JointGroupCommand"):
        raise RuntimeError("interbotix_xs_msgs.msg.JointGroupCommand is unavailable")
    return {
        **gate,
        "status": "READY_FOR_EXPLICIT_PUBLISHER_CONSTRUCTION",
        "ros_modules_imported": True,
        "publisher_constructed": False,
        "rospy_module": rospy_module,
        "message_module": message_module,
    }


def run_read_only_preflight(factory: ReadOnlyRosFactory) -> dict[str, object]:
    """Evaluate a read-only ROS graph snapshot without creating a publisher."""

    snapshot = dict(factory.read_only_snapshot())
    failed: list[str] = []
    for field, expected in EXPECTED_READ_ONLY_SNAPSHOT.items():
        if snapshot.get(field) != expected:
            failed.append(field)
    if tuple(snapshot.get("joint_names", ())) != ARM_JOINT_ORDER:
        failed.append("joint_order_verified")
    if snapshot.get("stop_path_verified") is not True:
        failed.append("stop_path_verified")
    return {
        "schema_version": 1,
        "status": "PASS" if not failed else "PARTIAL",
        "snapshot": snapshot,
        "expected_joint_order": list(ARM_JOINT_ORDER),
        "failed_gates": failed,
        "publisher_constructed": False,
        "network_mutation_performed": False,
        "commands_published": 0,
        "torque_changed": False,
    }

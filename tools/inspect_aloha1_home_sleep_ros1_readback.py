#!/usr/bin/env python3
"""Read-only ROS1 runtime inventory for the ALOHA follower-left driver."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ARM_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
JOINT_STATES = "/puppet_left/joint_states"
ARM_COMMAND = "/puppet_left/commands/joint_group"
GRIPPER_COMMAND = "/puppet_left/commands/joint_single"


def build_dry_run_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "NOT_RUN_EXPLICIT_READ_ONLY_ROS_FLAG_REQUIRED",
        "ros_connected": False,
        "subscriber_constructed": False,
        "publisher_constructed": False,
        "services_called": 0,
        "commands_published_by_inspector": 0,
    }


def classify_live_snapshot(snapshot: dict[str, object]) -> dict[str, Any]:
    nodes = [str(name) for name in snapshot["nodes"]]  # type: ignore[index]
    joint_names = [
        str(name) for name in snapshot["joint_names"]  # type: ignore[index]
    ]
    positions = [
        float(value) for value in snapshot["joint_positions"]  # type: ignore[index]
    ]
    velocities = [
        float(value) for value in snapshot["joint_velocities"]  # type: ignore[index]
    ]
    efforts = [
        float(value) for value in snapshot["joint_efforts"]  # type: ignore[index]
    ]
    forbidden_driver_nodes = [
        name
        for name in nodes
        if name.endswith("/xs_sdk")
        and name != "/puppet_left/xs_sdk"
    ]
    arm_publishers = [
        str(name)
        for name in snapshot["arm_command_publishers"]  # type: ignore[index]
    ]
    gripper_publishers = [
        str(name)
        for name in snapshot["gripper_command_publishers"]  # type: ignore[index]
    ]
    checks = {
        "puppet_left_driver_present": "/puppet_left/xs_sdk" in nodes,
        "no_forbidden_driver_nodes": not forbidden_driver_nodes,
        "joint_order": tuple(joint_names[:6]) == ARM_ORDER,
        "joint_count": len(joint_names) == 9,
        "finite_joint_state": all(
            math.isfinite(value) for value in positions + velocities + efforts
        ),
        "joint_states_type": snapshot.get("joint_states_type")
        == "sensor_msgs/JointState",
        "arm_command_type": snapshot.get("arm_command_type")
        == "interbotix_xs_msgs/JointGroupCommand",
        "gripper_command_type": snapshot.get("gripper_command_type")
        == "interbotix_xs_msgs/JointSingleCommand",
        "no_arm_command_publishers": not arm_publishers,
        "no_gripper_command_publishers": not gripper_publishers,
        "arm_command_subscriber": "/puppet_left/xs_sdk"
        in snapshot.get("arm_command_subscribers", []),
        "gripper_command_subscriber": "/puppet_left/xs_sdk"
        in snapshot.get("gripper_command_subscribers", []),
        "load_configs_false": snapshot.get("load_configs") is False,
        "left_mode_config": str(snapshot.get("mode_configs", "")).endswith(
            "/aloha/config/puppet_modes_left.yaml"
        ),
        "vx300s_motor_config": str(snapshot.get("motor_configs", "")).endswith(
            "/interbotix_xsarm_control/config/vx300s.yaml"
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "PASS_PUPPET_LEFT_READ_ONLY_RUNTIME"
            if all(checks.values())
            else "FAIL_PUPPET_LEFT_RUNTIME_SCOPE"
        ),
        "checks": checks,
        "arm_joint_order": joint_names[:6],
        "full_joint_order": joint_names,
        "joint_positions": positions,
        "joint_velocities": velocities,
        "joint_efforts": efforts,
        "nodes": nodes,
        "forbidden_driver_nodes": forbidden_driver_nodes,
        "arm_command_publishers": arm_publishers,
        "gripper_command_publishers": gripper_publishers,
        "commands_published_by_inspector": 0,
        "publisher_constructed": False,
        "services_called": 0,
        "snapshot": snapshot,
    }


def _topic_type(rostopic_module: Any, name: str) -> str | None:
    topic_type, _, _ = rostopic_module.get_topic_type(name, blocking=True)
    return str(topic_type) if topic_type else None


def run_live(*, output: Path) -> int:
    import rosgraph
    import rosnode
    import rospy
    import rostopic
    from sensor_msgs.msg import JointState

    rospy.init_node(
        "aloha1_home_sleep_read_only_inspector",
        anonymous=False,
        disable_signals=True,
    )
    message = rospy.wait_for_message(JOINT_STATES, JointState, timeout=5.0)
    publishers, subscribers, _ = rosgraph.Master(
        "/aloha1_home_sleep_read_only_inspector"
    ).getSystemState()
    publisher_map = {str(topic): list(nodes) for topic, nodes in publishers}
    subscriber_map = {str(topic): list(nodes) for topic, nodes in subscribers}
    snapshot: dict[str, object] = {
        "nodes": sorted(rosnode.get_node_names()),
        "joint_names": list(message.name),
        "joint_positions": list(message.position),
        "joint_velocities": list(message.velocity),
        "joint_efforts": list(message.effort),
        "joint_state_header": {
            "seq": int(message.header.seq),
            "stamp_ns": int(message.header.stamp.to_nsec()),
            "frame_id": str(message.header.frame_id),
        },
        "joint_states_type": _topic_type(rostopic, JOINT_STATES),
        "arm_command_type": _topic_type(rostopic, ARM_COMMAND),
        "gripper_command_type": _topic_type(rostopic, GRIPPER_COMMAND),
        "arm_command_publishers": publisher_map.get(ARM_COMMAND, []),
        "gripper_command_publishers": publisher_map.get(GRIPPER_COMMAND, []),
        "arm_command_subscribers": subscriber_map.get(ARM_COMMAND, []),
        "gripper_command_subscribers": subscriber_map.get(GRIPPER_COMMAND, []),
        "load_configs": rospy.get_param("/puppet_left/xs_sdk/load_configs"),
        "mode_configs": rospy.get_param("/puppet_left/xs_sdk/mode_configs"),
        "motor_configs": rospy.get_param("/puppet_left/xs_sdk/motor_configs"),
        "subscriber_constructed": True,
        "publisher_constructed": False,
        "services_called": 0,
    }
    report = classify_live_snapshot(snapshot)
    report["ros_connected"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": report["status"], "output": str(output)},
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_PUPPET_LEFT_READ_ONLY_RUNTIME" else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-read-only-ros", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if not args.execute_read_only_ros:
        report = build_dry_run_report()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
        return 2
    return run_live(output=output)


if __name__ == "__main__":
    raise SystemExit(main())

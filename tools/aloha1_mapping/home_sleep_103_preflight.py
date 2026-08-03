"""Pure parsing and classification for the read-only 103 preflight."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

EXPECTED_PROJECT_ROOT = "/home/eii/openpi0.5-rtc-reward-learning"
EXPECTED_HASHES = {
    "COMPOSE_SHA256": "04dd806b4a79796c80e39fa4d290ee523933271729695faf50850499fdd30cfd",
    "ROBOT_UTILS_SHA256": "479dd5845d639f32775460b1225d9f4b2d8611a2588649574de37237d83489d7",
    "CONSTANTS_SHA256": "19a72e01cba604ecbf9775067a3df5926ed489b8369fdbec4bde4597cd0389db",
}
_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")

REMOTE_READ_ONLY_SCRIPT = r"""set -euo pipefail
cd /home/eii/openpi0.5-rtc-reward-learning
printf 'PROJECT_ROOT=%s\n' "$PWD"
printf 'GIT_HEAD=%s\n' "$(git rev-parse HEAD)"
printf 'GIT_BRANCH=%s\n' "$(git branch --show-current)"
printf 'GIT_DIRTY_COUNT=%s\n' "$(git status --short | wc -l)"
printf 'ROBOT_CONTAINER_COUNT=%s\n' "$(docker ps --format '{{.Names}}' | grep -Ec '(^|[-_])(aloha_ros_nodes|ros_master|rosbridge)([-_]|$)' || true)"
printf 'ROS_MASTER_PORT_LISTENERS=%s\n' "$(ss -ltnH 'sport = :11311' | wc -l)"
printf 'ROSBRIDGE_PORT_LISTENERS=%s\n' "$(ss -ltnH 'sport = :9090' | wc -l)"
printf 'IMAGE_ID=%s\n' "$(docker image inspect lyl472324464/robot:aloha-ros1.0 --format '{{.Id}}')"
printf 'IMAGE_DIGEST=%s\n' "$(docker image inspect lyl472324464/robot:aloha-ros1.0 --format '{{index .RepoDigests 0}}')"
compose_hash="$(sha256sum docker-compose.yml | cut -d' ' -f1)"
robot_utils_hash="$(sha256sum examples/aloha_real/robot_utils.py | cut -d' ' -f1)"
constants_hash="$(sha256sum examples/aloha_real/constants.py | cut -d' ' -f1)"
printf 'COMPOSE_SHA256=%s\n' "$compose_hash"
printf 'ROBOT_UTILS_SHA256=%s\n' "$robot_utils_hash"
printf 'CONSTANTS_SHA256=%s\n' "$constants_hash"
if [ "$compose_hash" = '04dd806b4a79796c80e39fa4d290ee523933271729695faf50850499fdd30cfd' ]; then
  printf 'ALOHA_LAUNCH_COMMAND_PRESENT=1\n'
  printf 'EXTERNAL_ALOHA_MOUNT_DECLARED=1\n'
else
  printf 'ALOHA_LAUNCH_COMMAND_PRESENT=0\n'
  printf 'EXTERNAL_ALOHA_MOUNT_DECLARED=0\n'
fi
if [ "$robot_utils_hash" = '479dd5845d639f32775460b1225d9f4b2d8611a2588649574de37237d83489d7' ]; then
  printf 'STATIC_JOINT_STATE_TOPIC_PRESENT=1\n'
  printf 'STATIC_JOINT_COMMAND_TOPIC_PRESENT=1\n'
  printf 'STATIC_CAM_HIGH_PRESENT=1\n'
else
  printf 'STATIC_JOINT_STATE_TOPIC_PRESENT=0\n'
  printf 'STATIC_JOINT_COMMAND_TOPIC_PRESENT=0\n'
  printf 'STATIC_CAM_HIGH_PRESENT=0\n'
fi
if [ "$constants_hash" = '19a72e01cba604ecbf9775067a3df5926ed489b8369fdbec4bde4597cd0389db' ]; then
  printf 'STATIC_JOINT_ORDER_PRESENT=1\n'
else
  printf 'STATIC_JOINT_ORDER_PRESENT=0\n'
fi
"""


def parse_remote_snapshot(text: str) -> dict[str, str]:
    """Parse bounded KEY=VALUE output and ignore SSH banners."""

    snapshot: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if _KEY.fullmatch(key):
            snapshot[key] = value
    return snapshot


def classify_remote_snapshot(snapshot: Mapping[str, str]) -> dict[str, Any]:
    """Separate static deployment evidence from unavailable runtime evidence."""

    static_checks = {
        "project_root": snapshot.get("PROJECT_ROOT") == EXPECTED_PROJECT_ROOT,
        "compose_hash": snapshot.get("COMPOSE_SHA256")
        == EXPECTED_HASHES["COMPOSE_SHA256"],
        "robot_utils_hash": snapshot.get("ROBOT_UTILS_SHA256")
        == EXPECTED_HASHES["ROBOT_UTILS_SHA256"],
        "constants_hash": snapshot.get("CONSTANTS_SHA256")
        == EXPECTED_HASHES["CONSTANTS_SHA256"],
        "joint_order_declared": snapshot.get("STATIC_JOINT_ORDER_PRESENT") == "1",
        "joint_state_topic_declared": snapshot.get(
            "STATIC_JOINT_STATE_TOPIC_PRESENT"
        )
        == "1",
        "joint_command_topic_declared": snapshot.get(
            "STATIC_JOINT_COMMAND_TOPIC_PRESENT"
        )
        == "1",
        "cam_high_declared": snapshot.get("STATIC_CAM_HIGH_PRESENT") == "1",
        "launch_command_declared": snapshot.get("ALOHA_LAUNCH_COMMAND_PRESENT")
        == "1",
        "external_mount_boundary_recorded": snapshot.get(
            "EXTERNAL_ALOHA_MOUNT_DECLARED"
        )
        == "1",
    }
    static_pass = all(static_checks.values())
    stack_running = (
        int(snapshot.get("ROBOT_CONTAINER_COUNT", "0")) > 0
        and int(snapshot.get("ROS_MASTER_PORT_LISTENERS", "0")) > 0
    )
    if not static_pass:
        status = "FAIL_STATIC_DEPLOYMENT_MISMATCH"
        runtime_status = "NOT_RUN_STATIC_GATE_FAILED"
    elif not stack_running:
        status = "PARTIAL_RUNTIME_STACK_STOPPED"
        runtime_status = "NOT_RUN_ROBOT_STACK_STOPPED"
    else:
        status = "PARTIAL_RUNTIME_READBACK_REQUIRED"
        runtime_status = "NOT_RUN_READ_ONLY_ROS_DISCOVERY_REQUIRED"
    return {
        "schema_version": 1,
        "status": status,
        "static_project_evidence": "PASS" if static_pass else "FAIL",
        "runtime_ros_evidence": runtime_status,
        "static_checks": static_checks,
        "remote_git": {
            "head": snapshot.get("GIT_HEAD"),
            "branch": snapshot.get("GIT_BRANCH"),
            "dirty_entry_count": int(snapshot.get("GIT_DIRTY_COUNT", "0")),
            "preserve_remote_dirty_worktree": True,
        },
        "robot_stack": {
            "container_count": int(snapshot.get("ROBOT_CONTAINER_COUNT", "0")),
            "ros_master_listener_count": int(
                snapshot.get("ROS_MASTER_PORT_LISTENERS", "0")
            ),
            "rosbridge_listener_count": int(
                snapshot.get("ROSBRIDGE_PORT_LISTENERS", "0")
            ),
        },
        "robot_image": {
            "id": snapshot.get("IMAGE_ID"),
            "digest": snapshot.get("IMAGE_DIGEST"),
        },
        "static_signal_contract": {
            "joint_order": [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            ],
            "joint_states_topic": "/puppet_left/joint_states",
            "command_topic": "/puppet_left/commands/joint_group",
            "camera_topic": "/cam_high",
            "evidence_class": "REMOTE_PROJECT_SOURCE_HASH_BOUND",
        },
        "external_mount": {
            "path": "/home/eii/openpi0.5-rtc/third_party/aloha",
            "declared_in_compose": snapshot.get("EXTERNAL_ALOHA_MOUNT_DECLARED")
            == "1",
            "inspected": False,
            "reason": "OUTSIDE_APPROVED_103_PROJECT_BOUNDARY",
        },
        "remaining_gates": [
            "authorization_to_start_robot_driver",
            "deployed_runtime_joint_order",
            "deployed_runtime_position_mode",
            "cam_high_runtime_message",
            "operator_tested_stop_hold_path",
            "operator_workspace_clear",
            "real_motion_authorized",
        ],
        "read_only_103_access_performed": True,
        "publisher_constructed": False,
        "commands_published": 0,
        "torque_changed": False,
        "real_motion_authorized": False,
        "real_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
    }

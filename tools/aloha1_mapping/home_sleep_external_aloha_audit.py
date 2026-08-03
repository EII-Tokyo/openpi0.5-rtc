"""Static audit helpers for the user-authorized external ROS1 ALOHA source.

The helpers are deliberately transport-free. They parse source text and a
candidate launch file; they do not import ROS, open serial devices, construct
publishers, or start a driver.
"""

from __future__ import annotations

import ast
import base64
from collections.abc import Mapping
import re
from typing import Any
import xml.etree.ElementTree as ET

import yaml

EXTERNAL_ALOHA_ROOT = "/home/eii/openpi0.5-rtc/third_party/aloha"
EXPECTED_FILES = (
    "launch/ros_nodes.launch",
    "launch/4arms_teleop.launch",
    "config/master_modes_left.yaml",
    "config/master_modes_right.yaml",
    "config/puppet_modes_left.yaml",
    "config/puppet_modes_right.yaml",
    "aloha_scripts/realsense_publisher.py",
    "aloha_scripts/robot_utils.py",
    "aloha_scripts/sleep.py",
    "msg/RGBGrayscaleImage.msg",
    "LICENSE",
)

REMOTE_EXTERNAL_READ_ONLY_SCRIPT = r"""set -euo pipefail
cd /home/eii/openpi0.5-rtc/third_party/aloha
printf 'EXTERNAL_ROOT=%s\n' "$PWD"
printf 'GIT_TOPLEVEL=%s\n' "$(git rev-parse --show-toplevel)"
printf 'GIT_HEAD=%s\n' "$(git rev-parse HEAD)"
printf 'GIT_BRANCH=%s\n' "$(git branch --show-current)"
printf 'GIT_DIRTY_COUNT=%s\n' "$(git status --short | wc -l)"
printf 'GIT_ORIGIN=%s\n' "$(git remote get-url origin)"
for path in \
  launch/ros_nodes.launch \
  launch/4arms_teleop.launch \
  config/master_modes_left.yaml \
  config/master_modes_right.yaml \
  config/puppet_modes_left.yaml \
  config/puppet_modes_right.yaml \
  aloha_scripts/realsense_publisher.py \
  aloha_scripts/robot_utils.py \
  aloha_scripts/sleep.py \
  msg/RGBGrayscaleImage.msg \
  LICENSE
do
  key="$(printf '%s' "$path" | tr '[:lower:]/.' '[:upper:]___')"
  printf 'FILE_%s_SHA256=%s\n' "$key" "$(sha256sum "$path" | cut -d' ' -f1)"
  printf 'FILE_%s_B64=%s\n' "$key" "$(base64 -w0 "$path")"
done
"""

_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _args(include: ET.Element) -> dict[str, str]:
    return {
        str(arg.get("name")): str(arg.get("value", arg.get("default", "")))
        for arg in include.findall("arg")
        if arg.get("name")
    }


def _driver_includes(root: ET.Element) -> list[dict[str, str]]:
    return [
        _args(include)
        for include in root.findall("include")
        if str(include.get("file", "")).endswith("/xsarm_control.launch")
    ]


def _launch_args(root: ET.Element) -> dict[str, str]:
    return {
        str(arg.get("name")): str(arg.get("value", arg.get("default", "")))
        for arg in root.findall("arg")
        if arg.get("name")
    }


def _resolve_arg(value: str, launch_args: Mapping[str, str]) -> str:
    match = re.fullmatch(r"\$\(arg ([A-Za-z0-9_]+)\)", value)
    return launch_args.get(match.group(1), value) if match else value


def _camera_names(source: str) -> list[str]:
    match = re.search(r"camera_names\s*=\s*(\[[^\]]*\])", source, re.DOTALL)
    if match is None:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def audit_external_aloha_source(
    *,
    ros_nodes_launch: str,
    puppet_left_modes: str,
    realsense_source: str,
    sleep_source: str,
) -> dict[str, Any]:
    """Classify whether the deployed source is safe for a left-only replay."""

    launch_root = ET.fromstring(ros_nodes_launch)
    driver_records = _driver_includes(launch_root)
    launch_args = _launch_args(launch_root)
    robot_names = [
        _resolve_arg(record.get("robot_name", ""), launch_args)
        for record in driver_records
    ]
    camera_nodes = [
        str(node.get("name"))
        for node in launch_root.findall("node")
        if node.get("type") == "realsense_publisher.py"
    ]
    modes = yaml.safe_load(puppet_left_modes)
    arm_mode = modes["groups"]["arm"]
    gripper_mode = modes["singles"]["gripper"]
    names = _camera_names(realsense_source)
    requires_four_cameras = (
        set(names)
        == {"cam_left_wrist", "cam_high", "cam_low", "cam_right_wrist"}
        and "missing_cams" in realsense_source
        and "raise Exception" in realsense_source
    )
    commands_both_puppets = all(
        token in sleep_source
        for token in ("puppet_left", "puppet_right", "move_arms")
    )
    left_only = robot_names == ["puppet_left"]
    return {
        "schema_version": 1,
        "status": (
            "PASS_STATIC_LEFT_ONLY_SCOPE"
            if left_only and not camera_nodes and not commands_both_puppets
            else "REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY"
        ),
        "driver_scope": {
            "include_count": len(driver_records),
            "robot_names": robot_names,
            "left_only": left_only,
        },
        "puppet_left_mode": {
            "port": modes.get("port"),
            "arm_operating_mode": arm_mode.get("operating_mode"),
            "arm_torque_enable": arm_mode.get("torque_enable"),
            "gripper_operating_mode": gripper_mode.get("operating_mode"),
            "gripper_torque_enable": gripper_mode.get("torque_enable"),
            "evidence_class": "REMOTE_SOURCE_READBACK",
        },
        "camera_scope": {
            "launch_nodes": camera_nodes,
            "camera_names": names,
            "requires_four_cameras": requires_four_cameras,
            "hardware_reset_present": "hardware_reset()" in realsense_source,
            "suitable_for_cam_high_only": False,
        },
        "sleep_scope": {
            "commands_both_puppets": commands_both_puppets,
            "suitable_for_left_only": not commands_both_puppets,
        },
        "side_effect_boundary": {
            "starting_existing_ros_nodes_launch_would_start_real_drivers": True,
            "starting_puppet_left_candidate_would_enable_arm_torque": bool(
                arm_mode.get("torque_enable")
            ),
            "starting_puppet_left_candidate_would_enable_gripper_torque": bool(
                gripper_mode.get("torque_enable")
            ),
            "existing_camera_process_resets_devices": "hardware_reset()"
            in realsense_source,
        },
        "real_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
    }


def validate_left_only_launch(text: str) -> dict[str, Any]:
    """Validate the inert left-only launch candidate without invoking ROS."""

    root = ET.fromstring(text)
    includes = _driver_includes(root)
    camera_nodes = [
        str(node.get("name", ""))
        for node in root.findall("node")
        if "camera" in str(node.get("name", "")).lower()
        or "realsense" in str(node.get("type", "")).lower()
    ]
    record = includes[0] if len(includes) == 1 else {}
    launch_args = _launch_args(root)
    mode_config = record.get("mode_configs", "")
    if mode_config == "$(arg puppet_modes_left)":
        mode_config = launch_args.get("puppet_modes_left", mode_config)
    robot_names = [item.get("robot_name", "") for item in includes]
    checks = {
        "one_driver_include": len(includes) == 1,
        "robot_name": robot_names == ["puppet_left"],
        "robot_model": record.get("robot_model") == "vx300s",
        "mode_config": mode_config.endswith("/puppet_modes_left.yaml"),
        "load_configs": record.get("load_configs") == "false",
        "use_sim": record.get("use_sim") == "false",
        "no_camera_node": not camera_nodes,
        "no_master_or_right": not any(
            token in text for token in ("master_left", "master_right", "puppet_right")
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "PASS_STATIC_LEFT_ONLY_SCOPE"
            if all(checks.values())
            else "FAIL_STATIC_LEFT_ONLY_SCOPE"
        ),
        "checks": checks,
        "robot_names": robot_names,
        "robot_model": record.get("robot_model"),
        "mode_config": mode_config,
        "load_configs": record.get("load_configs"),
        "use_sim": record.get("use_sim"),
        "camera_nodes": camera_nodes,
        "real_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
    }


def parse_snapshot(text: str) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if _KEY.fullmatch(key):
            snapshot[key] = value
    return snapshot


def _path_key(path: str) -> str:
    return path.translate(str.maketrans({"/": "_", ".": "_"})).upper()


def snapshot_file(snapshot: Mapping[str, str], path: str) -> str:
    key = _path_key(path)
    encoded = snapshot[f"FILE_{key}_B64"]
    return base64.b64decode(encoded).decode("utf-8")


def build_external_audit_report(
    snapshot: Mapping[str, str], *, candidate_text: str
) -> dict[str, Any]:
    """Build the machine report from one bounded external-source snapshot."""

    existing = audit_external_aloha_source(
        ros_nodes_launch=snapshot_file(snapshot, "launch/ros_nodes.launch"),
        puppet_left_modes=snapshot_file(
            snapshot, "config/puppet_modes_left.yaml"
        ),
        realsense_source=snapshot_file(
            snapshot, "aloha_scripts/realsense_publisher.py"
        ),
        sleep_source=snapshot_file(snapshot, "aloha_scripts/sleep.py"),
    )
    candidate = validate_left_only_launch(candidate_text)
    file_manifest = []
    for path in EXPECTED_FILES:
        sha_key = f"FILE_{_path_key(path)}_SHA256"
        if sha_key in snapshot:
            file_manifest.append(
                {
                    "path": f"{EXTERNAL_ALOHA_ROOT}/{path}",
                    "sha256": snapshot[sha_key],
                    "evidence_class": "REMOTE_SOURCE_READBACK",
                }
            )
    ready = (
        existing["status"] == "REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY"
        and candidate["status"] == "PASS_STATIC_LEFT_ONLY_SCOPE"
    )
    return {
        "schema_version": 1,
        "status": (
            "READY_FOR_MINIMAL_START_AUTHORIZATION"
            if ready
            else "BLOCKED_STATIC_SOURCE_OR_CANDIDATE"
        ),
        "scope": "READ_ONLY_EXTERNAL_ALOHA_SOURCE_AND_INERT_LAUNCH_CANDIDATE",
        "external_repository": {
            "local_path": snapshot.get("EXTERNAL_ROOT"),
            "git_toplevel": snapshot.get("GIT_TOPLEVEL"),
            "origin": snapshot.get("GIT_ORIGIN"),
            "branch": snapshot.get("GIT_BRANCH"),
            "commit": snapshot.get("GIT_HEAD"),
            "dirty_entry_count": int(snapshot.get("GIT_DIRTY_COUNT", "0")),
            "license": "MIT",
            "license_evidence": f"{EXTERNAL_ALOHA_ROOT}/LICENSE",
            "preserve_remote_dirty_worktree": True,
        },
        "file_manifest": file_manifest,
        "official_xsarm_control_source": {
            "repository": (
                "https://github.com/Interbotix/interbotix_ros_manipulators.git"
            ),
            "branch": "noetic",
            "commit": "0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9",
            "path": (
                "interbotix_ros_xsarms/interbotix_xsarm_control/launch/"
                "xsarm_control.launch"
            ),
            "sha256": (
                "58d4f5511bf71b8b7408891e3bfe9b43c4ceec2eec08291b33596882dcd01c5c"
            ),
            "license": "BSD-3-Clause",
            "evidence_class": "OFFICIAL_PINNED_SOURCE",
        },
        "existing_deployment": existing,
        "left_only_candidate": candidate,
        "camera_decision": {
            "existing_four_camera_publisher_accepted": False,
            "reason": (
                "REQUIRES_ALL_FOUR_SERIALS_AND_CALLS_HARDWARE_RESET; "
                "NOT_A_CAM_HIGH_ONLY_PATH"
            ),
            "cam_high_single_camera_runtime": "NOT_RUN_AUTHORIZATION_REQUIRED",
        },
        "stop_hold_decision": {
            "torque_off_accepted_as_generic_stop": False,
            "zero_joint_command_accepted_as_generic_stop": False,
            "operator_tested_stop_hold_path": "NOT_VERIFIED",
        },
        "remaining_gates": [
            "explicit_authorization_to_start_puppet_left_driver",
            "runtime_joint_order_and_position_mode_readback",
            "operator_tested_stop_hold_path",
            "cam_high_single_camera_runtime_path",
            "operator_workspace_clear",
            "explicit_real_motion_authorization",
        ],
        "authorization": {
            "external_source_read_only_authorized": True,
            "driver_started": False,
            "ros_publisher_constructed": False,
            "commands_published": 0,
            "torque_changed": False,
            "real_motion_authorized": False,
            "real_execution": "NOT_RUN_AUTHORIZATION_REQUIRED",
        },
    }

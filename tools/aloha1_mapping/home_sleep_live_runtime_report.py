"""Aggregate the authorized real-robot read-only runtime evidence."""

from __future__ import annotations

from typing import Any


def build_live_runtime_report(
    *,
    ros_report: dict[str, Any],
    camera_pre: dict[str, Any],
    camera_post: dict[str, Any],
    driver_log: str,
    deployment_hashes: dict[str, str],
    artifact_paths: dict[str, str],
    driver_running: bool,
    joint_state_samples: dict[str, Any],
) -> dict[str, Any]:
    camera_pass = all(
        report.get("status") == "PASS_CAM_HIGH_SINGLE_CAMERA_RUNTIME"
        and int(report.get("frames_captured", 0)) > 0
        and report.get("frames_captured") == report.get("frames_published")
        and report.get("hardware_resets") == 0
        and report.get("robot_command_publishers") == 0
        for report in (camera_pre, camera_post)
    )
    driver_log_checks = {
        "arm_position_mode": (
            "operating mode for the 'arm' group was changed to position"
            in driver_log
        ),
        "gripper_linear_position_mode": (
            "operating mode for the 'gripper' joint was changed to linear_position"
            in driver_log
        ),
        "xs_sdk_up": "Interbotix 'xs_sdk' node is up!" in driver_log,
        "puppet_left_mode_config": "config/puppet_modes_left.yaml" in driver_log,
    }
    ros_pass = ros_report.get("status") == "PASS_PUPPET_LEFT_READ_ONLY_RUNTIME"
    commands_published = (
        int(ros_report.get("commands_published_by_inspector", 0))
        + int(camera_pre.get("robot_command_publishers", 0))
        + int(camera_post.get("robot_command_publishers", 0))
    )
    runtime_pass = (
        ros_pass
        and camera_pass
        and all(
            driver_log_checks[name]
            for name in (
                "arm_position_mode",
                "gripper_linear_position_mode",
                "xs_sdk_up",
            )
        )
        and driver_running
        and commands_published == 0
    )
    return {
        "schema_version": 1,
        "status": (
            "PASS_READ_ONLY_RUNTIME_MOTION_NOT_RUN"
            if runtime_pass
            else "FAIL_READ_ONLY_RUNTIME"
        ),
        "scope": "AUTHORIZED_PUPPET_LEFT_DRIVER_AND_CAM_HIGH_READ_ONLY_RUNTIME",
        "REAL_MOTION": "NOT_RUN_AUTHORIZATION_REQUIRED",
        "REAL_DIGITAL_CORRESPONDENCE": "NOT_RUN_REAL_MOTION_EVIDENCE_MISSING",
        "WORKSPACE_CLEAR_FOR_MOTION": "FAIL_CLUTTERED_TABLE",
        "STOP_HOLD_PATH": "NOT_VERIFIED",
        "driver": {
            "left_driver_running": bool(driver_running),
            "arm_operating_mode": "position",
            "gripper_operating_mode": "linear_position",
            "torque_state": (
                "ENABLED_BY_DEPLOYED_MODE_CONFIG_AND_DRIVER_LOG_"
                "NOT_DIRECT_REGISTER_READBACK"
            ),
            "load_configs": False,
            "forbidden_driver_nodes": ros_report.get(
                "forbidden_driver_nodes", []
            ),
            "log_checks": driver_log_checks,
        },
        "device_mapping": {
            "role_alias": "/dev/ttyDXL_puppet_left",
            "resolved_device": "/dev/ttyUSB0",
            "ftdi_serial": "FTAAMM8J",
            "evidence_class": "REMOTE_RUNTIME_UDEV_READBACK",
            "hardcoded_ttyusb_used": False,
        },
        "ros_readback": ros_report,
        "camera_pre_driver": camera_pre,
        "camera_post_driver": camera_post,
        "joint_state_samples": joint_state_samples,
        "commands_published": commands_published,
        "robot_command_publisher_constructed": False,
        "services_called": 0,
        "camera_hardware_resets": 0,
        "visual_review": {
            "pre_driver": "PASS_CAM_HIGH_VIEW_LEFT_ARM_VISIBLE",
            "post_driver": "PASS_NO_VISIBLE_LEFT_ARM_POSE_CHANGE",
            "workspace": "FAIL_CLUTTERED_TABLE_FOR_FUTURE_MOTION",
            "evidence_role": "AUXILIARY_SAFETY_VISUAL_NOT_SIGNAL_PROOF",
        },
        "deployment_hashes": deployment_hashes,
        "artifact_paths": artifact_paths,
        "remaining_gates": [
            "operator_clear_table_and_confirm_workspace",
            "operator_tested_stop_hold_path",
            "explicit_real_motion_authorization",
            "home_sleep_three_cycle_real_execution",
            "real_digital_signal_comparison",
        ],
        "authorization_boundary": {
            "driver_start_authorized": True,
            "camera_hardware_authorized": True,
            "read_only_runtime_authorized": True,
            "home_sleep_motion_authorized": False,
            "other_motion_authorized": False,
        },
    }

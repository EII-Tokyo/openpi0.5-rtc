#!/usr/bin/env python3
"""Audit the official ALOHA ROS 2 Sleep command against its own limit chain."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import build_home_sleep_samples
from tools.aloha1_mapping.home_sleep_correspondence import evaluate_interbotix_group_limit_gate

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
RESEARCH_ROOT = ROOT / ".codex/artifacts/20260803-aloha-home-sleep-root-cause"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(
    *,
    source_id: str,
    repository: str,
    branch: str,
    commit: str,
    license_name: str,
    path: Path,
    role: str,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "id": source_id,
        "repository": repository,
        "branch": branch,
        "commit": commit,
        "license": license_name,
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "role": role,
    }


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    function = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise ValueError(f"missing function {name}")
    return function


def _calls_attribute(function: ast.FunctionDef, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attribute
    ]


def _inspect_python_semantics(robot_utils: Path, arm_module: Path) -> dict[str, bool]:
    robot_tree = ast.parse(robot_utils.read_text(encoding="utf-8"))
    move_arms = _function(robot_tree, "move_arms")
    sleep_arms = _function(robot_tree, "sleep_arms")
    set_calls = _calls_attribute(move_arms, "set_joint_positions")
    ignored = any(
        isinstance(parent, ast.Expr)
        for call in set_calls
        for parent in ast.walk(move_arms)
        if isinstance(parent, ast.Expr) and parent.value is call
    )
    sleep_uses_group_value = any(
        isinstance(node, ast.Attribute) and node.attr == "joint_sleep_positions" for node in ast.walk(sleep_arms)
    )

    arm_tree = ast.parse(arm_module.read_text(encoding="utf-8"))
    check_limits = _function(arm_tree, "_check_joint_limits")
    set_positions = _function(arm_tree, "set_joint_positions")
    go_sleep = _function(arm_tree, "go_to_sleep_pose")
    set_checks = bool(_calls_attribute(set_positions, "_check_joint_limits"))
    set_publishes = bool(_calls_attribute(set_positions, "_publish_commands"))
    go_sleep_checks = bool(_calls_attribute(go_sleep, "_check_joint_limits"))
    go_sleep_publishes = bool(_calls_attribute(go_sleep, "_publish_commands"))
    whole_group_loop = any(
        isinstance(node, ast.For)
        and any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node))
        for node in ast.walk(check_limits)
    )
    return {
        "aloha_sleep_reads_group_sleep_positions": sleep_uses_group_value,
        "aloha_sleep_uses_set_joint_positions": bool(set_calls),
        "set_joint_positions_return_value_ignored": ignored,
        "set_joint_positions_checks_whole_group": set_checks and set_publishes and whole_group_loop,
        "generic_go_to_sleep_pose_bypasses_python_limit_check": (go_sleep_publishes and not go_sleep_checks),
    }


def _xacro_limits(path: Path) -> tuple[list[float], list[float]]:
    text = path.read_text(encoding="utf-8")
    property_pattern = re.compile(r'<xacro:property name="(?P<name>[a-z_]+)"\s+value="(?P<value>[^"]+)"/>')
    properties = {match.group("name"): match.group("value") for match in property_pattern.finditer(text)}
    pi_offset = float(properties["pi_offset"])

    def parse_limit(name: str) -> float:
        expression = properties[name]
        radians_match = re.fullmatch(r"\$\{radians\((-?[0-9.]+)\)\}", expression)
        if radians_match:
            return math.radians(float(radians_match.group(1)))
        if expression == "${-pi + pi_offset}":
            return -math.pi + pi_offset
        if expression == "${pi - pi_offset}":
            return math.pi - pi_offset
        raise ValueError(f"unsupported Xacro limit expression for {name}: {expression}")

    lower: list[float] = []
    upper: list[float] = []
    for joint in ARM_JOINT_ORDER:
        lower.append(parse_limit(f"{joint}_limit_lower"))
        upper.append(parse_limit(f"{joint}_limit_upper"))
    return lower, upper


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sleep_from_git(repo: Path, ref: str) -> list[float]:
    relative = "interbotix_ros_xsarms/interbotix_xsarm_control/config/aloha_vx300s.yaml"
    text = _git(repo, "show", f"{ref}:{relative}")
    payload = yaml.safe_load(text)
    return [float(value) for value in payload["sleep_positions"][:6]]


def _branch_history(repo: Path) -> list[dict[str, Any]]:
    result = []
    for branch in ("humble", "iron", "jazzy", "rolling", "main", "noetic"):
        ref = f"origin/{branch}"
        head = _git(repo, "rev-parse", ref)
        head_date = _git(repo, "show", "-s", "--format=%aI", ref)
        file_commit = _git(
            repo,
            "log",
            "-1",
            "--format=%H",
            ref,
            "--",
            "interbotix_ros_xsarms/interbotix_xsarm_control/config/aloha_vx300s.yaml",
        )
        result.append(
            {
                "branch": branch,
                "head_commit": head,
                "head_author_date": head_date,
                "sleep_file_last_commit": file_commit,
                "sleep_rad": _sleep_from_git(repo, ref),
            }
        )
    return result


def _isaac_sleep_saturation(
    telemetry: Path,
    *,
    lower_rad: list[float],
    upper_rad: list[float],
) -> dict[str, Any]:
    """Verify from telemetry that each illegal target stopped at its own DOF limit."""

    with telemetry.open(newline="", encoding="utf-8") as stream:
        sleep_rows = [row for row in csv.DictReader(stream) if str(row["segment"]).endswith("sleep_hold")]
    if not sleep_rows:
        raise ValueError("telemetry contains no sleep_hold row")
    row = sleep_rows[-1]
    target = [float(value) for value in json.loads(row["target_arm_q"])]
    readback = [float(value) for value in json.loads(row["left_q"])[:6]]
    saturated = []
    for index, (name, goal, actual, lower, upper) in enumerate(
        zip(ARM_JOINT_ORDER, target, readback, lower_rad, upper_rad, strict=True)
    ):
        bound = lower if goal < lower else upper if goal > upper else None
        if bound is None:
            continue
        tolerance = 8.0 * 2.0**-23 * max(1.0, abs(bound))
        if abs(actual - bound) <= tolerance:
            saturated.append(
                {
                    "joint_name": name,
                    "joint_index": index,
                    "target_rad": goal,
                    "readback_rad": actual,
                    "saturated_bound_rad": bound,
                    "absolute_bound_error_rad": abs(actual - bound),
                    "float32_tolerance_rad": tolerance,
                }
            )
    expected = [
        name
        for name, goal, lower, upper in zip(ARM_JOINT_ORDER, target, lower_rad, upper_rad, strict=True)
        if not lower <= goal <= upper
    ]
    return {
        "status": (
            "VERIFIED_INDIVIDUAL_DOF_LIMIT_SATURATION"
            if [item["joint_name"] for item in saturated] == expected
            else "INCONCLUSIVE"
        ),
        "telemetry_absolute_path": str(telemetry.resolve(strict=True)),
        "telemetry_sha256": _sha256(telemetry),
        "selected_segment": row["segment"],
        "target_arm_q_rad": target,
        "left_arm_readback_rad": readback,
        "saturated_joints": saturated,
    }


def _limit_conflicts(values: list[float], lower_rad: list[float], upper_rad: list[float]) -> list[dict[str, Any]]:
    conflicts = []
    for index, (name, value, lower, upper) in enumerate(
        zip(ARM_JOINT_ORDER, values, lower_rad, upper_rad, strict=True)
    ):
        if not lower <= value <= upper:
            conflicts.append(
                {
                    "joint_name": name,
                    "joint_index": index,
                    "target_rad": value,
                    "lower_rad": lower,
                    "upper_rad": upper,
                    "violation_rad": max(lower - value, value - upper),
                }
            )
    return conflicts


def build_root_cause_report(
    *,
    official_sleep_rad: list[float],
    previous_sleep_rad: list[float],
    lower_rad: list[float],
    upper_rad: list[float],
    gate_result: dict[str, Any],
    source_facts: dict[str, bool],
    runtime_facts: dict[str, Any],
    source_records: list[dict[str, Any]],
    branch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the evidence-layered root-cause classification."""

    conflicts = _limit_conflicts(official_sleep_rad, lower_rad, upper_rad)
    previous_conflicts = _limit_conflicts(previous_sleep_rad, lower_rad, upper_rad)
    required_facts = (
        "aloha_sleep_uses_set_joint_positions",
        "set_joint_positions_checks_whole_group",
        "set_joint_positions_return_value_ignored",
        "generic_go_to_sleep_pose_bypasses_python_limit_check",
        "xs_sdk_group_callback_adds_no_urdf_limit_check",
    )
    runtime_verified = (
        runtime_facts.get("isaac_sleep_saturation", {}).get("status") == "VERIFIED_INDIVIDUAL_DOF_LIMIT_SATURATION"
    )
    verified = (
        [item["joint_name"] for item in conflicts] == ["shoulder", "elbow", "wrist_angle"]
        and not previous_conflicts
        and all(source_facts.get(name) is True for name in required_facts)
        and runtime_verified
        and gate_result.get("command_semantics") == "REJECT_WHOLE_GROUP_SAMPLE"
    )
    return {
        "schema_version": 1,
        "status": "VERIFIED_ROOT_CAUSE" if verified else "INCONCLUSIVE",
        "classification": (
            "OFFICIAL_ROS2_ALOHA_SLEEP_CONFIGURATION_OUTSIDE_ITS_OWN_URDF_LIMITS" if verified else "INCONCLUSIVE"
        ),
        "product": "Stationary ALOHA follower / aloha_vx300s",
        "joint_order": list(ARM_JOINT_ORDER),
        "official_ros2_sleep_rad": official_sleep_rad,
        "previous_official_sleep_rad": previous_sleep_rad,
        "urdf_lower_rad": lower_rad,
        "urdf_upper_rad": upper_rad,
        "limit_conflicts": conflicts,
        "official_ros2_sleep_within_limits": not conflicts,
        "previous_sleep_within_limits": not previous_conflicts,
        "interbotix_group_gate_emulation": gate_result,
        "source_facts": source_facts,
        "runtime_facts": runtime_facts,
        "source_records": source_records,
        "branch_history": branch_history,
        "video_interpretation": "PASS_TRAJECTORY_VISUAL",
        "exact_sleep_endpoint_status": "FAIL",
        "signal_correspondence_status": "PARTIAL",
        "real_execution_status": "NOT_RUN_UNAUTHORIZED",
        "real_robot_access_performed": False,
        "hardware_goal_register_outcome": "NOT_RUNTIME_VERIFIED",
        "hardware_boundary": (
            "ROBOTIS documents Goal Position as constrained by Min/Max Position Limit in "
            "single-turn Position Control Mode. The exact result of a broadcast Sync Write "
            "outside that range was not tested on hardware; the ALOHA set_joint_positions "
            "path rejects the group earlier."
        ),
        "isaac_interpretation": (
            "The existing runner submitted every sample and PhysX independently stopped "
            "each conflicting DOF at its joint limit. That stable visual motion is not the "
            "same command-layer behavior as Interbotix whole-group rejection."
        ),
        "recommended_digital_contract": (
            "Preserve the visual run as trajectory evidence, but model the selected real API "
            "path explicitly before any hardware comparison. Do not widen USD limits and do "
            "not silently replace the pinned humble Sleep vector."
        ),
        "official_legal_candidate": {
            "value_rad": previous_sleep_rad,
            "status": "OFFICIAL_LEGAL_CANDIDATE_REQUIRES_VERSION_SELECTION",
            "automatically_selected": False,
        },
        "final_or_default_asset_modified": False,
        "task8_status": "COMPLETE_WITH_NO_PROMOTION",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Sleep limit root-cause audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Video interpretation: `{report['video_interpretation']}`",
        f"- Exact Sleep endpoint: `{report['exact_sleep_endpoint_status']}`",
        f"- Signal correspondence: `{report['signal_correspondence_status']}`",
        f"- Real execution: `{report['real_execution_status']}`",
        "",
        "## What happened",
        "",
        "The video is valid evidence that the simulated arm moved smoothly through three "
        "cycles and returned Home. It was not a visual-motion failure. The mismatch is at "
        "the command boundary: the pinned ROS 2 ALOHA Sleep vector exceeds the pinned "
        "`aloha_vx300s` URDF limits for three joints.",
        "",
        "The official ALOHA helper interpolates and calls `set_joint_positions`. The "
        "Interbotix Python API rejects the whole group sample as soon as any one joint is "
        "illegal. The Isaac runner instead kept submitting commands and PhysX stopped each "
        "joint independently. These two behaviors can look similarly safe in a video while "
        "being different signals.",
        "",
        "## Conflicting joints",
        "",
        "| Joint | Sleep target | URDF lower | URDF upper | Violation |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| `{joint_name}` | {target_rad:.9f} | {lower_rad:.9f} | {upper_rad:.9f} | {violation_rad:.9f} |".format(**item)
        for item in report["limit_conflicts"]
    )
    gate = report["interbotix_group_gate_emulation"]
    lines.extend(
        [
            "",
            "## Deterministic API emulation",
            "",
            f"- First rejected outbound sample: `{gate['first_rejected_segment_sample']}` (zero-based of 250).",
            f"- Accepted outbound samples: `{gate['accepted_sample_count']}`.",
            f"- First rejecting joint: `{gate['first_rejected_joint_names']}`.",
            f"- Last publishable command: `{gate['last_published_q_rad']}` rad.",
            "- Semantics: `REJECT_WHOLE_GROUP_SAMPLE`; no per-joint clamp.",
            "",
            "## Source-history boundary",
            "",
            "The original official ALOHA variant used the in-range Sleep vector "
            f"`{report['previous_official_sleep_rad']}`. PR #225 changed only the ROS 2 "
            "motor-config Sleep vector to reduce arm drop after torque-off; the ALOHA "
            "ViperX URDF limits were not widened in that change. The ROS 1 `main/noetic` "
            "branches stopped before this ROS 2 change, so their older value is not an "
            "automatic replacement for the pinned `humble` configuration.",
            "",
            "## Decision",
            "",
            report["recommended_digital_contract"],
            "",
            "No real robot was contacted and no final/default asset was modified.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aloha1_home_sleep_correspondence.yaml")
    parser.add_argument(
        "--xacro",
        type=Path,
        default=ROOT / ".codex/artifacts/20260802-aloha1-official-model-first/sources/"
        "interbotix_manipulators_b66d5b905725351dd71d3251a06cd3f4c777940f/"
        "aloha_vx300s.urdf.xacro",
    )
    parser.add_argument(
        "--toolbox-arm",
        type=Path,
        default=RESEARCH_ROOT / "toolboxes_probe/interbotix_xs_toolbox/"
        "interbotix_xs_modules/interbotix_xs_modules/xs_robot/arm.py",
    )
    parser.add_argument(
        "--core-xs-sdk",
        type=Path,
        default=RESEARCH_ROOT / "core_probe/interbotix_ros_xseries/interbotix_xs_sdk/src/xs_sdk_obj.cpp",
    )
    parser.add_argument(
        "--driver",
        type=Path,
        default=ROOT / ".codex/artifacts/20260802-aloha1-official-model-first/sources/"
        "interbotix_xs_driver_da27b8b2b6c7677844f74581b82c01829a834e1c/"
        "xs_driver.cpp",
    )
    parser.add_argument("--history-repo", type=Path, default=RESEARCH_ROOT / "history_probe")
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=REPORT_ROOT / "aloha1_home_sleep_digital_telemetry_run_01.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "aloha1_sleep_limit_root_cause.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_ROOT / "aloha1_sleep_limit_root_cause.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    motor_path = (ROOT / config["sources"]["aloha_vx300s"]["local_path"]).resolve(strict=True)
    robot_utils = (ROOT / config["sources"]["robot_utils"]["local_path"]).resolve(strict=True)
    motor = yaml.safe_load(motor_path.read_text(encoding="utf-8"))
    official_sleep = [float(value) for value in motor["sleep_positions"][:6]]
    lower, upper = _xacro_limits(args.xacro)
    outbound = [
        sample
        for sample in build_home_sleep_samples(
            home=config["command"]["home_rad"],
            sleep=official_sleep,
            command_hz=int(config["command"]["command_rate_hz"]),
            move_seconds=int(config["command"]["move_seconds"]),
            hold_seconds=1,
            cycles=1,
        )
        if sample.segment == "cycle_01_home_to_sleep"
    ]
    gate = evaluate_interbotix_group_limit_gate(
        outbound,
        lower_rad=lower,
        upper_rad=upper,
        moving_time_s=2.0,
        velocity_limits_rad_s=[math.pi] * 6,
    )
    facts = _inspect_python_semantics(robot_utils, args.toolbox_arm)
    core_text = args.core_xs_sdk.read_text(encoding="utf-8")
    driver_text = args.driver.read_text(encoding="utf-8")
    facts.update(
        {
            "xs_sdk_group_callback_adds_no_urdf_limit_check": (
                "xs_driver->write_commands(msg->name, msg->cmd);" in core_text
            ),
            "xs_sdk_group_info_limits_come_from_urdf": (
                "ptr->limits->lower" in core_text and "ptr->limits->upper" in core_text
            ),
            "driver_converts_and_sync_writes_group_commands": (
                "convertRadian2Value" in driver_text and "dxl_wb.syncWrite" in driver_text
            ),
        }
    )
    records = [
        _source_record(
            source_id="aloha_robot_utils",
            repository=config["sources"]["robot_utils"]["repository"],
            branch=config["sources"]["robot_utils"]["branch"],
            commit=config["sources"]["robot_utils"]["commit"],
            license_name=config["sources"]["robot_utils"]["license"],
            path=robot_utils,
            role="official ALOHA interpolation and sleep_arms call path",
        ),
        _source_record(
            source_id="aloha_vx300s_motor_config",
            repository=config["sources"]["aloha_vx300s"]["repository"],
            branch=config["sources"]["aloha_vx300s"]["branch"],
            commit=config["sources"]["aloha_vx300s"]["commit"],
            license_name=config["sources"]["aloha_vx300s"]["license"],
            path=motor_path,
            role="official exact-model Sleep vector and motor limits",
        ),
        _source_record(
            source_id="aloha_vx300s_xacro",
            repository="https://github.com/Interbotix/interbotix_ros_manipulators.git",
            branch="humble",
            commit="b66d5b905725351dd71d3251a06cd3f4c777940f",
            license_name="BSD-3-Clause",
            path=args.xacro,
            role="official exact-model URDF joint limits",
        ),
        _source_record(
            source_id="interbotix_arm_python_api",
            repository="https://github.com/Interbotix/interbotix_ros_toolboxes.git",
            branch="humble",
            commit="f52234371b9ec1cb1f5ee8241da32e6b8476fa5c",
            license_name="BSD-3-Clause",
            path=args.toolbox_arm,
            role="official group limit-check and publish semantics; file unchanged from compatible dbe4d7d",
        ),
        _source_record(
            source_id="interbotix_xs_sdk",
            repository="https://github.com/Interbotix/interbotix_ros_core.git",
            branch="humble",
            commit="af18d4fe24ba08e09a0f1e92afaca1863e3205de",
            license_name="BSD-3-Clause",
            path=args.core_xs_sdk,
            role="official ROS group callback and URDF group-info limits",
        ),
        _source_record(
            source_id="interbotix_xs_driver",
            repository="https://github.com/Interbotix/interbotix_xs_driver.git",
            branch="v0.3.3",
            commit="da27b8b2b6c7677844f74581b82c01829a834e1c",
            license_name="BSD-3-Clause",
            path=args.driver,
            role="official radian conversion and group Sync Write path",
        ),
    ]
    records.extend(
        [
            {
                "id": "trossen_vx300s_specification",
                "url": "https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html",
                "authority": "Trossen Robotics / Interbotix",
                "accessed": "2026-08-03",
                "confirms": "exact actuator models and firmware-defined default joint limits",
            },
            {
                "id": "robotis_xm540_w270_manual",
                "url": "https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/",
                "authority": "ROBOTIS",
                "accessed": "2026-08-03",
                "confirms": "Position Control Goal Position is constrained by Min/Max Position Limit",
            },
            {
                "id": "robotis_protocol_2",
                "url": "https://emanual.robotis.com/docs/en/dxl/protocol2/",
                "authority": "ROBOTIS",
                "accessed": "2026-08-03",
                "confirms": "out-of-range writes are Data Range Error; Sync Write uses broadcast ID",
            },
        ]
    )
    previous = _sleep_from_git(args.history_repo, "dbc6aefb53e956181fe97f60474f1ad292491f0c")
    report = build_root_cause_report(
        official_sleep_rad=official_sleep,
        previous_sleep_rad=previous,
        lower_rad=lower,
        upper_rad=upper,
        gate_result=gate,
        source_facts=facts,
        runtime_facts={
            "isaac_sleep_saturation": _isaac_sleep_saturation(
                args.telemetry,
                lower_rad=lower,
                upper_rad=upper,
            )
        },
        source_records=records,
        branch_history=_branch_history(args.history_repo),
    )
    report["history_change"] = {
        "pull_request": "https://github.com/Interbotix/interbotix_ros_manipulators/pull/225",
        "humble_merge_commit": "46e979438ee4a78c11d0a779d8aab31146b9e0cd",
        "change_commit": "5b90ea49cdfadf13b9e238e1b088147d98fd671d",
        "author_stated_reason": "change Sleep positions so the arm does not drop as much",
        "xacro_limits_changed_in_same_commit": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))
    return 0 if report["status"] == "VERIFIED_ROOT_CAUSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

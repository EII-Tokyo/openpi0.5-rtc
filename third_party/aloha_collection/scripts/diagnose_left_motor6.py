#!/usr/bin/env python3
"""
Diagnose follower arm motor 6 behavior while the left gripper grasps an object.

This script is intentionally read-only by default: it does not send motion commands or
change torque/mode settings. It samples ROS joint states plus Dynamixel registers and
writes a JSONL log that can be inspected after the failure happens.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


ARM_JOINTS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
TARGET_JOINT = "wrist_rotate"  # ALOHA/ViperX arm motor ID 6.
SINGLE_JOINTS = ["gripper"]

REGISTERS = [
    "Operating_Mode",
    "Torque_Enable",
    "Hardware_Error_Status",
    "Shutdown",
    "Present_Position",
    "Present_Velocity",
    "Present_Current",
    "Present_PWM",
    "Present_Input_Voltage",
    "Present_Temperature",
    "Current_Limit",
    "Goal_Position",
    "Goal_Current",
    "Goal_PWM",
    "Moving",
]
DYNAMIXEL_OPERATING_MODE_NAMES = {
    0: "current",
    1: "velocity",
    3: "position",
    4: "extended_position",
    5: "current_based_position",
    16: "pwm",
}
MOTOR_DIAG_JOINTS = ARM_JOINTS + ["gripper"]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def _safe_call(label: str, func) -> Any:
    try:
        return _jsonable(func())
    except Exception as exc:
        return {"error": f"{label}: {type(exc).__name__}: {exc}"}


def _decode_register_values(reg: str, values: Any) -> list[str] | None:
    if reg != "Operating_Mode" or not isinstance(values, list) or not values:
        return None
    decoded = []
    for value in values:
        try:
            mode_value = int(value)
        except (TypeError, ValueError):
            decoded.append("unknown")
            continue
        decoded.append(DYNAMIXEL_OPERATING_MODE_NAMES.get(mode_value, f"unknown_{mode_value}"))
    return decoded


def _joint_state_map(bot: InterbotixManipulatorXS) -> dict[str, dict[str, Any]]:
    msg = bot.core.joint_states
    names = list(msg.name)
    positions = list(msg.position)
    velocities = list(msg.velocity)
    efforts = list(msg.effort)

    state: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(names):
        state[name] = {
            "index": idx,
            "position": positions[idx] if idx < len(positions) else None,
            "velocity": velocities[idx] if idx < len(velocities) else None,
            "effort": efforts[idx] if idx < len(efforts) else None,
        }
    return state


def _read_registers(bot: InterbotixManipulatorXS, all_arm_joints: bool) -> dict[str, Any]:
    joints = ARM_JOINTS if all_arm_joints else [TARGET_JOINT]
    registers: dict[str, Any] = {}

    for joint in joints:
        registers[joint] = {}
        for reg in REGISTERS:
            values = _safe_call(
                f"{joint}.{reg}",
                lambda joint=joint, reg=reg: bot.core.robot_get_motor_registers(
                    "single", joint, reg
                ),
            )
            registers[joint][reg] = values
            decoded = _decode_register_values(reg, values)
            if decoded is not None:
                registers[joint][f"{reg}_decoded"] = decoded

    for joint in SINGLE_JOINTS:
        registers[joint] = {}
        for reg in REGISTERS:
            values = _safe_call(
                f"{joint}.{reg}",
                lambda joint=joint, reg=reg: bot.core.robot_get_motor_registers(
                    "single", joint, reg
                ),
            )
            registers[joint][reg] = values
            decoded = _decode_register_values(reg, values)
            if decoded is not None:
                registers[joint][f"{reg}_decoded"] = decoded

    return registers


def _read_control_modes(bot: InterbotixManipulatorXS) -> dict[str, Any]:
    modes = {}
    for joint in MOTOR_DIAG_JOINTS:
        values = _safe_call(
            f"{joint}.Operating_Mode",
            lambda joint=joint: bot.core.robot_get_motor_registers(
                "single", joint, "Operating_Mode"
            ),
        )
        modes[joint] = {
            "Operating_Mode": values,
            "Operating_Mode_decoded": _decode_register_values(
                "Operating_Mode", values
            ),
        }
    return modes


def sample(bot: InterbotixManipulatorXS, phase: str, all_arm_joints: bool) -> dict[str, Any]:
    arm_positions = _safe_call("arm positions", bot.arm.get_joint_positions)
    arm_velocities = _safe_call("arm velocities", bot.arm.get_joint_velocities)
    arm_efforts = _safe_call("arm efforts", bot.arm.get_joint_efforts)
    gripper_position = _safe_call("gripper position", bot.gripper.get_gripper_position)
    gripper_velocity = _safe_call("gripper velocity", bot.gripper.get_gripper_velocity)
    gripper_effort = _safe_call("gripper effort", bot.gripper.get_gripper_effort)

    return {
        "ts": time.time(),
        "phase": phase,
        "robot": bot.core.robot_name,
        "target_motor": {
            "id": 6,
            "joint": TARGET_JOINT,
            "arm_joint_index": 5,
        },
        "all_motor_control_modes": _read_control_modes(bot),
        "joint_states": _joint_state_map(bot),
        "api": {
            "arm_positions": arm_positions,
            "arm_velocities": arm_velocities,
            "arm_efforts": arm_efforts,
            "gripper_position": gripper_position,
            "gripper_velocity": gripper_velocity,
            "gripper_effort": gripper_effort,
        },
        "registers": _read_registers(bot, all_arm_joints),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_robot_phase: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in samples:
        robot = item.get("robot", "unknown")
        phase = item.get("phase", "unknown")
        by_robot_phase.setdefault(robot, {}).setdefault(phase, []).append(item)

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    summary: dict[str, Any] = {"samples": len(samples), "robots": {}}
    for robot, phases in by_robot_phase.items():
        summary["robots"][robot] = {}
        for phase, items in phases.items():
            target_positions = []
            target_velocities = []
            target_efforts = []
            gripper_efforts = []
            hw_errors = []
            torque_values = []
            mode_values = []

            for item in items:
                wrist = item.get("joint_states", {}).get(TARGET_JOINT, {})
                gripper = item.get("joint_states", {}).get("gripper", {})
                for src, dest in [
                    (wrist.get("position"), target_positions),
                    (wrist.get("velocity"), target_velocities),
                    (wrist.get("effort"), target_efforts),
                    (gripper.get("effort"), gripper_efforts),
                ]:
                    if isinstance(src, (int, float)) and math.isfinite(src):
                        dest.append(float(src))

                regs = item.get("registers", {}).get(TARGET_JOINT, {})
                hw_errors.append(
                    json.dumps(regs.get("Hardware_Error_Status"), sort_keys=True)
                )
                torque_values.append(
                    json.dumps(regs.get("Torque_Enable"), sort_keys=True)
                )
                mode_values.append(
                    json.dumps(regs.get("Operating_Mode"), sort_keys=True)
                )

            summary["robots"][robot][phase] = {
                f"{TARGET_JOINT}_position": stats(target_positions),
                f"{TARGET_JOINT}_velocity": stats(target_velocities),
                f"{TARGET_JOINT}_effort": stats(target_efforts),
                "gripper_effort": stats(gripper_efforts),
                f"{TARGET_JOINT}_hardware_error_values": sorted(set(hw_errors)),
                f"{TARGET_JOINT}_torque_enable_values": sorted(set(torque_values)),
                f"{TARGET_JOINT}_operating_mode_values": sorted(set(mode_values)),
            }
    return summary


def write_json_line(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def collect_phase(
    *,
    bot: InterbotixManipulatorXS,
    phase: str,
    seconds: float,
    rate_hz: float,
    all_arm_joints: bool,
    output: Path,
) -> list[dict[str, Any]]:
    interval = 1.0 / rate_hz
    count = max(1, int(seconds * rate_hz))
    samples: list[dict[str, Any]] = []
    print(
        f"\n[{phase}] collecting {count} samples from {bot.core.robot_name} "
        f"at {rate_hz:.1f} Hz..."
    )
    for idx in range(count):
        t0 = time.time()
        item = sample(bot, phase, all_arm_joints)
        samples.append(item)
        write_json_line(output, item)

        wrist = item.get("joint_states", {}).get(TARGET_JOINT, {})
        gripper = item.get("joint_states", {}).get("gripper", {})
        if idx == 0 or idx == count - 1 or idx % max(1, int(rate_hz)) == 0:
            print(
                f"  {idx + 1:03d}/{count:03d} "
                f"{TARGET_JOINT}: pos={wrist.get('position')} "
                f"vel={wrist.get('velocity')} effort={wrist.get('effort')} | "
                f"gripper: pos={gripper.get('position')} effort={gripper.get('effort')}"
            )

        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic logger for follower motor 6 / wrist_rotate."
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["leader_left", "leader_right", "follower_left", "follower_right"],
        help="Robot names to sample. Default: leader_left leader_right follower_left follower_right",
    )
    parser.add_argument(
        "--robot-model",
        default=None,
        help="Optional model override for all robots. Defaults are inferred from robot name.",
    )
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--grasp-seconds", type=float, default=12.0)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path. Default: logs/motor6_diag_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--all-arm-joints",
        action="store_true",
        help="Also read registers for arm motors 1-5. Slower, but useful for comparison.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not pause between baseline and grasp phases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        print("--rate-hz must be > 0", file=sys.stderr)
        return 2

    output = (
        Path(args.output)
        if args.output
        else Path("logs") / f"motor6_diag_{int(time.time())}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Read-only motor diagnostic")
    print(f"  robots: {', '.join(args.robots)}")
    print(f"  target: motor ID 6 / {TARGET_JOINT}")
    print(f"  output: {output}")
    print()
    print("Make sure the normal ALOHA bringup is running before starting this script.")
    print("During the grasp phase, close the LEFT gripper on the object and keep it there.")

    robot_startup()
    node = create_interbotix_global_node()
    samples: list[dict[str, Any]] = []
    try:
        bots = [
            InterbotixManipulatorXS(
                robot_model=(
                    args.robot_model
                    or ("aloha_wx250s" if robot_name.startswith("leader_") else "aloha_vx300s")
                ),
                robot_name=robot_name,
                node=node,
                iterative_update_fk=False,
            )
            for robot_name in args.robots
        ]

        metadata = {
            "ts": time.time(),
            "kind": "metadata",
            "robot_names": args.robots,
            "robot_model": args.robot_model,
            "target_motor": {"id": 6, "joint": TARGET_JOINT},
            "control_mode_joints": MOTOR_DIAG_JOINTS,
            "registers": REGISTERS,
            "operating_mode_names": DYNAMIXEL_OPERATING_MODE_NAMES,
            "all_arm_joints": args.all_arm_joints,
        }
        write_json_line(output, metadata)

        for bot in bots:
            samples.extend(
                collect_phase(
                    bot=bot,
                    phase="baseline_open_or_no_load",
                    seconds=args.baseline_seconds,
                    rate_hz=args.rate_hz,
                    all_arm_joints=args.all_arm_joints,
                    output=output,
                )
            )

        if not args.no_wait:
            input(
                "\nNow close the LEFT gripper on the object. "
                "Press Enter immediately after it starts grasping..."
            )

        for bot in bots:
            samples.extend(
                collect_phase(
                    bot=bot,
                    phase="grasp_loaded",
                    seconds=args.grasp_seconds,
                    rate_hz=args.rate_hz,
                    all_arm_joints=args.all_arm_joints,
                    output=output,
                )
            )

        summary = summarize(samples)
        summary["kind"] = "summary"
        summary["ts"] = time.time()
        write_json_line(output, summary)
        print("\nSummary")
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"\nSaved log: {output}")
        return 0
    finally:
        robot_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

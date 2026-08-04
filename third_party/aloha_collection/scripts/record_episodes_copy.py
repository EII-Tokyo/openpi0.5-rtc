#!/usr/bin/env python3
"""ALOHA teleoperation recorder with continuous current-pose collection.

Accepted episodes transfer to a bounded background saver. Four camera streams
are encoded concurrently through FFmpeg, preferring NVENC when it is actually
usable. After a normal pedal stop, leader and follower arms hold their current
poses; a debounced dual-leader-gripper open-to-close gesture restores
teleoperation without an intermediate HOME move.

The existing ``s``/``m``/``r``/``Ctrl+C`` safe-stop and final sleep behavior is
preserved. ``--return-home-between-episodes`` restores the legacy per-episode
HOME workflow.
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dm_env
from aloha.async_save_worker import SaveJob, SaveWorker
from aloha.camera_runtime import CameraRuntime
from aloha.real_env import get_action, make_real_env
from aloha import robot_utils as _robot_utils
from aloha.keyboard_commands import RecorderKeyRouter
from aloha.command_coordinator import RecorderCommandCoordinator
from aloha.continuous_recorder import (
    SessionOutcome,
    run_continuous_session,
)
from aloha.episode_storage import EpisodeClaimCollision, StagedEpisode
from aloha.episode_image_spool import (
    EpisodeImageSpoolWriter,
    strip_and_spool_timestep,
)
from aloha.external_recovery import (
    ExternalRecoverySession,
    bounded_best_effort_log,
    supervise_external_recovery,
)
from aloha.episode_serialization import (
    EpisodeSavePayload,
    build_camera_map,
)
from aloha.current_pose_rearm import (
    hold_leader_arms_at_current_pose,
    wait_for_safe_current_pose_rearm,
)
from aloha.local_pedal import DEFAULT_PEDAL_PATH, FootPedalListener
from aloha.keyboard_listener import run_keyboard_listener
from aloha.episode_attempt import (
    AttemptArtifact,
    AttemptDecision,
    AttemptOutcome,
    EpisodeAttemptRunner,
    check_episode_index,
    cleanup_attempt_artifact,
    find_next_available_episode_index,
    guarded_teleop_step,
    join_motion_thread_safely,
    prepare_return_modes,
    request_diagnostic_stop,
    restore_teleop_modes,
    sample_registers_interruptibly,
    stop_diagnostic_worker,
    wait_for_diagnostic_interval,
)
from aloha.gripper_control import (
    configure_follower_gripper_mode,
)
from aloha.interbotix_service import (
    set_gravity_compensation_with_timeout,
    set_operating_modes_with_timeout,
    torque_enable_with_timeout,
)
from aloha.motor_diagnostics import (
    diagnostic_registers_for_robot,
    read_register_values_with_timeout,
)
from aloha.record_trigger import (
    RecordingEvents,
    RecordingTriggerController,
    TriggerResult,
)
from aloha.remote_trigger import DEFAULT_SOCKET_PATH, TriggerSocketServer
from aloha.recovery_lease import RecoveryLease
from aloha.robot_health import (
    RobotHealthMonitor,
    RobotHealthUnavailable,
    attach_joint_state_subscriptions,
)
from aloha.safe_motion import move_robots_guarded, plan_motion_duration
from aloha.safe_sleep import (
    SafeSleepReport,
)
from aloha.safe_sleep_runtime import initialize_ros_context
from aloha.safe_stop import (
    SafeStopController,
)
from aloha.safety_state import RecoveryIdentity, publish_safety_state

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import json
import numpy as np
import os
import random
import rclpy
from rclpy.signals import SignalHandlerOptions
import time
from typing import Dict, List
from tqdm import tqdm

# ★ 新增：信号、键盘、线程、子进程支持
import signal
import threading

disable_gravity_compensation = _robot_utils.disable_gravity_compensation
enable_gravity_compensation = _robot_utils.enable_gravity_compensation
FOLLOWER_GRIPPER_JOINT_CLOSE = _robot_utils.FOLLOWER_GRIPPER_JOINT_CLOSE
FOLLOWER_GRIPPER_JOINT_OPEN = _robot_utils.FOLLOWER_GRIPPER_JOINT_OPEN
FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN = _robot_utils.FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN
get_arm_gripper_positions = _robot_utils.get_arm_gripper_positions
get_arm_joint_positions = _robot_utils.get_arm_joint_positions
ImageRecorder = _robot_utils.ImageRecorder
LEADER_GRIPPER_CLOSE_THRESH = _robot_utils.LEADER_GRIPPER_CLOSE_THRESH
LEADER_GRIPPER_JOINT_CLOSE = _robot_utils.LEADER_GRIPPER_JOINT_CLOSE
LEADER_GRIPPER_JOINT_OPEN = _robot_utils.LEADER_GRIPPER_JOINT_OPEN
LEADER_GRIPPER_JOINT_NORMALIZE_FN = _robot_utils.LEADER_GRIPPER_JOINT_NORMALIZE_FN
LEADER_GRIPPER_JOINT_MID = _robot_utils.LEADER_GRIPPER_JOINT_MID
load_yaml_file = _robot_utils.load_yaml_file
move_arms = _robot_utils.move_arms
move_grippers = _robot_utils.move_grippers
torque_off = _robot_utils.torque_off
torque_on = _robot_utils.torque_on


def set_follower_arm_operating_mode(
    bot: InterbotixManipulatorXS,
    continuous_roll_joints: bool = False,
) -> None:
    func = getattr(_robot_utils, "set_follower_arm_operating_mode", None)
    if func is not None:
        try:
            func(
                bot,
                continuous_roll_joints=continuous_roll_joints,
            )
        except TypeError:
            func(bot)
            if continuous_roll_joints:
                for joint_name in ("forearm_roll", "wrist_rotate"):
                    bot.core.robot_torque_enable("single", joint_name, False)
                    bot.core.robot_set_operating_modes("single", joint_name, "ext_position")
                    try:
                        mode = bot.core.robot_get_motor_registers("single", joint_name, "Operating_Mode")
                        print(f"[follower-mode] {bot.core.robot_name}.{joint_name} Operating_Mode={mode} (ext_position)")
                    except Exception as e:
                        print(f"[follower-mode] {bot.core.robot_name}.{joint_name} verify failed: {e}")
        return
    bot.core.robot_set_operating_modes("group", "arm", "position")
    if continuous_roll_joints:
        for joint_name in ("forearm_roll", "wrist_rotate"):
            bot.core.robot_torque_enable("single", joint_name, False)
            bot.core.robot_set_operating_modes("single", joint_name, "ext_position")
            try:
                mode = bot.core.robot_get_motor_registers("single", joint_name, "Operating_Mode")
                print(f"[follower-mode] {bot.core.robot_name}.{joint_name} Operating_Mode={mode} (ext_position)")
            except Exception as e:
                print(f"[follower-mode] {bot.core.robot_name}.{joint_name} verify failed: {e}")


# ★ 退出事件
STOP_NO_SAVE_EVENT = threading.Event()   # 不保存
STOP_AND_SAVE_EVENT = threading.Event()  # 保存
SKIP_SLEEP_EVENT = threading.Event()     # 'r' 结果标记；最终仍执行 fail-closed sleep
RETURN_TO_START_EVENT = threading.Event()  # ★ 录制中按 b 键：回到初始位置并保存
RECORDING_STARTED_EVENT = threading.Event()  # ★ 第一次按 b 键：开始真正写入 episode 数据
DISCARD_AND_RETRY_EVENT = threading.Event()  # 放弃当前 attempt，回到初始位置后重新等待 b
PROGRAM_EXIT_EVENT = threading.Event()  # 主程序退出时才结束终端键盘监听
_SAFE_STOP_CONTROLLER = SafeStopController(
    STOP_NO_SAVE_EVENT,
    STOP_AND_SAVE_EVENT,
    SKIP_SLEEP_EVENT,
)

# ★ 记录 robot_base（为了 sleep.py 传参）
_GLOBAL_ROBOT_BASE = None

# ★ Ctrl+C 时的特殊保存路径
_CTRL_C_SAVE_DIR = None

# ★ 按 b 后始终回到初始位置；此开关只控制是否把回程过程写入 episode，默认写入
_SAVE_RETURN_TO_START_DATA_ON_B = True
_RETURN_HOME_BETWEEN_EPISODES = False
_VIDEO_ENCODER_BACKEND = "auto"
_REARM_MAX_JOINT_ERROR_RAD = 0.1
_REARM_DEBOUNCE_SAMPLES = 3
_OPENING_HOME_MIN_SECONDS = 1.0
_OPENING_MAX_JOINT_SPEED = 0.4
_RETURN_HOME_MIN_SECONDS = 1.0
_RETURN_HOME_MAX_JOINT_SPEED = 0.4
_RETURN_HOME_ARRIVAL_TOLERANCE_RAD = 0.10
_RETURN_HOME_STABLE_SAMPLES = 3
_JOINT_STATE_MOVING_TIMEOUT = 0.30
_JOINT_STATE_IDLE_TIMEOUT = 0.75
_HEALTH_WATCHDOG_RATE_HZ = 10.0
_HEALTH_GATE_TIMEOUT_SECONDS = 2.0
_HEALTH_REQUIRED_CONSECUTIVE = 3
_TELEOP_LEADER_MAX_AGE_SECONDS = 0.10
_SAFE_STATE_OWNER_GRACE_SECONDS = 2.0
SAVE_DRAIN_BEFORE_RECOVERY_TIMEOUT_SECONDS = 30.0
SAVE_ABORT_TIMEOUT_SECONDS = 5.0

# ★ 开始采集触发方式：
#   - gripper: 默认。opening_ceremony 中两只 leader 夹爪同时闭合后开始采集。
#   - b: 保持按 b 开始采集、再次按 b 结束并回到初始位置。
_START_RECORDING_TRIGGER = "gripper"
_TRIGGER_CONTROLLER = None
_COMMAND_COORDINATOR = None
_KEY_ROUTER = None
_COMMAND_LOCK = threading.RLock()
INTERBOTIX_RETURN_SERVICE_TIMEOUT_SEC = 2.0
ARM_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
# 左右臂均使用经确认的采集专用位姿。
LEFT_ACQUISITION_START_ARM_QPOS = [
    0.0,
    -0.96,
    1.16,
    1.57,
    0.0,
    -1.57,
]
RIGHT_ACQUISITION_START_ARM_QPOS = [
    0.0,
    -0.96,
    1.16,
    0.0,
    0.0,
    0.0,
]
DEFAULT_START_ARM_POSE = {
    "left_arm": list(LEFT_ACQUISITION_START_ARM_QPOS),
    "right_arm": list(RIGHT_ACQUISITION_START_ARM_QPOS),
}
RANDOM_START_POSITIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "sampled_start_positions_1000_structured.json"
_ENABLE_RANDOM_START_POSITIONS = False
_RANDOM_START_POSITION_CACHE = None


def _set_operating_modes_bounded(
    bot,
    cmd_type: str,
    name: str,
    mode: str,
    *,
    profile_type: str = "velocity",
    profile_velocity: int = 0,
    profile_acceleration: int = 0,
) -> None:
    set_operating_modes_with_timeout(
        bot,
        cmd_type,
        name,
        mode,
        timeout_sec=INTERBOTIX_RETURN_SERVICE_TIMEOUT_SEC,
        profile_type=profile_type,
        profile_velocity=profile_velocity,
        profile_acceleration=profile_acceleration,
    )


def _configure_follower_gripper_mode_bounded(
    follower_name: str,
    follower,
) -> None:
    configure_follower_gripper_mode(
        follower_name,
        follower,
        set_operating_modes=_set_operating_modes_bounded,
    )


def _torque_enable_bounded(
    bot,
    cmd_type: str,
    name: str,
    enable: bool,
) -> None:
    torque_enable_with_timeout(
        bot,
        cmd_type,
        name,
        enable,
        timeout_sec=INTERBOTIX_RETURN_SERVICE_TIMEOUT_SEC,
    )


def _torque_on_bounded(bot) -> None:
    _torque_enable_bounded(bot, "group", "arm", True)
    _torque_enable_bounded(bot, "single", "gripper", True)


def _torque_off_bounded(bot) -> None:
    _torque_enable_bounded(bot, "group", "arm", False)
    _torque_enable_bounded(bot, "single", "gripper", False)


def _set_gravity_compensation_bounded(bot, enabled: bool) -> None:
    set_gravity_compensation_with_timeout(
        bot,
        enabled,
        timeout_sec=INTERBOTIX_RETURN_SERVICE_TIMEOUT_SEC,
    )


def _enable_gravity_compensation_bounded(bot) -> None:
    _set_gravity_compensation_bounded(bot, True)


def _disable_gravity_compensation_bounded(bot) -> None:
    _set_gravity_compensation_bounded(bot, False)


def _set_follower_arm_mode_bounded(
    bot,
    continuous_roll_joints: bool = False,
) -> None:
    _set_operating_modes_bounded(bot, "group", "arm", "position")
    if continuous_roll_joints:
        for joint_name in ("forearm_roll", "wrist_rotate"):
            _torque_enable_bounded(bot, "single", joint_name, False)
            _set_operating_modes_bounded(
                bot,
                "single",
                joint_name,
                "ext_position",
            )


def _sampled_qpos_gripper_to_follower_joint(value: float) -> float:
    """Convert a normalized sampled qpos gripper value to a follower joint target."""
    normalized_value = min(1.0, max(0.0, float(value)))
    return float(FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN(normalized_value))
FOLLOWER_GRIPPER_CURRENT_LIMITS = {
    "follower_left": 300,
    "follower_right": 550,
}
_FOLLOWER_GRIPPER_CURRENT_LIMIT_OVERRIDES = {}
FOLLOWER_GRIPPER_SHUTDOWN = 20
MIN_FPS_THRESHOLD = 30
DYNAMIXEL_OPERATING_MODE_NAMES = {
    0: "current",
    1: "velocity",
    3: "position",
    4: "extended_position",
    5: "current_based_position",
    16: "pwm",
}
MOTOR_DIAG_JOINTS = ARM_JOINT_NAMES + ["gripper"]


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _robot_get_motor_registers_timeout(
    robot,
    cmd_type,
    name,
    reg,
    timeout_sec=0.02,
):
    return read_register_values_with_timeout(
        robot,
        cmd_type,
        name,
        reg,
        timeout_sec=timeout_sec,
    )


def _get_joint_entry(robot, joint_name: str) -> Dict[str, object]:
    msg = robot.core.joint_states
    names = list(msg.name)
    if joint_name not in names:
        return {"error": f"{joint_name} not in joint_states", "names": names}
    idx = names.index(joint_name)
    return {
        "index": idx,
        "position": msg.position[idx] if idx < len(msg.position) else None,
        "velocity": msg.velocity[idx] if idx < len(msg.velocity) else None,
        "effort": msg.effort[idx] if idx < len(msg.effort) else None,
    }


def _decode_register_values(reg: str, values):
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


def _read_control_modes(
    robot,
    stop_event: threading.Event = None,
) -> Dict[str, object]:
    def read_mode(joint):
        try:
            values = _robot_get_motor_registers_timeout(
                robot, "single", joint, "Operating_Mode", timeout_sec=0.02
            )
            return {
                "Operating_Mode": values,
                "Operating_Mode_decoded": _decode_register_values(
                    "Operating_Mode", values
                ),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    results, interrupted = sample_registers_interruptibly(
        list(MOTOR_DIAG_JOINTS),
        read_register=read_mode,
        stop_event=stop_event,
    )
    modes = {joint: value for joint, value in results}
    if interrupted:
        modes["_interrupted"] = True
    return modes


def _sample_motor6_diag(
    robot,
    stop_event: threading.Event = None,
) -> Dict[str, object]:
    sample = {
        "robot": robot.core.robot_name,
        "target_motor": {"id": 6, "joint": "wrist_rotate"},
        "all_motor_control_modes": {},
        "joint_states": {
            "wrist_rotate": None,
            "gripper": None,
        },
        "registers": {
            "wrist_rotate": {},
            "gripper": {},
        },
    }
    if stop_event is not None and stop_event.is_set():
        sample["interrupted"] = True
        return sample

    sample["all_motor_control_modes"] = _read_control_modes(
        robot,
        stop_event=stop_event,
    )
    if stop_event is not None and stop_event.is_set():
        sample["interrupted"] = True
        return sample

    sample["joint_states"] = {
        "wrist_rotate": _get_joint_entry(robot, "wrist_rotate"),
        "gripper": _get_joint_entry(robot, "gripper"),
    }

    registers = diagnostic_registers_for_robot(robot.core.robot_name)
    requests = [
        (joint, reg)
        for joint in ("wrist_rotate", "gripper")
        for reg in registers
    ]

    def read_register(request):
        joint, reg = request
        try:
            values = _robot_get_motor_registers_timeout(
                robot,
                "single",
                joint,
                reg,
                timeout_sec=0.02,
            )
            return {
                "value": values,
                "decoded": _decode_register_values(reg, values),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    register_results, interrupted = sample_registers_interruptibly(
        requests,
        read_register=read_register,
        stop_event=stop_event,
    )
    for (joint, reg), result in register_results:
        if "error" in result:
            sample["registers"][joint][reg] = {"error": result["error"]}
            continue
        sample["registers"][joint][reg] = result["value"]
        if result["decoded"] is not None:
            sample["registers"][joint][f"{reg}_decoded"] = result["decoded"]
    if interrupted:
        sample["interrupted"] = True
    return sample


def _motor6_diagnostics_worker(
    robots: Dict[str, InterbotixManipulatorXS],
    output_path: str,
    stop_event: threading.Event,
    rate_hz: float,
) -> None:
    robot_names = [
        name
        for name in ("leader_left", "leader_right", "follower_left", "follower_right")
        if name in robots
    ]
    interval = 1.0 / max(0.1, float(rate_hz))
    os.makedirs(os.path.dirname(output_path), exist_ok=True, mode=0o755)
    with open(output_path, "a", encoding="utf-8") as f:
        metadata = {
            "kind": "metadata",
            "ts": time.time(),
            "robots": robot_names,
            "target_motor": {"id": 6, "joint": "wrist_rotate"},
            "control_mode_joints": MOTOR_DIAG_JOINTS,
            "registers_by_robot": {
                name: diagnostic_registers_for_robot(name)
                for name in robot_names
            },
            "operating_mode_names": DYNAMIXEL_OPERATING_MODE_NAMES,
        }
        f.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
        for name in robot_names:
            if stop_event.is_set():
                break
            mode_snapshot = {
                "kind": "control_mode_snapshot",
                "ts": time.time(),
                "sample": _sample_motor6_diag(
                    robots[name],
                    stop_event=stop_event,
                ),
            }
            f.write(json.dumps(_jsonable(mode_snapshot), ensure_ascii=False, sort_keys=True) + "\n")
            if stop_event.is_set():
                break
        f.flush()

        while not stop_event.is_set():
            started = time.time()
            for name in robot_names:
                if stop_event.is_set():
                    break
                item = {
                    "kind": "sample",
                    "ts": time.time(),
                    "sample": _sample_motor6_diag(
                        robots[name],
                        stop_event=stop_event,
                    ),
                }
                f.write(json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True) + "\n")
                if stop_event.is_set():
                    break
            f.flush()
            wait_for_diagnostic_interval(
                stop_event,
                max(0.0, interval - (time.time() - started)),
            )


def _load_random_start_positions() -> List[Dict[str, object]]:
    """加载随机初始位姿；失败时回退到默认位姿。"""
    global _RANDOM_START_POSITION_CACHE

    if _RANDOM_START_POSITION_CACHE is not None:
        return _RANDOM_START_POSITION_CACHE

    fallback_pose = [{
        "left_arm": [float(v) for v in DEFAULT_START_ARM_POSE["left_arm"]],
        "right_arm": [float(v) for v in DEFAULT_START_ARM_POSE["right_arm"]],
        "left_gripper": float(LEADER_GRIPPER_JOINT_MID),
        "right_gripper": float(LEADER_GRIPPER_JOINT_MID),
    }]

    if not RANDOM_START_POSITIONS_PATH.exists():
        print(f"[start-pose] 随机初始位姿文件不存在，回退默认位姿: {RANDOM_START_POSITIONS_PATH}")
        _RANDOM_START_POSITION_CACHE = fallback_pose
        return _RANDOM_START_POSITION_CACHE

    try:
        payload = json.loads(RANDOM_START_POSITIONS_PATH.read_text())
        positions = payload.get("positions", [])
        loaded_positions: List[Dict[str, object]] = []
        for item in positions:
            left_arm = item.get("left_arm")
            right_arm = item.get("right_arm")
            left_gripper = item.get("left_gripper")
            right_gripper = item.get("right_gripper")
            if not isinstance(left_arm, list) or len(left_arm) != 6:
                continue
            if not isinstance(right_arm, list) or len(right_arm) != 6:
                continue
            if left_gripper is None or right_gripper is None:
                continue
            loaded_positions.append({
                "left_arm": [float(v) for v in left_arm],
                "right_arm": [float(v) for v in right_arm],
                "left_gripper": float(left_gripper),
                "right_gripper": float(right_gripper),
            })

        if not loaded_positions:
            raise ValueError("no valid sampled start positions")

        print(f"[start-pose] 已加载 {len(loaded_positions)} 个随机初始位姿: {RANDOM_START_POSITIONS_PATH}")
        _RANDOM_START_POSITION_CACHE = loaded_positions
        return _RANDOM_START_POSITION_CACHE
    except Exception as exc:
        print(f"[start-pose] 加载随机初始位姿失败，回退默认位姿: {exc}")
        _RANDOM_START_POSITION_CACHE = fallback_pose
        return _RANDOM_START_POSITION_CACHE


def _choose_episode_start_arm_pose() -> Dict[str, object]:
    """为当前 episode 选择一组固定的起始位姿。"""
    left_gripper = random.random()
    right_gripper = random.random()
    if not _ENABLE_RANDOM_START_POSITIONS:
        print(
            "[start-pose] 已禁用随机初始臂位姿，使用默认臂位姿；"
            f"随机夹爪: left={left_gripper:.3f}, right={right_gripper:.3f}"
        )
        return {
            "left_arm": [float(v) for v in DEFAULT_START_ARM_POSE["left_arm"]],
            "right_arm": [float(v) for v in DEFAULT_START_ARM_POSE["right_arm"]],
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
        }

    sampled_positions = _load_random_start_positions()
    selected_pose = dict(random.choice(sampled_positions))
    print(
        "[start-pose] 当前 episode 选择起始位姿: "
        f"left={selected_pose['left_arm']} gripper={left_gripper:.3f}, "
        f"right={selected_pose['right_arm']} gripper={right_gripper:.3f}"
    )
    return {
        "left_arm": [float(v) for v in selected_pose["left_arm"]],
        "right_arm": [float(v) for v in selected_pose["right_arm"]],
        "left_gripper": left_gripper,
        "right_gripper": right_gripper,
    }


def _signal_handler(sig, frame):
    """First Ctrl+C requests a safe stop; the second unwinds a blocked main thread."""
    global _CTRL_C_SAVE_DIR
    _CTRL_C_SAVE_DIR = None
    if sig == signal.SIGTERM:
        _SAFE_STOP_CONTROLLER.handle_sigterm()
    else:
        _SAFE_STOP_CONTROLLER.handle_sigint()


def _handle_pedal_failure(error: BaseException) -> None:
    coordinator = _COMMAND_COORDINATOR
    if coordinator is None:
        _SAFE_STOP_CONTROLLER.request_no_save(
            source="foot-pedal-failure",
        )
        return
    coordinator.request_no_save(source="foot-pedal-failure")


def _handle_b_trigger(source: str) -> TriggerResult:
    """Route every local or remote b command through one atomic controller."""
    coordinator = _COMMAND_COORDINATOR
    if coordinator is None:
        print(f"\n[{source}] 触发控制器尚未就绪，忽略 b。")
        return TriggerResult.IGNORED

    result = coordinator.handle_b()
    if result is TriggerResult.STARTED:
        if _RETURN_HOME_BETWEEN_EPISODES:
            print(
                "\n[b] 开始采集 episode 数据。再次按 b 将结束采集"
                "并回到初始位置..."
            )
        else:
            print(
                "\n[b] 开始采集 episode 数据。再次按 b 将结束采集"
                "并保持当前位置..."
            )
    elif result is TriggerResult.WRONG_START_MODE:
        print("\n[b] 当前为双夹爪闭合开始采集模式；采集开始后按 b 才会结束。")
    elif result is TriggerResult.NOT_READY:
        print(
            f"\n[{source}] 当前 episode 尚未准备完成，忽略本次 b；"
            "请等待准备完成提示后再踩脚踏。"
        )
    elif result is TriggerResult.NO_SAMPLES:
        print(
            f"\n[{source}] 尚未写入首个数据时间步，忽略本次停止；"
            "采集已继续。"
        )
    elif result is TriggerResult.STOPPED:
        if not _RETURN_HOME_BETWEEN_EPISODES:
            print(
                "\n[b] 停止采集并保存数据；leader/follower 将保持当前位置..."
            )
        elif _SAVE_RETURN_TO_START_DATA_ON_B:
            print("\n[b] 回到初始位置并保存数据（包含回到初始位置的过程）...")
        else:
            print("\n[b] 回到初始位置并保存数据（不保存回到初始位置的过程）...")
    return result


def _handle_d_trigger(source: str) -> TriggerResult:
    """Discard the active attempt and request a return to the retry position."""
    coordinator = _COMMAND_COORDINATOR
    if coordinator is None:
        print(f"\n[{source}] 触发控制器尚未就绪，忽略 d。")
        return TriggerResult.IGNORED

    result = coordinator.handle_d()
    if result is TriggerResult.DISCARD_STARTED:
        print("\n[d] 放弃当前 attempt，回到初始位置后重新等待 b...")
    elif result is TriggerResult.NOT_RECORDING:
        print("\n[d] 当前没有正在采集的 attempt，忽略。")
    elif result is TriggerResult.IGNORED:
        print("\n[d] 当前阶段不允许放弃 attempt，忽略。")
    return result


def _handle_remote_trigger(command: str) -> None:
    if command != "b":
        print(f"[remote-trigger] 忽略不支持的命令：{command!r}")
        return
    result = _handle_b_trigger("remote-trigger")
    print(f"[remote-trigger] b -> {result.value}")


def _handle_m_trigger() -> None:
    coordinator = _COMMAND_COORDINATOR
    if coordinator is not None and coordinator.request_save(
        skip_sleep=False,
        source="m",
    ):
        print("\n[m] 保存数据；完成后启动独立 safe-sleep 并退出...")


def _handle_s_trigger() -> None:
    coordinator = _COMMAND_COORDINATOR
    if coordinator is not None:
        coordinator.request_no_save_from_s()


def _handle_r_trigger() -> None:
    coordinator = _COMMAND_COORDINATOR
    if coordinator is not None and coordinator.request_save(
        skip_sleep=True,
        source="r",
    ):
        print(
            "\n[r] 保存数据并退出；按 fail-closed 策略随后启动"
            "独立 safe-sleep..."
        )


def _handle_ignored_retry_key(ch: str) -> None:
    print(f"\n[keyboard] 正在放弃当前 attempt 并回到初始位置，忽略 {ch!r}。")


def _handle_keyboard_key(ch: str) -> None:
    """Route one terminal key without owning the listener lifetime.

    - 'm' 保存并 sleep
    - 's' 不保存并 sleep，然后退出
    - 'r' 保存并退出；按 fail-closed 策略仍执行独立 safe-sleep
    - 'd' 放弃当前 attempt，回到初始位置后继续等待采集
    - in --start-trigger b mode, first 'b' starts recording
    - while recording, 'b' returns to start and saves; whether the return segment is saved is controlled by args
    """
    router = _KEY_ROUTER
    if router is None:
        print(f"\n[keyboard] 按键路由器尚未就绪，忽略 {ch!r}。")
        return
    router.handle(ch)


def _keyboard_listener():
    """Keep terminal hotkeys active until the recorder process exits."""
    run_keyboard_listener(PROGRAM_EXIT_EVENT, _handle_keyboard_key)


def _plan_synchronized_return_duration(
    leader_bots: Dict[str, object],
    selected_start_pose: Dict[str, object],
    *,
    minimum_seconds: float = _RETURN_HOME_MIN_SECONDS,
    max_joint_speed: float = _RETURN_HOME_MAX_JOINT_SPEED,
) -> float:
    required = {"leader_left", "leader_right"}
    missing = sorted(required - set(leader_bots))
    if missing:
        raise ValueError(
            "return HOME requires leader_left and leader_right; missing: "
            + ", ".join(missing)
        )
    return max(
        plan_motion_duration(
            get_arm_joint_positions(leader_bots[robot_name]),
            selected_start_pose[f"{robot_name.removeprefix('leader_')}_arm"][
                :6
            ],
            minimum_seconds=minimum_seconds,
            max_joint_speed=max_joint_speed,
        )
        for robot_name in sorted(required)
    )


def verify_acquisition_home_arrival(
    robots: Dict[str, object],
    selected_start_pose: Dict[str, object],
    *,
    read_positions=get_arm_joint_positions,
    tolerance: float = _RETURN_HOME_ARRIVAL_TOLERANCE_RAD,
) -> Dict[str, float]:
    required = (
        "leader_left",
        "leader_right",
        "follower_left",
        "follower_right",
    )
    missing = [name for name in required if name not in robots]
    if missing:
        raise ValueError(
            "acquisition HOME verification is missing robots: "
            + ", ".join(missing)
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("acquisition HOME tolerance must be positive")

    errors = {}
    for robot_name in required:
        suffix = robot_name.rsplit("_", 1)[1]
        target = tuple(
            float(value)
            for value in selected_start_pose[f"{suffix}_arm"][:6]
        )
        actual = tuple(
            float(value) for value in read_positions(robots[robot_name])
        )
        if len(target) != 6 or len(actual) != 6:
            raise ValueError(
                f"{robot_name} acquisition HOME requires six arm joints"
            )
        if not all(np.isfinite(value) for value in target + actual):
            raise ValueError(
                f"{robot_name} acquisition HOME positions must be finite"
            )
        max_error = max(
            abs(actual_value - target_value)
            for actual_value, target_value in zip(actual, target)
        )
        errors[robot_name] = max_error
        if max_error > tolerance:
            raise RuntimeError(
                f"{robot_name} acquisition HOME max error "
                f"{max_error:.3f} rad exceeds {tolerance:.3f} rad"
            )
    return errors


def lock_robots_at_acquisition_home(
    robots: Dict[str, object],
    selected_start_pose: Dict[str, object],
    *,
    set_modes=_set_operating_modes_bounded,
    torque_enable=_torque_enable_bounded,
    command_positions=None,
) -> None:
    required = (
        "leader_left",
        "leader_right",
        "follower_left",
        "follower_right",
    )
    missing = [name for name in required if name not in robots]
    if missing:
        raise ValueError(
            "acquisition HOME lock is missing robots: "
            + ", ".join(missing)
        )
    if command_positions is None:
        command_positions = lambda robot, target: (
            robot.arm.set_joint_positions(target, blocking=False)
        )

    for robot_name in required:
        suffix = robot_name.rsplit("_", 1)[1]
        target = tuple(
            float(value)
            for value in selected_start_pose[f"{suffix}_arm"][:6]
        )
        robot = robots[robot_name]
        set_modes(robot, "group", "arm", "position")
        torque_enable(robot, "group", "arm", True)
        accepted = command_positions(robot, target)
        if accepted is False:
            raise RuntimeError(
                f"{robot_name} rejected acquisition HOME lock command"
            )


def _return_to_start_position(
    env,
    robots: Dict[str, InterbotixManipulatorXS],
    dt: float,
    config: Dict,
    start_arm_pose: Dict[str, object] = None,
    continuous_roll_joints: bool = False,
    joint_unwrapper=None,
    timestep_transform=None,
    health: RobotHealthMonitor = None,
    stop_requested=lambda: False,
) -> tuple:
    """
    让机器人回到初始位置，并记录整个过程的数据。
    ★ 修改：只移动 leader 到初始位置，follower 通过 env.step 跟随 leader 运动。
    这样实现 leader 带领 follower 回到初始位置。

    返回:
        return_to_start_timesteps: 回到初始位置过程中的 timesteps 列表
        return_to_start_actions: 回到初始位置过程中的 actions 列表
    """
    leader_bots = {name: bot for name, bot in robots.items() if "leader" in name}
    follower_bots = {name: bot for name, bot in robots.items() if "follower" in name}

    pairs = []
    leader_suffixes = {name.split('_', 1)[1]: bot for name, bot in leader_bots.items()}
    follower_suffixes = {name.split('_', 1)[1]: bot for name, bot in follower_bots.items()}

    for suffix, leader_bot in leader_suffixes.items():
        if suffix in follower_suffixes:
            follower_bot = follower_suffixes.pop(suffix)
            pairs.append((suffix, leader_bot, follower_bot))

    # 准备回到初始位置。任何模式/扭矩服务超时都必须在运动线程启动前失败。
    print("[回到初始位置] 准备机器人...")
    prepare_return_modes(
        leader_bots,
        follower_bots,
        continuous_roll_joints=continuous_roll_joints,
        set_operating_modes=_set_operating_modes_bounded,
        set_follower_arm_mode=_set_follower_arm_mode_bounded,
        configure_follower_gripper=_configure_follower_gripper_mode_bounded,
        torque_on=_torque_on_bounded,
    )

    # 记录回到初始位置的数据
    return_to_start_timesteps = []
    return_to_start_actions = []
    return_to_start_dt_history = []

    # Retry and accepted-save returns both target this session's selected pose.
    selected_start_pose = start_arm_pose or {
        "left_arm": DEFAULT_START_ARM_POSE["left_arm"][:],
        "right_arm": DEFAULT_START_ARM_POSE["right_arm"][:],
    }
    print(
        "[回到初始位置] 目标臂位姿: "
        f"left={selected_start_pose['left_arm']}, right={selected_start_pose['right_arm']}"
    )

    print("[回到初始位置] 开始移动：leader 先移动，follower 跟随...")
    moving_time = _plan_synchronized_return_duration(
        leader_bots,
        selected_start_pose,
    )
    gripper_moving_time = 0.3  # 夹爪移动时间

    # ★ 在单独的线程中只移动 leader 到初始位置
    import threading

    move_complete = threading.Event()
    move_error = [None]
    random_gripper_normalized_by_suffix = {}

    def move_leader_thread():
        try:
            # ★ 只移动 leader 到初始位置
            leader_bot_list = []
            leader_target_poses = []

            for suffix, leader_bot, follower_bot in pairs:
                if suffix not in {"left", "right"}:
                    raise ValueError(f"Unsupported leader-follower suffix '{suffix}'. Expected 'left' or 'right'.")
                start_arm_qpos = selected_start_pose[f"{suffix}_arm"][:6]
                leader_bot_list.append(leader_bot)
                leader_target_poses.append(start_arm_qpos)

            print(f"[回到初始位置] 移动 {len(leader_bot_list)} 个 leader 到初始位置（{moving_time}秒）...")
            move_arms(
                bot_list=leader_bot_list,
                target_pose_list=leader_target_poses,
                moving_time=moving_time,
                dt=dt,
            )
            # ★ 手臂回到初始位置后，leader 夹爪先张开，再移动到完全随机状态；
            # follower 会在主线程 env.step 中跟随。复用本 session 选定的夹爪状态，
            # 使 discard 后的下一次 attempt 从完全相同的起始位姿开始。
            print(f"[回到初始位置] 张开 {len(leader_bots)} 个 leader 夹爪...")
            move_grippers(
                list(leader_bots.values()),
                [LEADER_GRIPPER_JOINT_OPEN] * len(leader_bots),
                moving_time=gripper_moving_time,
                dt=dt,
            )
            leader_random_bots = []
            leader_random_targets = []
            for name, leader_bot in leader_bots.items():
                suffix = name.split('_', 1)[1]
                normalized = float(selected_start_pose.get(f"{suffix}_gripper", 0.5))
                random_gripper_normalized_by_suffix[suffix] = normalized
                target = (
                    LEADER_GRIPPER_JOINT_CLOSE
                    + normalized * (LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE)
                )
                leader_random_bots.append(leader_bot)
                leader_random_targets.append(target)

            print(f"[回到初始位置] 移动到随机夹爪状态: {leader_random_targets}")
            move_grippers(
                leader_random_bots,
                leader_random_targets,
                moving_time=gripper_moving_time,
                dt=dt,
            )
            move_complete.set()
        except Exception as e:
            move_error[0] = e
            move_complete.set()

    # 启动移动线程
    move_thread = threading.Thread(target=move_leader_thread, daemon=False)
    move_thread.start()

    # ★ 在主循环中：读取 leader 位置，让 follower 跟随
    num_steps = int((moving_time + gripper_moving_time * 2) / dt)  # 总步数（包括张开和随机夹爪移动）
    pending_error = None
    try:
        for step in range(num_steps):
            if move_complete.is_set():
                break

            t0 = time.time()
            # ★ 获取当前 action（leader 的位置）
            action = get_action(
                robots,
                joint_unwrapper=joint_unwrapper,
                use_continuous_joints=continuous_roll_joints,
            )
            t1 = time.time()
            # ★ 关键：使用 env.step 让 follower 跟随 leader
            ts = env.step(action, get_obs=True)
            t2 = time.time()

            return_to_start_timesteps.append(
                timestep_transform(ts) if timestep_transform else ts
            )
            return_to_start_actions.append(action)
            return_to_start_dt_history.append([t0, t1, t2])

            time.sleep(max(0, dt - (time.time() - t0)))
    except BaseException as exc:
        pending_error = exc
    finally:
        try:
            join_motion_thread_safely(
                move_thread,
                nominal_timeout=moving_time + gripper_moving_time * 2 + 1.0,
                logger=print,
            )
        except BaseException as exc:
            if pending_error is None:
                pending_error = exc

    if pending_error is not None:
        raise pending_error
    if move_error[0]:
        raise RuntimeError(f"回到初始位置失败: {move_error[0]}")

    # ★ move_complete 置位后主循环可能立刻退出，补发几帧最终随机状态，确保 follower 收到随机夹爪命令。
    settle_steps = max(3, int(0.3 / dt))
    for _ in range(settle_steps):
        t0 = time.time()
        action = get_action(
            robots,
            joint_unwrapper=joint_unwrapper,
            use_continuous_joints=continuous_roll_joints,
        )
        t1 = time.time()
        ts = env.step(action, get_obs=True)
        t2 = time.time()
        return_to_start_timesteps.append(
            timestep_transform(ts) if timestep_transform else ts
        )
        return_to_start_actions.append(action)
        return_to_start_dt_history.append([t0, t1, t2])
        time.sleep(max(0, dt - (time.time() - t0)))

    # ★ 最后再直接设置 follower 到同一个随机夹爪状态，避免后续保持在未完全到位的位置。
    # 模式切换失败必须传播到上层 no-save 清理，不能宣告 retry 已就绪。
    follower_gripper_bots = []
    follower_gripper_targets = []
    for name, bot in follower_bots.items():
        _set_operating_modes_bounded(bot, "single", "gripper", "position")
        suffix = name.split('_', 1)[1]
        normalized = random_gripper_normalized_by_suffix.get(suffix, 0.5)
        follower_gripper_bots.append(bot)
        follower_gripper_targets.append(
            float(FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN(normalized))
        )
    move_grippers(
        follower_gripper_bots,
        follower_gripper_targets,
        moving_time=0.5,
        dt=dt,
    )
    print(
        "[回到初始位置] follower 夹爪已最终移动到随机状态: "
        f"{follower_gripper_targets}"
    )

    if health is None:
        raise ValueError("health monitor is required for acquisition HOME")
    robot_names = set(robots)
    _wait_for_health_gate(
        health,
        robot_names,
        phase="return_home_arrival",
        max_age=_JOINT_STATE_MOVING_TIMEOUT,
        stop_requested=stop_requested,
    )
    verify_acquisition_home_arrival(robots, selected_start_pose)
    lock_robots_at_acquisition_home(robots, selected_start_pose)

    for _ in range(_RETURN_HOME_STABLE_SAMPLES):
        _wait_for_health_gate(
            health,
            robot_names,
            phase="return_home_locked",
            max_age=_JOINT_STATE_MOVING_TIMEOUT,
            stop_requested=stop_requested,
        )
        verify_acquisition_home_arrival(robots, selected_start_pose)
        t0 = time.time()
        action = get_action(
            robots,
            joint_unwrapper=joint_unwrapper,
            use_continuous_joints=continuous_roll_joints,
        )
        t1 = time.time()
        ts = env.step(action, get_obs=True)
        t2 = time.time()
        return_to_start_timesteps.append(
            timestep_transform(ts) if timestep_transform else ts
        )
        return_to_start_actions.append(action)
        return_to_start_dt_history.append([t0, t1, t2])
        time.sleep(max(0, dt - (time.time() - t0)))

    print(f"[回到初始位置] 完成，共记录 {len(return_to_start_timesteps)} 个时间步")
    return (
        return_to_start_timesteps,
        return_to_start_actions,
        return_to_start_dt_history,
    )


def _wait_for_health_gate(
    health: RobotHealthMonitor,
    robot_names,
    *,
    phase: str,
    max_age: float,
    stop_requested,
) -> None:
    if health.fault_event.is_set():
        fault = health.first_fault
        raise RobotHealthUnavailable(
            "opening blocked by latched health fault: "
            f"{fault.robot_name}/{fault.phase}/{fault.reason}"
        )
    snapshots = health.wait_for_fresh(
        robot_names,
        consecutive=_HEALTH_REQUIRED_CONSECUTIVE,
        max_age=max_age,
        timeout=_HEALTH_GATE_TIMEOUT_SECONDS,
        stop_requested=stop_requested,
    )
    if health.fault_event.is_set():
        fault = health.first_fault
        raise RobotHealthUnavailable(
            "opening blocked by latched health fault: "
            f"{fault.robot_name}/{fault.phase}/{fault.reason}"
        )
    for robot_name in sorted(snapshots):
        snapshot = snapshots[robot_name]
        print(
            f"[health] {robot_name} phase={phase} "
            f"age={snapshot.message_age:.3f} "
            f"sequence={snapshot.sequence} PASS"
        )


def _configure_opening_pair_modes(
    *,
    suffix: str,
    leader_bot,
    follower_bot,
    continuous_roll_joints: bool,
) -> None:
    follower_name = f"follower_{suffix}"
    current_limits = dict(FOLLOWER_GRIPPER_CURRENT_LIMITS)
    current_limits.update(_FOLLOWER_GRIPPER_CURRENT_LIMIT_OVERRIDES)

    _torque_enable_bounded(follower_bot, "single", "gripper", False)
    follower_bot.core.robot_reboot_motors("single", "gripper", True)
    current_limit = current_limits.get(follower_name)
    if current_limit is not None:
        follower_bot.core.robot_set_motor_registers(
            "single",
            "gripper",
            "Current_Limit",
            current_limit,
        )
        follower_bot.core.robot_set_motor_registers(
            "single",
            "gripper",
            "Shutdown",
            FOLLOWER_GRIPPER_SHUTDOWN,
        )
        print(
            f"[opening_ceremony] {follower_name} "
            f"gripper Current_Limit={current_limit}"
        )

    _set_follower_arm_mode_bounded(
        follower_bot,
        continuous_roll_joints=continuous_roll_joints,
    )
    _configure_follower_gripper_mode_bounded(
        follower_name,
        follower_bot,
    )
    _set_operating_modes_bounded(
        leader_bot,
        "group",
        "arm",
        "position",
    )
    _set_operating_modes_bounded(
        leader_bot,
        "single",
        "gripper",
        "position",
    )


def _torque_on_opening_pair(leader_bot, follower_bot) -> None:
    _torque_on_bounded(follower_bot)
    _torque_on_bounded(leader_bot)


def _prepare_opening_pair(
    *,
    suffix: str,
    leader_bot,
    follower_bot,
    health: RobotHealthMonitor,
    dt: float,
    start_arm_qpos,
    leader_gripper_qpos: float,
    follower_gripper_qpos: float,
    continuous_roll_joints: bool,
    opening_home_min_seconds: float,
    opening_max_joint_speed: float,
    joint_state_idle_timeout: float,
    joint_state_moving_timeout: float,
    stop_requested,
) -> None:
    pair_names = {f"leader_{suffix}", f"follower_{suffix}"}
    _wait_for_health_gate(
        health,
        pair_names,
        phase=f"pre_mode:{suffix}",
        max_age=joint_state_idle_timeout,
        stop_requested=stop_requested,
    )
    _configure_opening_pair_modes(
        suffix=suffix,
        leader_bot=leader_bot,
        follower_bot=follower_bot,
        continuous_roll_joints=continuous_roll_joints,
    )
    _wait_for_health_gate(
        health,
        pair_names,
        phase=f"post_mode:{suffix}",
        max_age=joint_state_idle_timeout,
        stop_requested=stop_requested,
    )
    _torque_on_opening_pair(leader_bot, follower_bot)
    _wait_for_health_gate(
        health,
        pair_names,
        phase=f"post_torque:{suffix}",
        max_age=joint_state_moving_timeout,
        stop_requested=stop_requested,
    )

    current_by_name = {
        f"leader_{suffix}": get_arm_joint_positions(leader_bot),
        f"follower_{suffix}": get_arm_joint_positions(follower_bot),
    }
    duration = max(
        plan_motion_duration(
            current,
            start_arm_qpos,
            minimum_seconds=opening_home_min_seconds,
            max_joint_speed=opening_max_joint_speed,
        )
        for current in current_by_name.values()
    )
    max_delta = max(
        abs(float(target) - float(current))
        for current_positions in current_by_name.values()
        for current, target in zip(current_positions, start_arm_qpos)
    )
    print(
        f"[opening] {suffix} max_delta={max_delta:.3f} "
        f"planned_duration={duration:.3f}"
    )

    with health.arm_scope(
        pair_names,
        phase=f"opening_home:{suffix}",
        max_age=joint_state_moving_timeout,
        latch_global=True,
    ) as scope:
        move_robots_guarded(
            robots={
                f"leader_{suffix}": leader_bot,
                f"follower_{suffix}": follower_bot,
            },
            targets={
                f"leader_{suffix}": start_arm_qpos,
                f"follower_{suffix}": start_arm_qpos,
            },
            dt=dt,
            duration=duration,
            fault_event=scope.fault_event,
            health_check=scope.raise_if_faulted,
            sleep=time.sleep,
        )

    move_grippers(
        [leader_bot, follower_bot],
        [leader_gripper_qpos, follower_gripper_qpos],
        moving_time=0.5,
        dt=dt,
    )


def _prepare_opening_pairs(
    *,
    pairs,
    health: RobotHealthMonitor,
    dt: float,
    selected_start_pose,
    continuous_roll_joints: bool,
    opening_home_min_seconds: float,
    opening_max_joint_speed: float,
    joint_state_idle_timeout: float,
    joint_state_moving_timeout: float,
    stop_requested,
) -> None:
    prepared_pairs = []
    for suffix, leader_bot, follower_bot in pairs:
        if suffix not in {"left", "right"}:
            raise ValueError(
                f"Unsupported leader-follower suffix '{suffix}'. "
                "Expected 'left' or 'right'."
            )
        start_arm_qpos = selected_start_pose[f"{suffix}_arm"][:6]
        start_gripper_normalized = float(
            selected_start_pose[f"{suffix}_gripper"]
        )
        leader_gripper_qpos = (
            LEADER_GRIPPER_JOINT_CLOSE
            + start_gripper_normalized
            * (LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE)
        )
        follower_gripper_qpos = _sampled_qpos_gripper_to_follower_joint(
            start_gripper_normalized
        )
        prepared_pairs.append(
            (
                suffix,
                leader_bot,
                follower_bot,
                start_arm_qpos,
                start_gripper_normalized,
                leader_gripper_qpos,
                follower_gripper_qpos,
            )
        )

    failures = {}
    failures_lock = threading.Lock()

    def prepare_pair(prepared) -> None:
        (
            suffix,
            leader_bot,
            follower_bot,
            start_arm_qpos,
            start_gripper_normalized,
            leader_gripper_qpos,
            follower_gripper_qpos,
        ) = prepared
        try:
            _prepare_opening_pair(
                suffix=suffix,
                leader_bot=leader_bot,
                follower_bot=follower_bot,
                health=health,
                dt=dt,
                start_arm_qpos=start_arm_qpos,
                leader_gripper_qpos=leader_gripper_qpos,
                follower_gripper_qpos=follower_gripper_qpos,
                continuous_roll_joints=continuous_roll_joints,
                opening_home_min_seconds=opening_home_min_seconds,
                opening_max_joint_speed=opening_max_joint_speed,
                joint_state_idle_timeout=joint_state_idle_timeout,
                joint_state_moving_timeout=joint_state_moving_timeout,
                stop_requested=stop_requested,
            )
            print(
                f"[opening_ceremony] {suffix} 随机初始夹爪 "
                f"normalized={start_gripper_normalized:.3f}, "
                f"leader={leader_gripper_qpos:.5f}, "
                f"follower={follower_gripper_qpos:.5f}"
            )
        except BaseException as exc:
            with failures_lock:
                failures[suffix] = exc

    workers = [
        threading.Thread(
            target=prepare_pair,
            args=(prepared,),
            name=f"aloha-opening-{prepared[0]}",
            daemon=False,
        )
        for prepared in prepared_pairs
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    if failures:
        details = "; ".join(
            f"{suffix}: {type(exc).__name__}: {exc}"
            for suffix, exc in sorted(failures.items())
        )
        first_suffix = sorted(failures)[0]
        raise RuntimeError(
            f"opening pair failure(s): {details}"
        ) from failures[first_suffix]


def opening_ceremony(
    robots: Dict[str, InterbotixManipulatorXS],
    gravity_compensation: bool,
    dt: float,
    health: RobotHealthMonitor,
    start_arm_pose: Dict[str, object] = None,
    continuous_roll_joints: bool = False,
    opening_home_min_seconds: float = _OPENING_HOME_MIN_SECONDS,
    opening_max_joint_speed: float = _OPENING_MAX_JOINT_SPEED,
    joint_state_idle_timeout: float = _JOINT_STATE_IDLE_TIMEOUT,
    joint_state_moving_timeout: float = _JOINT_STATE_MOVING_TIMEOUT,
    stop_requested=lambda: False,
) -> None:
    """Move all leader-follower pairs of robots to a starting pose for demonstration."""
    leader_bots = {name: bot for name, bot in robots.items() if "leader" in name}
    follower_bots = {name: bot for name, bot in robots.items() if "follower" in name}

    pairs = []
    leader_suffixes = {name.split('_', 1)[1]: bot for name, bot in leader_bots.items()}
    follower_suffixes = {name.split('_', 1)[1]: bot for name, bot in follower_bots.items()}

    for suffix, leader_bot in leader_suffixes.items():
        if suffix in follower_suffixes:
            follower_bot = follower_suffixes.pop(suffix)
            pairs.append((suffix, leader_bot, follower_bot))
        else:
            raise ValueError(
                f"Unmatched leader suffix '{suffix}' found. Every leader should have a corresponding follower with the same suffix."
            )

    if follower_suffixes:
        unmatched_suffixes = ', '.join(follower_suffixes.keys())
        raise ValueError(
            f"Unmatched follower suffix(es) found: {unmatched_suffixes}. Every follower should have a corresponding leader with the same suffix."
        )

    if not pairs:
        raise ValueError("No valid leader-follower pairs found in the robot dictionary.")

    pair_order = {"left": 0, "right": 1}
    pairs.sort(key=lambda item: pair_order.get(item[0], 99))
    print(f"[opening_ceremony] 找到 {len(pairs)} 对 leader-follower")
    for i, (suffix, leader_bot, follower_bot) in enumerate(pairs):
        print(f"  Pair {i+1} ({suffix}): {leader_bot.core.robot_name} -> {follower_bot.core.robot_name}")

    selected_start_pose = start_arm_pose or {
        "left_arm": DEFAULT_START_ARM_POSE["left_arm"][:],
        "right_arm": DEFAULT_START_ARM_POSE["right_arm"][:],
        "left_gripper": LEADER_GRIPPER_JOINT_MID,
        "right_gripper": LEADER_GRIPPER_JOINT_MID,
    }
    try:
        _prepare_opening_pairs(
            pairs=pairs,
            health=health,
            dt=dt,
            selected_start_pose=selected_start_pose,
            continuous_roll_joints=continuous_roll_joints,
            opening_home_min_seconds=opening_home_min_seconds,
            opening_max_joint_speed=opening_max_joint_speed,
            joint_state_idle_timeout=joint_state_idle_timeout,
            joint_state_moving_timeout=joint_state_moving_timeout,
            stop_requested=stop_requested,
        )
    except Exception:
        fault = health.first_fault
        if fault is not None:
            print(
                f"[fault] {fault.robot_name} phase={fault.phase} "
                f"reason={fault.reason} age={fault.message_age:.3f}"
            )
        raise

    # 让 leader 的夹爪解除扭矩（演示阶段）
    for leader_bot in leader_bots.values():
        _torque_enable_bounded(
            leader_bot,
            "single",
            "gripper",
            False,
        )

    if _START_RECORDING_TRIGGER == "gripper":
        print("Close both leader grippers to enable following and start recording")
    else:
        print("Close both leader grippers to enable follower following, then press b to start recording")
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = all(
            get_arm_gripper_positions(leader_bot) < LEADER_GRIPPER_CLOSE_THRESH
            for leader_bot in leader_bots.values()
        )
        # ★ 期间也可以响应退出事件
        if STOP_NO_SAVE_EVENT.is_set() or STOP_AND_SAVE_EVENT.is_set():
            break
        time.sleep(dt / 10)

    restore_teleop_modes(
        robots,
        gravity_compensation=gravity_compensation,
        continuous_roll_joints=continuous_roll_joints,
        set_follower_arm_mode=_set_follower_arm_mode_bounded,
        set_operating_modes=_set_operating_modes_bounded,
        configure_follower_gripper=_configure_follower_gripper_mode_bounded,
        torque_enable=_torque_enable_bounded,
        torque_on=_torque_on_bounded,
        torque_off=_torque_off_bounded,
        enable_gravity_compensation=_enable_gravity_compensation_bounded,
    )

    print("Started!")


@dataclass
class RecorderRuntime:
    node: object
    env: object
    health: RobotHealthMonitor
    health_subscriptions: list[object]
    camera_runtime: object | None = None
    dt: float = 0.02
    home_initialized: bool = False
    camera_closed: bool = False
    health_stopped: bool = False
    subscriptions_cleared: bool = False
    ros_shutdown: bool = False
    quiesced: bool = False
    last_saved_episode_name: str | None = None
    terminal_save_source: str | None = None


class RecorderRuntimeShutdownError(RuntimeError):
    """Recorder ROS teardown failed after all stages were attempted."""


def hold_leaders_for_current_pose_rearm(
    robots: Dict[str, object],
    gravity_compensation: bool,
) -> None:
    """Hold leader arms at the pedal-stop pose while grippers remain free."""
    leader_bots = {
        name: robot
        for name, robot in robots.items()
        if name.startswith("leader_")
    }
    hold_leader_arms_at_current_pose(
        leader_bots,
        gravity_compensation=gravity_compensation,
        read_positions=get_arm_joint_positions,
        disable_gravity_compensation=(
            _disable_gravity_compensation_bounded
        ),
        set_position_mode=lambda robot: _set_operating_modes_bounded(
            robot,
            "group",
            "arm",
            "position",
        ),
        command_positions=lambda robot, positions: (
            robot.arm.set_joint_positions(
                positions,
                blocking=False,
            )
        ),
        torque_enable=_torque_enable_bounded,
    )


def prepare_current_pose_save_stop(
    attempt,
    *,
    robots: Dict[str, object],
    gravity_compensation: bool,
    force_no_save,
    leader_hold_policy: str = "best-effort",
    hold_leaders=hold_leaders_for_current_pose_rearm,
    stop_diagnostics=stop_diagnostic_worker,
    logger=print,
) -> bool:
    """Apply the configured Leader hold policy before saving."""
    valid_policies = {"strict", "best-effort", "off"}
    if leader_hold_policy not in valid_policies:
        raise ValueError(
            "leader_hold_policy must be one of: strict, best-effort, off"
        )

    if leader_hold_policy == "off":
        logger(
            "[current-pose hold] 已按 off 策略跳过 Leader 锁定；"
            "Leader 未被机械锁定，episode 仍将保存。"
        )
    else:
        try:
            hold_leaders(robots, gravity_compensation)
        except Exception as exc:
            if leader_hold_policy == "strict":
                logger(
                    "[current-pose hold] Leader 锁定失败，strict 策略将"
                    f"不保存并进入安全清理: {exc}"
                )
                force_no_save("leader stop-pose hold failed")
                return False
            logger(
                "[current-pose hold] 警告：Leader 锁定失败；Leader 未被"
                f"机械锁定，episode 仍将保存: {exc}"
            )

    stop_diagnostics(attempt)
    logger(
        "\n[b键] 已停止采集；follower 保持当前位置（最后命令位置），"
        "episode 已准备交给后台保存。"
    )
    return True


def _arm_positions_by_suffix(
    robots: Dict[str, object],
    *,
    role: str,
) -> dict[str, object]:
    prefix = f"{role}_"
    return {
        name[len(prefix) :]: get_arm_joint_positions(robot)
        for name, robot in robots.items()
        if name.startswith(prefix)
    }


def rearm_current_pose(
    *,
    robots: Dict[str, object],
    gravity_compensation: bool,
    continuous_roll_joints: bool,
    max_joint_error_rad: float = _REARM_MAX_JOINT_ERROR_RAD,
    debounce_samples: int = _REARM_DEBOUNCE_SAMPLES,
    abort_requested=lambda: False,
    health_check=lambda: None,
    post_restore_health_gate=lambda: None,
) -> bool:
    """Restore teleoperation only after a safe current-pose gesture."""
    leader_bots = {
        name: robot
        for name, robot in robots.items()
        if name.startswith("leader_")
    }

    return wait_for_safe_current_pose_rearm(
        read_grippers=lambda: {
            name.removeprefix("leader_"): get_arm_gripper_positions(robot)
            for name, robot in leader_bots.items()
        },
        read_leader_positions=lambda: _arm_positions_by_suffix(
            robots,
            role="leader",
        ),
        read_follower_positions=lambda: _arm_positions_by_suffix(
            robots,
            role="follower",
        ),
        restore_teleop=lambda: restore_teleop_modes(
            robots,
            gravity_compensation=gravity_compensation,
            continuous_roll_joints=continuous_roll_joints,
            set_follower_arm_mode=_set_follower_arm_mode_bounded,
            set_operating_modes=_set_operating_modes_bounded,
            configure_follower_gripper=(
                _configure_follower_gripper_mode_bounded
            ),
            torque_enable=_torque_enable_bounded,
            torque_on=_torque_on_bounded,
            torque_off=_torque_off_bounded,
            enable_gravity_compensation=(
                _enable_gravity_compensation_bounded
            ),
        ),
        stop_requested=lambda: (
            STOP_NO_SAVE_EVENT.is_set() or STOP_AND_SAVE_EVENT.is_set()
            or abort_requested()
        ),
        max_joint_error_rad=max_joint_error_rad,
        debounce_samples=debounce_samples,
        sleep=time.sleep,
        health_check=health_check,
        post_restore_health_gate=post_restore_health_gate,
    )


def prepare_episode_start(
    runtime: RecorderRuntime,
    *,
    robots: Dict[str, object],
    gravity_compensation: bool,
    dt: float,
    start_arm_pose: Dict[str, object],
    continuous_roll_joints: bool,
    return_home_between_episodes: bool,
    opening_home_min_seconds: float = _OPENING_HOME_MIN_SECONDS,
    opening_max_joint_speed: float = _OPENING_MAX_JOINT_SPEED,
    joint_state_idle_timeout: float = _JOINT_STATE_IDLE_TIMEOUT,
    joint_state_moving_timeout: float = _JOINT_STATE_MOVING_TIMEOUT,
    opening=opening_ceremony,
    rearm=rearm_current_pose,
) -> bool:
    """Use HOME once, then rearm at the held pose for later episodes."""
    if not runtime.home_initialized or return_home_between_episodes:
        opening(
            robots=robots,
            gravity_compensation=gravity_compensation,
            dt=dt,
            health=runtime.health,
            start_arm_pose=start_arm_pose,
            continuous_roll_joints=continuous_roll_joints,
            opening_home_min_seconds=opening_home_min_seconds,
            opening_max_joint_speed=opening_max_joint_speed,
            joint_state_idle_timeout=joint_state_idle_timeout,
            joint_state_moving_timeout=joint_state_moving_timeout,
            stop_requested=lambda: (
                STOP_NO_SAVE_EVENT.is_set()
                or STOP_AND_SAVE_EVENT.is_set()
            ),
        )
        runtime.home_initialized = True
        return not (
            STOP_NO_SAVE_EVENT.is_set() or STOP_AND_SAVE_EVENT.is_set()
        )
    _wait_for_health_gate(
        runtime.health,
        set(robots),
        phase="current_pose_rearm",
        max_age=_TELEOP_LEADER_MAX_AGE_SECONDS,
        stop_requested=lambda: (
            STOP_NO_SAVE_EVENT.is_set()
            or STOP_AND_SAVE_EVENT.is_set()
        ),
    )
    return rearm(
        robots=robots,
        gravity_compensation=gravity_compensation,
        continuous_roll_joints=continuous_roll_joints,
        post_restore_health_gate=lambda: _wait_for_health_gate(
            runtime.health,
            set(robots),
            phase="current_pose_rearm_post_restore",
            max_age=_TELEOP_LEADER_MAX_AGE_SECONDS,
            stop_requested=lambda: (
                STOP_NO_SAVE_EVENT.is_set()
                or STOP_AND_SAVE_EVENT.is_set()
            ),
        ),
    )


def _require_fresh_leaders(
    runtime: RecorderRuntime,
    *,
    phase: str,
) -> None:
    """Reject a control iteration before reading any stale leader cache."""
    leaders = {
        robot_name
        for robot_name in runtime.env.robots
        if robot_name.startswith("leader_")
    }
    runtime.health.require_fresh(
        leaders,
        max_age=_TELEOP_LEADER_MAX_AGE_SECONDS,
        phase=phase,
    )


def should_return_attempt_to_home(
    decision: AttemptDecision,
    *,
    return_home_between_episodes: bool,
) -> bool:
    """Discard/retry always returns HOME; accepted saves do so only on opt-in."""
    return (
        decision is AttemptDecision.DISCARD
        or (
            decision is AttemptDecision.SAVE
            and return_home_between_episodes
        )
    )


def handoff_episode_save(
    save_worker: SaveWorker,
    payload: EpisodeSavePayload,
    trigger_controller: RecordingTriggerController,
) -> bool:
    """Transfer episode ownership before releasing the next pedal cycle."""
    save_worker.submit(SaveJob(payload.dataset_name, payload))
    return trigger_controller.complete_save_handoff()


def create_recorder_runtime(
    *,
    config: Dict,
    torque_base: bool,
    continuous_roll_joints: bool,
    health_watchdog_rate_hz: float = _HEALTH_WATCHDOG_RATE_HZ,
) -> RecorderRuntime:
    """创建进程级 ROS/机器人运行时；失败时回滚已启动的全局资源。"""
    node = create_interbotix_global_node("aloha")
    health = None
    camera_runtime = None
    try:
        camera_runtime = CameraRuntime.create(
            config=config,
            context=node.context,
        )
        env = make_real_env(
            node=node,
            setup_robots=False,
            setup_base=config.get("base", False),
            torque_base=torque_base,
            config=config,
            continuous_roll_joints=continuous_roll_joints,
            image_recorder=camera_runtime.image_recorder,
        )
        robot_startup(node)
        health = RobotHealthMonitor(
            watchdog_rate_hz=health_watchdog_rate_hz,
        )
        health_subscriptions = attach_joint_state_subscriptions(
            node,
            health,
            {
                robot_name: _expected_joint_state_names(robot)
                for robot_name, robot in env.robots.items()
            },
        )
        health.start()
        try:
            publish_safety_state("RUNNING")
        except Exception as state_error:
            print(f"[safety-state] RUNNING 发布失败: {state_error}")
    except BaseException:
        if health is not None:
            try:
                health.stop()
            except BaseException as health_stop_error:
                print(
                    "[startup-rollback] 健康监控停止失败: "
                    f"{health_stop_error}"
                )
        if camera_runtime is not None:
            try:
                camera_runtime.close()
            except BaseException as camera_stop_error:
                print(
                    "[startup-rollback] 相机运行时停止失败: "
                    f"{camera_stop_error}"
                )
        try:
            robot_shutdown()
        except BaseException as shutdown_error:
            print(f"[startup-rollback] 关闭未完成运行时失败: {shutdown_error}")
        raise
    return RecorderRuntime(
        node=node,
        env=env,
        health=health,
        health_subscriptions=health_subscriptions,
        camera_runtime=camera_runtime,
        dt=1 / config.get("fps", 50),
    )


def _expected_joint_state_names(robot) -> set[str]:
    """Read expected ROS joint names from already initialized interfaces."""
    arm_names = getattr(
        getattr(robot.arm, "group_info", None),
        "joint_names",
        (),
    )
    gripper_names = getattr(
        getattr(robot.gripper, "gripper_info", None),
        "joint_names",
        (),
    )
    expected = set(arm_names) | set(gripper_names)
    if not expected:
        raise ValueError(
            f"{robot.core.robot_name} exposes no expected joint-state names"
        )
    return expected


def _quiesce_recorder_runtime(runtime: RecorderRuntime) -> None:
    """Permanently stop the recorder ROS runtime without moving a robot."""

    failures = []

    if not runtime.camera_closed:
        camera = runtime.camera_runtime
        if camera is None:
            runtime.camera_closed = True
        else:
            try:
                camera.close()
            except BaseException as exc:
                failures.append(("camera.close", exc))
            else:
                runtime.camera_runtime = None
                runtime.camera_closed = True

    if not runtime.health_stopped:
        try:
            runtime.health.stop()
        except BaseException as exc:
            failures.append(("health.stop", exc))
        else:
            runtime.health_stopped = True

    if not runtime.subscriptions_cleared:
        try:
            runtime.health_subscriptions.clear()
        except BaseException as exc:
            failures.append(("subscriptions.clear", exc))
        else:
            runtime.subscriptions_cleared = True

    if not runtime.ros_shutdown:
        try:
            robot_shutdown(runtime.node)
        except BaseException as exc:
            failures.append(("robot_shutdown", exc))
        else:
            runtime.ros_shutdown = True

    runtime.quiesced = all(
        (
            runtime.camera_closed,
            runtime.health_stopped,
            runtime.subscriptions_cleared,
            runtime.ros_shutdown,
        )
    )

    if failures:
        details = "; ".join(
            f"{stage}: {type(error).__name__}: {error}"
            for stage, error in failures
        )
        raise RecorderRuntimeShutdownError(details)


def finalize_recorder_runtime(
    runtime: RecorderRuntime,
    *,
    outcome: SessionOutcome,
    save_worker: SaveWorker,
    robot_name: str,
    gravity_compensation_active: bool = False,
    supervise_recovery=supervise_external_recovery,
    publish_state=publish_safety_state,
    lease_factory=RecoveryLease.acquire,
    clock=time.monotonic,
    logger=print,
    log_interval_seconds: float = 10.0,
    save_drain_timeout_seconds: float = (
        SAVE_DRAIN_BEFORE_RECOVERY_TIMEOUT_SECONDS
    ),
    save_abort_timeout_seconds: float = SAVE_ABORT_TIMEOUT_SECONDS,
) -> SafeSleepReport:
    """Quiesce the recorder and hand all recovery to standalone children."""

    _SAFE_STOP_CONTROLLER.begin_cleanup()
    allow_pose_deviation = (
        outcome is SessionOutcome.EXIT_DISCARD_AND_SLEEP
        and _SAFE_STOP_CONTROLLER.allow_pose_deviation
    )
    deferred_errors: list[BaseException] = []
    deferred_error_stages: set[str] = set()
    logging_enabled = True

    def defer_error(stage: str, error: BaseException) -> None:
        nonlocal logging_enabled
        if stage in deferred_error_stages:
            return
        deferred_error_stages.add(stage)
        deferred_errors.append(error)
        if not logging_enabled:
            return
        try:
            bounded_best_effort_log(
                logger,
                f"[recorder-finalizer] deferred {stage}: "
                f"{type(error).__name__}: {error}",
            )
        except BaseException as log_error:
            deferred_errors.append(log_error)
            logging_enabled = False

    def emit_stage(message: str) -> None:
        try:
            bounded_best_effort_log(logger, message)
        except BaseException as exc:
            defer_error("operator stage log", exc)

    save_drained = False
    try:
        save_worker.drain(timeout=save_drain_timeout_seconds)
        save_drained = True
    except BaseException as exc:
        defer_error("save drain", exc)
        try:
            save_worker.abort(timeout=save_abort_timeout_seconds)
        except BaseException as abort_exc:
            defer_error("save abort", abort_exc)
    if (
        save_drained
        and runtime.last_saved_episode_name is not None
        and runtime.terminal_save_source in {"m", "r"}
    ):
        emit_stage(
            f"[{runtime.terminal_save_source}] "
            f"{runtime.last_saved_episode_name} 已保存完成，"
            "开始独立 safe-sleep。",
        )
    try:
        _quiesce_recorder_runtime(runtime)
    except BaseException as exc:
        defer_error("recorder quiesce", exc)

    def _retry_guidance() -> str:
        guidance = getattr(
            _SAFE_STOP_CONTROLLER,
            "retry_guidance",
            None,
        )
        if callable(guidance):
            return guidance()
        return (
            "保持 UNSAFE_HOLD。请由运维在交互终端接管并执行"
            "独立恢复。"
        )

    def wait_for_explicit_retry(reason: object) -> None:
        nonlocal logging_enabled
        last_log = float("-inf")
        while True:
            try:
                now = clock()
            except BaseException as clock_error:
                defer_error("unsafe wait clock", clock_error)
                now = last_log
            if logging_enabled and now - last_log >= log_interval_seconds:
                try:
                    bounded_best_effort_log(
                        logger,
                        "[UNSAFE_HOLD] standalone recovery cannot continue: "
                        f"{reason}. {_retry_guidance()}",
                    )
                except BaseException as log_error:
                    deferred_errors.append(log_error)
                    logging_enabled = False
                last_log = now
            try:
                if _SAFE_STOP_CONTROLLER.wait_for_safety_retry(
                    timeout=1.0
                ):
                    return
            except BaseException as wait_error:
                defer_error("unsafe retry wait", wait_error)
                time.sleep(1.0)

    def publish_for_lease(lease, state):
        publish_state(
            state,
            recovery=RecoveryIdentity(
                recovery_id=lease.metadata.recovery_id,
                owner_pid=lease.metadata.owner_pid,
                source=lease.metadata.source,
            ),
            context_ok=False,
        )

    def prepare_attempt(recovery_id):
        lease = lease_factory(
            source="recorder",
            robot=robot_name,
            recovery_id=recovery_id,
        )
        try:
            publish_for_lease(
                lease,
                "EXTERNAL_RECOVERY_REQUIRED",
            )
        finally:
            lease.release()

    def wait_for_restart(recovery_id, error):
        lease = lease_factory(
            source="recorder",
            robot=robot_name,
            recovery_id=recovery_id,
        )
        try:
            publish_for_lease(lease, "UNSAFE_HOLD")
            last_log = float("-inf")
            while not _SAFE_STOP_CONTROLLER.wait_for_safety_retry(
                timeout=1.0
            ):
                now = clock()
                if now - last_log >= log_interval_seconds:
                    bounded_best_effort_log(
                        logger,
                        "[UNSAFE_HOLD] standalone recovery exited; "
                        f"recovery_id={recovery_id}; error={error}. "
                        f"{_retry_guidance()}",
                    )
                    last_log = now
        finally:
            lease.release()

    _SAFE_STOP_CONTROLLER.enter_unsafe_hold()
    # Motion ownership cannot transfer while the recorder ROS node may live.
    while not runtime.ros_shutdown:
        wait_for_explicit_retry("old recorder ROS runtime is still active")
        try:
            _quiesce_recorder_runtime(runtime)
        except BaseException as exc:
            defer_error("recorder quiesce retry", exc)

    emit_stage(
        "[handoff] recorder ROS runtime 已关闭。",
    )

    recovery_session = ExternalRecoverySession()
    try:
        report = supervise_recovery(
            robot_name=robot_name,
            gravity_compensation_active=gravity_compensation_active,
            allow_pose_deviation=allow_pose_deviation,
            sleep_script=Path(__file__).with_name("sleep.py"),
            retry_requested=_SAFE_STOP_CONTROLLER.wait_for_safety_retry,
            prepare_attempt=prepare_attempt,
            wait_for_restart=wait_for_restart,
            logger=logger,
            session=recovery_session,
        )
        if report is None or not report.safe_to_stop:
            raise RuntimeError(
                "external supervisor returned without a safe report"
            )
    except BaseException as exc:
        defer_error("external supervisor", exc)
        raise deferred_errors[0]

    for arm_name, result in sorted(report.results.items()):
        emit_stage(
            f"[safe-sleep] {arm_name}: status={result.status.value} "
            "torque_off_verified="
            f"{str(result.torque_off_verified).lower()}",
        )
    if allow_pose_deviation:
        emit_stage(
            "[SAFE_TO_STOP] s 退出：四臂扭矩关闭已验证；"
            "姿态仅作诊断并已记录。",
        )
    else:
        emit_stage(
            "[SAFE_TO_STOP] 四臂均已归位并验证扭矩关闭。",
        )

    try:
        _SAFE_STOP_CONTROLLER.leave_unsafe_hold()
    except BaseException as exc:
        defer_error("leave unsafe hold", exc)

    if deferred_errors:
        raise deferred_errors[0]
    return report


def capture_one_episode(
    runtime: RecorderRuntime,
    save_worker: SaveWorker,
    episode_idx: int,
    max_timesteps: int,
    dataset_dir: str,
    allow_existing: bool,
    torque_base: bool = False,
    gravity_compensation: bool = False,
    config: Dict = None,
    continuous_roll_joints: bool = False,
    motor6_diagnostics: bool = False,
    motor6_diagnostics_rate_hz: float = 0.5,
    return_home_between_episodes: bool = False,
    video_encoder_backend: str = "auto",
    leader_hold_policy: str = "best-effort",
    rearm_max_joint_error_rad: float = _REARM_MAX_JOINT_ERROR_RAD,
    rearm_debounce_samples: int = _REARM_DEBOUNCE_SAMPLES,
    opening_home_min_seconds: float = _OPENING_HOME_MIN_SECONDS,
    opening_max_joint_speed: float = _OPENING_MAX_JOINT_SPEED,
    joint_state_idle_timeout: float = _JOINT_STATE_IDLE_TIMEOUT,
    joint_state_moving_timeout: float = _JOINT_STATE_MOVING_TIMEOUT,
) -> SessionOutcome:
    """使用进程级运行时采集、验证并原子完成一条 episode。"""
    IS_MOBILE = config.get("base", False)
    DT = 1 / config.get("fps", 50)
    joint_unwrapper = (
        _robot_utils.JointPositionUnwrapper()
        if continuous_roll_joints and hasattr(_robot_utils, "JointPositionUnwrapper")
        else None
    )
    env = runtime.env
    staged = None
    save_worker.raise_if_failed()

    # ====== 生成保存用的相机名称（将 camera_ 改为 cam_） ======
    # YAML 中的 name 保持不变（用于 ROS namespace），但保存到 HDF5 时自动转换
    # 例如：camera_high -> cam_high, camera_wrist_right -> cam_wrist_right
    # def convert_camera_name_for_save(ros_name: str) -> str:
    #     """将 ROS namespace 名称转换为保存用的名称：camera_xxx -> cam_xxx"""
    #     if ros_name.startswith("camera_"):
    #         return "cam_" + ros_name[7:]  # 去掉 "camera_" 前缀（7个字符），加上 "cam_" 前缀
    #     # 如果已经是 cam_ 开头或其他格式，保持原样
    #     return ros_name

    # # 为每个相机生成保存用的名称
    # for camera in config.get("cameras", {}).get("camera_instances", []):
    #     ros_name = camera["name"]  # ROS namespace（YAML 中的原始名称，如 camera_high）
    #     save_name = convert_camera_name_for_save(ros_name)  # 保存到 HDF5 用的名称（如 cam_high）
    #     camera["yaml_name"] = save_name  # 保存转换后的名称
    #     if ros_name != save_name:
    #         print(f"[相机名称转换] {ros_name} (ROS namespace) -> {save_name} (保存到 HDF5)")
    #     else:
    #         print(f"[相机名称] {ros_name} (ROS namespace，无需转换)")

    # # ====== 打印相机配置信息 ======
    # camera_names_ros = [camera["name"] for camera in config.get("cameras", {}).get("camera_instances", [])]
    # camera_names_yaml = [camera.get("yaml_name", camera["name"]) for camera in config.get("cameras", {}).get("camera_instances", [])]
    # color_image_topic_name = config.get("cameras", {}).get("common_parameters", {}).get("color_image_topic_name", None)
    # print(f"[相机配置] 配置信息:")
    # print(f"  - ROS namespace（用于订阅，来自 YAML name）: {camera_names_ros}")
    # print(f"  - 保存名称（用于 HDF5，自动转换）: {camera_names_yaml}")
    # print(f"  - Topic 格式: {color_image_topic_name}")
    # if color_image_topic_name:
    #     print(f"  - 实际订阅的 topics:")
    #     for cam_name in camera_names_ros:
    #         topic = color_image_topic_name.format(cam_name)
    #         if not topic.startswith('/'):
    #             topic = '/' + topic
    #         print(f"    * {cam_name} -> {topic}")

    try:
        os.makedirs(dataset_dir, exist_ok=True, mode=0o755)
        requested_episode_idx = episode_idx
        while True:
            try:
                staged = StagedEpisode.create(dataset_dir, episode_idx)
                break
            except EpisodeClaimCollision:
                print(
                    f"[episode-claim] episode_{episode_idx} 已被其他采集进程占用，"
                    "选择下一个可用索引。"
                )
                episode_idx = find_next_available_episode_index(
                    dataset_dir,
                    start_index=episode_idx + 1,
                )
        if episode_idx != requested_episode_idx:
            allow_existing = False

        dataset_name = f"episode_{episode_idx}"
        episode_dir = str(staged.staging_path)
        print(f"[路径准备] episode: {dataset_name}")
        print(f"[路径准备] 独占 staging 目录: {episode_dir}")

        # ====== 开场流程 ======
        episode_start_arm_pose = _choose_episode_start_arm_pose()
        episode_ready = prepare_episode_start(
            runtime,
            robots=env.robots,
            gravity_compensation=gravity_compensation,
            dt=DT,
            start_arm_pose=episode_start_arm_pose,
            continuous_roll_joints=continuous_roll_joints,
            return_home_between_episodes=return_home_between_episodes,
            opening_home_min_seconds=opening_home_min_seconds,
            opening_max_joint_speed=opening_max_joint_speed,
            joint_state_idle_timeout=joint_state_idle_timeout,
            joint_state_moving_timeout=joint_state_moving_timeout,
            rearm=lambda **kwargs: rearm_current_pose(
                **kwargs,
                max_joint_error_rad=rearm_max_joint_error_rad,
                debounce_samples=rearm_debounce_samples,
                abort_requested=lambda: save_worker.failed,
                health_check=lambda: runtime.health.require_fresh(
                    set(runtime.env.robots),
                    max_age=_TELEOP_LEADER_MAX_AGE_SECONDS,
                    phase="current_pose_rearm_wait",
                ),
            ),
        )
        if not episode_ready:
            save_worker.raise_if_failed()
            return SessionOutcome.EXIT_DISCARD_AND_SLEEP

        # ====== 获取相机名称（使用 YAML 中的原始名称，不是 ROS namespace） ======
        # 注意：ImageRecorder 使用的是 ROS namespace，但我们需要使用 YAML 中的原始名称来保存数据
        # camera_names = []  # YAML 中的原始名称，用于保存到 HDF5
        # ros_to_yaml_name_map = {}  # ROS namespace -> YAML name 的映射
        # for camera in config.get("cameras", {}).get("camera_instances", []):
        #     # 优先使用 yaml_name（如果存在），否则使用 name
        #     yaml_name = camera.get("yaml_name", camera["name"])
        #     ros_name = camera["name"]  # 此时 name 已经是 ROS namespace（如果应用了映射）
        #     camera_names.append(yaml_name)
        #     ros_to_yaml_name_map[ros_name] = yaml_name

        # print(f"[相机名称映射] YAML 名称（用于保存到 HDF5）: {camera_names}")
        # print(f"[相机名称映射] ROS namespace -> YAML 名称映射: {ros_to_yaml_name_map}")

        # ====== 采集循环 ======
        ts = env.reset(fake=True)

        # ★ 验证所有相机是否就绪（在采集开始前）
        print("\n[相机验证] 检查所有相机是否就绪...")
        test_images = env.get_images()
        camera_config = config.get("cameras", {}).get("camera_instances", [])
        camera_names_yaml = [camera["name"] for camera in camera_config]

        missing_cameras = []
        none_cameras = []
        for cam_name in camera_names_yaml:
            if cam_name not in test_images:
                missing_cameras.append(cam_name)
            elif test_images[cam_name] is None:
                none_cameras.append(cam_name)

        if missing_cameras:
            print(f"[错误] 以下相机在 observation 中不存在: {missing_cameras}")
            raise RuntimeError(f"相机配置错误: {missing_cameras} 在 observation 中不存在")

        if none_cameras:
            print(f"[警告] 以下相机图像为 None（可能未启动或订阅失败）: {none_cameras}")
            print(f"[警告] 等待 2 秒后重试...")
            time.sleep(2.0)
            test_images_retry = env.get_images()
            still_none = [cam for cam in none_cameras if test_images_retry.get(cam) is None]
            if still_none:
                print(f"[错误] 以下相机仍然无法获取图像: {still_none}")
                print(f"[错误] 请检查相机是否已启动，ROS topic 是否正确")
                raise RuntimeError(f"相机未就绪: {still_none}")
            else:
                print(f"[相机验证] ✓ 所有相机已恢复")
        else:
            print(f"[相机验证] ✓ 所有相机就绪（共 {len(camera_names_yaml)} 个）")

        if _TRIGGER_CONTROLLER is None:
            raise RuntimeError("触发控制器未初始化，无法开放采集")
        preparation_completed = _TRIGGER_CONTROLLER.complete_preparation(
            auto_start=_START_RECORDING_TRIGGER == "gripper",
        )
        if not preparation_completed:
            save_worker.raise_if_failed()
            print("[采集] episode 准备完成前已收到退出请求，不开放采集。")
            return SessionOutcome.EXIT_DISCARD_AND_SLEEP

        if _START_RECORDING_TRIGGER == "gripper":
            print("[采集] 双 leader 夹爪已闭合，默认模式：立即开始采集 episode 数据。")
            print("[采集] 再按 b 将结束采集，并按参数决定是否记录回到初始位置的数据。")
        else:
            print("[采集] 准备完成。follower 已进入跟随状态，但尚未写入 episode 数据。")
            print("[采集] 请调整 leader 到采集起点，然后按 b 开始采集；再次按 b 结束采集。")

        accepted_attempt = {}
        active_attempt = {"value": None}

        def store_attempt_timestep(attempt, timestep):
            writer = attempt.resources.get("image_spool_writer")
            if writer is None:
                writer = EpisodeImageSpoolWriter(
                    staged.staging_path,
                    tuple(camera_names_yaml),
                )
                attempt.resources["image_spool_writer"] = writer
            attempt.timesteps.append(
                strip_and_spool_timestep(writer, timestep)
            )

        def wait_for_attempt_start(_attempt):
            while not RECORDING_STARTED_EVENT.is_set():
                save_worker.raise_if_failed()
                if STOP_NO_SAVE_EVENT.is_set() or STOP_AND_SAVE_EVENT.is_set():
                    return
                t0 = time.time()
                action, _, _ = guarded_teleop_step(
                    health_check=lambda: _require_fresh_leaders(
                        runtime,
                        phase="teleop_wait",
                    ),
                    read_action=lambda: get_action(
                        env.robots,
                        joint_unwrapper=joint_unwrapper,
                        use_continuous_joints=continuous_roll_joints,
                    ),
                    command=lambda value: env.step(
                        value,
                        get_obs=False,
                    ),
                    clock=time.time,
                )
                time.sleep(max(0, DT - (time.time() - t0)))

        def stop_attempt_diagnostics(attempt):
            stop_diagnostic_worker(attempt, timeout=2.0)

        def collect_attempt(attempt):
            active_attempt["value"] = attempt
            ts = env.get_observation()
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=env.get_reward(),
                discount=None,
                observation=ts,
            )
            store_attempt_timestep(attempt, ts)
            attempt.resources["start_time"] = time.time()

            if motor6_diagnostics:
                artifact = AttemptArtifact.create(dataset_dir, dataset_name)
                stop_event = threading.Event()
                diag_thread = threading.Thread(
                    target=_motor6_diagnostics_worker,
                    args=(
                        env.robots,
                        str(artifact.diagnostic_path),
                        stop_event,
                        motor6_diagnostics_rate_hz,
                    ),
                    daemon=True,
                )
                attempt.resources["artifact"] = artifact
                attempt.resources["diagnostic_stop_event"] = stop_event
                attempt.resources["diagnostic_thread"] = diag_thread
                diag_thread.start()
                print(
                    "[motor6诊断] 已开启 leader/follower 四臂 6号电机/夹爪诊断: "
                    f"{artifact.diagnostic_path} @ {motor6_diagnostics_rate_hz:.1f}Hz"
                )

            attempt.resources["max_timesteps_reached"] = False
            for _ in tqdm(range(max_timesteps)):
                save_worker.raise_if_failed()
                if STOP_NO_SAVE_EVENT.is_set():
                    break
                if RETURN_TO_START_EVENT.is_set() or (
                    STOP_AND_SAVE_EVENT.is_set()
                    and not RETURN_TO_START_EVENT.is_set()
                ):
                    break
                t0 = time.time()
                action, ts, t1 = guarded_teleop_step(
                    health_check=lambda: _require_fresh_leaders(
                        runtime,
                        phase="episode_collection",
                    ),
                    read_action=lambda: get_action(
                        env.robots,
                        joint_unwrapper=joint_unwrapper,
                        use_continuous_joints=continuous_roll_joints,
                    ),
                    command=env.step,
                    clock=time.time,
                )
                t2 = time.time()
                store_attempt_timestep(attempt, ts)
                attempt.actions.append(action)
                _TRIGGER_CONTROLLER.mark_sample_recorded()
                attempt.dt_history.append([t0, t1, t2])
                time.sleep(max(0, DT - (time.time() - t0)))
            else:
                attempt.resources["max_timesteps_reached"] = True

            if STOP_NO_SAVE_EVENT.is_set():
                return AttemptDecision.EXIT_NO_SAVE
            if DISCARD_AND_RETRY_EVENT.is_set():
                return AttemptDecision.DISCARD
            return AttemptDecision.SAVE

        def return_attempt_to_start(attempt, decision):
            if (
                RETURN_TO_START_EVENT.is_set()
                and not should_return_attempt_to_home(
                    decision,
                    return_home_between_episodes=(
                        return_home_between_episodes
                    ),
                )
            ):
                return prepare_current_pose_save_stop(
                    attempt,
                    robots=env.robots,
                    gravity_compensation=gravity_compensation,
                    leader_hold_policy=leader_hold_policy,
                    stop_diagnostics=stop_attempt_diagnostics,
                    force_no_save=_SAFE_STOP_CONTROLLER.force_no_save,
                )

            if not RETURN_TO_START_EVENT.is_set():
                stop_attempt_diagnostics(attempt)
                return True
            if not should_return_attempt_to_home(
                decision,
                return_home_between_episodes=return_home_between_episodes,
            ):
                stop_attempt_diagnostics(attempt)
                return True

            label = "d键" if decision is AttemptDecision.DISCARD else "b键"
            save_return_data = (
                decision is AttemptDecision.SAVE
                and _SAVE_RETURN_TO_START_DATA_ON_B
            )
            print(f"\n[{label}] 开始回到本轮选定的初始位置...")
            if decision is AttemptDecision.SAVE:
                print(f"[{label}] 当前已记录 {len(attempt.actions)} 个操纵数据时间步")
            request_diagnostic_stop(attempt)
            try:
                (
                    return_timesteps,
                    return_actions,
                    return_dt_history,
                ) = _return_to_start_position(
                    env,
                    env.robots,
                    DT,
                    config,
                    start_arm_pose=attempt.start_pose,
                    continuous_roll_joints=continuous_roll_joints,
                    joint_unwrapper=joint_unwrapper,
                    timestep_transform=(
                        (
                            lambda timestep: strip_and_spool_timestep(
                                attempt.resources[
                                    "image_spool_writer"
                                ],
                                timestep,
                            )
                        )
                        if save_return_data
                        else None
                    ),
                    health=runtime.health,
                    stop_requested=lambda: (
                        STOP_NO_SAVE_EVENT.is_set()
                        or STOP_AND_SAVE_EVENT.is_set()
                    ),
                )
                stop_attempt_diagnostics(attempt)
            except Exception as exc:
                print(f"[{label}] 回到初始位置失败，将不保存并进入安全清理: {exc}")
                _SAFE_STOP_CONTROLLER.force_no_save(
                    f"{label} return-to-start failed"
                )
                return False

            if decision is AttemptDecision.DISCARD:
                try:
                    restore_teleop_modes(
                        env.robots,
                        gravity_compensation=gravity_compensation,
                        continuous_roll_joints=continuous_roll_joints,
                        set_follower_arm_mode=_set_follower_arm_mode_bounded,
                        set_operating_modes=_set_operating_modes_bounded,
                        configure_follower_gripper=(
                            _configure_follower_gripper_mode_bounded
                        ),
                        torque_enable=_torque_enable_bounded,
                        torque_on=_torque_on_bounded,
                        torque_off=_torque_off_bounded,
                        enable_gravity_compensation=_enable_gravity_compensation_bounded,
                    )
                except Exception as exc:
                    print(f"[{label}] 恢复 teleop 模式失败，将安全退出: {exc}")
                    _SAFE_STOP_CONTROLLER.force_no_save(
                        "retry teleop mode restoration failed"
                    )
                    return False

            if save_return_data:
                collected_action_count = len(attempt.actions)
                attempt.timesteps.extend(return_timesteps)
                attempt.actions.extend(return_actions)
                attempt.dt_history.extend(return_dt_history)
                print(f"[{label}] 回到初始位置完成，额外记录了 {len(return_timesteps)} 个时间步")
                print(
                    f"[{label}] 总计数据: {len(attempt.actions)} 个时间步"
                    f"（{collected_action_count} 个操纵 + {len(return_timesteps)} 个回到初始位置）"
                )
            elif decision is AttemptDecision.SAVE:
                print(f"[{label}] 回到初始位置完成，丢弃回程 {len(return_timesteps)} 个时间步")
            else:
                print(f"[{label}] 已放弃当前 attempt；回位数据未进入下一次 attempt。")
            return True

        def discard_attempt(attempt):
            cleaned = cleanup_attempt_artifact(
                attempt,
                stop_diagnostics=stop_attempt_diagnostics,
                force_no_save=_SAFE_STOP_CONTROLLER.force_no_save,
                logger=print,
            )
            spool_writer = attempt.resources.get("image_spool_writer")
            if spool_writer is not None:
                spool_writer.discard()
            attempt.timesteps.clear()
            attempt.actions.clear()
            attempt.dt_history.clear()
            return cleaned

        def commit_attempt(attempt):
            if not attempt.resources.get("diagnostics_stopped"):
                stop_attempt_diagnostics(attempt)
            accepted_attempt["value"] = attempt

        def reset_after_retry():
            env.reset(fake=True)
            print("[d] 当前 attempt 已放弃，已回到同一初始位置；等待下一次 b 开始采集。")

        def complete_retry(*, auto_start):
            if _TRIGGER_CONTROLLER is None:
                raise RuntimeError("触发控制器未初始化，无法完成 retry")
            return _TRIGGER_CONTROLLER.complete_retry(auto_start=auto_start)

        runner = EpisodeAttemptRunner(
            dataset_name=dataset_name,
            start_pose=episode_start_arm_pose,
            wait_for_start=wait_for_attempt_start,
            collect=collect_attempt,
            return_to_start=return_attempt_to_start,
            discard_attempt=discard_attempt,
            commit_attempt=commit_attempt,
            complete_retry=complete_retry,
            reset_timestep=reset_after_retry,
            is_exit_requested=STOP_NO_SAVE_EVENT.is_set,
            auto_start_after_retry=_START_RECORDING_TRIGGER == "gripper",
        )
        attempt_outcome = runner.run()

        if attempt_outcome is AttemptOutcome.EXIT_NO_SAVE:
            print("\n[退出] 不保存数据。")
            return SessionOutcome.EXIT_DISCARD_AND_SLEEP

        attempt = accepted_attempt["value"]
        timesteps = attempt.timesteps
        actions = attempt.actions
        actual_dt_history = attempt.dt_history
        start_time = attempt.resources["start_time"]


        print(f"Avg fps: { (len(actions)) / max(1e-6, (time.time() - start_time)) }")
        # ====== 采集停止后的安全处理（关键修复：确保 leader 能执行 sleep） ======
        leaders = (
            {
                name: bot
                for name, bot in env.robots.items()
                if "leader" in name
            }
            if return_home_between_episodes
            else {}
        )
        print("[pre-sleep] leaders detected:", list(leaders.keys()) or "<NONE>")
        for name, robot in leaders.items():
            try:
                if gravity_compensation:
                    _disable_gravity_compensation_bounded(robot)
                    time.sleep(0.1)
                _set_operating_modes_bounded(
                    robot,
                    "group",
                    "arm",
                    "position",
                )
                _set_operating_modes_bounded(
                    robot,
                    "single",
                    "gripper",
                    "position",
                )
                _torque_on_bounded(robot)
            except Exception as e:
                print(f"[leader:{name}] pre-sleep prepare error: {e}")

        # follower：普通停止时打开；按 b 回到初始位置后保持最终随机夹爪状态。
        follower_bots = {name: bot for name, bot in env.robots.items() if "follower" in name}
        if (
            RETURN_TO_START_EVENT.is_set()
            or not return_home_between_episodes
        ):
            print("[follower] follower 保持本轮结束位置。")
        else:
            for name, bot in follower_bots.items():
                try:
                    _set_operating_modes_bounded(
                        bot,
                        "single",
                        "gripper",
                        "position",
                    )
                except Exception as e:
                    print(f"[follower:{name}] set gripper->position failed: {e}")
            try:
                move_grippers(
                    list(follower_bots.values()),
                    [FOLLOWER_GRIPPER_JOINT_OPEN] * len(follower_bots),
                    moving_time=0.5,
                    dt=DT,
                )
            except Exception as e:
                print(f"[follower] move_grippers(open) failed: {e}")

        # ====== 保存策略 ======
        # Ctrl+C / 's'：STOP_NO_SAVE_EVENT；两者都执行 sleep
        if STOP_NO_SAVE_EVENT.is_set():
            print("\n[退出] 不保存数据。")
            return SessionOutcome.EXIT_DISCARD_AND_SLEEP

        freq_mean = print_dt_diagnosis(actual_dt_history) if actual_dt_history else 0.0
        if not STOP_AND_SAVE_EVENT.is_set() and freq_mean < MIN_FPS_THRESHOLD and len(actions) >= 10:
            print(f"\n\nfreq_mean is {freq_mean}, lower than 30; discarding and exiting safely.\n\n")
            return SessionOutcome.EXIT_FAILURE_AND_SLEEP

        if not actions:
            print(
                "[采集] 本轮没有写入 action；丢弃 staging，"
                "不提交后台保存，并进入安全清理。"
            )
            return SessionOutcome.EXIT_DISCARD_AND_SLEEP

        camera_names = [
            camera["name"]
            for camera in config.get("cameras", {}).get(
                "camera_instances",
                [],
            )
        ]
        camera_map = build_camera_map(camera_names)
        spool_writer = attempt.resources.get("image_spool_writer")
        if spool_writer is None:
            raise RuntimeError("accepted attempt has no image spool")
        image_spool = spool_writer.seal(
            selected_frame_count=len(actions)
        )
        payload = EpisodeSavePayload(
            staged=staged,
            dataset_name=dataset_name,
            timesteps=tuple(timesteps[: len(actions)]),
            actions=tuple(actions),
            camera_map=camera_map,
            video_fps=float(config.get("fps", 50)),
            total_joint_size=7 * len(follower_bots),
            is_mobile=IS_MOBILE,
            continuous_roll_joints=continuous_roll_joints,
            allow_existing=allow_existing,
            video_backend=video_encoder_backend,
            artifact=attempt.resources.get("artifact"),
            image_spool=image_spool,
        )
        max_timesteps_reached = bool(
            attempt.resources.get("max_timesteps_reached")
        )
        terminal_save = (
            STOP_AND_SAVE_EVENT.is_set() or max_timesteps_reached
        )
        if terminal_save:
            save_worker.submit(SaveJob(payload.dataset_name, payload))
            handoff_completed = True
        else:
            if _TRIGGER_CONTROLLER is None:
                raise RuntimeError("触发控制器未初始化，无法移交保存")
            handoff_completed = handoff_episode_save(
                save_worker,
                payload,
                _TRIGGER_CONTROLLER,
            )

        # Ownership now belongs to the worker; this frame must not discard it.
        active_attempt["value"] = None
        staged = None
        attempt.timesteps.clear()
        attempt.actions.clear()
        attempt.dt_history.clear()
        if not handoff_completed:
            _SAFE_STOP_CONTROLLER.force_no_save(
                "save handoff trigger transition failed"
            )
            return SessionOutcome.EXIT_FAILURE_AND_SLEEP

        if terminal_save:
            runtime.last_saved_episode_name = dataset_name
            if STOP_AND_SAVE_EVENT.is_set():
                runtime.terminal_save_source = (
                    "r" if SKIP_SLEEP_EVENT.is_set() else "m"
                )
            else:
                runtime.terminal_save_source = "max-timesteps"

        print(
            f"[保存] {dataset_name} 已移交后台 worker；"
            "当前位置可用于下一轮重新唤醒。"
        )
        if STOP_AND_SAVE_EVENT.is_set():
            return (
                SessionOutcome.EXIT_SAVE_WITHOUT_SLEEP
                if SKIP_SLEEP_EVENT.is_set()
                else SessionOutcome.EXIT_SAVE_AND_SLEEP
            )
        if max_timesteps_reached:
            return SessionOutcome.EXIT_SAVE_AND_SLEEP
        return SessionOutcome.CONTINUE_NEXT_EPISODE

    finally:
        try:
            # Any not-yet-committed artifact must be stopped before unlinking.
            attempt_state = locals().get("active_attempt")
            if attempt_state:
                unfinished_attempt = attempt_state.get("value")
                if unfinished_attempt is not None:
                    discard_attempt(unfinished_attempt)
        except BaseException as cleanup_error:
            unfinished_path = "<no diagnostic artifact>"
            try:
                unfinished_artifact = unfinished_attempt.resources.get("artifact")
                if unfinished_artifact is not None:
                    unfinished_path = str(unfinished_artifact.diagnostic_path)
            except BaseException:
                pass
            print(
                f"[attempt-cleanup] {unfinished_path}: "
                f"unexpected cleanup failure: {cleanup_error}"
            )
            _SAFE_STOP_CONTROLLER.force_no_save("attempt cleanup failed")
        finally:
            if staged is not None:
                try:
                    staged.discard()
                except BaseException as staging_cleanup_error:
                    print(
                        "[episode-cleanup] staging 清理失败，将强制不保存并退出: "
                        f"{staging_cleanup_error}"
                    )
                    _SAFE_STOP_CONTROLLER.force_no_save(
                        "episode staging cleanup failed"
                    )


def get_auto_index(dataset_dir: str, dataset_name_prefix: str = "", data_suffix: str = "hdf5") -> int:
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        episode_name = f"{dataset_name_prefix}episode_{i}"
        episode_dir = os.path.join(dataset_dir, episode_name)
        legacy_episode_file = os.path.join(dataset_dir, f"{episode_name}.{data_suffix}")
        if not os.path.isdir(episode_dir) and not os.path.isfile(legacy_episode_file):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes.")


def print_dt_diagnosis(actual_dt_history: List[List[float]]) -> float:
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]
    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean if dt_mean > 0 else 0.0
    print(f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}")
    return freq_mean


def debug() -> None:
    print("====== Debug mode ======")
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        image_recorder.print_diagnostics()


def main(args: Dict[str, any]) -> None:
    """Main 入口：支持 Ctrl+C、m、s、r 的优雅退出。"""
    global _GLOBAL_ROBOT_BASE
    global _ENABLE_RANDOM_START_POSITIONS
    global _SAVE_RETURN_TO_START_DATA_ON_B
    global _RETURN_HOME_BETWEEN_EPISODES
    global _VIDEO_ENCODER_BACKEND
    global _REARM_MAX_JOINT_ERROR_RAD
    global _REARM_DEBOUNCE_SAMPLES
    global _START_RECORDING_TRIGGER
    global _FOLLOWER_GRIPPER_CURRENT_LIMIT_OVERRIDES
    global _TRIGGER_CONTROLLER
    global _COMMAND_COORDINATOR
    global _KEY_ROUTER
    global _SAFE_STOP_CONTROLLER
    torque_base = args.get("enable_base_torque", False)
    gravity_compensation = args.get("gravity_compensation", False)
    continuous_roll_joints = bool(args.get("continuous_roll_joints", True))
    robot_base = args.get("robot", "")
    _GLOBAL_ROBOT_BASE = robot_base
    _ENABLE_RANDOM_START_POSITIONS = bool(args.get("enable_random_start_positions", False))
    _SAVE_RETURN_TO_START_DATA_ON_B = bool(args.get("save_return_to_start_data_on_b", True))
    _RETURN_HOME_BETWEEN_EPISODES = bool(
        args.get("return_home_between_episodes", False)
    )
    _VIDEO_ENCODER_BACKEND = args.get("video_encoder", "auto")
    leader_hold_policy = args.get("leader_hold_policy", "best-effort")
    if leader_hold_policy not in {"strict", "best-effort", "off"}:
        raise ValueError(
            "leader_hold_policy must be one of: strict, best-effort, off"
        )
    pedal_debounce_seconds = float(
        args.get("pedal_debounce_seconds", 1.0)
    )
    if pedal_debounce_seconds < 0:
        raise ValueError(
            "pedal_debounce_seconds must be non-negative"
        )
    opening_home_min_seconds = float(
        args.get(
            "opening_home_min_seconds",
            _OPENING_HOME_MIN_SECONDS,
        )
    )
    opening_max_joint_speed = float(
        args.get(
            "opening_max_joint_speed",
            _OPENING_MAX_JOINT_SPEED,
        )
    )
    joint_state_moving_timeout = float(
        args.get(
            "joint_state_moving_timeout",
            _JOINT_STATE_MOVING_TIMEOUT,
        )
    )
    joint_state_idle_timeout = float(
        args.get(
            "joint_state_idle_timeout",
            _JOINT_STATE_IDLE_TIMEOUT,
        )
    )
    health_watchdog_rate_hz = float(
        args.get(
            "health_watchdog_rate_hz",
            _HEALTH_WATCHDOG_RATE_HZ,
        )
    )
    if (
        not np.isfinite(opening_home_min_seconds)
        or opening_home_min_seconds < 1.0
    ):
        raise ValueError(
            "opening_home_min_seconds must be at least 1.0"
        )
    if (
        not np.isfinite(opening_max_joint_speed)
        or opening_max_joint_speed <= 0
    ):
        raise ValueError("opening_max_joint_speed must be positive")
    if (
        not np.isfinite(joint_state_moving_timeout)
        or joint_state_moving_timeout <= 0
    ):
        raise ValueError("joint_state_moving_timeout must be positive")
    if (
        not np.isfinite(joint_state_idle_timeout)
        or joint_state_idle_timeout <= 0
    ):
        raise ValueError("joint_state_idle_timeout must be positive")
    if (
        not np.isfinite(health_watchdog_rate_hz)
        or health_watchdog_rate_hz <= 0
    ):
        raise ValueError("health_watchdog_rate_hz must be positive")
    _REARM_MAX_JOINT_ERROR_RAD = float(
        args.get("rearm_max_joint_error_rad", 0.1)
    )
    _REARM_DEBOUNCE_SAMPLES = int(
        args.get("rearm_debounce_samples", 3)
    )
    if _REARM_MAX_JOINT_ERROR_RAD <= 0:
        raise ValueError("rearm_max_joint_error_rad must be positive")
    if _REARM_DEBOUNCE_SAMPLES < 1:
        raise ValueError("rearm_debounce_samples must be at least one")
    _START_RECORDING_TRIGGER = args.get("start_trigger", "gripper")
    _FOLLOWER_GRIPPER_CURRENT_LIMIT_OVERRIDES = {
        "follower_left": int(args.get("left_gripper_current_limit", FOLLOWER_GRIPPER_CURRENT_LIMITS["follower_left"])),
        "follower_right": int(args.get("right_gripper_current_limit", FOLLOWER_GRIPPER_CURRENT_LIMITS["follower_right"])),
    }
    RECORDING_STARTED_EVENT.clear()
    RETURN_TO_START_EVENT.clear()
    DISCARD_AND_RETRY_EVENT.clear()
    STOP_NO_SAVE_EVENT.clear()
    STOP_AND_SAVE_EVENT.clear()
    SKIP_SLEEP_EVENT.clear()
    PROGRAM_EXIT_EVENT.clear()

    _SAFE_STOP_CONTROLLER = SafeStopController(
        STOP_NO_SAVE_EVENT,
        STOP_AND_SAVE_EVENT,
        SKIP_SLEEP_EVENT,
        lock=_COMMAND_LOCK,
    )
    _TRIGGER_CONTROLLER = RecordingTriggerController(
        RecordingEvents(
            recording_started=RECORDING_STARTED_EVENT,
            return_to_start=RETURN_TO_START_EVENT,
            discard_and_retry=DISCARD_AND_RETRY_EVENT,
            stop_and_save=STOP_AND_SAVE_EVENT,
            stop_no_save=STOP_NO_SAVE_EVENT,
            skip_sleep=SKIP_SLEEP_EVENT,
        ),
        start_trigger=_START_RECORDING_TRIGGER,
        lock=_COMMAND_LOCK,
    )
    _COMMAND_COORDINATOR = RecorderCommandCoordinator(
        _TRIGGER_CONTROLLER,
        _SAFE_STOP_CONTROLLER,
        lock=_COMMAND_LOCK,
    )
    _KEY_ROUTER = RecorderKeyRouter(
        get_phase=lambda: _TRIGGER_CONTROLLER.phase,
        on_b=lambda: _handle_b_trigger("keyboard"),
        on_d=lambda: _handle_d_trigger("keyboard"),
        on_m=_handle_m_trigger,
        on_s=_handle_s_trigger,
        on_r=_handle_r_trigger,
        on_ignored=_handle_ignored_retry_key,
    )

    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    task_config = load_yaml_file("task", base_path=base_path)
    task = task_config["tasks"].get(args.get("task_name"))

    dataset_dir = os.path.expanduser(task.get("dataset_dir"))
    max_timesteps = task.get("episode_len")

    if args["episode_idx"] is not None:
        episode_idx = args["episode_idx"]
    else:
        episode_idx = find_next_available_episode_index(
            dataset_dir,
            start_index=0,
        )

    episode_decision = check_episode_index(
        dataset_dir=dataset_dir,
        episode_idx=episode_idx,
    )
    if not episode_decision.proceed:
        return

    initial_episode_idx = episode_idx
    print(f"episode_{initial_episode_idx}\n")
    print("[运行参数]")
    print(f"  - start_trigger={_START_RECORDING_TRIGGER}")
    print(f"  - pedal_device={args.get('pedal_device', DEFAULT_PEDAL_PATH)}")
    print(f"  - random_start_positions={_ENABLE_RANDOM_START_POSITIONS}")
    print(f"  - save_return_to_start_data_on_b={_SAVE_RETURN_TO_START_DATA_ON_B}")
    print(f"  - return_home_between_episodes={_RETURN_HOME_BETWEEN_EPISODES}")
    print(f"  - video_encoder={_VIDEO_ENCODER_BACKEND}")
    print(f"  - leader_hold_policy={leader_hold_policy}")
    print(f"  - pedal_debounce_seconds={pedal_debounce_seconds}")
    print(f"  - opening_home_min_seconds={opening_home_min_seconds}")
    print(f"  - opening_max_joint_speed={opening_max_joint_speed}")
    print(f"  - joint_state_moving_timeout={joint_state_moving_timeout}")
    print(f"  - joint_state_idle_timeout={joint_state_idle_timeout}")
    print(f"  - health_watchdog_rate_hz={health_watchdog_rate_hz}")
    print(f"  - rearm_max_joint_error_rad={_REARM_MAX_JOINT_ERROR_RAD}")
    print(f"  - rearm_debounce_samples={_REARM_DEBOUNCE_SAMPLES}")
    print(f"  - continuous_roll_joints={continuous_roll_joints}")
    print(f"  - motor6_diagnostics={bool(args.get('motor6_diagnostics', False))}")
    print(f"  - motor6_diagnostics_rate_hz={float(args.get('motor6_diagnostics_rate_hz', 0.5))}")
    print(f"  - follower_gripper_current_limits={_FOLLOWER_GRIPPER_CURRENT_LIMIT_OVERRIDES}")
    print(f"  - dataset_dir={dataset_dir}")
    print(f"  - episode_idx={initial_episode_idx}")

    initialize_ros_context(
        ok=rclpy.ok,
        init=rclpy.init,
        no_signal_handlers=SignalHandlerOptions.NO,
    )

    # ★ 注册 Ctrl+C 信号
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    trigger_socket = Path(args.get("trigger_socket", DEFAULT_SOCKET_PATH))
    pedal_context = (
        FootPedalListener(
            args.get("pedal_device", DEFAULT_PEDAL_PATH),
            lambda: _handle_b_trigger("foot-pedal"),
            debounce_seconds=pedal_debounce_seconds,
            on_failure=_handle_pedal_failure,
        )
        if _START_RECORDING_TRIGGER == "b"
        else nullcontext()
    )
    save_worker = SaveWorker(capacity=1)
    _SAFE_STOP_CONTROLLER.set_retry_input_available(sys.stdin.isatty())
    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()
    session_failed = False
    try:
        with TriggerSocketServer(trigger_socket, _handle_remote_trigger), pedal_context:
            print(f"[remote-trigger] 监听 Unix socket：{trigger_socket}")
            outcome = run_continuous_session(
                create_runtime=lambda: create_recorder_runtime(
                    config=config,
                    torque_base=torque_base,
                    continuous_roll_joints=continuous_roll_joints,
                    health_watchdog_rate_hz=health_watchdog_rate_hz,
                ),
                capture_episode=lambda runtime, index: capture_one_episode(
                    runtime=runtime,
                    save_worker=save_worker,
                    episode_idx=index,
                    max_timesteps=max_timesteps,
                    dataset_dir=dataset_dir,
                    allow_existing=index == initial_episode_idx
                    and episode_decision.allow_existing,
                    torque_base=torque_base,
                    gravity_compensation=gravity_compensation,
                    config=config,
                    continuous_roll_joints=continuous_roll_joints,
                    motor6_diagnostics=bool(
                        args.get("motor6_diagnostics", False)
                    ),
                    motor6_diagnostics_rate_hz=float(
                        args.get("motor6_diagnostics_rate_hz", 0.5)
                    ),
                    return_home_between_episodes=(
                        _RETURN_HOME_BETWEEN_EPISODES
                    ),
                    video_encoder_backend=_VIDEO_ENCODER_BACKEND,
                    leader_hold_policy=leader_hold_policy,
                    rearm_max_joint_error_rad=(
                        _REARM_MAX_JOINT_ERROR_RAD
                    ),
                    rearm_debounce_samples=_REARM_DEBOUNCE_SAMPLES,
                    opening_home_min_seconds=opening_home_min_seconds,
                    opening_max_joint_speed=opening_max_joint_speed,
                    joint_state_idle_timeout=joint_state_idle_timeout,
                    joint_state_moving_timeout=(
                        joint_state_moving_timeout
                    ),
                ),
                next_index=lambda index: find_next_available_episode_index(
                    dataset_dir,
                    start_index=index + 1,
                ),
                initial_index=initial_episode_idx,
                final_cleanup=lambda runtime, outcome: (
                    finalize_recorder_runtime(
                        runtime,
                        outcome=outcome,
                        save_worker=save_worker,
                        robot_name=robot_base,
                        gravity_compensation_active=(
                            gravity_compensation
                        ),
                    )
                ),
            )
            print(f"[session] 采集会话结束: {outcome.value}")
    except BaseException:
        session_failed = True
        raise
    finally:
        PROGRAM_EXIT_EVENT.set()
        kb_thread.join(timeout=1.0)
        if kb_thread.is_alive():
            print("[keyboard] 警告：监听线程未在 1 秒内退出。")
        print("[保存] 等待后台 worker 完成已接收的 episode...")
        try:
            save_worker.shutdown(
                timeout=SAVE_ABORT_TIMEOUT_SECONDS,
                raise_failure=True,
            )
        except BaseException as shutdown_error:
            if not session_failed:
                raise
            bounded_best_effort_log(
                print,
                "[保存] 后台 worker 收尾失败（会话已有错误）: "
                f"{type(shutdown_error).__name__}: {shutdown_error}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launches robot teleoperation with specified parameters.")
    parser.add_argument("-t", "--task_name", action="store", type=str, help="Task name to specify the teleoperation task.", required=True)
    parser.add_argument("--episode_idx", action="store", type=int, help="Episode index to name the dataset file. Auto-generated if not provided.", default=None, required=False)
    parser.add_argument("-b", "--enable_base_torque", action="store_true", help=("Enable base torque for mobile robots during recording. Allows joystick control or other manual methods."))
    parser.add_argument("-g", "--gravity_compensation", action="store_true", help="Enable gravity compensation for leader robots at the start of teleoperation.")
    parser.add_argument("-r", "--robot", action="store", type=str, help="Robot setup configuration (e.g., aloha_solo, aloha_stationary, aloha_mobile).", required=True)
    parser.add_argument(
        "--start-trigger",
        choices=("gripper", "b"),
        default="gripper",
        help=(
            "How to start writing episode data. "
            "'gripper' starts after both leader grippers are closed in opening_ceremony (default); "
            "'b' keeps the old first-b-to-start behavior while follower follows before recording."
        ),
    )
    parser.add_argument(
        "--trigger-socket",
        default=str(DEFAULT_SOCKET_PATH),
        help="Private Unix socket for local recording trigger commands.",
    )
    parser.add_argument(
        "--pedal-device",
        default=str(DEFAULT_PEDAL_PATH),
        help="Stable Linux input path for the local PCsensor foot switch.",
    )
    parser.add_argument(
        "--pedal-debounce-seconds",
        type=float,
        default=1.0,
        help=(
            "Minimum interval between accepted local PCsensor presses. "
            "Default: 1.0 seconds."
        ),
    )
    parser.add_argument(
        "--opening-home-min-seconds",
        type=float,
        default=_OPENING_HOME_MIN_SECONDS,
        help=(
            "Minimum duration for initial HOME motion. Safety floor: "
            "4.0 seconds."
        ),
    )
    parser.add_argument(
        "--opening-max-joint-speed",
        type=float,
        default=_OPENING_MAX_JOINT_SPEED,
        help="Maximum planned initial HOME joint speed in rad/s.",
    )
    parser.add_argument(
        "--joint-state-moving-timeout",
        type=float,
        default=_JOINT_STATE_MOVING_TIMEOUT,
        help="Maximum joint-state age during guarded motion in seconds.",
    )
    parser.add_argument(
        "--joint-state-idle-timeout",
        type=float,
        default=_JOINT_STATE_IDLE_TIMEOUT,
        help="Maximum joint-state age at idle safety gates in seconds.",
    )
    parser.add_argument(
        "--health-watchdog-rate-hz",
        type=float,
        default=_HEALTH_WATCHDOG_RATE_HZ,
        help="Joint-state watchdog check rate. It performs no bus reads.",
    )
    parser.add_argument(
        "--random-start-positions",
        action="store_true",
        dest="enable_random_start_positions",
        help="Enable sampled random start poses from config/sampled_start_positions_1000_structured.json.",
    )
    parser.add_argument(
        "--disable-random-start-positions",
        action="store_false",
        dest="enable_random_start_positions",
        help="Use the original fixed start pose. This is the default.",
    )
    parser.add_argument(
        "--no-save-return-to-start-on-b",
        action="store_false",
        dest="save_return_to_start_data_on_b",
        help="When pressing b, return to the start pose but only save collected data, not the return-to-start segment.",
    )
    parser.add_argument(
        "--return-home-between-episodes",
        action="store_true",
        help=(
            "Compatibility mode: return HOME after each accepted episode. "
            "By default the follower holds the stop pose and the next episode "
            "uses the dual-leader-gripper open-to-close rearm gesture."
        ),
    )
    parser.add_argument(
        "--leader-hold-policy",
        choices=("strict", "best-effort", "off"),
        default="best-effort",
        help=(
            "Current-pose Leader hold behavior: strict aborts saving on "
            "hold failure; best-effort warns and saves; off skips the hold."
        ),
    )
    parser.add_argument(
        "--video-encoder",
        choices=("auto", "nvenc", "cpu"),
        default="auto",
        help=(
            "MP4 encoder backend. 'auto' probes h264_nvenc and falls back "
            "to libx264; 'nvenc' fails closed if GPU encoding is unavailable."
        ),
    )
    parser.add_argument(
        "--rearm-max-joint-error-rad",
        type=float,
        default=0.1,
        help=(
            "Maximum leader/follower joint error allowed before current-pose "
            "following is restored. Default: 0.1 rad."
        ),
    )
    parser.add_argument(
        "--rearm-debounce-samples",
        type=int,
        default=3,
        help=(
            "Consecutive dual-gripper open and close samples required for "
            "current-pose rearm. Default: 3."
        ),
    )
    parser.add_argument(
        "--continuous-roll-joints",
        action="store_true",
        dest="continuous_roll_joints",
        help=(
            "Enable continuous angle unwrapping for leader joints 4/6 and set follower "
            "forearm_roll/wrist_rotate to Dynamixel ext_position mode. This is the default."
        ),
    )
    parser.add_argument(
        "--no-continuous-roll-joints",
        action="store_false",
        dest="continuous_roll_joints",
        help="Disable continuous angle handling for joints 4/6 and use normal position mode.",
    )
    parser.add_argument(
        "--motor6-diagnostics",
        action="store_true",
        dest="motor6_diagnostics",
        help="Opt-in: record low-rate leader/follower motor 6 and gripper diagnostics.",
    )
    parser.add_argument(
        "--no-motor6-diagnostics",
        action="store_false",
        dest="motor6_diagnostics",
        help="Disable motor 6 diagnostics during collection.",
    )
    parser.add_argument(
        "--motor6-diagnostics-rate-hz",
        type=float,
        default=0.5,
        help="Sampling rate for opt-in motor 6 diagnostics JSONL. Default: 0.5 Hz.",
    )
    parser.add_argument(
        "--left-gripper-current-limit",
        type=int,
        default=FOLLOWER_GRIPPER_CURRENT_LIMITS["follower_left"],
        help="Current_Limit for follower_left gripper. Default: 300.",
    )
    parser.add_argument(
        "--right-gripper-current-limit",
        type=int,
        default=FOLLOWER_GRIPPER_CURRENT_LIMITS["follower_right"],
        help="Current_Limit for follower_right gripper. Default: 550.",
    )
    parser.set_defaults(
        enable_random_start_positions=False,
        save_return_to_start_data_on_b=True,
        return_home_between_episodes=False,
        continuous_roll_joints=True,
        motor6_diagnostics=False,
    )

    try:
        main(vars(parser.parse_args()))
    except KeyboardInterrupt:
        print("\n[shutdown] 安全停止完成。")

#!/usr/bin/env python3
"""Safely recover every responsive ALOHA arm to verified sleep."""

from pathlib import Path
import signal
import threading
import time

from aloha.gripper_control import (
    configure_follower_gripper_mode,
    restore_gripper_idle_modes,
)
from aloha.interbotix_service import (
    set_gravity_compensation_with_timeout,
    set_operating_modes_with_timeout,
    torque_enable_with_timeout,
)
from aloha.robot_health import (
    RobotHealthMonitor,
    RobotHealthUnavailable,
    attach_joint_state_subscriptions,
)
from aloha.robot_utils import (
    get_arm_joint_positions,
    load_yaml_file,
)
from aloha.recovery_lease import RecoveryLease
from aloha.safe_sleep import (
    SAFE_SLEEP_MIN_SECONDS,
    recover_robots_to_sleep,
)
from aloha.safe_sleep_runtime import (
    build_prepare_robot,
    initialize_robots_independently,
    initialize_ros_context,
    install_recovery_signal_handlers,
    merge_initialization_failures,
    parse_sleep_args,
)
from aloha.safe_stop import SafeStopController
from aloha.safety_state import RecoveryIdentity, publish_safety_state
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


SERVICE_TIMEOUT_SECONDS = 2.0
MOVING_TIMEOUT_SECONDS = 0.30
IDLE_TIMEOUT_SECONDS = 0.75
RECOVERY_LOG_MAX_BYTES = 1_000_000
SAFE_STATE_OWNER_GRACE_SECONDS = 2.0


def _restore_post_session_grippers(
    robots,
    logger,
    *,
    configure_follower=configure_follower_gripper_mode,
    restore_idle=restore_gripper_idle_modes,
    set_modes=set_operating_modes_with_timeout,
    set_torque=torque_enable_with_timeout,
    timeout_seconds=SERVICE_TIMEOUT_SECONDS,
):
    def set_gripper_operating_modes(
        robot,
        command_type,
        name,
        mode,
        *,
        profile_type="velocity",
        profile_velocity=0,
        profile_acceleration=0,
    ):
        set_modes(
            robot,
            command_type,
            name,
            mode,
            timeout_sec=timeout_seconds,
            profile_type=profile_type,
            profile_velocity=profile_velocity,
            profile_acceleration=profile_acceleration,
        )

    def configure_follower_gripper(robot_name, robot):
        configure_follower(
            robot_name,
            robot,
            set_operating_modes=set_gripper_operating_modes,
        )

    def torque_gripper(robot, command_type, name, enabled):
        set_torque(
            robot,
            command_type,
            name,
            enabled,
            timeout_sec=timeout_seconds,
        )

    restore_idle(
        robots,
        configure_follower_gripper=configure_follower_gripper,
        torque_enable=torque_gripper,
        logger=logger,
    )


def _publish_safe_recovery(
    report,
    robots,
    logger,
    publish_state,
    *,
    restore_grippers=_restore_post_session_grippers,
):
    restore_grippers(robots, logger)
    publish_state("SAFE_TO_STOP", report=report)


def _expected_joint_state_names(robot) -> set[str]:
    expected = set(robot.arm.group_info.joint_names)
    expected.update(robot.gripper.gripper_info.joint_names)
    if not expected:
        raise ValueError(
            f"{robot.core.robot_name} exposes no joint-state names"
        )
    return expected


class _RecoveryLogger:
    def __init__(self, recovery_id: str) -> None:
        self.path = Path(
            f"/tmp/aloha-safe-sleep-{recovery_id}.log"
        )
        self._file = self.path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._written = self.path.stat().st_size

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        encoded_size = len((message + "\n").encode("utf-8"))
        if self._written + encoded_size > RECOVERY_LOG_MAX_BYTES:
            return
        self._file.write(message + "\n")
        self._written += encoded_size

    def close(self) -> None:
        self._file.close()


def _print_report(report, logger=print) -> None:
    for robot_name, result in report.results.items():
        logger(
            f"[safe-sleep] {robot_name}: {result.status.value}; "
            f"{result.reason}"
        )


def main(argv=None) -> int:
    args = parse_sleep_args(argv)

    base_path = Path(__file__).resolve().parent.parent / "config"
    config = load_yaml_file(
        "robot",
        args.robot,
        base_path,
    ).get("robot", {})
    dt = 1 / config.get("fps", 50)
    lease = RecoveryLease.acquire(
        source="standalone",
        robot=args.robot,
        recovery_id=args.recovery_id,
    )
    identity = RecoveryIdentity(
        recovery_id=lease.metadata.recovery_id,
        owner_pid=lease.metadata.owner_pid,
        source=lease.metadata.source,
    )
    logger = _RecoveryLogger(identity.recovery_id)
    node = None
    health = None
    health_subscriptions = []
    rclpy = None
    try:
        import rclpy
        from rclpy.signals import SignalHandlerOptions

        initialize_ros_context(
            ok=rclpy.ok,
            init=rclpy.init,
            no_signal_handlers=SignalHandlerOptions.NO,
        )

        stop_no_save = threading.Event()
        stop_and_save = threading.Event()
        skip_sleep = threading.Event()
        controller = SafeStopController(
            stop_no_save,
            stop_and_save,
            skip_sleep,
            logger=logger,
        )
        install_recovery_signal_handlers(
            signal_module=signal,
            controller=controller,
        )
        controller.begin_cleanup()

        node = create_interbotix_global_node("aloha")
        arm_configs = (
            list(config.get("leader_arms", []))
            + list(config.get("follower_arms", []))
        )

        def create_robot(arm_config):
            logger(f"Initializing arm: {arm_config['name']}")
            return InterbotixManipulatorXS(
                robot_model=arm_config["model"],
                robot_name=arm_config["name"],
                node=node,
                iterative_update_fk=False,
            )

        robots, initialization_failures = (
            initialize_robots_independently(
                arm_configs,
                create_robot=create_robot,
                logger=logger,
            )
        )
        robot_startup(node)
        health = RobotHealthMonitor(watchdog_rate_hz=10.0)
        health_subscriptions.extend(
            attach_joint_state_subscriptions(
                node,
                health,
                {
                    name: _expected_joint_state_names(robot)
                    for name, robot in robots.items()
                },
            )
        )
        health.start()

        def publish_state(state, *, report=None):
            publish_safety_state(
                state,
                report=report,
                recovery=identity,
                context_ok=rclpy.ok(),
            )

        try:
            health.wait_for_fresh(
                robots,
                consecutive=3,
                max_age=IDLE_TIMEOUT_SECONDS,
                timeout=2.0,
                stop_requested=lambda: False,
            )
        except RobotHealthUnavailable as exc:
            logger(
                "[safe-sleep] 初始健康门未全通过；仍将跳过失联臂并回收"
                f"其余机械臂: {exc}"
            )

        prepare_robot = build_prepare_robot(
            gravity_compensation_active=(
                args.gravity_compensation_active
            ),
            timeout_seconds=SERVICE_TIMEOUT_SECONDS,
            set_gravity=set_gravity_compensation_with_timeout,
            set_mode=set_operating_modes_with_timeout,
            set_torque=torque_enable_with_timeout,
        )

        def torque_off_robot(_robot_name, robot):
            torque_enable_with_timeout(
                robot,
                "group",
                "arm",
                False,
                timeout_sec=SERVICE_TIMEOUT_SECONDS,
            )

        def recover():
            return merge_initialization_failures(
                recover_robots_to_sleep(
                    robots=robots,
                    health=health,
                    prepare_robot=prepare_robot,
                    torque_off_robot=torque_off_robot,
                    read_positions=get_arm_joint_positions,
                    safe_sleep_positions=config.get(
                        "safe_sleep_positions",
                        {},
                    ),
                    dt=dt,
                    minimum_seconds=SAFE_SLEEP_MIN_SECONDS,
                    max_joint_speed=0.4,
                    moving_timeout=MOVING_TIMEOUT_SECONDS,
                    idle_timeout=IDLE_TIMEOUT_SECONDS,
                    stop_requested=lambda: False,
                    allow_pose_deviation=args.allow_pose_deviation,
                    logger=logger,
                ),
                initialization_failures,
            )

        publish_state("RECOVERY_IN_PROGRESS")
        report = recover()
        _print_report(report, logger)
        if not report.safe_to_stop:
            publish_state("UNSAFE_HOLD", report=report)
            logger(
                "[UNSAFE_HOLD] 四臂回收已结束；失败臂不重试，"
                "safe-sleep 退出。"
            )
            return 2

        _publish_safe_recovery(
            report,
            robots,
            logger,
            publish_state,
        )
        time.sleep(SAFE_STATE_OWNER_GRACE_SECONDS)
        health.stop()
        del health_subscriptions
        robot_shutdown(node)
        return 0
    except BaseException as exc:
        logger(
            "[UNSAFE_HOLD] 独立回收初始化或执行异常；"
            "禁止关闭容器且不调用 robot_shutdown: "
            f"{type(exc).__name__}: {exc}"
        )
        try:
            publish_safety_state(
                "UNSAFE_HOLD",
                recovery=identity,
                context_ok=bool(
                    rclpy is not None and rclpy.ok()
                ),
            )
        except Exception as publish_exc:
            logger(
                "[UNSAFE_HOLD] 安全状态发布失败，主机停止门仍应"
                "因 owner 退出而拒绝关闭: "
                f"{type(publish_exc).__name__}: {publish_exc}"
            )
        if health is not None:
            try:
                health.stop()
            except Exception as health_exc:
                logger(
                    "[UNSAFE_HOLD] 健康监控停止失败: "
                    f"{type(health_exc).__name__}: {health_exc}"
                )
        return 2
    finally:
        logger.close()
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())

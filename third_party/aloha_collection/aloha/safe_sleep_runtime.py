"""Device-independent policy helpers for the standalone sleep command."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from aloha.safe_sleep import (
    RobotSleepResult,
    SafeSleepReport,
    SleepStatus,
)


def parse_sleep_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sleep",
        description="Move all ALOHA arms to independently verified sleep.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="Compatibility flag. Safe shutdown verifies every arm.",
    )
    parser.add_argument(
        "-r",
        "--robot",
        required=True,
        help="Robot configuration such as aloha_stationary.",
    )
    parser.add_argument(
        "--gravity-compensation-active",
        action="store_true",
        default=False,
        help=(
            "Declare that leader gravity compensation is active and must "
            "be disabled before recovery. Machine 103 leaves this unset."
        ),
    )
    parser.add_argument(
        "--skip-gravity-compensation",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--recovery-id",
        help="Recovery ID supplied by a recorder fallback.",
    )
    parser.add_argument(
        "--allow-pose-deviation",
        action="store_true",
        default=False,
        help=(
            "Recorder keyboard-s policy: record a finite sleep-pose "
            "deviation but continue strict torque-off verification."
        ),
    )
    parser.add_argument(
        "--fast-sleep",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag; safe-recovery timing "
            "is unchanged."
        ),
    )
    args = parser.parse_args(argv)
    if (
        args.gravity_compensation_active
        and args.skip_gravity_compensation
    ):
        parser.error(
            "--gravity-compensation-active conflicts with "
            "--skip-gravity-compensation"
        )
    if args.skip_gravity_compensation:
        args.gravity_compensation_active = False
    return args


def initialize_ros_context(
    *,
    ok: Callable[[], bool],
    init: Callable[..., None],
    no_signal_handlers: Any,
) -> None:
    """Initialize rclpy while leaving Unix signals to the application."""

    if not ok():
        init(signal_handler_options=no_signal_handlers)


def install_recovery_signal_handlers(
    *,
    signal_module,
    controller,
) -> None:
    """Install application-owned recovery handlers before robot creation."""

    signal_module.signal(
        signal_module.SIGINT,
        lambda *_: controller.handle_sigint(),
    )
    signal_module.signal(
        signal_module.SIGTERM,
        lambda *_: controller.handle_sigterm(),
    )
    if hasattr(signal_module, "SIGUSR1"):
        signal_module.signal(
            signal_module.SIGUSR1,
            lambda *_: controller.request_from_s(wake_main=False),
        )


def build_prepare_robot(
    *,
    gravity_compensation_active: bool,
    timeout_seconds: float,
    set_gravity: Callable[..., Any],
    set_mode: Callable[..., Any],
    set_torque: Callable[..., Any],
) -> Callable[[str, object], None]:
    """Build the bounded per-arm preparation callback."""

    def prepare_robot(robot_name: str, robot: object) -> None:
        if (
            gravity_compensation_active
            and robot_name.startswith("leader_")
        ):
            set_gravity(
                robot,
                False,
                timeout_sec=timeout_seconds,
            )
        set_mode(
            robot,
            "group",
            "arm",
            "position",
            timeout_sec=timeout_seconds,
        )
        set_torque(
            robot,
            "group",
            "arm",
            True,
            timeout_sec=timeout_seconds,
        )

    return prepare_robot


def initialize_robots_independently(
    arm_configs: Sequence[dict],
    *,
    create_robot: Callable[[dict], object],
    logger: Callable[[str], None],
) -> tuple[dict[str, object], dict[str, str]]:
    """Attempt every configured arm even when an earlier arm fails."""

    robots = {}
    failures = {}
    for arm_config in arm_configs:
        robot_name = arm_config["name"]
        try:
            robots[robot_name] = create_robot(arm_config)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures[robot_name] = reason
            logger(
                f"[safe-sleep] {robot_name} initialization failed: "
                f"{reason}"
            )
    return robots, failures


def merge_initialization_failures(
    report: SafeSleepReport,
    failures: dict[str, str],
) -> SafeSleepReport:
    """Preserve missing configured arms as fail-closed report entries."""

    results = dict(report.results)
    for robot_name, reason in failures.items():
        results[robot_name] = RobotSleepResult(
            robot_name=robot_name,
            status=SleepStatus.UNRESPONSIVE,
            max_error_rad=None,
            reason=reason,
            phase="initialize",
            torque_off_verified=False,
        )
    return SafeSleepReport(results=results)


def retry_failed_initializations(
    arm_configs: Sequence[dict],
    *,
    robots: dict[str, object],
    failures: dict[str, str],
    create_robot: Callable[[dict], object],
    logger: Callable[[str], None],
) -> tuple[dict[str, object], dict[str, str]]:
    """Stage retries without mutating live robot/failure registries."""

    missing_configs = [
        config
        for config in arm_configs
        if (
            config["name"] not in robots
            and config["name"] in failures
        )
    ]
    return initialize_robots_independently(
        missing_configs,
        create_robot=create_robot,
        logger=logger,
    )

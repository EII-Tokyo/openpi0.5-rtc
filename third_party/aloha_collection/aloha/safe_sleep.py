"""Per-robot sleep recovery that isolates unresponsive hardware."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Callable, Mapping, Sequence

from aloha.safe_motion import move_robots_guarded, plan_motion_duration


SAFE_HOME_POSITIONS = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
SAFE_SLEEP_MIN_SECONDS = 1.0
POST_TORQUE_STABILITY_RAD = 0.02


class SleepStatus(Enum):
    SLEPT_VERIFIED = "slept_verified"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"


@dataclass(frozen=True)
class RobotSleepResult:
    robot_name: str
    status: SleepStatus
    max_error_rad: float | None
    reason: str
    phase: str = "unknown"
    torque_off_verified: bool = False


@dataclass(frozen=True)
class SafeSleepReport:
    results: dict[str, RobotSleepResult]

    @property
    def safe_to_stop(self) -> bool:
        return bool(self.results) and all(
            item.status is SleepStatus.SLEPT_VERIFIED
            and item.torque_off_verified
            for item in self.results.values()
        )


def recover_robots_to_sleep(
    *,
    robots: Mapping[str, object],
    health,
    sleep_one: Callable[[str, object], RobotSleepResult] | None = None,
    prepare_robot: Callable[[str, object], None] | None = None,
    torque_off_robot: Callable[[str, object], None] | None = None,
    read_positions: Callable[[object], Sequence[float]] | None = None,
    safe_sleep_positions: Mapping[
        str,
        Mapping[str, float],
    ] | None = None,
    home_positions: Sequence[float] = SAFE_HOME_POSITIONS,
    dt: float = 0.02,
    minimum_seconds: float = SAFE_SLEEP_MIN_SECONDS,
    max_joint_speed: float = 0.4,
    moving_timeout: float = 0.30,
    idle_timeout: float = 0.75,
    gate_timeout: float = 2.0,
    max_error_rad: float = 0.10,
    settle_seconds: float = 0.5,
    settle_error_rad: float = POST_TORQUE_STABILITY_RAD,
    verification_samples: int = 3,
    settle_sleep: Callable[[float], None] = time.sleep,
    verification_clock: Callable[[], float] = time.monotonic,
    stop_requested: Callable[[], bool] = lambda: False,
    allow_pose_deviation: bool = False,
    move_guarded=move_robots_guarded,
    plan_duration=plan_motion_duration,
    logger: Callable[[str], None] = print,
) -> SafeSleepReport:
    """Attempt every responsive robot independently, leaders before followers."""

    _validate_positive_finite("moving_timeout", moving_timeout)
    _validate_positive_finite("idle_timeout", idle_timeout)
    _validate_positive_finite("gate_timeout", gate_timeout)
    _validate_positive_finite("max_error_rad", max_error_rad)
    _validate_positive_finite("settle_seconds", settle_seconds)
    _validate_positive_finite("settle_error_rad", settle_error_rad)
    if (
        isinstance(verification_samples, bool)
        or not isinstance(verification_samples, int)
        or verification_samples <= 0
    ):
        raise ValueError("verification_samples must be a positive integer")

    if not robots:
        return SafeSleepReport(results={})

    snapshots = {
        robot_name: health.snapshot(robot_name)
        for robot_name in robots
    }
    results: dict[str, RobotSleepResult] = {}
    responsive_names = []
    for robot_name, snapshot in snapshots.items():
        if (
            snapshot.valid
            and snapshot.consecutive_valid > 0
            and snapshot.message_age <= idle_timeout
        ):
            responsive_names.append(robot_name)
            continue
        reason = snapshot.reason or "joint_state_stale"
        results[robot_name] = RobotSleepResult(
            robot_name=robot_name,
            status=SleepStatus.UNRESPONSIVE,
            max_error_rad=None,
            reason=(
                f"{reason}; message_age={snapshot.message_age:.3f}s"
            ),
        )
        logger(
            f"[safe-sleep] {robot_name} UNRESPONSIVE: "
            f"{results[robot_name].reason}"
        )

    ordered_names = _recovery_order(responsive_names)
    if sleep_one is None:
        if prepare_robot is None or read_positions is None:
            raise ValueError(
                "prepare_robot and read_positions are required "
                "when sleep_one is not supplied"
            )

        def sleep_one(robot_name, robot):
            return _sleep_one_robot(
                robot_name=robot_name,
                robot=robot,
                health=health,
                prepare_robot=prepare_robot,
                torque_off_robot=torque_off_robot,
                read_positions=read_positions,
                safe_sleep_positions=safe_sleep_positions or {},
                home_positions=home_positions,
                dt=dt,
                minimum_seconds=minimum_seconds,
                max_joint_speed=max_joint_speed,
                moving_timeout=moving_timeout,
                gate_timeout=gate_timeout,
                max_error_rad=max_error_rad,
                idle_timeout=idle_timeout,
                settle_seconds=settle_seconds,
                settle_error_rad=settle_error_rad,
                verification_samples=verification_samples,
                settle_sleep=settle_sleep,
                verification_clock=verification_clock,
                stop_requested=stop_requested,
                allow_pose_deviation=allow_pose_deviation,
                move_guarded=move_guarded,
                plan_duration=plan_duration,
            )

    results_lock = threading.Lock()

    def recover_one(robot_name: str) -> None:
        try:
            result = sleep_one(robot_name, robots[robot_name])
            if not isinstance(result, RobotSleepResult):
                raise TypeError("sleep_one must return RobotSleepResult")
            if result.robot_name != robot_name:
                raise ValueError(
                    "sleep_one result robot_name does not match request"
                )
        except Exception as exc:
            result = RobotSleepResult(
                robot_name=robot_name,
                status=SleepStatus.FAILED,
                max_error_rad=None,
                reason=f"{type(exc).__name__}: {exc}",
            )
        with results_lock:
            results[robot_name] = result

    workers = [
        threading.Thread(
            target=recover_one,
            args=(robot_name,),
            name=f"aloha-safe-sleep-{robot_name}",
            daemon=False,
        )
        for robot_name in ordered_names
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    for robot_name in ordered_names:
        result = results[robot_name]
        logger(
            f"[safe-sleep] {robot_name} {result.status.value}: "
            f"{result.reason}"
        )

    return SafeSleepReport(results=results)


def _sleep_one_robot(
    *,
    robot_name: str,
    robot,
    health,
    prepare_robot: Callable[[str, object], None],
    torque_off_robot: Callable[[str, object], None] | None,
    read_positions: Callable[[object], Sequence[float]],
    safe_sleep_positions: Mapping[str, Mapping[str, float]],
    home_positions: Sequence[float],
    dt: float,
    minimum_seconds: float,
    max_joint_speed: float,
    moving_timeout: float,
    gate_timeout: float,
    max_error_rad: float,
    idle_timeout: float,
    settle_seconds: float,
    settle_error_rad: float,
    verification_samples: int,
    settle_sleep: Callable[[float], None],
    verification_clock: Callable[[], float],
    stop_requested: Callable[[], bool],
    allow_pose_deviation: bool,
    move_guarded,
    plan_duration,
) -> RobotSleepResult:
    target_sleep = resolve_safe_sleep_target(
        robot_name,
        robot.arm.group_info,
        safe_sleep_positions,
    )
    target_home = tuple(float(value) for value in home_positions)
    if len(target_home) != len(target_sleep):
        raise ValueError(
            f"{robot_name} HOME and sleep pose lengths do not match"
        )

    prepare_robot(robot_name, robot)
    health.wait_for_fresh(
        {robot_name},
        consecutive=3,
        max_age=moving_timeout,
        timeout=gate_timeout,
        stop_requested=stop_requested,
    )

    with health.arm_scope(
        {robot_name},
        phase=f"safe_sleep:{robot_name}",
        max_age=moving_timeout,
        latch_global=False,
    ) as scope:
        scope.raise_if_faulted()

        current = tuple(float(value) for value in read_positions(robot))
        home_duration = plan_duration(
            current,
            target_home,
            minimum_seconds=minimum_seconds,
            max_joint_speed=max_joint_speed,
        )
        move_guarded(
            robots={robot_name: robot},
            targets={robot_name: target_home},
            dt=dt,
            duration=home_duration,
            fault_event=scope.fault_event,
            health_check=scope.raise_if_faulted,
        )

        current_home = tuple(
            float(value) for value in read_positions(robot)
        )
        sleep_duration = plan_duration(
            current_home,
            target_sleep,
            minimum_seconds=minimum_seconds,
            max_joint_speed=max_joint_speed,
        )
        move_guarded(
            robots={robot_name: robot},
            targets={robot_name: target_sleep},
            dt=dt,
            duration=sleep_duration,
            fault_event=scope.fault_event,
            health_check=scope.raise_if_faulted,
        )

        (
            max_error,
            verification_error,
            pose_tolerance_exceeded,
        ) = _verify_pose_samples(
            robot_name=robot_name,
            robot=robot,
            health=health,
            scope=scope,
            read_positions=read_positions,
            target=target_sleep,
            sample_count=verification_samples,
            max_age=moving_timeout,
            timeout=gate_timeout,
            stop_requested=stop_requested,
            tolerance=max_error_rad,
            error_label="sleep pose error",
        )
        pose_deviation_reason = None
        if (
            verification_error is not None
            and allow_pose_deviation
            and pose_tolerance_exceeded
        ):
            pose_deviation_reason = verification_error
        elif verification_error is not None:
            return RobotSleepResult(
                robot_name=robot_name,
                status=SleepStatus.FAILED,
                max_error_rad=max_error,
                reason=verification_error,
                phase="verify_sleep_pose",
            )

        if torque_off_robot is None:
            raise ValueError("torque_off_robot is required")
        try:
            torque_off_robot(robot_name, robot)
        except Exception as exc:
            return RobotSleepResult(
                robot_name=robot_name,
                status=SleepStatus.FAILED,
                max_error_rad=max_error,
                reason=f"{type(exc).__name__}: {exc}",
                phase="torque_off",
                torque_off_verified=False,
            )

        settle_sleep(settle_seconds)
        (
            settle_error,
            target_displacement,
            verification_error,
        ) = _verify_stable_samples(
            robot_name=robot_name,
            robot=robot,
            health=health,
            scope=scope,
            read_positions=read_positions,
            target=target_sleep,
            required_transitions=verification_samples,
            max_age=idle_timeout,
            timeout=gate_timeout,
            stop_requested=stop_requested,
            tolerance=settle_error_rad,
            clock=verification_clock,
        )
        if verification_error is not None:
            return RobotSleepResult(
                robot_name=robot_name,
                status=SleepStatus.FAILED,
                max_error_rad=settle_error,
                reason=verification_error,
                phase="verify_settle",
                torque_off_verified=False,
            )

    if pose_deviation_reason is None:
        result_max_error = settle_error
        reason = (
            "sleep pose and torque-off stability verified; "
            f"max step={settle_error:.3f} rad; "
            f"target displacement={target_displacement:.3f} rad"
        )
    else:
        result_max_error = max_error
        reason = (
            "pose deviation accepted for s exit: "
            f"{pose_deviation_reason}; "
            "torque-off stability verified; "
            f"max step={settle_error:.3f} rad; "
            f"target displacement={target_displacement:.3f} rad"
        )

    return RobotSleepResult(
        robot_name=robot_name,
        status=SleepStatus.SLEPT_VERIFIED,
        max_error_rad=result_max_error,
        reason=reason,
        phase="complete",
        torque_off_verified=True,
    )


def _maximum_joint_error(
    actual: Sequence[float],
    expected: Sequence[float],
) -> float:
    if not actual or len(actual) != len(expected):
        raise ValueError("actual and expected pose lengths must match")
    if not all(
        math.isfinite(value)
        for value in (*actual, *expected)
    ):
        raise ValueError("pose positions must be finite")
    return round(
        max(
            abs(actual_value - expected_value)
            for actual_value, expected_value in zip(actual, expected)
        ),
        12,
    )


def _verify_pose_samples(
    *,
    robot_name: str,
    robot,
    health,
    scope,
    read_positions: Callable[[object], Sequence[float]],
    target: Sequence[float],
    sample_count: int,
    max_age: float,
    timeout: float,
    stop_requested: Callable[[], bool],
    tolerance: float,
    error_label: str,
) -> tuple[float | None, str | None, bool]:
    maximum = 0.0
    for sample_index in range(1, sample_count + 1):
        health.wait_for_fresh(
            {robot_name},
            consecutive=1,
            max_age=max_age,
            timeout=timeout,
            stop_requested=stop_requested,
        )
        scope.raise_if_faulted()
        try:
            positions = tuple(
                float(value) for value in read_positions(robot)
            )
            error = _maximum_joint_error(positions, target)
        except (TypeError, ValueError) as exc:
            return (
                None,
                f"{error_label} sample {sample_index}/{sample_count} "
                f"is invalid: {exc}",
                False,
            )
        maximum = max(maximum, error)
        if error > tolerance:
            return (
                maximum,
                f"{error_label} {error:.3f} rad exceeds "
                f"{tolerance:.3f} rad at sample "
                f"{sample_index}/{sample_count}",
                True,
            )
    return maximum, None, False


def _verify_stable_samples(
    *,
    robot_name: str,
    robot,
    health,
    scope,
    read_positions: Callable[[object], Sequence[float]],
    target: Sequence[float],
    required_transitions: int,
    max_age: float,
    timeout: float,
    stop_requested: Callable[[], bool],
    tolerance: float,
    clock: Callable[[], float],
) -> tuple[float | None, float | None, str | None]:
    deadline = clock() + timeout
    previous: tuple[float, ...] | None = None
    stable_transitions = 0
    maximum_stable_delta = 0.0
    maximum_observed_delta = 0.0
    maximum_target_displacement = 0.0
    sample_index = 0

    while stable_transitions < required_transitions:
        remaining = deadline - clock()
        if remaining <= 0:
            return (
                maximum_observed_delta,
                maximum_target_displacement,
                "post-torque stability did not stabilize for "
                f"{required_transitions} consecutive transitions within "
                f"{timeout:.3f}s; max step="
                f"{maximum_observed_delta:.3f} rad; target displacement="
                f"{maximum_target_displacement:.3f} rad",
            )

        health.wait_for_fresh(
            {robot_name},
            consecutive=1,
            max_age=max_age,
            timeout=remaining,
            stop_requested=stop_requested,
        )
        scope.raise_if_faulted()
        sample_index += 1
        try:
            positions = tuple(
                float(value) for value in read_positions(robot)
            )
            target_displacement = _maximum_joint_error(
                positions,
                target,
            )
            if previous is None:
                delta = None
            else:
                delta = _maximum_joint_error(positions, previous)
        except (TypeError, ValueError) as exc:
            return (
                None,
                None,
                f"post-torque stability sample {sample_index} "
                f"is invalid: {exc}",
            )

        maximum_target_displacement = max(
            maximum_target_displacement,
            target_displacement,
        )
        if delta is not None:
            maximum_observed_delta = max(
                maximum_observed_delta,
                delta,
            )
            if delta <= tolerance:
                stable_transitions += 1
                maximum_stable_delta = max(
                    maximum_stable_delta,
                    delta,
                )
            else:
                stable_transitions = 0
                maximum_stable_delta = 0.0
        previous = positions

    return (
        maximum_stable_delta,
        maximum_target_displacement,
        None,
    )


def _validate_positive_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be positive and finite")


def resolve_safe_sleep_target(
    robot_name: str,
    group_info,
    configured: Mapping[str, Mapping[str, float]],
    *,
    limit_margin_rad: float = 0.01,
) -> tuple[float, ...]:
    """Resolve a named target and reject unsafe values before any motion."""

    names = tuple(group_info.joint_names)
    lower_limits = tuple(
        float(value) for value in group_info.joint_lower_limits
    )
    upper_limits = tuple(
        float(value) for value in group_info.joint_upper_limits
    )
    model_sleep = tuple(
        float(value) for value in group_info.joint_sleep_positions
    )
    expected_length = len(names)
    for label, values in (
        ("lower limits", lower_limits),
        ("upper limits", upper_limits),
        ("model sleep pose", model_sleep),
    ):
        if len(values) != expected_length:
            raise ValueError(
                f"{robot_name} {label} length does not match joint names"
            )

    if (
        isinstance(limit_margin_rad, bool)
        or not isinstance(limit_margin_rad, (int, float))
        or not math.isfinite(limit_margin_rad)
        or limit_margin_rad < 0
    ):
        raise ValueError(
            "limit_margin_rad must be non-negative and finite"
        )

    configured_pose = configured.get(robot_name)
    if configured_pose is None:
        target = model_sleep
        active_margin = 0.0
    else:
        if set(configured_pose) != set(names):
            missing = sorted(set(names) - set(configured_pose))
            extra = sorted(set(configured_pose) - set(names))
            raise ValueError(
                f"{robot_name} safe pose joint names do not exactly match; "
                f"missing={missing}, extra={extra}"
            )
        target = tuple(float(configured_pose[name]) for name in names)
        active_margin = float(limit_margin_rad)

    for name, value, lower, upper in zip(
        names,
        target,
        lower_limits,
        upper_limits,
    ):
        if not all(math.isfinite(item) for item in (value, lower, upper)):
            raise ValueError(
                f"{robot_name}.{name} target and limits must be finite"
            )
        if lower > upper:
            raise ValueError(
                f"{robot_name}.{name} limits must be ordered "
                f"[{lower:.3f}, {upper:.3f}]"
            )
        if value < lower + active_margin or value > upper - active_margin:
            raise ValueError(
                f"{robot_name}.{name} target {value:.3f} must remain "
                f"inside limits [{lower:.3f}, {upper:.3f}] by "
                f"{active_margin:.3f} rad"
            )
    return target


def _recovery_order(robot_names: Sequence[str]) -> list[str]:
    names = set(robot_names)
    leaders = sorted(name for name in names if name.startswith("leader_"))
    followers = sorted(
        name for name in names if name.startswith("follower_")
    )
    known = set(leaders) | set(followers)
    return leaders + followers + sorted(names - known)

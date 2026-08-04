"""Speed-limited arm motion with a joint-state health abort gate."""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence


class GuardedMotionAborted(RuntimeError):
    """Raised before a control cycle when robot health is no longer safe."""


def plan_motion_duration(
    current: Sequence[float],
    target: Sequence[float],
    *,
    minimum_seconds: float = 4.0,
    max_joint_speed: float = 0.4,
) -> float:
    if (
        isinstance(minimum_seconds, bool)
        or not isinstance(minimum_seconds, (int, float))
        or minimum_seconds <= 0
        or not math.isfinite(minimum_seconds)
    ):
        raise ValueError("minimum_seconds must be positive and finite")
    if max_joint_speed <= 0 or not math.isfinite(max_joint_speed):
        raise ValueError("max_joint_speed must be positive and finite")
    if len(current) == 0 or len(current) != len(target):
        raise ValueError("current and target must have the same non-zero length")

    current_values = tuple(float(value) for value in current)
    target_values = tuple(float(value) for value in target)
    if not all(
        math.isfinite(value) for value in current_values + target_values
    ):
        raise ValueError("current and target positions must be finite")

    max_delta = max(
        abs(target_value - current_value)
        for current_value, target_value in zip(
            current_values,
            target_values,
        )
    )
    return max(
        float(minimum_seconds),
        max_delta / float(max_joint_speed),
    )


def move_robots_guarded(
    *,
    robots: Mapping[str, object],
    targets: Mapping[str, Sequence[float]],
    dt: float,
    duration: float,
    fault_event,
    health_check: Callable[[], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Interpolate one or more robots in synchronized, abortable cycles."""

    if not robots:
        raise ValueError("robots must not be empty")
    if set(robots) != set(targets):
        raise ValueError("robots and targets must contain the same names")
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError("duration must be positive and finite")

    trajectories: dict[str, tuple[tuple[float, ...], ...]] = {}
    interval_count = max(1, int(math.ceil(duration / dt)))
    command_count = interval_count + 1

    for robot_name, robot in robots.items():
        current = tuple(
            float(value) for value in robot.arm.get_joint_positions()
        )
        target = tuple(float(value) for value in targets[robot_name])
        if not current or len(current) != len(target):
            raise ValueError(
                f"{robot_name} current and target positions must have "
                "the same non-zero length"
            )
        if not all(math.isfinite(value) for value in current + target):
            raise ValueError(f"{robot_name} positions must be finite")

        trajectories[robot_name] = tuple(
            tuple(
                start + (end - start) * command_index / interval_count
                for start, end in zip(current, target)
            )
            for command_index in range(command_count)
        )

    sleep_interval = duration / interval_count
    for command_index in range(command_count):
        _raise_if_faulted(fault_event)
        health_check()
        _raise_if_faulted(fault_event)

        for robot_name, robot in robots.items():
            robot.arm.set_joint_positions(
                trajectories[robot_name][command_index],
                blocking=False,
            )

        if command_index < interval_count:
            sleep(sleep_interval)


def _raise_if_faulted(fault_event) -> None:
    if fault_event.is_set():
        raise GuardedMotionAborted(
            "joint-state health fault aborted guarded motion"
        )

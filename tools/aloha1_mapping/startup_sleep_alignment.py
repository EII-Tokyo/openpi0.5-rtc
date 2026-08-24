"""Pure helpers for the one-shot real-robot startup Sleep alignment.

This module deliberately has no ROS imports.  The remote ROS runner uses these
helpers after it has received the frozen, source-audited Sleep manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


ARM_JOINT_COUNT = 6
DEFAULT_RATE_HZ = 50
DEFAULT_MOVE_SECONDS = 5.0


def validate_sleep_manifest(manifest: Mapping[str, object]) -> tuple[list[float], list[str]]:
    """Return the frozen Sleep target and joint order after strict validation."""

    order = manifest.get("joint_order")
    target = manifest.get("sleep_rad")
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        raise ValueError("sleep manifest joint_order is missing")
    if not isinstance(target, Sequence) or isinstance(target, (str, bytes)):
        raise ValueError("sleep manifest sleep_rad is missing")
    if len(order) != ARM_JOINT_COUNT or len(target) != ARM_JOINT_COUNT:
        raise ValueError("Sleep alignment requires exactly six arm joints")
    values = [float(value) for value in target]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Sleep target contains a non-finite value")
    names = [str(value) for value in order]
    expected = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    if names != expected:
        raise ValueError(f"unexpected arm joint order: {names}")
    return values, names


def interpolate_targets(
    start: Sequence[float], target: Sequence[float], *, rate_hz: int, move_seconds: float
) -> list[list[float]]:
    """Create a bounded-rate linear trajectory including both endpoints."""

    if len(start) != ARM_JOINT_COUNT or len(target) != ARM_JOINT_COUNT:
        raise ValueError("alignment trajectory requires six values")
    if rate_hz <= 0 or not math.isfinite(float(move_seconds)) or move_seconds <= 0:
        raise ValueError("rate_hz and move_seconds must be positive")
    start_values = [float(value) for value in start]
    target_values = [float(value) for value in target]
    if not all(math.isfinite(value) for value in start_values + target_values):
        raise ValueError("alignment trajectory contains a non-finite value")
    count = max(1, int(round(float(rate_hz) * float(move_seconds))))
    return [
        [start_values[j] + (target_values[j] - start_values[j]) * (i / count) for j in range(ARM_JOINT_COUNT)]
        for i in range(count + 1)
    ]


def max_step_velocity(
    start: Sequence[float], target: Sequence[float], *, rate_hz: int, move_seconds: float
) -> float:
    """Return the largest commanded joint velocity in rad/s."""

    if len(start) != ARM_JOINT_COUNT or len(target) != ARM_JOINT_COUNT:
        raise ValueError("velocity calculation requires six values")
    return max(
        abs(float(a) - float(b)) / float(move_seconds)
        for a, b in zip(start, target)
    )

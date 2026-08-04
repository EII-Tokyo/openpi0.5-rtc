"""Follower gripper operating-mode profiles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_FOLLOWER_GRIPPER_PROFILES = {
    "follower_left": ("velocity", 50, 10),
    "follower_right": ("velocity", 0, 0),
}


def configure_follower_gripper_mode(
    follower_name: str,
    follower: Any,
    *,
    set_operating_modes: Callable[..., None],
) -> None:
    """Apply the configured position/current profile for one follower."""
    try:
        profile_type, profile_velocity, profile_acceleration = (
            _FOLLOWER_GRIPPER_PROFILES[follower_name]
        )
    except KeyError as exc:
        raise ValueError(
            f"unsupported follower gripper profile: {follower_name}"
        ) from exc

    set_operating_modes(
        follower,
        "single",
        "gripper",
        "current_based_position",
        profile_type=profile_type,
        profile_velocity=profile_velocity,
        profile_acceleration=profile_acceleration,
    )


def restore_gripper_idle_modes(
    robots: dict[str, Any],
    *,
    configure_follower_gripper: Callable[[str, Any], None],
    torque_enable: Callable[[Any, str, str, bool], None],
    logger: Callable[[str], None] = print,
) -> None:
    """Best-effort restoration of post-session gripper modes and torque."""
    for robot_name, robot in robots.items():
        try:
            if robot_name.startswith("leader_"):
                torque_enable(robot, "single", "gripper", False)
            elif robot_name.startswith("follower_"):
                configure_follower_gripper(robot_name, robot)
                torque_enable(robot, "single", "gripper", True)
        except Exception as exc:
            logger(
                f"[gripper-idle] {robot_name} restoration failed: {exc}"
            )

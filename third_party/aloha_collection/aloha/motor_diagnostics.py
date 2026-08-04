"""Low-pressure, model-aware motor diagnostic sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aloha.interbotix_service import (
    InterbotixServiceTimeout,
    wait_for_service_future,
)


CURRENT_REGISTERS = (
    "Present_Current",
    "Current_Limit",
    "Goal_Current",
)

_ORDERED_REGISTERS = (
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
)


def diagnostic_registers_for_robot(robot_name: str) -> tuple[str, ...]:
    """Return the conservative register set for the configured ALOHA arm role."""
    if robot_name.startswith("follower_"):
        return _ORDERED_REGISTERS
    return tuple(
        register
        for register in _ORDERED_REGISTERS
        if register not in CURRENT_REGISTERS
    )


def read_register_values_with_timeout(
    robot: Any,
    cmd_type: str,
    name: str,
    register: str,
    *,
    timeout_sec: float,
    request_factory: Callable[..., Any] | None = None,
) -> list[Any] | None:
    """Read one register without leaving an unfinished future queued."""
    if request_factory is None:
        from interbotix_xs_msgs.srv import RegisterValues

        request_factory = RegisterValues.Request
    future = robot.core.srv_get_reg.call_async(
        request_factory(cmd_type=cmd_type, name=name, reg=register)
    )
    try:
        result = wait_for_service_future(
            robot.core.get_node(),
            future,
            timeout_sec=timeout_sec,
            operation=(
                f"{robot.core.robot_name} {name} register {register}"
            ),
        )
    except InterbotixServiceTimeout:
        return None
    if result is None:
        return None
    return list(result.values)

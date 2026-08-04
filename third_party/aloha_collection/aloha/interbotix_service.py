"""Bounded service calls for safety-critical Interbotix transitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class InterbotixServiceTimeout(TimeoutError):
    """Raised when an Interbotix service future misses its safety bound."""


class InterbotixServiceError(RuntimeError):
    """Raised when an Interbotix service completes with an exception."""


def wait_for_service_future(
    node: Any,
    future: Any,
    *,
    timeout_sec: float,
    operation: str,
) -> Any:
    """Wait for a service future without permitting an infinite hardware stall."""
    try:
        node.wait_until_future_complete(future, timeout_sec=timeout_sec)
    except BaseException:
        if not future.done():
            future.cancel()
        raise
    if not future.done():
        future.cancel()
        raise InterbotixServiceTimeout(
            f"{operation} timed out after {timeout_sec:.2f}s"
        )
    exception = future.exception()
    if exception is not None:
        raise InterbotixServiceError(f"{operation} failed: {exception}")
    return future.result()


def set_operating_modes_with_timeout(
    robot: Any,
    cmd_type: str,
    name: str,
    mode: str,
    *,
    timeout_sec: float,
    profile_type: str = "velocity",
    profile_velocity: int = 0,
    profile_acceleration: int = 0,
    request_factory: Callable[..., Any] | None = None,
) -> Any:
    """Set a motor mode through a finite Interbotix service future."""
    if request_factory is None:
        from interbotix_xs_msgs.srv import OperatingModes

        request_factory = OperatingModes.Request
    future = robot.core.srv_set_op_modes.call_async(
        request_factory(
            cmd_type=cmd_type,
            name=name,
            mode=mode,
            profile_type=profile_type,
            profile_velocity=profile_velocity,
            profile_acceleration=profile_acceleration,
        )
    )
    return wait_for_service_future(
        robot.core.get_node(),
        future,
        timeout_sec=timeout_sec,
        operation=f"{robot.core.robot_name} {name} operating mode",
    )


def torque_enable_with_timeout(
    robot: Any,
    cmd_type: str,
    name: str,
    enable: bool,
    *,
    timeout_sec: float,
    request_factory: Callable[..., Any] | None = None,
) -> Any:
    """Enable or disable torque through a finite Interbotix service future."""
    if request_factory is None:
        from interbotix_xs_msgs.srv import TorqueEnable

        request_factory = TorqueEnable.Request
    future = robot.core.srv_torque.call_async(
        request_factory(
            cmd_type=cmd_type,
            name=name,
            enable=enable,
        )
    )
    return wait_for_service_future(
        robot.core.get_node(),
        future,
        timeout_sec=timeout_sec,
        operation=f"{robot.core.robot_name} {name} torque",
    )


def set_gravity_compensation_with_timeout(
    robot: Any,
    enabled: bool,
    *,
    timeout_sec: float,
    service_type: Any = None,
    request_factory: Callable[..., Any] | None = None,
) -> Any:
    """Enable or disable gravity compensation without unbounded service waits."""
    if service_type is None or request_factory is None:
        from std_srvs.srv import SetBool

        service_type = SetBool
        request_factory = SetBool.Request
    node = robot.core.get_node()
    service_name = f"{robot.core.ns}/gravity_compensation_enable"
    client = node.create_client(service_type, service_name)
    try:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise InterbotixServiceTimeout(
                f"{robot.core.robot_name} gravity compensation service "
                f"unavailable after {timeout_sec:.2f}s"
            )
        future = client.call_async(request_factory(data=enabled))
        return wait_for_service_future(
            node,
            future,
            timeout_sec=timeout_sec,
            operation=f"{robot.core.robot_name} gravity compensation",
        )
    finally:
        node.destroy_client(client)

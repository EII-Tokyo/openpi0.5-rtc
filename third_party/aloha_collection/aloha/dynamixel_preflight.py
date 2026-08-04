"""Bounded DYNAMIXEL readiness checks performed before ALOHA launch actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import yaml


@dataclass(frozen=True)
class MotorExpectation:
    joint_name: str
    motor_id: int


@dataclass(frozen=True)
class BusExpectation:
    robot_name: str
    port: str
    motors: tuple[MotorExpectation, ...]


class DynamixelConfigError(ValueError):
    """Raised when preflight configuration cannot be validated."""


class DynamixelPreflightError(RuntimeError):
    """Raised when a configured DYNAMIXEL bus fails all probe attempts."""


@dataclass(frozen=True)
class ProbeIssue:
    motor: MotorExpectation | None
    detail: str


@dataclass(frozen=True)
class MotorDiagnostics:
    hardware_error: int
    torque_enable: int
    present_voltage: int
    min_voltage: int
    max_voltage: int
    present_temperature: int
    temperature_limit: int


_HARDWARE_ERROR_NAMES = (
    (0x01, "input_voltage"),
    (0x04, "overheating"),
    (0x08, "motor_encoder"),
    (0x10, "electrical_shock_or_insufficient_power"),
    (0x20, "overload"),
)


def _read_register(
    packet_handler,
    port_handler,
    motor_id: int,
    address: int,
    size: int,
) -> int:
    reader = (
        packet_handler.read1ByteTxRx
        if size == 1
        else packet_handler.read2ByteTxRx
    )
    value, communication_result, packet_error = reader(
        port_handler,
        motor_id,
        address,
    )
    if communication_result != 0:
        raise DynamixelPreflightError(
            f"diagnostic address {address} communication result "
            f"{communication_result}"
        )
    if packet_error not in (0, 0x80):
        raise DynamixelPreflightError(
            f"diagnostic address {address} DYNAMIXEL error {packet_error}"
        )
    return value


def _read_motor_diagnostics(
    packet_handler,
    port_handler,
    motor_id: int,
) -> MotorDiagnostics:
    return MotorDiagnostics(
        temperature_limit=_read_register(
            packet_handler, port_handler, motor_id, 31, 1
        ),
        max_voltage=_read_register(
            packet_handler, port_handler, motor_id, 32, 2
        ),
        min_voltage=_read_register(
            packet_handler, port_handler, motor_id, 34, 2
        ),
        torque_enable=_read_register(
            packet_handler, port_handler, motor_id, 64, 1
        ),
        hardware_error=_read_register(
            packet_handler, port_handler, motor_id, 70, 1
        ),
        present_voltage=_read_register(
            packet_handler, port_handler, motor_id, 144, 2
        ),
        present_temperature=_read_register(
            packet_handler, port_handler, motor_id, 146, 1
        ),
    )


def _format_diagnostics(diagnostics: MotorDiagnostics) -> str:
    error_names = [
        name
        for bit, name in _HARDWARE_ERROR_NAMES
        if diagnostics.hardware_error & bit
    ]
    rendered_errors = ", ".join(error_names) if error_names else "none"
    return (
        f"hardware_error=0x{diagnostics.hardware_error:02X} "
        f"[{rendered_errors}], "
        f"voltage={diagnostics.present_voltage / 10:.1f}V "
        f"limits={diagnostics.min_voltage / 10:.1f}.."
        f"{diagnostics.max_voltage / 10:.1f}V, "
        f"temperature={diagnostics.present_temperature}C "
        f"limit={diagnostics.temperature_limit}C, "
        f"torque={diagnostics.torque_enable}"
    )


def _is_safe_historical_voltage_alert(
    diagnostics: MotorDiagnostics,
) -> bool:
    return (
        diagnostics.hardware_error == 0x01
        and diagnostics.torque_enable == 0
        and diagnostics.min_voltage
        <= diagnostics.present_voltage
        <= diagnostics.max_voltage
        and diagnostics.present_temperature
        < diagnostics.temperature_limit
    )


def _load_yaml_mapping(
    robot_name: str,
    path: str | Path,
    *,
    description: str,
) -> dict[str, Any]:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DynamixelConfigError(
            f"{robot_name}: could not load {description} {config_path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise DynamixelConfigError(
            f"{robot_name}: {description} {config_path} must contain a mapping"
        )
    return loaded


def load_bus_expectation(
    robot_name: str,
    mode_config_path: str | Path,
    motor_config_path: str | Path,
) -> BusExpectation:
    """Load one arm's serial port and ordered motor IDs from YAML."""
    mode_config = _load_yaml_mapping(
        robot_name,
        mode_config_path,
        description="mode config",
    )
    port = mode_config.get("port")
    if not isinstance(port, str) or not port.strip():
        raise DynamixelConfigError(
            f"{robot_name}: mode config port must be a non-empty string"
        )

    motor_config = _load_yaml_mapping(
        robot_name,
        motor_config_path,
        description="motor config",
    )
    configured_motors = motor_config.get("motors")
    if not isinstance(configured_motors, dict) or not configured_motors:
        raise DynamixelConfigError(
            f"{robot_name}: motor config motors must be a non-empty mapping"
        )

    motors: list[MotorExpectation] = []
    seen_ids: set[int] = set()
    for joint_name, settings in configured_motors.items():
        if not isinstance(joint_name, str) or not joint_name.strip():
            raise DynamixelConfigError(
                f"{robot_name}: motor joint name must be a non-empty string"
            )
        if not isinstance(settings, dict) or "ID" not in settings:
            raise DynamixelConfigError(
                f"{robot_name}: motor {joint_name} is missing ID"
            )
        motor_id = settings["ID"]
        if isinstance(motor_id, bool) or not isinstance(motor_id, int):
            raise DynamixelConfigError(
                f"{robot_name}: motor {joint_name} ID must be an integer"
            )
        if not 0 <= motor_id <= 252:
            raise DynamixelConfigError(
                f"{robot_name}: motor {joint_name} ID must be in range 0..252"
            )
        if motor_id in seen_ids:
            raise DynamixelConfigError(
                f"{robot_name}: duplicate motor ID {motor_id}"
            )
        seen_ids.add(motor_id)
        motors.append(MotorExpectation(joint_name=joint_name, motor_id=motor_id))

    return BusExpectation(
        robot_name=robot_name,
        port=port.strip(),
        motors=tuple(motors),
    )


def probe_bus(
    bus: BusExpectation,
    *,
    auto_reboot_input_voltage_alerts: bool = True,
    recovered_motors: set[tuple[str, int]] | None = None,
    reboot_delay: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] = print,
    port_handler_factory=None,
    packet_handler_factory=None,
) -> tuple[ProbeIssue, ...]:
    """Probe every expected motor and diagnose Protocol 2.0 alerts."""
    if reboot_delay < 0:
        raise ValueError("reboot_delay must be non-negative")
    if recovered_motors is None:
        recovered_motors = set()
    if port_handler_factory is None or packet_handler_factory is None:
        from dynamixel_sdk import PacketHandler, PortHandler

        port_handler_factory = port_handler_factory or PortHandler
        packet_handler_factory = packet_handler_factory or PacketHandler

    port_handler = port_handler_factory(bus.port)
    opened = False
    try:
        try:
            opened = bool(port_handler.openPort())
        except Exception as error:
            return (ProbeIssue(None, f"failed to open port: {error}"),)
        if not opened:
            return (ProbeIssue(None, f"failed to open port {bus.port}"),)
        if not port_handler.setBaudRate(1_000_000):
            return (
                ProbeIssue(
                    None,
                    f"failed to set baud rate 1000000 on {bus.port}",
                ),
            )

        packet_handler = packet_handler_factory(2.0)
        issues: list[ProbeIssue] = []
        for motor in bus.motors:
            try:
                _model_number, communication_result, dynamixel_error = (
                    packet_handler.ping(port_handler, motor.motor_id)
                )
            except Exception as error:
                return (ProbeIssue(None, f"ping failed: {error}"),)
            if communication_result != 0:
                issues.append(
                    ProbeIssue(
                        motor,
                        f"communication result {communication_result}",
                    )
                )
            elif dynamixel_error == 0x80:
                try:
                    diagnostics = _read_motor_diagnostics(
                        packet_handler,
                        port_handler,
                        motor.motor_id,
                    )
                except DynamixelPreflightError as error:
                    issues.append(ProbeIssue(motor, str(error)))
                except Exception as error:
                    issues.append(
                        ProbeIssue(
                            motor,
                            f"diagnostic read failed: {error}",
                        )
                    )
                else:
                    rendered_diagnostics = _format_diagnostics(diagnostics)
                    recovery_key = (bus.port, motor.motor_id)
                    if not auto_reboot_input_voltage_alerts:
                        issues.append(
                            ProbeIssue(
                                motor,
                                "automatic input-voltage alert recovery "
                                f"is disabled; {rendered_diagnostics}",
                            )
                        )
                    elif not _is_safe_historical_voltage_alert(diagnostics):
                        issues.append(
                            ProbeIssue(
                                motor,
                                "unsafe hardware alert; "
                                f"{rendered_diagnostics}",
                            )
                        )
                    elif recovery_key in recovered_motors:
                        issues.append(
                            ProbeIssue(
                                motor,
                                "input-voltage alert recurred after "
                                "automatic reboot; "
                                f"{rendered_diagnostics}",
                            )
                        )
                    else:
                        recovered_motors.add(recovery_key)
                        log_fn(
                            f"[DYNAMIXEL preflight] {bus.robot_name} "
                            f"{motor.joint_name} (ID {motor.motor_id}) "
                            "has a safe historical input-voltage alert; "
                            "automatically rebooting once; "
                            f"{rendered_diagnostics}"
                        )
                        try:
                            reboot_communication, reboot_error = (
                                packet_handler.reboot(
                                    port_handler,
                                    motor.motor_id,
                                )
                            )
                        except Exception as error:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    f"automatic reboot failed: {error}",
                                )
                            )
                            continue
                        if reboot_communication != 0:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "automatic reboot communication "
                                    f"result {reboot_communication}",
                                )
                            )
                            continue
                        if reboot_error not in (0, 0x80):
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "automatic reboot DYNAMIXEL error "
                                    f"{reboot_error}",
                                )
                            )
                            continue
                        sleep_fn(reboot_delay)
                        try:
                            (
                                _model_number,
                                recovery_communication,
                                recovery_error,
                            ) = packet_handler.ping(
                                port_handler,
                                motor.motor_id,
                            )
                        except Exception as error:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "post-reboot ping "
                                    f"failed: {error}",
                                )
                            )
                            continue
                        if recovery_communication != 0:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "post-reboot communication result "
                                    f"{recovery_communication}",
                                )
                            )
                            continue
                        if recovery_error != 0:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "post-reboot DYNAMIXEL error "
                                    f"{recovery_error}",
                                )
                            )
                            continue
                        try:
                            post_reboot_torque = _read_register(
                                packet_handler,
                                port_handler,
                                motor.motor_id,
                                64,
                                1,
                            )
                            post_reboot_hardware_error = _read_register(
                                packet_handler,
                                port_handler,
                                motor.motor_id,
                                70,
                                1,
                            )
                        except Exception as error:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "automatic reboot verification "
                                    f"failed: {error}",
                                )
                            )
                            continue
                        if post_reboot_hardware_error != 0:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "post-reboot hardware_error="
                                    f"0x{post_reboot_hardware_error:02X}",
                                )
                            )
                            continue
                        if post_reboot_torque != 0:
                            issues.append(
                                ProbeIssue(
                                    motor,
                                    "post-reboot torque="
                                    f"{post_reboot_torque}",
                                )
                            )
                            continue
                        log_fn(
                            f"[DYNAMIXEL preflight] {bus.robot_name} "
                            f"{motor.joint_name} (ID {motor.motor_id}) "
                            "recovered after one automatic reboot"
                        )
            elif dynamixel_error != 0:
                issues.append(
                    ProbeIssue(
                        motor,
                        f"DYNAMIXEL error {dynamixel_error}",
                    )
                )
        return tuple(issues)
    finally:
        if opened:
            port_handler.closePort()


def _format_issues(issues: Sequence[ProbeIssue]) -> str:
    rendered: list[str] = []
    for issue in issues:
        if issue.motor is None:
            rendered.append(issue.detail)
        else:
            rendered.append(
                f"{issue.motor.joint_name} "
                f"(ID {issue.motor.motor_id}): {issue.detail}"
            )
    return "; ".join(rendered)


def run_preflight(
    buses: Sequence[BusExpectation],
    *,
    attempts: int = 3,
    retry_delay: float = 1.0,
    auto_reboot_input_voltage_alerts: bool = True,
    reboot_delay: float = 1.0,
    probe_fn: Callable[
        [BusExpectation],
        tuple[ProbeIssue, ...],
    ]
    | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] = print,
) -> None:
    """Probe selected buses serially and abort after bounded failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if retry_delay < 0:
        raise ValueError("retry_delay must be non-negative")
    if reboot_delay < 0:
        raise ValueError("reboot_delay must be non-negative")

    recovered_motors: set[tuple[str, int]] = set()
    if probe_fn is None:
        def active_probe(bus: BusExpectation) -> tuple[ProbeIssue, ...]:
            return probe_bus(
                bus,
                auto_reboot_input_voltage_alerts=(
                    auto_reboot_input_voltage_alerts
                ),
                recovered_motors=recovered_motors,
                reboot_delay=reboot_delay,
                sleep_fn=sleep_fn,
                log_fn=log_fn,
            )
    else:
        active_probe = probe_fn

    for bus in buses:
        last_issues: tuple[ProbeIssue, ...] = ()
        for attempt in range(1, attempts + 1):
            last_issues = tuple(active_probe(bus))
            if not last_issues:
                log_fn(f"[DYNAMIXEL preflight] {bus.robot_name} passed")
                break
            log_fn(
                f"[DYNAMIXEL preflight] {bus.robot_name} "
                f"attempt {attempt}/{attempts} failed on {bus.port}: "
                f"{_format_issues(last_issues)}"
            )
            if attempt < attempts:
                sleep_fn(retry_delay)
        else:
            raise DynamixelPreflightError(
                f"{bus.robot_name} on {bus.port} failed at "
                f"attempt {attempts}/{attempts}: "
                f"{_format_issues(last_issues)}"
            )

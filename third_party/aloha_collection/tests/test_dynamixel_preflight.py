from pathlib import Path

import pytest
import yaml

import aloha.dynamixel_preflight as dynamixel_preflight_module
from aloha.dynamixel_preflight import (
    BusExpectation,
    DynamixelConfigError,
    DynamixelPreflightError,
    MotorExpectation,
    ProbeIssue,
    load_bus_expectation,
    probe_bus,
    run_preflight,
)


def write_yaml(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_load_bus_expectation_reads_port_joint_names_and_ids(tmp_path):
    mode_path = write_yaml(
        tmp_path / "mode.yaml",
        {"port": "/dev/ttyTEST0"},
    )
    motor_path = write_yaml(
        tmp_path / "motors.yaml",
        {
            "motors": {
                "waist": {"ID": 1},
                "gripper": {"ID": 9},
            }
        },
    )

    bus = load_bus_expectation("leader_left", mode_path, motor_path)

    assert bus == BusExpectation(
        robot_name="leader_left",
        port="/dev/ttyTEST0",
        motors=(
            MotorExpectation("waist", 1),
            MotorExpectation("gripper", 9),
        ),
    )


@pytest.mark.parametrize(
    ("mode_config", "motor_config", "message"),
    [
        ({}, {"motors": {"waist": {"ID": 1}}}, "port"),
        ({"port": "  "}, {"motors": {"waist": {"ID": 1}}}, "port"),
        ({"port": "/dev/test"}, {}, "motors"),
        ({"port": "/dev/test"}, {"motors": {}}, "motors"),
        (
            {"port": "/dev/test"},
            {"motors": {"waist": {"ID": 1}, "shoulder": {"ID": 1}}},
            "duplicate",
        ),
        (
            {"port": "/dev/test"},
            {"motors": {"waist": {"ID": True}}},
            "integer",
        ),
        (
            {"port": "/dev/test"},
            {"motors": {"waist": {"ID": -1}}},
            "0..252",
        ),
        (
            {"port": "/dev/test"},
            {"motors": {"waist": {"ID": 253}}},
            "0..252",
        ),
        (
            {"port": "/dev/test"},
            {"motors": {"waist": {}}},
            "ID",
        ),
    ],
)
def test_load_bus_expectation_rejects_invalid_configuration(
    tmp_path,
    mode_config,
    motor_config,
    message,
):
    mode_path = write_yaml(tmp_path / "mode.yaml", mode_config)
    motor_path = write_yaml(tmp_path / "motors.yaml", motor_config)

    with pytest.raises(DynamixelConfigError) as exc_info:
        load_bus_expectation("follower_left", mode_path, motor_path)

    rendered = str(exc_info.value)
    assert "follower_left" in rendered
    assert message in rendered


class FakePort:
    def __init__(self, *, opens=True, sets_baud=True):
        self.opens = opens
        self.sets_baud = sets_baud
        self.open_calls = 0
        self.baud_calls = []
        self.closed = 0

    def openPort(self):
        self.open_calls += 1
        return self.opens

    def setBaudRate(self, baud):
        self.baud_calls.append(baud)
        return self.sets_baud

    def closePort(self):
        self.closed += 1


class FakePacket:
    def __init__(
        self,
        responses=None,
        register_values=None,
        post_reboot_register_values=None,
        reboot_results=None,
    ):
        self.responses = responses or {}
        self.register_values = register_values or {}
        self.post_reboot_register_values = (
            post_reboot_register_values or {}
        )
        self.reboot_results = reboot_results or {}
        self.calls = []
        self.read_calls = []
        self.reboot_calls = []
        self.rebooted = set()

    def ping(self, port, motor_id):
        self.calls.append((port, motor_id))
        response = self.responses.get(motor_id, (1000 + motor_id, 0, 0))
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def _read(self, size, port, motor_id, address):
        self.read_calls.append((size, port, motor_id, address))
        key = (motor_id, address)
        if motor_id in self.rebooted and key in self.post_reboot_register_values:
            response = self.post_reboot_register_values[key]
        else:
            response = self.register_values[key]
        if isinstance(response, Exception):
            raise response
        return response

    def read1ByteTxRx(self, port, motor_id, address):
        return self._read(1, port, motor_id, address)

    def read2ByteTxRx(self, port, motor_id, address):
        return self._read(2, port, motor_id, address)

    def reboot(self, port, motor_id):
        self.reboot_calls.append(motor_id)
        result = self.reboot_results.get(motor_id, (0, 0))
        if result[0] == 0:
            self.rebooted.add(motor_id)
        return result


def sample_bus(robot_name="leader_left", port="/dev/ttyTEST0"):
    return BusExpectation(
        robot_name=robot_name,
        port=port,
        motors=(
            MotorExpectation("waist", 1),
            MotorExpectation("gripper", 9),
        ),
    )


def safe_voltage_alert_registers(motor_id):
    return {
        (motor_id, 31): (80, 0, 128),
        (motor_id, 32): (160, 0, 128),
        (motor_id, 34): (95, 0, 128),
        (motor_id, 64): (0, 0, 128),
        (motor_id, 70): (1, 0, 128),
        (motor_id, 144): (117, 0, 128),
        (motor_id, 146): (32, 0, 128),
    }


def test_probe_bus_pings_every_motor_with_protocol_2_and_closes_port():
    port = FakePort()
    packet = FakePacket()
    protocols = []

    issues = probe_bus(
        sample_bus(),
        port_handler_factory=lambda configured_port: (
            port if configured_port == "/dev/ttyTEST0" else None
        ),
        packet_handler_factory=lambda protocol: (
            protocols.append(protocol) or packet
        ),
    )

    assert issues == ()
    assert port.open_calls == 1
    assert port.baud_calls == [1_000_000]
    assert protocols == [2.0]
    assert packet.calls == [(port, 1), (port, 9)]
    assert port.closed == 1


@pytest.mark.parametrize(
    ("port", "packet", "message", "expected_close_calls"),
    [
        (FakePort(opens=False), FakePacket(), "open", 0),
        (FakePort(sets_baud=False), FakePacket(), "baud", 1),
        (
            FakePort(),
            FakePacket({1: RuntimeError("packet exploded")}),
            "packet exploded",
            1,
        ),
    ],
)
def test_probe_bus_closes_only_open_ports_on_transport_failures(
    port,
    packet,
    message,
    expected_close_calls,
):
    issues = probe_bus(
        sample_bus(),
        port_handler_factory=lambda _configured_port: port,
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert issues[0].motor is None
    assert message in issues[0].detail
    assert port.closed == expected_close_calls


def test_probe_bus_preserves_open_exception_without_unsafe_close():
    class FailedOpenPort(FakePort):
        def openPort(self):
            raise RuntimeError("serial device missing")

        def closePort(self):
            raise AssertionError("unopened SDK port must not be closed")

    issues = probe_bus(
        sample_bus(),
        port_handler_factory=lambda _configured_port: FailedOpenPort(),
        packet_handler_factory=lambda _protocol: FakePacket(),
    )

    assert issues == (ProbeIssue(None, "failed to open port: serial device missing"),)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ((0, 7, 0), "communication result 7"),
        ((0, 0, 4), "DYNAMIXEL error 4"),
    ],
)
def test_probe_bus_reports_exact_missing_motor(response, message):
    port = FakePort()
    packet = FakePacket({9: response})

    issues = probe_bus(
        sample_bus(),
        port_handler_factory=lambda _configured_port: port,
        packet_handler_factory=lambda _protocol: packet,
    )

    assert issues == (
        ProbeIssue(
            motor=MotorExpectation("gripper", 9),
            detail=message,
        ),
    )
    assert port.closed == 1


def test_probe_bus_reads_safety_diagnostics_for_alert():
    port = FakePort()
    packet = FakePacket(
        responses={9: (1009, 0, 128)},
        register_values=safe_voltage_alert_registers(9),
    )

    issues = probe_bus(
        sample_bus(),
        auto_reboot_input_voltage_alerts=False,
        port_handler_factory=lambda _configured_port: port,
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert issues[0].motor == MotorExpectation("gripper", 9)
    assert "input_voltage" in issues[0].detail
    assert "11.7V" in issues[0].detail
    assert {call[3] for call in packet.read_calls} == {
        31,
        32,
        34,
        64,
        70,
        144,
        146,
    }
    assert packet.reboot_calls == []


def test_probe_bus_reboots_one_safe_historical_voltage_alert():
    port = FakePort()
    packet = FakePacket(
        responses={
            9: [
                (1009, 0, 128),
                (1009, 0, 0),
            ]
        },
        register_values=safe_voltage_alert_registers(9),
        post_reboot_register_values={
            (9, 64): (0, 0, 0),
            (9, 70): (0, 0, 0),
        },
        reboot_results={9: (0, 128)},
    )
    recovered = set()
    sleeps = []
    messages = []

    issues = probe_bus(
        sample_bus(),
        recovered_motors=recovered,
        reboot_delay=1.0,
        sleep_fn=sleeps.append,
        log_fn=messages.append,
        port_handler_factory=lambda _configured_port: port,
        packet_handler_factory=lambda _protocol: packet,
    )

    assert issues == ()
    assert packet.reboot_calls == [9]
    assert recovered == {("/dev/ttyTEST0", 9)}
    assert sleeps == [1.0]
    assert any("automatically rebooting" in message for message in messages)
    assert any("recovered" in message for message in messages)
    assert port.closed == 1


@pytest.mark.parametrize(
    ("address", "value", "message"),
    [
        (64, 1, "torque=1"),
        (70, 0x04, "overheating"),
        (70, 0x21, "overload"),
        (144, 90, "voltage=9.0V"),
        (146, 80, "temperature=80C"),
    ],
)
def test_probe_bus_does_not_reboot_unsafe_alerts(
    address,
    value,
    message,
):
    registers = safe_voltage_alert_registers(9)
    registers[(9, address)] = (value, 0, 128)
    packet = FakePacket(
        responses={9: (1009, 0, 128)},
        register_values=registers,
    )

    issues = probe_bus(
        sample_bus(),
        recovered_motors=set(),
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert packet.reboot_calls == []
    assert len(issues) == 1
    assert message in issues[0].detail


def test_probe_bus_reports_diagnostic_read_exception():
    registers = safe_voltage_alert_registers(9)
    registers[(9, 70)] = RuntimeError("register read exploded")
    packet = FakePacket(
        responses={9: (1009, 0, 128)},
        register_values=registers,
    )

    issues = probe_bus(
        sample_bus(),
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert issues == (
        ProbeIssue(
            MotorExpectation("gripper", 9),
            "diagnostic read failed: register read exploded",
        ),
    )
    assert packet.reboot_calls == []


@pytest.mark.parametrize(
    ("reboot_result", "message"),
    [
        ((7, 0), "reboot communication result 7"),
        ((0, 4), "reboot DYNAMIXEL error 4"),
    ],
)
def test_probe_bus_reports_reboot_failures(reboot_result, message):
    packet = FakePacket(
        responses={9: (1009, 0, 128)},
        register_values=safe_voltage_alert_registers(9),
        reboot_results={9: reboot_result},
    )
    recovered = set()

    issues = probe_bus(
        sample_bus(),
        recovered_motors=recovered,
        sleep_fn=lambda _delay: None,
        log_fn=lambda _message: None,
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert message in issues[0].detail
    assert packet.reboot_calls == [9]
    assert recovered == {("/dev/ttyTEST0", 9)}


@pytest.mark.parametrize(
    (
        "post_ping",
        "post_torque",
        "post_hardware_error",
        "message",
    ),
    [
        ((1009, 7, 0), 0, 0, "post-reboot communication result 7"),
        ((1009, 0, 128), 0, 1, "post-reboot DYNAMIXEL error 128"),
        ((1009, 0, 0), 0, 1, "post-reboot hardware_error=0x01"),
        ((1009, 0, 0), 1, 0, "post-reboot torque=1"),
    ],
)
def test_probe_bus_reports_post_reboot_verification_failures(
    post_ping,
    post_torque,
    post_hardware_error,
    message,
):
    packet = FakePacket(
        responses={
            9: [
                (1009, 0, 128),
                post_ping,
            ]
        },
        register_values=safe_voltage_alert_registers(9),
        post_reboot_register_values={
            (9, 64): (post_torque, 0, post_ping[2]),
            (9, 70): (post_hardware_error, 0, post_ping[2]),
        },
        reboot_results={9: (0, 128)},
    )

    issues = probe_bus(
        sample_bus(),
        recovered_motors=set(),
        sleep_fn=lambda _delay: None,
        log_fn=lambda _message: None,
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert message in issues[0].detail
    assert packet.reboot_calls == [9]


def test_probe_bus_stops_post_reboot_checks_after_failed_ping():
    packet = FakePacket(
        responses={
            9: [
                (1009, 0, 128),
                (0, 7, 0),
            ]
        },
        register_values=safe_voltage_alert_registers(9),
        reboot_results={9: (0, 128)},
    )

    issues = probe_bus(
        sample_bus(),
        recovered_motors=set(),
        sleep_fn=lambda _delay: None,
        log_fn=lambda _message: None,
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert "post-reboot communication result 7" in issues[0].detail
    assert len(packet.read_calls) == 7


def test_probe_bus_does_not_reboot_alert_that_recurred_in_session():
    packet = FakePacket(
        responses={9: (1009, 0, 128)},
        register_values=safe_voltage_alert_registers(9),
    )
    recovered = {("/dev/ttyTEST0", 9)}

    issues = probe_bus(
        sample_bus(),
        recovered_motors=recovered,
        port_handler_factory=lambda _configured_port: FakePort(),
        packet_handler_factory=lambda _protocol: packet,
    )

    assert len(issues) == 1
    assert "recurred after automatic reboot" in issues[0].detail
    assert packet.reboot_calls == []


def test_run_preflight_checks_buses_in_order_without_unneeded_sleep():
    buses = (
        sample_bus("leader_left", "/dev/leader"),
        sample_bus("follower_left", "/dev/follower"),
    )
    calls = []
    sleeps = []

    run_preflight(
        buses,
        probe_fn=lambda bus: calls.append(bus.robot_name) or (),
        sleep_fn=sleeps.append,
        log_fn=lambda _message: None,
    )

    assert calls == ["leader_left", "follower_left"]
    assert sleeps == []


def test_run_preflight_retries_twice_then_continues_after_third_success():
    bus = sample_bus()
    issue = ProbeIssue(MotorExpectation("gripper", 9), "no status packet")
    responses = [(issue,), (issue,), ()]
    sleeps = []
    messages = []

    run_preflight(
        (bus,),
        attempts=3,
        retry_delay=1.0,
        probe_fn=lambda _bus: responses.pop(0),
        sleep_fn=sleeps.append,
        log_fn=messages.append,
    )

    assert responses == []
    assert sleeps == [1.0, 1.0]
    assert any("attempt 2/3" in message for message in messages)
    assert any("leader_left passed" in message for message in messages)


def test_run_preflight_aborts_after_final_failure_without_later_bus():
    failed_bus = sample_bus("follower_left", "/dev/follower-left")
    later_bus = sample_bus("follower_right", "/dev/follower-right")
    issue = ProbeIssue(MotorExpectation("gripper", 9), "no status packet")
    calls = []

    with pytest.raises(DynamixelPreflightError) as exc_info:
        run_preflight(
            (failed_bus, later_bus),
            attempts=3,
            retry_delay=0.25,
            probe_fn=lambda bus: calls.append(bus.robot_name) or (issue,),
            sleep_fn=lambda _delay: None,
            log_fn=lambda _message: None,
        )

    assert calls == ["follower_left", "follower_left", "follower_left"]
    rendered = str(exc_info.value)
    assert "follower_left" in rendered
    assert "/dev/follower-left" in rendered
    assert "gripper" in rendered
    assert "ID 9" in rendered
    assert "3/3" in rendered


def test_run_preflight_reuses_one_recovery_set_across_attempts(monkeypatch):
    bus = sample_bus("follower_left", "/dev/follower-left")
    issue = ProbeIssue(bus.motors[-1], "alert recurred")
    recovered_sets = []

    def fake_default_probe(probed_bus, **kwargs):
        recovered_motors = kwargs["recovered_motors"]
        recovered_sets.append(recovered_motors)
        recovered_motors.add((probed_bus.port, 9))
        return (issue,)

    monkeypatch.setattr(
        dynamixel_preflight_module,
        "probe_bus",
        fake_default_probe,
    )

    with pytest.raises(DynamixelPreflightError):
        run_preflight(
            (bus,),
            attempts=3,
            retry_delay=0,
            probe_fn=None,
            sleep_fn=lambda _delay: None,
            log_fn=lambda _message: None,
        )

    assert len(recovered_sets) == 3
    assert len({id(state) for state in recovered_sets}) == 1
    assert recovered_sets[0] == {("/dev/follower-left", 9)}


@pytest.mark.parametrize(
    ("attempts", "retry_delay", "message"),
    [(0, 1.0, "attempts"), (1, -0.1, "retry_delay")],
)
def test_run_preflight_rejects_invalid_retry_configuration(
    attempts,
    retry_delay,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_preflight(
            (sample_bus(),),
            attempts=attempts,
            retry_delay=retry_delay,
        )


def test_preflight_rejects_negative_reboot_delay():
    with pytest.raises(ValueError, match="reboot_delay"):
        run_preflight(
            (sample_bus(),),
            reboot_delay=-0.1,
        )

    with pytest.raises(ValueError, match="reboot_delay"):
        probe_bus(
            sample_bus(),
            reboot_delay=-0.1,
        )

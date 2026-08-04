import io
from dataclasses import dataclass

from aloha.foot_pedal_relay import (
    EV_KEY,
    KEY_B,
    ForwardResult,
    PedalEventFilter,
    PedalRelay,
    PersistentSshTransport,
    build_ssh_command,
    deduplicate_device_paths,
    open_input_devices,
)


@dataclass
class Event:
    type: int
    code: int
    value: int


class FakeTransport:
    def __init__(self, connected=True):
        self.connected = connected
        self.writes = []

    def send(self, command, now):
        if not self.connected:
            return False
        self.writes.append((command, now))
        return True


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


def key_event(code=KEY_B, value=1, event_type=EV_KEY):
    return Event(type=event_type, code=code, value=value)


def test_filter_accepts_only_debounced_key_b_press():
    event_filter = PedalEventFilter(debounce_seconds=0.4)

    assert event_filter.accept(key_event(), now=1.0)
    assert not event_filter.accept(key_event(value=0), now=1.1)
    assert not event_filter.accept(key_event(value=2), now=1.2)
    assert not event_filter.accept(key_event(), now=1.3)
    assert event_filter.accept(key_event(), now=1.5)


def test_filter_rejects_other_key_and_event_type():
    event_filter = PedalEventFilter(debounce_seconds=0.4)

    assert not event_filter.accept(key_event(code=30), now=1.0)
    assert not event_filter.accept(key_event(event_type=2), now=1.0)


def test_filter_accepts_configured_mouse_button_code():
    event_filter = PedalEventFilter(debounce_seconds=0.4, event_code=272)

    assert event_filter.accept(key_event(code=272), now=1.0)
    assert not event_filter.accept(key_event(code=KEY_B), now=1.5)


def test_relay_forwards_configured_mouse_button_as_b():
    transport = FakeTransport()
    relay = PedalRelay(transport, event_code=272)

    assert relay.process_event(key_event(code=272), now=1.0) is ForwardResult.SENT
    assert transport.writes == [("b", 1.0)]


def test_disconnected_transport_drops_instead_of_replaying():
    transport = FakeTransport(connected=False)
    relay = PedalRelay(transport, debounce_seconds=0.4)

    assert relay.process_event(key_event(), now=1.0) is ForwardResult.DROPPED
    transport.connected = True
    assert transport.writes == []
    assert relay.process_event(key_event(), now=1.5) is ForwardResult.SENT
    assert transport.writes == [("b", 1.5)]


def test_ignored_event_does_not_contact_transport():
    transport = FakeTransport()
    relay = PedalRelay(transport)

    assert relay.process_event(key_event(value=0), now=1.0) is None
    assert transport.writes == []


def test_ssh_command_is_fixed_argument_list():
    command = build_ssh_command("aloha")

    assert command[:4] == ["ssh", "-T", "-o", "BatchMode=yes"]
    assert command[-7:] == [
        "docker",
        "exec",
        "-i",
        "aloha2-collect",
        "python3",
        "/root/interbotix_ws/src/aloha/scripts/send_record_trigger.py",
        "--stream",
    ]


def test_transport_spawns_without_shell_and_sends_current_line():
    calls = []
    process = FakeProcess()

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    transport = PersistentSshTransport(
        build_ssh_command("aloha"),
        popen_factory=popen,
    )

    assert transport.ensure_connected(now=0.0)
    assert transport.send("b", now=1.0)
    assert process.stdin.getvalue() == "b\n"
    command, kwargs = calls[0]
    assert isinstance(command, list)
    assert kwargs.get("shell", False) is False


def test_transport_uses_bounded_backoff_after_spawn_failure():
    attempts = []

    def failing_popen(command, **kwargs):
        attempts.append(command)
        raise OSError("ssh unavailable")

    transport = PersistentSshTransport(
        build_ssh_command("aloha"),
        popen_factory=failing_popen,
        initial_backoff=1.0,
        maximum_backoff=4.0,
    )

    assert not transport.ensure_connected(now=0.0)
    assert not transport.ensure_connected(now=0.5)
    assert not transport.ensure_connected(now=1.0)
    assert not transport.ensure_connected(now=2.9)
    assert not transport.ensure_connected(now=3.0)
    assert len(attempts) == 3


def test_broken_pipe_drops_press_and_closes_process():
    process = FakeProcess()

    class BrokenInput:
        def write(self, value):
            raise BrokenPipeError

        def flush(self):
            raise AssertionError("flush must not run")

    process.stdin = BrokenInput()
    transport = PersistentSshTransport(
        build_ssh_command("aloha"),
        popen_factory=lambda *args, **kwargs: process,
    )

    assert transport.ensure_connected(now=0.0)
    assert not transport.send("b", now=1.0)
    assert process.returncode == -15


def test_device_paths_are_deduplicated_by_real_event_node(tmp_path):
    event3 = tmp_path / "event3"
    event7 = tmp_path / "event7"
    event3.touch()
    event7.touch()
    usb3 = tmp_path / "usb-port7-event-kbd"
    usbv2_3 = tmp_path / "usbv2-port7-event-kbd"
    usb7 = tmp_path / "usb-port8-event-kbd"
    usb3.symlink_to(event3)
    usbv2_3.symlink_to(event3)
    usb7.symlink_to(event7)

    assert deduplicate_device_paths([usbv2_3, usb7, usb3]) == [usb3, usb7]


def test_open_input_devices_supports_unhashable_handles(tmp_path):
    class UnhashableDevice:
        __hash__ = None

        def __init__(self, path):
            self.path = path

    paths = [tmp_path / "event3", tmp_path / "event7"]

    pairs = open_input_devices(paths, UnhashableDevice)

    assert [(pair[0].path, pair[1]) for pair in pairs] == [
        (str(paths[0]), paths[0]),
        (str(paths[1]), paths[1]),
    ]

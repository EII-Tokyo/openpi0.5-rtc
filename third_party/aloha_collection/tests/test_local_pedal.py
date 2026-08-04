import os
from pathlib import Path
import struct
import time

import pytest

import aloha.local_pedal as local_pedal
from aloha.local_pedal import (
    FootPedalEventDecoder,
    FootPedalListener,
    FootPedalUnavailable,
)
from aloha.record_trigger import (
    RecordingEvents,
    RecordingTriggerController,
    TriggerResult,
)


INPUT_EVENT = struct.Struct("llHHI")
EV_KEY = 1
KEY_B = 48
ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "scripts/record_episodes_copy.py"


def event_bytes(
    event_type=EV_KEY,
    code=KEY_B,
    value=1,
    *,
    timestamp=0.0,
):
    seconds = int(timestamp)
    microseconds = int(round((timestamp - seconds) * 1_000_000))
    return INPUT_EVENT.pack(
        seconds,
        microseconds,
        event_type,
        code,
        value,
    )


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def make_pipe_listener(read_fd, on_press, **overrides):
    options = {
        "open_device": lambda _path, _flags: read_fd,
        "close_device": lambda _fd: None,
        "grab_device": lambda _fd: None,
        "release_device": lambda _fd: None,
        "poll_interval": 0.01,
    }
    options.update(overrides)
    return FootPedalListener("/stable/pedal", on_press, **options)


def test_decoder_accepts_only_key_b_press():
    decoder = FootPedalEventDecoder()

    assert decoder.feed(event_bytes()) == ["b"]
    assert decoder.feed(event_bytes(value=1, timestamp=0.1)) == []
    assert decoder.feed(event_bytes(value=0)) == []
    assert decoder.feed(event_bytes(value=2)) == []
    assert decoder.feed(event_bytes(code=30)) == []
    assert decoder.feed(event_bytes(event_type=2)) == []
    assert decoder.feed(
        event_bytes(value=1, timestamp=1.1)
    ) == ["b"]


def test_decoder_reassembles_fragmented_input_event():
    decoder = FootPedalEventDecoder()
    payload = event_bytes()

    assert decoder.feed(payload[:7]) == []
    assert decoder.feed(payload[7:]) == ["b"]


def test_decoder_handles_multiple_records_per_read():
    decoder = FootPedalEventDecoder()

    assert decoder.feed(
        event_bytes(value=1, timestamp=1.0)
        + event_bytes(value=0, timestamp=1.1)
        + event_bytes(value=1, timestamp=2.1)
    ) == ["b", "b"]


def test_decoder_requires_release_before_accepting_another_press():
    decoder = FootPedalEventDecoder(debounce_seconds=0.0)

    assert decoder.feed(event_bytes(value=1, timestamp=1.0)) == ["b"]
    assert decoder.feed(event_bytes(value=1, timestamp=1.1)) == []
    assert decoder.feed(event_bytes(value=0, timestamp=1.2)) == []
    assert decoder.feed(event_bytes(value=1, timestamp=1.3)) == ["b"]


def test_decoder_drops_second_macro_inside_debounce_window():
    decoder = FootPedalEventDecoder(debounce_seconds=1.0)

    first = event_bytes(value=1, timestamp=10.0)
    release = event_bytes(value=0, timestamp=10.1)
    early_second = event_bytes(value=1, timestamp=10.2)
    early_release = event_bytes(value=0, timestamp=10.3)
    later_second = event_bytes(value=1, timestamp=11.1)

    assert decoder.feed(
        first + release + early_second + early_release
    ) == ["b"]
    assert decoder.feed(later_second) == ["b"]


def test_decoder_ignores_key_repeat_without_rearming():
    decoder = FootPedalEventDecoder(debounce_seconds=0.0)

    assert decoder.feed(
        event_bytes(value=1, timestamp=1.0)
        + event_bytes(value=2, timestamp=1.1)
        + event_bytes(value=0, timestamp=1.2)
        + event_bytes(value=1, timestamp=1.3)
    ) == ["b", "b"]


def test_decoder_rejects_negative_debounce():
    with pytest.raises(ValueError, match="non-negative"):
        FootPedalEventDecoder(debounce_seconds=-0.1)


def test_listener_delivers_one_callback_for_one_press():
    read_fd, write_fd = os.pipe()
    received = []
    listener = make_pipe_listener(
        read_fd,
        lambda: received.append("b"),
    )
    try:
        with listener:
            os.write(write_fd, event_bytes(value=1) + event_bytes(value=0))
            wait_until(lambda: received == ["b"])
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_listener_drops_a_second_macro_inside_debounce_window():
    read_fd, write_fd = os.pipe()
    received = []
    logs = []
    listener = make_pipe_listener(
        read_fd,
        lambda: received.append("b"),
        debounce_seconds=1.0,
        logger=logs.append,
    )
    try:
        listener.start()
        os.write(
            write_fd,
            event_bytes(value=1, timestamp=10.0)
            + event_bytes(value=0, timestamp=10.1)
            + event_bytes(value=1, timestamp=10.2)
            + event_bytes(value=0, timestamp=10.3),
        )
        wait_until(
            lambda: any("丢弃防抖窗口" in item for item in logs)
        )
        assert received == ["b"]
    finally:
        listener.close()
        os.close(read_fd)
        os.close(write_fd)


def test_listener_opens_and_grabs_synchronously_before_start_returns():
    calls = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda path, flags: calls.append(("open", path, flags)) or 17,
        grab_device=lambda fd: calls.append(("grab", fd)),
        release_device=lambda fd: calls.append(("release", fd)),
        close_device=lambda fd: calls.append(("close", fd)),
        wait_readable=lambda *_args: ([], [], []),
        poll_interval=0.01,
    )

    listener.start()
    try:
        assert calls[0][0] == "open"
        assert calls[1] == ("grab", 17)
    finally:
        listener.close()

    assert calls[-2:] == [("release", 17), ("close", 17)]


def test_listener_open_failure_is_fail_closed_before_thread_start():
    listener = FootPedalListener(
        "/missing/pedal",
        lambda: None,
        open_device=lambda _path, _flags: (_ for _ in ()).throw(
            FileNotFoundError("missing")
        ),
        grab_device=lambda _fd: pytest.fail("grab must not run"),
    )

    with pytest.raises(FootPedalUnavailable, match="/missing/pedal"):
        listener.start()

    assert listener._thread is None


def test_listener_grab_failure_closes_descriptor_and_fails_closed():
    closes = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 19,
        grab_device=lambda _fd: (_ for _ in ()).throw(
            PermissionError("busy")
        ),
        close_device=closes.append,
    )

    with pytest.raises(FootPedalUnavailable, match="exclusive"):
        listener.start()

    assert closes == [19]
    assert listener._thread is None


def test_listener_close_releases_once_and_is_idempotent():
    calls = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 23,
        grab_device=lambda _fd: None,
        release_device=lambda fd: calls.append(("release", fd)),
        close_device=lambda fd: calls.append(("close", fd)),
        wait_readable=lambda *_args: ([], [], []),
        poll_interval=0.01,
    )

    listener.start()
    listener.close()
    listener.close()

    assert calls == [("release", 23), ("close", 23)]


def test_listener_close_still_closes_when_explicit_release_fails():
    closes = []
    logs = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 29,
        grab_device=lambda _fd: None,
        release_device=lambda _fd: (_ for _ in ()).throw(
            OSError("release failed")
        ),
        close_device=closes.append,
        wait_readable=lambda *_args: ([], [], []),
        poll_interval=0.01,
        logger=logs.append,
    )

    listener.start()
    listener.close()

    assert closes == [29]
    assert any("release failed" in message for message in logs)


def test_listener_thread_start_failure_releases_and_closes(monkeypatch):
    calls = []

    class FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(local_pedal.threading, "Thread", FailingThread)
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 43,
        grab_device=lambda fd: calls.append(("grab", fd)),
        release_device=lambda fd: calls.append(("release", fd)),
        close_device=lambda fd: calls.append(("close", fd)),
        logger=lambda _message: None,
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        listener.start()

    assert calls == [("grab", 43), ("release", 43), ("close", 43)]
    assert listener._descriptor is None
    assert listener._thread is None


def test_listener_close_is_bounded_while_device_is_idle():
    read_fd, write_fd = os.pipe()
    listener = make_pipe_listener(
        read_fd,
        lambda: None,
    )
    try:
        listener.start()
        started = time.monotonic()
        listener.close()
        assert time.monotonic() - started < 1.0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_two_pedal_presses_share_recording_trigger_controller():
    read_fd, write_fd = os.pipe()
    controller = RecordingTriggerController(
        RecordingEvents.create(),
        start_trigger="b",
    )
    assert controller.complete_preparation() is True
    results = []
    listener = make_pipe_listener(
        read_fd,
        lambda: results.append(controller.handle_b()),
    )
    try:
        listener.start()
        os.write(
            write_fd,
            event_bytes(value=1, timestamp=10.0)
            + event_bytes(value=0, timestamp=10.1),
        )
        wait_until(lambda: results == [TriggerResult.STARTED])
        assert controller.mark_sample_recorded() is True
        os.write(
            write_fd,
            event_bytes(value=1, timestamp=11.1)
            + event_bytes(value=0, timestamp=11.2),
        )
        wait_until(
            lambda: results == [TriggerResult.STARTED, TriggerResult.STOPPED]
        )
    finally:
        listener.close()
        os.close(read_fd)
        os.close(write_fd)


def test_recorder_owns_local_pedal_listener_for_b_mode():
    source = RECORDER.read_text(encoding="utf-8")
    failure_handler = source.split(
        "def _handle_pedal_failure(",
        1,
    )[1].split("def _handle_b_trigger(", 1)[0]

    assert "from contextlib import nullcontext" in source
    assert "DEFAULT_PEDAL_PATH" in source
    assert "FootPedalListener" in source
    assert 'args.get("pedal_device", DEFAULT_PEDAL_PATH)' in source
    assert 'if _START_RECORDING_TRIGGER == "b"' in source
    assert '_handle_b_trigger("foot-pedal")' in source
    assert "with TriggerSocketServer" in source
    assert "pedal_context" in source
    assert '"--pedal-device"' in source
    assert "def _handle_pedal_failure(" in source
    assert "on_failure=_handle_pedal_failure" in source
    assert 'coordinator.request_no_save(source="foot-pedal-failure")' in source
    assert "run_keyboard_listener(" in source
    assert "print(" not in failure_handler


def test_recorder_wires_pedal_debounce_option():
    source = RECORDER.read_text(encoding="utf-8")

    assert '"--pedal-debounce-seconds"' in source
    assert 'args.get("pedal_debounce_seconds", 1.0)' in source
    assert "debounce_seconds=pedal_debounce_seconds" in source


def test_listener_runtime_eof_reports_failure_once_without_reopening():
    failures = []
    opens = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda path, _flags: opens.append(path) or 31,
        grab_device=lambda _fd: None,
        release_device=lambda _fd: None,
        close_device=lambda _fd: None,
        read_device=lambda _fd, _size: b"",
        wait_readable=lambda read, _write, _errors, _timeout: (read, [], []),
        on_failure=failures.append,
        poll_interval=0.01,
    )

    listener.start()
    try:
        wait_until(lambda: len(failures) == 1)
        time.sleep(0.03)
    finally:
        listener.close()

    assert len(failures) == 1
    assert "ended" in str(failures[0]).lower()
    assert opens == ["/stable/pedal"]


def test_listener_runtime_read_error_reports_failure_once():
    failures = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 37,
        grab_device=lambda _fd: None,
        release_device=lambda _fd: None,
        close_device=lambda _fd: None,
        read_device=lambda _fd, _size: (_ for _ in ()).throw(
            OSError("disconnected")
        ),
        wait_readable=lambda read, _write, _errors, _timeout: (read, [], []),
        on_failure=failures.append,
        poll_interval=0.01,
    )

    listener.start()
    try:
        wait_until(lambda: len(failures) == 1)
    finally:
        listener.close()

    assert len(failures) == 1
    assert "disconnected" in str(failures[0])


def test_listener_runtime_wait_error_reports_failure_once():
    failures = []
    listener = FootPedalListener(
        "/stable/pedal",
        lambda: None,
        open_device=lambda _path, _flags: 41,
        grab_device=lambda _fd: None,
        release_device=lambda _fd: None,
        close_device=lambda _fd: None,
        wait_readable=lambda *_args: (_ for _ in ()).throw(
            OSError("poll failed")
        ),
        on_failure=failures.append,
        poll_interval=0.01,
    )

    listener.start()
    try:
        wait_until(lambda: len(failures) == 1)
    finally:
        listener.close()

    assert len(failures) == 1
    assert "poll failed" in str(failures[0])

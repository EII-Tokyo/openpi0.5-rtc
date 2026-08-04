import os
import termios
import threading
import time

from aloha.keyboard_commands import RecorderKeyRouter
from aloha.keyboard_listener import run_keyboard_listener
from aloha.record_trigger import (
    RecordingEvents,
    RecordingPhase,
    RecordingTriggerController,
    TriggerResult,
)


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_router_maps_each_supported_key_once_outside_retry():
    calls = {key: 0 for key in "bdmsr"}

    def count(key):
        def callback():
            calls[key] += 1

        return callback

    router = RecorderKeyRouter(
        get_phase=lambda: RecordingPhase.WAITING_FOR_B,
        on_b=count("b"),
        on_d=count("d"),
        on_m=count("m"),
        on_s=count("s"),
        on_r=count("r"),
        on_ignored=lambda key: None,
    )

    for key in "bdmsr":
        router.handle(key)

    assert calls == {key: 1 for key in "bdmsr"}


def test_listener_and_router_keep_s_available_while_retry_blocks_other_keys():
    events = RecordingEvents.create()
    program_exit = threading.Event()
    controller = RecordingTriggerController(events, start_trigger="b")
    assert controller.complete_preparation() is True
    calls = {"b": 0, "d": 0, "m": 0, "s": 0, "r": 0}
    ignored = []
    results = []

    def handle_b():
        calls["b"] += 1
        results.append(controller.handle_b())

    def handle_d():
        calls["d"] += 1
        results.append(controller.handle_d())

    def count(key):
        def callback():
            calls[key] += 1

        return callback

    router = RecorderKeyRouter(
        get_phase=lambda: controller.phase,
        on_b=handle_b,
        on_d=handle_d,
        on_m=count("m"),
        on_s=count("s"),
        on_r=count("r"),
        on_ignored=ignored.append,
    )
    master_fd, slave_fd = os.openpty()
    stdin = os.fdopen(slave_fd, "r", encoding="utf-8", buffering=1)
    listener = threading.Thread(
        target=run_keyboard_listener,
        args=(program_exit, router.handle),
        kwargs={"stdin": stdin, "poll_timeout": 0.01},
    )
    listener.start()

    try:
        _wait_until(
            lambda: not (termios.tcgetattr(stdin.fileno())[3] & termios.ICANON)
        )
        os.write(master_fd, b"b")
        _wait_until(lambda: controller.phase is RecordingPhase.RECORDING)

        os.write(master_fd, b"d")
        _wait_until(lambda: controller.phase is RecordingPhase.RETURNING_TO_RETRY)
        assert calls["d"] == 1
        assert results == [TriggerResult.STARTED, TriggerResult.DISCARD_STARTED]
        assert events.discard_and_retry.is_set()
        assert events.return_to_start.is_set()
        assert not events.stop_and_save.is_set()
        assert not events.stop_no_save.is_set()
        assert not events.skip_sleep.is_set()

        os.write(master_fd, b"b")
        _wait_until(lambda: ignored == ["b"])
        os.write(master_fd, b"m")
        _wait_until(lambda: ignored == ["b", "m"])
        os.write(master_fd, b"r")
        _wait_until(lambda: ignored == ["b", "m", "r"])
        os.write(master_fd, b"d")
        _wait_until(lambda: ignored == ["b", "m", "r", "d"])
        os.write(master_fd, b"s")
        _wait_until(lambda: calls["s"] == 1)
        assert calls == {"b": 1, "d": 1, "m": 0, "s": 1, "r": 0}
        assert ignored == ["b", "m", "r", "d"]
        assert not program_exit.is_set()
        assert listener.is_alive()

        controller.complete_retry()
        assert controller.phase is RecordingPhase.WAITING_FOR_B

        os.write(master_fd, b"b")
        _wait_until(lambda: controller.phase is RecordingPhase.RECORDING)
        assert controller.mark_sample_recorded() is True
        os.write(master_fd, b"b")
        _wait_until(lambda: controller.phase is RecordingPhase.RETURNING_TO_SAVE)

        assert calls["b"] == 3
        assert results[-2:] == [TriggerResult.STARTED, TriggerResult.STOPPED]
    finally:
        program_exit.set()
        listener.join(timeout=1.0)
        stdin.close()
        os.close(master_fd)

    assert not listener.is_alive()

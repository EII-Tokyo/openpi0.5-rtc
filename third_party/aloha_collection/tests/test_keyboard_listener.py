import io
import os
import pty
import termios
import threading
import time

from aloha.keyboard_listener import run_keyboard_listener


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_listener_reads_multiple_keys_and_restores_terminal_on_program_exit():
    master_fd, slave_fd = pty.openpty()
    slave = os.fdopen(slave_fd, "r", encoding="utf-8", buffering=1, closefd=False)
    original_settings = termios.tcgetattr(slave_fd)
    exit_event = threading.Event()
    received = []
    thread = threading.Thread(
        target=run_keyboard_listener,
        args=(exit_event, received.append),
        kwargs={"stdin": slave, "logger": received.append},
    )

    try:
        thread.start()
        wait_until(lambda: termios.tcgetattr(slave_fd) != original_settings)
        os.write(master_fd, b"b")
        wait_until(lambda: "b" in received)
        os.write(master_fd, b"s")
        wait_until(lambda: "s" in received)

        time.sleep(0.15)
        assert thread.is_alive()

        exit_event.set()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert received.count("b") == 1
        assert received.count("s") == 1
        assert termios.tcgetattr(slave_fd) == original_settings
    finally:
        exit_event.set()
        thread.join(timeout=1.0)
        slave.close()
        os.close(slave_fd)
        os.close(master_fd)


def test_non_tty_disables_hotkeys_without_instructing_operator_to_press_s():
    logs = []
    received = []

    run_keyboard_listener(
        threading.Event(),
        received.append,
        stdin=io.StringIO("s"),
        logger=logs.append,
    )

    assert received == []
    assert logs == [
        "[keyboard] stdin is not a TTY; keyboard hotkeys disabled for this run."
    ]
    assert all("按 s" not in message for message in logs)

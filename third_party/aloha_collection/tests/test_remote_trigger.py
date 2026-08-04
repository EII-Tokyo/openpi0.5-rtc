import os
import socket
import stat
import time

import pytest

from aloha.remote_trigger import (
    CommandParser,
    ProtocolError,
    TriggerSocketServer,
    TriggerUnavailable,
    send_command,
)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_parser_accepts_split_b_frame():
    parser = CommandParser(max_bytes=8)
    assert parser.feed(b"") == []
    assert parser.feed(b"b") == []
    assert parser.feed(b"\n") == ["b"]


def test_parser_rejects_unsupported_command():
    parser = CommandParser(max_bytes=8)
    with pytest.raises(ProtocolError, match="unsupported command"):
        parser.feed(b"m\n")


def test_parser_rejects_oversized_frame():
    parser = CommandParser(max_bytes=8)
    with pytest.raises(ProtocolError, match="too large"):
        parser.feed(b"123456789")


def test_socket_server_delivers_b_sets_mode_and_removes_socket(tmp_path):
    received = []
    path = tmp_path / "trigger.sock"

    with TriggerSocketServer(path, received.append):
        wait_until(path.exists)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        send_command(path, "b")
        wait_until(lambda: received == ["b"])

    assert not path.exists()


def test_socket_server_recovers_stale_socket(tmp_path):
    path = tmp_path / "trigger.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(path))
    stale.close()

    with TriggerSocketServer(path, lambda command: None):
        wait_until(path.exists)

    assert not path.exists()


def test_socket_server_refuses_live_socket(tmp_path):
    path = tmp_path / "trigger.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    try:
        with pytest.raises(TriggerUnavailable, match="already active"):
            TriggerSocketServer(path, lambda command: None).start()
    finally:
        listener.close()
        os.unlink(path)


def test_send_command_rejects_invalid_local_command(tmp_path):
    with pytest.raises(ProtocolError, match="unsupported command"):
        send_command(tmp_path / "missing.sock", "m")


def test_send_command_reports_missing_server(tmp_path):
    with pytest.raises(TriggerUnavailable, match="unavailable"):
        send_command(tmp_path / "missing.sock", "b")

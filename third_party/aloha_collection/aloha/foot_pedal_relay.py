"""Filtering and SSH transport for the remote ALOHA foot pedal."""

from __future__ import annotations

import subprocess
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence


EV_KEY = 1
KEY_B = 48


def deduplicate_device_paths(paths: Sequence[Path]) -> list[Path]:
    """Return one stable alias for each underlying event device."""
    selected: dict[Path, Path] = {}
    for path in sorted(paths):
        real_device = path.resolve()
        selected.setdefault(real_device, path)
    return sorted(selected.values())


def open_input_devices(paths: Sequence[Path], factory: Callable) -> list[tuple[object, Path]]:
    """Open paths without requiring evdev device handles to be hashable."""
    return [(factory(str(path)), path) for path in paths]


class ForwardResult(Enum):
    SENT = "sent"
    DROPPED = "dropped"


class PedalEventFilter:
    def __init__(self, debounce_seconds: float = 0.4, *, event_code: int = KEY_B):
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds must not be negative")
        self._debounce_seconds = debounce_seconds
        self._event_code = event_code
        self._last_press = float("-inf")

    def accept(self, event, *, now: float) -> bool:
        if event.type != EV_KEY or event.code != self._event_code or event.value != 1:
            return False
        if now - self._last_press < self._debounce_seconds:
            return False
        self._last_press = now
        return True


class PedalRelay:
    def __init__(
        self,
        transport,
        *,
        debounce_seconds: float = 0.4,
        event_code: int = KEY_B,
    ):
        self._transport = transport
        self._filter = PedalEventFilter(debounce_seconds, event_code=event_code)

    def process_event(self, event, *, now: float) -> Optional[ForwardResult]:
        if not self._filter.accept(event, now=now):
            return None
        if self._transport.send("b", now):
            return ForwardResult.SENT
        return ForwardResult.DROPPED


def build_ssh_command(host: str) -> list[str]:
    if not host or host.startswith("-"):
        raise ValueError("invalid SSH host")
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        host,
        "docker",
        "exec",
        "-i",
        "aloha2-collect",
        "python3",
        "/root/interbotix_ws/src/aloha/scripts/send_record_trigger.py",
        "--stream",
    ]


class PersistentSshTransport:
    """Maintain a fixed SSH subprocess without buffering pedal events."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        popen_factory: Callable = subprocess.Popen,
        initial_backoff: float = 1.0,
        maximum_backoff: float = 16.0,
    ):
        if not command:
            raise ValueError("command must not be empty")
        if initial_backoff <= 0 or maximum_backoff < initial_backoff:
            raise ValueError("invalid backoff limits")
        self.command = list(command)
        self._popen_factory = popen_factory
        self._initial_backoff = initial_backoff
        self._maximum_backoff = maximum_backoff
        self._backoff = initial_backoff
        self._next_retry = 0.0
        self._process = None
        self._lock = threading.RLock()

    def ensure_connected(self, *, now: float) -> bool:
        with self._lock:
            if self._is_connected():
                return True
            self._close_process()
            if now < self._next_retry:
                return False
            try:
                process = self._popen_factory(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    shell=False,
                )
            except OSError:
                self._schedule_retry(now)
                return False
            if process.stdin is None:
                process.terminate()
                self._schedule_retry(now)
                return False
            self._process = process
            self._backoff = self._initial_backoff
            self._next_retry = 0.0
            return True

    def send(self, command: str, now: float) -> bool:
        if command != "b":
            raise ValueError("unsupported relay command")
        with self._lock:
            if not self._is_connected():
                return False
            try:
                self._process.stdin.write("b\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._close_process()
                self._schedule_retry(now)
                return False
            return True

    def close(self) -> None:
        with self._lock:
            self._close_process()

    def _is_connected(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _schedule_retry(self, now: float) -> None:
        self._next_retry = now + self._backoff
        self._backoff = min(self._maximum_backoff, self._backoff * 2)

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()

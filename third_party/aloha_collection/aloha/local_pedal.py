"""Local Linux input support for the PCsensor recording foot pedal."""

from __future__ import annotations

import fcntl
import os
import select
import struct
import threading
from pathlib import Path
from typing import Callable, Optional


INPUT_EVENT_STRUCT = struct.Struct("llHHI")
EV_KEY = 1
KEY_B = 48
EVIOCGRAB = 0x40044590
DEFAULT_PEDAL_PATH = Path(
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
)


class FootPedalUnavailable(RuntimeError):
    """The configured pedal cannot be exclusively owned by the recorder."""


def grab_input_device(descriptor: int) -> None:
    fcntl.ioctl(descriptor, EVIOCGRAB, 1)


def release_input_device(descriptor: int) -> None:
    fcntl.ioctl(descriptor, EVIOCGRAB, 0)


class FootPedalEventDecoder:
    """Decode complete Linux input records and emit debounced commands."""

    def __init__(
        self,
        *,
        debounce_seconds: float = 1.0,
        logger: Callable[[str], None] = print,
    ) -> None:
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds must be non-negative")
        self._buffer = bytearray()
        self._debounce_seconds = float(debounce_seconds)
        self._logger = logger
        self._pressed = False
        self._last_accepted_at: Optional[float] = None

    def feed(self, data: bytes) -> list[str]:
        self._buffer.extend(data)
        commands: list[str] = []
        record_size = INPUT_EVENT_STRUCT.size
        while len(self._buffer) >= record_size:
            raw = bytes(self._buffer[:record_size])
            del self._buffer[:record_size]
            seconds, microseconds, event_type, code, value = (
                INPUT_EVENT_STRUCT.unpack(raw)
            )
            if event_type != EV_KEY or code != KEY_B:
                continue
            if value == 0:
                self._pressed = False
                continue
            if value != 1:
                continue
            event_time = (
                float(seconds) + float(microseconds) / 1_000_000.0
            )
            if self._pressed:
                self._logger(
                    "[foot-pedal] 丢弃重复按下事件（尚未释放）"
                )
                continue
            self._pressed = True
            if (
                self._last_accepted_at is not None
                and event_time - self._last_accepted_at
                < self._debounce_seconds
            ):
                self._logger(
                    "[foot-pedal] 丢弃防抖窗口内的重复触发"
                )
                continue
            self._last_accepted_at = event_time
            self._logger("[foot-pedal] 接受一次按下触发")
            commands.append("b")
        return commands


class FootPedalListener:
    """Supervise one stable Linux input path for the recorder lifetime."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        on_press: Callable[[], None],
        *,
        open_device: Callable[[str, int], int] = os.open,
        read_device: Callable[[int, int], bytes] = os.read,
        close_device: Callable[[int], None] = os.close,
        wait_readable: Callable = select.select,
        grab_device: Callable[[int], None] = grab_input_device,
        release_device: Callable[[int], None] = release_input_device,
        on_failure: Callable[[BaseException], None] | None = None,
        poll_interval: float = 0.2,
        debounce_seconds: float = 1.0,
        logger: Callable[[str], None] = print,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("listener poll interval must be positive")
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds must be non-negative")
        self.path = Path(path)
        self._on_press = on_press
        self._open_device = open_device
        self._read_device = read_device
        self._close_device = close_device
        self._wait_readable = wait_readable
        self._grab_device = grab_device
        self._release_device = release_device
        self._on_failure = on_failure or (lambda _error: None)
        self._poll_interval = poll_interval
        self._debounce_seconds = float(debounce_seconds)
        self._logger = logger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._descriptor: Optional[int] = None
        self._state_lock = threading.Lock()
        self._failure_reported = False

    def __enter__(self) -> "FootPedalListener":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> "FootPedalListener":
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            descriptor = None
            try:
                descriptor = self._open_device(
                    str(self.path),
                    os.O_RDONLY | os.O_NONBLOCK,
                )
                self._grab_device(descriptor)
            except OSError as exc:
                if descriptor is not None:
                    try:
                        self._close_device(descriptor)
                    except OSError:
                        pass
                raise FootPedalUnavailable(
                    f"exclusive foot pedal unavailable: {self.path}: {exc}"
                ) from exc

            self._descriptor = descriptor
            self._failure_reported = False
            self._stop.clear()
            try:
                self._logger(
                    f"[foot-pedal] exclusive device acquired: {self.path}"
                )
                self._thread = threading.Thread(
                    target=self._run,
                    args=(descriptor,),
                    name="aloha-local-foot-pedal",
                    daemon=True,
                )
                self._thread.start()
            except BaseException:
                self._stop.set()
                self._thread = None
                self._descriptor = None
                try:
                    self._release_device(descriptor)
                except OSError:
                    pass
                finally:
                    try:
                        self._close_device(descriptor)
                    except OSError:
                        pass
                raise
        return self

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(1.0, self._poll_interval * 2))

        with self._state_lock:
            descriptor = self._descriptor
            self._descriptor = None
        if descriptor is None:
            return
        try:
            self._release_device(descriptor)
            self._logger(
                f"[foot-pedal] exclusive device released: {self.path}"
            )
        except OSError as exc:
            self._logger(f"[foot-pedal] exclusive release failed: {exc}")
        finally:
            try:
                self._close_device(descriptor)
            except OSError as exc:
                self._logger(f"[foot-pedal] device close failed: {exc}")

    def _run(self, descriptor: int) -> None:
        try:
            self._read_open_device(descriptor)
        except (OSError, ValueError) as exc:
            if not self._stop.is_set():
                self._report_failure(exc)

    def _report_failure(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure_reported:
                return
            self._failure_reported = True
        self._logger(
            "[foot-pedal] runtime device failure; requesting "
            f"no-save safe stop: {error}"
        )
        self._on_failure(error)

    def _read_open_device(self, descriptor: int) -> None:
        decoder = FootPedalEventDecoder(
            debounce_seconds=self._debounce_seconds,
            logger=self._logger,
        )
        while not self._stop.is_set():
            readable, _, _ = self._wait_readable(
                [descriptor],
                [],
                [],
                self._poll_interval,
            )
            if not readable:
                continue
            data = self._read_device(descriptor, INPUT_EVENT_STRUCT.size * 64)
            if not data:
                raise OSError("foot pedal event stream ended")
            for command in decoder.feed(data):
                if command == "b":
                    self._on_press()

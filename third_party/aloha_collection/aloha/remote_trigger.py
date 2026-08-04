"""Local, bounded command transport for ALOHA recording triggers."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Callable, Optional


DEFAULT_SOCKET_PATH = Path("/tmp/aloha-record-trigger.sock")


class ProtocolError(ValueError):
    """A trigger command did not match the bounded wire protocol."""


class TriggerUnavailable(ConnectionError):
    """The local trigger server is not available."""


class CommandParser:
    """Parse newline-delimited, ASCII trigger commands."""

    def __init__(self, max_bytes: int = 16):
        if max_bytes < 2:
            raise ValueError("max_bytes must be at least 2")
        self._buffer = bytearray()
        self._max_bytes = max_bytes

    def feed(self, data: bytes) -> list[str]:
        self._buffer.extend(data)
        commands: list[str] = []
        while b"\n" in self._buffer:
            raw, _, rest = self._buffer.partition(b"\n")
            self._buffer[:] = rest
            if len(raw) + 1 > self._max_bytes:
                raise ProtocolError("command frame too large")
            try:
                command = raw.decode("ascii", errors="strict").strip().lower()
            except UnicodeDecodeError as exc:
                raise ProtocolError("command must be ASCII") from exc
            if command != "b":
                raise ProtocolError("unsupported command")
            commands.append(command)
        if len(self._buffer) > self._max_bytes:
            raise ProtocolError("command frame too large")
        return commands


class TriggerSocketServer:
    """Serve trigger commands on a private Unix-domain socket."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        on_command: Callable[[str], None],
        *,
        max_bytes: int = 16,
    ):
        self.path = Path(path)
        self._on_command = on_command
        self._max_bytes = max_bytes
        self._listener: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._socket_identity: Optional[tuple[int, int]] = None

    def __enter__(self) -> "TriggerSocketServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> "TriggerSocketServer":
        if self._listener is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_socket()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.path))
            os.chmod(self.path, 0o600)
            socket_stat = self.path.stat()
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
            listener.listen(4)
            listener.settimeout(0.2)
        except Exception:
            listener.close()
            self._unlink_owned_socket()
            raise

        self._stop.clear()
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve,
            name="aloha-trigger-socket",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)
        self._unlink_owned_socket()

    def _remove_stale_socket(self) -> None:
        if not self.path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self.path))
        except OSError:
            self.path.unlink(missing_ok=True)
        else:
            raise TriggerUnavailable(f"trigger socket already active: {self.path}")
        finally:
            probe.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                parser = CommandParser(self._max_bytes)
                while not self._stop.is_set():
                    try:
                        data = connection.recv(64)
                    except OSError:
                        break
                    if not data:
                        break
                    try:
                        commands = parser.feed(data)
                    except ProtocolError:
                        break
                    for command in commands:
                        self._on_command(command)

    def _unlink_owned_socket(self) -> None:
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            socket_stat = self.path.stat()
        except FileNotFoundError:
            return
        if (socket_stat.st_dev, socket_stat.st_ino) == identity:
            self.path.unlink()


def send_command(
    path: os.PathLike[str] | str,
    command: str,
    *,
    timeout: float = 1.0,
) -> None:
    """Send one current command without retrying or buffering it."""
    normalized = command.strip().lower()
    if normalized != "b":
        raise ProtocolError("unsupported command")

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        client.sendall(b"b\n")
    except OSError as exc:
        raise TriggerUnavailable(f"trigger server unavailable: {path}") from exc
    finally:
        client.close()

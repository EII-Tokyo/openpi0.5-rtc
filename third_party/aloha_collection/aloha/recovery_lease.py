"""Exclusive ownership for safety-critical robot recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import TextIO
import uuid


DEFAULT_RECOVERY_LEASE_PATH = Path("/tmp/aloha-safe-sleep.lock")


class RecoveryLeaseBusy(RuntimeError):
    """Raised when another process already owns safe-sleep recovery."""


@dataclass(frozen=True)
class RecoveryLeaseMetadata:
    recovery_id: str
    owner_pid: int
    source: str
    robot: str
    started_wall_time: str


class RecoveryLease:
    """An advisory process lease whose file also exposes owner metadata."""

    def __init__(
        self,
        *,
        path: Path,
        handle: TextIO,
        metadata: RecoveryLeaseMetadata,
    ) -> None:
        self.path = path
        self._handle: TextIO | None = handle
        self.metadata = metadata

    @classmethod
    def acquire(
        cls,
        *,
        path: Path | str = DEFAULT_RECOVERY_LEASE_PATH,
        source: str,
        robot: str,
        recovery_id: str | None = None,
    ) -> "RecoveryLease":
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = destination.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise RecoveryLeaseBusy(
                f"safe-sleep recovery is already owned by {owner}"
            ) from exc

        metadata = RecoveryLeaseMetadata(
            recovery_id=recovery_id or uuid.uuid4().hex,
            owner_pid=os.getpid(),
            source=source,
            robot=robot,
            started_wall_time=datetime.now(timezone.utc).isoformat(),
        )
        handle.seek(0)
        handle.truncate()
        json.dump(asdict(metadata), handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        return cls(
            path=destination,
            handle=handle,
            metadata=metadata,
        )

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "RecoveryLease":
        return self

    def __exit__(self, *_args) -> None:
        self.release()

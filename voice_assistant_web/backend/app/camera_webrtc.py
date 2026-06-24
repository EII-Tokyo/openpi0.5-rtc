from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import uuid


@dataclass(frozen=True)
class CameraWebRTCSession:
    session_id: str
    cameras: list[str]
    codec: str
    created_at: float
    expires_at: float
    status: str = "signaling"


class CameraWebRTCSessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, CameraWebRTCSession] = {}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def create(self, cameras: list[str], codec: str, ttl_seconds: float, max_sessions: int) -> CameraWebRTCSession | None:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            if len(self._sessions) >= max_sessions:
                return None
            session = CameraWebRTCSession(
                session_id=uuid.uuid4().hex,
                cameras=list(cameras),
                codec=codec,
                created_at=now,
                expires_at=now + max(1.0, ttl_seconds),
            )
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> CameraWebRTCSession | None:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> CameraWebRTCSession | None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return None
            return CameraWebRTCSession(
                session_id=session.session_id,
                cameras=session.cameras,
                codec=session.codec,
                created_at=session.created_at,
                expires_at=session.expires_at,
                status="closed",
            )

    def _expire_locked(self, now: float) -> None:
        expired = [session_id for session_id, session in self._sessions.items() if session.expires_at <= now]
        for session_id in expired:
            self._sessions.pop(session_id, None)

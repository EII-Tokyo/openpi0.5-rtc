from __future__ import annotations

from datetime import UTC
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from uuid import uuid4

from .models import CaptureStatus
from .models import PreflightReport
from .models import PreflightStatus
from .models import SessionRecord

_SAFE_NAME = re.compile(r"^[\w .-]+$", re.UNICODE)


class SessionNotFoundError(KeyError):
    pass


class SessionTransitionError(RuntimeError):
    pass


class SessionStore:
    def __init__(self, root: Path):
        self._root = root

    def create(self, name: str) -> SessionRecord:
        normalized = name.strip()
        if not normalized or len(normalized) > 96 or _SAFE_NAME.fullmatch(normalized) is None:
            raise ValueError("Session name contains unsupported characters")
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)
        now = datetime.now(UTC)
        record = SessionRecord(
            id=f"cal-{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}",
            name=normalized,
            state="SETUP",
            created_at_utc=now,
            updated_at_utc=now,
        )
        session_dir = self._root / record.id
        (session_dir / "artifacts").mkdir(parents=True, exist_ok=False, mode=0o700)
        session_dir.chmod(0o700)
        self._atomic_write_json(session_dir / "session.json", record.model_dump(mode="json"))
        return record

    def get(self, session_id: str) -> SessionRecord:
        path = self._session_path(session_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SessionNotFoundError(session_id) from exc
        record = SessionRecord.model_validate(payload)
        artifact_path = path.parent / "artifacts/preflight.json"
        if artifact_path.is_file():
            report = PreflightReport.model_validate_json(artifact_path.read_text(encoding="utf-8"))
            record = record.model_copy(update={"latest_preflight": report})
        return record

    def record_preflight(self, session_id: str, report: PreflightReport) -> SessionRecord:
        record = self.get(session_id)
        if record.latest_preflight_sha256 is not None:
            raise SessionTransitionError("Preflight artifact already exists; create a new session to rerun")
        artifact_path = self._session_path(session_id).parent / "artifacts/preflight.json"
        payload = report.model_dump_json(indent=2).encode("utf-8")
        self._exclusive_write(artifact_path, payload)
        digest = hashlib.sha256(payload).hexdigest()
        now = datetime.now(UTC)
        state = {
            PreflightStatus.READY: "PREFLIGHT_READY",
            PreflightStatus.BLOCKED: "PREFLIGHT_BLOCKED",
            PreflightStatus.FAILED: "PREFLIGHT_FAILED",
        }[report.status]
        updated = record.model_copy(
            update={
                "state": state,
                "updated_at_utc": now,
                "latest_preflight_sha256": digest,
                "latest_preflight": report,
            }
        )
        session_payload = updated.model_dump(mode="json", exclude={"latest_preflight"})
        self._atomic_write_json(self._session_path(session_id), session_payload)
        return updated

    def assert_intrinsics_start_allowed(self, session_id: str) -> None:
        record = self.get(session_id)
        if record.state != "PREFLIGHT_READY":
            raise SessionTransitionError(
                f"Intrinsics capture requires PREFLIGHT_READY, current state is {record.state}"
            )

    def record_intrinsics_start(self, session_id: str, status: CaptureStatus) -> SessionRecord:
        self.assert_intrinsics_start_allowed(session_id)
        if status.state != "STREAMING" or not status.pipeline_started:
            raise SessionTransitionError("Capture agent did not report a streaming pipeline")
        artifact_path = self._session_path(session_id).parent / "artifacts/intrinsics_start.json"
        payload = (status.model_dump_json(indent=2) + "\n").encode("utf-8")
        self._exclusive_write(artifact_path, payload)
        record = self.get(session_id)
        updated = record.model_copy(update={"state": "INTRINSICS_CAPTURING", "updated_at_utc": datetime.now(UTC)})
        session_payload = updated.model_dump(mode="json", exclude={"latest_preflight"})
        self._atomic_write_json(self._session_path(session_id), session_payload)
        return updated

    def record_workflow_artifact(
        self,
        session_id: str,
        *,
        expected_state: str,
        next_state: str,
        artifact_name: str,
        payload: Any,
    ) -> SessionRecord:
        record = self.get(session_id)
        if record.state != expected_state:
            raise SessionTransitionError(
                f"{next_state} requires {expected_state}, current state is {record.state}"
            )
        if not re.fullmatch(r"[a-z0-9_]+\.json", artifact_name):
            raise ValueError("workflow artifact name is not safe")
        if hasattr(payload, "model_dump_json"):
            encoded = (payload.model_dump_json(indent=2) + "\n").encode("utf-8")
        else:
            encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        artifact_path = self._session_path(session_id).parent / "artifacts" / artifact_name
        self._exclusive_write(artifact_path, encoded)
        updated = record.model_copy(update={"state": next_state, "updated_at_utc": datetime.now(UTC)})
        session_payload = updated.model_dump(mode="json", exclude={"latest_preflight"})
        self._atomic_write_json(self._session_path(session_id), session_payload)
        return updated

    def read_workflow_artifact(self, session_id: str, artifact_name: str) -> dict:
        self.get(session_id)
        if not re.fullmatch(r"[a-z0-9_]+\.json", artifact_name):
            raise ValueError("workflow artifact name is not safe")
        path = self._session_path(session_id).parent / "artifacts" / artifact_name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SessionTransitionError(f"required workflow artifact is missing: {artifact_name}") from exc

    def export_output_dir(self, session_id: str) -> Path:
        self.get(session_id)
        return self._session_path(session_id).parent / "artifacts" / "export"

    def write_attempt_artifact(self, session_id: str, artifact_name: str, payload: Any) -> Path:
        self.get(session_id)
        if not re.fullmatch(r"[a-z0-9_]+\.json", artifact_name):
            raise ValueError("attempt artifact name is not safe")
        if hasattr(payload, "model_dump_json"):
            encoded = (payload.model_dump_json(indent=2) + "\n").encode("utf-8")
        else:
            encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path = self._session_path(session_id).parent / "artifacts" / artifact_name
        self._exclusive_write(path, encoded)
        return path

    def write_numbered_attempt(self, session_id: str, prefix: str, payload: Any) -> Path:
        self.get(session_id)
        if not re.fullmatch(r"[a-z0-9_]+", prefix):
            raise ValueError("attempt prefix is not safe")
        artifacts = self._session_path(session_id).parent / "artifacts"
        existing = sorted(artifacts.glob(f"{prefix}_attempt_*.json"))
        return self.write_attempt_artifact(
            session_id,
            f"{prefix}_attempt_{len(existing) + 1:03d}.json",
            payload,
        )

    def read_latest_attempt(self, session_id: str, prefix: str) -> dict:
        self.get(session_id)
        if not re.fullmatch(r"[a-z0-9_]+", prefix):
            raise ValueError("attempt prefix is not safe")
        artifacts = self._session_path(session_id).parent / "artifacts"
        existing = sorted(artifacts.glob(f"{prefix}_attempt_*.json"))
        if not existing:
            raise SessionTransitionError(f"required attempt is missing: {prefix}")
        return json.loads(existing[-1].read_text(encoding="utf-8"))

    def _session_path(self, session_id: str) -> Path:
        if not re.fullmatch(r"cal-[0-9T]+-[0-9a-f]{8}", session_id):
            raise SessionNotFoundError(session_id)
        return self._root / session_id / "session.json"

    @staticmethod
    def _exclusive_write(path: Path, payload: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.chmod(0o600)
        temporary_path.replace(path)

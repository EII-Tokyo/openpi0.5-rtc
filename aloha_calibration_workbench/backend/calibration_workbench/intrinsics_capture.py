from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import threading
from typing import Any, Protocol

from .models import CaptureStatus
from .models import CharucoObservation
from .models import FactoryIntrinsics
from .models import OwnershipState
from .models import PreflightReport
from .models import PreflightStatus
from .models import ProductionProfile
from .models import SampleRecord

_SESSION_ID = re.compile(r"cal-[0-9T]+-[0-9a-f]{8}")
_ENABLED_STAGE1_ROLES = {"cam_high"}


class CaptureConflictError(RuntimeError):
    pass


class CaptureNotRunningError(RuntimeError):
    pass


@dataclass(frozen=True)
class FramePacket:
    rgb: Any
    frame_number: int
    device_timestamp_ms: float


@dataclass(frozen=True)
class PreviewFrame:
    jpeg: bytes
    observation: CharucoObservation


class PreflightRunner(Protocol):
    def run(self) -> PreflightReport: ...


class RunningCamera(Protocol):
    def factory_intrinsics(self) -> FactoryIntrinsics: ...

    def next_frame(self) -> FramePacket: ...

    def stop(self) -> None: ...


class CameraBackend(Protocol):
    def start(self, serial: str, profile: ProductionProfile) -> RunningCamera: ...


class FrameAnalyzer(Protocol):
    def analyze(
        self,
        frame: FramePacket,
        intrinsics: FactoryIntrinsics,
    ) -> tuple[CharucoObservation, bytes, bytes]: ...


class IntrinsicsCaptureService:
    """Own exactly one color-only RealSense pipeline and immutable sample artifacts."""

    def __init__(
        self,
        *,
        preflight: PreflightRunner,
        profile: ProductionProfile,
        backend: CameraBackend,
        analyzer: FrameAnalyzer,
        artifact_root: Path,
    ) -> None:
        self._preflight = preflight
        self._profile = profile
        self._backend = backend
        self._analyzer = analyzer
        self._artifact_root = artifact_root
        self._lock = threading.RLock()
        self._camera: RunningCamera | None = None
        self._status = CaptureStatus(state="IDLE")
        self._sample_count = 0

    def status(self) -> CaptureStatus:
        with self._lock:
            return self._status.model_copy(deep=True)

    def start(self, *, session_id: str, role: str) -> CaptureStatus:
        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Invalid calibration session identifier")
        if role not in _ENABLED_STAGE1_ROLES:
            raise ValueError(f"Camera role is not enabled in Stage 1: {role}")
        with self._lock:
            if self._camera is not None:
                raise CaptureConflictError(
                    f"Camera {self._status.role} is already streaming; stop it before starting {role}"
                )
            report = self._preflight.run()
            if report.status is not PreflightStatus.READY:
                raise CaptureConflictError(f"Preflight is {report.status.value}; capture was not started")
            selected = next((camera for camera in report.cameras if camera.role == role), None)
            if selected is None or not selected.identity_match or not selected.production_profile_supported:
                raise CaptureConflictError(f"Camera {role} did not pass identity/profile preflight")
            if selected.ownership is not OwnershipState.FREE:
                raise CaptureConflictError(f"Camera {role} ownership is {selected.ownership.value}")

            running = self._backend.start(selected.expected_serial, self._profile)
            try:
                intrinsics = running.factory_intrinsics()
                if (intrinsics.width, intrinsics.height) != (self._profile.width, self._profile.height):
                    raise RuntimeError("Active stream intrinsics do not match the production resolution")
                status = CaptureStatus(
                    state="STREAMING",
                    session_id=session_id,
                    role=role,
                    serial=selected.expected_serial,
                    profile=self._profile,
                    factory_intrinsics=intrinsics,
                    pipeline_started=True,
                    depth_stream_started=False,
                    robot_command_api=False,
                )
                camera_root = self._camera_root(session_id, role)
                camera_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                camera_root.chmod(0o700)
                self.write_exclusive_artifact(
                    camera_root / "factory_intrinsics.json",
                    (status.model_dump_json(indent=2) + "\n").encode("utf-8"),
                )
            except Exception:
                running.stop()
                raise
            self._camera = running
            self._status = status
            self._sample_count = 0
            return self.status()

    def preview(self) -> PreviewFrame:
        with self._lock:
            observation, jpeg, _ = self._analyze_next()
            self._status = self._status.model_copy(update={"latest_observation": observation})
            return PreviewFrame(jpeg=jpeg, observation=observation)

    def capture_sample(self) -> SampleRecord:
        with self._lock:
            observation, _, png = self._analyze_next()
            self._sample_count += 1
            sample_id = f"S-{self._sample_count:03d}"
            partition = "HELD_OUT" if self._sample_count % 5 == 0 else "SOLVE"
            accepted = observation.board_detected and observation.charuco_corner_count >= 8
            reason = "board-and-corners-detected" if accepted else "insufficient-charuco-corners"
            assert self._status.session_id is not None
            assert self._status.role is not None
            sample_dir = self._camera_root(self._status.session_id, self._status.role) / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            sample_dir.chmod(0o700)
            image_path = sample_dir / f"{sample_id}.png"
            self.write_exclusive_artifact(image_path, png)
            image_sha256 = hashlib.sha256(png).hexdigest()
            record = SampleRecord(
                id=sample_id,
                session_id=self._status.session_id,
                role=self._status.role,
                partition=partition,
                accepted=accepted,
                reason=reason,
                observation=observation,
                image_sha256=image_sha256,
            )
            metadata_path = sample_dir / f"{sample_id}.json"
            metadata_bytes = (record.model_dump_json(indent=2) + "\n").encode("utf-8")
            self.write_exclusive_artifact(metadata_path, metadata_bytes)
            self._status = self._status.model_copy(update={"latest_observation": observation})
            return record

    def stop(self) -> CaptureStatus:
        with self._lock:
            if self._camera is not None:
                self._camera.stop()
            self._camera = None
            self._status = CaptureStatus(state="IDLE")
            self._sample_count = 0
            return self.status()

    def _analyze_next(self) -> tuple[CharucoObservation, bytes, bytes]:
        if self._camera is None or self._status.factory_intrinsics is None:
            raise CaptureNotRunningError("No camera pipeline is active")
        frame = self._camera.next_frame()
        return self._analyzer.analyze(frame, self._status.factory_intrinsics)

    def _camera_root(self, session_id: str, role: str) -> Path:
        return self._artifact_root / session_id / role

    @staticmethod
    def write_exclusive_artifact(path: Path, payload: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

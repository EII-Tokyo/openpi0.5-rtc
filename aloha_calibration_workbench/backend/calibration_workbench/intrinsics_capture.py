from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Protocol

import cv2
import numpy as np

from .models import CaptureStatus
from .models import CharucoObservation
from .models import FactoryIntrinsics
from .models import OwnershipState
from .models import PreflightReport
from .models import PreflightStatus
from .models import ProductionProfile
from .models import SampleRecord
from .workflow import FactoryCameraSnapshot
from .workflow import TransformRecord
from .workflow import WorldOriginCaptureBatch

_SESSION_ID = re.compile(r"cal-[0-9T]+-[0-9a-f]{8}")
_ENABLED_STAGE1_ROLES = {"cam_high"}
_AUTO_EXPOSURE_WARMUP_FRAMES = 30


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


@dataclass(frozen=True)
class RgbSnapshot:
    jpeg: bytes
    attempt_id: str
    frame_number: int
    device_timestamp_ms: float
    image_sha256: str


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
    """Consume one color source at a time and write immutable sample artifacts."""

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

    def _assert_camera_source_available(self, camera: object) -> None:
        shared_source = bool(getattr(self._backend, "shared_source", False))
        expected = OwnershipState.ROS_SOURCE if shared_source else OwnershipState.FREE
        ownership = getattr(camera, "ownership")
        if ownership is not expected:
            mode = "ROS subscription" if shared_source else "exclusive RealSense"
            raise CaptureConflictError(
                f"Camera {getattr(camera, 'role')} ownership is {ownership.value}; "
                f"{mode} source requires {expected.value}"
            )

    def snapshot_factory_intrinsics(self) -> list[FactoryCameraSnapshot]:
        """Open each configured color stream sequentially and freeze its factory K/D."""
        with self._lock:
            if self._camera is not None:
                raise CaptureConflictError("stop the active camera before freezing all factory intrinsics")
            report = self._preflight.run()
            if report.status is not PreflightStatus.READY:
                raise CaptureConflictError(f"Preflight is {report.status.value}; factory snapshot was not started")
            snapshots: list[FactoryCameraSnapshot] = []
            for camera in sorted(report.cameras, key=lambda item: item.role):
                if not camera.identity_match or not camera.production_profile_supported:
                    raise CaptureConflictError(f"Camera {camera.role} did not pass identity/profile preflight")
                self._assert_camera_source_available(camera)
                running = self._backend.start(camera.expected_serial, self._profile)
                try:
                    intrinsics = running.factory_intrinsics()
                    if (intrinsics.width, intrinsics.height) != (self._profile.width, self._profile.height):
                        raise RuntimeError(
                            f"Camera {camera.role} factory intrinsics do not match production resolution"
                        )
                    snapshots.append(
                        FactoryCameraSnapshot(
                            role=camera.role,
                            serial=camera.expected_serial,
                            firmware=camera.firmware,
                            profile=self._profile,
                            intrinsics=intrinsics,
                            depth_stream_started=False,
                        )
                    )
                finally:
                    running.stop()
            return snapshots

    def capture_world_origin_batch(
        self,
        *,
        session_id: str,
        tag_size_m: float,
        tag_plane_height_m: float,
        frame_count: int = 200,
    ) -> WorldOriginCaptureBatch:
        """Capture one immutable, RGB-only AprilTag attempt from camera_high."""
        from .apriltag_analyzer import AprilTagAnalyzer

        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Invalid calibration session identifier")
        if not 150 <= frame_count <= 300:
            raise ValueError("world-origin frame_count must be between 150 and 300")
        with self._lock:
            if self._camera is not None:
                raise CaptureConflictError("stop the active camera before capturing world origin")
            report = self._preflight.run()
            if report.status is not PreflightStatus.READY:
                raise CaptureConflictError(f"Preflight is {report.status.value}; world capture was not started")
            selected = next((camera for camera in report.cameras if camera.role == "cam_high"), None)
            if selected is None or not selected.identity_match or not selected.production_profile_supported:
                raise CaptureConflictError("Camera cam_high did not pass identity/profile preflight")
            self._assert_camera_source_available(selected)
            running = self._backend.start(selected.expected_serial, self._profile)
            try:
                intrinsics = running.factory_intrinsics()
                analyzer = AprilTagAnalyzer(tag_id=0, tag_size_m=tag_size_m)
                # Librealsense examples discard the first 30 frames so automatic
                # exposure and other camera settings can settle before evidence capture.
                for _ in range(_AUTO_EXPOSURE_WARMUP_FRAMES):
                    running.next_frame()
                attempts_root = self._camera_root(session_id, "cam_high") / "world_origin"
                attempts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                attempt_number = 1 + len([path for path in attempts_root.iterdir() if path.is_dir()])
                attempt_root = attempts_root / f"A-{attempt_number:03d}"
                frames_root = attempt_root / "frames"
                frames_root.mkdir(parents=True, exist_ok=False, mode=0o700)
                samples = []
                for index in range(frame_count):
                    detection = analyzer.analyze(running.next_frame(), intrinsics)
                    frame_id = f"F-{index + 1:03d}"
                    self.write_exclusive_artifact(frames_root / f"{frame_id}.png", detection.png)
                    image_sha256 = hashlib.sha256(detection.png).hexdigest()
                    enriched_sample = None
                    if detection.sample is not None:
                        enriched_sample = detection.sample.model_copy(
                            update={
                                "frame_id": frame_id,
                                "device_timestamp_ms": detection.device_timestamp_ms,
                                "image_sha256": image_sha256,
                            }
                        )
                    metadata = {
                        "id": frame_id,
                        "frame_number": detection.frame_number,
                        "device_timestamp_ms": detection.device_timestamp_ms,
                        "detected": detection.detected,
                        "corners_px": detection.corners_px,
                        "image_sha256": image_sha256,
                        "sample": None if enriched_sample is None else enriched_sample.model_dump(mode="json"),
                    }
                    self.write_exclusive_artifact(
                        frames_root / f"{frame_id}.json",
                        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                    )
                    if enriched_sample is not None:
                        samples.append(enriched_sample)
                world_from_tag = np.eye(4, dtype=np.float64)
                world_from_tag[2, 3] = tag_plane_height_m
                batch = WorldOriginCaptureBatch(
                    samples=samples,
                    world_from_tag=TransformRecord(
                        source_frame="tag",
                        target_frame="table_world",
                        matrix=world_from_tag.tolist(),
                    ),
                    total_frames=frame_count,
                    detected_frames=len(samples),
                )
                self.write_exclusive_artifact(
                    attempt_root / "capture_manifest.json",
                    (batch.model_dump_json(indent=2) + "\n").encode("utf-8"),
                )
                return batch
            finally:
                running.stop()

    def capture_table_snapshot(self, *, session_id: str) -> RgbSnapshot:
        """Capture and persist the exact RGB frame returned for table-dot clicks."""

        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Invalid calibration session identifier")
        with self._lock:
            if self._camera is not None:
                raise CaptureConflictError("stop the active camera before capturing a table snapshot")
            report = self._preflight.run()
            if report.status is not PreflightStatus.READY:
                raise CaptureConflictError(
                    f"Preflight is {report.status.value}; table snapshot was not started"
                )
            selected = next((camera for camera in report.cameras if camera.role == "cam_high"), None)
            if selected is None or not selected.identity_match or not selected.production_profile_supported:
                raise CaptureConflictError("Camera cam_high did not pass identity/profile preflight")
            self._assert_camera_source_available(selected)
            running = self._backend.start(selected.expected_serial, self._profile)
            try:
                # Let auto-exposure settle without authoring those warm-up frames as evidence.
                for _ in range(_AUTO_EXPOSURE_WARMUP_FRAMES):
                    running.next_frame()
                packet = running.next_frame()
                rgb = np.asarray(packet.rgb)
                expected_shape = (self._profile.height, self._profile.width, 3)
                if rgb.shape != expected_shape or rgb.dtype != np.uint8:
                    raise RuntimeError(
                        f"camera_high RGB frame must be uint8 {expected_shape}, got {rgb.dtype} {rgb.shape}"
                    )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                png_ok, png = cv2.imencode(".png", bgr)
                jpeg_ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not png_ok or not jpeg_ok:
                    raise RuntimeError("OpenCV failed to encode table snapshot")
                attempts_root = self._camera_root(session_id, "cam_high") / "table_snapshot"
                attempts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                attempt_number = 1 + len([path for path in attempts_root.iterdir() if path.is_dir()])
                attempt_id = f"A-{attempt_number:03d}"
                attempt_root = attempts_root / attempt_id
                attempt_root.mkdir(parents=True, exist_ok=False, mode=0o700)
                png_bytes = png.tobytes()
                image_sha256 = hashlib.sha256(png_bytes).hexdigest()
                self.write_exclusive_artifact(attempt_root / "frame.png", png_bytes)
                metadata = {
                    "attempt_id": attempt_id,
                    "role": "cam_high",
                    "profile": self._profile.model_dump(mode="json"),
                    "frame_number": packet.frame_number,
                    "device_timestamp_ms": packet.device_timestamp_ms,
                    "image_sha256": image_sha256,
                    "depth_stream_started": False,
                }
                self.write_exclusive_artifact(
                    attempt_root / "frame.json",
                    (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                )
                return RgbSnapshot(
                    jpeg=jpeg.tobytes(),
                    attempt_id=attempt_id,
                    frame_number=packet.frame_number,
                    device_timestamp_ms=packet.device_timestamp_ms,
                    image_sha256=image_sha256,
                )
            finally:
                running.stop()

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
            self._assert_camera_source_available(selected)

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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from calibration_workbench.intrinsics_capture import CaptureConflictError
from calibration_workbench.intrinsics_capture import IntrinsicsCaptureService
from calibration_workbench.models import CharucoObservation
from calibration_workbench.models import FactoryIntrinsics
from calibration_workbench.models import OwnershipState
from calibration_workbench.models import PreflightCamera
from calibration_workbench.models import PreflightReport
from calibration_workbench.models import PreflightStatus
from calibration_workbench.models import ProductionProfile
import pytest


def _ready_report() -> PreflightReport:
    return PreflightReport(
        status=PreflightStatus.READY,
        registry_source="/project/aloha_stationary.yaml",
        registry_sha256="a" * 64,
        cameras=[
            PreflightCamera(
                role="cam_high",
                expected_serial="130322270656",
                connected=True,
                identity_match=True,
                actual_serial="130322270656",
                product_name="Intel RealSense D405",
                production_profile_supported=True,
                ownership=OwnershipState.FREE,
            ),
            PreflightCamera(
                role="cam_low",
                expected_serial="218622270440",
                connected=True,
                identity_match=True,
                actual_serial="218622270440",
                product_name="Intel RealSense D405",
                production_profile_supported=True,
                ownership=OwnershipState.FREE,
            ),
        ],
        issues=[],
    )


class FakePreflight:
    def run(self) -> PreflightReport:
        return _ready_report()


@dataclass
class FakeFrame:
    rgb: object = None
    frame_number: int = 17
    device_timestamp_ms: float = 123.5

    def __post_init__(self) -> None:
        if self.rgb is None:
            self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)


class FakeRunningCamera:
    def __init__(self) -> None:
        self.stopped = False
        self.frame_calls = 0

    def factory_intrinsics(self) -> FactoryIntrinsics:
        return FactoryIntrinsics(
            width=640,
            height=480,
            fx=601.0,
            fy=602.0,
            cx=319.5,
            cy=239.5,
            distortion_model="brown_conrady",
            distortion_coefficients=[0.1, -0.1, 0.0, 0.0, 0.0],
        )

    def next_frame(self) -> FakeFrame:
        self.frame_calls += 1
        return FakeFrame()

    def stop(self) -> None:
        self.stopped = True


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ProductionProfile]] = []
        self.running = FakeRunningCamera()

    def start(self, serial: str, profile: ProductionProfile) -> FakeRunningCamera:
        self.calls.append((serial, profile))
        return self.running


class FakeSharedRosBackend(FakeBackend):
    shared_source = True


class FakeRosPreflight:
    def run(self) -> PreflightReport:
        report = _ready_report()
        return report.model_copy(
            update={
                "exclusive_capture_required": False,
                "cameras": [
                    camera.model_copy(update={"ownership": OwnershipState.ROS_SOURCE})
                    for camera in report.cameras
                ],
            }
        )


class FakeAnalyzer:
    def analyze(self, frame: FakeFrame, intrinsics: FactoryIntrinsics):
        observation = CharucoObservation(
            board_detected=True,
            marker_count=12,
            charuco_corner_count=18,
            blur_variance=155.0,
            black_clip_percent=0.2,
            white_clip_percent=0.1,
            centroid_x=0.45,
            centroid_y=0.52,
            board_area_percent=22.0,
            reprojection_rms_px=0.31,
            frame_number=frame.frame_number,
            device_timestamp_ms=frame.device_timestamp_ms,
        )
        return observation, b"jpeg-bytes", b"png-bytes"


def _service(tmp_path: Path):
    backend = FakeBackend()
    service = IntrinsicsCaptureService(
        preflight=FakePreflight(),
        profile=ProductionProfile(width=640, height=480, fps=60, format="rgb8"),
        backend=backend,
        analyzer=FakeAnalyzer(),
        artifact_root=tmp_path,
    )
    return service, backend


def test_starts_exact_profile_and_exports_factory_intrinsics(tmp_path: Path):
    service, backend = _service(tmp_path)

    status = service.start(session_id="cal-20260805T120000-1234abcd", role="cam_high")

    assert backend.calls[0][0] == "130322270656"
    assert backend.calls[0][1].model_dump() == {
        "stream": "color",
        "width": 640,
        "height": 480,
        "fps": 60,
        "format": "rgb8",
    }
    assert status.state == "STREAMING"
    assert status.role == "cam_high"
    assert status.factory_intrinsics.fx == 601.0
    assert status.pipeline_started is True
    assert status.depth_stream_started is False
    assert status.robot_command_api is False
    assert (tmp_path / "cal-20260805T120000-1234abcd/cam_high/factory_intrinsics.json").is_file()


def test_enforces_one_active_camera_and_stops_the_owned_pipeline(tmp_path: Path):
    service, backend = _service(tmp_path)
    service.start(session_id="cal-20260805T120000-1234abcd", role="cam_high")

    with pytest.raises(CaptureConflictError, match="cam_high"):
        service.start(session_id="cal-20260805T120000-1234abcd", role="cam_high")

    stopped = service.stop()
    assert stopped.state == "IDLE"
    assert backend.running.stopped is True


def test_preview_and_sample_preserve_device_timestamp_and_immutable_artifacts(tmp_path: Path):
    service, _ = _service(tmp_path)
    session_id = "cal-20260805T120000-1234abcd"
    service.start(session_id=session_id, role="cam_high")

    preview = service.preview()
    sample = service.capture_sample()

    assert preview.jpeg == b"jpeg-bytes"
    assert preview.observation.device_timestamp_ms == 123.5
    assert sample.partition == "SOLVE"
    assert sample.accepted is True
    assert sample.observation.reprojection_rms_px == 0.31
    sample_dir = tmp_path / session_id / "cam_high/samples"
    assert (sample_dir / "S-001.png").read_bytes() == b"png-bytes"
    assert (sample_dir / "S-001.json").is_file()
    with pytest.raises(FileExistsError):
        service.write_exclusive_artifact(sample_dir / "S-001.png", b"replacement")


def test_rejects_invalid_session_identifier_before_any_device_access(tmp_path: Path):
    service, backend = _service(tmp_path)

    with pytest.raises(ValueError, match="session"):
        service.start(session_id="../../escape", role="cam_high")

    assert backend.calls == []


def test_table_snapshot_returns_and_persists_the_same_evidence_frame(tmp_path: Path):
    service, backend = _service(tmp_path)
    session_id = "cal-20260805T120000-1234abcd"

    snapshot = service.capture_table_snapshot(session_id=session_id)

    assert snapshot.attempt_id == "A-001"
    assert snapshot.frame_number == 17
    assert snapshot.jpeg.startswith(b"\xff\xd8")
    attempt = tmp_path / session_id / "cam_high/table_snapshot/A-001"
    assert (attempt / "frame.png").is_file()
    assert snapshot.image_sha256 == __import__("hashlib").sha256((attempt / "frame.png").read_bytes()).hexdigest()
    assert (attempt / "frame.json").is_file()
    assert backend.running.frame_calls == 31
    assert backend.running.stopped is True


def test_table_snapshot_accepts_a_verified_shared_ros_source(tmp_path: Path):
    backend = FakeSharedRosBackend()
    service = IntrinsicsCaptureService(
        preflight=FakeRosPreflight(),
        profile=ProductionProfile(width=640, height=480, fps=60, format="rgb8"),
        backend=backend,
        analyzer=FakeAnalyzer(),
        artifact_root=tmp_path,
    )

    snapshot = service.capture_table_snapshot(session_id="cal-20260805T120000-1234abcd")

    assert snapshot.attempt_id == "A-001"
    assert backend.running.frame_calls == 31

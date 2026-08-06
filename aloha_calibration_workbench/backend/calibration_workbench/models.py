from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field


class OwnershipState(StrEnum):
    FREE = "FREE"
    ROS_SOURCE = "ROS_SOURCE"
    BUSY = "BUSY"
    UNKNOWN = "UNKNOWN"


class PreflightStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class IssueSeverity(StrEnum):
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"
    ERROR = "ERROR"


class ProductionProfile(BaseModel):
    stream: str = "color"
    width: int
    height: int
    fps: int
    format: str = "rgb8"


class CandidateCamera(BaseModel):
    role: str
    config_name: str
    serial: str


class CandidateRegistry(BaseModel):
    source_path: str
    source_sha256: str
    cameras: list[CandidateCamera]
    profile: ProductionProfile


class PreflightIssue(BaseModel):
    code: str
    severity: IssueSeverity
    message: str
    camera_role: str | None = None


class PreflightCamera(BaseModel):
    role: str
    expected_serial: str
    connected: bool
    identity_match: bool
    actual_serial: str | None = None
    product_name: str | None = None
    firmware: str | None = None
    recommended_firmware: str | None = None
    usb_type: str | None = None
    physical_port: str | None = None
    production_profile_supported: bool = False
    ownership: OwnershipState = OwnershipState.UNKNOWN
    owner_processes: list[str] = Field(default_factory=list)
    video_nodes: list[str] = Field(default_factory=list)


class PreflightReport(BaseModel):
    status: PreflightStatus
    registry_source: str
    registry_sha256: str
    cameras: list[PreflightCamera]
    issues: list[PreflightIssue]
    captured_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    robot_command_api: bool = False
    browser_time_used: bool = False
    pipeline_started: bool = False
    hardware_reset_called: bool = False
    exclusive_capture_required: bool = True


class CreateSessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=96)


class SessionRecord(BaseModel):
    id: str
    name: str
    state: str
    created_at_utc: datetime
    updated_at_utc: datetime
    latest_preflight_sha256: str | None = None
    latest_preflight: PreflightReport | None = None


class FactoryIntrinsics(BaseModel):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: list[float]


class CharucoObservation(BaseModel):
    board_detected: bool
    marker_count: int
    charuco_corner_count: int
    blur_variance: float
    black_clip_percent: float
    white_clip_percent: float
    centroid_x: float | None = None
    centroid_y: float | None = None
    board_area_percent: float | None = None
    reprojection_rms_px: float | None = None
    frame_number: int
    device_timestamp_ms: float
    captured_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CaptureStatus(BaseModel):
    state: str
    session_id: str | None = None
    role: str | None = None
    serial: str | None = None
    profile: ProductionProfile | None = None
    factory_intrinsics: FactoryIntrinsics | None = None
    latest_observation: CharucoObservation | None = None
    pipeline_started: bool = False
    depth_stream_started: bool = False
    robot_command_api: bool = False
    board_definition: str = "DICT_5X5_100 · 7x5 · square 0.030 m · marker 0.022 m"
    acceptance_policy: str = "engineering-candidate-v1 · not NVIDIA acceptance criteria"


class IntrinsicsStartRequest(BaseModel):
    session_id: str
    role: str


class IntrinsicsRoleRequest(BaseModel):
    role: str


class SampleRecord(BaseModel):
    id: str
    session_id: str
    role: str
    partition: str
    accepted: bool
    reason: str
    observation: CharucoObservation
    image_sha256: str

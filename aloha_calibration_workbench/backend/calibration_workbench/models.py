from __future__ import annotations

from datetime import UTC
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class OwnershipState(str, Enum):
    FREE = "FREE"
    BUSY = "BUSY"
    UNKNOWN = "UNKNOWN"


class PreflightStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class IssueSeverity(str, Enum):
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

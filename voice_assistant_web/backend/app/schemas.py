from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RuntimeStatePayload(BaseModel):
    timestamp: float | None = None
    mode: str = "waiting"
    current_task: str | None = None
    qpos: list[float] = Field(default_factory=list)
    effort: list[float] = Field(default_factory=list)
    joint_effort: dict[str, Any] = Field(default_factory=dict)
    joint_temperature: dict[str, Any] = Field(default_factory=dict)
    latest_action: list[float] = Field(default_factory=list)
    rlt_actor_enabled: bool = False
    rlt_chunk_q_min: float | None = None
    rlt_vla_chunk_q_min: float | None = None
    rlt_actor_chunk_q_min: float | None = None


class RealtimePayload(BaseModel):
    robot: RuntimeStatePayload
    camera_status: dict[str, bool]
    camera_timestamps: dict[str, float | None] = Field(default_factory=dict)
    camera_jpeg_b64: dict[str, str] = Field(default_factory=dict)


class VoiceRequest(BaseModel):
    text: str
    language: str = "en"


class VoiceResponse(BaseModel):
    transcript: str
    reply_text: str
    task_number: str | None
    task_name: str | None
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class RLTLabelRequest(BaseModel):
    label: str


class RLTReplayStatus(BaseModel):
    replay_dir: str
    latest_episode: str | None = None
    terminal_label: str | None = None
    terminal_success: int | None = None
    num_steps: int | None = None
    num_chunks: int | None = None


class RLTTrajectoryRecord(BaseModel):
    path: str
    name: str
    terminal_label: str | None = None
    terminal_success: int | None = None
    num_steps: int
    num_chunks: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    camera_names: list[str] = Field(default_factory=list)
    trim_start_step: int
    trim_end_step: int
    mtime: float


class RLTTrajectoryListResponse(BaseModel):
    replay_dir: str
    trajectories: list[RLTTrajectoryRecord] = Field(default_factory=list)


class RLTTrajectoryTrimRequest(BaseModel):
    path: str
    trim_start_step: int
    trim_end_step: int
    terminal_label: str | None = None

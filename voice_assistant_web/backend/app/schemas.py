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
    runtime_qpos: list[float] = Field(default_factory=list)
    ros_qpos: list[float] = Field(default_factory=list)
    latest_action: list[float] = Field(default_factory=list)
    hierarchical: dict[str, Any] = Field(default_factory=dict)


class RealtimePayload(BaseModel):
    robot: RuntimeStatePayload
    camera_status: dict[str, bool]
    camera_timestamps: dict[str, float | None] = Field(default_factory=dict)


class VoiceRequest(BaseModel):
    text: str
    language: str = "en"
    dataset_dir: str | None = None
    manual_dataset_dir: str | None = None
    include_bottle_description: bool = True
    include_bottle_position: bool = False
    include_bottle_state: bool = True
    include_subtask: bool = True
    forced_low_level_subtask: str | None = None


class RuntimeConfigRequest(BaseModel):
    dataset_dir: str | None = None
    manual_dataset_dir: str | None = None
    include_bottle_description: bool | None = None
    include_bottle_position: bool | None = None
    include_bottle_state: bool | None = None
    include_subtask: bool | None = None
    forced_low_level_subtask: str | None = None


class TranslateRequest(BaseModel):
    text: str
    target_language: str


class TranslateResponse(BaseModel):
    translated_text: str


class AnnouncementAudioRequest(BaseModel):
    text: str
    target_language: str


class AnnouncementAudioResponse(BaseModel):
    translated_text: str
    audio_base64: str | None = None
    audio_mime_type: str | None = None


class VoiceResponse(BaseModel):
    transcript: str
    reply_text: str
    task_number: str | None
    task_name: str | None
    audio_base64: str | None = None
    audio_mime_type: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)

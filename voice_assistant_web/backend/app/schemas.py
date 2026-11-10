from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, StrictInt


OPERATORS = ("Jia Long", "Xinzheng He", "Zhaonan Ma")


class InferenceSettings(BaseModel):
    camera_low_position: Literal["front", "rear"] | None = None
    operator: Literal["Jia Long", "Xinzheng He", "Zhaonan Ma"] | None = None
    rating: StrictInt | None = Field(default=None, ge=1, le=5)


class HealthResponse(BaseModel):
    status: str = "ok"


class RecordingAnnotations(BaseModel):
    """Outcome labels are independent from requested inference conditioning."""
    rating: StrictInt | None = Field(default=None, ge=0, le=5)
    is_error: bool = False
    note: str = Field(default="", max_length=2000)


class RecordingCommand(BaseModel):
    """Recording does not start a task; resume is an explicit separate action."""
    action: Literal["start", "stop", "resume"]
    annotations: RecordingAnnotations = Field(default_factory=RecordingAnnotations)
    batch_name: str | None = Field(default=None, min_length=1, max_length=96, pattern=r"^\w[\w.-]*$")


class RuntimeStatePayload(BaseModel):
    inference_prompt: str | None = None
    recording: dict[str, Any] = Field(default_factory=dict)
    camera_low_position: Literal["front", "rear"] | None = None
    operator: str | None = None
    rating: int | None = None
    timestamp: float | None = None
    mode: str = "waiting"
    current_task: str | None = None
    qpos: list[float] = Field(default_factory=list)
    effort: list[float] = Field(default_factory=list)
    joint_effort: dict[str, Any] = Field(default_factory=dict)
    joint_temperature: dict[str, Any] = Field(default_factory=dict)
    latest_action: list[float] = Field(default_factory=list)


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

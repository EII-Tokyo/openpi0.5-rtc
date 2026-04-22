from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .low_level_subtask_defaults import DEFAULT_STATE_SUBTASK_PAIRS, DEFAULT_SUBTASK_CATALOG


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
    camera_jpeg_b64: dict[str, str] = Field(
        default_factory=dict,
        description="各相机 JPEG 的 base64；按 camera_refresh_ms 节流推送，空对象表示本 tick 不更新画面",
    )


class SubtaskCatalogEntryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subtask: str = Field(min_length=1, max_length=512)
    is_start_subtask: bool = False
    good_bad_action: Literal["good action", "bad action", "normal"] | None = None


class StateSubtaskPairPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bottle_state: str = Field(min_length=1, max_length=512)
    subtask: str = Field(min_length=1, max_length=512)


def _default_subtask_catalog() -> list[SubtaskCatalogEntryPayload]:
    return [SubtaskCatalogEntryPayload.model_validate(x) for x in DEFAULT_SUBTASK_CATALOG]


def _default_state_subtask_pairs() -> list[StateSubtaskPairPayload]:
    return [StateSubtaskPairPayload(bottle_state=a, subtask=b) for a, b in DEFAULT_STATE_SUBTASK_PAIRS]


class VoiceRequest(BaseModel):
    text: str
    language: str = "en"
    dataset_dir: str | None = None
    manual_dataset_dir: str | None = None
    include_bottle_description: bool = True
    lock_bottle_description: bool = True
    include_bottle_position: bool = False
    include_bottle_state: bool = True
    include_subtask: bool = True
    forced_low_level_subtask: str | None = None
    hdf5_recent_seconds: float = 5.0
    video_memory_num_frames: int = 1


class RuntimeConfigRequest(BaseModel):
    dataset_dir: str | None = None
    manual_dataset_dir: str | None = None
    include_bottle_description: bool | None = None
    lock_bottle_description: bool | None = None
    include_bottle_position: bool | None = None
    include_bottle_state: bool | None = None
    include_subtask: bool | None = None
    forced_low_level_subtask: str | None = None
    hdf5_recent_seconds: float | None = None
    video_memory_num_frames: int | None = None
    high_level_source: Literal["gpt", "service", "qwen"] | None = None
    gpt_model: str | None = None
    gpt_image_mode: Literal["high_only", "all_cameras"] | None = None
    announcement_language: Literal["zh", "ja"] | None = None
    api_base: str | None = None
    ws_base: str | None = None
    camera_refresh_ms: int | None = None
    ui_language: Literal["en", "ja", "zh"] | None = None
    subtask_catalog: list[SubtaskCatalogEntryPayload] | None = None
    state_subtask_pairs: list[StateSubtaskPairPayload] | None = None


class RuntimeConfigPayload(BaseModel):
    dataset_dir: str = ""
    manual_dataset_dir: str = ""
    include_bottle_description: bool = True
    lock_bottle_description: bool = True
    include_bottle_position: bool = False
    include_bottle_state: bool = True
    include_subtask: bool = True
    forced_low_level_subtask: str | None = None
    hdf5_recent_seconds: float = 5.0
    video_memory_num_frames: int = 1
    high_level_source: Literal["gpt", "service", "qwen"] = "gpt"
    gpt_model: str = "gpt-5.4"
    gpt_image_mode: Literal["high_only", "all_cameras"] = "all_cameras"
    announcement_language: Literal["zh", "ja"] = "zh"
    api_base: str = ""
    ws_base: str = ""
    camera_refresh_ms: int = 100
    ui_language: Literal["en", "ja", "zh"] = "en"
    subtask_catalog: list[SubtaskCatalogEntryPayload] = Field(default_factory=_default_subtask_catalog)
    state_subtask_pairs: list[StateSubtaskPairPayload] = Field(default_factory=_default_state_subtask_pairs)


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

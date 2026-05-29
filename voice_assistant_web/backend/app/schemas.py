from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RuntimeStatePayload(BaseModel):
    timestamp: float | None = None
    mode: str = "waiting"
    current_task: str | None = None
    qpos: list[float] = Field(default_factory=list)
    latest_action: list[float] = Field(default_factory=list)


class RealtimePayload(BaseModel):
    robot: RuntimeStatePayload
    camera_status: dict[str, bool]
    camera_timestamps: dict[str, float | None] = Field(default_factory=dict)
    camera_jpeg_b64: dict[str, str] = Field(default_factory=dict)
    rlt: RLTControlState


class RLTEvent(BaseModel):
    timestamp: float
    event: str
    detail: str = ""


class RLTControlState(BaseModel):
    phase: str = "idle"
    training_phase: str = "warmup"
    warmup_target: int = 100
    warmup_count: int = 0
    warmup_success: int = 0
    warmup_failure: int = 0
    warmup_attempts: int = 0
    warmup_invalid: int = 0
    auto_rollout_count: int = 0
    auto_rollout_success: int = 0
    auto_rollout_failure: int = 0
    auto_rollout_attempts: int = 0
    auto_rollout_invalid: int = 0
    actor_enabled: bool = False
    actor_ready: bool = False
    actor_effective: bool = False
    actor_locked_reason: str | None = "warmup"
    beta: float = 10.0
    intervention_scale: float = 0.25
    max_delta: float = 0.1
    active_key_region_id: str | None = None
    score_deadline: float | None = None
    last_reward: int | None = None
    last_event: str | None = None
    wandb_url: str | None = None
    critic_loss: float | None = None
    actor_loss: float | None = None
    replay_size: int | None = None
    replay_shards: int | None = None
    bad_shards: int | None = None
    actor_checkpoint_path: str | None = None
    actor_checkpoint_step: int | None = None
    rl_token_checkpoint_path: str | None = None
    events: list[RLTEvent] = Field(default_factory=list)


class RLTScoreRequest(BaseModel):
    reward: int = Field(ge=0, le=1)
    source: str = "ui"


class RLTConfigRequest(BaseModel):
    warmup_target: int | None = Field(default=None, ge=1, le=100000)
    beta: float | None = Field(default=None, ge=0, le=1000)
    actor_enabled: bool | None = None
    intervention_scale: float | None = Field(default=None, ge=0, le=1)
    max_delta: float | None = Field(default=None, ge=0, le=10)
    wandb_url: str | None = None


class RLTControlRequest(BaseModel):
    source: str = "ui"
    note: str | None = None


class RobotTaskRequest(BaseModel):
    task_num: str = Field(pattern="^[145]$")
    source: str = "ui"

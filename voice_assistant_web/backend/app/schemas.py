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
    trainer_enabled: bool = False
    trainer_running: bool = False
    actor_enabled: bool = False
    actor_ready: bool = False
    actor_effective: bool = False
    actor_locked_reason: str | None = "warmup"
    beta: float = 10.0
    auto_beta_enabled: bool = False
    auto_beta_target_delta_norm: float | None = None
    auto_beta_delta_norm_ema: float | None = None
    auto_beta_q_advantage_ema: float | None = None
    auto_beta_critic_loss_ema: float | None = None
    auto_beta_reason: str | None = None
    intervention_scale: float = 0.25
    max_delta: float = 0.1
    critic_gate_enabled: bool = False
    critic_gate_margin: float = 0.0
    critic_gate_temperature: float = 0.05
    critic_ready: bool = False
    inference_actor_active: bool = False
    inference_delta_norm: float | None = None
    inference_gate_reason: str | None = None
    key_region_probability: float | None = None
    loaded_actor_step: int | None = None
    inference_reference_q_value: float | None = None
    inference_actor_q_value: float | None = None
    inference_q_advantage: float | None = None
    active_key_region_id: str | None = None
    score_deadline: float | None = None
    last_reward: int | None = None
    last_event: str | None = None
    wandb_url: str | None = None
    critic_loss: float | None = None
    critic_q1_loss: float | None = None
    critic_q2_loss: float | None = None
    actor_loss: float | None = None
    actor_q_value: float | None = None
    reference_q_value: float | None = None
    q_advantage: float | None = None
    actor_delta_norm: float | None = None
    q1_mean: float | None = None
    q2_mean: float | None = None
    target_q_mean: float | None = None
    q_gap: float | None = None
    actor_updated: bool | None = None
    publish_actor: bool | None = None
    trainer_step: int | None = None
    steps_per_sec: float | None = None
    success_episodes: int | None = None
    failure_episodes: int | None = None
    replay_action_horizon: int | None = None
    train_action_horizon: int | None = None
    rlt_metrics_timestamp: float | None = None
    replay_size: int | None = None
    replay_shards: int | None = None
    bad_shards: int | None = None
    trainable_replay_count: int = 0
    trainable_replay_success: int = 0
    trainable_replay_failure: int = 0
    trainable_replay_samples: int = 0
    trainable_replay_shards: int = 0
    invalid_replay_shards: int = 0
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
    trainer_enabled: bool | None = None
    intervention_scale: float | None = Field(default=None, ge=0, le=1)
    max_delta: float | None = Field(default=None, ge=0, le=10)
    critic_gate_enabled: bool | None = None
    critic_gate_margin: float | None = Field(default=None, ge=-1000, le=1000)
    critic_gate_temperature: float | None = Field(default=None, gt=0, le=1000)
    wandb_url: str | None = None


class RLTControlRequest(BaseModel):
    source: str = "ui"
    note: str | None = None


class RLTDiscardRequest(BaseModel):
    source: str = "ui"
    reason: str = "operator_discard"


class RLTVoidRequest(BaseModel):
    source: str = "ui"
    reason: str = "operator_void"


class RLTBatchSegmentRequest(BaseModel):
    key_region_ids: list[str] = Field(min_length=1, max_length=500)
    source: str = "ui"
    reason: str = "operator_batch_review"


class RLTKeyRegionCropRequest(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    source: str = "ui"
    reason: str = "operator_crop"


class RLTKeyRegionCropResponse(BaseModel):
    key_region_id: str
    status: str = "committed"
    trainable: bool = True
    shard_path: str
    source_shard_path: str
    crop_start_sec: float
    crop_end_sec: float
    crop_start_sample: int
    crop_end_sample: int
    num_replay_transitions: int
    manifest_path: str


class RLTSegmentRecord(BaseModel):
    key_region_id: str
    status: str
    phase: str
    reward: int | None = None
    shard_path: str | None = None
    num_replay_transitions: int = 0
    invalid_reason: str | None = None
    created_at: float
    updated_at: float


class RLTKeyRegionReviewRecord(BaseModel):
    key_region_id: str
    status: str = "untracked"
    trainable: bool = False
    incomplete_reason: str | None = None
    phase: str | None = None
    reward: int | None = None
    shard_path: str | None = None
    npz_exists: bool = False
    video_exists: bool = False
    manifest_exists: bool = False
    rollout_path: str | None = None
    segment_status: str | None = None
    train_eligible: bool | None = None
    replay_status: str | None = None
    voided: bool | None = None
    default_video_path: str | None = None
    video_paths: list[str] = Field(default_factory=list)
    task: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    score_time: float | None = None
    duration_seconds: float | None = None
    key_region_duration_seconds: float | None = None
    key_region_start_sec: float | None = None
    key_region_end_sec: float | None = None
    fps: float | None = None
    num_frames: int | None = None
    crop_start_sec: float | None = None
    crop_end_sec: float | None = None
    crop_start_sample: int | None = None
    crop_end_sample: int | None = None
    crop_original_num_replay_transitions: int | None = None
    num_replay_transitions: int = 0
    updated_at: float | None = None


class RobotTaskRequest(BaseModel):
    task_num: str = Field(pattern="^[145]$")
    source: str = "ui"

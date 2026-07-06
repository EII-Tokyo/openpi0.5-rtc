from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RuntimeStatePayload(BaseModel):
    timestamp: float | None = None
    runtime_timestamp: float | None = None
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


class CameraCapabilitiesResponse(BaseModel):
    preferred_transport: str = "mjpeg"
    transports: list[str] = Field(default_factory=lambda: ["mjpeg", "jpeg_ws"])
    cameras: list[str] = Field(default_factory=list)
    include_realtime_frames: bool = False
    webrtc: dict[str, object] = Field(default_factory=dict)


class CameraDiagnosticsRecord(BaseModel):
    has_frame: bool = False
    frame_age_seconds: float | None = None
    source_fps_recent: float | None = None
    encoded_fps_recent: float | None = None
    raw_frames_total: int = 0
    encoded_frames_total: int = 0
    dropped_frames_total: int = 0
    error_count: int = 0
    last_error: str | None = None
    last_encoding: str | None = None
    last_width: int | None = None
    last_height: int | None = None
    latest_jpeg_bytes: int = 0
    encode_ms_mean_recent: float | None = None
    encode_ms_max_recent: float | None = None


class CameraDiagnosticsResponse(BaseModel):
    bridge_running: bool = False
    bridge_error: str | None = None
    jpeg_quality: int = 70
    cameras: dict[str, CameraDiagnosticsRecord] = Field(default_factory=dict)


class CameraWebRTCSessionRequest(BaseModel):
    cameras: list[str] = Field(min_length=1, max_length=4)
    codec: str = "h264"


class CameraWebRTCSessionResponse(BaseModel):
    session_id: str
    status: str = "signaling"
    cameras: list[str]
    signaling_url: str
    expires_at: float
    fallback_transport: str = "mjpeg"
    message: str | None = None


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
    online_safety_enabled: bool = True
    online_safety_phase: str | None = "idle_wait_new_data"
    online_round_index: int | None = 0
    online_last_committed_shards: int | None = 0
    online_last_committed_success: int | None = 0
    online_last_committed_failure: int | None = 0
    online_round_start_shards: int | None = 0
    online_round_start_success: int | None = 0
    online_round_start_failure: int | None = 0
    online_critic_steps_remaining: int | None = 0
    online_actor_steps_remaining: int | None = 0
    online_best_critic_auc: float | None = None
    online_best_critic_q_gap: float | None = None
    online_rejection_reason: str | None = None
    online_target_delta_norm: float | None = 0.04
    online_min_new_shards_per_round: int = 10
    online_min_new_success_per_round: int = 5
    online_min_new_failure_per_round: int = 5
    online_critic_updates_per_round: int = 500
    online_actor_updates_per_round: int = 300
    online_critic_auc_min: float = 0.70
    online_critic_max_auc_drop: float = 0.02
    online_require_positive_q_gap: bool = True
    online_actor_max_delta_norm: float = 0.09
    online_actor_min_q_advantage: float = 0.0
    online_beta_initial: float = 30.0
    online_beta_min: float = 5.0
    online_beta_max: float = 30.0
    online_beta_decay_on_actor_accept: float = 0.9
    online_beta_increase_on_reject: float = 1.25
    online_target_delta_initial: float = 0.04
    online_target_delta_max: float = 0.10
    online_target_delta_increment: float = 0.01
    online_auto_train_critic: bool = False
    online_auto_train_actor: bool = False
    actor_enabled: bool = False
    actor_ready: bool = False
    actor_effective: bool = False
    force_actor_effective: bool = False
    actor_locked_reason: str | None = "warmup"
    beta: float = 10.0
    auto_beta_enabled: bool = True
    auto_beta_target_delta_norm: float | None = 0.06
    auto_beta_min: float = 1.0
    auto_beta_max: float = 30.0
    auto_beta_lr: float = 0.03
    auto_beta_ema_decay: float = 0.8
    auto_beta_update_interval: int = 100
    auto_beta_q_margin: float = 0.01
    auto_beta_delta_norm_ema: float | None = None
    auto_beta_q_advantage_ema: float | None = None
    auto_beta_critic_loss_ema: float | None = None
    auto_beta_reason: str | None = None
    intervention_scale: float = 0.25
    max_delta: float = 0.1
    actor_handoff_steps: int = 4
    actor_delta_ema_alpha: float = 0.35
    actor_speed_limit_preset: Literal["off", "80", "50", "20"] = "off"
    critic_gate_enabled: bool = True
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
    critic_burn_in_steps: int | None = 1000
    target_sync_step: int | None = None
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


class RLTCriticReportSummary(BaseModel):
    exists: bool = False
    round_id: str | None = None
    source_path: str | None = None
    report_path: str | None = None
    updated_at: float | None = None
    step: int | None = None
    auc: float | None = None
    q_gap: float | None = None
    success_q_mean: float | None = None
    failure_q_mean: float | None = None
    holdout_bellman_loss: float | None = None
    success_transitions: int | None = None
    failure_transitions: int | None = None
    is_critic_usable: bool | None = None
    warning_reason: str | None = None


class RLTScoreRequest(BaseModel):
    reward: int = Field(ge=0, le=1)
    source: str = "ui"


class RLTConfigRequest(BaseModel):
    warmup_target: int | None = Field(default=None, ge=1, le=100000)
    beta: float | None = Field(default=None, ge=0, le=1000)
    auto_beta_enabled: bool | None = None
    auto_beta_target_delta_norm: float | None = Field(default=None, gt=0, le=10)
    auto_beta_min: float | None = Field(default=None, gt=0, le=1000)
    auto_beta_max: float | None = Field(default=None, gt=0, le=1000)
    auto_beta_lr: float | None = Field(default=None, ge=0, le=10)
    auto_beta_ema_decay: float | None = Field(default=None, ge=0, lt=1)
    auto_beta_update_interval: int | None = Field(default=None, ge=1, le=100000)
    auto_beta_q_margin: float | None = Field(default=None, ge=-1000, le=1000)
    critic_burn_in_steps: int | None = Field(default=None, ge=0, le=1000000)
    actor_enabled: bool | None = None
    force_actor_effective: bool | None = None
    trainer_enabled: bool | None = None
    online_safety_enabled: bool | None = None
    online_min_new_shards_per_round: int | None = Field(default=None, ge=1, le=100000)
    online_min_new_success_per_round: int | None = Field(default=None, ge=0, le=100000)
    online_min_new_failure_per_round: int | None = Field(default=None, ge=0, le=100000)
    online_critic_updates_per_round: int | None = Field(default=None, ge=1, le=1000000)
    online_actor_updates_per_round: int | None = Field(default=None, ge=1, le=1000000)
    online_critic_auc_min: float | None = Field(default=None, ge=0, le=1)
    online_critic_max_auc_drop: float | None = Field(default=None, ge=0, le=1)
    online_require_positive_q_gap: bool | None = None
    online_actor_max_delta_norm: float | None = Field(default=None, gt=0, le=1000)
    online_actor_min_q_advantage: float | None = Field(default=None, ge=-1000, le=1000)
    online_beta_initial: float | None = Field(default=None, gt=0, le=1000)
    online_beta_min: float | None = Field(default=None, gt=0, le=1000)
    online_beta_max: float | None = Field(default=None, gt=0, le=1000)
    online_beta_decay_on_actor_accept: float | None = Field(default=None, gt=0, le=1)
    online_beta_increase_on_reject: float | None = Field(default=None, ge=1, le=100)
    online_target_delta_initial: float | None = Field(default=None, gt=0, le=1000)
    online_target_delta_max: float | None = Field(default=None, gt=0, le=1000)
    online_target_delta_increment: float | None = Field(default=None, ge=0, le=1000)
    online_auto_train_critic: bool | None = None
    online_auto_train_actor: bool | None = None
    intervention_scale: float | None = Field(default=None, ge=0, le=1)
    max_delta: float | None = Field(default=None, ge=0, le=10)
    actor_handoff_steps: int | None = Field(default=None, ge=0, le=50)
    actor_delta_ema_alpha: float | None = Field(default=None, ge=0, le=1)
    actor_speed_limit_preset: Literal["off", "80", "50", "20"] | None = None
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


class RLTKeyRegionRescoreRequest(BaseModel):
    reward: int = Field(ge=0, le=1)
    source: str = "ui"
    reason: str = "operator_rescore"


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
    batch: str | None = None
    status: str = "untracked"
    trainable: bool = False
    needs_crop: bool = False
    incomplete_reason: str | None = None
    phase: str | None = None
    reward: int | None = None
    shard_path: str | None = None
    npz_exists: bool = False
    video_exists: bool = False
    manifest_exists: bool = False
    rollout_path: str | None = None
    local_rollout_path: str | None = None
    local_shard_path: str | None = None
    actor_inference_kind: str | None = None
    actor_delta_p95: float | None = None
    actor_delta_max: float | None = None
    actor_delta_mean: float | None = None
    has_intervention_metadata: bool = False
    has_action_source: bool = False
    has_takeover_id: bool = False
    segment_status: str | None = None
    train_eligible: bool | None = None
    replay_status: str | None = None
    replay_state_grain: str | None = None
    requires_offline_reencode: bool | None = None
    formal_replay_state_grain: str | None = None
    formal_replay_ready: bool | None = None
    conversion_status: str | None = None
    conversion_reason: str | None = None
    missing_rlt_metadata: list[str] = Field(default_factory=list)
    voided: bool | None = None
    default_video_path: str | None = None
    video_paths: list[str] = Field(default_factory=list)
    task: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    score_time: float | None = None
    review_datetime: str | None = None
    start_datetime: str | None = None
    score_datetime: str | None = None
    crop_datetime: str | None = None
    updated_datetime: str | None = None
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


class RLTKeyRegionReviewSummary(BaseModel):
    total: int = 0
    trainable: int = 0
    needs_crop: int = 0
    formal_replay_ready: int = 0
    needs_offline_reencode: int = 0
    legacy_unmarked: int = 0
    success: int = 0
    failure: int = 0
    replay_samples: int = 0


class RLTKeyRegionReviewPage(BaseModel):
    items: list[RLTKeyRegionReviewRecord] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0
    next_offset: int | None = None
    summary: RLTKeyRegionReviewSummary = Field(default_factory=RLTKeyRegionReviewSummary)
    batches: list[str] = Field(default_factory=list)


class RLTExpertDemoRecord(BaseModel):
    episode_key: str
    dataset_id: str
    episode_index: int
    fps: float | None = None
    num_frames: int | None = None
    duration_seconds: float | None = None
    video_paths: list[str] = Field(default_factory=list)
    local_video_paths: list[str] = Field(default_factory=list)
    video_start_secs: list[float] = Field(default_factory=list)
    camera_count: int = 0
    missing_cameras: list[str] = Field(default_factory=list)
    camera_complete: bool = False
    source_dataset_path: str
    saved_crop_count: int = 0
    saved_crop_start_sec: float | None = None
    saved_crop_end_sec: float | None = None
    saved_crop_reward: int | None = None


class RLTExpertDemoCropSummary(BaseModel):
    total_episodes: int = 0
    cropped_episodes: int = 0
    remaining_episodes: int = 0
    saved_crops: int = 0


class RLTExpertDemoPage(BaseModel):
    items: list[RLTExpertDemoRecord] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0
    next_offset: int | None = None
    datasets: list[str] = Field(default_factory=list)
    crop_summary: RLTExpertDemoCropSummary = Field(default_factory=RLTExpertDemoCropSummary)


class RLTExpertDemoCropRequest(BaseModel):
    start_sec: float
    end_sec: float
    reward: int = 1


class RLTExpertDemoCropResponse(BaseModel):
    dataset_id: str
    episode_index: int
    start_sec: float
    end_sec: float
    reward: int = 1
    label: str = "expert"
    metadata_path: str | None = None


class RLTPreferenceStats(BaseModel):
    total_preferences: int = 0
    left_wins: int = 0
    right_wins: int = 0
    ties: int = 0
    both_bad: int = 0
    skipped: int = 0


class RLTPreferencePairResponse(BaseModel):
    left: RLTKeyRegionReviewRecord | None = None
    right: RLTKeyRegionReviewRecord | None = None
    stats: RLTPreferenceStats = Field(default_factory=RLTPreferenceStats)
    remaining_unseen_pairs: int = 0
    pair_type: str | None = None
    strategy: str = "budgeted"
    round_budget: int = 800
    round_labeled: int = 0
    round_remaining: int = 0


class RLTPreferenceRequest(BaseModel):
    left_key_region_id: str
    right_key_region_id: str
    preference: str = Field(pattern="^(left|right|tie|both_bad|skip)$")
    reason_tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    source: str = "ui"


class RLTPreferenceRecord(BaseModel):
    id: int
    left_key_region_id: str
    right_key_region_id: str
    pair_key: str
    preference: str
    pair_type: str | None = None
    strategy: str = "budgeted"
    sample_round: int = 1
    reason_tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    source: str = "ui"
    created_at: float


class RobotTaskRequest(BaseModel):
    task_num: str = Field(pattern="^[1459]$")
    source: str = "ui"

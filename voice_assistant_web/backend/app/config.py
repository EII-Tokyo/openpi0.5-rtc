from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class Settings:
    enable_ros: bool = os.getenv("EII_PILOT_ENABLE_ROS", "true").lower() not in {"0", "false", "no", "off"}
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    runtime_state_channel: str = os.getenv("RUNTIME_STATE_CHANNEL", "aloha_runtime_state")
    rlt_control_channel: str = os.getenv("RLT_CONTROL_CHANNEL", "aloha_rlt_control")
    rlt_state_channel: str = os.getenv("RLT_STATE_CHANNEL", "aloha_rlt_state")
    rlt_state_latest_key: str = field(
        default_factory=lambda: os.getenv("RLT_STATE_LATEST_KEY", f"{os.getenv('RLT_STATE_CHANNEL', 'aloha_rlt_state')}:latest")
    )
    camera_jpeg_quality: int = int(os.getenv("CAMERA_JPEG_QUALITY", "70"))
    realtime_hz: float = float(os.getenv("REALTIME_HZ", "10"))
    rollouts_root: str = os.getenv("ROLLOUTS_ROOT", "/app/rollouts")
    replay_root: str = os.getenv("REPLAY_ROOT", "/app/replay")
    rlt_state_path: str = os.getenv("RLT_STATE_PATH", "/app/segment_db/rlt_control_state.json")
    rlt_segment_db_path: str = os.getenv("RLT_SEGMENT_DB_PATH", "/app/segment_db/segments.sqlite3")
    rlt_default_warmup_target: int = int(os.getenv("RLT_DEFAULT_WARMUP_TARGET", "100"))
    rlt_default_beta: float = float(os.getenv("RLT_DEFAULT_BETA", "10.0"))
    rlt_default_intervention_scale: float = float(os.getenv("RLT_DEFAULT_INTERVENTION_SCALE", "0.25"))
    rlt_default_max_delta: float = float(os.getenv("RLT_DEFAULT_MAX_DELTA", "0.1"))
    rlt_rl_token_checkpoint_path: str = os.getenv(
        "RLT_RL_TOKEN_CHECKPOINT_PATH",
        "/app/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query/rl_token_2048_enc4_dec4_query_from_19000_20260528/12000",
    )
    allow_origins: list[str] = field(default_factory=lambda: os.getenv("ALLOW_ORIGINS", "*").split(","))


settings = Settings()

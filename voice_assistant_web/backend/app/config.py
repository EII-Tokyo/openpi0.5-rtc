from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


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
    camera_transport: str = os.getenv("EII_CAMERA_TRANSPORT", "mjpeg")
    realtime_include_camera_frames: bool = _env_bool("EII_REALTIME_INCLUDE_CAMERA_FRAMES", "false")
    camera_webrtc_enabled: bool = _env_bool("EII_CAMERA_WEBRTC_ENABLED", "false")
    camera_mjpeg_default_fps: float = float(os.getenv("EII_CAMERA_MJPEG_DEFAULT_FPS", "20"))
    camera_mjpeg_max_fps: float = float(os.getenv("EII_CAMERA_MJPEG_MAX_FPS", "30"))
    camera_webrtc_session_ttl_seconds: float = float(os.getenv("EII_CAMERA_WEBRTC_SESSION_TTL_SECONDS", "30"))
    camera_webrtc_max_sessions: int = int(os.getenv("EII_CAMERA_WEBRTC_MAX_SESSIONS", "4"))
    camera_webrtc_media_url: str = os.getenv("EII_CAMERA_WEBRTC_MEDIA_URL", "http://127.0.0.1:8013")
    realtime_hz: float = float(os.getenv("REALTIME_HZ", "10"))
    rollouts_root: str = os.getenv("ROLLOUTS_ROOT", "/app/rollouts")
    replay_root: str = os.getenv("REPLAY_ROOT", "/app/replay")
    rlt_online_run_root: str = os.getenv("RLT_ONLINE_RUN_ROOT", "/app/rlt_online/run")
    rlt_state_path: str = os.getenv("RLT_STATE_PATH", "/app/segment_db/rlt_control_state.json")
    rlt_segment_db_path: str = os.getenv("RLT_SEGMENT_DB_PATH", "/app/segment_db/segments.sqlite3")
    rlt_default_warmup_target: int = int(os.getenv("RLT_DEFAULT_WARMUP_TARGET", "100"))
    rlt_default_beta: float = float(os.getenv("RLT_DEFAULT_BETA", "10.0"))
    rlt_default_intervention_scale: float = float(os.getenv("RLT_DEFAULT_INTERVENTION_SCALE", "0.25"))
    rlt_default_max_delta: float = float(os.getenv("RLT_DEFAULT_MAX_DELTA", "0.1"))
    rlt_default_actor_handoff_steps: int = int(os.getenv("RLT_DEFAULT_ACTOR_HANDOFF_STEPS", "4"))
    rlt_default_actor_delta_ema_alpha: float = float(os.getenv("RLT_DEFAULT_ACTOR_DELTA_EMA_ALPHA", "0.35"))
    rlt_default_actor_speed_limit_preset: str = os.getenv("RLT_DEFAULT_ACTOR_SPEED_LIMIT_PRESET", "off")
    rlt_rl_token_checkpoint_path: str = os.getenv(
        "RLT_RL_TOKEN_CHECKPOINT_PATH",
        "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint",
    )
    allow_origins: list[str] = field(default_factory=lambda: os.getenv("ALLOW_ORIGINS", "*").split(","))


settings = Settings()

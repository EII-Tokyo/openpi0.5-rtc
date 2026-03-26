from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class Settings:
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    runtime_state_channel: str = os.getenv("RUNTIME_STATE_CHANNEL", "aloha_runtime_state")
    voice_command_channel: str = os.getenv("VOICE_COMMAND_CHANNEL", "aloha_voice_commands")
    camera_jpeg_quality: int = int(os.getenv("CAMERA_JPEG_QUALITY", "70"))
    realtime_hz: float = float(os.getenv("REALTIME_HZ", "10"))
    allow_origins: list[str] = field(default_factory=lambda: os.getenv("ALLOW_ORIGINS", "*").split(","))
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_tts_model: str = os.getenv("OPENAI_TTS_MODEL", "tts-1-hd")
    openai_tts_voice: str = os.getenv("OPENAI_TTS_VOICE", "nova")
    openai_transcription_model: str = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")


settings = Settings()

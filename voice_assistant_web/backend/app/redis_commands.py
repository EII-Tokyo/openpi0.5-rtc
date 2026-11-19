from __future__ import annotations

import json
import time
import os
import re
import uuid

import redis

from .config import settings
from .schemas import InferenceSettings

INFERENCE_SETTINGS_KEY = "aloha_inference_settings"


def read_inference_settings(redis_client) -> InferenceSettings:
    raw = redis_client.get(INFERENCE_SETTINGS_KEY)
    return InferenceSettings.model_validate_json(raw) if raw else InferenceSettings()

TASK_MAPPING = {
    "1": "Twist off the bottle cap",
    "2": "Rinse bottle",
    "3": "Stop and human hand control",
    "4": "Return to home position and save hdf5",
    "5": "Return to sleep position, save hdf5 and quit robot runtime",
    "6": "Leader follower demo",
}


EXP_INSERT = os.getenv("ALOHA_TASK_MODE", "legacy") == "exp_insert"
if EXP_INSERT:
    TASK_MAPPING.pop("1")
    TASK_MAPPING.pop("2")
    TASK_MAPPING.update({f"bottle_{i}": f"Execute the full insertion cycle with bottle {i}" for i in (1, 2, 3)})


def bottle_task_from_text(text: str) -> str | None:
    """Only unambiguous bottle selections; never reinterpret a manual/stop request."""
    if not EXP_INSERT:
        return None
    compact = re.sub(r"[\s。.!！,，]", "", text.lower())
    match = re.fullmatch(r"(?:bottle)?([123一二三])(?:号(?:瓶子|瓶)?|号的瓶子)?", compact)
    if not match:
        return None
    number = {"一": "1", "二": "2", "三": "3"}.get(match[1], match[1])
    return f"bottle_{number}"


def create_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def publish_task(redis_client: redis.Redis, task_num: str) -> dict:
    task_name = TASK_MAPPING[task_num]
    message = {
        "task": task_num,
        "task_name": task_name,
        "timestamp": time.time(),
        "command_id": str(uuid.uuid4()),
    }
    if task_num in {"1", "2"} or task_num.startswith("bottle_"):
        message.update(read_inference_settings(redis_client).model_dump())
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

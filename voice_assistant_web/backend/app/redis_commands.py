from __future__ import annotations

import json
import time

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
    }
    if task_num in {"1", "2"}:
        message.update(read_inference_settings(redis_client).model_dump())
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

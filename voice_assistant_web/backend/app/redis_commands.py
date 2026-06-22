from __future__ import annotations

import json
import time

import redis

from .config import settings

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


def publish_task(redis_client: redis.Redis, task_num: str, **extra_fields) -> dict:
    task_name = TASK_MAPPING.get(task_num, str(extra_fields.get("task_name", task_num)))
    message = {
        "task": task_num,
        "task_name": task_name,
        "timestamp": time.time(),
        **extra_fields,
    }
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

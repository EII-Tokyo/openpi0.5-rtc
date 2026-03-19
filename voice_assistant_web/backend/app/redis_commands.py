from __future__ import annotations

import json
import time

import redis

from .config import settings

TASK_MAPPING = {
    "1": "process all bottles",
    "2": "Stop and human hand control",
    "3": "Return to home position and save hdf5",
    "4": "Return to sleep position, save hdf5 and quit robot runtime",
}


def create_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def publish_task(
    redis_client: redis.Redis,
    task_num: str,
    *,
    dataset_dir: str | None = None,
    manual_dataset_dir: str | None = None,
) -> dict:
    task_name = TASK_MAPPING[task_num]
    message = {
        "task": task_num,
        "task_name": task_name,
        "timestamp": time.time(),
    }
    if dataset_dir:
        message["dataset_dir"] = dataset_dir
    if manual_dataset_dir:
        message["manual_dataset_dir"] = manual_dataset_dir
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

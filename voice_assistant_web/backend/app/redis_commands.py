from __future__ import annotations

import json
import time

import redis

from .config import settings

TASK_MAPPING = {
    "1": "Remove the label from the bottle with the knife in the right hand.",
    "2": "Do the followings: 1. If the bottle cap is facing left, rotate the bottle 180 degrees. 2. Pick up the bottle. 3. Twist off the bottle cap if the bottle has a cap. 4. Put the bottle into the box on the left. 5. Put the cap into the box on the right. If the bottle cap falls onto the table, pick it up. 6. Return to home position.",
    "3": "Stop and human hand control",
    "4": "Return to home position and save hdf5",
    "5": "Return to sleep position, save hdf5 and quit robot runtime",
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
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

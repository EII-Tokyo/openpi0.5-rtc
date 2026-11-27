from __future__ import annotations

import json
import time
import re
import uuid

import redis
from fastapi import HTTPException

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


TASK_MAPPING.update({f"bottle_{i}": f"Execute the full insertion cycle with bottle {i}" for i in (1, 2, 3)})


def runtime_capabilities(redis_client):
    raw = redis_client.get("aloha_task_capabilities")
    return json.loads(raw) if raw else None


def runtime_supports_bottles(redis_client):
    capabilities = runtime_capabilities(redis_client)
    return capabilities is not None and capabilities.get("mode") == "exp_insert"


def bottle_task_from_text(text: str) -> str | None:
    """Only unambiguous bottle selections; never reinterpret a manual/stop request."""
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


def publish_task(redis_client: redis.Redis, task_num: str, *, authorize_sol: bool = False) -> dict:
    # Stop/home/manual controls never depend on optional model capability state.
    capabilities = runtime_capabilities(redis_client) if task_num not in {"3", "4", "5", "6"} else None
    if capabilities is not None and task_num not in capabilities.get("task_ids", []):
        raise HTTPException(status_code=409, detail="The loaded model does not support this task; select a compatible model")
    if task_num.startswith("bottle_") and (capabilities is None or capabilities.get("mode") != "exp_insert"):
        raise HTTPException(status_code=409, detail="No active policy with automatic bottle-cycle support")
    task_name = TASK_MAPPING[task_num]
    message = {
        "task": task_num,
        "task_name": task_name,
        "timestamp": time.time(),
        "command_id": str(uuid.uuid4()),
    }
    if task_num in {"1", "2"} or task_num.startswith("bottle_"):
        message.update(read_inference_settings(redis_client).model_dump())
    if authorize_sol:
        from openpi.shared import sol_run_grant

        grant = message["command_id"]
        message["sol_run_grant"] = grant
        with redis_client.pipeline() as transaction:
            transaction.set(sol_run_grant.KEY, grant, ex=sol_run_grant.TTL_SECONDS)
            transaction.delete(sol_run_grant.CALIBRATION_KEY)
            transaction.publish(settings.voice_command_channel, json.dumps(message))
            _, _, listeners = transaction.execute()
        if not listeners:
            sol_run_grant.revoke(redis_client, grant)
            raise HTTPException(503, "Robot control service is offline")
    else:
        redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

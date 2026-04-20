from __future__ import annotations

import json
import time
from typing import Any

import redis

from .config import settings

TASK_MAPPING = {
    "1": "Process all bottles",
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
    include_bottle_description: bool = True,
    lock_bottle_description: bool = True,
    include_bottle_position: bool = False,
    include_bottle_state: bool = True,
    include_subtask: bool = True,
    forced_low_level_subtask: str | None = None,
    hdf5_recent_seconds: float = 5.0,
    video_memory_num_frames: int = 1,
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
    message["include_bottle_description"] = bool(include_bottle_description)
    message["lock_bottle_description"] = bool(lock_bottle_description)
    message["include_bottle_position"] = bool(include_bottle_position)
    message["include_bottle_state"] = bool(include_bottle_state)
    message["include_subtask"] = bool(include_subtask)
    message["hdf5_recent_seconds"] = max(0.0, float(hdf5_recent_seconds))
    message["video_memory_num_frames"] = int(video_memory_num_frames) if int(video_memory_num_frames) in (1, 4) else 1
    if isinstance(forced_low_level_subtask, str) and forced_low_level_subtask.strip():
        message["forced_low_level_subtask"] = forced_low_level_subtask.strip()
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message


def _serialize_subtask_catalog(catalog: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in catalog or []:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump(mode="json"))
        elif isinstance(e, dict):
            out.append(dict(e))
    return out


def _serialize_state_subtask_pairs(pairs: Any) -> list[list[str]]:
    out: list[list[str]] = []
    for p in pairs or []:
        if hasattr(p, "model_dump"):
            d = p.model_dump(mode="json")
            out.append([str(d.get("bottle_state", "")), str(d.get("subtask", ""))])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([str(p[0]), str(p[1])])
        elif isinstance(p, dict):
            out.append([str(p.get("bottle_state", "")), str(p.get("subtask", ""))])
    return out


def publish_runtime_config(
    redis_client: redis.Redis,
    *,
    dataset_dir: str | None = None,
    manual_dataset_dir: str | None = None,
    include_bottle_description: bool | None = None,
    lock_bottle_description: bool | None = None,
    include_bottle_position: bool | None = None,
    include_bottle_state: bool | None = None,
    include_subtask: bool | None = None,
    forced_low_level_subtask: str | None = None,
    hdf5_recent_seconds: float | None = None,
    video_memory_num_frames: int | None = None,
    high_level_source: str | None = None,
    gpt_model: str | None = None,
    gpt_image_mode: str | None = None,
    subtask_catalog: Any = None,
    state_subtask_pairs: Any = None,
) -> dict:
    message = {
        "timestamp": time.time(),
        "config_only": True,
    }
    if isinstance(dataset_dir, str):
        message["dataset_dir"] = dataset_dir.strip()
    if isinstance(manual_dataset_dir, str):
        message["manual_dataset_dir"] = manual_dataset_dir.strip()
    if isinstance(include_bottle_description, bool):
        message["include_bottle_description"] = include_bottle_description
    if isinstance(lock_bottle_description, bool):
        message["lock_bottle_description"] = lock_bottle_description
    if isinstance(include_bottle_position, bool):
        message["include_bottle_position"] = include_bottle_position
    if isinstance(include_bottle_state, bool):
        message["include_bottle_state"] = include_bottle_state
    if isinstance(include_subtask, bool):
        message["include_subtask"] = include_subtask
    if hdf5_recent_seconds is not None:
        message["hdf5_recent_seconds"] = max(0.0, float(hdf5_recent_seconds))
    if isinstance(video_memory_num_frames, int) and video_memory_num_frames in (1, 4):
        message["video_memory_num_frames"] = video_memory_num_frames
    if high_level_source in {"gpt", "service"}:
        message["high_level_source"] = high_level_source
    if isinstance(gpt_model, str) and gpt_model.strip():
        message["gpt_model"] = gpt_model.strip()
    if gpt_image_mode in {"high_only", "all_cameras"}:
        message["gpt_image_mode"] = gpt_image_mode
    if forced_low_level_subtask is None:
        message["forced_low_level_subtask"] = None
    elif isinstance(forced_low_level_subtask, str):
        message["forced_low_level_subtask"] = forced_low_level_subtask.strip()
    if subtask_catalog is not None:
        message["subtask_catalog"] = _serialize_subtask_catalog(subtask_catalog)
    if state_subtask_pairs is not None:
        message["state_subtask_pairs"] = _serialize_state_subtask_pairs(state_subtask_pairs)
    redis_client.publish(settings.voice_command_channel, json.dumps(message))
    return message

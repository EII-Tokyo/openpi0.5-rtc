"""Publish the effective launch settings for the local robot web UI."""

import dataclasses
import datetime
import enum
import json
import logging
import os
from pathlib import Path
import tempfile

ENV_KEYS = (
    "OPENPI_DEFER_INITIAL_HOME", "OPENPI_RINSE_RUNTIME",
    "ALOHA_HOME_POSITION_TOLERANCE", "ALOHA_HOME_DEPARTURE_TOLERANCE",
    "ALOHA_HOME_VELOCITY_TOLERANCE", "ALOHA_HOME_STABLE_SECONDS",
    "ALOHA_ROLLOUT_DIR", "ALOHA_FOOT_SWITCH", "ALOHA_TASK_1_PROMPT", "ALOHA_TASK_2_PROMPT",
)
CHECKPOINT_KEYS = ("adapt_to_pi", "action_horizon", "action_dim", "joint_position_history", "task_catalog_version")
SOL_KEYS = (
    "enabled", "model", "camera_keys", "image_history_seconds", "wire_protocol",
    "prompt_version", "archive_required", "reset_protocol", "request_interval_seconds",
    "reply_expiry_seconds", "request_schedule",
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def arguments(value):
    return dataclasses.asdict(value, dict_factory=lambda items: {
        key: val.value if isinstance(val, enum.Enum) else val for key, val in items
    })


def clear_snapshot(path):
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logging.warning("Could not remove runtime launch parameters", exc_info=True)


def publish_snapshot(path, args, started_at, metadata, video_input):
    if path is None:
        return
    runtime = metadata["runtime"]
    policy = metadata.get("policy_launch", {})
    high_level = runtime.get("high_level", {})
    snapshot = {
        "captured_at": utc_now(),
        "runtime_started_at": started_at,
        "server_started_at": policy.get("started_at"),
        "runtime": arguments(args),
        "policy_server": policy.get("parameters", {}),
        "checkpoint": {
            **video_input,
            **{key: runtime[key] for key in CHECKPOINT_KEYS if key in runtime},
        },
        "sol": ({key: high_level[key] for key in SOL_KEYS if key in high_level}
                if high_level else {"enabled": False}),
        "runtime_environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(snapshot, output, indent=2, allow_nan=False)
            output.write("\n")
        temporary.chmod(0o644)
        temporary.replace(path)
    except (OSError, ValueError, TypeError):
        logging.warning("Could not publish runtime launch parameters", exc_info=True)
    finally:
        if temporary is not None:
            clear_snapshot(temporary)

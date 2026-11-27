"""Wire the reviewed rinse protocol to the real-robot memory supervisor."""
import os
import uuid

from openpi_client.runtime.long_memory_high_level import (
    CAMERAS,
    GptMemoryBackend,
    LongMemoryPolicy,
    MemorySupervisor,
    PROMPT_VERSION,
)

from openpi.serving.high_level_status import RedisHighLevelStatus
from openpi.shared.completed_step_memory import INSERT_AND_RINSE
from openpi.shared.long_memory import LongMemoryConfig
from openpi.shared import sol_run_grant


def fast_usage_cost(usage):
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    details = usage.get("input_tokens_details", {})
    hits = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    write_details = details.get("cache_write_tokens_details", {})
    long_writes = write_details.get("ttl_12h_tokens", 0) + max(0, writes - sum(write_details.values()))
    uncached = max(0, usage["input_tokens"] - hits - writes)
    return (uncached * 2 + hits * .1 + (writes - long_writes) * 2.5
            + long_writes * 5 + usage["output_tokens"] * 10) * 2 / 1_000_000


class RinseBackend(GptMemoryBackend):
    def __init__(self, *, status_publisher, **kwargs):
        self.dwell_token = None
        super().__init__(status_publisher=status_publisher, dwell_confirmed=self._confirmed, **kwargs)

    def _confirmed(self):
        return bool(self.dwell_token and self.status_publisher.client.get(
            "aloha:rinse:dwell:" + self.dwell_token
        ))

    def _set_dwell_token(self, token):
        if self.dwell_token == token:
            return
        self.dwell_token = token
        self.notify("dwell", {"token": token})

    def applied_memory(self, state):
        if state.current_task == INSERT_AND_RINSE:
            if self.dwell_token is None:
                self._set_dwell_token(uuid.uuid4().hex)
        else:
            self._set_dwell_token(None)

    def applied_nozzles(self, tips, grant):
        self.notify("nozzle", {"tips": tips, "grant": grant})

    def reset_status(self, epoch):
        self._set_dwell_token(None)
        super().reset_status(epoch)


def install(policy, metadata):
    runtime = metadata["runtime"]
    video = runtime["video_input"]
    settings = runtime["long_memory"]
    if (not settings or not runtime["training_time_rtc"] or not runtime["nozzle_conditioning"]
            or video["num_frames"] != 1 or set(video["camera_keys"]) != set(CAMERAS)):
        raise ValueError("Sol Fast rinse requires the four-camera RTC rinse checkpoint")
    settings = LongMemoryConfig(**{**settings, "update_seconds": 1.0, "video_hz": 4.0})
    status = RedisHighLevelStatus.from_environment(
        model="gpt-6.2-sol", checkpoint_step=video["checkpoint_step"], interval_seconds=settings.update_seconds
    )
    status.state.update(service_tier="Fast", dwell_token=None)
    backend = RinseBackend(
        archive_root=os.environ["OPENPI_HIGH_LEVEL_DATASET_DIR"], status_publisher=status,
        authorization=lambda grant: sol_run_grant.active(status.client, grant),
        cost_from_usage=fast_usage_cost
    )
    supervisor = MemorySupervisor(backend, settings=settings)
    runtime["high_level"] = {
        "enabled": True, "reset_protocol": True, "model": backend.model, "service_tier": "fast",
        "prompt_version": PROMPT_VERSION, "camera_keys": list(CAMERAS), "image_history_seconds": 0,
    }
    return LongMemoryPolicy(policy, supervisor)

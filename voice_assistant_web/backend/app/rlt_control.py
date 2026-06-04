
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import threading
import time
from typing import Any
import uuid

import redis

from .config import settings
from .rlt_segment_ledger import RLTSegmentLedger
from .schemas import RLTConfigRequest
from .schemas import RLTControlRequest
from .schemas import RLTControlState
from .schemas import RLTEvent


class RLTControlStore:
    def __init__(self, redis_client: redis.Redis, segment_ledger: RLTSegmentLedger | None = None) -> None:
        self._redis = redis_client
        self._lock = threading.Lock()
        self._state_path = Path(settings.rlt_state_path)
        self._segment_ledger = segment_ledger or RLTSegmentLedger(settings.rlt_segment_db_path)
        self._running = False
        self._redis_thread: threading.Thread | None = None
        self._state = RLTControlState(
            warmup_target=settings.rlt_default_warmup_target,
            beta=settings.rlt_default_beta,
            intervention_scale=settings.rlt_default_intervention_scale,
            max_delta=settings.rlt_default_max_delta,
            rl_token_checkpoint_path=settings.rlt_rl_token_checkpoint_path,
        )
        self._load()
        if self._state.rl_token_checkpoint_path is None:
            self._state.rl_token_checkpoint_path = settings.rlt_rl_token_checkpoint_path
        with self._lock:
            self._apply_ledger_stats_locked()
            self._refresh_derived_locked()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._redis_thread = threading.Thread(target=self._listen_runtime_state, daemon=True)
        self._redis_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._redis_thread and self._redis_thread.is_alive():
            self._redis_thread.join(timeout=1.0)

    def snapshot(self) -> RLTControlState:
        with self._lock:
            self._apply_score_timeout_locked()
            self._apply_ledger_stats_locked()
            self._refresh_derived_locked()
            return self._state.model_copy(deep=True)

    def start_key_region(self, request: RLTControlRequest) -> RLTControlState:
        with self._lock:
            self._apply_score_timeout_locked()
            if self._state.phase != "idle":
                raise ValueError(f"Cannot start key region while phase={self._state.phase}")
            self._state.phase = "key_region"
            self._state.active_key_region_id = uuid.uuid4().hex
            self._state.score_deadline = None
            self._state.last_reward = None
            self._segment_ledger.record_started(self._state.active_key_region_id, phase=self._state.training_phase)
            self._add_event_locked("key_region_start", request.note or request.source)
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_start", {"source": request.source, "key_region_id": self._state.active_key_region_id})
            return self._state.model_copy(deep=True)

    def end_key_region(self, request: RLTControlRequest) -> RLTControlState:
        with self._lock:
            self._apply_score_timeout_locked()
            if self._state.phase != "key_region":
                raise ValueError(f"Cannot end key region while phase={self._state.phase}")
            self._state.phase = "await_score"
            self._state.score_deadline = time.time() + 10.0
            if self._state.active_key_region_id:
                self._segment_ledger.record_ended(self._state.active_key_region_id, phase=self._state.training_phase)
            self._add_event_locked("key_region_end", request.note or request.source)
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_end", {"source": request.source, "key_region_id": self._state.active_key_region_id})
            return self._state.model_copy(deep=True)

    def score_key_region(self, reward: int, *, source: str = "ui", timeout: bool = False) -> RLTControlState:
        with self._lock:
            if self._state.phase != "await_score":
                raise ValueError(f"Cannot score key region while phase={self._state.phase}")
            if reward not in (0, 1):
                raise ValueError("reward must be 0 or 1")
            key_region_id = self._state.active_key_region_id
            is_warmup = self._state.warmup_count < self._state.warmup_target
            if is_warmup:
                self._state.warmup_attempts += 1
            else:
                self._state.auto_rollout_attempts += 1
            self._state.phase = "pending_replay"
            self._state.score_deadline = None
            self._state.last_reward = reward
            event_name = "score_timeout_default_failure" if timeout else "score"
            self._add_event_locked(event_name, f"reward={reward}")
            if key_region_id:
                self._segment_ledger.record_accepted(key_region_id, reward=reward, phase=self._state.training_phase)
            self._refresh_derived_locked()
            self._persist_locked()
            return self._state.model_copy(deep=True)

    def confirm_key_region(self, *, source: str = "ui") -> RLTControlState:
        with self._lock:
            if self._state.phase != "pending_replay":
                raise ValueError(f"Cannot confirm key region while phase={self._state.phase}")
            key_region_id = self._state.active_key_region_id
            reward = self._state.last_reward
            if key_region_id is None or reward is None:
                raise ValueError("No scored key region is pending confirmation")
            self._state.phase = "idle"
            self._state.active_key_region_id = None
            self._add_event_locked("confirm", f"reward={reward}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked(
                "score",
                {"source": source, "reward": reward, "score_timeout": False, "key_region_id": key_region_id},
            )
            return self._state.model_copy(deep=True)

    def discard_key_region(self, *, source: str = "ui", reason: str = "operator_discard") -> RLTControlState:
        with self._lock:
            if self._state.phase not in {"key_region", "await_score", "pending_replay"}:
                raise ValueError(f"Cannot discard key region while phase={self._state.phase}")
            key_region_id = self._state.active_key_region_id
            if key_region_id:
                self._segment_ledger.record_discarded(key_region_id, phase=self._state.training_phase, reason=reason)
            self._state.phase = "idle"
            self._state.active_key_region_id = None
            self._state.score_deadline = None
            self._state.last_reward = None
            self._add_event_locked("discard", reason)
            self._apply_ledger_stats_locked()
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_discard", {"source": source, "reason": reason, "key_region_id": key_region_id})
            return self._state.model_copy(deep=True)

    def void_segment(self, key_region_id: str, *, source: str = "ui", reason: str = "operator_void") -> RLTControlState:
        with self._lock:
            self._segment_ledger.void_segment(key_region_id, reason=reason)
            self._apply_ledger_stats_locked()
            self._add_event_locked("void", f"{key_region_id}:{reason}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_void", {"source": source, "reason": reason, "key_region_id": key_region_id})
            return self._state.model_copy(deep=True)

    def list_segments(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            return self._segment_ledger.list_segments(limit=limit)

    def void_segments(
        self, key_region_ids: list[str], *, source: str = "ui", reason: str = "operator_batch_void"
    ) -> RLTControlState:
        with self._lock:
            changed = self._segment_ledger.void_segments(key_region_ids, reason=reason)
            self._apply_ledger_stats_locked()
            self._add_event_locked("batch_void", f"{len(changed)}:{reason}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_batch_void", {"source": source, "reason": reason, "key_region_ids": changed})
            return self._state.model_copy(deep=True)

    def restore_segments(
        self, key_region_ids: list[str], *, source: str = "ui", reason: str = "operator_restore"
    ) -> RLTControlState:
        with self._lock:
            changed = self._segment_ledger.restore_segments(key_region_ids, reason=reason)
            self._apply_ledger_stats_locked()
            self._add_event_locked("batch_restore", f"{len(changed)}:{reason}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_batch_restore", {"source": source, "reason": reason, "key_region_ids": changed})
            return self._state.model_copy(deep=True)

    def commit_key_region_from_files(
        self,
        *,
        key_region_id: str,
        phase: str,
        reward: int,
        shard_path: str,
        num_replay_transitions: int,
    ) -> RLTControlState:
        with self._lock:
            self._segment_ledger.record_committed(
                key_region_id,
                reward=reward,
                phase=phase,
                shard_path=shard_path,
                num_replay_transitions=num_replay_transitions,
            )
            self._apply_ledger_stats_locked()
            self._add_event_locked("reconcile_commit", f"{key_region_id}:{num_replay_transitions}")
            self._refresh_derived_locked()
            self._persist_locked()
            return self._state.model_copy(deep=True)

    def crop_key_region_from_files(
        self,
        *,
        key_region_id: str,
        phase: str,
        reward: int,
        shard_path: str,
        num_replay_transitions: int,
        source: str = "ui",
        reason: str = "operator_crop",
    ) -> RLTControlState:
        with self._lock:
            self._segment_ledger.record_cropped(
                key_region_id,
                reward=reward,
                phase=phase,
                shard_path=shard_path,
                num_replay_transitions=num_replay_transitions,
                reason=reason,
            )
            self._apply_ledger_stats_locked()
            self._add_event_locked("crop", f"{key_region_id}:{num_replay_transitions}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked(
                "key_region_crop",
                {
                    "source": source,
                    "reason": reason,
                    "key_region_id": key_region_id,
                    "shard_path": shard_path,
                    "num_replay_transitions": num_replay_transitions,
                },
            )
            return self._state.model_copy(deep=True)

    def delete_segments(
        self, key_region_ids: list[str], *, source: str = "ui", reason: str = "operator_delete"
    ) -> RLTControlState:
        with self._lock:
            changed = self._segment_ledger.delete_segments(key_region_ids)
            self._apply_ledger_stats_locked()
            self._add_event_locked("batch_delete", f"{len(changed)}:{reason}")
            self._refresh_derived_locked()
            self._persist_locked()
            self._publish_locked("key_region_batch_delete", {"source": source, "reason": reason, "key_region_ids": changed})
            return self._state.model_copy(deep=True)

    def update_config(self, request: RLTConfigRequest) -> RLTControlState:
        with self._lock:
            self._apply_score_timeout_locked()
            updates: dict[str, Any] = {}
            for key in (
                "warmup_target",
                "beta",
                "intervention_scale",
                "max_delta",
                "critic_gate_enabled",
                "critic_gate_margin",
                "critic_gate_temperature",
                "wandb_url",
            ):
                value = getattr(request, key, None)
                if value is not None:
                    setattr(self._state, key, value)
                    updates[key] = value
            if request.actor_enabled is not None:
                self._state.actor_enabled = request.actor_enabled
                updates["actor_enabled"] = request.actor_enabled
            if updates:
                self._add_event_locked("config_update", json.dumps(updates, sort_keys=True))
                self._apply_ledger_stats_locked()
                self._refresh_derived_locked()
                self._persist_locked()
                self._publish_locked("config_update", updates)
            return self._state.model_copy(deep=True)

    def update_runtime_metrics(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if payload.get("type") in {"rlt_replay_segment_written", "rlt_replay_segment_committed", "rlt_replay_segment_rejected"}:
                self._record_replay_ack_locked(payload)
            else:
                metric_keys = (
                    "critic_loss",
                    "critic_q1_loss",
                    "critic_q2_loss",
                    "actor_loss",
                    "actor_q_value",
                    "reference_q_value",
                    "q_advantage",
                    "actor_delta_norm",
                    "beta",
                    "auto_beta_enabled",
                    "auto_beta_target_delta_norm",
                    "auto_beta_delta_norm_ema",
                    "auto_beta_q_advantage_ema",
                    "auto_beta_critic_loss_ema",
                    "auto_beta_reason",
                    "q1_mean",
                    "q2_mean",
                    "target_q_mean",
                    "q_gap",
                    "actor_updated",
                    "publish_actor",
                    "trainer_step",
                    "steps_per_sec",
                    "success_episodes",
                    "failure_episodes",
                    "replay_action_horizon",
                    "train_action_horizon",
                    "replay_size",
                    "replay_shards",
                    "bad_shards",
                    "wandb_url",
                    "critic_gate_enabled",
                    "critic_gate_margin",
                    "critic_gate_temperature",
                    "critic_ready",
                    "inference_actor_active",
                    "inference_delta_norm",
                    "inference_gate_reason",
                    "key_region_probability",
                    "loaded_actor_step",
                    "inference_reference_q_value",
                    "inference_actor_q_value",
                    "inference_q_advantage",
                )
                for key in metric_keys:
                    if key in payload:
                        setattr(self._state, key, payload[key])
                if "timestamp" in payload:
                    self._state.rlt_metrics_timestamp = payload["timestamp"]
                latest_actor_path = payload.get("latest_actor_path")
                latest_actor_step = payload.get("latest_actor_step")
                if latest_actor_path is not None:
                    self._state.actor_checkpoint_path = latest_actor_path
                if latest_actor_step is not None:
                    self._state.actor_checkpoint_step = int(latest_actor_step)
                if latest_actor_path and latest_actor_step is not None and int(latest_actor_step) > 0:
                    self._state.actor_ready = True
            self._apply_ledger_stats_locked()
            self._refresh_derived_locked()
            self._persist_locked()

    def _record_replay_ack_locked(self, payload: dict[str, Any]) -> None:
        phase = str(payload.get("phase") or self._state.training_phase or "warmup")
        reward = int(payload.get("reward") or 0)
        replay_ready = bool(payload.get("replay_ready")) and int(payload.get("num_replay_transitions") or 0) > 0
        key_region_id = str(payload.get("key_region_id") or f"legacy-{time.time_ns()}")
        if replay_ready and payload.get("type") != "rlt_replay_segment_rejected":
            self._segment_ledger.record_committed(
                key_region_id,
                reward=reward,
                phase=phase,
                shard_path=payload.get("shard_path"),
                num_replay_transitions=int(payload.get("num_replay_transitions") or 0),
            )
            self._add_event_locked("rlt_replay_segment_committed", f"reward={reward} transitions={payload.get('num_replay_transitions')}")
        else:
            self._segment_ledger.record_rejected(
                key_region_id,
                phase=phase,
                reason=str(payload.get("replay_status") or "invalid"),
            )
            self._add_event_locked("rlt_replay_segment_rejected", str(payload.get("replay_status") or "invalid"))

    def _listen_runtime_state(self) -> None:
        pubsub = None
        try:
            pubsub = self._redis.pubsub()
            pubsub.subscribe(settings.rlt_state_channel)
            while self._running:
                message = pubsub.get_message(timeout=1.0)
                if not message or message["type"] != "message":
                    continue
                payload = json.loads(message["data"])
                self.update_runtime_metrics(payload)
        except Exception:
            logging.exception("RLT runtime state listener failed")
        finally:
            if pubsub is not None:
                with contextlib.suppress(Exception):
                    pubsub.close()

    def _apply_ledger_stats_locked(self) -> None:
        for key, value in self._segment_ledger.stats().items():
            setattr(self._state, key, value)
        self._apply_trainable_replay_stats_locked()

    def _apply_trainable_replay_stats_locked(self) -> None:
        success = 0
        failure = 0
        samples = 0
        trainable_shards = 0
        invalid_shards = 0
        for segment in self._segment_ledger.list_segments(limit=100000):
            if segment.get("status") != "committed" or not segment.get("shard_path"):
                continue
            path = Path(str(segment["shard_path"]))
            try:
                with np.load(path) as data:
                    done = np.asarray(data["done"]).astype(np.bool_)
                    reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
            except Exception:
                invalid_shards += 1
                continue
            if len(done) == 0 or not np.any(done):
                invalid_shards += 1
                continue
            terminal_rewards = np.sum(reward_seq[done], axis=-1)
            shard_success = int(np.sum(terminal_rewards > 0.0))
            shard_failure = int(len(terminal_rewards) - shard_success)
            success += shard_success
            failure += shard_failure
            trainable_shards += 1
            samples += int(len(done))
        self._state.trainable_replay_success = success
        self._state.trainable_replay_failure = failure
        self._state.trainable_replay_count = success + failure
        self._state.trainable_replay_samples = samples
        self._state.trainable_replay_shards = trainable_shards
        self._state.invalid_replay_shards = invalid_shards

    def _refresh_derived_locked(self) -> None:
        trainable_count = self._state.trainable_replay_count
        trainable_success = self._state.trainable_replay_success
        trainable_failure = self._state.trainable_replay_failure
        in_warmup = trainable_count < self._state.warmup_target
        replay_balance_ready = trainable_success > 0 and trainable_failure > 0
        self._state.training_phase = "warmup" if in_warmup else "rl"
        self._state.actor_effective = bool(
            self._state.actor_enabled and not in_warmup and replay_balance_ready and self._state.actor_ready
        )
        if in_warmup:
            self._state.actor_locked_reason = "warmup"
        elif not replay_balance_ready:
            self._state.actor_locked_reason = "replay_balance"
        elif not self._state.actor_ready:
            self._state.actor_locked_reason = "actor_not_ready"
        elif not self._state.actor_enabled:
            self._state.actor_locked_reason = "disabled"
        else:
            self._state.actor_locked_reason = None

    def _apply_score_timeout_locked(self) -> None:
        if (
            self._state.phase == "await_score"
            and self._state.score_deadline is not None
            and time.time() >= self._state.score_deadline
        ):
            self._score_timeout_locked()

    def _score_timeout_locked(self) -> None:
        key_region_id = self._state.active_key_region_id
        if key_region_id:
            self._segment_ledger.record_discarded(key_region_id, phase=self._state.training_phase, reason="score_timeout")
        self._state.phase = "idle"
        self._state.active_key_region_id = None
        self._state.score_deadline = None
        self._state.last_reward = None
        self._add_event_locked("score_timeout_discard", "score_timeout")
        self._apply_ledger_stats_locked()
        self._refresh_derived_locked()
        self._persist_locked()
        self._publish_locked(
            "key_region_discard",
            {"source": "timeout", "reason": "score_timeout", "key_region_id": key_region_id},
        )

    def _add_event_locked(self, event: str, detail: str = "") -> None:
        self._state.last_event = event
        self._state.events.append(RLTEvent(timestamp=time.time(), event=event, detail=detail))
        self._state.events = self._state.events[-50:]

    def _publish_locked(self, event_type: str, payload: dict[str, Any]) -> None:
        message = {
            "type": event_type,
            "timestamp": time.time(),
            "state": self._state.model_dump(exclude={"events"}),
            **payload,
        }
        try:
            self._redis.publish(settings.rlt_control_channel, json.dumps(message))
        except Exception:
            logging.exception("Failed to publish RLT control event")

    def _load(self) -> None:
        try:
            if self._state_path.exists():
                self._state = RLTControlState(**json.loads(self._state_path.read_text()))
                self._refresh_derived_locked()
        except Exception:
            logging.exception("Failed to load RLT control state from %s", self._state_path)

    def _persist_locked(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(self._state.model_dump_json(indent=2))
        except Exception:
            logging.exception("Failed to persist RLT control state to %s", self._state_path)

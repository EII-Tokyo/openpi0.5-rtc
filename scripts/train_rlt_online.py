import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import platform
import shutil
import time

from flax import nnx
from flax import serialization
import jax
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

from openpi.models import rlt
from openpi.training import rlt_eval
from openpi.training import rlt_replay_store
from openpi.training import rlt_training


@dataclasses.dataclass
class Args:
    replay_dir: pathlib.Path
    segment_db_path: pathlib.Path | None = None
    manifest_path: pathlib.Path | None = None
    holdout_manifest_path: pathlib.Path | None = None
    output_dir: pathlib.Path = pathlib.Path("./checkpoints/rlt_actor_critic/online")
    num_train_steps: int = 0
    batch_size: int = 64
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 1_000
    scan_interval: float = 1.0
    wait_sleep_seconds: float = 1.0
    min_replay_samples: int = 512
    min_replay_shards: int = 0
    min_success_episodes: int = 1
    min_failure_episodes: int = 1
    critic_burn_in_steps: int = 1_000
    online_safety_enabled: bool = True
    online_auto_train_critic: bool = False
    online_auto_train_actor: bool = False
    online_min_new_shards_per_round: int = 10
    online_min_new_success_per_round: int = 5
    online_min_new_failure_per_round: int = 5
    online_critic_updates_per_round: int = 500
    online_actor_updates_per_round: int = 300
    online_critic_auc_min: float = 0.70
    online_critic_max_auc_drop: float = 0.02
    online_require_positive_q_gap: bool = True
    online_actor_max_delta_norm: float = 0.09
    online_actor_min_q_advantage: float = 0.0
    online_beta_initial: float = 30.0
    online_beta_min: float = 5.0
    online_beta_max: float = 30.0
    online_beta_decay_on_actor_accept: float = 0.9
    online_beta_increase_on_reject: float = 1.25
    online_target_delta_initial: float = 0.04
    online_target_delta_max: float = 0.10
    online_target_delta_increment: float = 0.01
    auto_beta_enabled: bool = False
    auto_beta_target_delta_norm: float = 0.05
    auto_beta_min: float = 1.0
    auto_beta_max: float = 15.0
    auto_beta_lr: float = 0.03
    auto_beta_ema_decay: float = 0.95
    auto_beta_update_interval: int = 100
    auto_beta_q_margin: float = 0.005
    actor_min_replay_samples: int = 0
    actor_min_replay_shards: int = 0
    actor_min_success_episodes: int = 0
    actor_min_failure_episodes: int = 0
    max_replay_samples: int | None = None
    recursive_scan: bool = False
    policy_delay: int = 2
    actor_publish_interval: int = 500
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    target_actor_noise: bool = True
    actor_loss_mode: str = "td3"
    awbc_temperature: float = 0.2
    awbc_max_weight: float = 20.0
    awbc_min_advantage: float = 0.0
    awbc_max_action_delta_norm: float = 2.0
    train_action_horizon: int | None = 10
    expected_replay_action_horizon: int | None = 10
    wandb_enabled: bool = True
    wandb_project: str = "openpi"
    wandb_run_name: str = "rlt_actor_critic_online"
    wandb_api_key: str | None = dataclasses.field(
        default_factory=lambda: os.getenv("WANDB_API_KEY") or None,
        repr=False,
    )
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_state_channel: str = "aloha_rlt_state"
    redis_control_channel: str = "aloha_rlt_control"
    init_inference_actor_checkpoint: pathlib.Path | None = None
    online_treat_initial_replay_as_committed: bool = True
    online_initial_committed_shards: int | None = None
    overwrite: bool = False


@dataclasses.dataclass
class OnlineSafetyController:
    min_new_shards_per_round: int = 10
    min_new_success_per_round: int = 5
    min_new_failure_per_round: int = 5
    auto_train_critic: bool = False
    auto_train_actor: bool = False
    critic_updates_per_round: int = 500
    actor_updates_per_round: int = 300
    critic_auc_min: float = 0.70
    critic_max_auc_drop: float = 0.02
    require_positive_q_gap: bool = True
    actor_max_delta_norm: float = 0.09
    actor_min_q_advantage: float = 0.0
    beta_initial: float = 30.0
    beta_min: float = 5.0
    beta_max: float = 30.0
    beta_decay_on_actor_accept: float = 0.9
    beta_increase_on_reject: float = 1.25
    target_delta_initial: float = 0.04
    target_delta_max: float = 0.10
    target_delta_increment: float = 0.01
    phase: str = "idle_wait_new_data"
    last_committed_shards: int = 0
    last_committed_success: int = 0
    last_committed_failure: int = 0
    round_start_shards: int = 0
    round_start_success: int = 0
    round_start_failure: int = 0
    round_index: int = 0
    critic_steps_remaining: int = 0
    actor_steps_remaining: int = 0
    best_critic_auc: float | None = None
    best_critic_q_gap: float | None = None
    last_rejection_reason: str | None = None
    beta: float = dataclasses.field(init=False)
    target_delta_norm: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.beta = float(np.clip(self.beta_initial, self.beta_min, self.beta_max))
        self.target_delta_norm = float(self.target_delta_initial)

    def maybe_start_round(self, stats: rlt_replay_store.ReplayStats) -> bool:
        if self.phase != "idle_wait_new_data":
            return False
        new_shards = int(stats.num_shards) - int(self.last_committed_shards)
        if new_shards < self.min_new_shards_per_round:
            self.last_rejection_reason = "waiting_for_new_shards"
            return False
        new_success = int(stats.success_episodes) - int(self.last_committed_success)
        if new_success < self.min_new_success_per_round:
            self.last_rejection_reason = "waiting_for_new_success"
            return False
        new_failure = int(stats.failure_episodes) - int(self.last_committed_failure)
        if new_failure < self.min_new_failure_per_round:
            self.last_rejection_reason = "waiting_for_new_failure"
            return False
        if not self.auto_train_critic:
            self.last_rejection_reason = "critic_auto_train_disabled"
            return False
        self.round_index += 1
        self.round_start_shards = int(stats.num_shards)
        self.round_start_success = int(stats.success_episodes)
        self.round_start_failure = int(stats.failure_episodes)
        self.critic_steps_remaining = int(self.critic_updates_per_round)
        self.actor_steps_remaining = 0
        self.phase = "critic_candidate_training"
        self.last_rejection_reason = None
        return True

    def mark_bootstrap_committed(self, stats: rlt_replay_store.ReplayStats) -> None:
        self.last_committed_shards = int(stats.num_shards)
        self.last_committed_success = int(stats.success_episodes)
        self.last_committed_failure = int(stats.failure_episodes)
        self.round_start_shards = int(stats.num_shards)
        self.round_start_success = int(stats.success_episodes)
        self.round_start_failure = int(stats.failure_episodes)

    def step_allocation(self) -> dict[str, bool]:
        if self.phase == "critic_candidate_training":
            if self.critic_steps_remaining <= 0:
                self.phase = "critic_eval"
                return {"trainer_running": False, "actor_enabled": False}
            self.critic_steps_remaining -= 1
            if self.critic_steps_remaining == 0:
                self.phase = "critic_eval"
            return {"trainer_running": True, "actor_enabled": False}
        if self.phase == "actor_candidate_training":
            if self.actor_steps_remaining <= 0:
                self.phase = "actor_eval"
                return {"trainer_running": False, "actor_enabled": False}
            self.actor_steps_remaining -= 1
            if self.actor_steps_remaining == 0:
                self.phase = "actor_eval"
            return {"trainer_running": True, "actor_enabled": True}
        return {"trainer_running": False, "actor_enabled": False}

    def accept_critic(self, metric: dict | None) -> bool:
        accepted, reason = self._critic_decision(metric)
        if accepted:
            self.best_critic_auc = _finite_float(metric.get("auc")) if metric else None
            self.best_critic_q_gap = _finite_float(metric.get("q_gap")) if metric else None
            if not self.auto_train_actor:
                self.actor_steps_remaining = 0
                self.phase = "idle_wait_new_data"
                self.last_committed_shards = int(self.round_start_shards)
                self.last_committed_success = int(self.round_start_success)
                self.last_committed_failure = int(self.round_start_failure)
                self.last_rejection_reason = "actor_auto_train_disabled"
                return True
            self.actor_steps_remaining = int(self.actor_updates_per_round)
            self.phase = "actor_candidate_training"
            self.last_rejection_reason = None
            return True
        self.last_rejection_reason = reason
        self.phase = "idle_wait_new_data"
        self.last_committed_shards = int(self.round_start_shards)
        self.last_committed_success = int(self.round_start_success)
        self.last_committed_failure = int(self.round_start_failure)
        return False

    def accept_actor(self, metric: dict | None) -> bool:
        accepted, reason = self._actor_decision(metric)
        self.last_committed_shards = int(self.round_start_shards)
        self.last_committed_success = int(self.round_start_success)
        self.last_committed_failure = int(self.round_start_failure)
        self.phase = "idle_wait_new_data"
        if accepted:
            self.on_actor_accepted()
            self.last_rejection_reason = None
            return True
        self.on_actor_rejected()
        self.last_rejection_reason = reason
        return False

    def on_actor_accepted(self) -> None:
        self.beta = float(np.clip(self.beta * self.beta_decay_on_actor_accept, self.beta_min, self.beta_max))
        self.target_delta_norm = min(
            float(self.target_delta_max),
            float(self.target_delta_norm) + float(self.target_delta_increment),
        )

    def on_actor_rejected(self) -> None:
        self.beta = float(np.clip(self.beta * self.beta_increase_on_reject, self.beta_min, self.beta_max))
        self.target_delta_norm = max(
            float(self.target_delta_initial),
            float(self.target_delta_norm) - float(self.target_delta_increment),
        )

    def metrics(self) -> dict[str, float | int | str | bool | None]:
        return {
            "online_safety_phase": self.phase,
            "online_round_index": self.round_index,
            "online_last_committed_shards": self.last_committed_shards,
            "online_last_committed_success": self.last_committed_success,
            "online_last_committed_failure": self.last_committed_failure,
            "online_auto_train_critic": self.auto_train_critic,
            "online_auto_train_actor": self.auto_train_actor,
            "online_round_start_shards": self.round_start_shards,
            "online_round_start_success": self.round_start_success,
            "online_round_start_failure": self.round_start_failure,
            "online_min_new_success_per_round": self.min_new_success_per_round,
            "online_min_new_failure_per_round": self.min_new_failure_per_round,
            "online_critic_steps_remaining": self.critic_steps_remaining,
            "online_actor_steps_remaining": self.actor_steps_remaining,
            "online_best_critic_auc": self.best_critic_auc,
            "online_best_critic_q_gap": self.best_critic_q_gap,
            "online_rejection_reason": self.last_rejection_reason,
            "online_target_delta_norm": self.target_delta_norm,
        }

    def _critic_decision(self, metric: dict | None) -> tuple[bool, str | None]:
        if metric is None:
            return False, "missing_critic_metric"
        auc = _finite_float(metric.get("auc"))
        q_gap = _finite_float(metric.get("q_gap"))
        if auc is None:
            return False, "missing_critic_auc"
        if auc < self.critic_auc_min:
            return False, "critic_auc_below_min"
        if self.best_critic_auc is not None and auc < self.best_critic_auc - self.critic_max_auc_drop:
            return False, "critic_auc_regressed"
        if self.require_positive_q_gap and (q_gap is None or q_gap <= 0.0):
            return False, "critic_q_gap_not_positive"
        return True, None

    def _actor_decision(self, metric: dict | None) -> tuple[bool, str | None]:
        if metric is None:
            return False, "missing_actor_metric"
        q_advantage = _finite_float(metric.get("q_advantage"))
        actor_delta_norm = _finite_float(metric.get("actor_delta_norm"))
        if q_advantage is None or q_advantage <= self.actor_min_q_advantage:
            return False, "actor_q_advantage_too_low"
        if actor_delta_norm is not None and actor_delta_norm > self.actor_max_delta_norm:
            return False, "actor_delta_norm_too_high"
        failure_adv = _finite_float(metric.get("failure_actor_advantage_mean"))
        success_adv = _finite_float(metric.get("success_actor_advantage_mean"))
        if failure_adv is not None and success_adv is not None and failure_adv > success_adv:
            return False, "failure_actor_advantage_above_success"
        return True, None


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shape_metadata(shape: rlt_replay_store.ReplayShape | None) -> dict[str, int] | None:
    return None if shape is None else dataclasses.asdict(shape)


def _stats_metadata(stats: rlt_replay_store.ReplayStats | None) -> dict[str, int] | None:
    return None if stats is None else dataclasses.asdict(stats)


class RedisMetricsPublisher:
    def __init__(
        self,
        *,
        enabled: bool,
        channel: str,
        latest_key: str | None = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redis_client=None,
    ):
        self._enabled = enabled
        self._channel = channel
        self._latest_key = latest_key or f"{channel}:latest"
        self._client = redis_client
        self._warned = False
        if not self._enabled or self._client is not None:
            return
        try:
            import redis

            self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self._client.ping()
        except Exception as exc:  # pragma: no cover - depends on operator environment.
            self._enabled = False
            logging.warning("Disabling Redis RLT metrics publisher: %s", exc)

    def publish(self, payload: dict) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            encoded = json.dumps(payload, sort_keys=True)
            self._client.set(self._latest_key, encoded)
            self._client.publish(self._channel, encoded)
        except Exception as exc:
            if not self._warned:
                logging.warning("Failed to publish RLT trainer metrics to Redis: %s", exc)
                self._warned = True


class RedisControlSubscriber:
    def __init__(
        self,
        *,
        enabled: bool,
        channel: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redis_client=None,
    ):
        self._enabled = enabled
        self._channel = channel
        self._client = redis_client
        self._pubsub = None
        self._warned = False
        if not self._enabled:
            return
        try:
            if self._client is None:
                import redis

                self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                self._client.ping()
            self._pubsub = self._client.pubsub()
            self._pubsub.subscribe(channel)
        except Exception as exc:  # pragma: no cover - depends on operator environment.
            self._enabled = False
            self._pubsub = None
            logging.warning("Disabling Redis RLT control subscriber: %s", exc)

    def poll_update(self) -> dict[str, float | bool | int]:
        if not self._enabled or self._pubsub is None:
            return {}
        latest_update: dict[str, float | bool | int] = {}
        try:
            while True:
                message = self._pubsub.get_message(timeout=0.0)
                if message is None:
                    break
                if message.get("type") != "message":
                    continue
                try:
                    payload = json.loads(message.get("data", "{}"))
                except json.JSONDecodeError:
                    continue
                if payload.get("type") != "config_update":
                    continue
                if "beta" in payload:
                    try:
                        beta = float(payload["beta"])
                    except (TypeError, ValueError):
                        beta = -1.0
                    if beta >= 0.0:
                        latest_update["beta"] = beta
                if "trainer_enabled" in payload:
                    latest_update["trainer_enabled"] = bool(payload["trainer_enabled"])
                if "critic_burn_in_steps" in payload:
                    try:
                        critic_burn_in_steps = int(payload["critic_burn_in_steps"])
                    except (TypeError, ValueError):
                        critic_burn_in_steps = -1
                    if critic_burn_in_steps >= 0:
                        latest_update["critic_burn_in_steps"] = critic_burn_in_steps
                for key in (
                    "online_min_new_shards_per_round",
                    "online_min_new_success_per_round",
                    "online_min_new_failure_per_round",
                    "online_critic_updates_per_round",
                    "online_actor_updates_per_round",
                ):
                    if key not in payload:
                        continue
                    try:
                        value = int(payload[key])
                    except (TypeError, ValueError):
                        value = 0
                    if value >= 0:
                        latest_update[key] = value
                if "online_safety_enabled" in payload:
                    latest_update["online_safety_enabled"] = bool(payload["online_safety_enabled"])
                if "online_auto_train_critic" in payload:
                    latest_update["online_auto_train_critic"] = bool(payload["online_auto_train_critic"])
                if "online_auto_train_actor" in payload:
                    latest_update["online_auto_train_actor"] = bool(payload["online_auto_train_actor"])
                for key in (
                    "online_critic_auc_min",
                    "online_critic_max_auc_drop",
                    "online_actor_max_delta_norm",
                    "online_actor_min_q_advantage",
                    "online_beta_initial",
                    "online_beta_min",
                    "online_beta_max",
                    "online_beta_decay_on_actor_accept",
                    "online_beta_increase_on_reject",
                    "online_target_delta_initial",
                    "online_target_delta_max",
                    "online_target_delta_increment",
                ):
                    if key not in payload:
                        continue
                    value = _finite_float(payload[key])
                    if value is not None:
                        latest_update[key] = value
                if "online_require_positive_q_gap" in payload:
                    latest_update["online_require_positive_q_gap"] = bool(payload["online_require_positive_q_gap"])
                if "auto_beta_enabled" in payload:
                    latest_update["auto_beta_enabled"] = bool(payload["auto_beta_enabled"])
                for key in (
                    "auto_beta_target_delta_norm",
                    "auto_beta_min",
                    "auto_beta_max",
                    "auto_beta_lr",
                    "auto_beta_ema_decay",
                    "auto_beta_q_margin",
                ):
                    if key not in payload:
                        continue
                    value = _finite_float(payload[key])
                    if value is not None:
                        latest_update[key] = value
                if "auto_beta_update_interval" in payload:
                    try:
                        update_interval = int(payload["auto_beta_update_interval"])
                    except (TypeError, ValueError):
                        update_interval = 0
                    if update_interval >= 1:
                        latest_update["auto_beta_update_interval"] = update_interval
        except Exception as exc:
            if not self._warned:
                logging.warning("Failed to read RLT control update from Redis: %s", exc)
                self._warned = True
        return latest_update

    def poll_beta_update(self) -> float | None:
        update = self.poll_update()
        beta = update.get("beta")
        return float(beta) if beta is not None else None

    def close(self) -> None:
        if self._pubsub is None:
            return
        try:
            self._pubsub.close()
        except Exception:
            pass



@dataclasses.dataclass(frozen=True)
class AutoBetaUpdate:
    beta: float
    changed: bool
    reason: str
    metrics: dict[str, float | bool | str | None]


class AutoBetaController:
    def __init__(
        self,
        *,
        beta: float,
        target_delta_norm: float,
        beta_min: float,
        beta_max: float,
        lr: float,
        ema_decay: float,
        q_margin: float,
        update_interval: int,
    ):
        if target_delta_norm <= 0:
            raise ValueError("target_delta_norm must be positive")
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("beta range must satisfy 0 < beta_min <= beta_max")
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.beta = float(np.clip(beta, beta_min, beta_max))
        self.target_delta_norm = float(target_delta_norm)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.lr = float(lr)
        self.ema_decay = float(ema_decay)
        self.q_margin = float(q_margin)
        self.update_interval = int(update_interval)
        self.delta_norm_ema: float | None = None
        self.q_advantage_ema: float | None = None
        self.critic_loss_ema: float | None = None
        self.reason = "initializing"

    def update_config(
        self,
        *,
        target_delta_norm: float | None = None,
        beta_min: float | None = None,
        beta_max: float | None = None,
        lr: float | None = None,
        ema_decay: float | None = None,
        q_margin: float | None = None,
        update_interval: int | None = None,
    ) -> None:
        target_delta_norm = self.target_delta_norm if target_delta_norm is None else float(target_delta_norm)
        beta_min = self.beta_min if beta_min is None else float(beta_min)
        beta_max = self.beta_max if beta_max is None else float(beta_max)
        lr = self.lr if lr is None else float(lr)
        ema_decay = self.ema_decay if ema_decay is None else float(ema_decay)
        q_margin = self.q_margin if q_margin is None else float(q_margin)
        update_interval = self.update_interval if update_interval is None else int(update_interval)
        if target_delta_norm <= 0:
            raise ValueError("target_delta_norm must be positive")
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("beta range must satisfy 0 < beta_min <= beta_max")
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.target_delta_norm = target_delta_norm
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.lr = lr
        self.ema_decay = ema_decay
        self.q_margin = q_margin
        self.update_interval = update_interval
        self.beta = float(np.clip(self.beta, self.beta_min, self.beta_max))
        self.reason = "config_updated"

    def update(self, *, step: int, metrics: dict) -> AutoBetaUpdate:
        critic_loss = _finite_float(metrics.get("critic_loss"))
        if critic_loss is not None:
            self.critic_loss_ema = self._ema(self.critic_loss_ema, critic_loss)

        actor_updated = bool(metrics.get("actor_updated", True))
        delta_norm = _finite_float(metrics.get("actor_delta_norm"))
        q_advantage = _finite_float(metrics.get("q_advantage"))
        if actor_updated:
            if delta_norm is not None:
                self.delta_norm_ema = self._ema(self.delta_norm_ema, delta_norm)
            if q_advantage is not None:
                self.q_advantage_ema = self._ema(self.q_advantage_ema, q_advantage)
        else:
            self.reason = "waiting_for_actor_update"
            return self._result(changed=False)

        if int(step) % self.update_interval != 0:
            self.reason = "waiting_for_update_interval"
            return self._result(changed=False)
        if self.delta_norm_ema is None or self.q_advantage_ema is None:
            self.reason = "waiting_for_actor_metrics"
            return self._result(changed=False)

        previous = self.beta
        if self.q_advantage_ema < self.q_margin:
            self.beta *= math.exp(self.lr)
            self.reason = "q_advantage_below_margin"
        else:
            ratio = self.delta_norm_ema / self.target_delta_norm
            if ratio > 1.0:
                self.beta *= math.exp(self.lr * (ratio - 1.0))
                self.reason = "delta_above_target"
            elif ratio < 1.0:
                self.beta *= math.exp(self.lr * (ratio - 1.0))
                self.reason = "delta_below_target_q_positive"
            else:
                self.reason = "stable"
        self.beta = float(np.clip(self.beta, self.beta_min, self.beta_max))
        return self._result(changed=not math.isclose(previous, self.beta, rel_tol=0.0, abs_tol=1e-12))

    def _ema(self, old: float | None, new: float) -> float:
        if old is None:
            return float(new)
        return self.ema_decay * old + (1.0 - self.ema_decay) * float(new)

    def _result(self, *, changed: bool) -> AutoBetaUpdate:
        return AutoBetaUpdate(beta=self.beta, changed=changed, reason=self.reason, metrics=self.metrics())

    def metrics(self) -> dict[str, float | bool | str | None]:
        return {
        "auto_beta_enabled": True,
        "auto_beta_target_delta_norm": self.target_delta_norm,
        "auto_beta_min": self.beta_min,
        "auto_beta_max": self.beta_max,
        "auto_beta_lr": self.lr,
        "auto_beta_ema_decay": self.ema_decay,
        "auto_beta_update_interval": self.update_interval,
        "auto_beta_q_margin": self.q_margin,
        "auto_beta_delta_norm_ema": self.delta_norm_ema,
        "auto_beta_q_advantage_ema": self.q_advantage_ema,
        "auto_beta_critic_loss_ema": self.critic_loss_ema,
            "auto_beta_reason": self.reason,
        }


def _finite_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _disabled_auto_beta_metrics(config: dict[str, float | int]) -> dict[str, float | bool | str | None]:
    return {
        "auto_beta_enabled": False,
        "auto_beta_target_delta_norm": float(config["target_delta_norm"]),
        "auto_beta_min": float(config["beta_min"]),
        "auto_beta_max": float(config["beta_max"]),
        "auto_beta_lr": float(config["lr"]),
        "auto_beta_ema_decay": float(config["ema_decay"]),
        "auto_beta_update_interval": int(config["update_interval"]),
        "auto_beta_q_margin": float(config["q_margin"]),
        "auto_beta_delta_norm_ema": None,
        "auto_beta_q_advantage_ema": None,
        "auto_beta_critic_loss_ema": None,
        "auto_beta_reason": "manual_beta",
    }


def _build_auto_beta_controller(beta: float, config: dict[str, float | int]) -> AutoBetaController:
    return AutoBetaController(
        beta=beta,
        target_delta_norm=float(config["target_delta_norm"]),
        beta_min=float(config["beta_min"]),
        beta_max=float(config["beta_max"]),
        lr=float(config["lr"]),
        ema_decay=float(config["ema_decay"]),
        q_margin=float(config["q_margin"]),
        update_interval=int(config["update_interval"]),
    )

def _with_runtime_beta(state: rlt_training.RLTTrainState, beta: float) -> rlt_training.RLTTrainState:
    model = nnx.merge(state.model_def, state.params)
    beta = float(beta)
    if float(model.config.beta) == beta:
        return state
    config = dataclasses.replace(model.config, beta=beta)
    model.config = config
    model.actor.config = config
    model.critic.q1.config = config
    model.critic.q2.config = config
    model.target_actor.config = config
    model.target_critic.q1.config = config
    model.target_critic.q2.config = config
    return dataclasses.replace(state, model_def=nnx.graphdef(model), params=nnx.state(model))


def _load_inference_actor_checkpoint(
    state: rlt_training.RLTTrainState,
    checkpoint_dir: pathlib.Path,
) -> tuple[rlt_training.RLTTrainState, dict]:
    checkpoint_dir = _resolve_inference_checkpoint_dir(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    actor_path = checkpoint_dir / "actor.msgpack"
    critic_path = checkpoint_dir / "critic.msgpack"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")
    if not actor_path.exists():
        raise FileNotFoundError(f"actor.msgpack not found in {checkpoint_dir}")
    if not critic_path.exists():
        raise FileNotFoundError(f"critic.msgpack not found in {checkpoint_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("type") != "rlt_inference_actor":
        raise ValueError(f"{checkpoint_dir} is not an RLT inference actor checkpoint")
    loaded_config = rlt.RLTConfig(**metadata["rlt_config"])
    model = nnx.merge(state.model_def, state.params)
    _assert_compatible_rlt_config(model.config, loaded_config, checkpoint_dir)

    actor_state = nnx.state(model.actor)
    actor_pure = serialization.from_bytes(actor_state.to_pure_dict(), actor_path.read_bytes())
    actor_state.replace_by_pure_dict(actor_pure)
    nnx.update(model.actor, actor_state)
    target_actor_state = nnx.state(model.target_actor)
    target_actor_state.replace_by_pure_dict(actor_pure)
    nnx.update(model.target_actor, target_actor_state)

    critic_state = nnx.state(model.critic)
    critic_pure = serialization.from_bytes(critic_state.to_pure_dict(), critic_path.read_bytes())
    critic_state.replace_by_pure_dict(critic_pure)
    nnx.update(model.critic, critic_state)
    target_critic_state = nnx.state(model.target_critic)
    target_critic_state.replace_by_pure_dict(critic_pure)
    nnx.update(model.target_critic, target_critic_state)
    return dataclasses.replace(
        state,
        step=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        params=nnx.state(model),
        model_def=nnx.graphdef(model),
    ), metadata


def _resolve_inference_checkpoint_dir(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    if path.name == "LATEST":
        return pathlib.Path(path.read_text().strip()).expanduser()
    if path.is_dir() and (path / "LATEST").exists():
        return pathlib.Path((path / "LATEST").read_text().strip()).expanduser()
    return path


def _assert_compatible_rlt_config(current: rlt.RLTConfig, loaded: rlt.RLTConfig, checkpoint_dir: pathlib.Path) -> None:
    fields = ("z_dim", "proprio_dim", "action_horizon", "action_dim", "hidden_dim", "num_layers")
    mismatches = {
        field: (getattr(current, field), getattr(loaded, field))
        for field in fields
        if getattr(current, field) != getattr(loaded, field)
    }
    if mismatches:
        raise ValueError(f"Incompatible RLT checkpoint {checkpoint_dir}: {mismatches}")


def _json_float(value):
    if value is None:
        return None
    return float(value)


def _json_bool(value):
    if value is None:
        return None
    return bool(value)


def _format_log_metric(key: str, value) -> str:
    if isinstance(value, bool):
        return f"{key}={int(value)}"
    if isinstance(value, int | float | np.number):
        return f"{key}={float(value):.4f}"
    return f"{key}={value}"


def _build_metrics_payload(
    *,
    step: int,
    reduced: dict,
    stats: rlt_replay_store.ReplayStats,
    replay_shape: rlt_replay_store.ReplayShape | None,
    train_shape: rlt_replay_store.ReplayShape | None,
    actor_enabled: bool,
    trainer_enabled: bool,
    trainer_running: bool,
    critic_burn_in_steps: int,
    target_sync_step: int | None = None,
    latest_actor_path: str | None = None,
    latest_actor_step: int | None = None,
    wandb_url: str | None = None,
) -> dict:
    q1_mean = _json_float(reduced.get("q1_mean"))
    q2_mean = _json_float(reduced.get("q2_mean"))
    return {
        "type": "rlt_trainer_metrics",
        "timestamp": time.time(),
        "trainer_step": int(step),
        "critic_loss": _json_float(reduced.get("critic_loss")),
        "critic_q1_loss": _json_float(reduced.get("critic_q1_loss")),
        "critic_q2_loss": _json_float(reduced.get("critic_q2_loss")),
        "actor_loss": _json_float(reduced.get("actor_loss")),
        "actor_q_value": _json_float(reduced.get("actor_q_value")),
        "reference_q_value": _json_float(reduced.get("reference_q_value")),
        "q_advantage": _json_float(reduced.get("q_advantage")),
        "actor_delta_norm": _json_float(reduced.get("actor_delta_norm")),
        "q1_mean": q1_mean,
        "q2_mean": q2_mean,
        "target_q_mean": _json_float(reduced.get("target_q_mean")),
        "q_gap": None if q1_mean is None or q2_mean is None else abs(q1_mean - q2_mean),
        "actor_updated": _json_bool(reduced.get("actor_updated")),
        "publish_actor": _json_bool(reduced.get("publish_actor")),
        "beta": _json_float(reduced.get("beta")),
        "auto_beta_enabled": _json_bool(reduced.get("auto_beta_enabled")),
        "auto_beta_target_delta_norm": _json_float(reduced.get("auto_beta_target_delta_norm")),
        "auto_beta_min": _json_float(reduced.get("auto_beta_min")),
        "auto_beta_max": _json_float(reduced.get("auto_beta_max")),
        "auto_beta_lr": _json_float(reduced.get("auto_beta_lr")),
        "auto_beta_ema_decay": _json_float(reduced.get("auto_beta_ema_decay")),
        "auto_beta_update_interval": None
        if reduced.get("auto_beta_update_interval") is None
        else int(reduced.get("auto_beta_update_interval")),
        "auto_beta_q_margin": _json_float(reduced.get("auto_beta_q_margin")),
        "auto_beta_delta_norm_ema": _json_float(reduced.get("auto_beta_delta_norm_ema")),
        "auto_beta_q_advantage_ema": _json_float(reduced.get("auto_beta_q_advantage_ema")),
        "auto_beta_critic_loss_ema": _json_float(reduced.get("auto_beta_critic_loss_ema")),
        "auto_beta_reason": reduced.get("auto_beta_reason"),
        "replay_size": int(stats.replay_size),
        "wandb_url": wandb_url,
        "actor_enabled": bool(actor_enabled),
        "trainer_enabled": bool(trainer_enabled),
        "trainer_running": bool(trainer_running),
        "critic_burn_in_steps": int(critic_burn_in_steps),
        "target_sync_step": None if target_sync_step is None else int(target_sync_step),
        "latest_actor_path": latest_actor_path,
        "latest_actor_step": None if latest_actor_step is None else int(latest_actor_step),
        "replay_shards": int(stats.num_shards),
        "success_episodes": int(stats.success_episodes),
        "failure_episodes": int(stats.failure_episodes),
        "bad_shards": int(stats.bad_shards),
        "replay_action_horizon": 0 if replay_shape is None else int(replay_shape.action_horizon),
        "train_action_horizon": 0 if train_shape is None else int(train_shape.action_horizon),
        "steps_per_sec": _json_float(reduced.get("steps_per_sec")),
        "online_safety_enabled": _json_bool(reduced.get("online_safety_enabled")),
        "online_safety_phase": reduced.get("online_safety_phase"),
        "online_round_index": None if reduced.get("online_round_index") is None else int(reduced.get("online_round_index")),
        "online_last_committed_shards": None
        if reduced.get("online_last_committed_shards") is None
        else int(reduced.get("online_last_committed_shards")),
        "online_auto_train_critic": _json_bool(reduced.get("online_auto_train_critic")),
        "online_auto_train_actor": _json_bool(reduced.get("online_auto_train_actor")),
        "online_round_start_shards": None
        if reduced.get("online_round_start_shards") is None
        else int(reduced.get("online_round_start_shards")),
        "online_critic_steps_remaining": None
        if reduced.get("online_critic_steps_remaining") is None
        else int(reduced.get("online_critic_steps_remaining")),
        "online_actor_steps_remaining": None
        if reduced.get("online_actor_steps_remaining") is None
        else int(reduced.get("online_actor_steps_remaining")),
        "online_best_critic_auc": _json_float(reduced.get("online_best_critic_auc")),
        "online_best_critic_q_gap": _json_float(reduced.get("online_best_critic_q_gap")),
        "online_rejection_reason": reduced.get("online_rejection_reason"),
        "online_target_delta_norm": _json_float(reduced.get("online_target_delta_norm")),
    }


def _reduce_numeric_infos(infos: list[dict[str, object]]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    if not infos:
        return reduced
    for key in infos[0]:
        values = [item.get(key) for item in infos]
        if any(value is None for value in values):
            continue
        first = np.asarray(values[0])
        if first.ndim != 0 or first.dtype.kind not in "biuf":
            continue
        reduced[key] = float(np.mean([np.asarray(value) for value in values]))
    return reduced


def _wandb_url() -> str | None:
    run = getattr(wandb, "run", None)
    if run is None:
        return None
    try:
        return run.get_url()
    except Exception:
        return None


def _save_actor_for_inference(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    *,
    action_horizon: int,
    replay_shape: rlt_replay_store.ReplayShape | None = None,
    train_shape: rlt_replay_store.ReplayShape | None = None,
    replay_stats: rlt_replay_store.ReplayStats | None = None,
) -> pathlib.Path:
    actor_dir = output_dir / "inference_actor" / f"{step:08d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_params = rlt_training.actor_params_for_inference(state).to_pure_dict()
    critic_params = rlt_training.critic_params_for_inference(state).to_pure_dict()
    actor_bytes = serialization.to_bytes(actor_params)
    critic_bytes = serialization.to_bytes(critic_params)
    _atomic_write_bytes(actor_dir / "actor.msgpack", actor_bytes)
    _atomic_write_bytes(actor_dir / "critic.msgpack", critic_bytes)
    model = nnx.merge(state.model_def, state.params)
    actor_loss_config = {
        "actor_loss_mode": rlt_training.actor_loss_mode_name(int(state.actor_loss_mode)),
        "awbc_temperature": float(state.awbc_temperature),
        "awbc_max_weight": float(state.awbc_max_weight),
        "awbc_min_advantage": float(state.awbc_min_advantage),
        "awbc_max_action_delta_norm": float(state.awbc_max_action_delta_norm),
    }
    _atomic_write_text(
        actor_dir / "metadata.json",
        json.dumps(
            {
                "format_version": 1,
                "created_at_unix": time.time(),
                "created_at_iso": datetime.datetime.now(datetime.UTC).isoformat(),
                "source_script": "scripts/train_rlt_online.py",
                "host": platform.node(),
                "step": int(step),
                "type": "rlt_inference_actor",
                "note": "Stable actor export. Runtime should switch only at chunk/idle boundary.",
                "actor_file": "actor.msgpack",
                "actor_sha256": _sha256_bytes(actor_bytes),
                "critic_file": "critic.msgpack",
                "critic_sha256": _sha256_bytes(critic_bytes),
                "action_horizon": int(action_horizon),
                "rlt_config": dataclasses.asdict(model.config),
                "actor_loss_config": actor_loss_config,
                "replay_shape": _shape_metadata(replay_shape),
                "train_shape": _shape_metadata(train_shape),
                "replay_stats": _stats_metadata(replay_stats),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    latest_path = output_dir / "inference_actor" / "LATEST"
    _atomic_write_text(latest_path, str(actor_dir))
    return actor_dir


def _save_training_checkpoint(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    store: rlt_replay_store.RLTReplayStore,
) -> pathlib.Path:
    checkpoint_dir = output_dir / "checkpoints" / f"{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(state.step),
        "params": _to_msgpackable_state(state.params),
        "actor_opt_state": _to_msgpackable_state(state.actor_opt_state),
        "critic_opt_state": _to_msgpackable_state(state.critic_opt_state),
    }
    _atomic_write_bytes(checkpoint_dir / "train_state.msgpack", serialization.to_bytes(payload))
    _atomic_write_text(
        checkpoint_dir / "metadata.json",
        json.dumps(
            {
                "step": int(state.step),
                "replay_stats": dataclasses.asdict(store.stats),
                "replay_shape": None if store.shape is None else dataclasses.asdict(store.shape),
                "train_shape": None if store.sample_shape is None else dataclasses.asdict(store.sample_shape),
                "loaded_shards": [str(path) for path in store.loaded_paths],
            },
            indent=2,
        ),
    )
    _atomic_write_text(output_dir / "checkpoints" / "LATEST", str(checkpoint_dir))
    return checkpoint_dir


def _to_msgpackable_state(value):
    state_dict = serialization.to_state_dict(value)
    return _convert_nnx_state_objects(state_dict)


def _convert_nnx_state_objects(value):
    if isinstance(value, nnx.State):
        return _convert_nnx_state_objects(value.to_pure_dict())
    if isinstance(value, dict):
        return {str(key): _convert_nnx_state_objects(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_convert_nnx_state_objects(item) for item in value]
    return value


def _init_wandb(args: Args, store: rlt_replay_store.RLTReplayStore) -> None:
    if args.wandb_enabled:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        config = {
            key: str(value) if isinstance(value, pathlib.Path) else value for key, value in dataclasses.asdict(args).items()
        }
        config["wandb_api_key"] = "<set>" if args.wandb_api_key else "<unset>"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **config,
                "initial_replay_stats": dataclasses.asdict(store.stats),
            },
        )
    else:
        wandb.init(mode="disabled")


def _wait_for_replay(args: Args, store: rlt_replay_store.RLTReplayStore) -> None:
    with tqdm.tqdm(total=None, dynamic_ncols=True, desc="waiting for replay") as progress:
        while not store.ready(
            min_replay_samples=args.min_replay_samples,
            min_success_episodes=args.min_success_episodes,
            min_failure_episodes=args.min_failure_episodes,
            min_replay_shards=args.min_replay_shards,
        ):
            store.scan()
            stats = store.stats
            progress.set_postfix(
                replay=stats.replay_size,
                success=stats.success_episodes,
                failure=stats.failure_episodes,
                shards=stats.num_shards,
            )
            time.sleep(args.wait_sleep_seconds)


def _actor_updates_enabled(args: Args, store: rlt_replay_store.RLTReplayStore, step: int) -> bool:
    if step < args.critic_burn_in_steps:
        return False
    stats = store.stats
    actor_min_replay = args.actor_min_replay_samples or args.min_replay_samples
    actor_min_shards = args.actor_min_replay_shards or args.min_replay_shards
    actor_min_success = args.actor_min_success_episodes or args.min_success_episodes
    actor_min_failure = args.actor_min_failure_episodes or args.min_failure_episodes
    return (
        stats.replay_size >= actor_min_replay
        and stats.num_shards >= actor_min_shards
        and stats.success_episodes >= actor_min_success
        and stats.failure_episodes >= actor_min_failure
    )


def _state_for_actor_gate(
    state: rlt_training.RLTTrainState,
    *,
    actor_enabled: bool,
    policy_delay: int,
    actor_publish_interval: int,
) -> rlt_training.RLTTrainState:
    if actor_enabled:
        return dataclasses.replace(
            state,
            policy_delay=policy_delay,
            actor_publish_interval=actor_publish_interval,
        )
    return dataclasses.replace(
        state,
        policy_delay=1_000_000_000,
        actor_publish_interval=0,
    )


def _build_online_controller(args: Args) -> OnlineSafetyController:
    return OnlineSafetyController(
        min_new_shards_per_round=args.online_min_new_shards_per_round,
        min_new_success_per_round=args.online_min_new_success_per_round,
        min_new_failure_per_round=args.online_min_new_failure_per_round,
        auto_train_critic=args.online_auto_train_critic,
        auto_train_actor=args.online_auto_train_actor,
        critic_updates_per_round=args.online_critic_updates_per_round,
        actor_updates_per_round=args.online_actor_updates_per_round,
        critic_auc_min=args.online_critic_auc_min,
        critic_max_auc_drop=args.online_critic_max_auc_drop,
        require_positive_q_gap=args.online_require_positive_q_gap,
        actor_max_delta_norm=args.online_actor_max_delta_norm,
        actor_min_q_advantage=args.online_actor_min_q_advantage,
        beta_initial=args.online_beta_initial,
        beta_min=args.online_beta_min,
        beta_max=args.online_beta_max,
        beta_decay_on_actor_accept=args.online_beta_decay_on_actor_accept,
        beta_increase_on_reject=args.online_beta_increase_on_reject,
        target_delta_initial=args.online_target_delta_initial,
        target_delta_max=args.online_target_delta_max,
        target_delta_increment=args.online_target_delta_increment,
    )


def _update_online_controller_config(controller: OnlineSafetyController, control_update: dict) -> None:
    mapping = {
        "online_min_new_shards_per_round": ("min_new_shards_per_round", int),
        "online_min_new_success_per_round": ("min_new_success_per_round", int),
        "online_min_new_failure_per_round": ("min_new_failure_per_round", int),
        "online_auto_train_critic": ("auto_train_critic", bool),
        "online_auto_train_actor": ("auto_train_actor", bool),
        "online_critic_updates_per_round": ("critic_updates_per_round", int),
        "online_actor_updates_per_round": ("actor_updates_per_round", int),
        "online_critic_auc_min": ("critic_auc_min", float),
        "online_critic_max_auc_drop": ("critic_max_auc_drop", float),
        "online_actor_max_delta_norm": ("actor_max_delta_norm", float),
        "online_actor_min_q_advantage": ("actor_min_q_advantage", float),
        "online_beta_min": ("beta_min", float),
        "online_beta_max": ("beta_max", float),
        "online_beta_decay_on_actor_accept": ("beta_decay_on_actor_accept", float),
        "online_beta_increase_on_reject": ("beta_increase_on_reject", float),
        "online_target_delta_max": ("target_delta_max", float),
        "online_target_delta_increment": ("target_delta_increment", float),
    }
    for source_key, (target_key, caster) in mapping.items():
        if source_key in control_update:
            setattr(controller, target_key, caster(control_update[source_key]))
    if "online_require_positive_q_gap" in control_update:
        controller.require_positive_q_gap = bool(control_update["online_require_positive_q_gap"])


def _evaluate_candidate_critic(
    *,
    args: Args,
    checkpoint_dir: pathlib.Path,
    store: rlt_replay_store.RLTReplayStore,
    output_dir: pathlib.Path,
    round_index: int,
) -> dict | None:
    paths = _candidate_holdout_paths(args=args, store=store, round_index=round_index)
    if not paths:
        return None
    try:
        result = rlt_eval.evaluate_holdout_checkpoints(
            checkpoint_dirs=[checkpoint_dir],
            holdout_paths=paths,
            output_dir=output_dir,
            score_batch_size=512,
        )
    except Exception as exc:
        logging.warning("Online candidate critic eval failed: %s", exc)
        return None
    return result.best_metric


def _build_replay_store(args: Args) -> rlt_replay_store.RLTReplayStore:
    return rlt_replay_store.RLTReplayStore(
        args.replay_dir,
        max_replay_samples=args.max_replay_samples,
        recursive=args.recursive_scan,
        sample_action_horizon=args.train_action_horizon,
        segment_db_path=args.segment_db_path,
        manifest_path=args.manifest_path,
    )


def _candidate_holdout_paths(
    *,
    args: Args,
    store: rlt_replay_store.RLTReplayStore,
    round_index: int,
) -> list[pathlib.Path]:
    if args.holdout_manifest_path is not None:
        paths = rlt_eval.find_replay_shards(args.replay_dir, manifest_path=args.holdout_manifest_path)
        return list(paths)
    paths = list(store.loaded_paths)
    if len(paths) < 2:
        return []
    split = rlt_eval.split_shards(paths, holdout_ratio=0.2, seed=args.seed + round_index)
    return list(split.holdout_paths)


def _validate_online_safety_inputs(args: Args) -> None:
    if args.online_safety_enabled and args.holdout_manifest_path is None:
        raise ValueError(
            "online_safety_enabled requires holdout_manifest_path. "
            "Do not run online safety training with a holdout split derived from the training store."
        )


def _validate_train_holdout_disjoint(args: Args, *, train_paths: list[pathlib.Path] | tuple[pathlib.Path, ...] | None) -> None:
    if args.holdout_manifest_path is None:
        return
    if args.manifest_path is not None:
        train_paths = rlt_eval.find_replay_shards(args.replay_dir, manifest_path=args.manifest_path)
    if train_paths is None:
        return
    train_paths = set(pathlib.Path(path).expanduser().resolve() for path in train_paths)
    holdout_paths = set(rlt_eval.find_replay_shards(args.replay_dir, manifest_path=args.holdout_manifest_path))
    overlap = sorted(train_paths & holdout_paths)
    if overlap:
        examples = ", ".join(str(path) for path in overlap[:3])
        raise ValueError(
            f"Online train and holdout manifests overlap on {len(overlap)} shard(s): {examples}. "
            "Holdout shards must be eval-only and must never participate in training."
        )


def _prepare_output_dir(args: Args) -> None:
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(args: Args) -> None:
    _init_logging()
    logging.info("Running online RLT trainer on %s", platform.node())

    _validate_online_safety_inputs(args)
    store = _build_replay_store(args)
    store.scan()
    _wait_for_replay(args, store)
    if store.shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    replay_shape = store.shape
    if replay_shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    if args.expected_replay_action_horizon is not None and replay_shape.action_horizon != args.expected_replay_action_horizon:
        raise ValueError(
            f"Expected replay action horizon {args.expected_replay_action_horizon}, got {replay_shape.action_horizon}"
        )
    shape = store.sample_shape
    if shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    _validate_train_holdout_disjoint(args, train_paths=list(store.loaded_paths))
    logging.info("Replay ready: %s, replay_shape=%s, train_shape=%s", store.stats, replay_shape, shape)

    _prepare_output_dir(args)

    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=shape.z_dim,
            proprio_dim=shape.proprio_dim,
            action_horizon=shape.action_horizon,
            action_dim=shape.action_dim,
        ),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_delay=args.policy_delay,
        actor_publish_interval=args.actor_publish_interval,
        target_actor_noise=args.target_actor_noise,
        actor_loss_mode=args.actor_loss_mode,
        awbc_temperature=args.awbc_temperature,
        awbc_max_weight=args.awbc_max_weight,
        awbc_min_advantage=args.awbc_min_advantage,
        awbc_max_action_delta_norm=args.awbc_max_action_delta_norm,
    )
    state = rlt_training.init_train_state(config, jax.random.key(args.seed))
    init_checkpoint_metadata = None
    if args.init_inference_actor_checkpoint is not None:
        state, init_checkpoint_metadata = _load_inference_actor_checkpoint(state, args.init_inference_actor_checkpoint)
        logging.info(
            "Initialized online RLT from inference checkpoint step=%s path=%s",
            init_checkpoint_metadata.get("step"),
            args.init_inference_actor_checkpoint,
        )
    online_safety_enabled = bool(args.online_safety_enabled)
    online_controller = _build_online_controller(args)
    accepted_state = state
    if online_safety_enabled:
        if args.online_initial_committed_shards is not None:
            online_controller.last_committed_shards = int(args.online_initial_committed_shards)
        elif args.online_treat_initial_replay_as_committed:
            online_controller.mark_bootstrap_committed(store.stats)
        state = _with_runtime_beta(state, online_controller.beta)
        accepted_state = state
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, beta=online_controller.beta))
    replay_rng = np.random.default_rng(args.seed)
    latest_target_sync_step: int | None = None
    _init_wandb(args, store)
    metrics_publisher = RedisMetricsPublisher(
        enabled=args.redis_enabled,
        channel=args.redis_state_channel,
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
    )
    control_subscriber = RedisControlSubscriber(
        enabled=args.redis_enabled,
        channel=args.redis_control_channel,
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
    )
    runtime_beta = float(config.model.beta)
    trainer_enabled = False
    auto_beta_enabled = bool(args.auto_beta_enabled)
    auto_beta_config: dict[str, float | int] = {
        "target_delta_norm": float(args.auto_beta_target_delta_norm),
        "beta_min": float(args.auto_beta_min),
        "beta_max": float(args.auto_beta_max),
        "lr": float(args.auto_beta_lr),
        "ema_decay": float(args.auto_beta_ema_decay),
        "q_margin": float(args.auto_beta_q_margin),
        "update_interval": int(args.auto_beta_update_interval),
    }
    auto_beta_controller = None
    latest_auto_beta_metrics: dict[str, float | bool | str | int | None] = _disabled_auto_beta_metrics(auto_beta_config)
    if auto_beta_enabled:
        auto_beta_controller = _build_auto_beta_controller(runtime_beta, auto_beta_config)
        latest_auto_beta_metrics = auto_beta_controller.metrics()

    initial_actor_dir = _save_actor_for_inference(
        state,
        args.output_dir,
        0,
        action_horizon=shape.action_horizon,
        replay_shape=replay_shape,
        train_shape=shape,
        replay_stats=store.stats,
    )
    _save_training_checkpoint(state, args.output_dir, 0, store)
    metrics_publisher.publish(
        _build_metrics_payload(
            step=0,
            reduced={
                "beta": runtime_beta,
                "online_safety_enabled": online_safety_enabled,
                **latest_auto_beta_metrics,
                **online_controller.metrics(),
            },
            stats=store.stats,
            replay_shape=store.shape,
            train_shape=store.sample_shape,
            actor_enabled=False,
            trainer_enabled=trainer_enabled,
            trainer_running=False,
            critic_burn_in_steps=args.critic_burn_in_steps,
            target_sync_step=latest_target_sync_step,
            latest_actor_path=str(initial_actor_dir),
            latest_actor_step=0,
            wandb_url=_wandb_url(),
        )
    )

    latest_actor_path = str(initial_actor_dir)
    latest_actor_step = 0

    last_scan_time = 0.0
    last_idle_metrics_time = 0.0
    log_start_time = time.perf_counter()
    infos: list[dict[str, np.ndarray]] = []
    progress_total = None if args.num_train_steps <= 0 else args.num_train_steps
    with tqdm.tqdm(total=progress_total, dynamic_ncols=True, desc="training") as progress:
        while args.num_train_steps <= 0 or int(state.step) < args.num_train_steps:
            now = time.monotonic()
            if now - last_scan_time >= args.scan_interval:
                store.scan()
                last_scan_time = now

            control_update = control_subscriber.poll_update()
            if "trainer_enabled" in control_update:
                trainer_enabled = bool(control_update["trainer_enabled"])
            if "online_safety_enabled" in control_update:
                online_safety_enabled = bool(control_update["online_safety_enabled"])
            _update_online_controller_config(online_controller, control_update)
            if "critic_burn_in_steps" in control_update:
                args.critic_burn_in_steps = int(control_update["critic_burn_in_steps"])
            auto_beta_config_changed = False
            if "auto_beta_enabled" in control_update:
                auto_beta_enabled = bool(control_update["auto_beta_enabled"])
                auto_beta_config_changed = True
            auto_beta_key_map = {
                "auto_beta_target_delta_norm": "target_delta_norm",
                "auto_beta_min": "beta_min",
                "auto_beta_max": "beta_max",
                "auto_beta_lr": "lr",
                "auto_beta_ema_decay": "ema_decay",
                "auto_beta_q_margin": "q_margin",
                "auto_beta_update_interval": "update_interval",
            }
            for update_key, config_key in auto_beta_key_map.items():
                if update_key in control_update:
                    auto_beta_config[config_key] = control_update[update_key]  # type: ignore[assignment]
                    auto_beta_config_changed = True
            if auto_beta_config_changed:
                if auto_beta_enabled:
                    if auto_beta_controller is None:
                        auto_beta_controller = _build_auto_beta_controller(runtime_beta, auto_beta_config)
                    else:
                        auto_beta_controller.update_config(
                            target_delta_norm=float(auto_beta_config["target_delta_norm"]),
                            beta_min=float(auto_beta_config["beta_min"]),
                            beta_max=float(auto_beta_config["beta_max"]),
                            lr=float(auto_beta_config["lr"]),
                            ema_decay=float(auto_beta_config["ema_decay"]),
                            q_margin=float(auto_beta_config["q_margin"]),
                            update_interval=int(auto_beta_config["update_interval"]),
                        )
                    if auto_beta_controller.beta != runtime_beta:
                        runtime_beta = auto_beta_controller.beta
                        state = _with_runtime_beta(state, runtime_beta)
                    latest_auto_beta_metrics = auto_beta_controller.metrics()
                    logging.info("Updated auto beta config: %s", latest_auto_beta_metrics)
                else:
                    latest_auto_beta_metrics = _disabled_auto_beta_metrics(auto_beta_config)
                    logging.info("Disabled auto beta; manual beta controls are active")
            beta_update = None if auto_beta_enabled else control_update.get("beta")
            if beta_update is not None and float(beta_update) != runtime_beta:
                runtime_beta = float(beta_update)
                state = _with_runtime_beta(state, runtime_beta)
                logging.info("Updated runtime RLT beta to %.4f", runtime_beta)
            if online_safety_enabled:
                if auto_beta_controller is not None:
                    auto_beta_controller.update_config(
                        target_delta_norm=float(online_controller.target_delta_norm),
                        beta_min=float(online_controller.beta_min),
                        beta_max=float(online_controller.beta_max),
                        lr=float(auto_beta_config["lr"]),
                        ema_decay=float(auto_beta_config["ema_decay"]),
                        q_margin=float(auto_beta_config["q_margin"]),
                        update_interval=int(auto_beta_config["update_interval"]),
                    )
                if not math.isclose(runtime_beta, online_controller.beta, rel_tol=0.0, abs_tol=1e-12):
                    runtime_beta = online_controller.beta
                    state = _with_runtime_beta(state, runtime_beta)
                latest_auto_beta_metrics = {
                    **latest_auto_beta_metrics,
                    "auto_beta_target_delta_norm": online_controller.target_delta_norm,
                    "auto_beta_min": online_controller.beta_min,
                    "auto_beta_max": online_controller.beta_max,
                }

            if not store.ready(
                min_replay_samples=args.min_replay_samples,
                min_success_episodes=args.min_success_episodes,
                min_failure_episodes=args.min_failure_episodes,
                min_replay_shards=args.min_replay_shards,
            ):
                time.sleep(args.wait_sleep_seconds)
                continue

            if not trainer_enabled:
                if now - last_idle_metrics_time >= max(float(args.log_interval), 1.0):
                    stats = store.stats
                    metrics_publisher.publish(
                        _build_metrics_payload(
                            step=int(state.step),
                            reduced={
                                "beta": runtime_beta,
                                "online_safety_enabled": online_safety_enabled,
                                **latest_auto_beta_metrics,
                                **online_controller.metrics(),
                            },
                            stats=stats,
                            replay_shape=store.shape,
                            train_shape=store.sample_shape,
                            actor_enabled=False,
                            trainer_enabled=False,
                            trainer_running=False,
                            critic_burn_in_steps=args.critic_burn_in_steps,
                            target_sync_step=latest_target_sync_step,
                            latest_actor_path=latest_actor_path,
                            latest_actor_step=latest_actor_step,
                            wandb_url=_wandb_url(),
                        )
                    )
                    logging.info("RLT trainer idle; waiting for frontend start command. replay=%s", stats)
                    last_idle_metrics_time = now
                time.sleep(args.wait_sleep_seconds)
                continue

            if online_safety_enabled:
                if online_controller.phase == "idle_wait_new_data":
                    online_controller.maybe_start_round(store.stats)
                if online_controller.phase == "critic_eval":
                    current_step = int(state.step)
                    candidate_dir = _save_actor_for_inference(
                        state,
                        args.output_dir / "candidates" / f"round_{online_controller.round_index:06d}" / "critic",
                        current_step,
                        action_horizon=shape.action_horizon,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        replay_stats=store.stats,
                    )
                    metric = _evaluate_candidate_critic(
                        args=args,
                        checkpoint_dir=candidate_dir,
                        store=store,
                        output_dir=args.output_dir / "candidates" / f"round_{online_controller.round_index:06d}" / "critic_eval",
                        round_index=online_controller.round_index,
                    )
                    if online_controller.accept_critic(metric):
                        accepted_state = state
                        _save_actor_for_inference(
                            accepted_state,
                            args.output_dir / "best_critic",
                            current_step,
                            action_horizon=shape.action_horizon,
                            replay_shape=store.shape,
                            train_shape=store.sample_shape,
                            replay_stats=store.stats,
                        )
                        latest_target_sync_step = None
                        logging.info("Accepted online candidate critic round=%d metric=%s", online_controller.round_index, metric)
                    else:
                        state = accepted_state
                        logging.warning(
                            "Rejected online candidate critic round=%d reason=%s metric=%s",
                            online_controller.round_index,
                            online_controller.last_rejection_reason,
                            metric,
                        )
                    continue
                if online_controller.phase == "actor_eval":
                    current_step = int(state.step)
                    actor_metric = _reduce_numeric_infos(infos) if infos else {}
                    if online_controller.accept_actor(actor_metric):
                        accepted_state = state
                        actor_dir = _save_actor_for_inference(
                            accepted_state,
                            args.output_dir / "best_actor",
                            current_step,
                            action_horizon=shape.action_horizon,
                            replay_shape=store.shape,
                            train_shape=store.sample_shape,
                            replay_stats=store.stats,
                        )
                        latest_actor_path = str(actor_dir)
                        latest_actor_step = current_step
                        logging.info("Accepted online candidate actor round=%d metric=%s", online_controller.round_index, actor_metric)
                    else:
                        state = accepted_state
                        logging.warning(
                            "Rejected online candidate actor round=%d reason=%s metric=%s",
                            online_controller.round_index,
                            online_controller.last_rejection_reason,
                            actor_metric,
                        )
                    infos = []
                    if not math.isclose(runtime_beta, online_controller.beta, rel_tol=0.0, abs_tol=1e-12):
                        runtime_beta = online_controller.beta
                        state = _with_runtime_beta(state, runtime_beta)
                    continue
                allocation = online_controller.step_allocation()
                if not allocation["trainer_running"]:
                    if now - last_idle_metrics_time >= max(float(args.log_interval), 1.0):
                        metrics_publisher.publish(
                            _build_metrics_payload(
                                step=int(state.step),
                                reduced={
                                    "beta": runtime_beta,
                                    "online_safety_enabled": online_safety_enabled,
                                    **latest_auto_beta_metrics,
                                    **online_controller.metrics(),
                                },
                                stats=store.stats,
                                replay_shape=store.shape,
                                train_shape=store.sample_shape,
                                actor_enabled=False,
                                trainer_enabled=trainer_enabled,
                                trainer_running=False,
                                critic_burn_in_steps=args.critic_burn_in_steps,
                                target_sync_step=latest_target_sync_step,
                                latest_actor_path=latest_actor_path,
                                latest_actor_step=latest_actor_step,
                                wandb_url=_wandb_url(),
                            )
                        )
                        logging.info("Online RLT safety idle phase=%s replay=%s", online_controller.phase, store.stats)
                        last_idle_metrics_time = now
                    time.sleep(args.wait_sleep_seconds)
                    continue
                forced_actor_enabled = bool(allocation["actor_enabled"])
            else:
                forced_actor_enabled = None

            next_step = int(state.step) + 1
            actor_enabled = (
                forced_actor_enabled
                if forced_actor_enabled is not None
                else _actor_updates_enabled(args, store, next_step)
            )
            if actor_enabled and latest_target_sync_step is None:
                state = rlt_training.sync_target_params(state)
                latest_target_sync_step = int(state.step)
                logging.info(
                    "Hard-synced target actor/critic before actor updates at step=%d burn_in=%d",
                    latest_target_sync_step,
                    args.critic_burn_in_steps,
                )
            state = _state_for_actor_gate(
                state,
                actor_enabled=actor_enabled,
                policy_delay=args.policy_delay,
                actor_publish_interval=0 if online_safety_enabled else args.actor_publish_interval,
            )
            batch = store.sample_batch(replay_rng, args.batch_size)
            train_rng = jax.random.fold_in(jax.random.key(args.seed), int(state.step))
            state, info = rlt_training.train_step(state, batch, train_rng)
            info = jax.device_get(info)
            current_step = int(state.step)
            if auto_beta_enabled and auto_beta_controller is not None:
                auto_beta_update = auto_beta_controller.update(step=current_step, metrics=info)
                latest_auto_beta_metrics = auto_beta_update.metrics
                if auto_beta_update.changed and auto_beta_update.beta != runtime_beta:
                    runtime_beta = auto_beta_update.beta
                    state = _with_runtime_beta(state, runtime_beta)
                    logging.info(
                        "Auto beta updated step=%d beta=%.4f reason=%s",
                        current_step,
                        runtime_beta,
                        auto_beta_update.reason,
                    )
            info = {**info, "beta": np.asarray(runtime_beta), **latest_auto_beta_metrics}
            infos.append(info)

            if bool(info["publish_actor"]):
                actor_dir = _save_actor_for_inference(
                    state,
                    args.output_dir,
                    current_step,
                    action_horizon=shape.action_horizon,
                    replay_shape=store.shape,
                    train_shape=store.sample_shape,
                    replay_stats=store.stats,
                )
                latest_actor_path = str(actor_dir)
                latest_actor_step = current_step
                metrics_publisher.publish(
                    _build_metrics_payload(
                        step=current_step,
                        reduced={
                            "beta": runtime_beta,
                            "online_safety_enabled": online_safety_enabled,
                            **latest_auto_beta_metrics,
                            **online_controller.metrics(),
                        },
                        stats=store.stats,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        actor_enabled=actor_enabled,
                        trainer_enabled=trainer_enabled,
                        trainer_running=True,
                        critic_burn_in_steps=args.critic_burn_in_steps,
                        target_sync_step=latest_target_sync_step,
                        latest_actor_path=latest_actor_path,
                        latest_actor_step=latest_actor_step,
                        wandb_url=_wandb_url(),
                    )
                )
                logging.info("Published inference actor at step=%d path=%s", current_step, actor_dir)
            if current_step % args.save_interval == 0:
                checkpoint_dir = _save_training_checkpoint(state, args.output_dir, current_step, store)
                _save_actor_for_inference(
                    state,
                    args.output_dir / "snapshots",
                    current_step,
                    action_horizon=shape.action_horizon,
                    replay_shape=store.shape,
                    train_shape=store.sample_shape,
                    replay_stats=store.stats,
                )
                logging.info("Saved RLT training checkpoint at step=%d path=%s", current_step, checkpoint_dir)
            if current_step % args.log_interval == 0 and infos:
                reduced = _reduce_numeric_infos(infos)
                stats = store.stats
                reduced.update(
                    {
                        "actor_enabled": float(actor_enabled),
                        "replay_size": float(stats.replay_size),
                        "replay_shards": float(stats.num_shards),
                        "replay_action_horizon": float(store.shape.action_horizon if store.shape else 0),
                        "train_action_horizon": float(store.sample_shape.action_horizon if store.sample_shape else 0),
                        "success_episodes": float(stats.success_episodes),
                        "failure_episodes": float(stats.failure_episodes),
                        "steps_per_sec": args.log_interval / max(time.perf_counter() - log_start_time, 1e-6),
                    }
                )
                reduced.update(latest_auto_beta_metrics)
                reduced.update(
                    {
                        "online_safety_enabled": online_safety_enabled,
                        **online_controller.metrics(),
                    }
                )
                metrics_publisher.publish(
                    _build_metrics_payload(
                        step=current_step,
                        reduced=reduced,
                        stats=stats,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        actor_enabled=actor_enabled,
                        trainer_enabled=trainer_enabled,
                        trainer_running=True,
                        critic_burn_in_steps=args.critic_burn_in_steps,
                        target_sync_step=latest_target_sync_step,
                        latest_actor_path=latest_actor_path,
                        latest_actor_step=latest_actor_step,
                        wandb_url=_wandb_url(),
                    )
                )
                wandb.log({f"rlt/{key}": value for key, value in reduced.items()}, step=current_step)
                logging.info("step=%d %s", current_step, " ".join(_format_log_metric(k, v) for k, v in reduced.items()))
                infos = []
                log_start_time = time.perf_counter()
            progress.update(1)


if __name__ == "__main__":
    main(tyro.cli(Args))

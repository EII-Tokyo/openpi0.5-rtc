import dataclasses
import datetime
import hashlib
import json
import logging
import math
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
from openpi.training import rlt_replay_store
from openpi.training import rlt_training


@dataclasses.dataclass
class Args:
    replay_dir: pathlib.Path
    segment_db_path: pathlib.Path | None = None
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
    critic_burn_in_steps: int = 0
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
    target_actor_noise: bool = False
    train_action_horizon: int | None = 10
    expected_replay_action_horizon: int | None = 10
    wandb_enabled: bool = True
    wandb_project: str = "openpi"
    wandb_run_name: str = "rlt_actor_critic_online"
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_state_channel: str = "aloha_rlt_state"
    redis_control_channel: str = "aloha_rlt_control"
    overwrite: bool = False


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
        "latest_actor_path": latest_actor_path,
        "latest_actor_step": None if latest_actor_step is None else int(latest_actor_step),
        "replay_shards": int(stats.num_shards),
        "success_episodes": int(stats.success_episodes),
        "failure_episodes": int(stats.failure_episodes),
        "bad_shards": int(stats.bad_shards),
        "replay_action_horizon": 0 if replay_shape is None else int(replay_shape.action_horizon),
        "train_action_horizon": 0 if train_shape is None else int(train_shape.action_horizon),
        "steps_per_sec": _json_float(reduced.get("steps_per_sec")),
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
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **{
                    key: str(value) if isinstance(value, pathlib.Path) else value
                    for key, value in dataclasses.asdict(args).items()
                },
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


def main(args: Args) -> None:
    _init_logging()
    logging.info("Running online RLT trainer on %s", platform.node())

    store = rlt_replay_store.RLTReplayStore(
        args.replay_dir,
        max_replay_samples=args.max_replay_samples,
        recursive=args.recursive_scan,
        sample_action_horizon=args.train_action_horizon,
        segment_db_path=args.segment_db_path,
    )
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
    logging.info("Replay ready: %s, replay_shape=%s, train_shape=%s", store.stats, replay_shape, shape)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    state = rlt_training.init_train_state(config, jax.random.key(args.seed))
    replay_rng = np.random.default_rng(args.seed)
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
            reduced={"beta": runtime_beta, **latest_auto_beta_metrics},
            stats=store.stats,
            replay_shape=store.shape,
            train_shape=store.sample_shape,
            actor_enabled=False,
            trainer_enabled=trainer_enabled,
            trainer_running=False,
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
                            reduced={"beta": runtime_beta, **latest_auto_beta_metrics},
                            stats=stats,
                            replay_shape=store.shape,
                            train_shape=store.sample_shape,
                            actor_enabled=False,
                            trainer_enabled=False,
                            trainer_running=False,
                            latest_actor_path=latest_actor_path,
                            latest_actor_step=latest_actor_step,
                            wandb_url=_wandb_url(),
                        )
                    )
                    logging.info("RLT trainer idle; waiting for frontend start command. replay=%s", stats)
                    last_idle_metrics_time = now
                time.sleep(args.wait_sleep_seconds)
                continue

            next_step = int(state.step) + 1
            actor_enabled = _actor_updates_enabled(args, store, next_step)
            state = _state_for_actor_gate(
                state,
                actor_enabled=actor_enabled,
                policy_delay=args.policy_delay,
                actor_publish_interval=args.actor_publish_interval,
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
                        reduced={"beta": runtime_beta, **latest_auto_beta_metrics},
                        stats=store.stats,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        actor_enabled=actor_enabled,
                        trainer_enabled=trainer_enabled,
                        trainer_running=True,
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

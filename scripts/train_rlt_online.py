import dataclasses
import datetime
import hashlib
import json
import logging
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
    expected_replay_action_horizon: int | None = None
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
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redis_client=None,
    ):
        self._enabled = enabled
        self._channel = channel
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
            self._client.publish(self._channel, json.dumps(payload, sort_keys=True))
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

    def poll_beta_update(self) -> float | None:
        if not self._enabled or self._pubsub is None:
            return None
        latest_beta = None
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
                if payload.get("type") != "config_update" or "beta" not in payload:
                    continue
                try:
                    beta = float(payload["beta"])
                except (TypeError, ValueError):
                    continue
                if beta >= 0.0:
                    latest_beta = beta
        except Exception as exc:
            if not self._warned:
                logging.warning("Failed to read RLT control update from Redis: %s", exc)
                self._warned = True
        return latest_beta

    def close(self) -> None:
        if self._pubsub is None:
            return
        try:
            self._pubsub.close()
        except Exception:
            pass


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


def _build_metrics_payload(
    *,
    step: int,
    reduced: dict,
    stats: rlt_replay_store.ReplayStats,
    replay_shape: rlt_replay_store.ReplayShape | None,
    train_shape: rlt_replay_store.ReplayShape | None,
    actor_enabled: bool,
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
        "replay_size": int(stats.replay_size),
        "wandb_url": wandb_url,
        "actor_enabled": bool(actor_enabled),
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
            reduced={},
            stats=store.stats,
            replay_shape=store.shape,
            train_shape=store.sample_shape,
            actor_enabled=False,
            latest_actor_path=str(initial_actor_dir),
            latest_actor_step=0,
            wandb_url=_wandb_url(),
        )
    )

    latest_actor_path = str(initial_actor_dir)
    latest_actor_step = 0

    last_scan_time = 0.0
    log_start_time = time.perf_counter()
    infos: list[dict[str, np.ndarray]] = []
    progress_total = None if args.num_train_steps <= 0 else args.num_train_steps
    with tqdm.tqdm(total=progress_total, dynamic_ncols=True, desc="training") as progress:
        while args.num_train_steps <= 0 or int(state.step) < args.num_train_steps:
            now = time.monotonic()
            if now - last_scan_time >= args.scan_interval:
                store.scan()
                last_scan_time = now

            if not store.ready(
                min_replay_samples=args.min_replay_samples,
                min_success_episodes=args.min_success_episodes,
                min_failure_episodes=args.min_failure_episodes,
                min_replay_shards=args.min_replay_shards,
            ):
                time.sleep(args.wait_sleep_seconds)
                continue

            next_step = int(state.step) + 1
            beta_update = control_subscriber.poll_beta_update()
            if beta_update is not None and beta_update != runtime_beta:
                runtime_beta = beta_update
                state = _with_runtime_beta(state, runtime_beta)
                logging.info("Updated runtime RLT beta to %.4f", runtime_beta)

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
            infos.append(info)

            current_step = int(state.step)
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
                        reduced={},
                        stats=store.stats,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        actor_enabled=actor_enabled,
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
                reduced = {
                    key: float(np.mean([np.asarray(item[key]) for item in infos]))
                    for key in infos[0]
                    if np.asarray(infos[0][key]).ndim == 0
                }
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
                metrics_publisher.publish(
                    _build_metrics_payload(
                        step=current_step,
                        reduced=reduced,
                        stats=stats,
                        replay_shape=store.shape,
                        train_shape=store.sample_shape,
                        actor_enabled=actor_enabled,
                        latest_actor_path=latest_actor_path,
                        latest_actor_step=latest_actor_step,
                        wandb_url=_wandb_url(),
                    )
                )
                wandb.log({f"rlt/{key}": value for key, value in reduced.items()}, step=current_step)
                logging.info("step=%d %s", current_step, " ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
                infos = []
                log_start_time = time.perf_counter()
            progress.update(1)


if __name__ == "__main__":
    main(tyro.cli(Args))

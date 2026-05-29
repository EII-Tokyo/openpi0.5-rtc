import dataclasses
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
    output_dir: pathlib.Path = pathlib.Path("./checkpoints/rlt_actor_critic/online")
    num_train_steps: int = 0
    batch_size: int = 64
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 1_000
    scan_interval: float = 1.0
    wait_sleep_seconds: float = 1.0
    min_replay_samples: int = 512
    min_success_episodes: int = 1
    min_failure_episodes: int = 1
    critic_burn_in_steps: int = 0
    actor_min_replay_samples: int = 0
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


def _save_actor_for_inference(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    *,
    action_horizon: int,
) -> pathlib.Path:
    actor_dir = output_dir / "inference_actor" / f"{step:08d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_params = rlt_training.actor_params_for_inference(state).to_pure_dict()
    _atomic_write_bytes(actor_dir / "actor.msgpack", serialization.to_bytes(actor_params))
    _atomic_write_text(
        actor_dir / "metadata.json",
        json.dumps(
            {
                "step": step,
                "type": "rlt_inference_actor",
                "note": "Stable actor export. Runtime should switch only at chunk/idle boundary.",
                "action_horizon": action_horizon,
            },
            indent=2,
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
    actor_min_success = args.actor_min_success_episodes or args.min_success_episodes
    actor_min_failure = args.actor_min_failure_episodes or args.min_failure_episodes
    return (
        stats.replay_size >= actor_min_replay
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
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = rlt_replay_store.RLTReplayStore(
        args.replay_dir,
        max_replay_samples=args.max_replay_samples,
        recursive=args.recursive_scan,
        sample_action_horizon=args.train_action_horizon,
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

    _save_actor_for_inference(state, args.output_dir, 0, action_horizon=shape.action_horizon)
    _save_training_checkpoint(state, args.output_dir, 0, store)

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
            ):
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
            infos.append(info)

            current_step = int(state.step)
            if bool(info["publish_actor"]):
                actor_dir = _save_actor_for_inference(state, args.output_dir, current_step, action_horizon=shape.action_horizon)
                logging.info("Published inference actor at step=%d path=%s", current_step, actor_dir)
            if current_step % args.save_interval == 0:
                checkpoint_dir = _save_training_checkpoint(state, args.output_dir, current_step, store)
                _save_actor_for_inference(state, args.output_dir / "snapshots", current_step, action_horizon=shape.action_horizon)
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
                wandb.log({f"rlt/{key}": value for key, value in reduced.items()}, step=current_step)
                logging.info("step=%d %s", current_step, " ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
                infos = []
                log_start_time = time.perf_counter()
            progress.update(1)


if __name__ == "__main__":
    main(tyro.cli(Args))

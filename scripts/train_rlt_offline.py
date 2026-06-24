from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import platform
import shutil
import time

import jax
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training
from scripts import train_rlt_online


@dataclasses.dataclass
class Args:
    replay_dir: pathlib.Path
    output_dir: pathlib.Path = pathlib.Path("./checkpoints/rlt_actor_critic/offline")
    num_train_steps: int = 10_000
    batch_size: int = 64
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 1_000
    min_replay_samples: int = 512
    min_replay_shards: int = 0
    min_success_episodes: int = 1
    min_failure_episodes: int = 1
    critic_burn_in_steps: int = 1_000
    actor_min_replay_samples: int = 0
    actor_min_replay_shards: int = 0
    actor_min_success_episodes: int = 0
    actor_min_failure_episodes: int = 0
    max_replay_samples: int | None = None
    recursive_scan: bool = False
    segment_db_path: pathlib.Path | None = None
    policy_delay: int = 2
    actor_publish_interval: int = 500
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    beta: float = 10.0
    target_actor_noise: bool = True
    actor_loss_mode: str = "td3"
    awbc_temperature: float = 0.2
    awbc_max_weight: float = 20.0
    awbc_min_advantage: float = 0.0
    awbc_max_action_delta_norm: float = 2.0
    train_action_horizon: int | None = 10
    expected_replay_action_horizon: int | None = 10
    wandb_enabled: bool = True
    wandb_project: str = "openpi-rlt-offline"
    wandb_run_name: str = "rlt_actor_critic_offline"
    overwrite: bool = False


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_replay_store(args: Args) -> rlt_replay_store.RLTReplayStore:
    return rlt_replay_store.RLTReplayStore(
        args.replay_dir,
        max_replay_samples=args.max_replay_samples,
        recursive=args.recursive_scan,
        sample_action_horizon=args.train_action_horizon,
        segment_db_path=args.segment_db_path,
    )


def _build_training_config(
    args: Args,
    shape: rlt_replay_store.ReplayShape,
) -> rlt_training.RLTTrainingConfig:
    return rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=shape.z_dim,
            proprio_dim=shape.proprio_dim,
            action_horizon=shape.action_horizon,
            action_dim=shape.action_dim,
            beta=args.beta,
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


def _require_ready(args: Args, store: rlt_replay_store.RLTReplayStore) -> None:
    store.scan()
    if not store.ready(
        min_replay_samples=args.min_replay_samples,
        min_success_episodes=args.min_success_episodes,
        min_failure_episodes=args.min_failure_episodes,
        min_replay_shards=args.min_replay_shards,
    ):
        raise ValueError(
            "Replay dataset is not ready: "
            f"stats={store.stats} required_samples={args.min_replay_samples} "
            f"required_success={args.min_success_episodes} required_failure={args.min_failure_episodes} "
            f"required_shards={args.min_replay_shards}"
        )
    if store.shape is None or store.sample_shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    if args.expected_replay_action_horizon is not None and store.shape.action_horizon != args.expected_replay_action_horizon:
        raise ValueError(
            f"Expected replay action horizon {args.expected_replay_action_horizon}, got {store.shape.action_horizon}"
        )


def _init_wandb(args: Args, store: rlt_replay_store.RLTReplayStore) -> None:
    if not args.wandb_enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            **{key: str(value) if isinstance(value, pathlib.Path) else value for key, value in dataclasses.asdict(args).items()},
            "replay_stats": dataclasses.asdict(store.stats),
            "replay_shape": None if store.shape is None else dataclasses.asdict(store.shape),
            "train_shape": None if store.sample_shape is None else dataclasses.asdict(store.sample_shape),
        },
    )


def _write_summary(
    args: Args,
    store: rlt_replay_store.RLTReplayStore,
    *,
    final_step: int,
    latest_actor_path: str,
    target_sync_step: int | None,
) -> None:
    summary = {
        "host": platform.node(),
        "source_script": "scripts/train_rlt_offline.py",
        "final_step": final_step,
        "latest_actor_path": latest_actor_path,
        "target_sync_step": target_sync_step,
        "args": {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in dataclasses.asdict(args).items()},
        "replay_stats": dataclasses.asdict(store.stats),
        "replay_shape": None if store.shape is None else dataclasses.asdict(store.shape),
        "train_shape": None if store.sample_shape is None else dataclasses.asdict(store.sample_shape),
        "loaded_shards": [str(path) for path in store.loaded_paths],
    }
    train_rlt_online._atomic_write_text(args.output_dir / "training_summary.json", json.dumps(summary, indent=2))


def main(args: Args) -> None:
    _init_logging()
    logging.info("Running offline RLT trainer on %s", platform.node())
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = _build_replay_store(args)
    _require_ready(args, store)
    replay_shape = store.shape
    train_shape = store.sample_shape
    if replay_shape is None or train_shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    logging.info("Replay ready: stats=%s replay_shape=%s train_shape=%s", store.stats, replay_shape, train_shape)

    config = _build_training_config(args, train_shape)
    state = rlt_training.init_train_state(config, jax.random.key(args.seed))
    replay_rng = np.random.default_rng(args.seed)
    _init_wandb(args, store)

    latest_actor_dir = train_rlt_online._save_actor_for_inference(
        state,
        args.output_dir,
        0,
        action_horizon=train_shape.action_horizon,
        replay_shape=replay_shape,
        train_shape=train_shape,
        replay_stats=store.stats,
    )
    train_rlt_online._save_training_checkpoint(state, args.output_dir, 0, store)
    latest_target_sync_step: int | None = None
    infos: list[dict[str, np.ndarray]] = []
    log_start = time.perf_counter()

    with tqdm.tqdm(total=args.num_train_steps, dynamic_ncols=True, desc="offline-rlt") as progress:
        while int(state.step) < args.num_train_steps:
            next_step = int(state.step) + 1
            actor_enabled = train_rlt_online._actor_updates_enabled(args, store, next_step)
            if actor_enabled and latest_target_sync_step is None:
                state = rlt_training.sync_target_params(state)
                latest_target_sync_step = int(state.step)
                logging.info("Hard-synced target actor/critic before actor updates at step=%d", latest_target_sync_step)
            state = train_rlt_online._state_for_actor_gate(
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
                latest_actor_dir = train_rlt_online._save_actor_for_inference(
                    state,
                    args.output_dir,
                    current_step,
                    action_horizon=train_shape.action_horizon,
                    replay_shape=replay_shape,
                    train_shape=train_shape,
                    replay_stats=store.stats,
                )
            if current_step % args.save_interval == 0:
                train_rlt_online._save_training_checkpoint(state, args.output_dir, current_step, store)
                train_rlt_online._save_actor_for_inference(
                    state,
                    args.output_dir / "snapshots",
                    current_step,
                    action_horizon=train_shape.action_horizon,
                    replay_shape=replay_shape,
                    train_shape=train_shape,
                    replay_stats=store.stats,
                )
            if current_step % args.log_interval == 0 and infos:
                reduced = train_rlt_online._reduce_numeric_infos(infos)
                reduced.update(
                    {
                        "actor_enabled": float(actor_enabled),
                        "replay_size": float(store.stats.replay_size),
                        "replay_shards": float(store.stats.num_shards),
                        "success_episodes": float(store.stats.success_episodes),
                        "failure_episodes": float(store.stats.failure_episodes),
                        "steps_per_sec": args.log_interval / max(time.perf_counter() - log_start, 1e-6),
                    }
                )
                wandb.log({f"rlt/{key}": value for key, value in reduced.items()}, step=current_step)
                logging.info("step=%d %s", current_step, " ".join(train_rlt_online._format_log_metric(k, v) for k, v in reduced.items()))
                infos = []
                log_start = time.perf_counter()
            progress.update(1)

    _write_summary(
        args,
        store,
        final_step=int(state.step),
        latest_actor_path=str(latest_actor_dir),
        target_sync_step=latest_target_sync_step,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

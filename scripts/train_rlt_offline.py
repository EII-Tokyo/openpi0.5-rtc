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
from flax import serialization

from openpi.models import rlt
from openpi.training import rlt_eval
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
    training_stage: str = "critic_actor"
    critic_auc_threshold: float | None = None
    require_positive_q_gap: bool = False
    actor_min_replay_samples: int = 0
    actor_min_replay_shards: int = 0
    actor_min_success_episodes: int = 0
    actor_min_failure_episodes: int = 0
    init_critic_checkpoint: pathlib.Path | None = None
    max_replay_samples: int | None = None
    recursive_scan: bool = False
    segment_db_path: pathlib.Path | None = None
    manifest_path: pathlib.Path | None = None
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
    eval_holdout_critic: bool = False
    holdout_ratio: float = 0.2
    holdout_seed: int = 42
    eval_holdout_every_steps: int = 1_000
    holdout_score_batch_size: int = 512


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
        manifest_path=args.manifest_path,
    )


def _prepare_holdout_split(args: Args) -> tuple[pathlib.Path | None, pathlib.Path | None, dict[str, int]]:
    if not args.eval_holdout_critic:
        return None, None, {}
    shards = rlt_eval.find_replay_shards(
        args.replay_dir,
        recursive=args.recursive_scan,
        segment_db_path=args.segment_db_path,
        manifest_path=args.manifest_path,
    )
    split = rlt_eval.split_shards(shards, holdout_ratio=args.holdout_ratio, seed=args.holdout_seed)
    split_dir = args.output_dir / "holdout_split"
    train_manifest = rlt_eval.write_manifest(split.train_paths, split_dir / "train_manifest.jsonl")
    holdout_manifest = rlt_eval.write_manifest(split.holdout_paths, split_dir / "holdout_manifest.jsonl")
    summary = {
        "num_total_shards": len(shards),
        "num_train_shards": len(split.train_paths),
        "num_holdout_shards": len(split.holdout_paths),
    }
    train_rlt_online._atomic_write_text(
        split_dir / "summary.json",
        json.dumps(
            {
                **summary,
                "holdout_ratio": args.holdout_ratio,
                "holdout_seed": args.holdout_seed,
                "train_manifest": str(train_manifest),
                "holdout_manifest": str(holdout_manifest),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return train_manifest, holdout_manifest, summary


def _build_train_store(args: Args, train_manifest: pathlib.Path | None) -> rlt_replay_store.RLTReplayStore:
    if train_manifest is None:
        return _build_replay_store(args)
    return rlt_replay_store.RLTReplayStore(
        args.replay_dir,
        max_replay_samples=args.max_replay_samples,
        recursive=args.recursive_scan,
        sample_action_horizon=args.train_action_horizon,
        manifest_path=train_manifest,
    )


def _build_training_config(
    args: Args,
    shape: rlt_replay_store.ReplayShape,
) -> rlt_training.RLTTrainingConfig:
    critic_lr = 0.0 if args.training_stage == "actor_only" else args.critic_lr
    return rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=shape.z_dim,
            proprio_dim=shape.proprio_dim,
            action_horizon=shape.action_horizon,
            action_dim=shape.action_dim,
            beta=args.beta,
        ),
        actor_lr=args.actor_lr,
        critic_lr=critic_lr,
        policy_delay=args.policy_delay,
        actor_publish_interval=args.actor_publish_interval,
        target_actor_noise=args.target_actor_noise,
        actor_loss_mode=args.actor_loss_mode,
        awbc_temperature=args.awbc_temperature,
        awbc_max_weight=args.awbc_max_weight,
        awbc_min_advantage=args.awbc_min_advantage,
        awbc_max_action_delta_norm=args.awbc_max_action_delta_norm,
    )


def _validate_training_stage(stage: str) -> None:
    if stage not in {"critic_only", "actor_only", "critic_actor"}:
        raise ValueError("training_stage must be one of: critic_only, actor_only, critic_actor")


def _critic_gate_allows_actor(args: Args, metric: dict | None) -> bool:
    if args.critic_auc_threshold is None and not args.require_positive_q_gap:
        return True
    if metric is None:
        return False
    if args.critic_auc_threshold is not None:
        auc = metric.get("auc")
        if auc is None or float(auc) < float(args.critic_auc_threshold):
            return False
    if args.require_positive_q_gap:
        q_gap = metric.get("q_gap")
        if q_gap is None or float(q_gap) <= 0.0:
            return False
    return True


def _actor_updates_allowed(
    args: Args,
    *,
    stats: rlt_replay_store.ReplayStats,
    step: int,
    critic_gate_open: bool,
) -> bool:
    _validate_training_stage(args.training_stage)
    if args.training_stage == "critic_only":
        return False
    if not critic_gate_open:
        return False
    if args.training_stage == "critic_actor" and step < args.critic_burn_in_steps:
        return False
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


def _load_critic_checkpoint(state: rlt_training.RLTTrainState, checkpoint_path: pathlib.Path) -> rlt_training.RLTTrainState:
    train_state_path = checkpoint_path
    if checkpoint_path.is_dir():
        train_state_path = checkpoint_path / "train_state.msgpack"
    payload = serialization.msgpack_restore(train_state_path.read_bytes())
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{train_state_path} is not an RLT training checkpoint")
    return rlt_training.load_critic_params_from_state_dict(state, payload["params"], reset_step=True)


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
        "manifest_path": None if args.manifest_path is None else str(args.manifest_path),
        "replay_stats": dataclasses.asdict(store.stats),
        "replay_shape": None if store.shape is None else dataclasses.asdict(store.shape),
        "train_shape": None if store.sample_shape is None else dataclasses.asdict(store.sample_shape),
        "loaded_shards": [str(path) for path in store.loaded_paths],
    }
    train_rlt_online._atomic_write_text(args.output_dir / "training_summary.json", json.dumps(summary, indent=2))


def _write_used_manifest(output_dir: pathlib.Path, store: rlt_replay_store.RLTReplayStore) -> pathlib.Path:
    manifest_path = output_dir / "used_manifest.jsonl"
    rows = [{"shard_path": str(path)} for path in store.loaded_paths]
    train_rlt_online._atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    return manifest_path


def _write_latest_actor_pointer(output_dir: pathlib.Path, actor_path: pathlib.Path | str) -> None:
    train_rlt_online._atomic_write_text(output_dir / "latest_actor_path.txt", str(actor_path) + "\n")


def main(args: Args) -> None:
    _init_logging()
    _validate_training_stage(args.training_stage)
    logging.info("Running offline RLT trainer on %s", platform.node())
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest, holdout_manifest, holdout_summary = _prepare_holdout_split(args)
    store = _build_train_store(args, train_manifest)
    _require_ready(args, store)
    _write_used_manifest(args.output_dir, store)
    replay_shape = store.shape
    train_shape = store.sample_shape
    if replay_shape is None or train_shape is None:
        raise ValueError(f"No valid replay shards found in {args.replay_dir}")
    logging.info("Replay ready: stats=%s replay_shape=%s train_shape=%s", store.stats, replay_shape, train_shape)

    config = _build_training_config(args, train_shape)
    state = rlt_training.init_train_state(config, jax.random.key(args.seed))
    if args.init_critic_checkpoint is not None:
        state = _load_critic_checkpoint(state, args.init_critic_checkpoint)
        logging.info("Initialized critic from %s; actor remains freshly initialized", args.init_critic_checkpoint)
    if args.training_stage == "actor_only":
        state = rlt_training.sync_target_params(state)
        logging.info("Synced target critic before actor_only training")
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
    if args.eval_holdout_critic and holdout_manifest is not None:
        initial_eval = rlt_eval.evaluate_holdout_checkpoints(
            checkpoint_dirs=[latest_actor_dir],
            holdout_paths=rlt_eval.find_replay_shards(args.replay_dir, manifest_path=holdout_manifest),
            output_dir=args.output_dir / "holdout_eval" / f"{0:08d}",
            score_batch_size=args.holdout_score_batch_size,
        )
        latest_critic_metric = initial_eval.best_metric
    else:
        latest_critic_metric = None
    _write_latest_actor_pointer(args.output_dir, latest_actor_dir)
    latest_target_sync_step: int | None = None
    infos: list[dict[str, np.ndarray]] = []
    log_start = time.perf_counter()

    with tqdm.tqdm(total=args.num_train_steps, dynamic_ncols=True, desc="offline-rlt") as progress:
        while int(state.step) < args.num_train_steps:
            next_step = int(state.step) + 1
            critic_gate_open = _critic_gate_allows_actor(args, latest_critic_metric)
            actor_enabled = _actor_updates_allowed(
                args,
                stats=store.stats,
                step=next_step,
                critic_gate_open=critic_gate_open,
            )
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
                train_rlt_online._atomic_write_text(
                    latest_actor_dir / "metrics.json",
                    json.dumps({"critic_loss": float(info["critic_loss"])}, indent=2, sort_keys=True),
                )
                _write_latest_actor_pointer(args.output_dir, latest_actor_dir)
            if current_step % args.save_interval == 0:
                train_rlt_online._save_training_checkpoint(state, args.output_dir, current_step, store)
                snapshot_dir = train_rlt_online._save_actor_for_inference(
                    state,
                    args.output_dir / "snapshots",
                    current_step,
                    action_horizon=train_shape.action_horizon,
                    replay_shape=replay_shape,
                    train_shape=train_shape,
                    replay_stats=store.stats,
                )
                train_rlt_online._atomic_write_text(
                    snapshot_dir / "metrics.json",
                    json.dumps({"critic_loss": float(info["critic_loss"])}, indent=2, sort_keys=True),
                )
                if (
                    args.eval_holdout_critic
                    and holdout_manifest is not None
                    and args.eval_holdout_every_steps > 0
                    and current_step % args.eval_holdout_every_steps == 0
                ):
                    eval_result = rlt_eval.evaluate_holdout_checkpoints(
                        checkpoint_dirs=[snapshot_dir],
                        holdout_paths=rlt_eval.find_replay_shards(args.replay_dir, manifest_path=holdout_manifest),
                        output_dir=args.output_dir / "holdout_eval" / f"{current_step:08d}",
                        score_batch_size=args.holdout_score_batch_size,
                    )
                    latest_critic_metric = eval_result.best_metric
            if current_step % args.log_interval == 0 and infos:
                reduced = train_rlt_online._reduce_numeric_infos(infos)
                reduced.update(
                    {
                        "actor_enabled": float(actor_enabled),
                        "replay_size": float(store.stats.replay_size),
                        "replay_shards": float(store.stats.num_shards),
                        "success_episodes": float(store.stats.success_episodes),
                        "failure_episodes": float(store.stats.failure_episodes),
                        "critic_gate_open": float(critic_gate_open),
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
    if args.eval_holdout_critic and holdout_manifest is not None:
        final_eval = rlt_eval.evaluate_holdout_checkpoints(
            checkpoint_dirs=rlt_eval.discover_inference_checkpoints(args.output_dir / "snapshots"),
            holdout_paths=rlt_eval.find_replay_shards(args.replay_dir, manifest_path=holdout_manifest),
            output_dir=args.output_dir / "holdout_eval",
            score_batch_size=args.holdout_score_batch_size,
        )
        if final_eval.best_metric is not None:
            logging.info("Best holdout critic: %s", final_eval.best_metric["checkpoint_path"])


if __name__ == "__main__":
    main(tyro.cli(Args))

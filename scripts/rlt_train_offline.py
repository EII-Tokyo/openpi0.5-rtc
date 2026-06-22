from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils import data as torch_data

from openpi.rlt import actor_critic
from openpi.rlt import replay


def _maybe_init_wandb(
    args: argparse.Namespace,
    config: actor_critic.RLTActorCriticConfig,
    dataset: replay.ReplayDataset,
    *,
    train_size: int,
    val_size: int,
):
    if not args.wandb_project:
        return None
    try:
        import wandb  # pytype: disable=import-error
    except ImportError as exc:
        raise ImportError("Install wandb or omit --wandb-project to disable logging.") from exc

    run_config = {
        **dataclasses.asdict(config),
        "replay_dir": str(args.replay_dir),
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "start_step": args.start_step,
        "init_actor_critic_checkpoint": args.init_actor_critic_checkpoint,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "shuffle": not args.no_shuffle,
        "drop_last": args.drop_last,
        "dataset_size": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "val_episode_policy": "fixed_1_success_5_failure",
        "split_by_episode": True,
        "train_episode_label_filter": args.train_episode_label_filter,
        "balance_train_episode_labels": args.balance_train_episode_labels,
        "oversample_success_train_episodes": args.oversample_success_train_episodes,
        "repeat_success_train_episodes": args.repeat_success_train_episodes,
        "near_terminal_window": args.near_terminal_window,
        "near_terminal_sample_weight": args.near_terminal_sample_weight,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=run_config,
        tags=args.wandb_tag,
    )


def _to_float(value: Any) -> float:
    return float(np.asarray(value))


def _tree_diff_summary(params, target_params, prefix: str) -> dict[str, float]:
    diffs = jax.tree.leaves(jax.tree.map(lambda p, t: np.asarray(p) - np.asarray(t), params, target_params))
    targets = jax.tree.leaves(jax.tree.map(np.asarray, target_params))
    if not diffs:
        return {}
    flat_diff = np.concatenate([x.reshape(-1) for x in diffs])
    flat_target = np.concatenate([x.reshape(-1) for x in targets])
    l2 = float(np.linalg.norm(flat_diff))
    target_l2 = float(np.linalg.norm(flat_target))
    return {
        f"{prefix}/l2": l2,
        f"{prefix}/relative_l2": l2 / max(target_l2, 1e-12),
        f"{prefix}/mean_abs": float(np.mean(np.abs(flat_diff))),
        f"{prefix}/max_abs": float(np.max(np.abs(flat_diff))),
    }


def _q_target_diff_summary(params, target_params) -> dict[str, float]:
    return {
        **_tree_diff_summary(params["critic1"], target_params["critic1"], "target_diff/critic1"),
        **_tree_diff_summary(params["critic2"], target_params["critic2"], "target_diff/critic2"),
        **_tree_diff_summary(
            {"critic1": params["critic1"], "critic2": params["critic2"]},
            {"critic1": target_params["critic1"], "critic2": target_params["critic2"]},
            "target_diff/critic_total",
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _training_args_summary(args: argparse.Namespace, config_overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "args": _json_safe(vars(args)),
        "config_overrides": _json_safe(config_overrides),
        "effective_runtime": {
            "shuffle": not args.no_shuffle,
            "val_episode_policy": "fixed_1_success_5_failure",
            "split_by_episode": True,
            "rlt_token_layernorm": "model",
        },
    }


def _split_summary(
    *,
    replay_dir: Path,
    dataset: replay.ReplayDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_episodes: np.ndarray,
    val_episodes: np.ndarray,
    initial_train_label_counts: dict[str, int],
    train_label_counts: dict[str, int],
    val_label_counts: dict[str, int],
    args: argparse.Namespace,
    success_oversample_stats: dict[str, Any],
    success_repeat_stats: dict[str, Any],
    near_terminal_stats: dict[str, Any],
    step: int | None = None,
) -> dict[str, Any]:
    summary = {
        "replay_dir": str(replay_dir),
        "dataset_size": int(len(dataset)),
        "train_size": int(train_indices.size),
        "val_size": int(val_indices.size),
        "train_episodes": [int(x) for x in train_episodes.tolist()],
        "val_episodes": [int(x) for x in val_episodes.tolist()],
        "initial_train_episode_label_counts": initial_train_label_counts,
        "train_episode_label_counts": train_label_counts,
        "val_episode_label_counts": val_label_counts,
        "train_episode_label_filter": args.train_episode_label_filter,
        "balance_train_episode_labels": bool(args.balance_train_episode_labels),
        "oversample_success_train_episodes": success_oversample_stats,
        "repeat_success_train_episodes": success_repeat_stats,
        "near_terminal_sampling": near_terminal_stats,
        "source_files": getattr(dataset, "source_files", []),
        "seed": int(args.seed),
        "val_episode_policy": "fixed_1_success_5_failure",
    }
    if step is not None:
        summary["step"] = int(step)
    return summary


def _split_indices_by_episode(dataset: replay.ReplayDataset, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    unique_episodes = np.unique(episode_ids)
    rewards = np.asarray(dataset.data["reward"])
    done = np.asarray(dataset.data["done"]).astype(bool)

    def is_success_episode(episode) -> bool:
        episode_mask = episode_ids == episode
        terminal_rewards = rewards[episode_mask & done]
        return bool(terminal_rewards.size and float(terminal_rewards[-1]) > 0.5)

    success_episodes = np.asarray([episode for episode in unique_episodes if is_success_episode(episode)], dtype=unique_episodes.dtype)
    failure_episodes = np.asarray([episode for episode in unique_episodes if not is_success_episode(episode)], dtype=unique_episodes.dtype)
    if success_episodes.size < 1 or failure_episodes.size < 5:
        raise ValueError(
            "Fixed validation split requires at least 1 success episode and 5 failure episodes, got "
            f"success={success_episodes.size} failure={failure_episodes.size}"
        )

    rng = np.random.default_rng(seed)
    val_success = rng.choice(success_episodes, size=1, replace=False)
    val_failure = rng.choice(failure_episodes, size=5, replace=False)
    val_episodes = set(np.concatenate([val_success, val_failure]).tolist())
    val_mask = np.asarray([episode in val_episodes for episode in episode_ids], dtype=bool)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    return all_indices[~val_mask], all_indices[val_mask]


def _episode_label_map(dataset: replay.ReplayDataset, episodes: np.ndarray) -> dict[int, str]:
    return {int(episode_id): _episode_label(dataset, int(episode_id)) for episode_id in episodes}


def _episode_label_counts(dataset: replay.ReplayDataset, indices: np.ndarray) -> dict[str, int]:
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    episodes = np.unique(episode_ids[indices]) if indices.size else np.asarray([], dtype=np.int32)
    labels = _episode_label_map(dataset, episodes)
    return {
        "success": sum(1 for label in labels.values() if label == "success"),
        "failure": sum(1 for label in labels.values() if label == "failure"),
    }


def _balance_indices_by_episode_label(dataset: replay.ReplayDataset, indices: np.ndarray, *, seed: int) -> np.ndarray:
    if indices.size == 0:
        return indices
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    episodes = np.unique(episode_ids[indices])
    labels = _episode_label_map(dataset, episodes)
    success_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "success"], dtype=episodes.dtype)
    failure_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "failure"], dtype=episodes.dtype)
    if success_episodes.size == 0 or failure_episodes.size == 0:
        raise ValueError(
            "Cannot balance train split by episode label because one class is empty: "
            f"success={success_episodes.size} failure={failure_episodes.size}"
        )

    keep_count = int(min(success_episodes.size, failure_episodes.size))
    rng = np.random.default_rng(seed)
    success_keep = rng.choice(success_episodes, size=keep_count, replace=False)
    failure_keep = rng.choice(failure_episodes, size=keep_count, replace=False)
    keep_episodes = set(np.concatenate([success_keep, failure_keep]).tolist())
    keep_mask = np.asarray([episode in keep_episodes for episode in episode_ids], dtype=bool)
    index_mask = np.zeros((len(dataset),), dtype=bool)
    index_mask[indices] = True
    balanced = np.flatnonzero(index_mask & keep_mask).astype(np.int64)
    rng.shuffle(balanced)
    return balanced


def _filter_indices_by_episode_label(dataset: replay.ReplayDataset, indices: np.ndarray, label: str) -> np.ndarray:
    if label == "all" or indices.size == 0:
        return indices
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    episodes = np.unique(episode_ids[indices])
    labels = _episode_label_map(dataset, episodes)
    keep_episodes = {int(episode) for episode in episodes if labels[int(episode)] == label}
    keep_mask = np.asarray([int(episode) in keep_episodes for episode in episode_ids], dtype=bool)
    index_mask = np.zeros((len(dataset),), dtype=bool)
    index_mask[indices] = True
    return np.flatnonzero(index_mask & keep_mask).astype(np.int64)


def _oversample_success_episodes_to_match_failures(
    dataset: replay.ReplayDataset,
    indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if indices.size == 0:
        return indices, {"enabled": True}
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    episodes = np.unique(episode_ids[indices])
    labels = _episode_label_map(dataset, episodes)
    success_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "success"], dtype=episodes.dtype)
    failure_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "failure"], dtype=episodes.dtype)
    if success_episodes.size == 0 or failure_episodes.size == 0:
        raise ValueError(
            "Cannot oversample success episodes because one class is empty: "
            f"success={success_episodes.size} failure={failure_episodes.size}"
        )
    stats = {
        "enabled": True,
        "initial_success_episodes": int(success_episodes.size),
        "initial_failure_episodes": int(failure_episodes.size),
        "sampled_success_episodes": int(success_episodes.size),
        "sampled_failure_episodes": int(failure_episodes.size),
    }
    if success_episodes.size >= failure_episodes.size:
        return indices, stats

    rng = np.random.default_rng(seed)
    sampled_success = rng.choice(success_episodes, size=failure_episodes.size, replace=True)
    episode_sequence = np.concatenate([failure_episodes, sampled_success])
    rng.shuffle(episode_sequence)
    episode_chunks = [np.flatnonzero(episode_ids == episode).astype(np.int64) for episode in episode_sequence]
    oversampled = np.concatenate(episode_chunks).astype(np.int64)
    stats["sampled_success_episodes"] = int(sampled_success.size)
    stats["sampled_failure_episodes"] = int(failure_episodes.size)
    return oversampled, stats


def _repeat_success_episodes(
    dataset: replay.ReplayDataset,
    indices: np.ndarray,
    *,
    repeat: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    if repeat <= 1:
        return indices, {"enabled": False, "repeat": int(repeat)}
    if indices.size == 0:
        return indices, {"enabled": True, "repeat": int(repeat)}
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    episodes = np.unique(episode_ids[indices])
    labels = _episode_label_map(dataset, episodes)
    success_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "success"], dtype=episodes.dtype)
    failure_episodes = np.asarray([episode for episode in episodes if labels[int(episode)] == "failure"], dtype=episodes.dtype)
    if success_episodes.size == 0:
        raise ValueError("Cannot repeat success episodes because train split has no success episodes")

    episode_sequence = [*failure_episodes.tolist(), *np.repeat(success_episodes, repeat).tolist()]
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_sequence)
    episode_chunks = [np.flatnonzero(episode_ids == episode).astype(np.int64) for episode in episode_sequence]
    repeated = np.concatenate(episode_chunks).astype(np.int64)
    return repeated, {
        "enabled": True,
        "repeat": int(repeat),
        "initial_success_episodes": int(success_episodes.size),
        "initial_failure_episodes": int(failure_episodes.size),
        "sampled_success_episodes": int(success_episodes.size * repeat),
        "sampled_failure_episodes": int(failure_episodes.size),
    }


def _near_terminal_weights(
    dataset: replay.ReplayDataset,
    indices: np.ndarray,
    *,
    window: int,
    sample_weight: float,
) -> np.ndarray | None:
    if window <= 0 or sample_weight <= 1.0:
        return None
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    weights = np.ones((indices.shape[0],), dtype=np.float64)
    for episode_id in np.unique(episode_ids[indices]):
        positions = np.flatnonzero(episode_ids[indices] == episode_id)
        if positions.size:
            weights[positions[-window:]] = float(sample_weight)
    return weights


def _episode_label(dataset: replay.ReplayDataset, episode_id: int) -> str:
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    mask = episode_ids == episode_id
    done = np.asarray(dataset.data["done"])[mask].astype(bool)
    rewards = np.asarray(dataset.data["reward"])[mask]
    terminal_rewards = rewards[done]
    if terminal_rewards.size and float(terminal_rewards[-1]) > 0.5:
        return "success"
    return "failure"


def _find_curve_episodes(dataset: replay.ReplayDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    episode_ids = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    selected = {}
    for label in ("success", "failure"):
        for episode_id in np.unique(episode_ids[indices]):
            if _episode_label(dataset, int(episode_id)) == label:
                selected[label] = np.where(episode_ids == episode_id)[0]
                break
    return selected


def _batch_from_indices(dataset: replay.ReplayDataset, indices: np.ndarray) -> replay.ReplayBatch:
    return replay.ReplayBatch(**{key: np.asarray(value[indices]) for key, value in dataset.data.items()})


def _write_curve_csv(path: Path, step_index: np.ndarray, scores: dict[str, np.ndarray], reward: np.ndarray, done: np.ndarray) -> None:
    rows = np.zeros(
        step_index.shape[0],
        dtype=[
            ("step_index", "i4"),
            ("executed_q", "f4"),
            ("actor_q", "f4"),
            ("actor_minus_executed_q", "f4"),
            ("actor_mae", "f4"),
            ("reward", "f4"),
            ("done", "i4"),
        ],
    )
    rows["step_index"] = step_index.astype(np.int32)
    rows["executed_q"] = scores["executed_q"]
    rows["actor_q"] = scores["actor_q"]
    rows["actor_minus_executed_q"] = scores["actor_q"] - scores["executed_q"]
    rows["actor_mae"] = scores["actor_mae"]
    rows["reward"] = reward.astype(np.float32)
    rows["done"] = done.astype(np.int32)
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=",".join(rows.dtype.names),
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f", "%d"],
    )


def _plot_q_curve(path: Path, *, title: str, step_index: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.plot(step_index, scores["executed_q"], label="Q(executed action)", linewidth=1.4)
    ax.plot(step_index, scores["actor_q"], label="Q(actor action)", linewidth=1.4)
    ax.axhline(0.0, color="#7A828F", linewidth=1.0, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Replay step index")
    ax.set_ylabel("Critic Q, min(Q1, Q2)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _make_dataloader(
    dataset: replay.ReplayDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    seed: int,
    sample_weights: np.ndarray | None = None,
):
    subset = torch_data.Subset(dataset, indices.tolist())
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if not shuffle:
            raise ValueError("sample_weights requires shuffle=True")
        sampler = torch_data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return torch_data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=replay.collate_replay_batch,
        generator=generator,
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def _save(
    path: Path,
    params,
    target_params,
    config,
    *,
    actor_opt=None,
    critic_opt=None,
    training_args: dict[str, Any] | None = None,
) -> dict[str, float]:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, params), f)
    with (path / "target_params.pkl").open("wb") as f:
        pickle.dump(jax.tree.map(np.asarray, target_params), f)
    if actor_opt is not None and critic_opt is not None:
        with (path / "optimizer_state.pkl").open("wb") as f:
            pickle.dump(
                {
                    "actor": jax.tree.map(np.asarray, actor_opt),
                    "critic": jax.tree.map(np.asarray, critic_opt),
                },
                f,
            )
    diff_summary = _q_target_diff_summary(jax.tree.map(np.asarray, params), jax.tree.map(np.asarray, target_params))
    (path / "target_diff.json").write_text(json.dumps(diff_summary, indent=2, sort_keys=True) + "\n")
    config_json = dataclasses.asdict(config)
    if training_args is not None:
        config_json["training"] = training_args
    (path / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n")
    return diff_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default="/tmp/openpi-rlt-replay")
    parser.add_argument("--output-dir", default="/tmp/openpi-rlt-actor")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--bc-warmup-steps", type=int, default=None)
    parser.add_argument("--rl-loss-coef", type=float, default=None)
    parser.add_argument("--bc-coef", type=float, default=None)
    parser.add_argument("--action-smooth-coef", type=float, default=None)
    parser.add_argument("--action-accel-coef", type=float, default=None)
    parser.add_argument("--critic-loss-coef", type=float, default=None)
    parser.add_argument("--actor-update-period", type=int, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--actor-hidden-layers", type=int, default=None)
    parser.add_argument("--critic-hidden-layers", type=int, default=None)
    parser.add_argument("--target-policy-noise", type=float, default=None)
    parser.add_argument("--train-episode-label-filter", choices=("all", "success", "failure"), default="all")
    parser.add_argument("--balance-train-episode-labels", action="store_true")
    parser.add_argument("--oversample-success-train-episodes", action="store_true")
    parser.add_argument("--repeat-success-train-episodes", type=int, default=1)
    parser.add_argument("--near-terminal-window", type=int, default=0)
    parser.add_argument("--near-terminal-sample-weight", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--init-actor-critic-checkpoint", default=None)
    parser.add_argument("--q-curve-every", type=int, default=500)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tag", action="append", default=[])
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    dataset = replay.ReplayDataset(replay_dir)
    train_indices, val_indices = _split_indices_by_episode(dataset, seed=args.seed)
    initial_train_label_counts = _episode_label_counts(dataset, train_indices)
    val_label_counts = _episode_label_counts(dataset, val_indices)
    success_oversample_stats = {"enabled": False}
    success_repeat_stats = {"enabled": False, "repeat": int(args.repeat_success_train_episodes)}
    train_indices = _filter_indices_by_episode_label(dataset, train_indices, args.train_episode_label_filter)
    sampling_modes = [
        bool(args.balance_train_episode_labels),
        bool(args.oversample_success_train_episodes),
        int(args.repeat_success_train_episodes) > 1,
    ]
    if sum(sampling_modes) > 1:
        raise ValueError(
            "--balance-train-episode-labels, --oversample-success-train-episodes, "
            "and --repeat-success-train-episodes > 1 are mutually exclusive"
        )
    if args.train_episode_label_filter != "all" and any(sampling_modes):
        raise ValueError("--train-episode-label-filter cannot be combined with train episode balancing/oversampling")
    if args.balance_train_episode_labels:
        train_indices = _balance_indices_by_episode_label(dataset, train_indices, seed=args.seed + 17)
    if args.oversample_success_train_episodes:
        train_indices, success_oversample_stats = _oversample_success_episodes_to_match_failures(
            dataset,
            train_indices,
            seed=args.seed + 23,
        )
    if args.repeat_success_train_episodes > 1:
        train_indices, success_repeat_stats = _repeat_success_episodes(
            dataset,
            train_indices,
            repeat=args.repeat_success_train_episodes,
            seed=args.seed + 29,
        )
    train_label_counts = _episode_label_counts(dataset, train_indices)
    train_sample_weights = _near_terminal_weights(
        dataset,
        train_indices,
        window=args.near_terminal_window,
        sample_weight=args.near_terminal_sample_weight,
    )
    near_terminal_stats = {
        "enabled": train_sample_weights is not None,
        "window": int(args.near_terminal_window),
        "sample_weight": float(args.near_terminal_sample_weight),
        "weighted_samples": int(np.count_nonzero(train_sample_weights > 1.0)) if train_sample_weights is not None else 0,
    }
    if train_indices.size == 0:
        raise ValueError(f"Empty train split: dataset_size={len(dataset)}")
    dataloader = _make_dataloader(
        dataset,
        batch_size=args.batch_size,
        indices=train_indices,
        shuffle=not args.no_shuffle,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_weights=train_sample_weights,
    )
    val_loader = None
    if val_indices.size > 0:
        val_loader = _make_dataloader(
            dataset,
            indices=val_indices,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            seed=args.seed + 1,
        )
    train_eval_loader = _make_dataloader(
        dataset,
        indices=train_indices,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        seed=args.seed + 2,
    )
    split_episode_id = np.asarray(getattr(dataset, "split_episode_id", dataset.data["episode_id"]))
    train_episodes = np.unique(split_episode_id[train_indices])
    val_episodes = np.unique(split_episode_id[val_indices]) if val_indices.size else np.asarray([], dtype=np.int32)
    curve_indices = _find_curve_episodes(dataset, val_indices if val_indices.size else train_indices)
    print(
        f"dataset_size={len(dataset)} train_size={train_indices.size} val_size={val_indices.size} "
        f"train_episodes={train_episodes.tolist()} val_episodes={val_episodes.tolist()}"
    )
    print(
        "episode_label_counts="
        + json.dumps(
            {
                "initial_train": initial_train_label_counts,
                "train_episode_label_filter": args.train_episode_label_filter,
                "train": train_label_counts,
                "val": val_label_counts,
                "balance_train_episode_labels": bool(args.balance_train_episode_labels),
                "oversample_success_train_episodes": success_oversample_stats,
                "repeat_success_train_episodes": success_repeat_stats,
                "near_terminal_sampling": near_terminal_stats,
            },
            sort_keys=True,
        )
    )
    print("rlt_token_layernorm=model")
    if len(dataloader) == 0:
        raise ValueError(
            f"DataLoader produced zero batches: dataset_size={len(dataset)} "
            f"batch_size={args.batch_size} drop_last={args.drop_last}"
        )
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None
    config_overrides = {}
    if args.bc_coef is not None:
        config_overrides["bc_coef"] = args.bc_coef
    if args.action_smooth_coef is not None:
        config_overrides["action_smooth_coef"] = args.action_smooth_coef
    if args.action_accel_coef is not None:
        config_overrides["action_accel_coef"] = args.action_accel_coef
    if args.critic_loss_coef is not None:
        config_overrides["critic_loss_coef"] = args.critic_loss_coef
    if args.bc_warmup_steps is not None:
        config_overrides["bc_warmup_steps"] = args.bc_warmup_steps
    if args.rl_loss_coef is not None:
        config_overrides["rl_loss_coef"] = args.rl_loss_coef
    if args.actor_update_period is not None:
        config_overrides["actor_update_period"] = args.actor_update_period
    if args.actor_lr is not None:
        config_overrides["actor_lr"] = args.actor_lr
    if args.critic_lr is not None:
        config_overrides["critic_lr"] = args.critic_lr
    if args.hidden_dim is not None:
        config_overrides["hidden_dim"] = args.hidden_dim
    if args.actor_hidden_layers is not None:
        config_overrides["actor_hidden_layers"] = args.actor_hidden_layers
    if args.critic_hidden_layers is not None:
        config_overrides["critic_hidden_layers"] = args.critic_hidden_layers
    if args.target_policy_noise is not None:
        config_overrides["target_policy_noise"] = args.target_policy_noise
    config = dataclasses.replace(actor_critic.RLTActorCriticConfig(), **config_overrides)
    training_args = _training_args_summary(args, config_overrides)
    wandb_run = _maybe_init_wandb(args, config, dataset, train_size=int(train_indices.size), val_size=int(val_indices.size))
    rng = jax.random.key(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = actor_critic.init_actor_critic_params(init_rng, config)
    target_params = jax.tree.map(lambda x: x.copy(), params)
    restored_optimizer_state = None
    if args.init_actor_critic_checkpoint:
        init_checkpoint = Path(args.init_actor_critic_checkpoint)
        with (init_checkpoint / "params.pkl").open("rb") as f:
            params = pickle.load(f)
        target_path = init_checkpoint / "target_params.pkl"
        if target_path.exists():
            with target_path.open("rb") as f:
                target_params = pickle.load(f)
        else:
            target_params = jax.tree.map(lambda x: x.copy(), params)
        optimizer_path = init_checkpoint / "optimizer_state.pkl"
        if optimizer_path.exists():
            with optimizer_path.open("rb") as f:
                restored_optimizer_state = pickle.load(f)
            print(f"restored_optimizer_state_from={optimizer_path}")
        else:
            print(f"optimizer_state_missing={optimizer_path}")
        print(f"initialized_actor_critic_from={init_checkpoint}")
    actor_tx = optax.adam(config.actor_lr)
    critic_tx = optax.adam(config.critic_lr)
    critic_opt = critic_tx.init({"critic1": params["critic1"], "critic2": params["critic2"]})
    if restored_optimizer_state is not None:
        critic_opt = restored_optimizer_state["critic"]

    @jax.jit
    def critic_step(params, target_params, opt_state, batch, rng):
        def loss_for_critics(critic_params):
            full_params = dict(params)
            full_params["critic1"] = critic_params["critic1"]
            full_params["critic2"] = critic_params["critic2"]
            return actor_critic.critic_loss(full_params, target_params, batch, config, rng)

        critic_params = {"critic1": params["critic1"], "critic2": params["critic2"]}
        (loss, metrics), grads = jax.value_and_grad(loss_for_critics, has_aux=True)(critic_params)
        updates, opt_state = critic_tx.update(grads, opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, updates)
        params = dict(params)
        params["critic1"] = critic_params["critic1"]
        params["critic2"] = critic_params["critic2"]
        return params, opt_state, loss, metrics

    actor_opt = actor_tx.init(params["actor"])
    if restored_optimizer_state is not None:
        actor_opt = restored_optimizer_state["actor"]

    @jax.jit
    def actor_step(params, target_params, opt_state, batch, rng, step_idx):
        def loss_for_actor(actor_params):
            actor_only_params = dict(params)
            actor_only_params["actor"] = actor_params
            return actor_critic.actor_loss(actor_only_params, batch, config, rng, step_idx)

        (loss, metrics), grads = jax.value_and_grad(loss_for_actor, has_aux=True)(params["actor"])
        updates, opt_state = actor_tx.update(grads, opt_state, params["actor"])
        new_actor = optax.apply_updates(params["actor"], updates)
        params = dict(params)
        params["actor"] = new_actor
        return params, target_params, opt_state, loss, metrics

    @jax.jit
    def eval_step(params, target_params, batch, rng):
        critic_loss, critic_metrics = actor_critic.critic_loss(
            params,
            target_params,
            batch,
            config,
            rng,
        )
        action = actor_critic.actor_apply(
            params["actor"],
            batch.rlt_token,
            batch.state,
            batch.reference_action_chunk,
            config,
        )
        target = batch.executed_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]
        mask = batch.executed_action_mask[:, : config.rlt_chunk_horizon].astype(jnp.float32)[..., None]
        denom = jnp.maximum(jnp.sum(mask) * config.action_dim, 1.0)
        mse = jnp.sum(jnp.square(action - target) * mask) / denom
        mae = jnp.sum(jnp.abs(action - target) * mask) / denom
        actor_q1 = actor_critic.critic_apply(params["critic1"], batch.rlt_token, batch.state, action, config)
        actor_q2 = actor_critic.critic_apply(params["critic2"], batch.rlt_token, batch.state, action, config)
        executed_q1 = actor_critic.critic_apply(params["critic1"], batch.rlt_token, batch.state, target, config)
        executed_q2 = actor_critic.critic_apply(params["critic2"], batch.rlt_token, batch.state, target, config)
        return {
            "critic_loss": critic_loss,
            "critic_loss_unweighted": critic_metrics["critic_loss_unweighted"],
            "q1": critic_metrics["q1"],
            "target_q": critic_metrics["target_q"],
            "actor_mse": mse,
            "actor_mae": mae,
            "actor_q": jnp.mean(jnp.minimum(actor_q1, actor_q2)),
            "executed_q": jnp.mean(jnp.minimum(executed_q1, executed_q2)),
            "reward_mean": jnp.mean(batch.reward),
            "done_mean": jnp.mean(batch.done.astype(jnp.float32)),
        }

    @jax.jit
    def q_curve_step(params, batch):
        action = actor_critic.actor_apply(
            params["actor"],
            batch.rlt_token,
            batch.state,
            batch.reference_action_chunk,
            config,
        )
        executed_target = batch.executed_action_chunk[:, : config.rlt_chunk_horizon, : config.action_dim]
        executed_q1 = actor_critic.critic_apply(params["critic1"], batch.rlt_token, batch.state, executed_target, config)
        executed_q2 = actor_critic.critic_apply(params["critic2"], batch.rlt_token, batch.state, executed_target, config)
        actor_q1 = actor_critic.critic_apply(params["critic1"], batch.rlt_token, batch.state, action, config)
        actor_q2 = actor_critic.critic_apply(params["critic2"], batch.rlt_token, batch.state, action, config)
        actor_mae = jnp.mean(jnp.abs(action - executed_target), axis=(1, 2))
        return {
            "executed_q": jnp.minimum(executed_q1, executed_q2),
            "actor_q": jnp.minimum(actor_q1, actor_q2),
            "actor_mae": actor_mae,
        }

    def evaluate(prefix: str, loader, step_idx: int) -> dict[str, float]:
        if loader is None:
            return {}
        totals: dict[str, float] = {}
        count = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.val_max_batches:
                break
            batch = replay.batch_to_jax(batch)
            eval_rng = jax.random.fold_in(rng, step_idx * 1000 + batch_idx)
            metrics = eval_step(params, target_params, batch, eval_rng)
            batch_size = int(batch.rlt_token.shape[0])
            for key, value in metrics.items():
                totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + _to_float(value) * batch_size
            count += batch_size
        if count == 0:
            return {}
        return {key: value / count for key, value in totals.items()}

    def log_q_curves(step_idx: int) -> dict[str, float]:
        if args.q_curve_every <= 0 or not curve_indices:
            return {}
        curve_dir = Path(args.output_dir) / "q_curves" / f"step_{step_idx:06d}"
        curve_dir.mkdir(parents=True, exist_ok=True)
        metrics = {}
        wandb_images = {}
        for label, indices in curve_indices.items():
            episode_id = int(split_episode_id[indices[0]])
            batch_np = _batch_from_indices(dataset, indices)
            batch = replay.batch_to_jax(batch_np)
            scores = jax.tree.map(np.asarray, q_curve_step(params, batch))
            step_index = np.asarray(dataset.data["step_index"])[indices]
            reward = np.asarray(dataset.data["reward"])[indices]
            done = np.asarray(dataset.data["done"])[indices]
            csv_path = curve_dir / f"val_{label}_episode_{episode_id:03d}_q_curve.csv"
            png_path = curve_dir / f"val_{label}_episode_{episode_id:03d}_q_curve.png"
            _write_curve_csv(csv_path, step_index, scores, reward, done)
            _plot_q_curve(
                png_path,
                title=f"step {step_idx}: validation {label} episode {episode_id}",
                step_index=step_index,
                scores=scores,
            )
            metrics[f"q_curve/{label}/executed_q_mean"] = float(np.mean(scores["executed_q"]))
            metrics[f"q_curve/{label}/actor_q_mean"] = float(np.mean(scores["actor_q"]))
            metrics[f"q_curve/{label}/actor_mae_mean"] = float(np.mean(scores["actor_mae"]))
            metrics[f"q_curve/{label}/terminal_reward"] = float(reward[-1]) if reward.size else 0.0
            if wandb_run is not None:
                import wandb

                wandb_images[f"q_curve/{label}/plot"] = wandb.Image(str(png_path))
        if wandb_run is not None and wandb_images:
            wandb_run.log(wandb_images, step=step_idx)
        return metrics

    data_iter = iter(dataloader)
    epoch = 0
    batch_in_epoch = 0
    for step_idx in range(args.max_steps):
        global_step = args.start_step + step_idx
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            batch_in_epoch = 0
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = replay.batch_to_jax(batch)
        current_batch_size = int(batch.rlt_token.shape[0])
        loader_info = {"epoch": epoch, "batch_in_epoch": batch_in_epoch, "batch_size": current_batch_size}
        batch_in_epoch += 1
        rng, update_rng = jax.random.split(rng)
        params, critic_opt, critic_loss, critic_metrics = critic_step(
            params,
            target_params,
            critic_opt,
            batch,
            update_rng,
        )
        if global_step % config.actor_update_period == 0:
            rng, actor_rng = jax.random.split(rng)
            params, target_params, actor_opt, actor_loss, actor_metrics = actor_step(
                params,
                target_params,
                actor_opt,
                batch,
                actor_rng,
                jnp.asarray(global_step, dtype=jnp.int32),
            )
            target_params = actor_critic.soft_update(params, target_params, config.tau)
            target_params = dict(target_params)
            target_params["actor"] = params["actor"]
            print(
                f"step={global_step} local_step={step_idx} epoch={loader_info['epoch']} "
                f"batch_in_epoch={loader_info['batch_in_epoch']} "
                f"batch_size={loader_info['batch_size']} "
                f"critic_loss={float(critic_loss):.6f} actor_loss={float(actor_loss):.6f} "
                f"critic_loss_unweighted={float(critic_metrics['critic_loss_unweighted']):.6f} "
                f"rl_loss={float(actor_metrics['actor_rl_loss']):.6f} "
                f"rl_loss_w={float(actor_metrics['actor_rl_loss_weighted']):.6f} "
                f"bc_loss={float(actor_metrics['bc_loss']):.6f} "
                f"bc_loss_w={float(actor_metrics['bc_loss_weighted']):.6f} "
                f"smooth_loss={float(actor_metrics['smooth_loss']):.6f} "
                f"smooth_loss_w={float(actor_metrics['smooth_loss_weighted']):.6f} "
                f"accel_loss={float(actor_metrics['accel_loss']):.6f} "
                f"accel_loss_w={float(actor_metrics['accel_loss_weighted']):.6f}"
            )
            log_data = {
                "train/critic_loss": _to_float(critic_loss),
                "train/critic_loss_unweighted": _to_float(critic_metrics["critic_loss_unweighted"]),
                "train/actor_loss": _to_float(actor_loss),
                "train/actor_rl_loss": _to_float(actor_metrics["actor_rl_loss"]),
                "train/actor_rl_active": _to_float(actor_metrics["actor_rl_active"]),
                "train/actor_rl_loss_weighted": _to_float(actor_metrics["actor_rl_loss_weighted"]),
                "train/rl_loss_coef": config.rl_loss_coef,
                "train/bc_loss": _to_float(actor_metrics["bc_loss"]),
                "train/bc_loss_weighted": _to_float(actor_metrics["bc_loss_weighted"]),
                "train/smooth_loss": _to_float(actor_metrics["smooth_loss"]),
                "train/smooth_loss_weighted": _to_float(actor_metrics["smooth_loss_weighted"]),
                "train/action_smooth_coef": config.action_smooth_coef,
                "train/accel_loss": _to_float(actor_metrics["accel_loss"]),
                "train/accel_loss_weighted": _to_float(actor_metrics["accel_loss_weighted"]),
                "train/action_accel_coef": config.action_accel_coef,
                "train/actor_q": _to_float(actor_metrics["actor_q"]),
                "train/actor_q1": _to_float(actor_metrics["actor_q1"]),
                "train/actor_q2": _to_float(actor_metrics["actor_q2"]),
                "train/bc_valid_steps": _to_float(actor_metrics["bc_valid_steps"]),
                "train/smooth_valid_steps": _to_float(actor_metrics["smooth_valid_steps"]),
                "train/accel_valid_steps": _to_float(actor_metrics["accel_valid_steps"]),
                "train/q1": _to_float(critic_metrics["q1"]),
                "train/target_q": _to_float(critic_metrics["target_q"]),
                "train/actor_lr": config.actor_lr,
                "train/critic_lr": config.critic_lr,
                "data/epoch": loader_info["epoch"],
                "data/batch_in_epoch": loader_info["batch_in_epoch"],
                "data/batch_size": loader_info["batch_size"],
            }
        else:
            print(
                f"step={global_step} local_step={step_idx} epoch={loader_info['epoch']} "
                f"batch_in_epoch={loader_info['batch_in_epoch']} "
                f"batch_size={loader_info['batch_size']} "
                f"critic_loss={float(critic_loss):.6f} "
                f"critic_loss_unweighted={float(critic_metrics['critic_loss_unweighted']):.6f} "
                f"q1={float(critic_metrics['q1']):.6f}"
            )
            log_data = {
                "train/critic_loss": _to_float(critic_loss),
                "train/critic_loss_unweighted": _to_float(critic_metrics["critic_loss_unweighted"]),
                "train/q1": _to_float(critic_metrics["q1"]),
                "train/target_q": _to_float(critic_metrics["target_q"]),
                "train/critic_lr": config.critic_lr,
                "data/epoch": loader_info["epoch"],
                "data/batch_in_epoch": loader_info["batch_in_epoch"],
                "data/batch_size": loader_info["batch_size"],
            }
        if args.val_every > 0 and (global_step % args.val_every == 0 or step_idx == args.max_steps - 1):
            log_data.update(evaluate("eval/train", train_eval_loader, global_step))
            log_data.update(evaluate("eval/val", val_loader, global_step))
            if "eval/val/actor_mae" in log_data:
                print(
                    f"eval step={global_step} "
                    f"train_actor_mae={log_data.get('eval/train/actor_mae', float('nan')):.6f} "
                    f"val_actor_mae={log_data['eval/val/actor_mae']:.6f} "
                    f"val_critic_loss={log_data.get('eval/val/critic_loss', float('nan')):.6f} "
                    f"val_executed_q={log_data.get('eval/val/executed_q', float('nan')):.6f} "
                    f"val_actor_q={log_data.get('eval/val/actor_q', float('nan')):.6f} "
                    f"val_target_q={log_data.get('eval/val/target_q', float('nan')):.6f} "
                    f"val_reward_mean={log_data.get('eval/val/reward_mean', float('nan')):.6f} "
                    f"val_done_mean={log_data.get('eval/val/done_mean', float('nan')):.6f}"
                )
        if args.q_curve_every > 0 and (global_step % args.q_curve_every == 0 or step_idx == args.max_steps - 1):
            log_data.update(log_q_curves(global_step))
        if wandb_run is not None:
            wandb_run.log(log_data, step=global_step)
        completed_step = global_step + 1
        if args.checkpoint_every > 0 and completed_step % args.checkpoint_every == 0:
            checkpoint_dir = Path(args.output_dir) / f"step_{completed_step:06d}" / "rlt_actor_critic"
            diff_summary = _save(
                checkpoint_dir,
                params,
                target_params,
                config,
                actor_opt=actor_opt,
                critic_opt=critic_opt,
                training_args=training_args,
            )
            (checkpoint_dir / "split.json").write_text(
                json.dumps(
                    _split_summary(
                        replay_dir=replay_dir,
                        dataset=dataset,
                        train_indices=train_indices,
                        val_indices=val_indices,
                        train_episodes=train_episodes,
                        val_episodes=val_episodes,
                        initial_train_label_counts=initial_train_label_counts,
                        train_label_counts=train_label_counts,
                        val_label_counts=val_label_counts,
                        args=args,
                        success_oversample_stats=success_oversample_stats,
                        success_repeat_stats=success_repeat_stats,
                        near_terminal_stats=near_terminal_stats,
                        step=completed_step,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"saved_intermediate_actor_critic={checkpoint_dir}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "checkpoint/step": completed_step,
                        **{f"checkpoint/{key}": value for key, value in diff_summary.items()},
                    },
                    step=global_step,
                )

    out = Path(args.output_dir) / "rlt_actor_critic"
    diff_summary = _save(
        out,
        params,
        target_params,
        config,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        training_args=training_args,
    )
    split_summary = _split_summary(
        replay_dir=replay_dir,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        initial_train_label_counts=initial_train_label_counts,
        train_label_counts=train_label_counts,
        val_label_counts=val_label_counts,
        args=args,
        success_oversample_stats=success_oversample_stats,
        success_repeat_stats=success_repeat_stats,
        near_terminal_stats=near_terminal_stats,
    )
    (out / "split.json").write_text(json.dumps(split_summary, indent=2, sort_keys=True) + "\n")
    print(f"saved_actor_critic={out}")
    print("target_diff=" + json.dumps(diff_summary, sort_keys=True))
    if wandb_run is not None:
        wandb_run.summary["saved_actor_critic"] = str(out)
        for key, value in diff_summary.items():
            wandb_run.summary[key] = value
        wandb_run.finish()


if __name__ == "__main__":
    main()

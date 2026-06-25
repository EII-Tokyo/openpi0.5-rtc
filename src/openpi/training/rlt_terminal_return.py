from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from openpi.training import rlt_eval


@dataclasses.dataclass(frozen=True)
class TerminalReturnDataset:
    z_rl: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray
    sources: np.ndarray
    transition_indices: np.ndarray
    num_transitions: np.ndarray
    progress: np.ndarray
    shard_paths: np.ndarray
    terminal_rewards: np.ndarray
    rows: tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class DatasetSplit:
    name: str
    train_indices: np.ndarray
    holdout_indices: np.ndarray
    split_type: str
    holdout_source: str | None = None


def load_terminal_return_dataset(
    manifest_path: pathlib.Path | str,
    *,
    gamma: float,
    critical_ratio: float | None = None,
    failure_target: float = 0.0,
) -> TerminalReturnDataset:
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if critical_ratio is not None and not 0.0 < critical_ratio <= 1.0:
        raise ValueError("critical_ratio must be in (0, 1]")

    z_parts: list[np.ndarray] = []
    proprio_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    transition_indices: list[np.ndarray] = []
    num_transitions: list[np.ndarray] = []
    progress: list[np.ndarray] = []
    shard_paths: list[np.ndarray] = []
    terminal_rewards: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for row in _read_jsonl(pathlib.Path(manifest_path)):
        shard_path = pathlib.Path(row["shard_path"]).expanduser().resolve()
        source = str(row.get("batch") or _source_from_path(shard_path))
        with np.load(shard_path, allow_pickle=False) as data:
            z_rl = np.asarray(data["z_rl"], dtype=np.float32)
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            reference_action = np.asarray(data["reference_action"], dtype=np.float32)
            reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
            done = np.asarray(data["done"]).astype(np.bool_)

        n = int(z_rl.shape[0])
        if n <= 0:
            continue
        start = int(n * (1.0 - critical_ratio)) if critical_ratio is not None else 0
        start = min(max(start, 0), max(n - 1, 0))
        selected = np.arange(start, n, dtype=np.int64)
        terminal_reward = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
        label = int(terminal_reward > 0.0)
        episode_id = shard_path.stem
        denom = max(n - 1, 1)
        target = _terminal_targets_for_indices(
            selected,
            num_transitions=n,
            terminal_reward=terminal_reward,
            gamma=gamma,
            failure_target=failure_target,
        )

        z_parts.append(z_rl[selected])
        proprio_parts.append(proprio[selected])
        action_parts.append(action[selected])
        reference_parts.append(reference_action[selected])
        labels.append(np.full((len(selected),), label, dtype=np.int64))
        targets.append(target)
        episode_ids.append(np.asarray([episode_id] * len(selected), dtype=object))
        sources.append(np.asarray([source] * len(selected), dtype=object))
        transition_indices.append(selected.astype(np.int64))
        num_transitions.append(np.full((len(selected),), n, dtype=np.int64))
        progress.append(selected.astype(np.float32) / float(denom))
        shard_paths.append(np.asarray([str(shard_path)] * len(selected), dtype=object))
        terminal_rewards.append(np.full((len(selected),), terminal_reward, dtype=np.float32))
        rows.extend(
            {
                "episode_id": episode_id,
                "source": source,
                "shard_path": str(shard_path),
                "label_success": label,
                "target": float(value),
                "terminal_reward": terminal_reward,
                "transition_index": int(index),
                "num_transitions": n,
                "progress": float(index / denom),
            }
            for index, value in zip(selected, target, strict=True)
        )

    if not z_parts:
        raise ValueError(f"No terminal-return transitions loaded from {manifest_path}")
    return TerminalReturnDataset(
        z_rl=np.concatenate(z_parts, axis=0),
        proprio=np.concatenate(proprio_parts, axis=0),
        action=np.concatenate(action_parts, axis=0),
        reference_action=np.concatenate(reference_parts, axis=0),
        labels=np.concatenate(labels, axis=0),
        targets=np.concatenate(targets, axis=0),
        episode_ids=np.concatenate(episode_ids, axis=0),
        sources=np.concatenate(sources, axis=0),
        transition_indices=np.concatenate(transition_indices, axis=0),
        num_transitions=np.concatenate(num_transitions, axis=0),
        progress=np.concatenate(progress, axis=0),
        shard_paths=np.concatenate(shard_paths, axis=0),
        terminal_rewards=np.concatenate(terminal_rewards, axis=0),
        rows=tuple(rows),
    )


def episode_random_split(dataset: TerminalReturnDataset, *, holdout_ratio: float, seed: int) -> DatasetSplit:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    episodes = np.asarray(sorted(set(dataset.episode_ids.tolist())), dtype=object)
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for holdout split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    holdout_count = max(1, round(len(episodes) * holdout_ratio))
    holdout_count = min(holdout_count, len(episodes) - 1)
    holdout_eps = set(episodes[order[:holdout_count]].tolist())
    holdout_mask = np.asarray([episode in holdout_eps for episode in dataset.episode_ids], dtype=np.bool_)
    return DatasetSplit(
        name="episode_random",
        train_indices=np.flatnonzero(~holdout_mask),
        holdout_indices=np.flatnonzero(holdout_mask),
        split_type="episode_random",
    )


def leave_one_source_out_splits(dataset: TerminalReturnDataset) -> list[DatasetSplit]:
    splits: list[DatasetSplit] = []
    for source in sorted(set(dataset.sources.tolist())):
        holdout_mask = dataset.sources == source
        if np.all(holdout_mask) or not np.any(holdout_mask):
            continue
        splits.append(
            DatasetSplit(
                name=f"holdout_{source}",
                train_indices=np.flatnonzero(~holdout_mask),
                holdout_indices=np.flatnonzero(holdout_mask),
                split_type="leave_one_source_out",
                holdout_source=str(source),
            )
        )
    return splits


def intra_source_episode_splits(
    dataset: TerminalReturnDataset,
    *,
    holdout_ratio: float,
    seed: int,
) -> list[DatasetSplit]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    splits: list[DatasetSplit] = []
    for source in sorted(set(dataset.sources.tolist())):
        source_mask = dataset.sources == source
        source_episodes = np.asarray(sorted(set(dataset.episode_ids[source_mask].tolist())), dtype=object)
        if len(source_episodes) < 2:
            continue
        holdout_episodes: set[str] = set()
        for label in (0, 1):
            label_episodes = np.asarray(
                [episode for episode in source_episodes if _episode_label(dataset, str(episode)) == label],
                dtype=object,
            )
            if len(label_episodes) < 2:
                continue
            order = rng.permutation(len(label_episodes))
            holdout_count = max(1, round(len(label_episodes) * holdout_ratio))
            holdout_count = min(holdout_count, len(label_episodes) - 1)
            holdout_episodes.update(str(episode) for episode in label_episodes[order[:holdout_count]])
        if not holdout_episodes:
            order = rng.permutation(len(source_episodes))
            holdout_count = max(1, round(len(source_episodes) * holdout_ratio))
            holdout_count = min(holdout_count, len(source_episodes) - 1)
            holdout_episodes.update(str(episode) for episode in source_episodes[order[:holdout_count]])
        holdout_mask = source_mask & np.asarray([str(episode) in holdout_episodes for episode in dataset.episode_ids])
        train_mask = source_mask & ~holdout_mask
        if not np.any(train_mask) or not np.any(holdout_mask):
            continue
        splits.append(
            DatasetSplit(
                name=f"intra_{source}",
                train_indices=np.flatnonzero(train_mask),
                holdout_indices=np.flatnonzero(holdout_mask),
                split_type="intra_source_episode",
                holdout_source=str(source),
            )
        )
    return splits


def score_metrics(labels: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    success = scores[labels == 1]
    failure = scores[labels == 0]
    return {
        "auc": _nan_if_none(rlt_eval.auc_rank(labels, scores)),
        "q_success_mean": _nanmean(success),
        "q_failure_mean": _nanmean(failure),
        "q_gap": _nanmean(success) - _nanmean(failure) if success.size and failure.size else float("nan"),
        "target_success_mean": _nanmean(targets[labels == 1]),
        "target_failure_mean": _nanmean(targets[labels == 0]),
        "mse": float(np.mean(np.square(scores - targets))) if scores.size else float("nan"),
        "num_success_transitions": int(np.sum(labels == 1)),
        "num_failure_transitions": int(np.sum(labels == 0)),
    }


def dataset_stats(dataset: TerminalReturnDataset) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "num_transitions": int(len(dataset.labels)),
        "num_episodes": int(len(set(dataset.episode_ids.tolist()))),
        "num_success_transitions": int(np.sum(dataset.labels == 1)),
        "num_failure_transitions": int(np.sum(dataset.labels == 0)),
        "num_success_episodes": int(
            sum(_episode_label(dataset, episode) == 1 for episode in set(dataset.episode_ids.tolist()))
        ),
        "num_failure_episodes": int(
            sum(_episode_label(dataset, episode) == 0 for episode in set(dataset.episode_ids.tolist()))
        ),
        "by_source": {},
    }
    for source in sorted(set(dataset.sources.tolist())):
        mask = dataset.sources == source
        source_eps = set(dataset.episode_ids[mask].tolist())
        success_eps = sum(_episode_label(dataset, episode) == 1 for episode in source_eps)
        failure_eps = len(source_eps) - success_eps
        stats["by_source"][str(source)] = {
            "num_episodes": int(len(source_eps)),
            "num_success_episodes": int(success_eps),
            "num_failure_episodes": int(failure_eps),
            "num_success_transitions": int(np.sum(dataset.labels[mask] == 1)),
            "num_failure_transitions": int(np.sum(dataset.labels[mask] == 0)),
            "success_rate": float(success_eps / max(len(source_eps), 1)),
        }
    return stats


def _terminal_targets_for_indices(
    indices: np.ndarray,
    *,
    num_transitions: int,
    terminal_reward: float,
    gamma: float,
    failure_target: float,
) -> np.ndarray:
    if terminal_reward > 0.0:
        distance = (num_transitions - 1) - indices.astype(np.float32)
        return (float(terminal_reward) * (gamma**distance)).astype(np.float32)
    return np.full((len(indices),), float(failure_target), dtype=np.float32)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _source_from_path(path: pathlib.Path) -> str:
    for part in path.parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "unknown"


def _episode_label(dataset: TerminalReturnDataset, episode_id: str) -> int:
    mask = dataset.episode_ids == episode_id
    labels = set(dataset.labels[mask].tolist())
    if len(labels) != 1:
        raise ValueError(f"Episode {episode_id} has inconsistent labels: {labels}")
    return int(next(iter(labels)))


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _nanmean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")

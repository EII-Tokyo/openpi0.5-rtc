from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import re
from typing import Any

import numpy as np

from openpi.training import rlt_eval


@dataclasses.dataclass(frozen=True)
class TransitionDataset:
    z_rl: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    labels: np.ndarray
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


def load_transition_dataset(manifest_path: pathlib.Path | str, *, critical_ratio: float | None = None) -> TransitionDataset:
    if critical_ratio is not None and not 0.0 < critical_ratio <= 1.0:
        raise ValueError("critical_ratio must be in (0, 1]")
    z_parts: list[np.ndarray] = []
    proprio_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    next_z_parts: list[np.ndarray] = []
    next_proprio_parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    transition_indices: list[np.ndarray] = []
    num_transitions: list[np.ndarray] = []
    progress: list[np.ndarray] = []
    shard_paths_out: list[np.ndarray] = []
    terminal_rewards: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for row in _read_jsonl(pathlib.Path(manifest_path)):
        shard_path = pathlib.Path(row["shard_path"]).expanduser().resolve()
        source = str(row.get("batch") or _source_from_path(shard_path))
        with np.load(shard_path, allow_pickle=False) as data:
            z_rl = np.asarray(data["z_rl"], dtype=np.float32)
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            next_z_rl = np.asarray(data["next_z_rl"], dtype=np.float32)
            next_proprio = np.asarray(data["next_proprio"], dtype=np.float32)
            reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
            done = np.asarray(data["done"]).astype(np.bool_)
        n = int(z_rl.shape[0])
        start = int(n * (1.0 - critical_ratio)) if critical_ratio is not None else 0
        start = min(max(start, 0), max(n - 1, 0))
        selected = np.arange(start, n, dtype=np.int64)
        terminal_reward = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
        label = int(terminal_reward > 0.0)
        episode_id = shard_path.stem

        z_parts.append(z_rl[selected])
        proprio_parts.append(proprio[selected])
        action_parts.append(action[selected])
        next_z_parts.append(next_z_rl[selected])
        next_proprio_parts.append(next_proprio[selected])
        labels.append(np.full((len(selected),), label, dtype=np.int64))
        episode_ids.append(np.asarray([episode_id] * len(selected), dtype=object))
        sources.append(np.asarray([source] * len(selected), dtype=object))
        transition_indices.append(selected.astype(np.int64))
        num_transitions.append(np.full((len(selected),), n, dtype=np.int64))
        denom = max(n - 1, 1)
        progress.append(selected.astype(np.float32) / float(denom))
        shard_paths_out.append(np.asarray([str(shard_path)] * len(selected), dtype=object))
        terminal_rewards.append(np.full((len(selected),), terminal_reward, dtype=np.float32))
        rows.extend(
            {
                "episode_id": episode_id,
                "shard_path": str(shard_path),
                "source": source,
                "label_success": label,
                "terminal_reward": terminal_reward,
                "transition_idx": int(index),
                "num_transitions": n,
                "progress": float(index / denom),
            }
            for index in selected
        )

    if not z_parts:
        raise ValueError(f"No discriminator transitions loaded from {manifest_path}")
    return TransitionDataset(
        z_rl=np.concatenate(z_parts, axis=0),
        proprio=np.concatenate(proprio_parts, axis=0),
        action=np.concatenate(action_parts, axis=0),
        next_z_rl=np.concatenate(next_z_parts, axis=0),
        next_proprio=np.concatenate(next_proprio_parts, axis=0),
        labels=np.concatenate(labels, axis=0),
        episode_ids=np.concatenate(episode_ids, axis=0),
        sources=np.concatenate(sources, axis=0),
        transition_indices=np.concatenate(transition_indices, axis=0),
        num_transitions=np.concatenate(num_transitions, axis=0),
        progress=np.concatenate(progress, axis=0),
        shard_paths=np.concatenate(shard_paths_out, axis=0),
        terminal_rewards=np.concatenate(terminal_rewards, axis=0),
        rows=tuple(rows),
    )


def build_features(
    dataset: TransitionDataset,
    variant: str,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    state = np.concatenate([dataset.z_rl, dataset.proprio], axis=-1)
    if variant == "state_only":
        return state.astype(np.float32)
    next_state = np.concatenate([dataset.next_z_rl, dataset.next_proprio], axis=-1)
    delta_state = np.concatenate([dataset.next_z_rl - dataset.z_rl, dataset.next_proprio - dataset.proprio], axis=-1)
    if variant == "state_next_only":
        return np.concatenate([state, next_state, delta_state], axis=-1).astype(np.float32)
    if variant not in {
        "state_action",
        "shuffled_action",
        "state_action_next",
        "state_action_next_delta",
        "shuffled_action_next_delta",
    }:
        raise ValueError(f"Unsupported discriminator feature variant: {variant}")
    action = dataset.action
    if variant in {"shuffled_action", "shuffled_action_next_delta"}:
        rng = np.random.default_rng(0) if rng is None else rng
        order = rng.permutation(action.shape[0])
        action = action[order]
    flat_action = action.reshape(action.shape[0], -1)
    if variant in {"state_action", "shuffled_action"}:
        return np.concatenate([state, flat_action], axis=-1).astype(np.float32)
    if variant == "state_action_next":
        return np.concatenate([state, flat_action, next_state], axis=-1).astype(np.float32)
    return np.concatenate([state, flat_action, next_state, delta_state], axis=-1).astype(np.float32)


def episode_random_split(dataset: TransitionDataset, *, holdout_ratio: float, seed: int) -> DatasetSplit:
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


def leave_one_source_out_splits(dataset: TransitionDataset) -> list[DatasetSplit]:
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


def binary_classification_metrics(labels: np.ndarray, probs: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs >= threshold).astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    tp = int(np.sum((pred == 1) & pos))
    tn = int(np.sum((pred == 0) & neg))
    fp = int(np.sum((pred == 1) & neg))
    fn = int(np.sum((pred == 0) & pos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    success = probs[pos]
    failure = probs[neg]
    return {
        "auc": _nan_if_none(rlt_eval.auc_rank(labels, probs)),
        "accuracy": float(np.mean(pred == labels)) if labels.size else math.nan,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_D_success": _nanmean(success),
        "mean_D_failure": _nanmean(failure),
        "D_gap": _nanmean(success) - _nanmean(failure) if success.size and failure.size else math.nan,
        "num_success_transitions": int(np.sum(pos)),
        "num_failure_transitions": int(np.sum(neg)),
    }


def dataset_stats(dataset: TransitionDataset) -> dict[str, Any]:
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


def per_source_metrics(dataset: TransitionDataset, indices: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    labels = dataset.labels[indices]
    sources = dataset.sources[indices]
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        source_metrics = binary_classification_metrics(labels[mask], probs[mask])
        rows.append(
            {
                "source": str(source),
                "warning": bool(source_metrics["mean_D_failure"] > source_metrics["mean_D_success"]),
                **source_metrics,
            }
        )
    return rows


def calibration_rows(labels: np.ndarray, probs: np.ndarray, *, bins: int = 10) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        left, right = float(edges[i]), float(edges[i + 1])
        if i == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        bucket_labels = labels[mask]
        bucket_probs = probs[mask]
        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": int(np.sum(mask)),
                "mean_pred_prob": _nanmean(bucket_probs),
                "actual_success_rate": _nanmean(bucket_labels.astype(np.float64)),
            }
        )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _episode_label(dataset: TransitionDataset, episode_id: str) -> int:
    mask = dataset.episode_ids == episode_id
    values = dataset.labels[mask]
    return int(values[0]) if values.size else 0


def _source_from_path(path: pathlib.Path) -> str:
    text = path.as_posix()
    match = re.search(r"/(\d{4}-\d{2}-\d{2})/", text)
    if match:
        return match.group(1)
    return "unknown"


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _nan_if_none(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return math.nan if values.size == 0 else float(np.mean(values))

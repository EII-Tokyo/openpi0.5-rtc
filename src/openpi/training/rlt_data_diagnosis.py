from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import time
from typing import Any

import numpy as np


TRAJECTORY_POINTS = 16


@dataclasses.dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    source: str
    shard_path: str
    label: int
    terminal_reward: float
    num_transitions: int
    action_horizon: int
    z_mean: np.ndarray
    z_terminal: np.ndarray
    proprio_mean: np.ndarray
    proprio_terminal: np.ndarray
    action_mean: np.ndarray
    action_terminal: np.ndarray
    z_trajectory: np.ndarray
    proprio_trajectory: np.ndarray
    action_trajectory: np.ndarray
    action_delta_norm_mean: float
    action_smoothness_mean: float


def load_episode_summaries(
    manifest_path: pathlib.Path | str,
    *,
    sources: set[str] | None = None,
    min_transitions: int = 3,
) -> list[EpisodeSummary]:
    episodes: list[EpisodeSummary] = []
    for row in _read_jsonl(pathlib.Path(manifest_path)):
        shard_path = pathlib.Path(row["shard_path"]).expanduser().resolve()
        source = str(row.get("batch") or _source_from_path(shard_path))
        if sources is not None and source not in sources:
            continue
        with np.load(shard_path, allow_pickle=False) as data:
            z_rl = np.asarray(data["z_rl"], dtype=np.float32)
            proprio = np.asarray(data["proprio"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            reference_action = np.asarray(data["reference_action"], dtype=np.float32)
            reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
            done = np.asarray(data["done"]).astype(np.bool_)
        if len(z_rl) == 0:
            continue
        terminal_reward = float(reward_seq[done].sum()) if np.any(done) else float(reward_seq.sum())
        label = int(terminal_reward > 0.0)
        action_delta = action - reference_action
        action_smoothness = _chunk_smoothness(action)
        episodes.append(
            EpisodeSummary(
                episode_id=shard_path.stem,
                source=source,
                shard_path=str(shard_path),
                label=label,
                terminal_reward=terminal_reward,
                num_transitions=int(len(z_rl)),
                action_horizon=int(action.shape[1]),
                z_mean=np.mean(z_rl, axis=0),
                z_terminal=z_rl[-1],
                proprio_mean=np.mean(proprio, axis=0),
                proprio_terminal=proprio[-1],
                action_mean=np.mean(action.reshape(action.shape[0], -1), axis=0),
                action_terminal=action[-1].reshape(-1),
                z_trajectory=_resample_sequence(z_rl, TRAJECTORY_POINTS),
                proprio_trajectory=_resample_sequence(proprio, TRAJECTORY_POINTS),
                action_trajectory=_resample_sequence(action.reshape(action.shape[0], -1), TRAJECTORY_POINTS),
                action_delta_norm_mean=float(np.mean(np.linalg.norm(action_delta.reshape(action.shape[0], -1), axis=-1))),
                action_smoothness_mean=float(np.mean(action_smoothness)),
            )
        )
    return episodes


def source_stats(episodes: list[EpisodeSummary]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for source in sorted({episode.source for episode in episodes}):
        source_eps = [episode for episode in episodes if episode.source == source]
        success = [episode for episode in source_eps if episode.label == 1]
        failure = [episode for episode in source_eps if episode.label == 0]
        lengths = np.asarray([episode.num_transitions for episode in source_eps], dtype=np.float64)
        stats[source] = {
            "episodes": len(source_eps),
            "success_episodes": len(success),
            "failure_episodes": len(failure),
            "success_rate": len(success) / max(len(source_eps), 1),
            "length_mean": _mean_or_nan(lengths),
            "length_min": int(np.min(lengths)) if lengths.size else 0,
            "length_p10": _percentile_or_nan(lengths, 10),
            "length_p90": _percentile_or_nan(lengths, 90),
        }
    return stats


def slice_audit_rows(episodes: list[EpisodeSummary], *, min_transitions: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        issue = ""
        if episode.num_transitions < min_transitions:
            issue = "too_few_transitions"
        rows.append(
            {
                "source": episode.source,
                "episode_id": episode.episode_id,
                "label": episode.label,
                "terminal_reward": episode.terminal_reward,
                "num_transitions": episode.num_transitions,
                "action_horizon": episode.action_horizon,
                "action_delta_norm_mean": episode.action_delta_norm_mean,
                "action_smoothness_mean": episode.action_smoothness_mean,
                "suspected_issue": issue,
                "shard_path": episode.shard_path,
            }
        )
    return rows


def nearest_success_rows(episodes: list[EpisodeSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted({episode.source for episode in episodes}):
        source_eps = [episode for episode in episodes if episode.source == source]
        successes = [episode for episode in source_eps if episode.label == 1]
        failures = [episode for episode in source_eps if episode.label == 0]
        if not successes:
            continue
        z_scale = _median_pairwise_scale([episode.z_terminal for episode in source_eps])
        proprio_scale = _median_pairwise_scale([episode.proprio_terminal for episode in source_eps])
        action_scale = _median_pairwise_scale([episode.action_terminal for episode in source_eps])
        z_trajectory_scale = _median_pairwise_scale([episode.z_trajectory.reshape(-1) for episode in source_eps])
        proprio_trajectory_scale = _median_pairwise_scale(
            [episode.proprio_trajectory.reshape(-1) for episode in source_eps]
        )
        action_trajectory_scale = _median_pairwise_scale([episode.action_trajectory.reshape(-1) for episode in source_eps])
        for failure in failures:
            best: tuple[float, dict[str, Any]] | None = None
            for success in successes:
                z_distance = _normalized_l2(failure.z_terminal, success.z_terminal, z_scale)
                proprio_distance = _normalized_l2(failure.proprio_terminal, success.proprio_terminal, proprio_scale)
                action_distance = _normalized_l2(failure.action_terminal, success.action_terminal, action_scale)
                terminal_distance = 0.55 * z_distance + 0.25 * proprio_distance + 0.20 * action_distance
                z_trajectory_distance = _normalized_l2(
                    failure.z_trajectory.reshape(-1),
                    success.z_trajectory.reshape(-1),
                    z_trajectory_scale,
                )
                proprio_trajectory_distance = _normalized_l2(
                    failure.proprio_trajectory.reshape(-1),
                    success.proprio_trajectory.reshape(-1),
                    proprio_trajectory_scale,
                )
                action_trajectory_distance = _normalized_l2(
                    failure.action_trajectory.reshape(-1),
                    success.action_trajectory.reshape(-1),
                    action_trajectory_scale,
                )
                trajectory_distance = (
                    0.55 * z_trajectory_distance
                    + 0.25 * proprio_trajectory_distance
                    + 0.20 * action_trajectory_distance
                )
                combined = 0.20 * terminal_distance + 0.70 * trajectory_distance + 0.10 * action_trajectory_distance
                row = {
                    "source": source,
                    "failure_episode_id": failure.episode_id,
                    "nearest_success_episode_id": success.episode_id,
                    "combined_distance": combined,
                    "terminal_distance": terminal_distance,
                    "trajectory_distance": trajectory_distance,
                    "z_terminal_distance": z_distance,
                    "proprio_terminal_distance": proprio_distance,
                    "action_terminal_distance": action_distance,
                    "z_trajectory_distance": z_trajectory_distance,
                    "proprio_trajectory_distance": proprio_trajectory_distance,
                    "action_trajectory_distance": action_trajectory_distance,
                    "failure_num_transitions": failure.num_transitions,
                    "success_num_transitions": success.num_transitions,
                    "failure_action_delta_norm_mean": failure.action_delta_norm_mean,
                    "success_action_delta_norm_mean": success.action_delta_norm_mean,
                    "failure_shard_path": failure.shard_path,
                    "success_shard_path": success.shard_path,
                }
                if best is None or combined < best[0]:
                    best = (combined, row)
            if best is not None:
                rows.append(best[1])
    return sorted(rows, key=lambda row: (row["source"], row["combined_distance"]))


def hard_negative_rows(similarity_rows: list[dict[str, Any]], *, max_distance: float) -> list[dict[str, Any]]:
    candidates = [
        {
            **row,
            "failure_datetime": datetime_from_episode_id(str(row.get("failure_episode_id") or "")),
            "nearest_success_datetime": datetime_from_episode_id(str(row.get("nearest_success_episode_id") or "")),
            "recommended_use": "hard_negative",
            "reason": "failure close to nearest success in terminal and trajectory embedding/proprio/action",
        }
        for row in similarity_rows
        if float(row["combined_distance"]) <= max_distance
    ]
    return sorted(candidates, key=lambda row: (row["source"], row["combined_distance"]))


def datetime_from_episode_id(episode_id: str) -> str:
    matches = re.findall(r"\.crop_(\d{10,})", episode_id)
    if not matches:
        return ""
    timestamp_ms = int(matches[0])
    with np.errstate(all="ignore"):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_ms / 1000.0))


def label_audit_rows(episodes: list[EpisodeSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        issue = ""
        if episode.label == 1 and episode.terminal_reward <= 0.0:
            issue = "success_label_without_positive_reward"
        elif episode.label == 0 and episode.terminal_reward > 0.0:
            issue = "failure_label_with_positive_reward"
        rows.append(
            {
                "source": episode.source,
                "episode_id": episode.episode_id,
                "label": episode.label,
                "terminal_reward": episode.terminal_reward,
                "suspected_issue": issue,
                "shard_path": episode.shard_path,
            }
        )
    return rows


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


def _chunk_smoothness(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] <= 1:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    diffs = np.diff(actions, axis=1)
    return np.mean(np.linalg.norm(diffs, axis=-1), axis=-1)


def _resample_sequence(values: np.ndarray, points: int) -> np.ndarray:
    if points <= 0:
        raise ValueError("points must be positive")
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.shape[0] == 0:
        raise ValueError("sequence must not be empty")
    flat = sequence.reshape(sequence.shape[0], -1)
    if flat.shape[0] == points:
        return flat
    if flat.shape[0] == 1:
        return np.repeat(flat, points, axis=0)
    source_x = np.linspace(0.0, 1.0, flat.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, points, dtype=np.float32)
    resampled = np.empty((points, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        resampled[:, column] = np.interp(target_x, source_x, flat[:, column]).astype(np.float32)
    return resampled


def _median_pairwise_scale(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0
    values = []
    for i, left in enumerate(vectors):
        for right in vectors[i + 1 :]:
            values.append(float(np.linalg.norm(left - right)))
    median = float(np.median(values)) if values else 1.0
    return max(median, 1e-6)


def _normalized_l2(left: np.ndarray, right: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(left - right) / max(scale, 1e-6))


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")

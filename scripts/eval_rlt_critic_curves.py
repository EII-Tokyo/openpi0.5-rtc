"""Evaluate a trained RLT critic on replay shards and plot Q curves.

The critic predicts one scalar Q value per transition/action chunk. The plots
therefore show Q over the key-region transition index, not per action dimension
inside the chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sqlite3
from typing import Any

from flax import nnx
from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_trainable_manifest


def _resolve_actor_dir(path: pathlib.Path) -> pathlib.Path:
    if path.name == "LATEST":
        return pathlib.Path(path.read_text().strip())
    if path.is_dir() and (path / "LATEST").exists():
        return pathlib.Path((path / "LATEST").read_text().strip())
    if path.is_dir():
        return path
    raise FileNotFoundError(f"actor dir not found: {path}")


def _load_modules(actor_dir: pathlib.Path) -> tuple[rlt.RLTConfig, rlt.RLTTwinCritic, rlt.RLTActor | None, dict[str, Any]]:
    metadata_path = actor_dir / "metadata.json"
    critic_path = actor_dir / "critic.msgpack"
    actor_path = actor_dir / "actor.msgpack"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {actor_dir}")
    if not critic_path.exists():
        raise FileNotFoundError(f"critic.msgpack not found in {actor_dir}")
    metadata = json.loads(metadata_path.read_text())
    config = rlt.RLTConfig(**metadata["rlt_config"])

    critic = rlt.RLTTwinCritic(config, rngs=nnx.Rngs(0))
    critic_state = nnx.state(critic)
    pure_critic_state = serialization.from_bytes(critic_state.to_pure_dict(), critic_path.read_bytes())
    critic_state.replace_by_pure_dict(pure_critic_state)
    nnx.update(critic, critic_state)

    actor = None
    if actor_path.exists():
        actor = rlt.RLTActor(config, rngs=nnx.Rngs(0))
        actor_state = nnx.state(actor)
        pure_actor_state = serialization.from_bytes(actor_state.to_pure_dict(), actor_path.read_bytes())
        actor_state.replace_by_pure_dict(pure_actor_state)
        nnx.update(actor, actor_state)
    return config, critic, actor, metadata


def _find_shards(
    replay_dir: pathlib.Path,
    recursive: bool,
    segment_db_path: pathlib.Path | None,
    manifest_path: pathlib.Path | None,
) -> list[pathlib.Path]:
    if manifest_path is not None:
        return rlt_trainable_manifest.read_manifest_paths(manifest_path)
    if segment_db_path is not None:
        with sqlite3.connect(segment_db_path) as conn:
            rows = conn.execute(
                "SELECT shard_path FROM segments WHERE status = 'committed' AND shard_path IS NOT NULL"
            ).fetchall()
        return sorted(pathlib.Path(row[0]).resolve() for row in rows if row[0] and pathlib.Path(row[0]).exists())
    if recursive:
        return sorted(replay_dir.glob("**/shards/*.npz"))
    direct = sorted(replay_dir.glob("*.npz"))
    nested = sorted((replay_dir / "shards").glob("*.npz")) if (replay_dir / "shards").exists() else []
    return direct + nested


def _episode_label(data: np.lib.npyio.NpzFile) -> int:
    done = np.asarray(data["done"]).astype(bool)
    rewards = np.asarray(data["reward_seq"], dtype=np.float32)
    if done.any():
        terminal_reward = float(rewards[done].sum())
    else:
        terminal_reward = float(rewards.sum())
    return int(terminal_reward > 0.0)


def _auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        i = j
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    sum_pos = float(np.sum(original_ranks[labels == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _summary_for_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    success = scores[labels == 1]
    failure = scores[labels == 0]
    return {
        "auc": _auc_rank(labels, scores),
        "success_mean": None if success.size == 0 else float(np.mean(success)),
        "failure_mean": None if failure.size == 0 else float(np.mean(failure)),
        "success_median": None if success.size == 0 else float(np.median(success)),
        "failure_median": None if failure.size == 0 else float(np.median(failure)),
        "mean_gap_success_minus_failure": None
        if success.size == 0 or failure.size == 0
        else float(np.mean(success) - np.mean(failure)),
    }


def _summary_for_values(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _chunk_smoothness(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[1] <= 1:
        return np.zeros((actions.shape[0],), dtype=np.float32)
    diffs = np.diff(actions, axis=1)
    return np.mean(np.linalg.norm(diffs, axis=-1), axis=-1)


def _plot_mean_curves(rows: list[dict[str, Any]], out_path: pathlib.Path, score_key: str) -> None:
    bins = np.linspace(0.0, 1.0, 41)
    centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(10, 5.5))
    for label, name, color in [(1, "success", "#238443"), (0, "failure", "#b2182b")]:
        means = []
        lo = []
        hi = []
        for left, right in zip(bins[:-1], bins[1:], strict=True):
            values = [
                float(row[score_key])
                for row in rows
                if int(row["label"]) == label and left <= float(row["progress"]) <= right
            ]
            if values:
                arr = np.asarray(values)
                means.append(float(arr.mean()))
                lo.append(float(np.percentile(arr, 25)))
                hi.append(float(np.percentile(arr, 75)))
            else:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
        means_arr = np.asarray(means)
        plt.plot(centers, means_arr, label=name, color=color, linewidth=2.0)
        plt.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)
    plt.xlabel("normalized key-region progress")
    plt.ylabel(score_key)
    plt.title(f"Mean {score_key} over key-region progress")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_hist(rows: list[dict[str, Any]], out_path: pathlib.Path, score_key: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, name, color in [(1, "success", "#238443"), (0, "failure", "#b2182b")]:
        values = [float(row[score_key]) for row in rows if int(row["label"]) == label]
        plt.hist(values, bins=50, alpha=0.45, density=True, label=name, color=color)
    plt.xlabel(score_key)
    plt.ylabel("density")
    plt.title(f"{score_key} distribution by terminal label")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_sample_curves(rows: list[dict[str, Any]], out_path: pathlib.Path, score_key: str, max_per_class: int) -> None:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    selected: list[tuple[str, list[dict[str, Any]]]] = []
    counts = {0: 0, 1: 0}
    for episode_id, episode_rows in sorted(by_episode.items()):
        label = int(episode_rows[0]["label"])
        if counts[label] >= max_per_class:
            continue
        selected.append((episode_id, sorted(episode_rows, key=lambda item: int(item["transition_index"]))))
        counts[label] += 1
        if counts[0] >= max_per_class and counts[1] >= max_per_class:
            break
    plt.figure(figsize=(10, 5.5))
    for _, episode_rows in selected:
        label = int(episode_rows[0]["label"])
        color = "#238443" if label else "#b2182b"
        alpha = 0.5 if label else 0.35
        plt.plot(
            [float(row["progress"]) for row in episode_rows],
            [float(row[score_key]) for row in episode_rows],
            color=color,
            alpha=alpha,
            linewidth=1.2,
        )
    plt.xlabel("normalized key-region progress")
    plt.ylabel(score_key)
    plt.title(f"Sample {score_key} curves")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True, type=pathlib.Path)
    parser.add_argument("--replay-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--recursive-scan", action="store_true")
    parser.add_argument("--segment-db-path", type=pathlib.Path, default=None)
    parser.add_argument("--manifest-path", type=pathlib.Path, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--sample-curves-per-class", type=int, default=12)
    args = parser.parse_args()

    actor_dir = _resolve_actor_dir(args.actor_dir)
    config, critic, actor, metadata = _load_modules(actor_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def score_batch(z_rl, proprio, action, reference_action):
        x = rlt.make_state(z_rl, proprio)
        actual_q = critic.min_q(x, action)
        reference_q = critic.min_q(x, reference_action)
        if actor is None:
            actor_q = jnp.full_like(actual_q, jnp.nan)
            delta_norm = jnp.full_like(actual_q, jnp.nan)
            actor_action = jnp.full_like(action, jnp.nan)
        else:
            actor_action = actor(x, reference_action, sample=False)
            actor_q = critic.min_q(x, actor_action)
            delta = actor_action - reference_action
            delta_norm = jnp.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1)
        return actual_q, reference_q, actor_q, delta_norm, actor_action

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    z_pieces: list[np.ndarray] = []
    proprio_pieces: list[np.ndarray] = []
    action_pieces: list[np.ndarray] = []
    reference_pieces: list[np.ndarray] = []
    shards = _find_shards(args.replay_dir, args.recursive_scan, args.segment_db_path, args.manifest_path)
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    for shard_path in shards:
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                z_rl = np.asarray(data["z_rl"], dtype=np.float32)
                proprio = np.asarray(data["proprio"], dtype=np.float32)
                action = np.asarray(data["action"], dtype=np.float32)
                reference_action = np.asarray(data["reference_action"], dtype=np.float32)
                if action.shape[1] != config.action_horizon or action.shape[2] != config.action_dim:
                    raise ValueError(f"action shape {action.shape} does not match config {config.action_horizon}x{config.action_dim}")
                label = _episode_label(data)
                reward_sum = float(np.asarray(data["reward_seq"], dtype=np.float32).sum())
                done = np.asarray(data["done"]).astype(bool)
            row_start = len(rows)
            z_pieces.append(z_rl)
            proprio_pieces.append(proprio)
            action_pieces.append(action)
            reference_pieces.append(reference_action)
            n = int(z_rl.shape[0])
            episode_id = shard_path.stem
            for i in range(n):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "shard_path": str(shard_path),
                        "label": label,
                        "reward_sum": reward_sum,
                        "transition_index": i,
                        "num_transitions": n,
                        "progress": 0.0 if n <= 1 else i / (n - 1),
                        "done": bool(done[i]) if i < len(done) else False,
                        "q_actual": np.nan,
                        "q_reference": np.nan,
                        "q_actor": np.nan,
                        "actor_delta_norm": np.nan,
                        "_score_index": row_start + i,
                    }
                )
        except Exception as exc:  # Keep evaluation robust over partial local datasets.
            skipped.append({"path": str(shard_path), "error": str(exc)})

    if not rows:
        raise RuntimeError(f"no replay rows evaluated under {args.replay_dir}")

    all_z = np.concatenate(z_pieces, axis=0)
    all_proprio = np.concatenate(proprio_pieces, axis=0)
    all_action = np.concatenate(action_pieces, axis=0)
    all_reference = np.concatenate(reference_pieces, axis=0)
    total = all_z.shape[0]
    batch_size = int(args.score_batch_size)
    actual_scores = np.empty(total, dtype=np.float32)
    reference_scores = np.empty(total, dtype=np.float32)
    actor_scores = np.empty(total, dtype=np.float32)
    delta_norms = np.empty(total, dtype=np.float32)
    actor_actions = np.empty_like(all_action, dtype=np.float32)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        pad = batch_size - (end - start)
        z_batch = all_z[start:end]
        proprio_batch = all_proprio[start:end]
        action_batch = all_action[start:end]
        reference_batch = all_reference[start:end]
        if pad:
            z_batch = np.pad(z_batch, ((0, pad), (0, 0)))
            proprio_batch = np.pad(proprio_batch, ((0, pad), (0, 0)))
            action_batch = np.pad(action_batch, ((0, pad), (0, 0), (0, 0)))
            reference_batch = np.pad(reference_batch, ((0, pad), (0, 0), (0, 0)))
        actual_q, reference_q, actor_q, delta_norm, actor_action = jax.device_get(
            score_batch(z_batch, proprio_batch, action_batch, reference_batch)
        )
        valid = end - start
        actual_scores[start:end] = actual_q[:valid]
        reference_scores[start:end] = reference_q[:valid]
        actor_scores[start:end] = actor_q[:valid]
        delta_norms[start:end] = delta_norm[:valid]
        actor_actions[start:end] = actor_action[:valid]

    actor_smoothness = _chunk_smoothness(actor_actions)
    for row in rows:
        score_index = int(row.pop("_score_index"))
        row["q_actual"] = float(actual_scores[score_index])
        row["q_reference"] = float(reference_scores[score_index])
        row["q_actor"] = float(actor_scores[score_index])
        row["actor_delta_norm"] = float(delta_norms[score_index])
        row["actor_chunk_smoothness"] = float(actor_smoothness[score_index])

    csv_path = args.output_dir / "per_transition_q.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = np.asarray([int(row["label"]) for row in rows])
    terminal_rows = [row for row in rows if bool(row["done"])]
    summary = {
        "actor_dir": str(actor_dir),
        "actor_step": metadata.get("step"),
        "replay_dir": str(args.replay_dir),
        "segment_db_path": None if args.segment_db_path is None else str(args.segment_db_path),
        "manifest_path": None if args.manifest_path is None else str(args.manifest_path),
        "num_shards_seen": len(shards),
        "num_shards_evaluated": len({row["episode_id"] for row in rows}),
        "num_shards_skipped": len(skipped),
        "num_transitions": len(rows),
        "success_episodes": len({row["episode_id"] for row in rows if int(row["label"]) == 1}),
        "failure_episodes": len({row["episode_id"] for row in rows if int(row["label"]) == 0}),
        "success_transitions": int(np.sum(labels == 1)),
        "failure_transitions": int(np.sum(labels == 0)),
        "scores_all_transitions": {
            key: _summary_for_scores(labels, np.asarray([float(row[key]) for row in rows]))
            for key in ["q_actual", "q_reference", "q_actor"]
        },
        "scores_terminal_transitions": {
            key: _summary_for_scores(
                np.asarray([int(row["label"]) for row in terminal_rows]),
                np.asarray([float(row[key]) for row in terminal_rows]),
            )
            for key in ["q_actual", "q_reference", "q_actor"]
        },
        "actor_delta_norm": {
            "all": _summary_for_values(np.asarray([float(row["actor_delta_norm"]) for row in rows])),
            "terminal": _summary_for_values(np.asarray([float(row["actor_delta_norm"]) for row in terminal_rows])),
        },
        "actor_chunk_smoothness": {
            "all": _summary_for_values(np.asarray([float(row["actor_chunk_smoothness"]) for row in rows])),
            "terminal": _summary_for_values(np.asarray([float(row["actor_chunk_smoothness"]) for row in terminal_rows])),
        },
        "skipped": skipped[:50],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    for score_key in ["q_actual", "q_reference", "q_actor"]:
        _plot_mean_curves(rows, args.output_dir / f"{score_key}_mean_curve.png", score_key)
        _plot_hist(rows, args.output_dir / f"{score_key}_hist.png", score_key)
        _plot_sample_curves(
            rows,
            args.output_dir / f"{score_key}_sample_curves.png",
            score_key,
            max_per_class=args.sample_curves_per_class,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

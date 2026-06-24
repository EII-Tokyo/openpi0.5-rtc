from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import math
import pathlib
import sqlite3
from typing import Any

from flax import nnx
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_trainable_manifest


@dataclasses.dataclass(frozen=True)
class ReplaySplit:
    train_paths: tuple[pathlib.Path, ...]
    holdout_paths: tuple[pathlib.Path, ...]


@dataclasses.dataclass(frozen=True)
class CriticUsabilityDecision:
    is_critic_usable: bool
    warning_reason: str


@dataclasses.dataclass(frozen=True)
class HoldoutEvalResult:
    metrics: list[dict[str, Any]]
    best_metric: dict[str, Any] | None
    per_transition_rows: list[dict[str, Any]]
    skipped: list[dict[str, str]]


REPLAY_KEYS: tuple[str, ...] = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


def find_replay_shards(
    replay_dir: pathlib.Path,
    *,
    recursive: bool = False,
    segment_db_path: pathlib.Path | None = None,
    manifest_path: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    if manifest_path is not None:
        return sorted(path.resolve() for path in rlt_trainable_manifest.read_manifest_paths(manifest_path) if path.exists())
    if segment_db_path is not None:
        with sqlite3.connect(segment_db_path) as conn:
            rows = conn.execute(
                "SELECT shard_path FROM segments WHERE status = 'committed' AND shard_path IS NOT NULL"
            ).fetchall()
        return sorted(pathlib.Path(row[0]).expanduser().resolve() for row in rows if row[0] and pathlib.Path(row[0]).exists())
    if recursive:
        return sorted(path.resolve() for path in replay_dir.glob("**/shards/*.npz") if path.is_file())
    direct = sorted(path.resolve() for path in replay_dir.glob("*.npz") if path.is_file())
    nested_dir = replay_dir / "shards"
    nested = sorted(path.resolve() for path in nested_dir.glob("*.npz") if path.is_file()) if nested_dir.exists() else []
    return direct + nested


def split_shards(paths: list[pathlib.Path], *, holdout_ratio: float, seed: int) -> ReplaySplit:
    if not 0.0 <= holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")
    resolved = tuple(sorted(path.expanduser().resolve() for path in paths))
    if holdout_ratio == 0.0:
        return ReplaySplit(train_paths=resolved, holdout_paths=())
    if len(resolved) < 2:
        raise ValueError("At least two replay shards are required for a non-empty holdout split.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(resolved))
    holdout_count = max(1, round(len(resolved) * holdout_ratio))
    holdout_count = min(holdout_count, len(resolved) - 1)
    holdout_indices = {int(index) for index in order[:holdout_count]}
    train_paths = tuple(path for index, path in enumerate(resolved) if index not in holdout_indices)
    holdout_paths = tuple(path for index, path in enumerate(resolved) if index in holdout_indices)
    return ReplaySplit(train_paths=train_paths, holdout_paths=holdout_paths)


def write_manifest(paths: list[pathlib.Path] | tuple[pathlib.Path, ...], manifest_path: pathlib.Path) -> pathlib.Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        for path in paths:
            file.write(json.dumps({"shard_path": str(path.expanduser().resolve())}, sort_keys=True) + "\n")
    return manifest_path


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
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
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def judge_critic_usability(
    *,
    success_q_mean: float | None,
    failure_q_mean: float | None,
    auc: float | None,
    success_actor_advantage_mean: float | None,
    failure_actor_advantage_mean: float | None,
    q_gap_stability_warning: bool,
) -> CriticUsabilityDecision:
    warnings: list[str] = []
    if success_q_mean is None or failure_q_mean is None:
        warnings.append("missing_success_or_failure_q")
    elif failure_q_mean >= success_q_mean:
        warnings.append("failure_q_mean>=success_q_mean")
    if auc is None:
        warnings.append("missing_auc")
    elif auc <= 0.60:
        warnings.append("auc<=0.60")
    elif auc <= 0.70:
        warnings.append("auc<=0.70")
    if (
        success_actor_advantage_mean is not None
        and failure_actor_advantage_mean is not None
        and failure_actor_advantage_mean > success_actor_advantage_mean
    ):
        warnings.append("failure_actor_advantage>success_actor_advantage")
    if q_gap_stability_warning:
        warnings.append("checkpoint_q_gap_ranking_unstable")
    usable = (
        success_q_mean is not None
        and failure_q_mean is not None
        and success_q_mean > failure_q_mean
        and auc is not None
        and auc > 0.70
        and "failure_actor_advantage>success_actor_advantage" not in warnings
        and not q_gap_stability_warning
    )
    return CriticUsabilityDecision(is_critic_usable=usable, warning_reason=";".join(warnings))


def discover_inference_checkpoints(checkpoint_dir: pathlib.Path) -> list[pathlib.Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if (checkpoint_dir / "metadata.json").exists() and (checkpoint_dir / "critic.msgpack").exists():
        return [checkpoint_dir]
    candidates: list[pathlib.Path] = []
    for base in (
        checkpoint_dir,
        checkpoint_dir / "inference_actor",
        checkpoint_dir / "snapshots" / "inference_actor",
    ):
        if not base.exists():
            continue
        candidates.extend(
            path
            for path in sorted(base.iterdir())
            if path.is_dir() and (path / "metadata.json").exists() and (path / "critic.msgpack").exists()
        )
    return sorted(set(candidates), key=_checkpoint_sort_key)


def evaluate_holdout_checkpoints(
    *,
    checkpoint_dirs: list[pathlib.Path],
    holdout_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    score_batch_size: int = 512,
) -> HoldoutEvalResult:
    if not checkpoint_dirs:
        raise ValueError("No critic checkpoints were found.")
    if not holdout_paths:
        raise ValueError("No holdout replay shards were provided.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    all_transition_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    best_rows: list[dict[str, Any]] = []

    for checkpoint_dir in checkpoint_dirs:
        config, critic, actor, metadata = load_inference_modules(checkpoint_dir)
        arrays, rows, checkpoint_skipped = load_holdout_arrays(holdout_paths, config=config)
        skipped.extend(checkpoint_skipped)
        if arrays is None or not rows:
            raise RuntimeError(f"No holdout rows could be evaluated for checkpoint {checkpoint_dir}")
        scored_rows = score_holdout_rows(
            arrays,
            rows,
            critic=critic,
            actor=actor,
            config=config,
            score_batch_size=score_batch_size,
        )
        metric = summarize_checkpoint(
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
            rows=scored_rows,
            train_critic_loss=_read_train_critic_loss(checkpoint_dir),
        )
        metrics.append(metric)
        all_transition_rows.extend({"checkpoint_step": metric["step"], **row} for row in scored_rows)
        if best_checkpoint_metric(metrics) is metric:
            best_rows = scored_rows

    stability_warning = _q_gap_unstable(metrics)
    for metric in metrics:
        decision = judge_critic_usability(
            success_q_mean=_as_optional_float(metric["success_q_mean"]),
            failure_q_mean=_as_optional_float(metric["failure_q_mean"]),
            auc=_as_optional_float(metric["auc"]),
            success_actor_advantage_mean=_as_optional_float(metric["success_actor_advantage_mean"]),
            failure_actor_advantage_mean=_as_optional_float(metric["failure_actor_advantage_mean"]),
            q_gap_stability_warning=stability_warning,
        )
        metric["is_critic_usable"] = decision.is_critic_usable
        metric["warning_reason"] = decision.warning_reason

    best = best_checkpoint_metric(metrics)
    if best is not None:
        best_step = best["step"]
        best_rows = [row for row in all_transition_rows if row["checkpoint_step"] == best_step]
    write_metric_reports(metrics, output_dir)
    write_best_checkpoint(best, output_dir)
    write_markdown_report(metrics, best, output_dir, num_holdout_transitions=len(best_rows), skipped=skipped)
    write_plots(metrics, best_rows, output_dir)
    return HoldoutEvalResult(metrics=metrics, best_metric=best, per_transition_rows=all_transition_rows, skipped=skipped)


def load_inference_modules(actor_dir: pathlib.Path) -> tuple[rlt.RLTConfig, rlt.RLTTwinCritic, rlt.RLTActor | None, dict[str, Any]]:
    metadata = json.loads((actor_dir / "metadata.json").read_text())
    config = rlt.RLTConfig(**metadata["rlt_config"])
    critic = rlt.RLTTwinCritic(config, rngs=nnx.Rngs(0))
    critic_state = nnx.state(critic)
    pure_critic_state = serialization.from_bytes(critic_state.to_pure_dict(), (actor_dir / "critic.msgpack").read_bytes())
    critic_state.replace_by_pure_dict(pure_critic_state)
    nnx.update(critic, critic_state)
    actor = None
    actor_path = actor_dir / "actor.msgpack"
    if actor_path.exists():
        actor = rlt.RLTActor(config, rngs=nnx.Rngs(0))
        actor_state = nnx.state(actor)
        pure_actor_state = serialization.from_bytes(actor_state.to_pure_dict(), actor_path.read_bytes())
        actor_state.replace_by_pure_dict(pure_actor_state)
        nnx.update(actor, actor_state)
    return config, critic, actor, metadata


def load_holdout_arrays(
    holdout_paths: list[pathlib.Path],
    *,
    config: rlt.RLTConfig,
) -> tuple[dict[str, np.ndarray] | None, list[dict[str, Any]], list[dict[str, str]]]:
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in REPLAY_KEYS}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    offset = 0
    for shard_path in holdout_paths:
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                missing = [key for key in REPLAY_KEYS if key not in data]
                if missing:
                    raise ValueError(f"missing replay keys: {missing}")
                shard = {key: np.asarray(data[key]) for key in REPLAY_KEYS}
            shard = _slice_to_config_horizon(shard, config)
            label = _episode_label(shard)
            n = int(shard["z_rl"].shape[0])
            for key, value in shard.items():
                pieces[key].append(value)
            rows.extend(
                {
                    "row_index": offset + index,
                    "episode_id": shard_path.stem,
                    "shard_path": str(shard_path),
                    "label": label,
                    "transition_index": index,
                    "num_transitions": n,
                    "progress": 0.0 if n <= 1 else index / (n - 1),
                    "done": bool(shard["done"][index]),
                }
                for index in range(n)
            )
            offset += n
        except Exception as exc:
            skipped.append({"path": str(shard_path), "error": str(exc)})
    if offset == 0:
        return None, rows, skipped
    return {key: np.concatenate(value, axis=0) for key, value in pieces.items()}, rows, skipped


def score_holdout_rows(
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    *,
    critic: rlt.RLTTwinCritic,
    actor: rlt.RLTActor | None,
    config: rlt.RLTConfig,
    score_batch_size: int,
) -> list[dict[str, Any]]:
    del config

    @jax.jit
    def score_batch(z_rl, proprio, action, reference_action, reward_seq, next_z_rl, next_proprio, next_reference_action, done):
        x = rlt.make_state(z_rl, proprio)
        next_x = rlt.make_state(next_z_rl, next_proprio)
        predicted_q = critic.min_q(x, action)
        reference_q = critic.min_q(x, reference_action)
        if actor is None:
            actor_action = reference_action
            actor_q = jnp.full_like(predicted_q, jnp.nan)
            actor_advantage = jnp.full_like(predicted_q, jnp.nan)
            next_action = next_reference_action
        else:
            actor_action = actor(x, reference_action, sample=False)
            actor_q = critic.min_q(x, actor_action)
            actor_advantage = actor_q - reference_q
            next_action = actor(next_x, next_reference_action, sample=False)
        next_q = critic.min_q(next_x, next_action)
        target_q = rlt.td3_target(reward_seq, done, next_q, gamma=critic.q1.config.gamma)
        bellman_error = jnp.square(predicted_q - target_q)
        return predicted_q, reference_q, actor_q, actor_advantage, target_q, bellman_error

    total = int(arrays["z_rl"].shape[0])
    outputs = {key: np.empty(total, dtype=np.float32) for key in ("predicted_q", "reference_q", "actor_q", "actor_advantage", "target_q", "bellman_error")}
    for start in range(0, total, score_batch_size):
        end = min(start + score_batch_size, total)
        batch = {key: value[start:end] for key, value in arrays.items()}
        result = jax.device_get(score_batch(**{key: jnp.asarray(value) for key, value in batch.items()}))
        for name, values in zip(outputs.keys(), result, strict=True):
            outputs[name][start:end] = np.asarray(values[: end - start], dtype=np.float32)
    scored: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["row_index"])
        scored.append({**row, **{key: float(value[index]) for key, value in outputs.items()}})
    return scored


def summarize_checkpoint(
    *,
    checkpoint_dir: pathlib.Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    train_critic_loss: float | None,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows])
    predicted_q = np.asarray([float(row["predicted_q"]) for row in rows], dtype=np.float64)
    target_q = np.asarray([float(row["target_q"]) for row in rows], dtype=np.float64)
    bellman = np.asarray([float(row["bellman_error"]) for row in rows], dtype=np.float64)
    actor_q = np.asarray([float(row["actor_q"]) for row in rows], dtype=np.float64)
    reference_q = np.asarray([float(row["reference_q"]) for row in rows], dtype=np.float64)
    advantage = np.asarray([float(row["actor_advantage"]) for row in rows], dtype=np.float64)
    success_q = predicted_q[labels == 1]
    failure_q = predicted_q[labels == 0]
    success_advantage = advantage[labels == 1]
    failure_advantage = advantage[labels == 0]
    return {
        "step": int(metadata.get("step", _step_from_path(checkpoint_dir))),
        "checkpoint_path": str(checkpoint_dir),
        "train_critic_loss": _none_to_nan(train_critic_loss),
        "holdout_bellman_loss": _nanmean(bellman),
        "target_q_mean": _nanmean(target_q),
        "target_q_std": _nanstd(target_q),
        "predicted_q_mean": _nanmean(predicted_q),
        "predicted_q_std": _nanstd(predicted_q),
        "success_q_mean": _nanmean(success_q),
        "success_q_std": _nanstd(success_q),
        "failure_q_mean": _nanmean(failure_q),
        "failure_q_std": _nanstd(failure_q),
        "q_gap": _nanmean(success_q) - _nanmean(failure_q) if success_q.size and failure_q.size else math.nan,
        "auc": _none_to_nan(auc_rank(labels, predicted_q)),
        "actor_q_mean": _nanmean(actor_q),
        "reference_q_mean": _nanmean(reference_q),
        "actor_advantage_mean": _nanmean(advantage),
        "actor_advantage_std": _nanstd(advantage),
        "success_actor_advantage_mean": _nanmean(success_advantage),
        "failure_actor_advantage_mean": _nanmean(failure_advantage),
        "num_holdout_transitions": len(rows),
        "success_transitions": int(np.sum(labels == 1)),
        "failure_transitions": int(np.sum(labels == 0)),
        "is_critic_usable": False,
        "warning_reason": "",
    }


def best_checkpoint_metric(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not metrics:
        return None

    def key(metric: dict[str, Any]) -> tuple[float, float, float, float]:
        auc = _finite_or(metric.get("auc"), -1.0)
        q_gap = _finite_or(metric.get("q_gap"), -1e9)
        loss = _finite_or(metric.get("holdout_bellman_loss"), 1e9)
        advantage_gap = _finite_or(metric.get("success_actor_advantage_mean"), 0.0) - _finite_or(
            metric.get("failure_actor_advantage_mean"), 0.0
        )
        return (auc, q_gap, -loss, advantage_gap)

    return max(metrics, key=key)


def write_metric_reports(metrics: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    if not metrics:
        return
    fieldnames = list(metrics[0].keys())
    with (output_dir / "critic_holdout_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    (output_dir / "critic_holdout_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_best_checkpoint(best: dict[str, Any] | None, output_dir: pathlib.Path) -> None:
    if best is None:
        text = "No checkpoint evaluated.\n"
    else:
        text = "\n".join(
            [
                f"best checkpoint path: {best['checkpoint_path']}",
                "selection reason: highest holdout AUC, then positive q_gap, then lower holdout Bellman loss, then success/failure actor advantage ordering",
                f"AUC: {best['auc']}",
                f"q_gap: {best['q_gap']}",
                f"success_q_mean: {best['success_q_mean']}",
                f"failure_q_mean: {best['failure_q_mean']}",
                f"actor_advantage_summary: mean={best['actor_advantage_mean']} success={best['success_actor_advantage_mean']} failure={best['failure_actor_advantage_mean']}",
                f"warning_reason: {best.get('warning_reason') or ''}",
                "",
            ]
        )
    (output_dir / "best_critic_checkpoint.txt").write_text(text, encoding="utf-8")


def write_markdown_report(
    metrics: list[dict[str, Any]],
    best: dict[str, Any] | None,
    output_dir: pathlib.Path,
    *,
    num_holdout_transitions: int,
    skipped: list[dict[str, str]],
) -> None:
    if best is None:
        report = "# Critic Holdout Report\n\nNo checkpoint evaluated.\n"
    else:
        usable = "可用" if best["is_critic_usable"] else "不可靠"
        report = f"""# Critic Holdout Report

## 数据概况

- 评估 checkpoint 数量: {len(metrics)}
- holdout transitions: {num_holdout_transitions}
- success transitions: {best['success_transitions']}
- failure transitions: {best['failure_transitions']}
- skipped shards: {len(skipped)}

## 最佳 checkpoint

- path: `{best['checkpoint_path']}`
- step: {best['step']}
- AUC: {best['auc']:.4f}
- q_gap: {best['q_gap']:.6f}
- success_q_mean: {best['success_q_mean']:.6f}
- failure_q_mean: {best['failure_q_mean']:.6f}
- holdout_bellman_loss: {best['holdout_bellman_loss']:.6f}
- critic 判断: **{usable}**
- warning_reason: `{best.get('warning_reason') or ''}`

## Actor advantage

- actor_q_mean: {best['actor_q_mean']:.6f}
- reference_q_mean: {best['reference_q_mean']:.6f}
- actor_advantage_mean: {best['actor_advantage_mean']:.6f}
- success_actor_advantage_mean: {best['success_actor_advantage_mean']:.6f}
- failure_actor_advantage_mean: {best['failure_actor_advantage_mean']:.6f}

## 解释

如果 success Q 均值高于 failure Q, 且 AUC 大于 0.70, 说明 critic 在 holdout 数据上具备基本排序能力。若 failure actor advantage 高于 success, 则说明 critic 可能在错误鼓励失败动作, 本报告会标记为不可靠。

时间曲线使用 replay shard 内的 `transition_index` 和 normalized progress。若原始 replay 没有真实 `episode_id` / `timestep` 字段, 这不是完整 episode 时间, 只是 key region 内部传播检查。

## 下一步建议

优先使用最佳 checkpoint 进行 actor 训练或部署前评估; 如果 critic 被标记为不可靠, 应先增加可区分 success/failure 的数据、检查 reward 标注, 或缩短/重切关键区域后重新训练 critic。
"""
    (output_dir / "critic_holdout_report.md").write_text(report, encoding="utf-8")


def write_plots(metrics: list[dict[str, Any]], best_rows: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics or not best_rows:
        return
    labels = np.asarray([int(row["label"]) for row in best_rows])
    q = np.asarray([float(row["predicted_q"]) for row in best_rows])
    advantage = np.asarray([float(row["actor_advantage"]) for row in best_rows])
    success_q = q[labels == 1]
    failure_q = q[labels == 0]

    plt.figure(figsize=(8, 5))
    if success_q.size:
        plt.hist(success_q, bins=40, alpha=0.5, density=True, label="success")
    if failure_q.size:
        plt.hist(failure_q, bins=40, alpha=0.5, density=True, label="failure")
    plt.xlabel("Q")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "q_distribution_success_failure.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot([success_q, failure_q], labels=["success", "failure"])
    plt.ylabel("Q")
    plt.tight_layout()
    plt.savefig(output_dir / "q_boxplot_success_failure.png", dpi=180)
    plt.close()

    steps = [metric["step"] for metric in metrics]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [metric["auc"] for metric in metrics], marker="o")
    plt.xlabel("checkpoint step")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_over_checkpoints.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [metric["q_gap"] for metric in metrics], marker="o")
    plt.xlabel("checkpoint step")
    plt.ylabel("success_q_mean - failure_q_mean")
    plt.tight_layout()
    plt.savefig(output_dir / "q_gap_over_checkpoints.png", dpi=180)
    plt.close()

    success_adv = advantage[labels == 1]
    failure_adv = advantage[labels == 0]
    plt.figure(figsize=(6, 5))
    plt.boxplot([success_adv[np.isfinite(success_adv)], failure_adv[np.isfinite(failure_adv)]], labels=["success", "failure"])
    plt.ylabel("Q(actor) - Q(reference)")
    plt.tight_layout()
    plt.savefig(output_dir / "actor_advantage_success_failure.png", dpi=180)
    plt.close()

    _plot_time_curves(best_rows, output_dir, label=1, filename="q_over_time_success.png")
    _plot_time_curves(best_rows, output_dir, label=0, filename="q_over_time_failure.png")
    _plot_mean_time_curves(best_rows, output_dir / "q_over_time_mean_success_failure.png")


def _plot_time_curves(rows: list[dict[str, Any]], output_dir: pathlib.Path, *, label: int, filename: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["label"]) == label:
            by_episode.setdefault(str(row["episode_id"]), []).append(row)
    for raw_episode_rows in by_episode.values():
        episode_rows = sorted(raw_episode_rows, key=lambda item: int(item["transition_index"]))
        plt.plot([row["progress"] for row in episode_rows], [row["predicted_q"] for row in episode_rows], alpha=0.35)
    plt.xlabel("normalized key-region progress")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def _plot_mean_time_curves(rows: list[dict[str, Any]], out_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, 31)
    centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(9, 5))
    for label, name in [(1, "success"), (0, "failure")]:
        means = []
        for left, right in itertools.pairwise(bins):
            values = [row["predicted_q"] for row in rows if int(row["label"]) == label and left <= float(row["progress"]) <= right]
            means.append(np.nan if not values else float(np.mean(values)))
        plt.plot(centers, means, marker="o", label=name)
    plt.xlabel("normalized key-region progress")
    plt.ylabel("mean Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _slice_to_config_horizon(arrays: dict[str, np.ndarray], config: rlt.RLTConfig) -> dict[str, np.ndarray]:
    sliced = dict(arrays)
    for key in ("action", "reference_action", "next_reference_action"):
        if sliced[key].shape[1] < config.action_horizon or sliced[key].shape[2] != config.action_dim:
            raise ValueError(f"{key} shape {sliced[key].shape} does not match config horizon/dim")
        sliced[key] = sliced[key][:, : config.action_horizon, : config.action_dim].astype(np.float32)
    if sliced["reward_seq"].shape[1] < config.action_horizon:
        raise ValueError(f"reward_seq shape {sliced['reward_seq'].shape} is shorter than config horizon")
    sliced["reward_seq"] = sliced["reward_seq"][:, : config.action_horizon].astype(np.float32)
    for key in ("z_rl", "next_z_rl", "proprio", "next_proprio"):
        sliced[key] = sliced[key].astype(np.float32)
    sliced["done"] = sliced["done"].astype(np.bool_)
    return sliced


def _episode_label(arrays: dict[str, np.ndarray]) -> int:
    done = arrays["done"].astype(np.bool_)
    rewards = arrays["reward_seq"].astype(np.float32)
    terminal_rewards = rewards[done].sum(axis=-1) if np.any(done) else np.asarray([], dtype=np.float32)
    return int(np.any(terminal_rewards > 0.0))


def _checkpoint_sort_key(path: pathlib.Path) -> tuple[int, str]:
    return (_step_from_path(path), str(path))


def _step_from_path(path: pathlib.Path) -> int:
    try:
        return int(path.name)
    except ValueError:
        return -1


def _none_to_nan(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return math.nan
    return float(np.nanmean(values))


def _nanstd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return math.nan
    return float(np.nanstd(values))


def _finite_or(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if math.isfinite(numeric) else fallback


def _as_optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _q_gap_unstable(metrics: list[dict[str, Any]]) -> bool:
    if len(metrics) < 3:
        return False
    gaps = np.asarray([_finite_or(metric.get("q_gap"), math.nan) for metric in metrics], dtype=np.float64)
    finite = gaps[np.isfinite(gaps)]
    if finite.size < 3:
        return False
    signs = np.sign(finite)
    return bool(np.any(signs > 0) and np.any(signs < 0))


def _read_train_critic_loss(checkpoint_dir: pathlib.Path) -> float | None:
    metrics_path = checkpoint_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None
    value = data.get("train_critic_loss") or data.get("critic_loss")
    return None if value is None else float(value)

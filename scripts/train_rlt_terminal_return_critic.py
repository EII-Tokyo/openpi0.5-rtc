from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import shutil
from typing import Any

from flax import nnx
from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro

from openpi.models import rlt
from openpi.training import rlt_terminal_return


@dataclasses.dataclass
class Args:
    manifest_path: pathlib.Path = pathlib.Path("local_rlt_manifests/trainable_clean_committed_20260623_time_sorted.jsonl")
    output_dir: pathlib.Path = pathlib.Path("local_rlt_runs/rlt_terminal_return_critic_20260624")
    num_train_steps: int = 5_000
    batch_size: int = 256
    holdout_ratio: float = 0.2
    seed: int = 0
    gamma: float = 0.99
    failure_target: float = 0.0
    train_action_horizon: int = 10
    critical_ratio: float = 0.4
    critic_lr: float = 3e-4
    log_interval: int = 250
    balanced_batches: bool = True
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    name: str
    critical_ratio: float | None


EXPERIMENTS = (
    ExperimentSpec("terminal-all", None),
    ExperimentSpec("terminal-critical", 0.4),
)


def _slice_horizon(dataset: rlt_terminal_return.TerminalReturnDataset, horizon: int) -> rlt_terminal_return.TerminalReturnDataset:
    if horizon <= 0:
        raise ValueError("train_action_horizon must be positive")
    if dataset.action.shape[1] < horizon:
        raise ValueError(f"Dataset action horizon {dataset.action.shape[1]} is shorter than requested {horizon}")
    return dataclasses.replace(
        dataset,
        action=dataset.action[:, :horizon, :],
        reference_action=dataset.reference_action[:, :horizon, :],
    )


def _init_critic(dataset: rlt_terminal_return.TerminalReturnDataset, *, horizon: int, seed: int) -> rlt.RLTTwinCritic:
    config = rlt.RLTConfig(
        z_dim=int(dataset.z_rl.shape[-1]),
        proprio_dim=int(dataset.proprio.shape[-1]),
        action_horizon=horizon,
        action_dim=int(dataset.action.shape[-1]),
    )
    return rlt.RLTTwinCritic(config, rngs=nnx.Rngs(jax.random.key(seed)))


def _train_critic(
    dataset: rlt_terminal_return.TerminalReturnDataset,
    train_indices: np.ndarray,
    *,
    horizon: int,
    seed: int,
    num_train_steps: int,
    batch_size: int,
    critic_lr: float,
    log_interval: int,
    balanced_batches: bool,
) -> tuple[rlt.RLTTwinCritic, list[dict[str, float]]]:
    critic = _init_critic(dataset, horizon=horizon, seed=seed)
    tx = optax.adam(critic_lr)
    graphdef, params = nnx.split(critic)
    opt_state = tx.init(params)
    rng = np.random.default_rng(seed)
    logs: list[dict[str, float]] = []

    @jax.jit
    def train_step(params, opt_state, x, action, target):
        def loss_fn(model_params):
            model = nnx.merge(graphdef, model_params)
            q1, q2 = model(x, action)
            q1_loss = jnp.mean(jnp.square(q1 - target))
            q2_loss = jnp.mean(jnp.square(q2 - target))
            loss = q1_loss + q2_loss
            return loss, {
                "loss": loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q_mean": jnp.mean(jnp.minimum(q1, q2)),
                "target_mean": jnp.mean(target),
            }

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info

    for step in range(1, num_train_steps + 1):
        if balanced_batches:
            indices = _sample_balanced_indices(dataset, train_indices, rng, batch_size)
        else:
            indices = rng.choice(train_indices, size=batch_size, replace=len(train_indices) < batch_size)
        x = rlt.make_state(jnp.asarray(dataset.z_rl[indices]), jnp.asarray(dataset.proprio[indices]))
        action = jnp.asarray(dataset.action[indices])
        target = jnp.asarray(dataset.targets[indices])
        params, opt_state, info = train_step(params, opt_state, x, action, target)

        if step == 1 or step % log_interval == 0 or step == num_train_steps:
            info_np = jax.device_get(info)
            logs.append(
                {
                    "step": float(step),
                    "loss": float(info_np["loss"]),
                    "q1_loss": float(info_np["q1_loss"]),
                    "q2_loss": float(info_np["q2_loss"]),
                    "q_mean": float(info_np["q_mean"]),
                    "target_mean": float(info_np["target_mean"]),
                }
            )
    return nnx.merge(graphdef, params), logs


def _sample_balanced_indices(
    dataset: rlt_terminal_return.TerminalReturnDataset,
    train_indices: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    groups: list[np.ndarray] = []
    for source in sorted(set(dataset.sources[train_indices].tolist())):
        source_mask = dataset.sources[train_indices] == source
        source_indices = train_indices[source_mask]
        for label in (0, 1):
            label_indices = source_indices[dataset.labels[source_indices] == label]
            if len(label_indices):
                groups.append(label_indices)
    if not groups:
        return rng.choice(train_indices, size=batch_size, replace=len(train_indices) < batch_size)
    per_group = batch_size // len(groups)
    remainder = batch_size % len(groups)
    pieces: list[np.ndarray] = []
    for group_index, group in enumerate(groups):
        count = per_group + (1 if group_index < remainder else 0)
        if count > 0:
            pieces.append(rng.choice(group, size=count, replace=len(group) < count))
    sampled = np.concatenate(pieces, axis=0)
    rng.shuffle(sampled)
    return sampled


def _score(
    critic: rlt.RLTTwinCritic,
    dataset: rlt_terminal_return.TerminalReturnDataset,
    indices: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    q_actual = np.empty((len(indices),), dtype=np.float32)
    q_reference = np.empty((len(indices),), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x = rlt.make_state(jnp.asarray(dataset.z_rl[batch_indices]), jnp.asarray(dataset.proprio[batch_indices]))
        actual = jnp.asarray(dataset.action[batch_indices])
        reference = jnp.asarray(dataset.reference_action[batch_indices])
        actual_q = critic.min_q(x, actual)
        reference_q = critic.min_q(x, reference)
        end = start + len(batch_indices)
        q_actual[start:end] = np.asarray(jax.device_get(actual_q), dtype=np.float32)
        q_reference[start:end] = np.asarray(jax.device_get(reference_q), dtype=np.float32)
    return q_actual, q_reference


def _evaluate_split(
    critic: rlt.RLTTwinCritic,
    dataset: rlt_terminal_return.TerminalReturnDataset,
    split: rlt_terminal_return.DatasetSplit,
    *,
    experiment: str,
) -> dict[str, Any]:
    q_actual, q_reference = _score(critic, dataset, split.holdout_indices)
    labels = dataset.labels[split.holdout_indices]
    targets = dataset.targets[split.holdout_indices]
    actual_metrics = rlt_terminal_return.score_metrics(labels, q_actual, targets)
    reference_metrics = rlt_terminal_return.score_metrics(labels, q_reference, targets)
    return {
        "experiment": experiment,
        "split": split.name,
        "split_type": split.split_type,
        "holdout_source": split.holdout_source or "",
        "holdout_auc": actual_metrics["auc"],
        "holdout_q_gap": actual_metrics["q_gap"],
        "holdout_mse": actual_metrics["mse"],
        "reference_auc": reference_metrics["auc"],
        "reference_q_gap": reference_metrics["q_gap"],
        "reference_mse": reference_metrics["mse"],
        "mean_q_actual_minus_reference": float(np.mean(q_actual - q_reference)),
        "success_q_actual_minus_reference": _mean_if_any((q_actual - q_reference)[labels == 1]),
        "failure_q_actual_minus_reference": _mean_if_any((q_actual - q_reference)[labels == 0]),
        "num_success_transitions": actual_metrics["num_success_transitions"],
        "num_failure_transitions": actual_metrics["num_failure_transitions"],
    }


def _per_source_rows(
    critic: rlt.RLTTwinCritic,
    dataset: rlt_terminal_return.TerminalReturnDataset,
    *,
    experiment: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(dataset.labels), dtype=np.int64)
    q_actual, q_reference = _score(critic, dataset, all_indices)
    for source in sorted(set(dataset.sources.tolist())):
        mask = dataset.sources == source
        labels = dataset.labels[mask]
        targets = dataset.targets[mask]
        actual_metrics = rlt_terminal_return.score_metrics(labels, q_actual[mask], targets)
        reference_metrics = rlt_terminal_return.score_metrics(labels, q_reference[mask], targets)
        rows.append(
            {
                "experiment": experiment,
                "source": str(source),
                "auc": actual_metrics["auc"],
                "q_gap": actual_metrics["q_gap"],
                "mse": actual_metrics["mse"],
                "reference_auc": reference_metrics["auc"],
                "reference_q_gap": reference_metrics["q_gap"],
                "num_success_transitions": actual_metrics["num_success_transitions"],
                "num_failure_transitions": actual_metrics["num_failure_transitions"],
            }
        )
    return rows


def _prediction_rows(
    critic: rlt.RLTTwinCritic,
    dataset: rlt_terminal_return.TerminalReturnDataset,
    *,
    experiment: str,
) -> list[dict[str, Any]]:
    indices = np.arange(len(dataset.labels), dtype=np.int64)
    q_actual, q_reference = _score(critic, dataset, indices)
    rows: list[dict[str, Any]] = []
    for idx, base in enumerate(dataset.rows):
        rows.append(
            {
                **base,
                "experiment": experiment,
                "q_actual": float(q_actual[idx]),
                "q_reference": float(q_reference[idx]),
                "q_actual_minus_reference": float(q_actual[idx] - q_reference[idx]),
            }
        )
    return rows


def _plot_mean_curves(rows: list[dict[str, Any]], out_path: pathlib.Path, *, score_key: str) -> None:
    bins = np.linspace(0.0, 1.0, 41)
    centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(10, 5.5))
    for label, name, color in [(1, "success", "#238443"), (0, "failure", "#b2182b")]:
        means = []
        for left, right in zip(bins[:-1], bins[1:], strict=True):
            values = [
                float(row[score_key])
                for row in rows
                if int(row["label_success"]) == label and left <= float(row["progress"]) <= right
            ]
            means.append(float(np.mean(values)) if values else np.nan)
        plt.plot(centers, means, label=name, color=color, linewidth=2.0)
    plt.xlabel("normalized key-region progress")
    plt.ylabel(score_key)
    plt.title(f"Terminal-return critic {score_key} over key-region progress")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output_dir: pathlib.Path,
    *,
    args: Args,
    stats: dict[str, Any],
    metrics: list[dict[str, Any]],
    per_source: list[dict[str, Any]],
) -> None:
    main_leave_one = [
        row for row in metrics if row["experiment"] == "terminal-all" and row["split_type"] == "leave_one_source_out"
    ]
    min_auc = min(float(row["holdout_auc"]) for row in main_leave_one)
    min_gap = min(float(row["holdout_q_gap"]) for row in main_leave_one)
    usable = min_auc >= 0.70 and min_gap >= 0.15
    reason = (
        "leave-one-date AUC/gap passed basic critic gates"
        if usable
        else f"leave-one-date min AUC {min_auc:.3f} or min Q gap {min_gap:.3f} below gate"
    )
    lines = [
        "# RLT Terminal-Return Critic Experiment",
        "",
        "## Setup",
        "",
        f"- manifest: `{args.manifest_path}`",
        f"- output_dir: `{args.output_dir}`",
        f"- target: `success ? gamma^(T-1-t) * terminal_reward : {args.failure_target}`",
        f"- gamma: `{args.gamma}`",
        f"- num_train_steps per split: `{args.num_train_steps}`",
        f"- train_action_horizon: `{args.train_action_horizon}`",
        "",
        "## Dataset",
        "",
        f"- episodes: `{stats['num_episodes']}`",
        f"- transitions: `{stats['num_transitions']}`",
        f"- success/failure episodes: `{stats['num_success_episodes']} / {stats['num_failure_episodes']}`",
        f"- success/failure transitions: `{stats['num_success_transitions']} / {stats['num_failure_transitions']}`",
        "",
        "| source | success episodes | failure episodes | success rate | success transitions | failure transitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, row in stats["by_source"].items():
        lines.append(
            f"| {source} | {row['num_success_episodes']} | {row['num_failure_episodes']} | "
            f"{row['success_rate']:.3f} | {row['num_success_transitions']} | {row['num_failure_transitions']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics Summary",
            "",
            "| experiment | split | holdout source | AUC | Q gap | MSE | ref AUC | ref Q gap | Q(actual-ref) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['holdout_source']} | "
            f"{float(row['holdout_auc']):.4f} | {float(row['holdout_q_gap']):.4f} | {float(row['holdout_mse']):.6f} | "
            f"{float(row['reference_auc']):.4f} | {float(row['reference_q_gap']):.4f} | "
            f"{float(row['mean_q_actual_minus_reference']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Source Final-Critic Diagnostics",
            "",
            "| experiment | source | AUC | Q gap | MSE | ref AUC | ref Q gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_source:
        lines.append(
            f"| {row['experiment']} | {row['source']} | {float(row['auc']):.4f} | {float(row['q_gap']):.4f} | "
            f"{float(row['mse']):.6f} | {float(row['reference_auc']):.4f} | {float(row['reference_q_gap']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            f"- Terminal-return critic usable for actor training now: `{'YES' if usable else 'NO'}`",
            f"- Reason: {reason}",
            "- Rule: require terminal-all leave-one-date min AUC >= 0.70 and min Q gap >= 0.15.",
            "",
            "## Interpretation",
            "",
            "- If episode-random is strong but leave-one-date is weak, critic is still learning date-specific shortcuts.",
            "- If q_reference is close to q_actual, the critic may rank states but still be weakly action-sensitive.",
            "- Actor training should only start after leave-one-date and action-sensitivity diagnostics are acceptable.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_critic(critic: rlt.RLTTwinCritic, output_dir: pathlib.Path, *, args: Args, dataset: rlt_terminal_return.TerminalReturnDataset) -> None:
    critic_state = nnx.state(critic).to_pure_dict()
    (output_dir / "critic.msgpack").write_bytes(serialization.to_bytes(critic_state))
    config = critic.q1.config
    metadata = {
        "type": "rlt_terminal_return_critic",
        "step": args.num_train_steps,
        "rlt_config": dataclasses.asdict(config),
        "manifest_path": str(args.manifest_path),
        "num_transitions": int(len(dataset.labels)),
        "num_episodes": int(len(set(dataset.episode_ids.tolist()))),
        "target": "success ? gamma^(T-1-t) * terminal_reward : failure_target",
        "gamma": args.gamma,
        "failure_target": args.failure_target,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean_if_any(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def main(args: Args) -> None:
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []
    training_logs: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    main_stats: dict[str, Any] | None = None

    for experiment in EXPERIMENTS:
        critical_ratio = args.critical_ratio if experiment.critical_ratio is not None else None
        dataset = rlt_terminal_return.load_terminal_return_dataset(
            args.manifest_path,
            gamma=args.gamma,
            critical_ratio=critical_ratio,
            failure_target=args.failure_target,
        )
        dataset = _slice_horizon(dataset, args.train_action_horizon)
        if experiment.name == "terminal-all":
            main_stats = rlt_terminal_return.dataset_stats(dataset)
        splits = [
            rlt_terminal_return.episode_random_split(dataset, holdout_ratio=args.holdout_ratio, seed=args.seed),
            *rlt_terminal_return.leave_one_source_out_splits(dataset),
            *rlt_terminal_return.intra_source_episode_splits(
                dataset,
                holdout_ratio=args.holdout_ratio,
                seed=args.seed,
            ),
        ]
        for split_index, split in enumerate(splits):
            critic, logs = _train_critic(
                dataset,
                split.train_indices,
                horizon=args.train_action_horizon,
                seed=args.seed + split_index,
                num_train_steps=args.num_train_steps,
                batch_size=args.batch_size,
                critic_lr=args.critic_lr,
                log_interval=args.log_interval,
                balanced_batches=args.balanced_batches,
            )
            metrics.append(_evaluate_split(critic, dataset, split, experiment=experiment.name))
            training_logs.extend({**row, "experiment": experiment.name, "split": split.name} for row in logs)

        final_critic, logs = _train_critic(
            dataset,
            np.arange(len(dataset.labels), dtype=np.int64),
            horizon=args.train_action_horizon,
            seed=args.seed + 10_000,
            num_train_steps=args.num_train_steps,
            batch_size=args.batch_size,
            critic_lr=args.critic_lr,
            log_interval=args.log_interval,
            balanced_batches=args.balanced_batches,
        )
        training_logs.extend({**row, "experiment": experiment.name, "split": "all_data_final"} for row in logs)
        per_source.extend(_per_source_rows(final_critic, dataset, experiment=experiment.name))
        rows = _prediction_rows(final_critic, dataset, experiment=experiment.name)
        prediction_rows.extend(rows)
        _plot_mean_curves(rows, args.output_dir / f"{experiment.name}_q_actual_curve.png", score_key="q_actual")
        _plot_mean_curves(rows, args.output_dir / f"{experiment.name}_q_reference_curve.png", score_key="q_reference")
        if experiment.name == "terminal-all":
            _save_critic(final_critic, args.output_dir, args=args, dataset=dataset)

    if main_stats is None:
        raise RuntimeError("terminal-all experiment did not run")

    _write_csv(args.output_dir / "metrics_summary.csv", metrics)
    _write_csv(args.output_dir / "per_source_metrics.csv", per_source)
    _write_csv(args.output_dir / "training_log.csv", training_logs)
    _write_csv(args.output_dir / "per_transition_predictions.csv", prediction_rows)
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(main_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(args.output_dir, args=args, stats=main_stats, metrics=metrics, per_source=per_source)
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))

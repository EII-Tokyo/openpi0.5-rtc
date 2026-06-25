#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tyro

from openpi.models.rlt_discriminator import RLTSuccessDiscriminator
from openpi.training import rlt_discriminator_data as disc_data


@dataclasses.dataclass
class Args:
    manifest_path: pathlib.Path = pathlib.Path("local_rlt_manifests/trainable_clean_committed_20260623_time_sorted.jsonl")
    output_dir: pathlib.Path = pathlib.Path("local_rlt_runs/rlt_success_discriminator_simple_20260624")
    seed: int = 0
    holdout_ratio: float = 0.2
    critical_ratio: float = 0.4
    num_epochs: int = 25
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 512
    device: str = "auto"
    experiment_preset: str = "simple"
    overwrite: bool = False


SIMPLE_EXPERIMENTS = (
    {"name": "D-all", "feature_variant": "state_action", "critical": False},
    {"name": "D-critical", "feature_variant": "state_action", "critical": True},
    {"name": "D-state-only", "feature_variant": "state_only", "critical": False},
    {"name": "D-shuffled-action", "feature_variant": "shuffled_action", "critical": False},
)

COMPLEX_EXPERIMENTS = (
    {"name": "D-complex", "feature_variant": "state_action_next_delta", "critical": False},
    {"name": "D-complex-critical", "feature_variant": "state_action_next_delta", "critical": True},
    {"name": "D-state-action", "feature_variant": "state_action", "critical": False},
    {"name": "D-state-next-only", "feature_variant": "state_next_only", "critical": False},
    {"name": "D-shuffled-action-complex", "feature_variant": "shuffled_action_next_delta", "critical": False},
)


def main(args: Args) -> None:
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    full_dataset = disc_data.load_transition_dataset(args.manifest_path)
    critical_dataset = disc_data.load_transition_dataset(args.manifest_path, critical_ratio=args.critical_ratio)
    experiments = _experiments_for_preset(args.experiment_preset)
    dataset_stats = disc_data.dataset_stats(full_dataset)
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "split_config.json").write_text(
        json.dumps(
            {
                "holdout_ratio": args.holdout_ratio,
                "seed": args.seed,
                "critical_ratio": args.critical_ratio,
                "split_types": ["episode_random", "leave_one_source_out"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_metrics: list[dict[str, Any]] = []
    all_per_source: list[dict[str, Any]] = []
    all_training_logs: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_model_state: dict[str, Any] | None = None
    best_time_rows: list[dict[str, Any]] = []
    best_calibration_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        dataset = critical_dataset if experiment["critical"] else full_dataset
        split_defs = [disc_data.episode_random_split(dataset, holdout_ratio=args.holdout_ratio, seed=args.seed)]
        split_defs.extend(disc_data.leave_one_source_out_splits(dataset))
        features = disc_data.build_features(
            dataset,
            str(experiment["feature_variant"]),
            rng=np.random.default_rng(args.seed),
        )
        for split in split_defs:
            result = _train_one(
                features,
                dataset.labels,
                split,
                input_dim=features.shape[1],
                args=args,
                device=device,
                experiment_name=str(experiment["name"]),
            )
            all_training_logs.extend(result["training_log"])
            train_probs = result["train_probs"]
            holdout_probs = result["holdout_probs"]
            train_metrics = disc_data.binary_classification_metrics(dataset.labels[split.train_indices], train_probs)
            holdout_metrics = disc_data.binary_classification_metrics(dataset.labels[split.holdout_indices], holdout_probs)
            metric_row = {
                "experiment": experiment["name"],
                "feature_variant": experiment["feature_variant"],
                "split": split.name,
                "split_type": split.split_type,
                "holdout_source": split.holdout_source or "",
                "input_dim": features.shape[1],
                "train_loss": result["train_loss"],
                "holdout_loss": result["holdout_loss"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
            }
            metric_row["warning"] = _warning_for_metric(metric_row)
            all_metrics.append(metric_row)
            for row in disc_data.per_source_metrics(dataset, split.holdout_indices, holdout_probs):
                all_per_source.append({"experiment": experiment["name"], "split": split.name, **row})

            record_score = _selection_score(metric_row)
            if best_record is None or record_score > _selection_score(best_record):
                best_record = metric_row
                best_model_state = result["model_state"]
                best_time_rows = _time_curve_rows(dataset, split.holdout_indices, holdout_probs, experiment["name"], split.name)
                best_calibration_rows = [
                    {"experiment": experiment["name"], "split": split.name, **row}
                    for row in disc_data.calibration_rows(dataset.labels[split.holdout_indices], holdout_probs)
                ]

    disc_data.write_csv(args.output_dir / "metrics_summary.csv", all_metrics)
    disc_data.write_csv(args.output_dir / "per_source_metrics.csv", all_per_source)
    disc_data.write_csv(args.output_dir / "training_log.csv", all_training_logs)
    disc_data.write_csv(args.output_dir / "episode_time_curves.csv", best_time_rows)
    disc_data.write_csv(args.output_dir / "calibration.csv", best_calibration_rows)
    if best_model_state is not None:
        torch.save(best_model_state, args.output_dir / "best_model.pt")
    _plot_time_curves(best_time_rows, args.output_dir / "episode_time_curves.png")
    _plot_calibration(best_calibration_rows, args.output_dir / "calibration.png")
    _write_report(args.output_dir / "report.md", args, dataset_stats, all_metrics, all_per_source, best_record)


def _train_one(
    features: np.ndarray,
    labels: np.ndarray,
    split: disc_data.DatasetSplit,
    *,
    input_dim: int,
    args: Args,
    device: torch.device,
    experiment_name: str,
) -> dict[str, Any]:
    model = RLTSuccessDiscriminator(input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_y = labels[split.train_indices].astype(np.float32)
    pos = float(np.sum(train_y == 1.0))
    neg = float(np.sum(train_y == 0.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features[split.train_indices]).float(),
            torch.from_numpy(train_y).float(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    log_rows = []
    best_state = None
    best_auc = -1.0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_probs, train_loss = _predict_loss(model, loss_fn, features[split.train_indices], labels[split.train_indices], device)
        holdout_probs, holdout_loss = _predict_loss(model, loss_fn, features[split.holdout_indices], labels[split.holdout_indices], device)
        holdout_auc = disc_data.binary_classification_metrics(labels[split.holdout_indices], holdout_probs)["auc"]
        log_rows.append(
            {
                "experiment": experiment_name,
                "split": split.name,
                "epoch": epoch,
                "batch_train_loss": float(np.mean(losses)) if losses else math.nan,
                "train_loss": train_loss,
                "holdout_loss": holdout_loss,
                "holdout_auc": holdout_auc,
            }
        )
        if np.isfinite(holdout_auc) and holdout_auc > best_auc:
            best_auc = float(holdout_auc)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_probs, train_loss = _predict_loss(model, loss_fn, features[split.train_indices], labels[split.train_indices], device)
    holdout_probs, holdout_loss = _predict_loss(model, loss_fn, features[split.holdout_indices], labels[split.holdout_indices], device)
    return {
        "train_probs": train_probs,
        "holdout_probs": holdout_probs,
        "train_loss": train_loss,
        "holdout_loss": holdout_loss,
        "training_log": log_rows,
        "model_state": {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_dim": input_dim,
            "experiment": experiment_name,
            "split": split.name,
            "created_at": time.time(),
        },
    }


def _predict_loss(
    model: RLTSuccessDiscriminator,
    loss_fn: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    xs = torch.from_numpy(features).float().to(device)
    ys = torch.from_numpy(labels.astype(np.float32)).float().to(device)
    with torch.no_grad():
        logits = model(xs)
        loss = loss_fn(logits, ys)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs.astype(np.float32), float(loss.detach().cpu())


def _time_curve_rows(dataset: disc_data.TransitionDataset, indices: np.ndarray, probs: np.ndarray, experiment: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for local_i, index in enumerate(indices):
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "episode_id": str(dataset.episode_ids[index]),
                "source": str(dataset.sources[index]),
                "transition_idx": int(dataset.transition_indices[index]),
                "num_transitions": int(dataset.num_transitions[index]),
                "progress": float(dataset.progress[index]),
                "label_success": int(dataset.labels[index]),
                "D_prob": float(probs[local_i]),
            }
        )
    return rows


def _warning_for_metric(row: dict[str, Any]) -> str:
    warnings = []
    if float(row["holdout_auc"]) < 0.70:
        warnings.append("holdout_auc<0.70")
    if float(row["holdout_D_gap"]) < 0.15:
        warnings.append("holdout_D_gap<0.15")
    if float(row["holdout_mean_D_failure"]) > float(row["holdout_mean_D_success"]):
        warnings.append("failure_mean>success_mean")
    return ";".join(warnings)


def _selection_score(row: dict[str, Any] | None) -> tuple[float, float, float]:
    if row is None:
        return (-1.0, -1.0, 0.0)
    return (
        _finite_or(row.get("holdout_auc"), -1.0),
        _finite_or(row.get("holdout_D_gap"), -1.0),
        -_finite_or(row.get("holdout_loss"), 1e9),
    )


def _finite_or(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


def _plot_time_curves(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not rows:
        return
    bins = np.linspace(0.0, 1.0, 21)
    centers = (bins[:-1] + bins[1:]) / 2.0
    plt.figure(figsize=(8, 5))
    for label, name, color in [(1, "success", "#238443"), (0, "failure", "#b2182b")]:
        means = []
        for left, right in zip(bins[:-1], bins[1:], strict=True):
            vals = [float(row["D_prob"]) for row in rows if int(row["label_success"]) == label and left <= float(row["progress"]) <= right]
            means.append(np.nan if not vals else float(np.mean(vals)))
        plt.plot(centers, means, label=name, color=color, linewidth=2)
    plt.xlabel("progress")
    plt.ylabel("D_prob")
    plt.title("Holdout D probability over key-region progress")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_calibration(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not rows:
        return
    x = [float(row["mean_pred_prob"]) for row in rows if np.isfinite(float(row["mean_pred_prob"]))]
    y = [float(row["actual_success_rate"]) for row in rows if np.isfinite(float(row["mean_pred_prob"]))]
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], color="#666", linestyle="--", linewidth=1)
    plt.scatter(x, y, color="#2166ac")
    plt.xlabel("mean predicted probability")
    plt.ylabel("actual success rate")
    plt.title("D calibration")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(
    path: pathlib.Path,
    args: Args,
    dataset_stats: dict[str, Any],
    metrics: list[dict[str, Any]],
    per_source: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> None:
    episode_random = [row for row in metrics if row["split_type"] == "episode_random"]
    leave_one = [row for row in metrics if row["split_type"] == "leave_one_source_out"]
    del episode_random
    main_name = "D-complex" if any(row["experiment"] == "D-complex" for row in metrics) else "D-all"
    state_baseline_name = "D-state-next-only" if main_name == "D-complex" else "D-state-only"
    shuffle_name = "D-shuffled-action-complex" if main_name == "D-complex" else "D-shuffled-action"
    d_all_leave_one = [row for row in leave_one if row["experiment"] == main_name]
    d_state_leave_one = [row for row in leave_one if row["experiment"] == state_baseline_name]
    d_shuffle_leave_one = [row for row in leave_one if row["experiment"] == shuffle_name]
    gate = _gate_decision(d_all_leave_one, d_state_leave_one, d_shuffle_leave_one)
    input_main = (
        "z_rl + proprio + action_chunk + next_z_rl + next_proprio + delta_state"
        if main_name == "D-complex"
        else "z_rl + proprio + action_chunk"
    )
    lines = [
        f"# RLT Success/Failure Discriminator {args.experiment_preset.title()} Experiment",
        "",
        "## Setup",
        "",
        f"- manifest: `{args.manifest_path}`",
        f"- output_dir: `{args.output_dir}`",
        f"- experiment_preset: `{args.experiment_preset}`",
        f"- input main: `{input_main}`",
        f"- critical_ratio: `{args.critical_ratio}`",
        f"- epochs: `{args.num_epochs}`",
        "",
        "## Dataset",
        "",
        f"- episodes: `{dataset_stats['num_episodes']}`",
        f"- transitions: `{dataset_stats['num_transitions']}`",
        f"- success/failure episodes: `{dataset_stats['num_success_episodes']} / {dataset_stats['num_failure_episodes']}`",
        f"- success/failure transitions: `{dataset_stats['num_success_transitions']} / {dataset_stats['num_failure_transitions']}`",
        "",
        "| source | success episodes | failure episodes | success rate | success transitions | failure transitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, item in dataset_stats["by_source"].items():
        lines.append(
            f"| {source} | {item['num_success_episodes']} | {item['num_failure_episodes']} | "
            f"{item['success_rate']:.3f} | {item['num_success_transitions']} | {item['num_failure_transitions']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics Summary",
            "",
            "| experiment | split | holdout source | AUC | D gap | balanced acc | warning |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['holdout_source']} | "
            f"{float(row['holdout_auc']):.4f} | {float(row['holdout_D_gap']):.4f} | "
            f"{float(row['holdout_balanced_accuracy']):.4f} | {row['warning']} |"
        )
    if best is not None:
        lines.extend(
            [
                "",
                "## Best Holdout Model",
                "",
                f"- experiment: `{best['experiment']}`",
                f"- split: `{best['split']}`",
                f"- AUC: `{float(best['holdout_auc']):.4f}`",
                f"- D_gap: `{float(best['holdout_D_gap']):.4f}`",
                f"- warning: `{best['warning']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-Source Warnings",
            "",
            "| experiment | split | source | AUC | D gap | failure > success |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in per_source:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['source']} | "
            f"{float(row['auc']):.4f} | {float(row['D_gap']):.4f} | {row['warning']} |"
        )
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            f"- Main model checked: `{main_name}`",
            f"- State baseline checked: `{state_baseline_name}`",
            f"- Shuffled-action baseline checked: `{shuffle_name}`",
            f"- D reward usable for critic training: `{'YES' if gate['usable'] else 'NO'}`",
            f"- Reason: {gate['reason']}",
            "- Recommended reward: `none`" if not gate["usable"] else "- Recommended reward: `conservative r_D = 2 * D_prob - 1`",
            "- Recommended beta: `none`" if not gate["usable"] else "- Recommended beta: `0.1 first`",
            "- Whether actor training should start now: `NO`",
            "",
            "Conclusion:",
            f"- D reward usable for critic training: {'YES' if gate['usable'] else 'NO'}",
            f"- Reason: {gate['reason']}",
            f"- Recommended reward: {'conservative' if gate['usable'] else 'none'}",
            f"- Recommended beta: {'0.1' if gate['usable'] else 'none'}",
            "- Whether actor training should start now: NO",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _gate_decision(d_all: list[dict[str, Any]], d_state: list[dict[str, Any]], d_shuffle: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    finite_all = [row for row in d_all if np.isfinite(float(row["holdout_auc"]))]
    if not finite_all:
        failures.append("no finite leave-one-date main-model AUC")
    else:
        min_auc = min(float(row["holdout_auc"]) for row in finite_all)
        min_gap = min(float(row["holdout_D_gap"]) for row in finite_all)
        if min_auc < 0.70:
            failures.append(f"leave-one-date main-model min AUC {min_auc:.3f} < 0.70")
        if min_gap < 0.15:
            failures.append(f"leave-one-date main-model min D_gap {min_gap:.3f} < 0.15")
    if d_all and d_state:
        all_mean = float(np.nanmean([float(row["holdout_auc"]) for row in d_all]))
        state_mean = float(np.nanmean([float(row["holdout_auc"]) for row in d_state]))
        if all_mean - state_mean < 0.05:
            failures.append(f"main-model gain over state baseline is small ({all_mean - state_mean:.3f})")
    if d_all and d_shuffle:
        all_mean = float(np.nanmean([float(row["holdout_auc"]) for row in d_all]))
        shuffle_mean = float(np.nanmean([float(row["holdout_auc"]) for row in d_shuffle]))
        if all_mean - shuffle_mean < 0.05:
            failures.append(f"correct action gain over shuffled-action is small ({all_mean - shuffle_mean:.3f})")
    return {"usable": not failures, "reason": "; ".join(failures) if failures else "passed discriminator gates"}


def _experiments_for_preset(preset: str) -> tuple[dict[str, Any], ...]:
    if preset == "simple":
        return SIMPLE_EXPERIMENTS
    if preset == "complex":
        return COMPLEX_EXPERIMENTS
    raise ValueError(f"Unsupported experiment_preset={preset!r}; expected 'simple' or 'complex'")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main(tyro.cli(Args))

#!/usr/bin/env python3
"""Train critic A/B on sidecar z_rl vs VLA-token-derived z_rl."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from openpi.training import rlt_terminal_return
from scripts import train_rlt_terminal_return_critic as terminal_critic
from scripts.compare_vla_same_forward_vs_sidecar_tokens import SelectedRow, _load_selected_rows


DEFAULT_INPUT_DIR = Path("local_rlt_runs/vla_same_forward_vs_sidecar_tokens_20260704_zrl")
DEFAULT_OUTPUT_DIR = Path("local_rlt_runs/vla_token_z_critic_ablation_20260704")
DEFAULT_KB_DIR = Path(
    "/home/eii/Documents/Notes/openpi0.5-rtc-reward-learning/70_Experiments/"
    "2026-07-04_vla_same_forward_2view_vs_sidecar_rltoken"
)


@dataclasses.dataclass(frozen=True)
class BuiltReplay:
    sidecar_manifest: Path
    vla_token_manifest: Path
    split_json: Path
    num_episodes: int
    num_transitions: int
    num_success_episodes: int
    num_failure_episodes: int


def _load_feature_file(path: Path, *, require_z: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    if require_z and "z_rl" not in payload:
        raise ValueError(f"{path} does not contain z_rl")
    return payload


def _group_selected_rows(rows: list[SelectedRow]) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, row in enumerate(rows):
        if row.key_region_id not in groups:
            groups[row.key_region_id] = []
            order.append(row.key_region_id)
        groups[row.key_region_id].append(idx)
    return [(key, groups[key]) for key in order]


def _episode_success(rows: list[SelectedRow], indices: list[int]) -> int:
    values = {int(rows[index].reward >= 0.5) for index in indices}
    if len(values) != 1:
        raise ValueError(f"Mixed rewards inside one selected key region: {values}")
    return values.pop()


def _build_split(rows: list[SelectedRow], *, holdout_ratio: float, seed: int) -> dict[str, Any]:
    groups = _group_selected_rows(rows)
    rng = np.random.default_rng(seed)
    holdout: set[str] = set()
    for label in (0, 1):
        label_groups = [key for key, indices in groups if _episode_success(rows, indices) == label]
        rng.shuffle(label_groups)
        if len(label_groups) < 2:
            continue
        count = max(1, round(len(label_groups) * holdout_ratio))
        count = min(count, len(label_groups) - 1)
        holdout.update(label_groups[:count])
    if not holdout:
        keys = [key for key, _ in groups]
        rng.shuffle(keys)
        count = max(1, round(len(keys) * holdout_ratio))
        count = min(count, len(keys) - 1)
        holdout.update(keys[:count])
    train = [key for key, _ in groups if key not in holdout]
    holdout_list = [key for key, _ in groups if key in holdout]
    return {
        "train_key_region_ids": train,
        "holdout_key_region_ids": holdout_list,
        "holdout_ratio": holdout_ratio,
        "seed": seed,
    }


def _write_one_variant(
    *,
    name: str,
    rows: list[SelectedRow],
    features: dict[str, np.ndarray],
    output_root: Path,
) -> Path:
    groups = _group_selected_rows(rows)
    manifest_rows: list[dict[str, Any]] = []
    for key_region_id, indices in groups:
        first = rows[indices[0]]
        source_shard = first.shard_path
        with np.load(source_shard, allow_pickle=False) as source:
            proprio = np.stack([np.asarray(source["proprio"][rows[index].local_row], dtype=np.float32) for index in indices])
        z_rl = np.stack([np.asarray(features["z_rl"][index], dtype=np.float32) for index in indices])
        action = np.stack([np.asarray(features["action"][index], dtype=np.float32) for index in indices])
        reference_action = np.stack([np.asarray(features["reference_action"][index], dtype=np.float32) for index in indices])
        reward = float(first.reward)
        horizon = int(action.shape[1])
        reward_seq = np.zeros((len(indices), horizon), dtype=np.float32)
        done = np.zeros((len(indices),), dtype=np.bool_)
        done[-1] = True
        if reward > 0.0:
            reward_seq[-1, -1] = reward
        shard_dir = output_root / name / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"key_region_{key_region_id}.npz"
        manifest = {
            "key_region_id": key_region_id,
            "source_shard_path": str(source_shard),
            "z_source": name,
            "reward": int(reward > 0.0),
            "selected_local_rows": [int(rows[index].local_row) for index in indices],
            "selected_frame_indices": [int(rows[index].frame_index) for index in indices],
        }
        np.savez_compressed(
            shard_path,
            z_rl=z_rl,
            proprio=proprio,
            action=action,
            reference_action=reference_action,
            reward_seq=reward_seq,
            next_z_rl=np.zeros_like(z_rl),
            next_proprio=np.zeros_like(proprio),
            next_reference_action=reference_action,
            done=done,
            manifest=np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        )
        manifest_rows.append(
            {
                "shard_path": str(shard_path.resolve()),
                "batch": "vla_token_z_ablation",
                "key_region_id": key_region_id,
                "reward": int(reward > 0.0),
            }
        )
    manifest_path = output_root / name / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows) + "\n", encoding="utf-8")
    return manifest_path


def build_replay(
    *,
    selected_csv: Path,
    sidecar_features: Path,
    vla_z_features: Path,
    output_dir: Path,
    holdout_ratio: float,
    seed: int,
) -> BuiltReplay:
    rows = _load_selected_rows(selected_csv)
    sidecar = _load_feature_file(sidecar_features, require_z=True)
    vla_z = _load_feature_file(vla_z_features, require_z=True)
    sidecar_ids = [str(x) for x in sidecar["row_ids"]]
    vla_ids = [str(x) for x in vla_z["row_ids"]]
    selected_ids = [row.row_id for row in rows]
    if sidecar_ids != selected_ids or vla_ids != selected_ids:
        raise ValueError("Selected rows and feature rows are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_manifest = _write_one_variant(name="sidecar_z", rows=rows, features=sidecar, output_root=output_dir)
    vla_manifest = _write_one_variant(name="vla_token_z", rows=rows, features=vla_z, output_root=output_dir)
    split = _build_split(rows, holdout_ratio=holdout_ratio, seed=seed)
    split_path = output_dir / "fixed_split.json"
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = _group_selected_rows(rows)
    return BuiltReplay(
        sidecar_manifest=sidecar_manifest,
        vla_token_manifest=vla_manifest,
        split_json=split_path,
        num_episodes=len(groups),
        num_transitions=len(rows),
        num_success_episodes=sum(_episode_success(rows, indices) == 1 for _, indices in groups),
        num_failure_episodes=sum(_episode_success(rows, indices) == 0 for _, indices in groups),
    )


def _split_indices(dataset: rlt_terminal_return.TerminalReturnDataset, split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_eps = set(split["train_key_region_ids"])
    holdout_eps = set(split["holdout_key_region_ids"])
    train = np.asarray([str(ep).replace("key_region_", "") in train_eps for ep in dataset.episode_ids], dtype=np.bool_)
    holdout = np.asarray([str(ep).replace("key_region_", "") in holdout_eps for ep in dataset.episode_ids], dtype=np.bool_)
    if not np.any(train) or not np.any(holdout):
        raise ValueError("Empty train or holdout split")
    return np.flatnonzero(train), np.flatnonzero(holdout)


def train_and_eval(
    *,
    manifest_path: Path,
    split_json: Path,
    output_dir: Path,
    num_train_steps: int,
    batch_size: int,
    seed: int,
    gamma: float,
    critic_lr: float,
    train_action_horizon: int,
) -> dict[str, Any]:
    dataset = rlt_terminal_return.load_terminal_return_dataset(manifest_path, gamma=gamma)
    dataset = terminal_critic._slice_horizon(dataset, train_action_horizon)
    split = json.loads(split_json.read_text(encoding="utf-8"))
    train_indices, holdout_indices = _split_indices(dataset, split)
    critic, logs = terminal_critic._train_critic(
        dataset,
        train_indices,
        horizon=train_action_horizon,
        seed=seed,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        critic_lr=critic_lr,
        log_interval=max(1, num_train_steps // 10),
        balanced_batches=True,
    )
    q_holdout, q_ref_holdout = terminal_critic._score(critic, dataset, holdout_indices)
    holdout_labels = dataset.labels[holdout_indices]
    holdout_targets = dataset.targets[holdout_indices]
    q_train, _ = terminal_critic._score(critic, dataset, train_indices)
    train_metrics = rlt_terminal_return.score_metrics(dataset.labels[train_indices], q_train, dataset.targets[train_indices])
    holdout_metrics = rlt_terminal_return.score_metrics(holdout_labels, q_holdout, holdout_targets)
    reference_metrics = rlt_terminal_return.score_metrics(holdout_labels, q_ref_holdout, holdout_targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)
    result = {
        "manifest_path": str(manifest_path),
        "num_train_steps": num_train_steps,
        "batch_size": batch_size,
        "seed": seed,
        "train_transitions": int(len(train_indices)),
        "holdout_transitions": int(len(holdout_indices)),
        "train_auc": train_metrics["auc"],
        "train_q_gap": train_metrics["q_gap"],
        "holdout_auc": holdout_metrics["auc"],
        "holdout_q_gap": holdout_metrics["q_gap"],
        "holdout_q_success_mean": holdout_metrics["q_success_mean"],
        "holdout_q_failure_mean": holdout_metrics["q_failure_mean"],
        "holdout_mse": holdout_metrics["mse"],
        "reference_holdout_auc": reference_metrics["auc"],
        "reference_holdout_q_gap": reference_metrics["q_gap"],
        "final_train_loss": float(logs[-1]["loss"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _write_report(output_dir: Path, sidecar: dict[str, Any], vla: dict[str, Any], built: BuiltReplay) -> None:
    winner = "vla_token_z" if (vla["holdout_auc"], vla["holdout_q_gap"]) > (sidecar["holdout_auc"], sidecar["holdout_q_gap"]) else "sidecar_z"
    report = f"""# Critic A/B：sidecar z_rl vs VLA-token z_rl

## 数据

- key region episodes: {built.num_episodes}
- transitions: {built.num_transitions}
- success / failure episodes: {built.num_success_episodes} / {built.num_failure_episodes}
- train / holdout transitions: {sidecar['train_transitions']} / {sidecar['holdout_transitions']}

## 结果

| z_rl 来源 | holdout AUC | holdout q_gap | success_q | failure_q | holdout MSE | train AUC | train q_gap | final train loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sidecar z_rl | {sidecar['holdout_auc']:.4f} | {sidecar['holdout_q_gap']:.6f} | {sidecar['holdout_q_success_mean']:.6f} | {sidecar['holdout_q_failure_mean']:.6f} | {sidecar['holdout_mse']:.6f} | {sidecar['train_auc']:.4f} | {sidecar['train_q_gap']:.6f} | {sidecar['final_train_loss']:.6f} |
| VLA token -> RLToken encoder | {vla['holdout_auc']:.4f} | {vla['holdout_q_gap']:.6f} | {vla['holdout_q_success_mean']:.6f} | {vla['holdout_q_failure_mean']:.6f} | {vla['holdout_mse']:.6f} | {vla['train_auc']:.4f} | {vla['train_q_gap']:.6f} | {vla['final_train_loss']:.6f} |

## 判断

本实验只改变 `z_rl` 来源，保持 replay、action、proprio、reward、split、critic 结构和训练步数一致。

按 holdout AUC 优先、q_gap 次优的规则，本次更好的结果是：`{winner}`。

如果 `VLA token -> RLToken encoder` 没有同时提高 holdout AUC 和 q_gap，就不能据此推进 runtime 架构大改。若它提高 reference-action probe 但 critic holdout 不提高，说明该表示更贴近 VLA 动作，但不一定更适合用当前稀疏 reward 训练 critic。
"""
    (output_dir / "critic_ablation_report.md").write_text(report, encoding="utf-8")


def _copy_report_to_kb(output_dir: Path, kb_dir: Path) -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    for name in ("critic_ablation_report.md", "summary.json"):
        source = output_dir / name
        if source.exists():
            target = kb_dir / f"critic_ablation_{name}"
            target.write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--num-train-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--holdout-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--train-action-horizon", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    built = build_replay(
        selected_csv=Path("local_rlt_runs/vla_same_forward_vs_sidecar_tokens_20260704_full/selected_samples.csv"),
        sidecar_features=args.input_dir / "sidecar_features.npz",
        vla_z_features=args.input_dir / "vla_same_forward_token_z_features.npz",
        output_dir=args.output_dir / "replay",
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    sidecar_metrics = train_and_eval(
        manifest_path=built.sidecar_manifest,
        split_json=built.split_json,
        output_dir=args.output_dir / "sidecar_z",
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        gamma=args.gamma,
        critic_lr=args.critic_lr,
        train_action_horizon=args.train_action_horizon,
    )
    jax.clear_caches()
    vla_metrics = train_and_eval(
        manifest_path=built.vla_token_manifest,
        split_json=built.split_json,
        output_dir=args.output_dir / "vla_token_z",
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        gamma=args.gamma,
        critic_lr=args.critic_lr,
        train_action_horizon=args.train_action_horizon,
    )
    summary = {
        "built_replay": dataclasses.asdict(built),
        "sidecar_z": sidecar_metrics,
        "vla_token_z": vla_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_report(args.output_dir, sidecar_metrics, vla_metrics, built)
    _copy_report_to_kb(args.output_dir, args.kb_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

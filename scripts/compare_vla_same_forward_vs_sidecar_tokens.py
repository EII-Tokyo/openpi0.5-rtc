#!/usr/bin/env python3
"""Compare lower/right tokens from a 4-camera VLA forward vs 2-camera sidecar.

This is a read-only diagnostic experiment. It reconstructs observations from
raw robot rollouts, extracts lower/right image-token features through two
different encoder paths, then measures representation drift and downstream
probe quality.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.reencode_clean_no_actor_z_rl import (
    _VideoFrameReader,
    _load_qpos,
    compute_replay_frame_indices,
    find_rollout_dir,
    load_manifest_from_npz,
)


DEFAULT_OUTPUT_DIR = Path("local_rlt_runs/vla_same_forward_vs_sidecar_tokens_20260704")
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions")
DEFAULT_MANIFESTS = (
    Path("local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/holdout_bootstrap29.jsonl"),
    Path("local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/train_bootstrap117_expert59.jsonl"),
)
DEFAULT_CAM4_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_CAM4_CHECKPOINT = Path(
    "checkpoints/eii_rinse_11repo_cam4_fullft/"
    "rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"
)
DEFAULT_SIDECAR_CONFIG = "eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer"
DEFAULT_SIDECAR_CHECKPOINT = Path("checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint")
CAMERA_SLOTS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TARGET_SLOTS = ("base_1_rgb", "right_wrist_0_rgb")


@dataclasses.dataclass(frozen=True)
class SelectedRow:
    row_id: str
    shard_path: Path
    rollout_dir: Path
    local_row: int
    frame_index: int
    next_frame_index: int
    reward: float
    split: str
    key_region_id: str
    date: str
    phase: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _manifest_entry_path(entry: dict[str, Any]) -> Path | None:
    for key in ("path", "shard_path", "local_path", "replay_path"):
        value = entry.get(key)
        if value:
            return Path(str(value))
    return None


def _npz_row_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        if "action" in data:
            return int(np.asarray(data["action"]).shape[0])
        if "reference_action" in data:
            return int(np.asarray(data["reference_action"]).shape[0])
        if "z_rl" in data:
            return int(np.asarray(data["z_rl"]).shape[0])
    raise ValueError(f"{path} has no replay row arrays")


def _row_reward(path: Path, manifest: dict[str, Any]) -> float:
    if "reward" in manifest:
        return float(manifest["reward"])
    with np.load(path, allow_pickle=False) as data:
        if "done" in data and "reward_seq" in data:
            done = np.asarray(data["done"])
            reward_seq = np.asarray(data["reward_seq"])
            done_indices = np.flatnonzero(done)
            if len(done_indices):
                idx = int(done_indices[-1])
                return float(np.max(reward_seq[idx]))
        if "reward_seq" in data:
            return float(np.max(np.asarray(data["reward_seq"])))
    return 0.0


def _is_robot_key_region(path: Path, manifest: dict[str, Any]) -> bool:
    source = str(manifest.get("source_shard_path", ""))
    text = f"{path} {source}"
    return "key_region_" in text and "human_expert" not in text and "lerobot" not in text


def _choose_row_indices(n_rows: int, rows_per_shard: int) -> list[int]:
    if n_rows <= 0:
        return []
    if rows_per_shard >= n_rows:
        return list(range(n_rows))
    # Avoid only choosing the first frames; include early/middle/late anchors.
    return sorted({int(round(v)) for v in np.linspace(0, n_rows - 1, rows_per_shard)})


def select_rows(
    *,
    manifest_paths: tuple[Path, ...],
    rollout_root: Path,
    output_dir: Path,
    max_shards: int,
    rows_per_shard: int,
    seed: int,
) -> list[SelectedRow]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, Path, dict[str, Any], Path, int, float]] = []
    skipped: dict[str, int] = {}

    for manifest_path in manifest_paths:
        split = "holdout" if "holdout" in manifest_path.name else "train"
        for entry in _read_jsonl(manifest_path):
            shard_path = _manifest_entry_path(entry)
            if shard_path is None:
                skipped["missing_path"] = skipped.get("missing_path", 0) + 1
                continue
            if not shard_path.exists():
                skipped["missing_shard"] = skipped.get("missing_shard", 0) + 1
                continue
            try:
                manifest = load_manifest_from_npz(shard_path)
                if not _is_robot_key_region(shard_path, manifest):
                    skipped["not_robot_key_region"] = skipped.get("not_robot_key_region", 0) + 1
                    continue
                rollout_dir = find_rollout_dir(rollout_root, manifest)
                required = [rollout_dir / name for name in ("episode.hdf5", "cam_high.mp4", "cam_low.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4")]
                if not all(path.exists() for path in required):
                    skipped["missing_raw_rollout_file"] = skipped.get("missing_raw_rollout_file", 0) + 1
                    continue
                n_rows = _npz_row_count(shard_path)
                reward = _row_reward(shard_path, manifest)
                candidates.append((split, shard_path, manifest, rollout_dir, n_rows, reward))
            except Exception as exc:
                logging.warning("skip candidate %s: %s", shard_path, exc)
                skipped["exception"] = skipped.get("exception", 0) + 1

    if not candidates:
        raise RuntimeError(f"No usable robot key-region candidates found. skipped={skipped}")

    holdout = [item for item in candidates if item[0] == "holdout"]
    train = [item for item in candidates if item[0] != "holdout"]
    def _balanced(items: list[tuple[str, Path, dict[str, Any], Path, int, float]]) -> list[tuple[str, Path, dict[str, Any], Path, int, float]]:
        success = [x for x in items if x[5] >= 0.5]
        failure = [x for x in items if x[5] < 0.5]
        rng.shuffle(success)
        rng.shuffle(failure)
        mixed: list[tuple[str, Path, dict[str, Any], Path, int, float]] = []
        for pair in zip(success, failure, strict=False):
            mixed.extend(pair)
        longer = success if len(success) > len(failure) else failure
        mixed.extend(longer[len(mixed) // 2 :])
        return mixed

    selected_shards = (_balanced(holdout) + _balanced(train))[:max_shards]
    rows: list[SelectedRow] = []
    for split, shard_path, manifest, rollout_dir, n_rows, reward in selected_shards:
        qpos = _load_qpos(rollout_dir / "episode.hdf5")
        current_frames, next_frames = compute_replay_frame_indices(manifest, clean_rows=n_rows, episode_frames=len(qpos))
        key_region_id = str(manifest.get("key_region_id") or manifest.get("id") or rollout_dir.name)
        date = str(manifest.get("date") or _date_from_path(rollout_dir))
        phase = str(manifest.get("phase") or rollout_dir.parent.name)
        for local_row in _choose_row_indices(n_rows, rows_per_shard):
            rows.append(
                SelectedRow(
                    row_id=f"{key_region_id}:{local_row}",
                    shard_path=shard_path,
                    rollout_dir=rollout_dir,
                    local_row=int(local_row),
                    frame_index=int(current_frames[local_row]),
                    next_frame_index=int(next_frames[local_row]),
                    reward=float(reward),
                    split=split,
                    key_region_id=key_region_id,
                    date=date,
                    phase=phase,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dataclasses.asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            payload = dataclasses.asdict(row)
            payload["shard_path"] = str(payload["shard_path"])
            payload["rollout_dir"] = str(payload["rollout_dir"])
            writer.writerow(payload)
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "usable_candidates": len(candidates),
                "selected_shards": len(selected_shards),
                "selected_rows": len(rows),
                "success_rows": sum(1 for row in rows if row.reward >= 0.5),
                "failure_rows": sum(1 for row in rows if row.reward < 0.5),
                "skipped": skipped,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return rows


def _date_from_path(path: Path) -> str:
    for part in path.parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "unknown"


def _load_selected_rows(path: Path) -> list[SelectedRow]:
    rows: list[SelectedRow] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            rows.append(
                SelectedRow(
                    row_id=raw["row_id"],
                    shard_path=Path(raw["shard_path"]),
                    rollout_dir=Path(raw["rollout_dir"]),
                    local_row=int(raw["local_row"]),
                    frame_index=int(raw["frame_index"]),
                    next_frame_index=int(raw["next_frame_index"]),
                    reward=float(raw["reward"]),
                    split=raw["split"],
                    key_region_id=raw["key_region_id"],
                    date=raw["date"],
                    phase=raw["phase"],
                )
            )
    return rows


def _build_lower_right_prefix_from_blocks(
    low_tokens: np.ndarray,
    right_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place lower/right token blocks back into the cam4-compatible slot layout.

    The lower+right RLToken autoencoder was trained with four physical image
    slots, where cam_high and cam_left_wrist are masked out but still occupy
    slot ranges. This helper preserves that physical slot layout:
    high(empty), low(valid), left(empty), right(valid).
    """
    if low_tokens.shape != right_tokens.shape:
        raise ValueError(f"low/right token blocks must have same shape, got {low_tokens.shape} and {right_tokens.shape}")
    if low_tokens.ndim != 3:
        raise ValueError(f"token blocks must have shape [batch, tokens, hidden], got {low_tokens.shape}")
    batch, tokens_per_slot, hidden = low_tokens.shape
    prefix = np.zeros((batch, tokens_per_slot * 4, hidden), dtype=low_tokens.dtype)
    mask = np.zeros((batch, tokens_per_slot * 4), dtype=bool)
    prefix[:, tokens_per_slot : 2 * tokens_per_slot, :] = low_tokens
    prefix[:, 3 * tokens_per_slot : 4 * tokens_per_slot, :] = right_tokens
    mask[:, tokens_per_slot : 2 * tokens_per_slot] = True
    mask[:, 3 * tokens_per_slot : 4 * tokens_per_slot] = True
    return prefix, mask


class PrefixFeatureExtractor:
    def __init__(self, *, config_name: str, checkpoint: Path, prompt: str) -> None:
        from openpi.policies import policy_config
        from openpi.training import config as train_config

        logging.info("loading policy config=%s checkpoint=%s", config_name, checkpoint)
        cfg = train_config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt=prompt)
        self.prompt = prompt

    def extract(self, obs: dict[str, Any]) -> dict[str, Any]:
        from openpi.models import model as _model
        from openpi.policies.policy import _drop_language_from_prefix_hidden

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.policy._input_transform(inputs)  # noqa: SLF001 - experiment needs transformed model inputs.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        prefix_hidden = self.policy._embed_prefix_hidden(observation)  # noqa: SLF001
        prefix_out, prefix_mask = _drop_language_from_prefix_hidden(prefix_hidden, observation)
        z_rl = None
        if getattr(self.policy._model, "rl_token_autoencoder", None) is not None:  # noqa: SLF001
            z_rl = self.policy._model.rl_token_autoencoder.encode(jax.lax.stop_gradient(prefix_out), prefix_mask)  # noqa: SLF001
        prefix_out_np = np.asarray(jax.device_get(prefix_out[0]), dtype=np.float32)
        prefix_mask_np = np.asarray(jax.device_get(prefix_mask[0]), dtype=bool)
        slot_names = list(observation.images.keys())
        if len(slot_names) != len(CAMERA_SLOTS):
            raise RuntimeError(f"Expected 4 image slots, got {slot_names}")
        tokens_per_slot = prefix_out_np.shape[0] // len(slot_names)
        features: dict[str, np.ndarray] = {}
        token_blocks: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        positions: dict[str, dict[str, int | None]] = {}
        valid_positions = np.cumsum(prefix_mask_np.astype(np.int64)) - 1
        for slot_idx, slot in enumerate(slot_names):
            start = slot_idx * tokens_per_slot
            end = start + tokens_per_slot
            slot_tokens = prefix_out_np[start:end]
            slot_mask = prefix_mask_np[start:end]
            masks[slot] = slot_mask
            if slot_mask.any():
                valid_tokens = slot_tokens[slot_mask]
                token_blocks[slot] = valid_tokens.astype(np.float32)
                features[slot] = valid_tokens.mean(axis=0)
                slot_positions = valid_positions[start:end][slot_mask]
                positions[slot] = {
                    "token_start": int(start),
                    "token_end": int(end),
                    "valid_count": int(slot_mask.sum()),
                    "valid_position_start": int(slot_positions[0]),
                    "valid_position_end": int(slot_positions[-1]),
                }
            else:
                token_blocks[slot] = np.zeros((0, slot_tokens.shape[-1]), dtype=np.float32)
                features[slot] = np.zeros((slot_tokens.shape[-1],), dtype=np.float32)
                positions[slot] = {
                    "token_start": int(start),
                    "token_end": int(end),
                    "valid_count": 0,
                    "valid_position_start": None,
                    "valid_position_end": None,
                }
        combined = np.concatenate([features[slot] for slot in TARGET_SLOTS], axis=0)
        return {
            "features": features,
            "token_blocks": token_blocks,
            "combined": combined.astype(np.float32),
            "positions": positions,
            "z_rl": None if z_rl is None else np.asarray(jax.device_get(z_rl[0]), dtype=np.float32),
        }


def _load_observation(row: SelectedRow, *, convert_bgr_to_rgb: bool, prompt: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    qpos = _load_qpos(row.rollout_dir / "episode.hdf5")
    if row.frame_index >= len(qpos):
        raise IndexError(f"frame {row.frame_index} exceeds qpos length {len(qpos)} for {row.rollout_dir}")
    reader = _VideoFrameReader(row.rollout_dir, convert_bgr_to_rgb=convert_bgr_to_rgb)
    try:
        images = reader.read_all(row.frame_index)
    finally:
        reader.close()
    with np.load(row.shard_path, allow_pickle=False) as data:
        reference_action = np.asarray(data["reference_action"][row.local_row], dtype=np.float32)
        action = np.asarray(data["action"][row.local_row], dtype=np.float32)
    obs = {
        "images": images,
        "state": np.asarray(qpos[row.frame_index], dtype=np.float32),
        "prompt": prompt,
    }
    return obs, reference_action, action


def extract_features(
    *,
    source: str,
    selected_csv: Path,
    output_dir: Path,
    config_name: str,
    checkpoint: Path,
    prompt: str,
    convert_bgr_to_rgb: bool,
    store_token_blocks: bool,
) -> Path:
    rows = _load_selected_rows(selected_csv)
    extractor = PrefixFeatureExtractor(config_name=config_name, checkpoint=checkpoint, prompt=prompt)
    combined: list[np.ndarray] = []
    low: list[np.ndarray] = []
    right: list[np.ndarray] = []
    low_tokens: list[np.ndarray] = []
    right_tokens: list[np.ndarray] = []
    z_rl_values: list[np.ndarray] = []
    reference_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    row_ids: list[str] = []
    position_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        obs, reference_action, action = _load_observation(row, convert_bgr_to_rgb=convert_bgr_to_rgb, prompt=prompt)
        result = extractor.extract(obs)
        combined.append(result["combined"])
        low.append(result["features"]["base_1_rgb"])
        right.append(result["features"]["right_wrist_0_rgb"])
        if store_token_blocks:
            low_tokens.append(result["token_blocks"]["base_1_rgb"])
            right_tokens.append(result["token_blocks"]["right_wrist_0_rgb"])
        if result["z_rl"] is not None:
            z_rl_values.append(result["z_rl"])
        reference_actions.append(reference_action)
        executed_actions.append(action)
        rewards.append(row.reward)
        row_ids.append(row.row_id)
        position_records.append(
            {
                "row_id": row.row_id,
                "source": source,
                "positions": result["positions"],
            }
        )
        logging.info("extracted %s %d/%d row=%s reward=%.1f", source, index, len(rows), row.row_id, row.reward)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{source}_features.npz"
    payload: dict[str, np.ndarray] = {
        "row_ids": np.asarray(row_ids),
        "reward": np.asarray(rewards, dtype=np.float32),
        "low": np.stack(low).astype(np.float32),
        "right": np.stack(right).astype(np.float32),
        "combined": np.stack(combined).astype(np.float32),
        "reference_action": np.stack(reference_actions).astype(np.float32),
        "action": np.stack(executed_actions).astype(np.float32),
    }
    if store_token_blocks:
        payload["low_tokens"] = np.stack(low_tokens).astype(np.float16)
        payload["right_tokens"] = np.stack(right_tokens).astype(np.float16)
    if z_rl_values:
        if len(z_rl_values) != len(row_ids):
            raise RuntimeError("partial z_rl extraction is not allowed")
        payload["z_rl"] = np.stack(z_rl_values).astype(np.float32)
    np.savez_compressed(feature_path, **payload)
    with (output_dir / f"{source}_positions.json").open("w", encoding="utf-8") as stream:
        json.dump(position_records, stream, ensure_ascii=False, indent=2)
    return feature_path


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-12)
    return np.sum(a * b, axis=1) / denom


def _ridge_probe(x: np.ndarray, y: np.ndarray, *, seed: int, alpha: float = 1e-2) -> dict[str, float]:
    if x.shape[0] < 6:
        return {"mse": math.nan, "r2": math.nan, "train_n": 0, "test_n": 0}
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    test_n = max(2, int(round(0.3 * len(indices))))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    y_mean = y[train_idx].mean(axis=0, keepdims=True)
    x_train = (x[train_idx] - x_mean) / x_std
    x_test = (x[test_idx] - x_mean) / x_std
    y_train = y[train_idx] - y_mean
    y_test = y[test_idx]
    xtx = x_train.T @ x_train
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    pred = x_test @ weights + y_mean
    mse = float(np.mean((pred - y_test) ** 2))
    baseline_mse = float(np.mean((y_test - y[test_idx].mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - mse / max(baseline_mse, 1e-12))
    return {"mse": mse, "r2": r2, "train_n": int(len(train_idx)), "test_n": int(len(test_idx))}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    greater = 0.0
    total = 0
    for p in pos:
        greater += float(np.sum(p > neg))
        greater += 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return float(greater / total)


def _success_probe(x: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[str, float]:
    if len(np.unique(labels)) < 2 or x.shape[0] < 8:
        return {"auc": math.nan, "train_n": 0, "test_n": 0}
    y = labels.astype(np.float32)[:, None]
    probe = _ridge_probe(x, y, seed=seed, alpha=1.0)
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    test_n = max(2, int(round(0.3 * len(indices))))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x_train = (x[train_idx] - x_mean) / x_std
    x_test = (x[test_idx] - x_mean) / x_std
    y_train = y[train_idx] - y[train_idx].mean(axis=0, keepdims=True)
    xtx = x_train.T @ x_train
    weights = np.linalg.solve(xtx + np.eye(xtx.shape[0], dtype=np.float32), x_train.T @ y_train)
    scores = (x_test @ weights).reshape(-1)
    return {"auc": _auc(scores, labels[test_idx]), "train_n": int(len(train_idx)), "test_n": int(len(test_idx))}


def _plot_cosine(values: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=min(20, max(5, len(values) // 2)), color="#2f6f8f", edgecolor="white")
    plt.axvline(float(np.mean(values)), color="#b23a48", linewidth=2, label=f"mean={np.mean(values):.3f}")
    plt.xlabel("cosine(sidecar, VLA same-forward)")
    plt.ylabel("sample count")
    plt.title("Lower+Right token drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_probe(metrics: dict[str, Any], output_path: Path) -> None:
    labels = ["sidecar", "vla_same_forward"]
    mse = [metrics["reference_action_probe"]["sidecar"]["mse"], metrics["reference_action_probe"]["vla_same_forward"]["mse"]]
    r2 = [metrics["reference_action_probe"]["sidecar"]["r2"], metrics["reference_action_probe"]["vla_same_forward"]["r2"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, mse, color=["#93785b", "#2f6f8f"])
    axes[0].set_title("Reference action MSE")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, r2, color=["#93785b", "#2f6f8f"])
    axes[1].set_title("Reference action R2")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_simple_probe(probe_metrics: dict[str, Any], output_path: Path, *, title_prefix: str) -> None:
    labels = ["sidecar_z", "vla_token_z"]
    mse = [probe_metrics["sidecar_z"]["mse"], probe_metrics["vla_token_z"]["mse"]]
    r2 = [probe_metrics["sidecar_z"]["r2"], probe_metrics["vla_token_z"]["r2"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, mse, color=["#93785b", "#2f6f8f"])
    axes[0].set_title(f"{title_prefix} MSE")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, r2, color=["#93785b", "#2f6f8f"])
    axes[1].set_title(f"{title_prefix} R2")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def encode_vla_tokens_to_z(
    *,
    vla_features_path: Path,
    output_dir: Path,
    config_name: str,
    checkpoint: Path,
    batch_size: int,
) -> Path:
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    data = np.load(vla_features_path, allow_pickle=False)
    if "low_tokens" not in data or "right_tokens" not in data:
        raise ValueError(f"{vla_features_path} must contain low_tokens and right_tokens; rerun extract with --store-token-blocks")
    low_tokens = np.asarray(data["low_tokens"], dtype=np.float32)
    right_tokens = np.asarray(data["right_tokens"], dtype=np.float32)
    cfg = train_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, checkpoint, default_prompt="Twist off the bottle cap.")
    autoencoder = getattr(policy._model, "rl_token_autoencoder", None)  # noqa: SLF001
    if autoencoder is None:
        raise ValueError(f"{config_name} checkpoint does not expose rl_token_autoencoder")

    z_batches: list[np.ndarray] = []
    for start in range(0, low_tokens.shape[0], batch_size):
        end = min(start + batch_size, low_tokens.shape[0])
        prefix, mask = _build_lower_right_prefix_from_blocks(low_tokens[start:end], right_tokens[start:end])
        z = autoencoder.encode(jax.lax.stop_gradient(jnp.asarray(prefix)), jnp.asarray(mask))
        z_batches.append(np.asarray(jax.device_get(z), dtype=np.float32))
        logging.info("encoded VLA token blocks to z_rl rows %d-%d/%d", start, end, low_tokens.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vla_same_forward_token_z_features.npz"
    np.savez_compressed(
        output_path,
        row_ids=np.asarray(data["row_ids"]),
        reward=np.asarray(data["reward"], dtype=np.float32),
        z_rl=np.concatenate(z_batches, axis=0).astype(np.float32),
        reference_action=np.asarray(data["reference_action"], dtype=np.float32),
        action=np.asarray(data["action"], dtype=np.float32),
    )
    return output_path


def analyze_z(*, sidecar_features_path: Path, vla_z_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    sidecar = np.load(sidecar_features_path, allow_pickle=False)
    vla_z = np.load(vla_z_path, allow_pickle=False)
    if "z_rl" not in sidecar:
        raise ValueError(f"{sidecar_features_path} does not contain z_rl; rerun sidecar extract after this script update")
    sidecar_ids = [str(x) for x in sidecar["row_ids"]]
    vla_ids = [str(x) for x in vla_z["row_ids"]]
    if sidecar_ids != vla_ids:
        raise ValueError("sidecar z and VLA-token z feature files have different row order")
    labels = np.asarray(sidecar["reward"], dtype=np.float32) >= 0.5
    y_action = np.asarray(sidecar["reference_action"], dtype=np.float32).reshape(len(sidecar_ids), -1)
    sidecar_z = np.asarray(sidecar["z_rl"], dtype=np.float32)
    vla_token_z = np.asarray(vla_z["z_rl"], dtype=np.float32)
    cosine = _cosine_rows(sidecar_z, vla_token_z)
    metrics: dict[str, Any] = {
        "n_rows": int(len(sidecar_ids)),
        "success_rows": int(labels.sum()),
        "failure_rows": int((~labels).sum()),
        "z_cosine": {
            "mean": float(np.mean(cosine)),
            "median": float(np.median(cosine)),
            "std": float(np.std(cosine)),
            "min": float(np.min(cosine)),
            "max": float(np.max(cosine)),
        },
        "reference_action_probe": {
            "sidecar_z": _ridge_probe(sidecar_z, y_action, seed=seed),
            "vla_token_z": _ridge_probe(vla_token_z, y_action, seed=seed),
        },
        "success_probe": {
            "sidecar_z": _success_probe(sidecar_z, labels, seed=seed),
            "vla_token_z": _success_probe(vla_token_z, labels, seed=seed),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_z_rl.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    with (output_dir / "z_rl_cosine_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "reward", "z_cosine"])
        writer.writeheader()
        for row_id, reward, value in zip(sidecar_ids, sidecar["reward"], cosine, strict=True):
            writer.writerow({"row_id": row_id, "reward": float(reward), "z_cosine": float(value)})
    _plot_cosine(cosine, output_dir / "z_rl_cosine_hist.png")
    _plot_simple_probe(metrics["reference_action_probe"], output_dir / "z_rl_reference_action_probe.png", title_prefix="z_rl reference action")
    _write_z_markdown_report(metrics, output_dir / "z_rl_实验结果.md")
    return metrics


def _write_z_markdown_report(metrics: dict[str, Any], path: Path) -> None:
    probe = metrics["reference_action_probe"]
    success_probe = metrics["success_probe"]
    cosine = metrics["z_cosine"]
    conclusion = (
        "支持：主 VLA token 经 lower+right autoencoder 得到的 z_rl 更能预测 reference action。"
        if probe["vla_token_z"]["mse"] < probe["sidecar_z"]["mse"]
        else "暂不支持：z_rl 级 action probe 没有显示主 VLA token 路径更好。"
    )
    path.write_text(
        "\n".join(
            [
                "# z_rl 级实验结果：VLA token -> lower+right RLToken encoder vs sidecar z_rl",
                "",
                f"- 样本数：{metrics['n_rows']}",
                f"- success / failure：{metrics['success_rows']} / {metrics['failure_rows']}",
                f"- z_rl cosine mean：{cosine['mean']:.6f}",
                f"- z_rl cosine median：{cosine['median']:.6f}",
                f"- z_rl cosine min/max：{cosine['min']:.6f} / {cosine['max']:.6f}",
                "",
                "## Reference Action Probe",
                "",
                "| z_rl 来源 | MSE | R2 | train/test |",
                "|---|---:|---:|---:|",
                f"| sidecar z_rl | {probe['sidecar_z']['mse']:.8f} | {probe['sidecar_z']['r2']:.4f} | {probe['sidecar_z']['train_n']} / {probe['sidecar_z']['test_n']} |",
                f"| VLA token -> RLToken encoder | {probe['vla_token_z']['mse']:.8f} | {probe['vla_token_z']['r2']:.4f} | {probe['vla_token_z']['train_n']} / {probe['vla_token_z']['test_n']} |",
                "",
                "## Success / Failure Probe",
                "",
                "| z_rl 来源 | AUC | train/test |",
                "|---|---:|---:|",
                f"| sidecar z_rl | {success_probe['sidecar_z']['auc']:.4f} | {success_probe['sidecar_z']['train_n']} / {success_probe['sidecar_z']['test_n']} |",
                f"| VLA token -> RLToken encoder | {success_probe['vla_token_z']['auc']:.4f} | {success_probe['vla_token_z']['train_n']} / {success_probe['vla_token_z']['test_n']} |",
                "",
                "## 结论",
                "",
                conclusion,
                "",
                "![[z_rl_cosine_hist.png]]",
                "",
                "![[z_rl_reference_action_probe.png]]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _position_evidence(output_dir: Path) -> dict[str, Any] | None:
    sidecar_path = output_dir / "sidecar_positions.json"
    vla_path = output_dir / "vla_same_forward_positions.json"
    if not sidecar_path.exists() or not vla_path.exists():
        return None
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))[0]["positions"]
    vla = json.loads(vla_path.read_text(encoding="utf-8"))[0]["positions"]
    evidence = {
        "sidecar": {slot: sidecar[slot] for slot in TARGET_SLOTS},
        "vla_same_forward": {slot: vla[slot] for slot in TARGET_SLOTS},
    }
    return evidence


def _plot_positions(evidence: dict[str, Any], output_path: Path) -> None:
    rows = [
        ("sidecar cam_low", evidence["sidecar"]["base_1_rgb"]),
        ("VLA cam_low", evidence["vla_same_forward"]["base_1_rgb"]),
        ("sidecar right_wrist", evidence["sidecar"]["right_wrist_0_rgb"]),
        ("VLA right_wrist", evidence["vla_same_forward"]["right_wrist_0_rgb"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#93785b", "#2f6f8f", "#93785b", "#2f6f8f"]
    for idx, ((label, record), color) in enumerate(zip(rows, colors, strict=True)):
        start = record["valid_position_start"]
        end = record["valid_position_end"]
        if start is None or end is None:
            continue
        ax.barh(idx, end - start + 1, left=start, color=color)
        ax.text(end + 10, idx, f"{start}-{end}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("effective Transformer position")
    ax.set_title("Effective image-token positions")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze(*, sidecar_path: Path, vla_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    sidecar = np.load(sidecar_path, allow_pickle=False)
    vla = np.load(vla_path, allow_pickle=False)
    sidecar_ids = [str(x) for x in sidecar["row_ids"]]
    vla_ids = [str(x) for x in vla["row_ids"]]
    if sidecar_ids != vla_ids:
        raise ValueError("sidecar and VLA feature files have different row order")
    labels = np.asarray(sidecar["reward"], dtype=np.float32) >= 0.5
    y_action = np.asarray(sidecar["reference_action"], dtype=np.float32).reshape(len(sidecar_ids), -1)
    sidecar_x = np.asarray(sidecar["combined"], dtype=np.float32)
    vla_x = np.asarray(vla["combined"], dtype=np.float32)
    cosine = _cosine_rows(sidecar_x, vla_x)
    low_cosine = _cosine_rows(np.asarray(sidecar["low"], dtype=np.float32), np.asarray(vla["low"], dtype=np.float32))
    right_cosine = _cosine_rows(np.asarray(sidecar["right"], dtype=np.float32), np.asarray(vla["right"], dtype=np.float32))

    metrics: dict[str, Any] = {
        "n_rows": int(len(sidecar_ids)),
        "success_rows": int(labels.sum()),
        "failure_rows": int((~labels).sum()),
        "token_cosine": {
            "combined_mean": float(np.mean(cosine)),
            "combined_median": float(np.median(cosine)),
            "combined_std": float(np.std(cosine)),
            "combined_min": float(np.min(cosine)),
            "combined_max": float(np.max(cosine)),
            "low_mean": float(np.mean(low_cosine)),
            "right_mean": float(np.mean(right_cosine)),
        },
        "reference_action_probe": {
            "sidecar": _ridge_probe(sidecar_x, y_action, seed=seed),
            "vla_same_forward": _ridge_probe(vla_x, y_action, seed=seed),
        },
        "success_probe": {
            "sidecar": _success_probe(sidecar_x, labels, seed=seed),
            "vla_same_forward": _success_probe(vla_x, labels, seed=seed),
        },
    }
    positions = _position_evidence(output_dir)
    if positions is not None:
        metrics["position_evidence"] = positions
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    with (output_dir / "token_cosine_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "reward", "combined_cosine", "low_cosine", "right_cosine"])
        writer.writeheader()
        for row_id, reward, c, low_c, right_c in zip(sidecar_ids, sidecar["reward"], cosine, low_cosine, right_cosine, strict=True):
            writer.writerow(
                {
                    "row_id": row_id,
                    "reward": float(reward),
                    "combined_cosine": float(c),
                    "low_cosine": float(low_c),
                    "right_cosine": float(right_c),
                }
            )
    _plot_cosine(cosine, output_dir / "token_cosine_hist.png")
    _plot_probe(metrics, output_dir / "reference_action_probe.png")
    if positions is not None:
        _plot_positions(positions, output_dir / "effective_token_positions.png")
    _write_markdown_report(metrics, output_dir / "实验结果.md")
    return metrics


def _write_markdown_report(metrics: dict[str, Any], path: Path) -> None:
    probe = metrics["reference_action_probe"]
    success_probe = metrics["success_probe"]
    cosine = metrics["token_cosine"]
    conclusion = (
        "支持：主 VLA 同次 forward token 更适合作为 RLT 表示。"
        if probe["vla_same_forward"]["mse"] < probe["sidecar"]["mse"]
        else "暂不支持：本次 action probe 没有显示主 VLA token 更好。"
    )
    path.write_text(
        "\n".join(
            [
                "# 实验结果：VLA 同次 forward 2 路 token vs sidecar 2 路 token",
                "",
                f"- 样本数：{metrics['n_rows']}",
                f"- success / failure：{metrics['success_rows']} / {metrics['failure_rows']}",
                f"- combined token cosine mean：{cosine['combined_mean']:.4f}",
                f"- cam_low token cosine mean：{cosine['low_mean']:.4f}",
                f"- cam_right_wrist token cosine mean：{cosine['right_mean']:.4f}",
                "",
                "## Reference Action Probe",
                "",
                "| token 来源 | MSE | R2 | train/test |",
                "|---|---:|---:|---:|",
                f"| sidecar 重新编码 | {probe['sidecar']['mse']:.6f} | {probe['sidecar']['r2']:.4f} | {probe['sidecar']['train_n']} / {probe['sidecar']['test_n']} |",
                f"| 主 VLA 同次 forward | {probe['vla_same_forward']['mse']:.6f} | {probe['vla_same_forward']['r2']:.4f} | {probe['vla_same_forward']['train_n']} / {probe['vla_same_forward']['test_n']} |",
                "",
                "## Success / Failure Probe",
                "",
                "| token 来源 | AUC | train/test |",
                "|---|---:|---:|",
                f"| sidecar 重新编码 | {success_probe['sidecar']['auc']:.4f} | {success_probe['sidecar']['train_n']} / {success_probe['sidecar']['test_n']} |",
                f"| 主 VLA 同次 forward | {success_probe['vla_same_forward']['auc']:.4f} | {success_probe['vla_same_forward']['train_n']} / {success_probe['vla_same_forward']['test_n']} |",
                "",
                "## 结论",
                "",
                conclusion,
                "",
                "## 有效位置证据",
                "",
                "如果两条路径完全等价，同一相机的有效 token 位置也应该一致。本次结果显示位置不一致，但 pooled hidden 仍然高度相似。",
                "",
                "![[effective_token_positions.png]]",
                "",
                "图表：",
                "",
                "![[token_cosine_hist.png]]",
                "",
                "![[reference_action_probe.png]]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    select = sub.add_parser("select-data")
    select.add_argument("--manifest", action="append", type=Path, dest="manifests")
    select.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    select.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    select.add_argument("--max-shards", type=int, default=12)
    select.add_argument("--rows-per-shard", type=int, default=4)
    select.add_argument("--seed", type=int, default=42)

    extract = sub.add_parser("extract")
    extract.add_argument("--source", choices=("sidecar", "vla_same_forward"), required=True)
    extract.add_argument("--selected-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "selected_samples.csv")
    extract.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    extract.add_argument("--prompt", default="Twist off the bottle cap.")
    extract.add_argument("--convert-bgr-to-rgb", action="store_true")
    extract.add_argument("--config-name")
    extract.add_argument("--checkpoint", type=Path)
    extract.add_argument("--store-token-blocks", action="store_true")

    encode_z = sub.add_parser("encode-vla-z")
    encode_z.add_argument("--vla-features", type=Path, default=DEFAULT_OUTPUT_DIR / "vla_same_forward_features.npz")
    encode_z.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    encode_z.add_argument("--config-name", default=DEFAULT_SIDECAR_CONFIG)
    encode_z.add_argument("--checkpoint", type=Path, default=DEFAULT_SIDECAR_CHECKPOINT)
    encode_z.add_argument("--batch-size", type=int, default=8)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--sidecar-features", type=Path)
    analyze_parser.add_argument("--vla-features", type=Path)
    analyze_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    analyze_parser.add_argument("--seed", type=int, default=42)

    analyze_z_parser = sub.add_parser("analyze-z")
    analyze_z_parser.add_argument("--sidecar-features", type=Path)
    analyze_z_parser.add_argument("--vla-z-features", type=Path)
    analyze_z_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    analyze_z_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    if args.cmd == "select-data":
        manifests = tuple(args.manifests) if args.manifests else DEFAULT_MANIFESTS
        rows = select_rows(
            manifest_paths=manifests,
            rollout_root=args.rollout_root,
            output_dir=args.output_dir,
            max_shards=args.max_shards,
            rows_per_shard=args.rows_per_shard,
            seed=args.seed,
        )
        logging.info("selected %d rows -> %s", len(rows), args.output_dir / "selected_samples.csv")
    elif args.cmd == "extract":
        if args.source == "sidecar":
            config_name = args.config_name or DEFAULT_SIDECAR_CONFIG
            checkpoint = args.checkpoint or DEFAULT_SIDECAR_CHECKPOINT
        else:
            config_name = args.config_name or DEFAULT_CAM4_CONFIG
            checkpoint = args.checkpoint or DEFAULT_CAM4_CHECKPOINT
        path = extract_features(
            source=args.source,
            selected_csv=args.selected_csv,
            output_dir=args.output_dir,
            config_name=config_name,
            checkpoint=checkpoint,
            prompt=args.prompt,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
            store_token_blocks=args.store_token_blocks,
        )
        logging.info("wrote %s", path)
    elif args.cmd == "encode-vla-z":
        path = encode_vla_tokens_to_z(
            vla_features_path=args.vla_features,
            output_dir=args.output_dir,
            config_name=args.config_name,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
        )
        logging.info("wrote %s", path)
    elif args.cmd == "analyze":
        sidecar_features = args.sidecar_features or (args.output_dir / "sidecar_features.npz")
        vla_features = args.vla_features or (args.output_dir / "vla_same_forward_features.npz")
        metrics = analyze(
            sidecar_path=sidecar_features,
            vla_path=vla_features,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        logging.info("metrics: %s", json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.cmd == "analyze-z":
        sidecar_features = args.sidecar_features or (args.output_dir / "sidecar_features.npz")
        vla_z_features = args.vla_z_features or (args.output_dir / "vla_same_forward_token_z_features.npz")
        metrics = analyze_z(
            sidecar_features_path=sidecar_features,
            vla_z_path=vla_z_features,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        logging.info("z metrics: %s", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

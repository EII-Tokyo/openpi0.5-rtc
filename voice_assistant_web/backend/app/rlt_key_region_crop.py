from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise ValueError("manifest.json is missing")
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain an object")
    return payload


def _duration_seconds(manifest: dict[str, Any]) -> float:
    duration = float(manifest.get("duration_seconds") or 0.0)
    if duration <= 0:
        start_time = manifest.get("start_time")
        end_time = manifest.get("end_time")
        if start_time is not None and end_time is not None:
            duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        raise ValueError("duration_seconds is required for crop")
    return duration


def _sample_bounds(*, start_sec: float, end_sec: float, duration_sec: float, sample_count: int) -> tuple[int, int]:
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative")
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    if sample_count <= 0:
        raise ValueError("replay shard has no samples")
    start = int(np.floor((start_sec / duration_sec) * sample_count))
    end = int(np.ceil((end_sec / duration_sec) * sample_count))
    start = min(max(start, 0), sample_count - 1)
    end = min(max(end, start + 1), sample_count)
    return start, end


def _train_horizon(manifest: dict[str, Any], reward_seq: np.ndarray | None) -> int:
    horizon = int(manifest.get("train_horizon") or manifest.get("train_chunk_horizon") or 10)
    if reward_seq is not None and reward_seq.ndim >= 2:
        horizon = min(max(horizon, 1), int(reward_seq.shape[1]))
    return horizon


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp_path, path)


def crop_key_region_files(
    rollout_dir: Path,
    shard_path: Path,
    output_shard_path: Path,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    manifest_path = rollout_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    duration_sec = _duration_seconds(manifest)
    if not shard_path.exists():
        raise ValueError("replay shard is missing")

    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    sample_count = 0
    for key, value in arrays.items():
        if key == "manifest":
            continue
        if value.ndim > 0:
            sample_count = int(value.shape[0])
            break
    start_index, end_index = _sample_bounds(
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=duration_sec,
        sample_count=sample_count,
    )
    cropped_count = end_index - start_index

    cropped: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key != "manifest" and value.ndim > 0 and int(value.shape[0]) == sample_count:
            cropped[key] = value[start_index:end_index]
        else:
            cropped[key] = value

    reward_seq = cropped.get("reward_seq")
    if reward_seq is not None and reward_seq.ndim >= 2 and cropped_count > 0:
        reward_seq = np.zeros_like(reward_seq)
        reward_index = _train_horizon(manifest, reward_seq) - 1
        reward_seq[-1, reward_index] = float(manifest.get("reward") or 0.0)
        cropped["reward_seq"] = reward_seq
    done = cropped.get("done")
    if done is not None and done.ndim == 1 and cropped_count > 0:
        next_done = np.zeros_like(done, dtype=np.bool_)
        next_done[-1] = True
        cropped["done"] = next_done

    original_count = int(manifest.get("crop_original_num_replay_transitions") or manifest.get("num_replay_transitions") or sample_count)
    source_shard_path = str(manifest.get("source_shard_path") or manifest.get("shard_path") or shard_path)
    manifest.update(
        {
            "shard_path": str(output_shard_path),
            "source_shard_path": source_shard_path,
            "crop_start_sec": float(start_sec),
            "crop_end_sec": float(end_sec),
            "crop_start_sample": int(start_index),
            "crop_end_sample": int(end_index),
            "crop_original_num_replay_transitions": original_count,
            "crop_duration_seconds": float(end_sec - start_sec),
            "num_replay_transitions": int(cropped_count),
            "replay_array_shapes": {key: list(value.shape) for key, value in cropped.items() if key != "manifest"},
            "replay_status": "ready",
            "replay_ready": True,
            "segment_status": "committed",
            "train_eligible": True,
            "voided": False,
        }
    )
    cropped["manifest"] = np.asarray(json.dumps(manifest))

    output_shard_path.parent.mkdir(parents=True, exist_ok=True)
    _write_npz(output_shard_path, cropped)
    _write_manifest(manifest_path, manifest)
    return manifest

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import bisect

import numpy as np

JITTER_PENALTIES: dict[str, float] = {
    "smooth": 0.0,
    "mild_jitter": 0.3,
    "severe_jitter": 1.0,
}
QUALITY_JITTER_LAMBDA = 0.25


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


def _replay_start_frames(manifest: dict[str, Any], *, sample_count: int) -> list[int] | None:
    try:
        frame_count = int(manifest.get("num_frames") or 0)
        train_horizon = int(manifest.get("train_horizon") or manifest.get("train_chunk_horizon") or 10)
        chunk_stride = int(manifest.get("chunk_stride") or 2)
    except (TypeError, ValueError):
        return None
    if frame_count <= 0 or train_horizon <= 0 or chunk_stride <= 0:
        return None
    last_start = frame_count - (2 * train_horizon)
    if last_start < 0:
        return None
    starts = list(range(0, last_start + 1, chunk_stride))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    if len(starts) != sample_count:
        return None
    return starts


def _replay_sample_bounds(
    manifest: dict[str, Any],
    *,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    sample_count: int,
) -> tuple[int, int]:
    starts = _replay_start_frames(manifest, sample_count=sample_count)
    if starts is None:
        return _sample_bounds(
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            sample_count=sample_count,
        )
    frame_count = int(manifest.get("num_frames") or 0)
    start_frame = int(np.floor((start_sec / duration_sec) * frame_count))
    end_frame = int(np.ceil((end_sec / duration_sec) * frame_count))
    start_frame = min(max(start_frame, 0), frame_count - 1)
    end_frame = min(max(end_frame, start_frame), frame_count - 1)
    start = bisect.bisect_left(starts, start_frame)
    end = bisect.bisect_right(starts, end_frame)
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
    start_index, end_index = _replay_sample_bounds(
        manifest,
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


def rescore_key_region_files(rollout_dir: Path, shard_path: Path, *, reward: int) -> dict[str, Any]:
    if reward not in (0, 1):
        raise ValueError("reward must be 0 or 1")
    manifest_path = rollout_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if not shard_path.exists():
        raise ValueError("replay shard is missing")

    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    reward_seq = arrays.get("reward_seq")
    if reward_seq is None or reward_seq.ndim < 2 or int(reward_seq.shape[0]) == 0:
        raise ValueError("replay shard reward_seq is missing or empty")
    reward_seq = np.zeros_like(reward_seq, dtype=np.float32)
    done = arrays.get("done")
    if done is not None and done.ndim == 1 and np.any(done):
        terminal_index = int(np.flatnonzero(done)[-1])
    else:
        terminal_index = int(reward_seq.shape[0] - 1)
    reward_index = _train_horizon(manifest, reward_seq) - 1
    reward_seq[terminal_index, reward_index] = float(reward)
    arrays["reward_seq"] = reward_seq

    manifest.update(
        {
            "reward": int(reward),
            "score_timeout": False,
            "rescore_time": float(__import__("time").time()),
            "replay_status": "ready",
            "replay_ready": True,
            "segment_status": "committed",
            "train_eligible": True,
            "voided": False,
        }
    )
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    _write_npz(shard_path, arrays)
    _write_manifest(manifest_path, manifest)
    return manifest


def update_key_region_quality_files(
    rollout_dir: Path,
    shard_path: Path,
    *,
    quality_score: int,
    jitter_level: str,
    actor_train_mode: str,
    notes: str | None = None,
) -> dict[str, Any]:
    if quality_score < 0 or quality_score > 4:
        raise ValueError("quality_score must be between 0 and 4")
    if jitter_level not in JITTER_PENALTIES:
        raise ValueError(f"unsupported jitter_level={jitter_level}")
    if actor_train_mode not in {"auto", "exclude", "low_weight", "normal", "strong"}:
        raise ValueError(f"unsupported actor_train_mode={actor_train_mode}")
    manifest_path = rollout_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if not shard_path.exists():
        raise ValueError("replay shard is missing")

    quality_task = float(quality_score) / 4.0
    jitter_penalty = JITTER_PENALTIES[jitter_level]
    quality_final = max(0.0, min(1.0, quality_task - QUALITY_JITTER_LAMBDA * jitter_penalty))
    quality_payload = {
        "quality_score": int(quality_score),
        "quality_task": float(quality_task),
        "jitter_level": jitter_level,
        "jitter_penalty": float(jitter_penalty),
        "quality_final": float(quality_final),
        "actor_train_mode": actor_train_mode,
        "quality_source": "human",
        "quality_version": 1,
        "quality_updated_at": float(__import__("time").time()),
        "quality_notes": notes,
    }

    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    manifest.update(quality_payload)
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    _write_npz(shard_path, arrays)
    _write_manifest(manifest_path, manifest)
    return manifest

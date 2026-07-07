from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

REPLAY_KEYS = (
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

STATE_DIRS = ("pending", "running", "ready", "failed")


@dataclasses.dataclass(frozen=True)
class AnchorTokenJob:
    path: Path
    payload: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class FormalReplayResult:
    shard_path: Path
    manifest: dict[str, Any]


def safe_key_region_name(key_region_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(key_region_id).strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("key_region_id must not be empty")
    return cleaned if cleaned.startswith("key_region_") else f"key_region_{cleaned}"


def ensure_job_dirs(job_root: Path) -> None:
    for name in STATE_DIRS:
        (job_root / name).mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def load_npz_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in data.files:
        return {}
    raw = data["manifest"]
    raw_value = (
        raw.item()
        if isinstance(raw, np.ndarray) and raw.shape == ()
        else raw.tolist()
        if isinstance(raw, np.ndarray)
        else raw
    )
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
        if not isinstance(parsed, dict):
            raise ValueError("npz manifest JSON is not an object")
        return parsed
    if isinstance(raw_value, dict):
        return raw_value
    raise ValueError(f"unsupported npz manifest type: {type(raw_value)!r}")


def write_pending_job(
    *,
    job_root: Path,
    manifest: dict[str, Any],
    rollout_dir: Path,
    source_shard_path: Path | None,
    overwrite: bool = True,
) -> AnchorTokenJob:
    ensure_job_dirs(job_root)
    key_region_id = str(manifest.get("key_region_id") or "")
    job_name = safe_key_region_name(key_region_id)
    path = job_root / "pending" / f"{job_name}.json"
    if path.exists() and not overwrite:
        return AnchorTokenJob(path=path, payload=load_json(path))
    payload = {
        "schema_version": 1,
        "status": "pending",
        "created_at": time.time(),
        "updated_at": time.time(),
        "key_region_id": key_region_id,
        "task": manifest.get("task"),
        "phase": manifest.get("phase"),
        "reward": manifest.get("reward"),
        "rollout_dir": str(Path(rollout_dir).resolve()),
        "source_runtime_cache_block_shard_path": None if source_shard_path is None else str(Path(source_shard_path).resolve()),
        "source_replay_state_grain": manifest.get("replay_state_grain"),
        "formal_replay_state_grain": "paper_subsampled_anchor",
        "subsampled_transition_semantics": "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
        "train_horizon": int(manifest.get("train_horizon") or manifest.get("train_chunk_horizon") or manifest.get("policy_horizon") or 0),
        "chunk_stride": int(manifest.get("chunk_stride") or manifest.get("train_chunk_stride") or 0),
        "num_frames": int(manifest.get("num_frames") or 0),
        "num_replay_transitions": int(manifest.get("num_replay_transitions") or 0),
        "requires_vla_same_forward": True,
        "sidecar_rl_token_allowed": False,
    }
    write_json_atomic(path, payload)
    return AnchorTokenJob(path=path, payload=payload)


def move_job(job_path: Path, status: str, *, extra: dict[str, Any] | None = None) -> AnchorTokenJob:
    if status not in STATE_DIRS:
        raise ValueError(f"unknown job status {status!r}; expected one of {STATE_DIRS}")
    payload = load_json(job_path)
    payload.update(extra or {})
    payload["status"] = status
    payload["updated_at"] = time.time()
    target = job_path.parent.parent / status / job_path.name
    write_json_atomic(target, payload)
    if target.resolve() != job_path.resolve() and job_path.exists():
        job_path.unlink()
    return AnchorTokenJob(path=target, payload=payload)


def list_jobs(job_root: Path, *, status: str = "pending") -> list[AnchorTokenJob]:
    if status not in STATE_DIRS:
        raise ValueError(f"unknown job status {status!r}; expected one of {STATE_DIRS}")
    job_dir = job_root / status
    if not job_dir.exists():
        return []
    return [AnchorTokenJob(path=path, payload=load_json(path)) for path in sorted(job_dir.glob("*.json"))]


def _load_required_arrays(path: Path, keys: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing arrays {missing}")
        arrays = {key: np.asarray(data[key]) for key in keys}
        manifest = load_npz_manifest(data)
    return arrays, manifest


def _validate_replacement_rows(
    *,
    source_path: Path,
    source_rows: int,
    cache_path: Path,
    cache_arrays: dict[str, np.ndarray],
) -> None:
    for key in ("z_rl", "next_z_rl", "proprio", "next_proprio"):
        rows = int(cache_arrays[key].shape[0])
        if rows != source_rows:
            raise ValueError(f"{cache_path} {key} rows {rows} != {source_path} action rows {source_rows}")
    for key in ("current_frames", "next_frames"):
        rows = int(cache_arrays[key].shape[0])
        if rows != source_rows:
            raise ValueError(f"{cache_path} {key} rows {rows} != {source_path} action rows {source_rows}")


def assemble_formal_replay_from_encoded_cache(
    *,
    job_path: Path,
    encoded_cache_path: Path,
    output_root: Path,
    overwrite: bool,
) -> FormalReplayResult:
    job = load_json(job_path)
    source = job.get("source_runtime_cache_block_shard_path")
    if not source:
        raise ValueError(f"{job_path} has no source_runtime_cache_block_shard_path")
    source_path = Path(source)
    arrays, source_manifest = _load_required_arrays(source_path, REPLAY_KEYS)
    cache_arrays, _cache_manifest = _load_required_arrays(
        encoded_cache_path,
        ("z_rl", "next_z_rl", "proprio", "next_proprio", "current_frames", "next_frames"),
    )
    source_rows = int(arrays["action"].shape[0])
    _validate_replacement_rows(
        source_path=source_path,
        source_rows=source_rows,
        cache_path=encoded_cache_path,
        cache_arrays=cache_arrays,
    )

    arrays["z_rl"] = np.asarray(cache_arrays["z_rl"], dtype=np.float32)
    arrays["next_z_rl"] = np.asarray(cache_arrays["next_z_rl"], dtype=np.float32)
    arrays["proprio"] = np.asarray(cache_arrays["proprio"], dtype=np.float32)
    arrays["next_proprio"] = np.asarray(cache_arrays["next_proprio"], dtype=np.float32)
    current_frames = np.asarray(cache_arrays["current_frames"], dtype=np.int64)
    next_frames = np.asarray(cache_arrays["next_frames"], dtype=np.int64)

    key_region_id = str(job.get("key_region_id") or source_manifest.get("key_region_id") or source_path.stem.removeprefix("key_region_"))
    output_shard_dir = output_root / "shards"
    output_shard_dir.mkdir(parents=True, exist_ok=True)
    out = output_shard_dir / f"{safe_key_region_name(key_region_id)}.npz"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")

    previous_shapes = {key: list(value.shape) for key, value in arrays.items()}
    manifest = dict(source_manifest)
    manifest.update(
        {
            "key_region_id": key_region_id,
            "reward": job.get("reward", source_manifest.get("reward")),
            "z_rl_source": "async_anchor_token_cache_vla_same_forward",
            "z_rl_dim": int(arrays["z_rl"].shape[-1]),
            "proprio_source": "vla_policy_input_transform_at_anchor_frame",
            "proprio_dim": int(arrays["proprio"].shape[-1]),
            "replay_state_grain": "paper_subsampled_anchor",
            "requires_offline_reencode": False,
            "formal_replay_state_grain": "paper_subsampled_anchor",
            "formal_replay_ready": True,
            "train_eligible": True,
            "subsampled_transition_semantics": "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
            "source_runtime_cache_block_shard_path": str(source_path.resolve()),
            "source_rollout_dir": job.get("rollout_dir"),
            "async_anchor_token_job_path": str(job_path.resolve()),
            "encoded_anchor_token_cache_path": str(Path(encoded_cache_path).resolve()),
            "conversion_cache_scope": "transition_anchor_frames",
            "train_horizon": int(job.get("train_horizon") or source_manifest.get("train_horizon") or 0),
            "chunk_stride": int(job.get("chunk_stride") or source_manifest.get("chunk_stride") or 0),
            "current_frames": [int(x) for x in current_frames],
            "next_frames": [int(x) for x in next_frames],
            "previous_replay_array_shapes": previous_shapes,
            "replay_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
    )
    payload = {**arrays, "manifest": np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))}
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    tmp_path.replace(out)

    manifest_row = {
        "shard_path": str(out.resolve()),
        "batch": "async_anchor_token_cache_paper_subsampled_anchor",
        "source_group": "async_anchor_token_cache",
        "key_region_id": key_region_id,
        "reward": manifest.get("reward"),
        "num_transitions": source_rows,
        "z_dim": int(arrays["z_rl"].shape[-1]),
        "replay_state_grain": "paper_subsampled_anchor",
    }
    return FormalReplayResult(shard_path=out, manifest=manifest_row)

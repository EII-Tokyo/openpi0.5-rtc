from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
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


def compute_anchor_starts(num_frames: int, train_horizon: int, chunk_stride: int) -> np.ndarray:
    if train_horizon <= 0:
        raise ValueError("train_horizon must be positive")
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    last_start = int(num_frames) - (2 * int(train_horizon))
    if last_start < 0:
        raise ValueError(f"episode has {num_frames} frames, shorter than 2 * horizon {2 * train_horizon}")
    starts = list(range(0, last_start + 1, int(chunk_stride)))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    return np.asarray(starts, dtype=np.int64)


def build_paper_replay_from_timeline_hdf5(
    hdf5_path: str | Path,
    *,
    train_horizon: int,
    chunk_stride: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(hdf5_path)
    with h5py.File(path, "r") as root:
        _require_dataset(root, "action")
        _require_dataset(root, "reference_action")
        _require_dataset(root, "rlt_timeline/z_rl")
        _require_dataset(root, "rlt_timeline/proprio")
        action = np.asarray(root["action"], dtype=np.float32)
        reference_action = np.asarray(root["reference_action"], dtype=np.float32)
        z_rl = np.asarray(root["rlt_timeline/z_rl"], dtype=np.float32)
        proprio = np.asarray(root["rlt_timeline/proprio"], dtype=np.float32)
        valid = np.asarray(root["rlt_timeline/valid"], dtype=np.bool_) if "rlt_timeline/valid" in root else None
        reward = float(root.attrs.get("reward", 0.0))
        key_region_id = _h5_attr_str(root.attrs.get("key_region_id", path.stem))
        z_rl_source = _h5_attr_str(root["rlt_timeline"].attrs.get("z_rl_source", ""))
        rl_token_checkpoint_path = _h5_attr_str(root["rlt_timeline"].attrs.get("rl_token_checkpoint_path", ""))

    _validate_timeline_arrays(
        path=path,
        action=action,
        reference_action=reference_action,
        z_rl=z_rl,
        proprio=proprio,
        valid=valid,
        z_rl_source=z_rl_source,
    )
    starts = compute_anchor_starts(len(action), train_horizon, chunk_stride)
    next_frames = starts + int(train_horizon)
    arrays = {
        "z_rl": z_rl[starts].astype(np.float32),
        "proprio": proprio[starts].astype(np.float32),
        "action": _windows(action, starts, train_horizon),
        "reference_action": _windows(reference_action, starts, train_horizon),
        "reward_seq": np.zeros((len(starts), int(train_horizon)), dtype=np.float32),
        "next_z_rl": z_rl[next_frames].astype(np.float32),
        "next_proprio": proprio[next_frames].astype(np.float32),
        "next_reference_action": _windows(reference_action, next_frames, train_horizon),
        "done": np.zeros((len(starts),), dtype=np.bool_),
    }
    arrays["done"][-1] = True
    arrays["reward_seq"][-1, int(train_horizon) - 1] = reward
    manifest = {
        "key_region_id": key_region_id,
        "reward": reward,
        "source_format": "rlt_timeline_hdf5",
        "source_hdf5_path": str(path.resolve()),
        "replay_state_grain": "paper_subsampled_anchor",
        "formal_replay_state_grain": "paper_subsampled_anchor",
        "formal_replay_ready": True,
        "train_eligible": True,
        "subsampled_transition_semantics": "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
        "z_rl_source": z_rl_source,
        "z_rl_dim": int(z_rl.shape[-1]),
        "proprio_dim": int(proprio.shape[-1]),
        "action_dim": int(action.shape[-1]),
        "train_horizon": int(train_horizon),
        "chunk_stride": int(chunk_stride),
        "current_frames": [int(frame) for frame in starts],
        "next_frames": [int(frame) for frame in next_frames],
        "rl_token_checkpoint_path": rl_token_checkpoint_path,
        "replay_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    return arrays, manifest


def write_paper_replay_shard_from_timeline_hdf5(
    hdf5_path: str | Path,
    output_path: str | Path,
    *,
    train_horizon: int,
    chunk_stride: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    arrays, manifest = build_paper_replay_from_timeline_hdf5(
        hdf5_path,
        train_horizon=train_horizon,
        chunk_stride=chunk_stride,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "shard_path": str(out.resolve())}
    payload = {**arrays, "manifest": np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))}
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    tmp_path.replace(out)
    return manifest


def _windows(array: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([array[int(start) : int(start) + int(horizon)] for start in starts], axis=0).astype(np.float32)


def _require_dataset(root: h5py.File, name: str) -> None:
    if name not in root:
        raise KeyError(f"{root.filename} is missing required dataset {name}")


def _validate_timeline_arrays(
    *,
    path: Path,
    action: np.ndarray,
    reference_action: np.ndarray,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    valid: np.ndarray | None,
    z_rl_source: str,
) -> None:
    if action.ndim != 2:
        raise ValueError(f"{path} action must have shape [T, action_dim], got {action.shape}")
    if reference_action.shape != action.shape:
        raise ValueError(f"{path} reference_action shape {reference_action.shape} != action shape {action.shape}")
    if z_rl.ndim != 2 or proprio.ndim != 2:
        raise ValueError(f"{path} z_rl and proprio must be rank-2 arrays")
    lengths = {len(action), len(reference_action), len(z_rl), len(proprio)}
    if len(lengths) != 1:
        raise ValueError(
            f"{path} timeline length mismatch: action={len(action)} reference_action={len(reference_action)} "
            f"z_rl={len(z_rl)} proprio={len(proprio)}"
        )
    if valid is not None:
        if valid.shape != (len(action),):
            raise ValueError(f"{path} rlt_timeline/valid shape {valid.shape} != ({len(action)},)")
        if not bool(np.all(valid)):
            raise ValueError(f"{path} contains invalid rlt_timeline rows")
    if not z_rl_source.startswith("vla_same_forward"):
        raise ValueError(f"{path} z_rl_source={z_rl_source!r} is not a vla_same_forward source")
    for name, array in (("action", action), ("reference_action", reference_action), ("z_rl", z_rl), ("proprio", proprio)):
        if not np.isfinite(array).all():
            raise ValueError(f"{path} {name} contains non-finite values")


def _h5_attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

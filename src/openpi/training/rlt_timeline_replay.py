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

POLICY_EVENT_ALIGNMENT_EXACT = "exact_event_pairs"
POLICY_EVENT_ALIGNMENT_TRUNK_SHARED = "trunk_shared"
DEFAULT_POLICY_EVENT_ALIGNMENT = POLICY_EVENT_ALIGNMENT_TRUNK_SHARED


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
    policy_event_alignment: str = DEFAULT_POLICY_EVENT_ALIGNMENT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_policy_event_alignment(policy_event_alignment)
    path = Path(hdf5_path)
    with h5py.File(path, "r") as root:
        _require_dataset(root, "action")
        _require_dataset(root, "reference_action")
        action = np.asarray(root["action"], dtype=np.float32)
        reference_action = np.asarray(root["reference_action"], dtype=np.float32)
        reward = float(root.attrs.get("reward", 0.0))
        key_region_id = _h5_attr_str(root.attrs.get("key_region_id", path.stem))
        rl_token_checkpoint_path = ""
        if "rlt_timeline/z_rl" in root and "rlt_timeline/proprio" in root:
            z_rl = np.asarray(root["rlt_timeline/z_rl"], dtype=np.float32)
            proprio = np.asarray(root["rlt_timeline/proprio"], dtype=np.float32)
            valid = np.asarray(root["rlt_timeline/valid"], dtype=np.bool_) if "rlt_timeline/valid" in root else None
            z_rl_source = _h5_attr_str(root["rlt_timeline"].attrs.get("z_rl_source", ""))
            rl_token_checkpoint_path = _h5_attr_str(root["rlt_timeline"].attrs.get("rl_token_checkpoint_path", ""))
            arrays, manifest = _build_from_complete_frame_timeline(
                path=path,
                action=action,
                reference_action=reference_action,
                z_rl=z_rl,
                proprio=proprio,
                valid=valid,
                z_rl_source=z_rl_source,
                reward=reward,
                key_region_id=key_region_id,
                train_horizon=train_horizon,
                chunk_stride=chunk_stride,
                rl_token_checkpoint_path=rl_token_checkpoint_path,
            )
            return arrays, manifest
        events = _read_policy_forward_events(root)

    return _build_from_policy_forward_events(
        path=path,
        action=action,
        reference_action=reference_action,
        events=events,
        reward=reward,
        key_region_id=key_region_id,
        train_horizon=train_horizon,
        chunk_stride=chunk_stride,
        rl_token_checkpoint_path=rl_token_checkpoint_path,
        policy_event_alignment=policy_event_alignment,
    )


def _build_from_complete_frame_timeline(
    *,
    path: Path,
    action: np.ndarray,
    reference_action: np.ndarray,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    valid: np.ndarray | None,
    z_rl_source: str,
    reward: float,
    key_region_id: str,
    train_horizon: int,
    chunk_stride: int,
    rl_token_checkpoint_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    arrays = _assemble_arrays(
        action=action,
        reference_action=reference_action,
        current_z_rl=z_rl[starts],
        current_proprio=proprio[starts],
        next_z_rl=z_rl[next_frames],
        next_proprio=proprio[next_frames],
        starts=starts,
        next_frames=next_frames,
        train_horizon=train_horizon,
        reward=reward,
    )
    manifest = _build_manifest(
        path=path,
        arrays=arrays,
        starts=starts,
        next_frames=next_frames,
        key_region_id=key_region_id,
        reward=reward,
        z_rl_source=z_rl_source,
        z_rl_dim=int(z_rl.shape[-1]),
        proprio_dim=int(proprio.shape[-1]),
        action_dim=int(action.shape[-1]),
        train_horizon=train_horizon,
        chunk_stride=chunk_stride,
        rl_token_checkpoint_path=rl_token_checkpoint_path,
        z_alignment="complete_frame_timeline",
    )
    return arrays, manifest


def _assemble_arrays(
    *,
    action: np.ndarray,
    reference_action: np.ndarray,
    current_z_rl: np.ndarray,
    current_proprio: np.ndarray,
    next_z_rl: np.ndarray,
    next_proprio: np.ndarray,
    starts: np.ndarray,
    next_frames: np.ndarray,
    train_horizon: int,
    reward: float,
) -> dict[str, np.ndarray]:
    arrays = {
        "z_rl": current_z_rl.astype(np.float32),
        "proprio": current_proprio.astype(np.float32),
        "action": _windows(action, starts, train_horizon),
        "reference_action": _windows(reference_action, starts, train_horizon),
        "reward_seq": np.zeros((len(starts), int(train_horizon)), dtype=np.float32),
        "next_z_rl": next_z_rl.astype(np.float32),
        "next_proprio": next_proprio.astype(np.float32),
        "next_reference_action": _windows(reference_action, next_frames, train_horizon),
        "done": np.zeros((len(starts),), dtype=np.bool_),
    }
    arrays["done"][-1] = True
    arrays["reward_seq"][-1, int(train_horizon) - 1] = reward
    return arrays


def _build_from_policy_forward_events(
    *,
    path: Path,
    action: np.ndarray,
    reference_action: np.ndarray,
    events: dict[str, np.ndarray | str],
    reward: float,
    key_region_id: str,
    train_horizon: int,
    chunk_stride: int,
    rl_token_checkpoint_path: str,
    policy_event_alignment: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_action_arrays(path=path, action=action, reference_action=reference_action)
    step_index = np.asarray(events["step_index"], dtype=np.int64)
    z_rl = np.asarray(events["z_rl"], dtype=np.float32)
    proprio = np.asarray(events["proprio"], dtype=np.float32)
    z_rl_source = str(events["z_rl_source"])
    _validate_policy_forward_events(
        path=path,
        action_len=len(action),
        step_index=step_index,
        z_rl=z_rl,
        proprio=proprio,
        z_rl_source=z_rl_source,
    )

    if policy_event_alignment == POLICY_EVENT_ALIGNMENT_EXACT:
        starts, next_frames, current_indices, next_indices = _select_exact_policy_event_pairs(
            path=path,
            action_len=len(action),
            step_index=step_index,
            train_horizon=train_horizon,
            chunk_stride=chunk_stride,
        )
        z_alignment = "policy_forward_event_exact_step_pairs"
        replay_state_grain = "paper_subsampled_anchor"
        subsampled_transition_semantics = "x_i_action_i_to_i_plus_c_next_x_i_plus_c"
    else:
        starts, next_frames, current_indices, next_indices = _select_trunk_shared_policy_events(
            path=path,
            action_len=len(action),
            step_index=step_index,
            train_horizon=train_horizon,
            chunk_stride=chunk_stride,
        )
        z_alignment = "policy_forward_event_trunk_shared"
        replay_state_grain = "trunk_shared_z_subsampled_anchor"
        subsampled_transition_semantics = (
            "x_i_action_i_to_i_plus_c_next_x_i_plus_c_with_real_forward_z_shared_inside_trunk"
        )
    arrays = _assemble_arrays(
        action=action,
        reference_action=reference_action,
        current_z_rl=z_rl[current_indices],
        current_proprio=proprio[current_indices],
        next_z_rl=z_rl[next_indices],
        next_proprio=proprio[next_indices],
        starts=starts,
        next_frames=next_frames,
        train_horizon=train_horizon,
        reward=reward,
    )
    manifest = _build_manifest(
        path=path,
        arrays=arrays,
        starts=starts,
        next_frames=next_frames,
        key_region_id=key_region_id,
        reward=reward,
        z_rl_source=z_rl_source,
        z_rl_dim=int(z_rl.shape[-1]),
        proprio_dim=int(proprio.shape[-1]),
        action_dim=int(action.shape[-1]),
        train_horizon=train_horizon,
        chunk_stride=chunk_stride,
        rl_token_checkpoint_path=rl_token_checkpoint_path,
        z_alignment=z_alignment,
        replay_state_grain=replay_state_grain,
        subsampled_transition_semantics=subsampled_transition_semantics,
    )
    return arrays, manifest


def _select_exact_policy_event_pairs(
    *,
    path: Path,
    action_len: int,
    step_index: np.ndarray,
    train_horizon: int,
    chunk_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_by_step = {int(step): idx for idx, step in enumerate(step_index)}
    candidate_starts = [
        int(step)
        for step in sorted(step_index)
        if int(step) + int(train_horizon) in event_by_step and int(step) + (2 * int(train_horizon)) <= int(action_len)
    ]
    selected_starts: list[int] = []
    for step in candidate_starts:
        if not selected_starts or step - selected_starts[-1] >= int(chunk_stride):
            selected_starts.append(step)
    if not selected_starts:
        raise ValueError(
            f"{path} has no exact policy-forward event pairs separated by train_horizon={train_horizon}; "
            "refusing to synthesize or copy z_rl"
        )

    starts = np.asarray(selected_starts, dtype=np.int64)
    next_frames = starts + int(train_horizon)
    current_indices = np.asarray([event_by_step[int(step)] for step in starts], dtype=np.int64)
    next_indices = np.asarray([event_by_step[int(step)] for step in next_frames], dtype=np.int64)
    return starts, next_frames, current_indices, next_indices


def _select_trunk_shared_policy_events(
    *,
    path: Path,
    action_len: int,
    step_index: np.ndarray,
    train_horizon: int,
    chunk_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(step_index)
    sorted_steps = np.asarray(step_index[order], dtype=np.int64)
    candidate_starts = compute_anchor_starts(action_len, train_horizon, chunk_stride)
    candidate_next_frames = candidate_starts + int(train_horizon)
    current_indices = np.searchsorted(sorted_steps, candidate_starts, side="right") - 1
    next_indices = np.searchsorted(sorted_steps, candidate_next_frames, side="right") - 1
    valid = (current_indices >= 0) & (next_indices >= 0)
    if not bool(np.any(valid)):
        raise ValueError(
            f"{path} has no policy-forward events at or before sampled anchors for train_horizon={train_horizon}"
        )
    starts = candidate_starts[valid]
    next_frames = candidate_next_frames[valid]
    current_event_indices = order[current_indices[valid]]
    next_event_indices = order[next_indices[valid]]
    return starts, next_frames, current_event_indices, next_event_indices


def _build_manifest(
    *,
    path: Path,
    arrays: dict[str, np.ndarray],
    starts: np.ndarray,
    next_frames: np.ndarray,
    key_region_id: str,
    reward: float,
    z_rl_source: str,
    z_rl_dim: int,
    proprio_dim: int,
    action_dim: int,
    train_horizon: int,
    chunk_stride: int,
    rl_token_checkpoint_path: str,
    z_alignment: str,
    replay_state_grain: str = "paper_subsampled_anchor",
    subsampled_transition_semantics: str = "x_i_action_i_to_i_plus_c_next_x_i_plus_c",
) -> dict[str, Any]:
    return {
        "key_region_id": key_region_id,
        "reward": reward,
        "source_format": "rlt_timeline_hdf5",
        "source_hdf5_path": str(path.resolve()),
        "replay_state_grain": replay_state_grain,
        "formal_replay_state_grain": replay_state_grain,
        "formal_replay_ready": True,
        "train_eligible": True,
        "subsampled_transition_semantics": subsampled_transition_semantics,
        "z_rl_source": z_rl_source,
        "z_alignment": z_alignment,
        "z_rl_dim": int(z_rl_dim),
        "proprio_dim": int(proprio_dim),
        "action_dim": int(action_dim),
        "train_horizon": int(train_horizon),
        "chunk_stride": int(chunk_stride),
        "current_frames": [int(frame) for frame in starts],
        "next_frames": [int(frame) for frame in next_frames],
        "rl_token_checkpoint_path": rl_token_checkpoint_path,
        "replay_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }


def write_paper_replay_shard_from_timeline_hdf5(
    hdf5_path: str | Path,
    output_path: str | Path,
    *,
    train_horizon: int,
    chunk_stride: int,
    policy_event_alignment: str = DEFAULT_POLICY_EVENT_ALIGNMENT,
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    arrays, manifest = build_paper_replay_from_timeline_hdf5(
        hdf5_path,
        train_horizon=train_horizon,
        chunk_stride=chunk_stride,
        policy_event_alignment=policy_event_alignment,
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


def _read_policy_forward_events(root: h5py.File) -> dict[str, np.ndarray | str]:
    if "rlt_policy_forward_events" not in root:
        raise KeyError(
            f"{root.filename} has neither complete /rlt_timeline z_rl/proprio nor /rlt_policy_forward_events"
        )
    group = root["rlt_policy_forward_events"]
    for name in ("step_index", "z_rl", "proprio"):
        if name not in group:
            raise KeyError(f"{root.filename} is missing rlt_policy_forward_events/{name}")
    z_rl_source = _h5_attr_str(group.attrs.get("z_rl_source", ""))
    if (not z_rl_source or z_rl_source == "missing") and "z_rl_source" in group:
        raw_sources = np.asarray(group["z_rl_source"])
        decoded = {_h5_attr_str(value) for value in raw_sources.tolist()}
        decoded.discard("missing")
        z_rl_source = next(iter(decoded)) if len(decoded) == 1 else "mixed"
    step_index = np.asarray(group["step_index"], dtype=np.int64)
    step_index_semantics = _h5_attr_str(group.attrs.get("step_index_semantics", ""))
    if step_index_semantics != "anchor_observation_step_index" and "action_start_index" in group:
        step_index = step_index - np.asarray(group["action_start_index"], dtype=np.int64)
    return {
        "step_index": step_index,
        "z_rl": np.asarray(group["z_rl"], dtype=np.float32),
        "proprio": np.asarray(group["proprio"], dtype=np.float32),
        "z_rl_source": z_rl_source,
        "step_index_semantics": "anchor_observation_step_index",
    }


def _validate_policy_event_alignment(value: str) -> None:
    if value not in {POLICY_EVENT_ALIGNMENT_EXACT, POLICY_EVENT_ALIGNMENT_TRUNK_SHARED}:
        raise ValueError(
            f"policy_event_alignment must be one of "
            f"{POLICY_EVENT_ALIGNMENT_EXACT!r}, {POLICY_EVENT_ALIGNMENT_TRUNK_SHARED!r}; got {value!r}"
        )


def _validate_action_arrays(*, path: Path, action: np.ndarray, reference_action: np.ndarray) -> None:
    if action.ndim != 2:
        raise ValueError(f"{path} action must have shape [T, action_dim], got {action.shape}")
    if reference_action.shape != action.shape:
        raise ValueError(f"{path} reference_action shape {reference_action.shape} != action shape {action.shape}")
    for name, array in (("action", action), ("reference_action", reference_action)):
        if not np.isfinite(array).all():
            raise ValueError(f"{path} {name} contains non-finite values")


def _validate_policy_forward_events(
    *,
    path: Path,
    action_len: int,
    step_index: np.ndarray,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    z_rl_source: str,
) -> None:
    if step_index.ndim != 1:
        raise ValueError(f"{path} policy forward step_index must be rank-1, got {step_index.shape}")
    if z_rl.ndim != 2 or proprio.ndim != 2:
        raise ValueError(f"{path} policy forward z_rl and proprio must be rank-2 arrays")
    if len(step_index) == 0:
        raise ValueError(f"{path} has no policy-forward events")
    if len({len(step_index), len(z_rl), len(proprio)}) != 1:
        raise ValueError(
            f"{path} policy-forward length mismatch: step_index={len(step_index)} z_rl={len(z_rl)} proprio={len(proprio)}"
        )
    if np.any(step_index < 0) or np.any(step_index >= int(action_len)):
        raise ValueError(f"{path} policy-forward step_index is outside action timeline length {action_len}")
    if len(np.unique(step_index)) != len(step_index):
        raise ValueError(f"{path} policy-forward step_index contains duplicates")
    if not z_rl_source.startswith("vla_same_forward"):
        raise ValueError(f"{path} policy-forward z_rl_source={z_rl_source!r} is not a vla_same_forward source")
    for name, array in (("policy_forward_z_rl", z_rl), ("policy_forward_proprio", proprio)):
        if not np.isfinite(array).all():
            raise ValueError(f"{path} {name} contains non-finite values")


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
    _validate_action_arrays(path=path, action=action, reference_action=reference_action)
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
    for name, array in (("z_rl", z_rl), ("proprio", proprio)):
        if not np.isfinite(array).all():
            raise ValueError(f"{path} {name} contains non-finite values")


def _h5_attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

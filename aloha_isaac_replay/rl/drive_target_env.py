from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos


@dataclasses.dataclass(frozen=True)
class DriveTargetReplayConfig:
    """Configuration shared by replay regression and future RL stepping."""

    side: str = "left"
    replay_mode: str = "left_arm_and_gripper"
    target_hold_steps: int = 1
    max_controlled_error: float = 0.02


@dataclasses.dataclass(frozen=True)
class StepMetrics:
    step_index: int
    controlled_max_abs_error: float
    controlled_rms_error: float
    target_limit_controlled_max_violation: float
    reward_ready: bool


def load_hdf5_qpos(path: str | Path, *, start: int | None = None, end: int | None = None) -> np.ndarray:
    """Load raw ALOHA 14D qpos without applying Isaac side effects."""

    import h5py

    episode = Path(path)
    with h5py.File(episode, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"Expected observations/qpos shape (T, >=14), got {qpos.shape} in {episode}")
    lo = 0 if start is None else int(start)
    hi = len(qpos) if end is None else int(end)
    seq = qpos[lo:hi]
    if seq.shape[0] < 2:
        raise ValueError(f"Need at least two qpos frames, got {seq.shape[0]} from {episode}")
    if not np.isfinite(seq).all():
        raise ValueError(f"HDF5 qpos contains NaN/Inf: {episode}")
    return np.asarray(seq[:, :14], dtype=np.float64)


def tracking_groups(
    dof_names: list[str], *, side: str, replay_mode: str, finger_dof_names: dict[str, str]
) -> dict[str, list[int]]:
    finger_indices = [
        dof_names.index(finger_dof_names["left_finger"]),
        dof_names.index(finger_dof_names["right_finger"]),
    ]
    groups: dict[str, list[int]] = {"gripper": finger_indices}
    if replay_mode == "left_arm_and_gripper":
        base_arm_names = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
        side_arm_names = tuple(f"{side}_{name}" for name in base_arm_names)
        arm_names = side_arm_names if all(name in dof_names for name in side_arm_names) else base_arm_names
        arm_indices = [dof_names.index(name) for name in arm_names if name in dof_names]
        groups["arm"] = arm_indices
        groups["controlled"] = arm_indices + finger_indices
    else:
        groups["controlled"] = finger_indices
    return groups


def tracking_step_errors(*, target: np.ndarray, actual: np.ndarray, groups: dict[str, list[int]]) -> dict[str, dict[str, float]]:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    rows: dict[str, dict[str, float]] = {}
    for name, indices in groups.items():
        if not indices:
            rows[name] = {"max_abs_error": float("nan"), "rms_error": float("nan")}
            continue
        group_error = error[np.asarray(indices, dtype=np.int64)]
        local_max_index = int(np.argmax(np.abs(group_error)))
        rows[name] = {
            "max_abs_error": float(np.max(np.abs(group_error))),
            "max_abs_error_dof_index": int(indices[local_max_index]),
            "max_abs_error_signed": float(group_error[local_max_index]),
            "rms_error": float(np.sqrt(np.mean(np.square(group_error)))),
        }
    return rows


def target_limit_violations(
    *, target: np.ndarray, limits: np.ndarray, groups: dict[str, list[int]]
) -> dict[str, dict[str, float]]:
    target_arr = np.asarray(target, dtype=np.float64)
    limits_arr = np.asarray(limits, dtype=np.float64)
    lower = limits_arr[:, 0]
    upper = limits_arr[:, 1]
    lower_violation = np.maximum(lower - target_arr, 0.0)
    upper_violation = np.maximum(target_arr - upper, 0.0)
    max_violation_by_dof = np.maximum(lower_violation, upper_violation)
    signed_violation_by_dof = np.where(upper_violation > 0.0, upper_violation, -lower_violation)
    rows: dict[str, dict[str, float]] = {}
    for name, indices in groups.items():
        if not indices:
            rows[name] = {"max_violation": float("nan"), "signed_violation": float("nan")}
            continue
        group_violation = max_violation_by_dof[np.asarray(indices, dtype=np.int64)]
        local_max_index = int(np.argmax(group_violation))
        dof_index = int(indices[local_max_index])
        rows[name] = {
            "max_violation": float(group_violation[local_max_index]),
            "max_violation_dof_index": dof_index,
            "signed_violation": float(signed_violation_by_dof[dof_index]),
        }
    return rows


def target_from_standard_qpos(
    *,
    current_target: np.ndarray,
    dof_names: list[str],
    side: str,
    qpos_frame: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
) -> np.ndarray:
    """Convert one raw 14D ALOHA qpos frame into an Isaac full-DOF target."""

    target = np.asarray(current_target, dtype=np.float64).reshape(-1).copy()
    if replay_mode == "left_arm_and_gripper":
        if mapping is None:
            raise ValueError("left_arm_and_gripper replay requires a mapping")
        side_prefix = f"{side}/"
        for arm_target in arm_only_targets_from_standard_qpos(qpos_frame, mapping, side=side):
            if not arm_target.isaac_dof_name.startswith(side_prefix):
                continue
            dof_name = arm_target.isaac_dof_name[len(side_prefix) :]
            target[dof_names.index(dof_name)] = float(arm_target.value)

    channel = 6 if side == "left" else 13
    fingers = standard_gripper_qpos_to_isaac_fingers(float(qpos_frame[channel]), side=side, limits=finger_qpos_limits)
    target[dof_names.index(finger_dof_names["left_finger"])] = float(fingers[f"{side}/left_finger"])
    target[dof_names.index(finger_dof_names["right_finger"])] = float(fingers[f"{side}/right_finger"])
    return target


def targets_from_hdf5_qpos(
    *,
    initial_target: np.ndarray,
    dof_names: list[str],
    side: str,
    qpos: np.ndarray,
    mapping_path: str | Path | None,
    replay_mode: str,
    finger_dof_names: dict[str, str],
    finger_qpos_limits: Any,
) -> list[np.ndarray]:
    mapping = load_mapping(mapping_path) if mapping_path is not None else None
    current = np.asarray(initial_target, dtype=np.float64).reshape(-1)
    return [
        target_from_standard_qpos(
            current_target=current,
            dof_names=dof_names,
            side=side,
            qpos_frame=frame,
            mapping=mapping,
            replay_mode=replay_mode,
            finger_dof_names=finger_dof_names,
            finger_qpos_limits=finger_qpos_limits,
        )
        for frame in qpos
    ]


def summarize_step(
    *,
    step_index: int,
    target: np.ndarray,
    actual: np.ndarray,
    limits: np.ndarray,
    groups: dict[str, list[int]],
    max_controlled_error: float,
) -> StepMetrics:
    tracking = tracking_step_errors(target=target, actual=actual, groups=groups)
    limits_row = target_limit_violations(target=target, limits=limits, groups=groups)
    controlled = tracking["controlled"]
    controlled_limit = limits_row["controlled"]
    reward_ready = bool(
        controlled["max_abs_error"] <= max_controlled_error and controlled_limit["max_violation"] <= 1e-9
    )
    return StepMetrics(
        step_index=int(step_index),
        controlled_max_abs_error=float(controlled["max_abs_error"]),
        controlled_rms_error=float(controlled["rms_error"]),
        target_limit_controlled_max_violation=float(controlled_limit["max_violation"]),
        reward_ready=reward_ready,
    )


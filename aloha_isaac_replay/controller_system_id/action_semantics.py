from __future__ import annotations

import numpy as np

from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES


ARM_ACTION_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)


def require_14d_array(array: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        if values.shape != (14,):
            raise ValueError(f"{name} must have shape (14,), got {values.shape}")
    elif values.ndim == 2:
        if values.shape[1] != 14:
            raise ValueError(f"{name} must have shape (T, 14), got {values.shape}")
    else:
        raise ValueError(f"{name} must be 1D or 2D, got ndim={values.ndim}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return values


def arm_action_from_raw_hdf5_action(action_14d: np.ndarray) -> np.ndarray:
    """Return the 12 arm dimensions from raw HDF5 action without OpenPI transforms."""
    action = require_14d_array(action_14d, name="action_14d")
    if action.ndim != 1:
        raise ValueError(f"action_14d must be one frame, got {action.shape}")
    return action[ARM_ACTION_INDICES].astype(np.float64, copy=True)


def arm_qpos_from_raw_hdf5_qpos(qpos_14d: np.ndarray) -> np.ndarray:
    qpos = require_14d_array(qpos_14d, name="qpos_14d")
    if qpos.ndim == 1:
        return qpos[ARM_ACTION_INDICES].astype(np.float64, copy=True)
    return qpos[:, ARM_ACTION_INDICES].astype(np.float64, copy=True)


def canonical_absolute_targets(actions_14d: np.ndarray) -> np.ndarray:
    """Convert raw HDF5 actions to canonical 12D absolute follower targets.

    This intentionally does not apply OpenPI `adapt_to_pi`, does not subtract state,
    and does not integrate action as a delta. The HDF5 arm action is already the
    runtime absolute command in standard ALOHA order.
    """
    actions = require_14d_array(actions_14d, name="actions_14d")
    if actions.ndim != 2:
        raise ValueError(f"actions_14d must be a sequence, got {actions.shape}")
    return actions[:, ARM_ACTION_INDICES].astype(np.float64, copy=True)


def canonical_arm_names() -> tuple[str, ...]:
    return ARM_ONLY_NAMES


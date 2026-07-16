from __future__ import annotations

import numpy as np


STANDARD_ALOHA_14D_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)

OPENPI_JOINT_FLIP_MASK = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float64)


def require_aloha_14d(values: np.ndarray, *, name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 14:
        raise ValueError(f"{name} must have final dimension 14, got shape {array.shape}")
    return array


def standard_to_openpi_internal(values: np.ndarray) -> np.ndarray:
    """Convert standard ALOHA 14D to current OpenPI internal 14D.

    This mirrors the active local OpenPI transform: sign flip only. The gripper angular
    conversion code is intentionally not applied because it is commented out in the
    current local `src/openpi/policies/aloha_policy.py`.
    """
    array = require_aloha_14d(values)
    return array * OPENPI_JOINT_FLIP_MASK


def openpi_internal_to_standard(values: np.ndarray) -> np.ndarray:
    array = require_aloha_14d(values)
    return array * OPENPI_JOINT_FLIP_MASK


def split_left_right(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = require_aloha_14d(values)
    return array[..., :7], array[..., 7:]


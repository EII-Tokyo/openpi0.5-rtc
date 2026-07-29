"""Product-of-exponentials FK reference for the Interbotix ALOHA ViperX 300S."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

SOURCE_CLASS = "aloha_vx300s"
SOURCE_FILE = (
    "external/ros2-essentials/aloha_ws/src/interbotix_ros_toolboxes/"
    "interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/"
    "xs_robot/mr_descriptions.py"
)
SOURCE_SHA256 = "9412f1496f0cf1f3e23995ba3f0c10f250624cdd3798274a7191b1cad6248388"

SLIST = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
    ],
    dtype=np.float64,
).T
Slist = SLIST

M = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.536494],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.42705],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _finite_six_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,):
        raise ValueError(f"{name} must have shape (6,)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def vec_to_se3(twist: Any) -> np.ndarray:
    """Map a finite six-vector ``[omega, v]`` to its se(3) matrix."""
    vector = _finite_six_vector(twist, name="twist")
    omega_x, omega_y, omega_z, velocity_x, velocity_y, velocity_z = vector
    return np.asarray(
        [
            [0.0, -omega_z, omega_y, velocity_x],
            [omega_z, 0.0, -omega_x, velocity_y],
            [-omega_y, omega_x, 0.0, velocity_z],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def fk_space(q: Any) -> np.ndarray:
    """Return the ALOHA ViperX 300S end-effector pose for six joint angles."""
    joint_angles = _finite_six_vector(q, name="q")
    transform = np.eye(4, dtype=np.float64)
    for joint_index, joint_angle in enumerate(joint_angles):
        transform = transform @ expm(
            vec_to_se3(SLIST[:, joint_index]) * joint_angle
        )
    return transform @ M

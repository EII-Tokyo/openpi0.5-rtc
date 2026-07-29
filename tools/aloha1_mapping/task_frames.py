from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ClosureError:
    translation_m: float
    rotation_rad: float


def rigid_transform(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> np.ndarray:
    rotation_value = np.asarray(rotation, dtype=np.float64)
    translation_value = np.asarray(translation, dtype=np.float64)
    if rotation_value.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    if translation_value.shape != (3,):
        raise ValueError("translation must contain exactly three values")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_value
    matrix[:3, 3] = translation_value
    return validate_rigid_transform(matrix)


def validate_rigid_transform(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("rigid transform must be finite 4x4")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        raise ValueError("invalid homogeneous row")

    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-10):
        raise ValueError("rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"rotation determinant is {determinant}")
    return value


def tabletop_task_frame(
    *,
    table_center_world_m: Sequence[float],
    table_size_world_m: Sequence[float],
) -> np.ndarray:
    center = np.asarray(table_center_world_m, dtype=np.float64)
    size = np.asarray(table_size_world_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("table center must contain exactly three finite values")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("table size must contain exactly three positive finite values")
    origin = center.copy()
    origin[2] += size[2] / 2.0
    return rigid_transform(np.eye(3), origin)


def closure_error(
    expected: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
) -> ClosureError:
    expected_value = validate_rigid_transform(expected)
    observed_value = validate_rigid_transform(observed)
    delta = np.linalg.inv(expected_value) @ observed_value
    return ClosureError(
        translation_m=float(np.linalg.norm(delta[:3, 3])),
        rotation_rad=float(Rotation.from_matrix(delta[:3, :3]).magnitude()),
    )

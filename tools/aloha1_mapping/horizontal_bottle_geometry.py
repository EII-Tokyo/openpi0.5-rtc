"""Pure geometry gates for the ALOHA1 horizontal Bottle500 task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

_EPSILON = 1e-12


@dataclass(frozen=True)
class DirectedAxis:
    a_world: tuple[float, float, float]
    b_world: tuple[float, float, float]
    unit: tuple[float, float, float]
    length_m: float


@dataclass(frozen=True)
class HorizontalPlacement:
    matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    a_world: tuple[float, float, float]
    b_world: tuple[float, float, float]
    axis_unit: tuple[float, float, float]
    lowest_point_world_z: float


def _vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be one 3-vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def _unit(value: Any, *, name: str) -> np.ndarray:
    vector = _vector(value, name=name)
    length = float(np.linalg.norm(vector))
    if length <= _EPSILON:
        raise ValueError(f"{name} must not be zero-length")
    return vector / length


def _affine_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be one 4x4 affine matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("transform must be finite")
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=_EPSILON,
    ):
        raise ValueError("transform bottom row must be affine")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if determinant <= _EPSILON:
        raise ValueError("transform determinant must be positive")
    return matrix


def _rotation_matrix(value: Any) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be one 3x3 matrix")
    if not np.isfinite(rotation).all():
        raise ValueError("rotation must be finite")
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("rotation determinant must be positive one")
    return rotation


def transform_points(points: Any, transform: Any) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    if (
        point_array.ndim != 2
        or point_array.shape[1] != 3
        or point_array.shape[0] == 0
    ):
        raise ValueError("points must be a non-empty Nx3 array")
    if not np.isfinite(point_array).all():
        raise ValueError("points must be finite")
    matrix = _affine_matrix(transform)
    homogeneous = np.column_stack(
        [point_array, np.ones(point_array.shape[0], dtype=np.float64)]
    )
    return (matrix @ homogeneous.T).T[:, :3]


def transform_directed_axis(
    a_local: Any,
    b_local: Any,
    transform: Any,
) -> DirectedAxis:
    local_a = _vector(a_local, name="axis A")
    local_b = _vector(b_local, name="axis B")
    if float(np.linalg.norm(local_b - local_a)) <= _EPSILON:
        raise ValueError("axis must not be zero-length")
    world = transform_points(np.stack([local_a, local_b]), transform)
    direction = world[1] - world[0]
    length = float(np.linalg.norm(direction))
    if length <= _EPSILON:
        raise ValueError("transformed axis must not be zero-length")
    unit = direction / length
    return DirectedAxis(
        a_world=tuple(float(value) for value in world[0]),
        b_world=tuple(float(value) for value in world[1]),
        unit=tuple(float(value) for value in unit),
        length_m=length,
    )


def canonical_bottle_axis(gripper_line: Any) -> np.ndarray:
    line = _vector(gripper_line, name="gripper line").copy()
    line[2] = 0.0
    length = float(np.linalg.norm(line))
    if length <= _EPSILON:
        raise ValueError("gripper line XY projection must not be zero")
    line /= length
    candidate = np.cross(line, np.asarray([0.0, 0.0, 1.0]))
    candidate /= np.linalg.norm(candidate)
    if candidate[0] < -_EPSILON or (
        abs(float(candidate[0])) <= _EPSILON
        and candidate[1] < 0.0
    ):
        candidate = -candidate
    candidate[2] = 0.0
    return candidate


def shortest_arc_rotation(source: Any, target: Any) -> np.ndarray:
    source_unit = _unit(source, name="source axis")
    target_unit = _unit(target, name="target axis")
    cosine = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if cosine >= 1.0 - _EPSILON:
        return np.eye(3, dtype=np.float64)
    if cosine <= -1.0 + _EPSILON:
        basis = np.eye(3)[int(np.argmin(np.abs(source_unit)))]
        axis = np.cross(source_unit, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    cross = np.cross(source_unit, target_unit)
    sine_squared = float(np.dot(cross, cross))
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + skew
        + (skew @ skew) * ((1.0 - cosine) / sine_squared)
    )


def point_on_axis(
    a_world: Any,
    axis_unit: Any,
    coordinate_m: float,
) -> list[float]:
    start = _vector(a_world, name="axis A")
    direction = _unit(axis_unit, name="axis unit")
    coordinate = float(coordinate_m)
    if not math.isfinite(coordinate):
        raise ValueError("axis coordinate must be finite")
    return [
        float(value)
        for value in start + coordinate * direction
    ]


def derive_horizontal_support_placement(
    *,
    local_collision_points: Any,
    rotation: Any,
    grasp_center_world_xy: Any,
    grasp_coordinate_m: float,
    table_top_z: float,
    setup_gap_m: float,
    axis_a_local: Any,
    axis_b_local: Any,
) -> HorizontalPlacement:
    points = np.asarray(local_collision_points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("local collision points must be one finite Nx3 array")
    orientation = _rotation_matrix(rotation)
    grasp_xy = np.asarray(grasp_center_world_xy, dtype=np.float64)
    if grasp_xy.shape != (2,) or not np.isfinite(grasp_xy).all():
        raise ValueError("grasp center XY must be one finite 2-vector")
    table_z = float(table_top_z)
    setup_gap = float(setup_gap_m)
    coordinate = float(grasp_coordinate_m)
    if not math.isfinite(table_z):
        raise ValueError("table top must be finite")
    if not math.isfinite(setup_gap) or setup_gap < 0.0:
        raise ValueError("setup gap must be finite and nonnegative")
    if not math.isfinite(coordinate) or coordinate < 0.0:
        raise ValueError("grasp coordinate must be finite and nonnegative")

    local_a = _vector(axis_a_local, name="axis A")
    local_b = _vector(axis_b_local, name="axis B")
    local_direction = local_b - local_a
    local_length = float(np.linalg.norm(local_direction))
    if local_length <= _EPSILON:
        raise ValueError("axis must not be zero-length")
    if coordinate > local_length + _EPSILON:
        raise ValueError("grasp coordinate must lie on the directed axis")
    local_grasp_point = (
        local_a + coordinate * local_direction / local_length
    )

    rotated_points = (orientation @ points.T).T
    rotated_grasp = orientation @ local_grasp_point
    translation = np.asarray(
        [
            grasp_xy[0] - rotated_grasp[0],
            grasp_xy[1] - rotated_grasp[1],
            table_z + setup_gap - float(rotated_points[:, 2].min()),
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = orientation
    matrix[:3, 3] = translation

    axis = transform_directed_axis(local_a, local_b, matrix)
    lowest = float(transform_points(points, matrix)[:, 2].min())
    return HorizontalPlacement(
        matrix=tuple(
            tuple(float(value) for value in row)
            for row in matrix
        ),
        a_world=axis.a_world,
        b_world=axis.b_world,
        axis_unit=axis.unit,
        lowest_point_world_z=lowest,
    )


def angle_degrees(first: Any, second: Any) -> float:
    first_unit = _unit(first, name="first vector")
    second_unit = _unit(second, name="second vector")
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def evaluate_geometry(
    *,
    axis_unit: Any,
    table_normal: Any,
    gripper_line: Any,
    approach_delta: Any,
    axis_vertical_angle_gate_deg: float,
    gripper_perpendicular_gate_deg: float,
    approach_direction_gate_deg: float,
) -> dict[str, Any]:
    axis = _unit(axis_unit, name="bottle axis")
    normal = _unit(table_normal, name="table normal")
    gripper = _vector(gripper_line, name="gripper line")
    approach = _unit(approach_delta, name="approach delta")
    axis_xy = axis.copy()
    axis_xy[2] = 0.0
    gripper_xy = gripper.copy()
    gripper_xy[2] = 0.0

    axis_to_normal = angle_degrees(axis, normal)
    gripper_to_axis = angle_degrees(gripper_xy, axis_xy)
    approach_to_negative_z = angle_degrees(
        approach,
        [0.0, 0.0, -1.0],
    )
    failed: list[str] = []
    if abs(axis_to_normal - 90.0) > float(axis_vertical_angle_gate_deg):
        failed.append("axis_horizontal")
    if abs(gripper_to_axis - 90.0) > float(
        gripper_perpendicular_gate_deg
    ):
        failed.append("gripper_axis_perpendicular")
    if approach_to_negative_z > float(approach_direction_gate_deg):
        failed.append("vertical_approach")
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_gates": failed,
        "axis_to_table_normal_deg": axis_to_normal,
        "gripper_line_to_axis_deg": gripper_to_axis,
        "approach_to_negative_z_deg": approach_to_negative_z,
    }

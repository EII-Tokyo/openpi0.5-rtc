"""Pure geometry measurements for follower-left/table collision validation."""

from __future__ import annotations

import math
from typing import Sequence


Point3 = tuple[float, float, float]
Matrix4 = Sequence[Sequence[float]]


def points_in_table_footprint(
    world_points: Sequence[Point3],
    table_from_world: Matrix4,
    half_extents_xy: tuple[float, float],
) -> list[Point3]:
    """Transform world points and retain those inside the table's local XY."""

    if len(table_from_world) != 4 or any(len(row) != 4 for row in table_from_world):
        raise ValueError("table_from_world must be 4x4")
    if (
        len(half_extents_xy) != 2
        or not all(math.isfinite(value) and value >= 0 for value in half_extents_xy)
    ):
        raise ValueError("half_extents_xy must be finite and nonnegative")

    result: list[Point3] = []
    for point in world_points:
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("world point must contain three finite values")
        vector = (*point, 1.0)
        transformed = [
            sum(
                float(table_from_world[row][column]) * vector[column]
                for column in range(4)
            )
            for row in range(4)
        ]
        if not all(math.isfinite(value) for value in transformed) or transformed[3] == 0:
            raise ValueError("non-finite homogeneous point")
        local = tuple(transformed[index] / transformed[3] for index in range(3))
        if (
            abs(local[0]) <= half_extents_xy[0]
            and abs(local[1]) <= half_extents_xy[1]
        ):
            result.append(local)
    return result


def minimum_local_z(points: Sequence[Point3]) -> float:
    """Return the lowest finite table-local Z from a nonempty point set."""

    if not points:
        raise ValueError("no points inside table footprint")
    values = [point[2] for point in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("local point Z must be finite")
    return min(values)


def maximum_point_error(
    visual_points: Sequence[Point3],
    collision_points: Sequence[Point3],
) -> float:
    """Return maximum ordered Euclidean error for corresponding mesh vertices."""

    if not visual_points or len(visual_points) != len(collision_points):
        raise ValueError("visual/collision topology mismatch")
    errors = []
    for visual, collision in zip(visual_points, collision_points, strict=True):
        if (
            len(visual) != 3
            or len(collision) != 3
            or not all(math.isfinite(value) for value in (*visual, *collision))
        ):
            raise ValueError("visual/collision points must be finite 3D values")
        errors.append(math.dist(visual, collision))
    return max(errors)

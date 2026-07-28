"""Numerical convex-hull pair audit helpers for Task 5 diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection


def _normalized_halfspaces(points: np.ndarray) -> tuple[np.ndarray, ConvexHull]:
    hull = ConvexHull(points)
    equations = np.asarray(hull.equations, dtype=np.float64).copy()
    norms = np.linalg.norm(equations[:, :3], axis=1)
    equations[:, :3] /= norms[:, None]
    equations[:, 3] /= norms
    return equations, hull


def convex_pair_relation(
    points_a: Sequence[Sequence[float]],
    points_b: Sequence[Sequence[float]],
    *,
    tolerance_m: float = 1.0e-8,
) -> dict[str, Any]:
    """Classify two cooked convex hulls by a normalized halfspace LP.

    The signed Chebyshev margin is positive for a volumetric intersection,
    zero at touching, and negative for separation. For intersection, the
    combined halfspaces are tessellated to obtain overlap volume.
    """

    array_a = np.asarray(points_a, dtype=np.float64)
    array_b = np.asarray(points_b, dtype=np.float64)
    equations_a, hull_a = _normalized_halfspaces(array_a)
    equations_b, hull_b = _normalized_halfspaces(array_b)
    equations = np.vstack([equations_a, equations_b])
    constraints = np.column_stack(
        [equations[:, :3], np.ones(len(equations))]
    )
    result = linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=constraints,
        b_ub=-equations[:, 3],
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"convex-pair LP failed: {result.message}")
    margin = float(result.x[3])
    overlap_volume = 0.0
    intersection_vertex_count = 0
    if margin > tolerance_m:
        intersection = HalfspaceIntersection(
            equations,
            np.asarray(result.x[:3], dtype=np.float64),
        )
        intersection_points = np.asarray(
            intersection.intersections,
            dtype=np.float64,
        )
        intersection_vertex_count = len(intersection_points)
        if intersection_vertex_count >= 4:
            overlap_volume = float(ConvexHull(intersection_points).volume)
        relation = "OVERLAP"
    elif margin < -tolerance_m:
        relation = "SEPARATED"
    else:
        relation = "TOUCHING_WITHIN_TOLERANCE"
    return {
        "relation": relation,
        "signed_chebyshev_margin_m": margin,
        "overlap_volume_m3": overlap_volume,
        "intersection_vertex_count": intersection_vertex_count,
        "tolerance_m": tolerance_m,
        "method": (
            "normalized convex-hull halfspace feasibility LP; positive "
            "margin means volumetric intersection"
        ),
        "hull_a": {
            "input_point_count": len(array_a),
            "vertex_count": len(hull_a.vertices),
            "face_count": len(hull_a.simplices),
            "volume_m3": float(hull_a.volume),
        },
        "hull_b": {
            "input_point_count": len(array_b),
            "vertex_count": len(hull_b.vertices),
            "face_count": len(hull_b.simplices),
            "volume_m3": float(hull_b.volume),
        },
    }


def collider_summary(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Summarize the convex hull that PhysX is requested to cook."""

    array = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(array)
    return {
        "source_point_count": len(array),
        "cooked_convex_vertex_count": len(hull.vertices),
        "cooked_convex_face_count": len(hull.simplices),
        "cooked_convex_volume_m3": float(hull.volume),
        "aabb_min_world_m": array.min(axis=0).tolist(),
        "aabb_max_world_m": array.max(axis=0).tolist(),
    }

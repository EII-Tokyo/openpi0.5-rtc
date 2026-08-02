"""Pure geometry construction for a contact-preserving compound collider."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError


def _unit(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(result))
    if length == 0.0:
        raise ValueError("normal must be non-zero")
    return result / length


def _deduplicate(points: list[np.ndarray], tolerance_m: float) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    scale = max(tolerance_m, np.finfo(float).eps)
    unique: dict[tuple[int, int, int], np.ndarray] = {}
    for point in points:
        key = tuple(round(float(value) / scale) for value in point)
        unique.setdefault(key, point)
    return np.asarray(list(unique.values()), dtype=np.float64)


def convex_triangle_topology(vertices: np.ndarray) -> dict[str, Any]:
    """Return deterministic outward-wound triangle topology for a convex point set."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("convex topology requires at least four 3-D vertices")
    hull = ConvexHull(points)
    centroid = points.mean(axis=0)
    triangles: list[tuple[int, int, int]] = []
    for simplex in hull.simplices:
        first, second, third = (int(index) for index in simplex)
        normal = np.cross(points[second] - points[first], points[third] - points[first])
        if float(np.dot(normal, centroid - points[first])) > 0.0:
            second, third = third, second
        triangles.append((first, second, third))
    triangles.sort(key=lambda value: tuple(sorted(value)))
    return {
        "face_vertex_counts": [3] * len(triangles),
        "face_vertex_indices": [index for triangle in triangles for index in triangle],
        "face_count": len(triangles),
        "volume_m3": float(hull.volume),
    }


def canonical_runtime_cooking_signature(report: dict[str, Any]) -> str:
    """Hash only cooked geometry/readback, excluding process and timing metadata."""
    payload = {
        "fingers": {
            side: {
                "pieces": [
                    {
                        "source_piece_index": piece["source_piece_index"],
                        "approximation_readback": piece["approximation_readback"],
                        "cooked": piece["cooked"],
                    }
                    for piece in finger["pieces"]
                ]
            }
            for side, finger in sorted(report["fingers"].items())
        }
    }

    def json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_contact_region_status(certificate: dict[str, Any], numeric_tolerance_m: float) -> str:
    """Apply the predeclared CAD contact-region coverage/crossing gate."""
    crossing = float(certificate.get("positive_exit_distance_max_m") or 0.0)
    coverage = certificate.get(
        "tolerance_adjusted_coverage_ratio",
        certificate["source_point_coverage_ratio"],
    )
    return "PASS" if coverage == 1.0 and crossing <= numeric_tolerance_m else "FAIL"


def tolerance_adjusted_contact_coverage(certificate: dict[str, Any], *, numeric_tolerance_m: float) -> dict[str, Any]:
    """Classify exact-ray misses caused only by bounded float32 cooking error."""
    adjusted = dict(certificate)
    uncovered = int(certificate["uncovered_count"])
    within_distance = uncovered == 0 or float(certificate["uncovered_nearest_surface_max_m"]) <= numeric_tolerance_m
    projection_values = (
        certificate.get("uncovered_nearest_surface_normal_projection_min_m"),
        certificate.get("uncovered_nearest_surface_normal_projection_max_m"),
    )
    within_normal_projection = uncovered == 0 or all(
        value is not None and abs(float(value)) <= numeric_tolerance_m for value in projection_values
    )
    quantization_boundary_count = uncovered if within_distance and within_normal_projection else 0
    adjusted_count = int(certificate["source_point_covered_count"]) + (quantization_boundary_count)
    sample_count = int(certificate["contact_sample_count"])
    adjusted.update(
        {
            "exact_ray_coverage_ratio": certificate["source_point_coverage_ratio"],
            "tolerance_adjusted_covered_count": adjusted_count,
            "tolerance_adjusted_coverage_ratio": (adjusted_count / sample_count if sample_count else 0.0),
            "quantization_boundary_sample_count": quantization_boundary_count,
            "coverage_numeric_tolerance_m": numeric_tolerance_m,
            "coverage_adjustment_rule": (
                "EXACT_RAY_OR_NEAREST_SURFACE_DISTANCE_AND_NORMAL_PROJECTION_WITHIN_DERIVED_NUMERIC_TOLERANCE"
            ),
        }
    )
    return adjusted


def classify_fresh_runtime_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Require two passing fresh processes with identical cooked geometry."""
    if len(runs) != 2:
        raise ValueError("exactly two runtime cooking reports are required")
    process_ids = [int(run["process_id"]) for run in runs]
    signatures = [str(run["deterministic_signature"]) for run in runs]
    fresh = len(set(process_ids)) == 2
    matching = len(set(signatures)) == 1
    passing = all(run["status"] == "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED" for run in runs)
    return {
        "status": (
            "PASS_DETERMINISTIC_FRESH_PROCESS_COOKING"
            if fresh and matching and passing
            else "FAIL_DETERMINISTIC_FRESH_PROCESS_COOKING"
        ),
        "fresh_processes": fresh,
        "matching_geometry_signatures": matching,
        "all_runtime_contact_region_gates_pass": passing,
        "process_ids": process_ids,
        "geometry_signatures": signatures,
    }


def compound_piece_prim_path(side: str, index: int) -> str:
    """Return the explicit geometry-only diagnostic prim path for one piece."""
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported finger side: {side}")
    if index < 0:
        raise ValueError("piece index must be non-negative")
    return f"/CadFingerCompoundContactCandidate/{side}_finger/piece_{index:03d}"


def transform_contact_candidate(candidate: dict[str, Any], matrix: np.ndarray) -> dict[str, Any]:
    """Apply one proper rigid transform to compound geometry and its audit frame."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("contact candidate transform must be 4x4")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12):
        raise ValueError("contact candidate transform rotation is not orthonormal")
    if not np.isclose(determinant, 1.0, atol=1.0e-12):
        raise ValueError("contact candidate transform must not mirror geometry")

    def points(values: Any) -> list[list[float]]:
        source = np.asarray(values, dtype=np.float64)
        transformed = source @ rotation.T + transform[:3, 3]
        return transformed.tolist()

    result = dict(candidate)
    transformed_pieces = []
    for piece in candidate["pieces"]:
        transformed_piece = dict(piece)
        transformed_piece["vertices"] = points(piece["vertices"])
        transformed_pieces.append(transformed_piece)
    result.update(
        {
            "pieces": transformed_pieces,
            "outward_normal": _unit(rotation @ np.asarray(candidate["outward_normal"], dtype=np.float64)).tolist(),
            "plane_point_m": points([candidate["plane_point_m"]])[0],
            "contact_rectangle_vertices_m": points(candidate["contact_rectangle_vertices_m"]),
            "rigid_transform_matrix": transform.tolist(),
            "rigid_transform_determinant": determinant,
            "mirror_used": False,
        }
    )
    return result


def clip_convex_piece_to_halfspace(
    piece: dict[str, Any],
    *,
    plane_point: np.ndarray,
    outward_normal: np.ndarray,
    numeric_tolerance_m: float,
) -> dict[str, Any] | None:
    """Clip one convex piece to ``n·(x-p)<=0`` without fitting geometry."""
    vertices = np.asarray(piece["vertices"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("convex piece requires at least four 3-D vertices")
    point = np.asarray(plane_point, dtype=np.float64)
    normal = _unit(outward_normal)
    hull = ConvexHull(vertices)
    signed = (vertices - point) @ normal
    candidates = []
    for vertex, distance in zip(vertices, signed, strict=True):
        if distance <= numeric_tolerance_m:
            candidate = vertex - normal * distance if distance > 0.0 else vertex
            candidates.append(candidate)

    edges = {
        tuple(sorted((int(first), int(second))))
        for simplex in hull.simplices
        for first, second in (
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        )
    }
    for first, second in sorted(edges):
        first_distance = float(signed[first])
        second_distance = float(signed[second])
        if (first_distance < 0.0 < second_distance) or (second_distance < 0.0 < first_distance):
            weight = first_distance / (first_distance - second_distance)
            intersection = vertices[first] + weight * (vertices[second] - vertices[first])
            candidates.append(intersection)

    clipped = _deduplicate(candidates, numeric_tolerance_m)
    if len(clipped) < 4:
        return None
    try:
        clipped_hull = ConvexHull(clipped)
    except QhullError:
        return None
    if float(clipped_hull.volume) <= numeric_tolerance_m**3:
        return None
    return {
        "vertices": clipped.tolist(),
        "source_vertex_count": len(vertices),
        "clipped_vertex_count": len(clipped),
        "volume_m3": float(clipped_hull.volume),
        "maximum_plane_signed_distance_m": float(np.max((clipped - point) @ normal)),
        "construction": "CONVEX_HALFSPACE_INTERSECTION",
    }


def triangular_contact_prism(
    triangle: np.ndarray,
    *,
    outward_normal: np.ndarray,
    depth_m: float,
) -> dict[str, Any]:
    """Extrude a planar contact triangle only toward the finger interior."""
    front = np.asarray(triangle, dtype=np.float64)
    if front.shape != (3, 3):
        raise ValueError("contact triangle must have shape (3, 3)")
    if depth_m <= 0.0:
        raise ValueError("contact prism depth must be positive")
    normal = _unit(outward_normal)
    back = front - normal * depth_m
    vertices = np.vstack((front, back))
    hull = ConvexHull(vertices)
    return {
        "vertices": vertices.tolist(),
        "front_vertex_count": 3,
        "depth_m": float(depth_m),
        "volume_m3": float(hull.volume),
        "construction": "CAD_CONTACT_TRIANGLE_INWARD_PRISM",
    }


def build_contact_preserving_candidate(
    *,
    cooked_pieces: list[dict[str, Any]],
    contact_triangles: np.ndarray,
    plane_point: np.ndarray,
    outward_normal: np.ndarray,
    contact_prism_depth_m: float,
    numeric_tolerance_m: float,
) -> dict[str, Any]:
    """Clip cooked body pieces and add exact-plane inward contact prisms."""
    clipped = [
        piece
        for source_piece in cooked_pieces
        if (
            piece := clip_convex_piece_to_halfspace(
                source_piece,
                plane_point=plane_point,
                outward_normal=outward_normal,
                numeric_tolerance_m=numeric_tolerance_m,
            )
        )
        is not None
    ]
    prisms = [
        triangular_contact_prism(
            triangle,
            outward_normal=outward_normal,
            depth_m=contact_prism_depth_m,
        )
        for triangle in np.asarray(contact_triangles, dtype=np.float64)
    ]
    return {
        "pieces": clipped + prisms,
        "source_cooked_piece_count": len(cooked_pieces),
        "clipped_body_piece_count": len(clipped),
        "discarded_body_piece_count": len(cooked_pieces) - len(clipped),
        "contact_prism_piece_count": len(prisms),
        "piece_count": len(clipped) + len(prisms),
        "contact_prism_depth_m": float(contact_prism_depth_m),
        "final_or_default_collider_modified": False,
    }

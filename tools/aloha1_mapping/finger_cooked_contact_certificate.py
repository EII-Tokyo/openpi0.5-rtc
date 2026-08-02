"""Measure cooked convex-union envelope along an authoritative contact normal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from tools.aloha1_mapping.collider_surface_certificate import FINGER_HASHES
from tools.aloha1_mapping.collider_surface_certificate import FINGER_PATHS
from tools.aloha1_mapping.collider_surface_certificate import _load_obj
from tools.aloha1_mapping.collider_surface_certificate import _sha256
from tools.aloha1_mapping.collider_surface_certificate import _surface_samples

EXPECTED_SUPPLIER_STEP_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
EXPECTED_BREP_FACE_INDICES = {"left": 117, "right": 128}


def load_exact_brep_contact_surface(
    report_paths: list[Path], side: str
) -> dict[str, Any]:
    """Load two deterministic OCCT B-Rep face-sampling process reports."""
    if side not in EXPECTED_BREP_FACE_INDICES:
        raise ValueError(f"unsupported finger side: {side}")
    if len(report_paths) != 2:
        raise ValueError("exactly two fresh FreeCAD reports are required")
    loaded = [
        (
            path.resolve(strict=True),
            json.loads(path.read_text(encoding="utf-8")),
        )
        for path in report_paths
    ]
    reports = [report for _, report in loaded]
    if any(report["status"] != "PASS" for report in reports):
        raise ValueError("a B-Rep contact sampling report did not pass")
    if any(
        report["classification"] != "EXACT_OCCT_BREP_CONTACT_FACE_SAMPLES"
        for report in reports
    ):
        raise ValueError("unexpected B-Rep sampling classification")
    if len({report["process_id"] for report in reports}) != 2:
        raise ValueError("B-Rep reports do not prove distinct processes")
    if len({report["deterministic_signature"] for report in reports}) != 1:
        raise ValueError("B-Rep contact samples are not deterministic")
    if any(
        report["source"]["sha256"] != EXPECTED_SUPPLIER_STEP_SHA256
        for report in reports
    ):
        raise ValueError("supplier STEP hash mismatch")
    if any(
        report["toolchain"]["required_freecad"] != "1.1.1"
        or report["toolchain"]["required_opencascade"] != "7.8.1"
        or report["toolchain"]["opencascade"] != "7.8.1"
        for report in reports
    ):
        raise ValueError("unexpected FreeCAD/OCCT runtime")
    if any(
        report["sampling"]["no_tessellation_used_for_points"] is not True
        for report in reports
    ):
        raise ValueError("B-Rep samples were replaced by tessellation samples")

    record = reports[0]["fingers"][side]
    if record["face_index_1_based"] != EXPECTED_BREP_FACE_INDICES[side]:
        raise ValueError(f"unexpected audited B-Rep face index: {side}")
    samples_m = np.asarray(record["samples_mm"], dtype=np.float64) * 0.001
    if samples_m.shape != (record["sample_count"], 3):
        raise ValueError("invalid B-Rep contact sample shape")
    normal = np.asarray(record["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return {
        "side": side,
        "samples_m": samples_m,
        "sample_count": len(samples_m),
        "normal": normal,
        "face_index_1_based": record["face_index_1_based"],
        "brep_membership_tolerance_m": (
            float(record["uv_grid"]["membership_tolerance_mm"]) * 0.001
        ),
        "source_geometry": "trimmed OCCT B-Rep face",
        "source_sha256": EXPECTED_SUPPLIER_STEP_SHA256,
        "fresh_process_count": 2,
        "deterministic_signature": reports[0]["deterministic_signature"],
        "reports": [str(path) for path, _ in loaded],
    }


def derive_cooked_brep_numeric_tolerance(
    points_m: np.ndarray,
    *,
    brep_membership_tolerance_m: float,
) -> dict[str, Any]:
    """Derive a comparison floor from OCCT and float32 coordinate precision."""
    points = np.asarray(points_m, dtype=np.float64)
    maximum_coordinate_m = float(np.max(np.abs(points)))
    float32_ulp_m = float(np.spacing(np.float32(maximum_coordinate_m)))
    float32_allowance_m = 8.0 * abs(float32_ulp_m)
    tolerance_m = max(brep_membership_tolerance_m, float32_allowance_m)
    return {
        "numeric_tolerance_m": tolerance_m,
        "brep_membership_tolerance_m": brep_membership_tolerance_m,
        "maximum_sample_coordinate_m": maximum_coordinate_m,
        "float32_ulp_at_maximum_coordinate_m": float32_ulp_m,
        "float32_quantization_allowance_m": float32_allowance_m,
        "float32_ulp_multiplier": 8,
        "derivation": "MAX(BREP_MEMBERSHIP_TOLERANCE,8_FLOAT32_ULP)",
    }


def classify_exact_brep_profiles(
    profiles_by_side: dict[str, dict[str, dict[str, Any]]],
    *,
    numeric_tolerance_m: float,
) -> dict[str, Any]:
    """Classify exact-face crossing separately from task-local usability."""
    if set(profiles_by_side) != {"left", "right"}:
        raise ValueError("both supplier-CAD handed fingers are required")
    crossing_records = []
    improvements = []
    for side, profiles in profiles_by_side.items():
        if set(profiles) != {"convexHull", "convexDecomposition"}:
            raise ValueError(f"both collision profiles are required for {side}")
        hull = float(profiles["convexHull"]["maximum_inward_crossing_m"])
        decomposition = float(
            profiles["convexDecomposition"]["maximum_inward_crossing_m"]
        )
        improvements.append(decomposition < hull)
        for approximation, crossing in (
            ("convexHull", hull),
            ("convexDecomposition", decomposition),
        ):
            crossing_records.append(
                {
                    "side": side,
                    "approximation": approximation,
                    "maximum_inward_crossing_m": crossing,
                    "crosses_beyond_numeric_tolerance": (
                        crossing > numeric_tolerance_m
                    ),
                }
            )
    crossing_count = sum(
        record["crosses_beyond_numeric_tolerance"]
        for record in crossing_records
    )
    if crossing_count == len(crossing_records):
        exact_surface_status = "ALL_PROFILES_CROSS_INWARD_CAD_SURFACE"
    elif crossing_count:
        exact_surface_status = "SOME_PROFILES_CROSS_INWARD_CAD_SURFACE"
    else:
        exact_surface_status = "NO_SAMPLED_INWARD_SURFACE_CROSSING"
    if all(improvements):
        decomposition_comparison = "DECOMPOSITION_REDUCES_CROSSING_BOTH_SIDES"
    elif any(improvements):
        decomposition_comparison = "DECOMPOSITION_MIXED_OR_WORSE"
    else:
        decomposition_comparison = "DECOMPOSITION_WORSE_OR_EQUAL_BOTH_SIDES"
    return {
        "exact_surface_status": exact_surface_status,
        "crossing_profile_count": crossing_count,
        "profile_count": len(crossing_records),
        "decomposition_comparison": decomposition_comparison,
        "crossing_records": crossing_records,
        "numeric_tolerance_m": numeric_tolerance_m,
        "asset_decision": (
            "REJECTED_EXACT_CAD_CONTACT_GATE"
            if crossing_count
            else "EXACT_GATE_PASS_NOT_PROMOTED"
        ),
        "task_local_approximate_collider_acceptance": (
            "HARD_BLOCKER_TASK_LOCAL_APPROXIMATION_TOLERANCE"
        ),
        "final_or_default_collider_modified": False,
    }


def load_supplier_contact_surface(root: Path, side: str) -> dict[str, Any]:
    suffix = f"{side}_finger_link"
    if suffix not in FINGER_PATHS:
        raise ValueError(f"unsupported finger side: {side}")
    root = root.resolve(strict=True)
    source_path = (root / FINGER_PATHS[suffix]).resolve(strict=True)
    source_hash = _sha256(source_path)
    if source_hash != FINGER_HASHES[suffix]:
        raise ValueError(f"supplier-CAD finger hash mismatch: {side}")
    geometry_probe = json.loads(
        (
            root
            / "reports/aloha1_mapping/aloha1_cad_source_geometry_probe.json"
        ).read_text(encoding="utf-8")
    )
    face = geometry_probe["finger_contact_surfaces"][side]
    vertices, faces = _load_obj(source_path)
    samples = _surface_samples(vertices, faces)
    normal = np.asarray(face["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    center = np.asarray(face["center_mm"], dtype=np.float64) * 0.001
    bbox = face["bbox_mm"]
    budget = 0.0002
    signed_plane_distance = (samples - center) @ normal
    selected = (
        (np.abs(signed_plane_distance) <= budget)
        & (samples[:, 0] >= float(bbox["XMin"]) * 0.001 - budget)
        & (samples[:, 0] <= float(bbox["XMax"]) * 0.001 + budget)
        & (samples[:, 1] >= float(bbox["YMin"]) * 0.001 - budget)
        & (samples[:, 1] <= float(bbox["YMax"]) * 0.001 + budget)
        & (samples[:, 2] >= float(bbox["ZMin"]) * 0.001 - budget)
        & (samples[:, 2] <= float(bbox["ZMax"]) * 0.001 + budget)
    )
    contact_samples = samples[selected]
    if not len(contact_samples):
        raise ValueError(f"no supplier-CAD contact samples selected: {side}")
    return {
        "side": side,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "source_face_count": len(faces),
        "cad_face_index": int(face["face_index"]),
        "cad_face_center_m": center.tolist(),
        "normal": normal.tolist(),
        "samples": contact_samples,
        "sample_count": len(contact_samples),
        "tessellation_error_budget_m": budget,
        "mirror_used": False,
        "selection_method": (
            "authoritative mesh vertices, edge midpoints and triangle centroids "
            "within the FreeCAD face AABB and 0.20 mm of its plane"
        ),
    }


def _piece_ray_interval(
    point: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
) -> tuple[float, float] | None:
    hull = ConvexHull(vertices)
    lower = -np.inf
    upper = np.inf
    numeric_scale = max(1.0, float(np.max(np.abs(vertices))))
    parallel_tolerance = 128.0 * np.finfo(float).eps * numeric_scale
    for equation in hull.equations:
        normal = equation[:3]
        offset = float(equation[3])
        value = float(np.dot(normal, point) + offset)
        slope = float(np.dot(normal, direction))
        if abs(slope) <= parallel_tolerance:
            if value > parallel_tolerance:
                return None
            continue
        boundary = -value / slope
        if slope > 0.0:
            upper = min(upper, boundary)
        else:
            lower = max(lower, boundary)
        if lower > upper + parallel_tolerance:
            return None
    return float(lower), float(upper)


def positive_union_exit_distance(
    point: np.ndarray,
    direction: np.ndarray,
    pieces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the positive ray exit from the convex-piece union containing point."""

    point = np.asarray(point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        raise ValueError("contact direction must be non-zero")
    direction = direction / direction_norm
    intervals = []
    numeric_scale = max(1.0, float(np.max(np.abs(point))))
    for index, piece in enumerate(pieces):
        vertices = np.asarray(piece["vertices"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError(f"invalid convex piece vertices at index {index}")
        interval = _piece_ray_interval(point, direction, vertices)
        if interval is None:
            continue
        numeric_scale = max(numeric_scale, float(np.max(np.abs(vertices))))
        intervals.append((interval[0], interval[1], index))
    tolerance = 256.0 * np.finfo(float).eps * numeric_scale
    containing = [
        interval
        for interval in intervals
        if interval[0] <= tolerance and interval[1] >= -tolerance
    ]
    if not containing:
        positive_entries = [
            max(0.0, lower)
            for lower, upper, _ in intervals
            if upper >= -tolerance and lower >= -tolerance
        ]
        return {
            "source_point_covered": False,
            "positive_exit_distance_m": None,
            "nearest_positive_entry_m": (
                float(min(positive_entries)) if positive_entries else None
            ),
            "contributing_piece_count": 0,
            "numeric_tolerance_m": tolerance,
        }

    current_end = max(interval[1] for interval in containing)
    contributing = {interval[2] for interval in containing}
    extended = True
    while extended:
        extended = False
        for lower, upper, index in intervals:
            if lower <= current_end + tolerance and upper > current_end + tolerance:
                current_end = upper
                contributing.add(index)
                extended = True
    return {
        "source_point_covered": True,
        "positive_exit_distance_m": max(0.0, float(current_end)),
        "nearest_positive_entry_m": 0.0,
        "contributing_piece_count": len(contributing),
        "numeric_tolerance_m": tolerance,
    }


def _closest_point_on_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    edge_ab = second - first
    edge_ac = third - first
    point_a = point - first
    d1 = float(np.dot(edge_ab, point_a))
    d2 = float(np.dot(edge_ac, point_a))
    if d1 <= 0.0 and d2 <= 0.0:
        return first
    point_b = point - second
    d3 = float(np.dot(edge_ab, point_b))
    d4 = float(np.dot(edge_ac, point_b))
    if d3 >= 0.0 and d4 <= d3:
        return second
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        weight = d1 / (d1 - d3)
        return first + weight * edge_ab
    point_c = point - third
    d5 = float(np.dot(edge_ab, point_c))
    d6 = float(np.dot(edge_ac, point_c))
    if d6 >= 0.0 and d5 <= d6:
        return third
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        weight = d2 / (d2 - d6)
        return first + weight * edge_ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return second + weight * (third - second)
    denominator = 1.0 / (va + vb + vc)
    weight_b = vb * denominator
    weight_c = vc * denominator
    return first + edge_ab * weight_b + edge_ac * weight_c


def _piece_surface_triangles(pieces: list[dict[str, Any]]) -> list[np.ndarray]:
    triangles = []
    for piece in pieces:
        vertices = np.asarray(piece["vertices"], dtype=np.float64)
        hull = ConvexHull(vertices)
        triangles.append(vertices[np.asarray(hull.simplices, dtype=np.int64)])
    return triangles


def _nearest_union_surface(
    point: np.ndarray,
    direction: np.ndarray,
    surface_triangles: list[np.ndarray],
) -> dict[str, Any]:
    best_distance = np.inf
    best_delta = None
    best_target = None
    best_piece = None
    for piece_index, triangles in enumerate(surface_triangles):
        for triangle in triangles:
            closest = _closest_point_on_triangle(
                point, triangle[0], triangle[1], triangle[2]
            )
            delta = closest - point
            distance = float(np.linalg.norm(delta))
            if distance < best_distance:
                best_distance = distance
                best_delta = delta
                best_target = closest
                best_piece = piece_index
    if best_delta is None or best_target is None:
        raise ValueError("no cooked convex surfaces are available")
    return {
        "distance_m": best_distance,
        "normal_projection_m": float(np.dot(best_delta, direction)),
        "target_point_m": best_target.tolist(),
        "piece_index": best_piece,
    }


def summarize_contact_envelope(
    contact_samples: np.ndarray,
    direction: np.ndarray,
    pieces: list[dict[str, Any]],
    *,
    tessellation_budget_m: float,
) -> dict[str, Any]:
    results = [
        positive_union_exit_distance(sample, direction, pieces)
        for sample in np.asarray(contact_samples, dtype=np.float64)
    ]
    covered = [item for item in results if item["source_point_covered"]]
    direction_array = np.asarray(direction, dtype=np.float64)
    direction_array /= np.linalg.norm(direction_array)
    surface_triangles = _piece_surface_triangles(pieces)
    uncovered_measurements = [
        _nearest_union_surface(sample, direction_array, surface_triangles)
        for sample, item in zip(contact_samples, results, strict=True)
        if not item["source_point_covered"]
    ]
    distances = np.asarray(
        [item["positive_exit_distance_m"] for item in covered],
        dtype=np.float64,
    )
    coverage_ratio = len(covered) / len(results) if results else 0.0
    uncovered_distances = np.asarray(
        [item["distance_m"] for item in uncovered_measurements], dtype=np.float64
    )
    deviations = np.concatenate((distances, uncovered_distances))
    maximum_deviation = float(np.max(deviations))
    uncovered_iterator = iter(uncovered_measurements)
    deviation_records = []
    for sample_index, (sample, item) in enumerate(
        zip(contact_samples, results, strict=True)
    ):
        source_point = np.asarray(sample, dtype=np.float64)
        if item["source_point_covered"]:
            distance = float(item["positive_exit_distance_m"])
            deviation_records.append(
                {
                    "sample_index": sample_index,
                    "kind": "COVERED_NORMAL_EXIT",
                    "distance_m": distance,
                    "source_point_m": source_point.tolist(),
                    "target_point_m": (
                        source_point + direction_array * distance
                    ).tolist(),
                }
            )
        else:
            measurement = next(uncovered_iterator)
            deviation_records.append(
                {
                    "sample_index": sample_index,
                    "kind": "UNCOVERED_NEAREST_SURFACE",
                    "distance_m": measurement["distance_m"],
                    "source_point_m": source_point.tolist(),
                    "target_point_m": measurement["target_point_m"],
                }
            )
    worst = max(deviation_records, key=lambda item: item["distance_m"])
    crossing_records = [
        record
        for record in deviation_records
        if record["kind"] == "COVERED_NORMAL_EXIT"
    ]
    worst_crossing = (
        max(crossing_records, key=lambda item: item["distance_m"])
        if crossing_records
        else None
    )
    if maximum_deviation > tessellation_budget_m:
        status = "FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET"
    else:
        status = "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
    return {
        "status": status,
        "contact_sample_count": len(results),
        "source_point_covered_count": len(covered),
        "source_point_coverage_ratio": coverage_ratio,
        "uncovered_count": len(results) - len(covered),
        "uncovered_nearest_surface_min_m": (
            float(np.min(uncovered_distances)) if len(uncovered_distances) else None
        ),
        "uncovered_nearest_surface_mean_m": (
            float(np.mean(uncovered_distances)) if len(uncovered_distances) else None
        ),
        "uncovered_nearest_surface_max_m": (
            float(np.max(uncovered_distances)) if len(uncovered_distances) else None
        ),
        "uncovered_nearest_surface_normal_projection_min_m": (
            min(item["normal_projection_m"] for item in uncovered_measurements)
            if uncovered_measurements
            else None
        ),
        "uncovered_nearest_surface_normal_projection_max_m": (
            max(item["normal_projection_m"] for item in uncovered_measurements)
            if uncovered_measurements
            else None
        ),
        "tessellation_error_budget_m": tessellation_budget_m,
        "maximum_contact_surface_deviation_m": maximum_deviation,
        "maximum_deviation_kind": worst["kind"],
        "maximum_deviation_sample_index": worst["sample_index"],
        "maximum_deviation_source_point_m": worst["source_point_m"],
        "maximum_deviation_target_point_m": worst["target_point_m"],
        "positive_exit_distance_min_m": (
            float(np.min(distances)) if len(distances) else None
        ),
        "positive_exit_distance_mean_m": (
            float(np.mean(distances)) if len(distances) else None
        ),
        "positive_exit_distance_p95_m": (
            float(np.quantile(distances, 0.95)) if len(distances) else None
        ),
        "positive_exit_distance_max_m": (
            float(np.max(distances)) if len(distances) else None
        ),
        "maximum_inward_crossing_sample_index": (
            worst_crossing["sample_index"] if worst_crossing else None
        ),
        "maximum_inward_crossing_source_point_m": (
            worst_crossing["source_point_m"] if worst_crossing else None
        ),
        "maximum_inward_crossing_target_point_m": (
            worst_crossing["target_point_m"] if worst_crossing else None
        ),
        "maximum_contributing_piece_count": max(
            (item["contributing_piece_count"] for item in results), default=0
        ),
        "method": (
            "ANALYTIC_NORMAL_RAY_UNION_FOR_COVERED_POINTS_AND_EXACT_TRIANGLE_"
            "NEAREST_SURFACE_FOR_UNCOVERED_POINTS"
        ),
    }


def classify_profile_comparison(
    profiles_by_side: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if set(profiles_by_side) != {"left", "right"}:
        raise ValueError("both supplier-CAD handed fingers are required")
    deltas = {}
    decomposition_passes = True
    hull_passes = True
    improvements = []
    for side, profiles in profiles_by_side.items():
        if set(profiles) != {"convexHull", "convexDecomposition"}:
            raise ValueError(f"both collision profiles are required for {side}")
        hull = profiles["convexHull"]
        decomposition = profiles["convexDecomposition"]
        hull_error = float(hull["maximum_contact_surface_deviation_m"])
        decomposition_error = float(
            decomposition["maximum_contact_surface_deviation_m"]
        )
        delta = hull_error - decomposition_error
        deltas[side] = {
            "hull_max_deviation_m": hull_error,
            "decomposition_max_deviation_m": decomposition_error,
            "hull_minus_decomposition_m": delta,
        }
        improvements.append(delta > 0.0)
        decomposition_passes &= (
            decomposition["status"]
            == "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
        )
        hull_passes &= hull["status"] == "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
    if all(improvements) and decomposition_passes and not hull_passes:
        classification = (
            "DECOMPOSITION_GEOMETRY_IMPROVES_WITHIN_BUDGET_NOT_PROMOTED"
        )
    elif all(improvements) and decomposition_passes and hull_passes:
        classification = (
            "BOTH_WITHIN_BUDGET_DECOMPOSITION_LOWER_ERROR_NOT_PROMOTED"
        )
    elif all(improvements):
        classification = "DECOMPOSITION_REDUCES_ERROR_OUTSIDE_BUDGET"
    elif any(not improved for improved in improvements):
        classification = "DECOMPOSITION_MIXED_OR_WORSE"
    else:
        classification = "NO_NUMERICAL_IMPROVEMENT"
    return {
        "classification": classification,
        "by_side": deltas,
        "final_or_default_collider_modified": False,
        "runtime_hold_claim": "NOT_MADE",
    }

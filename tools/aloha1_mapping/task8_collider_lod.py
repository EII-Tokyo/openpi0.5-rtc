from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.stats import qmc

from tools.aloha1_mapping.task8_optimization import summarize_numeric_samples

TASK_CONTACT_CRITICAL = {
    "gripper_link",
    "gripper_bar_link",
    "gripper_prop_link",
    "left_finger_link",
    "right_finger_link",
}
ENVIRONMENT_CLEARANCE_CRITICAL = {
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
}
VIRTUAL_FRAME_LINKS = {
    "ee_arm_link",
    "ee_gripper_link",
    "fingers_link",
}


def classify_link_role(link_suffix: str, *, has_collider: bool) -> str:
    if link_suffix in TASK_CONTACT_CRITICAL:
        return "task_contact_critical"
    if link_suffix in ENVIRONMENT_CLEARANCE_CRITICAL:
        return "environment_clearance_critical"
    if not has_collider and link_suffix in VIRTUAL_FRAME_LINKS:
        return "non_contact_visual_only"
    return "inconclusive"


def select_throughput_links(records: Sequence[Mapping[str, Any]]) -> list[str]:
    selected = []
    for record in records:
        if record.get("role") != "environment_clearance_critical":
            continue
        if int(record.get("source_convex_piece_count", 0)) <= 1:
            continue
        if record.get("source_brep_valid") is not True:
            continue
        if record.get("baseline_static_audit") != "PASS":
            continue
        if record.get("baseline_swept_audit") != "PASS":
            continue
        selected.append(str(record["link_suffix"]))
    return sorted(selected)


def _canonical_geometry_signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    triangles = []
    for face in faces:
        triangle = sorted(
            tuple(round(float(component), 12) for component in vertices[index])
            for index in face
        )
        triangles.append(triangle)
    payload = json.dumps(sorted(triangles), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_single_hull_geometry(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("convex hull requires at least four 3-D points")
    if not np.isfinite(points).all():
        raise ValueError("convex hull points must be finite")

    ordered = np.unique(points, axis=0)
    hull = ConvexHull(ordered)
    hull_indices = np.asarray(sorted(int(index) for index in hull.vertices), dtype=np.int64)
    vertices = ordered[hull_indices]
    remap = {int(old): new for new, old in enumerate(hull_indices)}
    oriented_faces = []
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        face = [remap[int(index)] for index in simplex]
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), equation[:3]) < 0.0:
            face[1], face[2] = face[2], face[1]
        oriented_faces.append(tuple(face))
    faces = np.asarray(sorted(oriented_faces), dtype=np.int64)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return {
        "vertices_m": vertices.tolist(),
        "faces": faces.tolist(),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "volume_m3": float(hull.volume),
        "area_m2": float(hull.area),
        "aabb_m": {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "extent": (maximum - minimum).tolist(),
        },
        "canonical_signature": _canonical_geometry_signature(vertices, faces),
    }


def split_face_components(faces: np.ndarray) -> list[list[int]]:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangle faces must have shape (N, 3)")
    parent = list(range(len(faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owners: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            previous = owners.setdefault(int(vertex_index), face_index)
            union(face_index, previous)
    grouped: dict[int, list[int]] = defaultdict(list)
    for face_index in range(len(faces)):
        grouped[find(face_index)].append(face_index)
    return sorted((sorted(indices) for indices in grouped.values()), key=lambda item: item[0])


def ordered_mesh_components(
    vertices: np.ndarray, faces: np.ndarray
) -> list[dict[str, Any]]:
    """Reproduce the component order used by the CAD-collider authoring tool."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    components = []
    for face_indices in split_face_components(faces):
        component_faces_source = faces[face_indices]
        used = np.unique(component_faces_source.reshape(-1))
        remap = {int(source): target for target, source in enumerate(used)}
        component_vertices = vertices[used]
        component_faces = np.asarray(
            [
                [remap[int(source)] for source in face]
                for face in component_faces_source
            ],
            dtype=np.int64,
        )
        minimum_point = min(tuple(float(value) for value in point) for point in component_vertices)
        components.append(
            {
                "vertices": component_vertices,
                "faces": component_faces,
                "source_vertex_indices": used,
                "source_face_indices": np.asarray(face_indices, dtype=np.int64),
                "minimum_point": list(minimum_point),
                "vertex_count": len(component_vertices),
                "face_count": len(component_faces),
                "geometry_signature": _canonical_geometry_signature(
                    component_vertices, component_faces
                ),
            }
        )
    components.sort(
        key=lambda item: (
            tuple(item["minimum_point"]),
            int(item["vertex_count"]),
            int(item["face_count"]) * 3,
        )
    )
    for piece_index, component in enumerate(components):
        component["piece_index"] = piece_index
    return components


def build_containment_pruning_certificate(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, Any]:
    """Prove whether one existing convex piece contains every other piece.

    The tolerance only covers float32 USD point authoring and half-space
    evaluation noise; it is not a geometry fitting allowance.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    components = ordered_mesh_components(vertices, faces)
    coordinate_scale = max(
        float(np.max(np.abs(vertices))),
        float(np.max(vertices.max(axis=0) - vertices.min(axis=0))),
        1.0e-3,
    )
    tolerance = float(64.0 * np.finfo(np.float32).eps * coordinate_scale)
    candidate_records = []
    for component in components:
        hull = ConvexHull(component["vertices"])
        signed_distances = (
            vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
        )
        maximum_outside = float(np.max(signed_distances))
        candidate_records.append(
            {
                "piece_index": int(component["piece_index"]),
                "maximum_outside_distance_m": maximum_outside,
                "contains_all_source_vertices": maximum_outside <= tolerance,
                "hull_volume_m3": float(hull.volume),
                "hull_area_m2": float(hull.area),
                "vertex_count": int(component["vertex_count"]),
                "face_count": int(component["face_count"]),
                "minimum_point": component["minimum_point"],
                "geometry_signature": component["geometry_signature"],
            }
        )
    containing = [
        record for record in candidate_records if record["contains_all_source_vertices"]
    ]
    if len(containing) != 1:
        return {
            "status": "NO_UNIQUE_EXISTING_CONTAINING_PIECE",
            "tolerance_m": tolerance,
            "tolerance_derivation": (
                "64 * float32_epsilon * max(abs_coordinate, AABB_extent, 1e-3 m)"
            ),
            "candidate_records": candidate_records,
        }

    retained_index = int(containing[0]["piece_index"])
    retained = components[retained_index]
    retained_hull = build_single_hull_geometry(retained["vertices"])
    full_hull = build_single_hull_geometry(vertices)
    maximum_outside = float(containing[0]["maximum_outside_distance_m"])
    return {
        "status": "VERIFIED_EXISTING_PIECE_CONTAINS_ALL_OTHERS",
        "retained_piece_index": retained_index,
        "removed_piece_indices": [
            index for index in range(len(components)) if index != retained_index
        ],
        "source_piece_count": len(components),
        "candidate_piece_count": 1,
        "maximum_outside_distance_m": maximum_outside,
        "tolerance_m": tolerance,
        "tolerance_derivation": (
            "64 * float32_epsilon * max(abs_coordinate, AABB_extent, 1e-3 m)"
        ),
        "full_hull_matches_retained_hull": (
            retained_hull["canonical_signature"] == full_hull["canonical_signature"]
        ),
        "retained_hull_signature": retained_hull["canonical_signature"],
        "full_hull_signature": full_hull["canonical_signature"],
        "retained_hull_volume_m3": retained_hull["volume_m3"],
        "full_hull_volume_m3": full_hull["volume_m3"],
        "component_records": candidate_records,
        "component_ordering": (
            "MATCHES tools/build_aloha1_cad_derived_collider_diagnostic_stage.py "
            "_connected_mesh_components: minimum point, vertex count, index count"
        ),
    }


def compare_profile_inventories(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    removed_collider_paths: Sequence[str],
) -> dict[str, Any]:
    """Require exact protected-state equality except declared collider removals."""

    protected_groups = ("articulations", "joints", "rigid_bodies", "visuals")
    protected_group_equality = {
        group: baseline.get(group) == candidate.get(group) for group in protected_groups
    }
    baseline_colliders = {
        str(record["path"]): record for record in baseline.get("colliders", [])
    }
    candidate_colliders = {
        str(record["path"]): record for record in candidate.get("colliders", [])
    }
    expected_removed = {str(path) for path in removed_collider_paths}
    actual_removed = set(baseline_colliders) - set(candidate_colliders)
    added = set(candidate_colliders) - set(baseline_colliders)
    retained = set(baseline_colliders) & set(candidate_colliders)
    drift = sorted(
        path
        for path in retained
        if baseline_colliders[path] != candidate_colliders[path]
    )
    unexpected_removed = sorted(actual_removed - expected_removed)
    missing_expected_removal = sorted(expected_removed - actual_removed)
    status = (
        "PASS"
        if all(protected_group_equality.values())
        and not unexpected_removed
        and not missing_expected_removal
        and not added
        and not drift
        else "FAIL"
    )
    return {
        "status": status,
        "protected_group_equality": protected_group_equality,
        "expected_removed_collider_paths": sorted(expected_removed),
        "actual_removed_collider_paths": sorted(actual_removed),
        "unexpected_removed_collider_paths": unexpected_removed,
        "missing_expected_removal_paths": missing_expected_removal,
        "unexpected_added_collider_paths": sorted(added),
        "retained_collider_drift_paths": drift,
        "baseline_collider_count": len(baseline_colliders),
        "candidate_collider_count": len(candidate_colliders),
    }


def _surface_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.vstack(
        (
            vertices,
            triangles.mean(axis=1),
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        )
    )


def compare_compound_to_single_hull(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    candidate = build_single_hull_geometry(vertices)
    candidate_vertices = np.asarray(candidate["vertices_m"], dtype=np.float64)
    candidate_faces = np.asarray(candidate["faces"], dtype=np.int64)
    components = split_face_components(faces)
    source_hulls = []
    source_volumes = []
    for face_indices in components:
        component_points = np.unique(vertices[faces[face_indices].reshape(-1)], axis=0)
        component_hull = ConvexHull(component_points)
        source_hulls.append(component_hull)
        source_volumes.append(float(component_hull.volume))

    hull = ConvexHull(candidate_vertices)
    signed_halfspaces = vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
    outside = np.max(signed_halfspaces, axis=1) > 1.0e-12
    source_samples = _surface_samples(vertices, faces)
    candidate_samples = _surface_samples(candidate_vertices, candidate_faces)
    outward = cKDTree(source_samples).query(candidate_samples, workers=1)[0]
    reverse = cKDTree(candidate_samples).query(source_samples, workers=1)[0]
    source_piece_volume_sum = float(sum(source_volumes))
    candidate_volume = float(candidate["volume_m3"])
    sample_power = 18
    unit_samples = qmc.Sobol(d=3, scramble=False).random_base2(sample_power)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    samples = qmc.scale(unit_samples, bounds_min, bounds_max)
    inside_union = np.zeros(len(samples), dtype=bool)
    for source_hull in source_hulls:
        equations = source_hull.equations
        for start in range(0, len(samples), 8192):
            stop = min(start + 8192, len(samples))
            inside_union[start:stop] |= np.all(
                samples[start:stop] @ equations[:, :3].T + equations[:, 3]
                <= 1.0e-12,
                axis=1,
            )
    aabb_volume = float(np.prod(bounds_max - bounds_min))
    occupancy = float(np.mean(inside_union))
    union_volume = aabb_volume * occupancy
    union_standard_error = aabb_volume * float(
        np.sqrt(occupancy * (1.0 - occupancy) / len(samples))
    )
    return {
        "source_component_count": len(components),
        "source_piece_volume_sum_m3": source_piece_volume_sum,
        "source_union_volume_estimate_m3": union_volume,
        "source_union_volume_estimate_standard_error_m3": union_standard_error,
        "source_union_volume_method": (
            "DETERMINISTIC_UNSCRAMBLED_SOBOL_AABB_INTEGRATION_AGAINST_COMPONENT_"
            "CONVEX_HULL_HALFSPACES"
        ),
        "source_union_volume_sample_count": len(samples),
        "candidate_piece_count": 1,
        "candidate_volume_m3": candidate_volume,
        "candidate_minus_source_union_estimate_m3": candidate_volume - union_volume,
        "candidate_volume_ratio": candidate_volume / union_volume,
        "candidate_minus_source_piece_sum_m3": candidate_volume - source_piece_volume_sum,
        "source_vertex_outside_candidate_count": int(np.count_nonzero(outside)),
        "inward_vertex_deviation_max_m": 0.0 if not np.any(outside) else None,
        "outward_sample_deviation_max_m": float(outward.max()),
        "outward_sample_deviation_rms_m": float(np.sqrt(np.mean(outward**2))),
        "source_to_candidate_sample_distance_max_m": float(reverse.max()),
        "surface_distance_method": (
            "VERTEX_FACE_CENTROID_EDGE_MIDPOINT_CKDTREE; approximate sampled "
            "distance, not exact CAD closest-point"
        ),
        "candidate_geometry": candidate,
    }


def rank_pair_merge_candidates(
    vertices: np.ndarray, faces: np.ndarray
) -> list[dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    components = split_face_components(faces)
    records = []
    for first, second in itertools.combinations(range(len(components)), 2):
        face_indices = sorted(components[first] + components[second])
        source_faces = faces[face_indices]
        source_vertex_indices = np.unique(source_faces.reshape(-1))
        source_vertices = vertices[source_vertex_indices]
        remap = {int(old): new for new, old in enumerate(source_vertex_indices)}
        source_faces_local = np.asarray(
            [[remap[int(index)] for index in face] for face in source_faces],
            dtype=np.int64,
        )
        candidate = build_single_hull_geometry(source_vertices)
        candidate_vertices = np.asarray(candidate["vertices_m"], dtype=np.float64)
        candidate_faces = np.asarray(candidate["faces"], dtype=np.int64)
        source_samples = _surface_samples(source_vertices, source_faces_local)
        candidate_samples = _surface_samples(candidate_vertices, candidate_faces)
        outward = cKDTree(source_samples).query(candidate_samples, workers=1)[0]
        hull = ConvexHull(candidate_vertices)
        signed_halfspaces = (
            source_vertices @ hull.equations[:, :3].T + hull.equations[:, 3]
        )
        outside = np.max(signed_halfspaces, axis=1) > 1.0e-12
        source_piece_volume_sum = 0.0
        for component_index in (first, second):
            component_points = np.unique(
                vertices[faces[components[component_index]].reshape(-1)], axis=0
            )
            source_piece_volume_sum += float(ConvexHull(component_points).volume)
        records.append(
            {
                "merged_component_indices": [first, second],
                "piece_reduction": 1,
                "source_piece_volume_sum_m3": source_piece_volume_sum,
                "candidate_volume_m3": float(candidate["volume_m3"]),
                "candidate_minus_source_piece_sum_m3": (
                    float(candidate["volume_m3"]) - source_piece_volume_sum
                ),
                "outward_sample_deviation_max_m": float(outward.max()),
                "outward_sample_deviation_rms_m": float(
                    np.sqrt(np.mean(outward**2))
                ),
                "source_vertex_outside_candidate_count": int(
                    np.count_nonzero(outside)
                ),
                "surface_distance_method": (
                    "VERTEX_FACE_CENTROID_EDGE_MIDPOINT_CKDTREE; approximate "
                    "sampled distance, not exact CAD closest-point"
                ),
                "candidate_geometry": candidate,
            }
        )
    return sorted(
        records,
        key=lambda record: (
            float(record["outward_sample_deviation_max_m"]),
            float(record["outward_sample_deviation_rms_m"]),
            record["merged_component_indices"],
        ),
    )


def summarize_profile_runs(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["profile"]), int(run["environment_count"]))].append(run)

    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (profile, environment_count), records in sorted(grouped.items()):
        if len(records) < 2:
            raise ValueError(
                f"profile={profile} environment_count={environment_count} "
                "requires at least two fresh-process runs"
            )
        numeric_keys = sorted(
            key
            for key in records[0]
            if key not in {"profile", "environment_count"}
            and all(isinstance(record.get(key), int | float) for record in records)
        )
        output[profile][str(environment_count)] = {
            key: summarize_numeric_samples([float(record[key]) for record in records])
            for key in numeric_keys
        }
    return dict(output)


def classify_benchmark_improvement(
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Require candidate physics ranges to beat baseline at every scale."""

    fidelity = summary.get("fidelity_profile", {})
    throughput = summary.get("throughput_profile", {})
    scales = sorted(set(fidelity) & set(throughput), key=int)
    if not scales:
        raise ValueError("benchmark profiles have no common environment scales")
    records = []
    for scale in scales:
        baseline = fidelity[scale]["physics_step_ms"]
        candidate = throughput[scale]["physics_step_ms"]
        baseline_mean = float(baseline["mean"])
        candidate_mean = float(candidate["mean"])
        records.append(
            {
                "environment_count": int(scale),
                "fidelity_mean_ms": baseline_mean,
                "throughput_mean_ms": candidate_mean,
                "mean_improvement_percent": (
                    100.0 * (baseline_mean - candidate_mean) / baseline_mean
                ),
                "fidelity_range_ms": [
                    float(baseline["min"]),
                    float(baseline["max"]),
                ],
                "throughput_range_ms": [
                    float(candidate["min"]),
                    float(candidate["max"]),
                ],
                "non_overlapping_improvement": (
                    float(candidate["max"]) < float(baseline["min"])
                ),
            }
        )
    all_improved = all(record["non_overlapping_improvement"] for record in records)
    return {
        "classification": (
            "PASS_CANDIDATE_NOT_PROMOTED"
            if all_improved
            else "NO_MEASURABLE_IMPROVEMENT"
        ),
        "criterion": (
            "THROUGHPUT_MAX_PHYSICS_STEP_MS_LT_FIDELITY_MIN_AT_EVERY_"
            "TESTED_ENVIRONMENT_SCALE"
        ),
        "all_scales_non_overlapping_improvement": all_improved,
        "scale_records": records,
    }


def summarize_hold_contact_telemetry(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize signed finger/Bottle500 hold contacts without zero clipping."""

    hold_rows = [record for record in rows if record.get("phase") == "HOLD"]
    if not hold_rows:
        raise ValueError("telemetry has no HOLD rows")
    output: dict[str, Any] = {"hold_frame_count": len(hold_rows)}
    for side in ("left", "right"):
        minimum_separations = []
        impulse_sums = []
        pair_paths: set[tuple[str, str]] = set()
        for record in hold_rows:
            contacts = [
                contact
                for contact in record.get("contacts", [])
                if f"{side}_finger_link" in (
                    f"{contact.get('actor0_path', '')} "
                    f"{contact.get('actor1_path', '')}"
                )
                and "Bottle500" in (
                    f"{contact.get('actor0_path', '')} "
                    f"{contact.get('actor1_path', '')}"
                )
            ]
            if not contacts:
                raise ValueError(f"HOLD frame is missing {side} finger contact")
            minimum_separations.append(
                min(float(contact["separation_m"]) for contact in contacts)
            )
            impulse_sums.append(
                sum(max(0.0, float(contact["impulse_ns"])) for contact in contacts)
            )
            pair_paths.update(
                (
                    str(contact.get("collider0_path", "")),
                    str(contact.get("collider1_path", "")),
                )
                for contact in contacts
            )
        output[f"{side}_finger"] = {
            "minimum_separation_m": summarize_numeric_samples(
                minimum_separations
            ),
            "positive_impulse_sum_ns": summarize_numeric_samples(impulse_sums),
            "geometric_contact_frame_count": sum(
                value <= 0.0 for value in minimum_separations
            ),
            "solver_active_frame_count": sum(value > 0.0 for value in impulse_sums),
            "collider_pair_paths": [list(pair) for pair in sorted(pair_paths)],
        }
    output["hold_bottle_origin_z_delta_m"] = float(
        hold_rows[-1]["bottle"]["position_world_m"][2]
        - hold_rows[0]["bottle"]["position_world_m"][2]
    )
    output["hold_drop_m"] = float(
        hold_rows[-1]["observation"]["hold_drop_m"]
    )
    return output

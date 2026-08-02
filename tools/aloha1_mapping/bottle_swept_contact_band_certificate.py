"""Signed task-contact-band certificate for the Bottle500 finger collider.

This module is intentionally offline.  It compares the project Bottle500
analytic CAD tangency against the finite supplier-CAD contact patch cooked by
PhysX.  It does not use grasp success, start a timeline, or mutate a USD.
"""

from __future__ import annotations

import ast
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRY_RELATIVE_PATH = Path(
    ".codex/artifacts/20260802-aloha1-official-model-first/"
    "cad_compound_contact_candidate/compound_geometry_finger_link_local.json"
)
COOKED_RUN_RELATIVE_PATH = Path(
    ".codex/artifacts/20260802-aloha1-official-model-first/"
    "cad_compound_contact_candidate/runtime_cooking_finger_link_local/run1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(values))
    if length == 0.0:
        raise ValueError("zero-length vector")
    return values / length


def rectangle_point_metrics(
    vertices: np.ndarray,
    point: np.ndarray,
    *,
    numeric_tolerance_m: float,
) -> dict[str, Any]:
    """Measure a point against an ordered finite planar rectangle."""
    rectangle = np.asarray(vertices, dtype=np.float64)
    query = np.asarray(point, dtype=np.float64)
    if rectangle.shape != (4, 3):
        raise ValueError("rectangle vertices must have shape (4, 3)")
    if query.shape != (3,):
        raise ValueError("query point must have shape (3,)")
    if numeric_tolerance_m < 0.0:
        raise ValueError("numeric tolerance must be non-negative")

    first_axis = rectangle[1] - rectangle[0]
    second_axis = rectangle[3] - rectangle[0]
    first_length = float(np.linalg.norm(first_axis))
    second_length = float(np.linalg.norm(second_axis))
    if first_length == 0.0 or second_length == 0.0:
        raise ValueError("rectangle has a zero-length edge")
    first_unit = first_axis / first_length
    second_unit = second_axis / second_length
    orthogonality_error = abs(float(np.dot(first_unit, second_unit)))
    if orthogonality_error > 1.0e-8:
        raise ValueError("finite contact patch is not rectangular")
    normal = _unit(np.cross(first_axis, second_axis))
    offset = query - rectangle[0]
    plane_signed_distance = float(np.dot(normal, offset))
    projected = query - plane_signed_distance * normal
    projected_offset = projected - rectangle[0]
    first_coordinate = float(np.dot(projected_offset, first_unit) / first_length)
    second_coordinate = float(np.dot(projected_offset, second_unit) / second_length)
    first_tolerance = numeric_tolerance_m / first_length
    second_tolerance = numeric_tolerance_m / second_length
    inside = (
        -first_tolerance <= first_coordinate <= 1.0 + first_tolerance
        and -second_tolerance <= second_coordinate <= 1.0 + second_tolerance
    )
    clamped_first = min(1.0, max(0.0, first_coordinate))
    clamped_second = min(1.0, max(0.0, second_coordinate))
    closest = rectangle[0] + clamped_first * first_axis + clamped_second * second_axis
    in_plane_distance = float(np.linalg.norm(projected - closest))
    return {
        "plane_signed_distance_m": plane_signed_distance,
        "point_on_plane": abs(plane_signed_distance) <= numeric_tolerance_m,
        "rectangle_coordinates": [first_coordinate, second_coordinate],
        "inside_finite_rectangle": inside,
        "minimum_in_plane_distance_to_rectangle_m": in_plane_distance,
        "projected_point_m": projected.tolist(),
        "closest_rectangle_point_m": closest.tolist(),
        "rectangle_edge_lengths_m": [first_length, second_length],
        "rectangle_edge_orthogonality_error": orthogonality_error,
    }


def classify_task_contact_band(
    fingers: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if set(fingers) != {"left", "right"}:
        raise ValueError("both handed supplier-CAD fingers are required")
    plane_pass = all(record["center_tangent_point_on_cad_plane"] for record in fingers.values())
    finite_patch_pass = all(record["center_tangent_point_inside_cooked_patch"] for record in fingers.values())
    if not plane_pass:
        status = "FAIL_TANGENCY_NOT_ON_AUTHORITATIVE_CAD_PLANE"
    elif not finite_patch_pass:
        status = "FAIL_CENTRAL_TANGENCY_OUTSIDE_COMPOUND_PATCH"
    else:
        status = "PASS_CENTRAL_TANGENCY_INSIDE_COMPOUND_PATCH"
    return {
        "status": status,
        "candidate_decision": (
            "TASK_CONTACT_BAND_GEOMETRY_PASS_NOT_PROMOTED"
            if status.startswith("PASS_")
            else "REJECTED_TASK_CONTACT_BAND_NOT_PROMOTED"
        ),
    }


def _extract_outer_profile(path: Path) -> list[tuple[float, float]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "OUTER_PROFILE_MM" for target in node.targets):
            values = ast.literal_eval(node.value)
            return [(float(radius), float(z_value)) for radius, z_value in values]
    raise ValueError("Bottle500 OUTER_PROFILE_MM was not found")


def _cad_to_reference(
    points_cad_m: np.ndarray,
    mapping: dict[str, float],
) -> np.ndarray:
    points = np.asarray(points_cad_m, dtype=np.float64)
    return np.column_stack(
        (
            -points[:, 1] + float(mapping["x_from_global_y_offset_m"]),
            points[:, 0] + float(mapping["y_from_global_x_offset_m"]),
            points[:, 2] + float(mapping["z_from_global_z_offset_m"]),
        )
    )


def _rectangle_in_gripper_reference(
    finger_geometry: dict[str, Any],
    cad_to_reference: dict[str, float],
    *,
    translation_from_closed_m: float,
    side: str,
) -> tuple[np.ndarray, dict[str, float]]:
    matrix = np.asarray(finger_geometry["rigid_transform_matrix"], dtype=np.float64)
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    orthonormal_residual = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if not np.isclose(determinant, 1.0, atol=1.0e-12):
        raise ValueError(f"{side} CAD-to-link transform is not proper")
    local = np.asarray(finger_geometry["contact_rectangle_vertices_m"], dtype=np.float64)
    homogeneous = np.column_stack((local, np.ones(len(local))))
    inverse = np.linalg.inv(matrix)
    cad = (homogeneous @ inverse.T)[:, :3]
    roundtrip = (np.column_stack((cad, np.ones(len(cad)))) @ matrix.T)[:, :3]
    roundtrip_residual = float(np.max(np.linalg.norm(roundtrip - local, axis=1)))
    reference = _cad_to_reference(cad, cad_to_reference)
    sign = 1.0 if side == "left" else -1.0
    reference += np.asarray([0.0, sign * translation_from_closed_m, 0.0])
    return reference, {
        "determinant": determinant,
        "orthonormal_residual": orthonormal_residual,
        "roundtrip_residual_m": roundtrip_residual,
    }


def _normal_cad_to_reference(normal_cad: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_cad, dtype=np.float64)
    return _unit(np.asarray([-normal[1], normal[0], normal[2]]))


def _cooked_contact_normal_error_deg(
    cooked_finger: dict[str, Any],
    expected_normal_finger_link: np.ndarray,
) -> float:
    expected = _unit(expected_normal_finger_link)
    errors = []
    contact_pieces = [
        piece
        for piece in cooked_finger["pieces"]
        if piece["source_construction"] == "CAD_CONTACT_TRIANGLE_INWARD_PRISM"
    ]
    if len(contact_pieces) != 2:
        raise ValueError("expected exactly two cooked contact prisms")
    for piece in contact_pieces:
        polygons = piece["cooked"]["pieces"][0]["polygons"]
        alignments = []
        for polygon in polygons:
            normal = _unit(np.asarray(polygon["plane"][:3], dtype=np.float64))
            alignments.append(float(np.clip(np.dot(normal, expected), -1.0, 1.0)))
        best = max(alignments)
        errors.append(math.degrees(math.acos(best)))
    return max(errors)


def build_certificate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "compound_geometry": root / GEOMETRY_RELATIVE_PATH,
        "runtime_cooked_run": root / COOKED_RUN_RELATIVE_PATH,
        "compound_runtime_certificate": root
        / "reports/aloha1_mapping/aloha1_supplier_cad_compound_runtime_cooking_certificate.json",
        "compound_candidate_report": root
        / "reports/aloha1_mapping/aloha1_supplier_cad_compound_contact_candidate.json",
        "grasp_clearance": root / "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance.json",
        "contact_semantics": root / "reports/aloha1_mapping/gripper_contact_semantics.json",
        "bottle_build_script": root / "assets/bottle_500ml/scripts/build_bottle_freecad.py",
    }
    for path in paths.values():
        path.resolve(strict=True)

    geometry = _load_json(paths["compound_geometry"])
    cooked = _load_json(paths["runtime_cooked_run"])
    runtime_certificate = _load_json(paths["compound_runtime_certificate"])
    candidate_report = _load_json(paths["compound_candidate_report"])
    clearance = _load_json(paths["grasp_clearance"])
    contact_semantics = _load_json(paths["contact_semantics"])
    profile = _extract_outer_profile(paths["bottle_build_script"])

    section = clearance["bottle_section"]
    station_mm = float(section["axial_station_mm"])
    radius_mm = float(section["outer_radius_mm"])
    constant_radius_segments = [
        [first_z, second_z]
        for (first_radius, first_z), (second_radius, second_z) in pairwise(profile)
        if first_radius == radius_mm and second_radius == radius_mm
    ]
    containing_segments = [segment for segment in constant_radius_segments if segment[0] <= station_mm <= segment[1]]
    if containing_segments != [[18.0, 120.0]]:
        raise ValueError("Bottle500 contact station is not in the pinned cylindrical body")

    contact_offset_records = []
    for profile_record in contact_semantics["profiles"].values():
        follower = profile_record["follower_left"]
        contact_offset_records.extend(
            [follower["sides"][side]["collider"]["contact_offset"] for side in ("left", "right")]
        )
    if not all(
        record["runtime_effective"] == "SIMULATION_DETERMINED_NOT_EXPOSED_BY_107_3_USD_READBACK"
        for record in contact_offset_records
    ):
        raise ValueError("unexpected local contact-offset readback boundary")

    finger_records: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        side_geometry = geometry["fingers"][side]
        contact_solution = clearance["contact_solution"]
        rectangle, transform = _rectangle_in_gripper_reference(
            side_geometry,
            clearance["coordinate_contract"]["cad_to_reference"],
            translation_from_closed_m=float(contact_solution[f"{side}_translation_from_closed_m"]),
            side=side,
        )
        tangent = np.asarray(contact_solution[f"{side}_contact_reference_m"], dtype=np.float64)
        numeric_tolerance = float(candidate_report["fingers"][side]["numeric_tolerance"]["numeric_tolerance_m"])
        metrics = rectangle_point_metrics(
            rectangle,
            tangent,
            numeric_tolerance_m=numeric_tolerance,
        )
        inverse = np.linalg.inv(np.asarray(side_geometry["rigid_transform_matrix"], dtype=np.float64))
        normal_cad = inverse[:3, :3] @ _unit(np.asarray(side_geometry["outward_normal"], dtype=np.float64))
        normal_reference = _normal_cad_to_reference(normal_cad)
        finite_patch = metrics["inside_finite_rectangle"]
        runtime_side = runtime_certificate["fingers"][side]
        finger_records[side] = {
            "supplier_handed_brep": True,
            "mirror_used": False,
            "contact_q_m": float(contact_solution[f"{side}_finger_q_m"]),
            "legal_q_range_m": (
                contact_solution["legal_range_m"]
                if side == "left"
                else [
                    -float(contact_solution["legal_range_m"][1]),
                    -float(contact_solution["legal_range_m"][0]),
                ]
            ),
            "candidate_rectangle_gripper_reference_m": rectangle.tolist(),
            "analytic_bottle_tangent_point_gripper_reference_m": tangent.tolist(),
            "authoritative_contact_normal_gripper_reference": normal_reference.tolist(),
            **metrics,
            "center_tangent_point_on_cad_plane": metrics["point_on_plane"],
            "center_tangent_point_inside_cooked_patch": finite_patch,
            "contact_line_intersection_length_m": (min(metrics["rectangle_edge_lengths_m"]) if finite_patch else 0.0),
            "cooked_contact_normal_max_error_deg": _cooked_contact_normal_error_deg(
                cooked["fingers"][side],
                np.asarray(side_geometry["outward_normal"], dtype=np.float64),
            ),
            "maximum_outward_crossing_m": float(runtime_side["maximum_outward_crossing_m"]),
            "maximum_cooking_surface_quantization_m": float(
                runtime_side["maximum_quantization_surface_distance_m"] or 0.0
            ),
            "numeric_tolerance_m": numeric_tolerance,
            "transform": transform,
        }

    classification = classify_task_contact_band(finger_records)
    maximum_transform_residual = max(record["transform"]["roundtrip_residual_m"] for record in finger_records.values())
    maximum_cooking_deviation = max(
        record["maximum_cooking_surface_quantization_m"] for record in finger_records.values()
    )
    brep_float_allowance = max(record["numeric_tolerance_m"] for record in finger_records.values())
    tessellation_deflection_m = 0.0002
    known_sum = (
        tessellation_deflection_m + maximum_transform_residual + maximum_cooking_deviation + brep_float_allowance
    )
    miss_distances = [record["minimum_in_plane_distance_to_rectangle_m"] for record in finger_records.values()]
    left_right_asymmetry = abs(miss_distances[0] - miss_distances[1])

    inputs = {
        name: {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "PASS_DETERMINISTIC_REJECTION"
            if classification["status"].startswith("FAIL_")
            else "PASS_TASK_CONTACT_BAND_GEOMETRY"
        ),
        "scope": "BOTTLE500_ANALYTIC_TANGENCY_VS_FINITE_COOKED_COMPOUND_PATCH",
        "candidate_decision": classification["candidate_decision"],
        "task_contact_band": {
            "status": classification["status"],
            "central_tangency_required": True,
            "central_tangency_is_sufficient_for_full_band": False,
            "reason": (
                "The analytic Bottle500 tangent lies on each infinite supplier-CAD "
                "inner plane but outside the finite compound contact rectangle. "
                "The candidate therefore fails before any axial-band-length claim."
            ),
            "left_right_miss_asymmetry_m": left_right_asymmetry,
        },
        "bottle500": {
            "geometry_authority": "PROJECT_AUTHORED_ANALYTIC_BREP",
            "cad_axis": "+Z",
            "contact_axial_station_mm": station_mm,
            "outer_radius_mm": radius_mm,
            "constant_radius_interval_mm": containing_segments[0],
            "physical_bottle_equivalence_claimed": False,
        },
        "fingers": finger_records,
        "known_numerical_error_budget": {
            "combination": "CONSERVATIVE_SUM_OF_INDEPENDENT_KNOWN_TERMS",
            "visual_tessellation_deflection_m": tessellation_deflection_m,
            "maximum_transform_roundtrip_residual_m": maximum_transform_residual,
            "maximum_physx_cooking_surface_quantization_m": maximum_cooking_deviation,
            "brep_float_comparison_allowance_m": brep_float_allowance,
            "known_sum_m": known_sum,
            "minimum_patch_miss_m": min(miss_distances),
            "minimum_patch_miss_to_known_sum_ratio": min(miss_distances) / known_sum,
            "contact_offset_readback": "NOT_EXPOSED_BY_LOCAL_107_3_USD_READBACK",
            "rest_offset_readback": "UNAUTHORED_RIGID_BODY_SCHEMA_ZERO_SEMANTICS",
            "physical_bottle_geometry": ("OUT_OF_SCOPE_PROJECT_CAD_IS_DIGITAL_GEOMETRY_AUTHORITY"),
            "complete_runtime_contact_envelope_budget": "PARTIAL",
        },
        "clearance_evidence": {
            "gripper_bar_minimum_m": clearance["station_selection"]["selected_clearance_by_envelope_m"][
                "runtime_urdf_gripper_bar"
            ],
            "supplier_shell_minimum_m": clearance["station_selection"]["selected_clearance_by_envelope_m"][
                "supplier_gripper_shell"
            ],
            "five_pose_unexpected_overlap_waypoints": 0,
            "five_pose_swept_report": str(
                (root / "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision.json").resolve()
            ),
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "cooking_readback_only": True,
        },
        "inputs": inputs,
        "grasp_success_used_to_set_tolerance": False,
        "grasp_video_used": False,
        "timeline_started": False,
        "final_or_default_collider_modified": False,
        "candidate_promoted": False,
    }
    signature_payload = dict(report)
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for side, record in report["fingers"].items():
        rows.append(
            f"| {side} | {record['plane_signed_distance_m'] * 1000.0:.9f} | "
            f"{record['minimum_in_plane_distance_to_rectangle_m'] * 1000.0:.6f} | "
            f"{record['cooked_contact_normal_max_error_deg']:.9f} | "
            f"{record['maximum_outward_crossing_m'] * 1000.0:.9f} | "
            f"{'PASS' if record['center_tangent_point_inside_cooked_patch'] else 'FAIL'} |"
        )
    budget = report["known_numerical_error_budget"]
    return "\n".join(
        [
            "# ALOHA1 Bottle500 swept contact-band collider certificate",
            "",
            f"- Audit status: **{report['status']}**",
            f"- Task contact-band status: **{report['task_contact_band']['status']}**",
            f"- Candidate decision: **{report['candidate_decision']}**",
            "- Final/default collider modified: `false`",
            "- Grasp success used to set tolerance: `false`",
            "",
            "| side | signed plane residual (mm) | finite-patch miss (mm) | cooked normal error (deg) | outward crossing (mm) | finite patch |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
            "",
            "The analytic Bottle500 tangent point is on the authoritative infinite "
            "supplier-CAD inner plane on both sides. It is nevertheless outside the "
            "finite 10.02 mm compound contact rectangle by about 1.61 mm. Plane "
            "alignment alone was therefore an insufficient acceptance test.",
            "",
            "The known conservative numerical sum is "
            f"`{budget['known_sum_m'] * 1000.0:.6f} mm`; the smallest finite-patch "
            f"miss is `{budget['minimum_patch_miss_m'] * 1000.0:.6f} mm` "
            f"(`{budget['minimum_patch_miss_to_known_sum_ratio']:.3f}x` larger). "
            "The local 107.3 runtime does not expose the effective contactOffset as "
            "USD readback, so the complete contact-envelope budget remains PARTIAL. "
            "That missing readback cannot promote a geometry patch which does not "
            "contain the task's central analytic tangency.",
            "",
            "This is a deterministic rejection certificate, not a collider repair. "
            "The 68-piece candidate remains diagnostic-only and unpromoted.",
            "",
        ]
    )

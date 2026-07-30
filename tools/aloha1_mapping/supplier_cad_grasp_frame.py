"""Derive the supplier-CAD two-finger grasp frame from audited pad surfaces."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_MESH_SHA256 = {
    "left": "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
    "right": "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
}
CAD_TO_FINGER_LINK = {
    "left": np.asarray(
        [
            [0.0, -1.0, 0.0, -0.49899999973392],
            [1.0, 0.0, 0.0, -0.021099890257662776],
            [0.0, 0.0, 1.0, -0.42680133373174],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            [0.0, -1.0, 0.0, -0.49899999973392],
            [1.0, 0.0, 0.0, 0.020900109742337226],
            [0.0, 0.0, 1.0, -0.42680133373174],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}
INWARD_AXIS = {
    "left": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "right": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
}
CLOSED_FINGER_Q_M = {"left": 0.021, "right": -0.021}
GRIPPER_REFERENCE_FROM_FINGERS_LINK_X_M = 0.0687
GRIPPER_REFERENCE_FROM_EE_GRIPPER_X_M = 0.1072
MINIMUM_INWARD_NORMAL_DOT = math.cos(math.radians(12.0))
COPLANAR_NORMAL_COSINE = 1.0 - 1e-8
COPLANAR_OFFSET_TOLERANCE_M = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_clearance_grasp_frame(
    *,
    clearance_report_path: Path,
    screenshot_review_path: Path,
    expected_clearance_sha256: str,
    expected_screenshot_sha256: str,
) -> dict[str, Any]:
    """Load the user-approved complete-gripper clearance frame fail-closed.

    The older whole-pad-face centroid remains available to historical audit
    code in this module, but it is not authorized as a runtime grasp origin.
    """

    clearance_path = clearance_report_path.resolve(strict=True)
    screenshot_path = screenshot_review_path.resolve(strict=True)
    clearance_sha256 = _sha256(clearance_path)
    screenshot_sha256 = _sha256(screenshot_path)
    if clearance_sha256 != expected_clearance_sha256:
        raise RuntimeError(
            "supplier-CAD clearance report SHA-256 mismatch: "
            f"{clearance_sha256}"
        )
    if screenshot_sha256 != expected_screenshot_sha256:
        raise RuntimeError(
            "supplier-CAD screenshot review SHA-256 mismatch: "
            f"{screenshot_sha256}"
        )

    clearance = json.loads(clearance_path.read_text(encoding="utf-8"))
    screenshot = json.loads(screenshot_path.read_text(encoding="utf-8"))
    if (
        clearance.get("status") != "PASS"
        or clearance.get("classification")
        != "COMPLETE_SUPPLIER_GRIPPER_PROJECT_BOTTLE_CLEARANCE"
        or clearance.get("task8") != "NOT_RUN"
    ):
        raise RuntimeError("complete-gripper clearance gate is not PASS")
    if (
        screenshot.get("status") != "PASS"
        or screenshot.get("final_visual_judgment") != "PASS"
        or screenshot.get("all_individually_visually_reviewed") is not True
        or screenshot.get("task8") != "NOT_RUN"
        or screenshot.get("user_review", {}).get("status") != "PASS"
    ):
        raise RuntimeError(
            "supplier-CAD screenshot/user-review gate is not PASS"
        )
    if any(
        capture.get("visual_review") != "PASS"
        for capture in screenshot.get("captures", [])
    ):
        raise RuntimeError("a supplier-CAD screenshot capture is not PASS")
    screenshot_source = Path(
        screenshot["source_geometry_report"]["absolute_path"]
    ).resolve(strict=True)
    if screenshot_source != clearance_path:
        raise RuntimeError(
            "screenshot review does not reference the frozen clearance report"
        )

    frame = clearance["grasp_frame"]
    contact = clearance["contact_solution"]
    rejected = clearance["rejected_run13"]
    transform = np.asarray(
        frame["reference_from_grasp"],
        dtype=np.float64,
    )
    validate_shape = transform.shape == (4, 4)
    if not validate_shape or not np.isfinite(transform).all():
        raise RuntimeError("clearance grasp frame is not a finite 4x4 matrix")
    determinant = float(np.linalg.det(transform[:3, :3]))
    if not np.isclose(determinant, 1.0, atol=1e-12):
        raise RuntimeError(
            f"clearance grasp-frame determinant is {determinant}"
        )
    if frame.get("status") != "PASS":
        raise RuntimeError("clearance grasp frame is not PASS")
    if frame.get("ee_endpoint_is_grasp_center") is not False:
        raise RuntimeError("official EE helper was aliased to grasp center")
    if (
        rejected.get("classification")
        != "REJECTED_WHOLE_PAD_FACE_CENTROID_NOT_EFFECTIVE_GRASP_CENTER"
    ):
        raise RuntimeError("whole-pad-face centroid rejection is missing")
    if contact.get("symmetric") is not True:
        raise RuntimeError("clearance contact solution is not symmetric")

    left_target = float(contact["left_finger_q_m"])
    right_target = float(contact["right_finger_q_m"])
    if not np.isclose(left_target, -right_target, atol=1e-12):
        raise RuntimeError("clearance finger targets are not antisymmetric")
    return {
        "status": "PASS",
        "classification": (
            "FROZEN_SUPPLIER_CAD_COMPLETE_GRIPPER_CLEARANCE_FRAME"
        ),
        "clearance_report": {
            "absolute_path": str(clearance_path),
            "sha256": clearance_sha256,
            "semantic_signature": clearance["semantic_signature"],
            "status": clearance["status"],
        },
        "screenshot_gate": {
            "absolute_path": str(screenshot_path),
            "sha256": screenshot_sha256,
            "status": screenshot["status"],
            "vision_model_reviewed": (
                screenshot["all_individually_visually_reviewed"]
            ),
            "user_confirmed": True,
            "user_confirmation": screenshot["user_review"]["confirmation"],
        },
        "reference_from_grasp": transform.tolist(),
        "origin_reference_m": [
            float(value) for value in frame["origin_reference_m"]
        ],
        "bottle_axis_center_from_grasp_m": [
            float(value)
            for value in frame["bottle_axis_center_from_grasp_m"]
        ],
        "approach_axis_reference": [
            float(value) for value in frame["approach_axis_reference"]
        ],
        "finger_line_axis_reference": [
            float(value) for value in frame["finger_line_axis_reference"]
        ],
        "bottle_axis_reference": [
            float(value) for value in frame["bottle_axis_reference"]
        ],
        "rotation_determinant": determinant,
        "finger_targets_m": {
            "left_finger": left_target,
            "right_finger": right_target,
        },
        "contact_points_reference_m": {
            "left": [
                float(value)
                for value in contact["left_contact_reference_m"]
            ],
            "right": [
                float(value)
                for value in contact["right_contact_reference_m"]
            ],
        },
        "legal_range_m": [
            float(value) for value in contact["legal_range_m"]
        ],
        "selected_minimum_margin_m": float(
            clearance["station_selection"]["selected_minimum_margin_m"]
        ),
        "official_ee_helper_semantics": "NOT_GRASP_CENTER",
        "whole_pad_face_centroid_use": "REJECTED",
        "source_geometry_modified": False,
        "task8": "NOT_RUN",
    }


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [
                int(token.split("/", 1)[0]) - 1
                for token in line.split()[1:]
            ]
            if len(face) != 3:
                raise ValueError(f"non-triangle OBJ face in {path}")
            faces.append(face)
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError(f"invalid OBJ vertices: {path}")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"invalid OBJ triangles: {path}")
    if (
        face_array.min(initial=0) < 0
        or face_array.max(initial=-1) >= len(vertex_array)
    ):
        raise ValueError(f"OBJ face index out of bounds: {path}")
    return vertex_array, face_array


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )
    return (homogeneous @ matrix.T)[:, :3]


def _json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def _connected_planar_patches(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    inward_axis: np.ndarray,
) -> list[dict[str, Any]]:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    cross_norm = np.linalg.norm(cross, axis=1)
    if np.any(cross_norm <= 1e-15):
        raise ValueError("supplier CAD OBJ contains degenerate triangles")
    areas = cross_norm / 2.0
    normals = cross / cross_norm[:, None]
    plane_offsets = -np.einsum(
        "ij,ij->i",
        normals,
        triangles[:, 0],
    )
    candidate_ids = np.flatnonzero(
        normals @ inward_axis >= MINIMUM_INWARD_NORMAL_DOT
    )
    if not len(candidate_ids):
        raise ValueError("no inward-facing supplier CAD triangles")

    parent = {int(index): int(index) for index in candidate_ids}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_vertex: dict[int, list[int]] = {}
    for index in candidate_ids:
        for vertex_index in faces[index]:
            by_vertex.setdefault(int(vertex_index), []).append(int(index))
    for incident in by_vertex.values():
        for position, left in enumerate(incident):
            for right in incident[position + 1 :]:
                if (
                    float(np.dot(normals[left], normals[right]))
                    >= COPLANAR_NORMAL_COSINE
                    and abs(
                        float(
                            plane_offsets[left] - plane_offsets[right]
                        )
                    )
                    <= COPLANAR_OFFSET_TOLERANCE_M
                ):
                    union(left, right)

    groups: dict[int, list[int]] = {}
    for index in candidate_ids:
        groups.setdefault(find(int(index)), []).append(int(index))

    patches = []
    for indices in groups.values():
        index_array = np.asarray(indices, dtype=np.int64)
        patch_triangles = triangles[index_array]
        patch_areas = areas[index_array]
        area = float(np.sum(patch_areas))
        centroids = np.mean(patch_triangles, axis=1)
        centroid = np.sum(
            centroids * patch_areas[:, None],
            axis=0,
        ) / area
        normal = np.sum(
            normals[index_array] * patch_areas[:, None],
            axis=0,
        )
        normal /= np.linalg.norm(normal)
        points = patch_triangles.reshape(-1, 3)
        patches.append(
            {
                "triangle_count": len(indices),
                "triangle_indices": sorted(indices),
                "area_m2": area,
                "centroid_finger_link_m": _json_vector(centroid),
                "normal_finger_link": _json_vector(normal),
                "inward_normal_dot": float(np.dot(normal, inward_axis)),
                "aabb_finger_link_m": {
                    "min": _json_vector(np.min(points, axis=0)),
                    "max": _json_vector(np.max(points, axis=0)),
                },
            }
        )
    return sorted(
        patches,
        key=lambda record: float(record["area_m2"]),
        reverse=True,
    )


def _finger_record(side: str, path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_MESH_SHA256[side]:
        raise ValueError(
            f"unexpected {side} supplier CAD mesh SHA-256: {actual_sha256}"
        )
    cad_vertices, faces = _load_obj(path)
    vertices = _transform_points(cad_vertices, CAD_TO_FINGER_LINK[side])
    patches = _connected_planar_patches(
        vertices=vertices,
        faces=faces,
        inward_axis=INWARD_AXIS[side],
    )
    selected = dict(patches[0])
    selected["area_ratio_to_next_candidate"] = float(
        selected["area_m2"] / patches[1]["area_m2"]
    )
    selected["selection_rule"] = (
        "LARGEST_CONNECTED_COPLANAR_INWARD_FACING_SURFACE"
    )
    selected["distal_x_max_residual_to_mesh_m"] = float(
        np.max(vertices[:, 0])
        - selected["aabb_finger_link_m"]["max"][0]
    )
    return {
        "source_obj": {
            "absolute_path": str(path),
            "sha256": actual_sha256,
            "vertex_count": len(vertices),
            "triangle_count": len(faces),
        },
        "cad_to_finger_link_matrix": [
            _json_vector(row) for row in CAD_TO_FINGER_LINK[side]
        ],
        "cad_to_finger_link_determinant": float(
            np.linalg.det(CAD_TO_FINGER_LINK[side][:3, :3])
        ),
        "mesh_aabb_finger_link_m": {
            "min": _json_vector(np.min(vertices, axis=0)),
            "max": _json_vector(np.max(vertices, axis=0)),
        },
        "candidate_patch_count": len(patches),
        "selected_inner_pad": selected,
        "candidate_patches": patches,
    }


def derive_supplier_cad_grasp_frame(
    *,
    left_obj_path: Path,
    right_obj_path: Path,
) -> dict[str, Any]:
    fingers = {
        "left": _finger_record("left", left_obj_path),
        "right": _finger_record("right", right_obj_path),
    }
    centers_finger = {
        side: np.asarray(
            fingers[side]["selected_inner_pad"][
                "centroid_finger_link_m"
            ],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    centers_reference = {
        side: centers_finger[side]
        + np.asarray(
            [
                GRIPPER_REFERENCE_FROM_FINGERS_LINK_X_M,
                CLOSED_FINGER_Q_M[side],
                0.0,
            ],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    midpoint_reference = (
        centers_reference["left"] + centers_reference["right"]
    ) / 2.0
    midpoint_ee = midpoint_reference - np.asarray(
        [GRIPPER_REFERENCE_FROM_EE_GRIPPER_X_M, 0.0, 0.0],
        dtype=np.float64,
    )
    center_line = centers_reference["right"] - centers_reference["left"]
    left_normal = np.asarray(
        fingers["left"]["selected_inner_pad"]["normal_finger_link"],
        dtype=np.float64,
    )
    right_normal = np.asarray(
        fingers["right"]["selected_inner_pad"]["normal_finger_link"],
        dtype=np.float64,
    )
    handed_normal_transform = np.diag([1.0, -1.0, -1.0])
    symmetry = {
        "mirror_operation_applied": False,
        "comparison_operator": (
            "CAD_HANDED_PAIR_180_DEGREE_X_SYMMETRY_FOR_VECTOR_COMPARISON_ONLY"
        ),
        "center_x_residual_m": float(
            abs(
                centers_reference["left"][0]
                - centers_reference["right"][0]
            )
        ),
        "center_y_sum_abs_m": float(
            abs(
                centers_reference["left"][1]
                + centers_reference["right"][1]
            )
        ),
        "center_z_sum_abs_m": float(
            abs(
                centers_reference["left"][2]
                + centers_reference["right"][2]
            )
        ),
        "normal_handed_pair_residual": float(
            np.linalg.norm(
                handed_normal_transform @ left_normal - right_normal
            )
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "SUPPLIER_CAD_DISTAL_INNER_PAD_FRAME_DERIVED",
        "method": (
            "AREA_WEIGHTED_LARGEST_CONNECTED_COPLANAR_INWARD_SURFACE_PAIR"
        ),
        "source_geometry_modified": False,
        "tessellation": {
            "freecad": "1.1.1",
            "opencascade": "7.8.1",
            "linear_deflection_mm": 0.20,
            "angular_deflection_deg": 20.0,
        },
        "fingers": fingers,
        "closed_reference_pair": {
            "finger_joint_positions_m": dict(CLOSED_FINGER_Q_M),
            "fingers_link_origin_from_gripper_reference_m": [
                GRIPPER_REFERENCE_FROM_FINGERS_LINK_X_M,
                0.0,
                0.0,
            ],
            "left_center_gripper_reference_m": _json_vector(
                centers_reference["left"]
            ),
            "right_center_gripper_reference_m": _json_vector(
                centers_reference["right"]
            ),
            "midpoint_gripper_reference_m": _json_vector(
                midpoint_reference
            ),
            "midpoint_ee_gripper_frame_m": _json_vector(midpoint_ee),
            "center_line_gripper_reference_m": _json_vector(center_line),
            "center_line_length_m": float(np.linalg.norm(center_line)),
            "symmetry": symmetry,
        },
        "frame_semantics": {
            "gripper_reference_frame": "follower_left_gripper_link",
            "official_ee_helper_frame": "follower_left_ee_gripper_link",
            "official_ee_helper_offset_from_reference_m": [
                GRIPPER_REFERENCE_FROM_EE_GRIPPER_X_M,
                0.0,
                0.0,
            ],
            "official_ee_helper_is_pad_center": bool(
                np.linalg.norm(midpoint_ee) <= 1e-6
            ),
            "required_grasp_origin": (
                "SUPPLIER_CAD_PAD_PAIR_MIDPOINT_NOT_GENERIC_EE_ORIGIN"
            ),
        },
        "thresholds": {
            "minimum_inward_normal_dot": MINIMUM_INWARD_NORMAL_DOT,
            "coplanar_normal_cosine": COPLANAR_NORMAL_COSINE,
            "coplanar_offset_tolerance_m": (
                COPLANAR_OFFSET_TOLERANCE_M
            ),
        },
        "task8": "NOT_RUN",
    }


def compare_brep_mesh_pad_evidence(
    *,
    mesh_report: dict[str, Any],
    brep_report: dict[str, Any],
) -> dict[str, Any]:
    thresholds = {
        "maximum_centroid_residual_m": 0.0002,
        "maximum_normal_angle_deg": 0.1,
        "maximum_relative_area_error": 0.001,
    }
    fingers = {}
    for side in ("left", "right"):
        mesh_pad = mesh_report["fingers"][side]["selected_inner_pad"]
        brep_pad = brep_report["fingers"][side]["selected_inner_pad"]
        mesh_center = np.asarray(
            mesh_pad["centroid_finger_link_m"],
            dtype=np.float64,
        )
        brep_center = np.asarray(
            brep_pad["center_finger_link_m"],
            dtype=np.float64,
        )
        mesh_normal = np.asarray(
            mesh_pad["normal_finger_link"],
            dtype=np.float64,
        )
        brep_normal = np.asarray(
            brep_pad["normal_finger_link"],
            dtype=np.float64,
        )
        centroid_residual = float(np.linalg.norm(mesh_center - brep_center))
        cosine = float(
            np.clip(np.dot(mesh_normal, brep_normal), -1.0, 1.0)
        )
        normal_angle = float(np.degrees(np.arccos(cosine)))
        mesh_area = float(mesh_pad["area_m2"])
        brep_area = float(brep_pad["area_mm2"]) * 1e-6
        relative_area_error = abs(mesh_area - brep_area) / brep_area
        fingers[side] = {
            "status": (
                "PASS"
                if (
                    centroid_residual
                    <= thresholds["maximum_centroid_residual_m"]
                    and normal_angle
                    <= thresholds["maximum_normal_angle_deg"]
                    and relative_area_error
                    <= thresholds["maximum_relative_area_error"]
                )
                else "FAIL"
            ),
            "mesh_triangle_count": int(mesh_pad["triangle_count"]),
            "brep_face_index_1_based": int(
                brep_pad["face_index_1_based"]
            ),
            "centroid_residual_m": centroid_residual,
            "normal_angle_deg": normal_angle,
            "mesh_area_m2": mesh_area,
            "brep_area_m2": brep_area,
            "relative_area_error": relative_area_error,
        }
    return {
        "status": (
            "PASS"
            if (
                mesh_report.get("status") == "PASS"
                and brep_report.get("status") == "PASS"
                and all(
                    record["status"] == "PASS"
                    for record in fingers.values()
                )
            )
            else "FAIL"
        ),
        "classification": "BREP_AND_CONTROLLED_MESH_SELECT_SAME_PAD_FACES",
        "fingers": fingers,
        "maximum_centroid_residual_m": max(
            record["centroid_residual_m"]
            for record in fingers.values()
        ),
        "maximum_normal_angle_deg": max(
            record["normal_angle_deg"] for record in fingers.values()
        ),
        "maximum_relative_area_error": max(
            record["relative_area_error"]
            for record in fingers.values()
        ),
        "thresholds": thresholds,
    }

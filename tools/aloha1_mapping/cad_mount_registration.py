"""Register supplier gripper-shell CAD to the follower gripper link by datums."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from tools.aloha1_mapping.cad_finger_installation import CAD_GLOBAL_TO_GRIPPER_ROTATION
from tools.aloha1_mapping.cad_finger_installation import determinant3

DATUM_KEYS = ("x_min", "y_min", "y_max", "z_max")
REGISTRATION_THRESHOLD_M = 0.0002
PLANE_TOLERANCE_M = 5.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rpy_matrix(values: list[float]) -> np.ndarray:
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],
            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if element is None:
        return matrix
    xyz = [float(value) for value in element.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in element.get("rpy", "0 0 0").split()]
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = xyz
    return matrix


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            indices = [
                int(value.split("/")[0]) - 1 for value in line.split()[1:]
            ]
            if len(indices) != 3:
                raise RuntimeError(f"non-triangle OBJ face in {path}")
            faces.append(indices)
    if not vertices or not faces:
        raise RuntimeError(f"empty OBJ mesh: {path}")
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise RuntimeError(f"invalid binary STL: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + triangle_count * 50:
        raise RuntimeError(f"unexpected binary STL size: {path}")
    record_type = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(
        payload,
        dtype=record_type,
        offset=84,
        count=triangle_count,
    )
    vertices = records["vertices"].astype(np.float64).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _transform_points(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (vertices, np.ones(len(vertices), dtype=np.float64))
    )
    return (matrix @ homogeneous.T).T[:, :3]


def _link_transforms_from(
    root: ET.Element,
    root_link: str,
) -> dict[str, np.ndarray]:
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.get("link", ""), []).append(
            (
                child.get("link", ""),
                _origin_matrix(joint.find("origin")),
            )
        )
    transforms = {root_link: np.eye(4, dtype=np.float64)}
    pending = [root_link]
    while pending:
        parent = pending.pop()
        for child, relative in children.get(parent, []):
            if child in transforms:
                continue
            transforms[child] = transforms[parent] @ relative
            pending.append(child)
    return transforms


def _urdf_visual_mesh_in_gripper_frame(
    urdf_path: Path,
    *,
    link_suffix: str,
    expected_mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    gripper_link = next(
        link.get("name", "")
        for link in root.findall("link")
        if link.get("name", "").endswith("_gripper_link")
        and not link.get("name", "").endswith("_ee_gripper_link")
    )
    target_link = next(
        link
        for link in root.findall("link")
        if link.get("name", "").endswith(link_suffix)
    )
    target_name = target_link.get("name", "")
    visual = target_link.find("visual")
    if visual is None:
        raise RuntimeError(f"visual missing for {target_name}")
    mesh = visual.find("geometry/mesh")
    if mesh is None:
        raise RuntimeError(f"visual mesh missing for {target_name}")
    filename = mesh.get("filename", "")
    if not filename.endswith(expected_mesh_path.name):
        raise RuntimeError(
            f"unexpected mesh for {target_name}: {filename}"
        )
    scale = np.asarray(
        [float(value) for value in mesh.get("scale", "1 1 1").split()],
        dtype=np.float64,
    )
    vertices, faces = _load_binary_stl(expected_mesh_path)
    vertices = vertices * scale
    transforms = _link_transforms_from(root, gripper_link)
    matrix = transforms[target_name] @ _origin_matrix(visual.find("origin"))
    return _transform_points(vertices, matrix), faces, matrix


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )


def _datum(
    vertices: np.ndarray,
    faces: np.ndarray,
    key: str,
) -> dict[str, Any]:
    axis_name, extreme_name = key.split("_")
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    coordinate = (
        float(vertices[:, axis].min())
        if extreme_name == "min"
        else float(vertices[:, axis].max())
    )
    triangles = vertices[faces]
    on_plane = np.max(
        np.abs(triangles[:, :, axis] - coordinate),
        axis=1,
    ) <= PLANE_TOLERANCE_M
    areas = _triangle_areas(vertices, faces)
    return {
        "axis": axis_name,
        "extreme": extreme_name,
        "coordinate_m": coordinate,
        "triangle_count": int(on_plane.sum()),
        "area_m2": float(areas[on_plane].sum()),
        "plane_tolerance_m": PLANE_TOLERANCE_M,
    }


def build_mount_registration_report(
    *,
    probe_manifest_path: Path,
    cad_shell_obj_path: Path,
    follower_urdf_path: Path,
    gripper_stl_path: Path,
    gripper_bar_stl_path: Path,
) -> dict[str, Any]:
    """Compare supplier and Stage geometry only on controlled planar datums."""
    manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    shell = manifest["objects"]["Part__Feature006"]
    placement = np.asarray(shell["placement_matrix_mm"], dtype=np.float64)
    cad_vertices_global, cad_faces = _load_obj(cad_shell_obj_path)
    rotation = np.asarray(
        CAD_GLOBAL_TO_GRIPPER_ROTATION,
        dtype=np.float64,
    )
    cad_vertices = (
        rotation
        @ (
            cad_vertices_global
            - placement[:3, 3][np.newaxis, :] * 0.001
        ).T
    ).T

    gripper_vertices, gripper_faces, gripper_matrix = (
        _urdf_visual_mesh_in_gripper_frame(
            follower_urdf_path,
            link_suffix="_gripper_link",
            expected_mesh_path=gripper_stl_path,
        )
    )
    bar_vertices, bar_faces, bar_matrix = (
        _urdf_visual_mesh_in_gripper_frame(
            follower_urdf_path,
            link_suffix="_gripper_bar_link",
            expected_mesh_path=gripper_bar_stl_path,
        )
    )
    stage_vertices = np.vstack((gripper_vertices, bar_vertices))
    stage_faces = np.vstack(
        (gripper_faces, bar_faces + len(gripper_vertices))
    )

    datums: dict[str, Any] = {}
    for key in DATUM_KEYS:
        cad = _datum(cad_vertices, cad_faces, key)
        stage = _datum(stage_vertices, stage_faces, key)
        residual = abs(cad["coordinate_m"] - stage["coordinate_m"])
        nonzero = (
            cad["triangle_count"] > 0
            and cad["area_m2"] > 0.0
            and stage["triangle_count"] > 0
            and stage["area_m2"] > 0.0
        )
        datums[key] = {
            "cad": cad,
            "stage": stage,
            "absolute_coordinate_residual_m": residual,
            "within_tessellation_threshold": (
                residual <= REGISTRATION_THRESHOLD_M
            ),
            "nonzero_planar_support": nonzero,
            "status": (
                "PASS"
                if nonzero and residual <= REGISTRATION_THRESHOLD_M
                else "FAIL"
            ),
        }

    proper_rotation = math.isclose(
        determinant3(CAD_GLOBAL_TO_GRIPPER_ROTATION),
        1.0,
        abs_tol=1.0e-12,
    )
    all_datums_pass = all(
        record["status"] == "PASS" for record in datums.values()
    )
    gates = {
        "source_hash_matches_probe": (
            manifest["source_sha256"]
            == "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
        ),
        "proper_rotation_no_mirror": proper_rotation,
        "four_nonzero_planar_datums": all(
            record["nonzero_planar_support"]
            for record in datums.values()
        ),
        "all_coordinate_residuals_within_0p20mm": all_datums_pass,
        "general_icp_excluded_from_decision": True,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "method": "CONTROLLED_ORTHOGONAL_PLANAR_DATUM_REGISTRATION",
        "threshold_m": REGISTRATION_THRESHOLD_M,
        "threshold_source": (
            "fixed supplier-CAD visual tessellation linear deflection 0.20 mm"
        ),
        "source_cad": {
            "absolute_path": manifest["source"],
            "sha256": manifest["source_sha256"],
            "freecad_version": manifest["freecad_version"],
            "opencascade_version": manifest["opencascade_version"],
        },
        "cad_shell": {
            "object_name": "Part__Feature006",
            "label": shell["label"],
            "obj_absolute_path": str(cad_shell_obj_path.resolve()),
            "obj_sha256": _sha256(cad_shell_obj_path),
            "vertex_count": len(cad_vertices),
            "triangle_count": len(cad_faces),
            "source_placement_matrix_mm": shell["placement_matrix_mm"],
            "cad_global_to_gripper_rotation": rotation.tolist(),
            "rotation_determinant": float(np.linalg.det(rotation)),
        },
        "stage_reference_geometry": {
            "follower_urdf_absolute_path": str(follower_urdf_path.resolve()),
            "follower_urdf_sha256": _sha256(follower_urdf_path),
            "gripper_stl": {
                "absolute_path": str(gripper_stl_path.resolve()),
                "sha256": _sha256(gripper_stl_path),
                "gripper_frame_matrix": gripper_matrix.tolist(),
            },
            "gripper_bar_stl": {
                "absolute_path": str(gripper_bar_stl_path.resolve()),
                "sha256": _sha256(gripper_bar_stl_path),
                "gripper_frame_matrix": bar_matrix.tolist(),
            },
        },
        "datums": datums,
        "decision_boundary": {
            "full_surface_icp_used": False,
            "reason": (
                "Supplier CAD and URDF meshes are different revisions and "
                "general ICP admitted local minima; it is retained only as "
                "rejected exploratory evidence and cannot select orientation."
            ),
            "classification": "SUPPLIER_CAD_TO_STAGE_DATUM_REGISTRATION",
            "physical_measurement": False,
        },
        "gates": gates,
    }


def render_mount_registration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA Viper supplier-CAD mounting datum registration",
        "",
        f"- Status: `{report['status']}`",
        f"- Method: `{report['method']}`",
        f"- Threshold: `{report['threshold_m']:.7f} m`",
        "- Full-surface ICP used for decision: `false`",
        "- Physical measurement: `false`",
        "",
        "| Datum | CAD coordinate (m) | Stage coordinate (m) | Residual (m) | CAD triangles/area | Stage triangles/area | Status |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, record in report["datums"].items():
        lines.append(
            f"| `{key}` | {record['cad']['coordinate_m']:.12g} | "
            f"{record['stage']['coordinate_m']:.12g} | "
            f"{record['absolute_coordinate_residual_m']:.12g} | "
            f"{record['cad']['triangle_count']} / "
            f"{record['cad']['area_m2']:.12g} m² | "
            f"{record['stage']['triangle_count']} / "
            f"{record['stage']['area_m2']:.12g} m² | "
            f"`{record['status']}` |"
        )
    lines.extend(
        [
            "",
            "This report validates a controlled supplier-CAD-to-Stage datum "
            "registration. It is not a physical measurement and does not "
            "validate collision, contact, or grasping.",
            "",
        ]
    )
    return "\n".join(lines)

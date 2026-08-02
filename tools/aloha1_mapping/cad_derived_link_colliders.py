"""Build traceable, isolated ALOHA CAD-derived collision candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from tools.aloha1_mapping.cad_finger_installation import CAD_GLOBAL_TO_GRIPPER_ROTATION

SUPPORTED_PROFILES = {"CAD_SUBPART_COMPOUND_CONVEX_HULL"}
SOURCE_SHA256 = "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
MAIN_LINK_SUFFIXES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
    "gripper_link",
)
FINGER_INPUTS = {
    "left_finger_link": (
        "left_finger.obj",
        "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
    ),
    "right_finger_link": (
        "right_finger.obj",
        "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
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
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if element is None:
        return matrix
    matrix[:3, 3] = [float(value) for value in element.get("xyz", "0 0 0").split()]
    matrix[:3, :3] = _rpy_matrix([float(value) for value in element.get("rpy", "0 0 0").split()])
    return matrix


def _zero_pose_link_transforms(root: ET.Element, root_link: str) -> dict[str, np.ndarray]:
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.get("link", ""), []).append(
            (child.get("link", ""), _origin_matrix(joint.find("origin")))
        )
    transforms = {root_link: np.eye(4, dtype=np.float64)}
    pending = [root_link]
    while pending:
        parent = pending.pop()
        for child, relative in children.get(parent, []):
            if child not in transforms:
                transforms[child] = transforms[parent] @ relative
                pending.append(child)
    return transforms


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            if len(face) != 3:
                raise ValueError(f"non-triangle OBJ face in {path}")
            faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"empty OBJ mesh: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"invalid binary STL: {path}")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + count * 50:
        raise ValueError(f"unexpected binary STL size: {path}")
    records = np.frombuffer(
        payload,
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
        offset=84,
        count=count,
    )
    vertices = records["vertices"].astype(np.float64).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    return (matrix @ homogeneous.T).T[:, :3]


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        "# ALOHA supplier-CAD diagnostic collision candidate",
        "# coordinate frame: owning URDF link; unit: metre",
    ]
    lines.extend("v " + " ".join(format(float(value), ".17g") for value in point) for point in vertices)
    lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _canonical_signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    triangles = []
    for face in faces:
        coordinates = [tuple(round(float(value), 9) for value in vertices[index]) for index in face]
        triangles.append(tuple(sorted(coordinates)))
    payload = json.dumps(sorted(triangles), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _aabb(vertices: np.ndarray) -> dict[str, list[float]]:
    return {
        "minimum_m": vertices.min(axis=0).tolist(),
        "maximum_m": vertices.max(axis=0).tolist(),
        "extent_m": np.ptp(vertices, axis=0).tolist(),
    }


def _surface_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.vstack(
        (
            vertices,
            triangles.mean(axis=1),
            (triangles[:, 0] + triangles[:, 1]) * 0.5,
            (triangles[:, 1] + triangles[:, 2]) * 0.5,
            (triangles[:, 2] + triangles[:, 0]) * 0.5,
        )
    )


def _surface_deviation(
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
) -> dict[str, Any]:
    candidate = _surface_samples(candidate_vertices, candidate_faces)
    reference = _surface_samples(reference_vertices, reference_faces)
    forward = cKDTree(reference).query(candidate, workers=1)[0]
    reverse = cKDTree(candidate).query(reference, workers=1)[0]
    return {
        "method": "VERTICES_CENTROIDS_EDGE_MIDPOINTS_BIDIRECTIONAL_CKDTREE",
        "candidate_sample_count": len(candidate),
        "reference_sample_count": len(reference),
        "candidate_to_reference_max_m": float(forward.max()),
        "candidate_to_reference_rms_m": float(np.sqrt(np.mean(forward**2))),
        "reference_to_candidate_max_m": float(reverse.max()),
        "reference_to_candidate_rms_m": float(np.sqrt(np.mean(reverse**2))),
        "decision_use": "DIAGNOSTIC_REVISION_DIFFERENCE_NOT_REGISTRATION_GATE",
    }


def _run_freecad(
    *, freecadcmd: Path, extractor: Path, source_step: Path, parent: Path, label: str
) -> tuple[Path, Path]:
    output = Path(tempfile.mkdtemp(prefix=f"{label}_", dir=parent)) / "output"
    log = output.parent / "freecad.log"
    environment = dict(os.environ)
    environment.update(
        {
            "ALOHA_VIPER_STEP": str(source_step.resolve(strict=True)),
            "ALOHA_CAD_LINK_OUTPUT_DIR": str(output),
        }
    )
    result = subprocess.run(
        [str(freecadcmd.resolve(strict=True)), str(extractor.resolve(strict=True))],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0 or not (output / "manifest.json").is_file():
        raise RuntimeError(f"FreeCAD extraction failed; see {log}")
    return output / "manifest.json", log


def _urdf_visual_reference(*, root: ET.Element, link_name: str, mesh_root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    link = next(item for item in root.findall("link") if item.get("name") == link_name)
    visual = link.find("visual")
    if visual is None:
        raise ValueError(f"missing visual for {link_name}")
    mesh = visual.find("geometry/mesh")
    if mesh is None:
        raise ValueError(f"missing visual mesh for {link_name}")
    mesh_path = mesh_root / Path(mesh.get("filename", "")).name
    vertices, faces = _load_binary_stl(mesh_path.resolve(strict=True))
    scale = np.asarray([float(value) for value in mesh.get("scale", "1 1 1").split()])
    vertices = vertices * scale
    return _transform(vertices, _origin_matrix(visual.find("origin"))), faces, mesh_path


def _gripper_fixed_group_reference(
    *,
    root: ET.Element,
    zero_pose: dict[str, np.ndarray],
    mesh_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    gripper_name = "follower_left_gripper_link"
    bar_name = "follower_left_gripper_bar_link"
    gripper_vertices, gripper_faces, gripper_path = _urdf_visual_reference(
        root=root,
        link_name=gripper_name,
        mesh_root=mesh_root,
    )
    bar_vertices, bar_faces, bar_path = _urdf_visual_reference(
        root=root,
        link_name=bar_name,
        mesh_root=mesh_root,
    )
    bar_to_gripper = np.linalg.inv(zero_pose[gripper_name]) @ zero_pose[bar_name]
    bar_vertices = _transform(bar_vertices, bar_to_gripper)
    return (
        np.vstack((gripper_vertices, bar_vertices)),
        np.vstack((gripper_faces, bar_faces + len(gripper_vertices))),
        [gripper_path, bar_path],
    )


def _verified_toolchain(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "opencascade_version": "7.8.1",
        "mesher_api": "MeshPart.meshFromShape",
        "linear_deflection_mm": 0.2,
        "angular_deflection_deg": 20.0,
        "relative_deflection": False,
    }
    for manifest in (run_a, run_b):
        if manifest["freecad_version"][:3] != ["1", "1", "1"]:
            raise RuntimeError(f"unexpected FreeCAD readback: {manifest['freecad_version']}")
        for key, value in expected.items():
            if manifest[key] != value:
                raise RuntimeError(f"unexpected pinned toolchain value {key}: {manifest[key]}")
    if any(run_a[key] != run_b[key] for key in expected):
        raise RuntimeError("FreeCAD run toolchain readbacks differ")
    if run_a["freecad_version"] != run_b["freecad_version"]:
        raise RuntimeError("FreeCAD version readbacks differ")
    return {
        "freecad_version": "1.1.1",
        "freecad_version_raw": run_a["freecad_version"],
        **expected,
    }


def _cad_to_link_matrix(
    *, suffix: str, zero_pose: dict[str, np.ndarray], robot_name: str, record: dict[str, Any]
) -> tuple[np.ndarray, str]:
    rotation = np.asarray(CAD_GLOBAL_TO_GRIPPER_ROTATION, dtype=np.float64)
    cad_to_base = np.eye(4, dtype=np.float64)
    cad_to_base[:3, :3] = rotation
    if suffix == "gripper_link":
        placement = np.asarray(record["source_placement_matrix_mm"], dtype=np.float64)
        matrix = cad_to_base.copy()
        matrix[:3, 3] = -rotation @ (placement[:3, 3] * 0.001)
        return matrix, "VERIFIED_GRIPPER_PLANAR_DATUM_REGISTRATION"
    link_name = f"{robot_name}_{suffix}"
    return (
        np.linalg.inv(zero_pose[link_name]) @ cad_to_base,
        "URDF_ZERO_POSE_FK_INVERSE_WITH_VERIFIED_CAD_AXIS_ROTATION",
    )


def build_candidate(*, source_step: Path, source_stage: Path, output_root: Path, profile: str) -> dict[str, Any]:
    """Build one isolated collision candidate; profile is mandatory."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported collider profile: {profile}")
    if _sha256(source_step) != SOURCE_SHA256:
        raise ValueError("supplier CAD hash drift")
    if _sha256(source_stage) != STAGE_SHA256:
        raise ValueError("approved source Stage hash drift")

    root_dir = Path(__file__).resolve().parents[2]
    artifact_root = root_dir / ".codex/artifacts/20260802-aloha1-cad-derived-colliders/phase3_tessellation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    freecadcmd = root_dir / "local_tools/freecad-tessellation/freecadcmd"
    extractor = root_dir / "tools/aloha1_mapping/extract_cad_derived_link_meshes_freecad.py"
    run_a_path, run_a_log = _run_freecad(
        freecadcmd=freecadcmd,
        extractor=extractor,
        source_step=source_step,
        parent=artifact_root,
        label="run_a",
    )
    run_b_path, run_b_log = _run_freecad(
        freecadcmd=freecadcmd,
        extractor=extractor,
        source_step=source_step,
        parent=artifact_root,
        label="run_b",
    )
    run_a = json.loads(run_a_path.read_text(encoding="utf-8"))
    run_b = json.loads(run_b_path.read_text(encoding="utf-8"))
    toolchain = _verified_toolchain(run_a, run_b)

    output_root = output_root.resolve()
    geometry_root = output_root / "geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)
    urdf = root_dir / "generated/urdf/follower_left.urdf"
    urdf_root = ET.parse(urdf).getroot()
    zero_pose = _zero_pose_link_transforms(urdf_root, "follower_left_base_link")
    mesh_root = (
        root_dir / "external/ros2-essentials/aloha_ws/src/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/"
        "aloha_vx300s_meshes"
    )

    suffix_results: dict[str, dict[str, Any]] = {}
    for suffix in MAIN_LINK_SUFFIXES:
        left = run_a["records"][suffix]
        right = run_b["records"][suffix]
        deterministic = all(
            left.get(key) == right.get(key)
            for key in (
                "status",
                "obj_sha256",
                "canonical_geometry_sha256",
                "vertex_count",
                "triangle_count",
                "aabb_mm",
                "brep_volume_mm3",
                "connected_components",
                "degenerate_triangle_count",
                "source_placement_matrix_mm",
            )
        )
        if not deterministic:
            raise RuntimeError(f"non-deterministic FreeCAD result for {suffix}")
        if left["status"] != "PASS":
            suffix_results[suffix] = {
                **left,
                "run_a_matches_run_b": True,
                "output_obj": None,
                "registration_method": None,
                "cad_to_link_matrix": np.eye(4).tolist(),
                "transform_determinant": 1.0,
                "mirror_used": False,
                "approximation": "convexHull",
                "convex_piece_count": left["connected_components"],
                "surface_deviation": None,
            }
            continue
        source_obj = Path(left["obj_path"])
        vertices, faces = _load_obj(source_obj)
        matrix, method = _cad_to_link_matrix(
            suffix=suffix,
            zero_pose=zero_pose,
            robot_name="follower_left",
            record=left,
        )
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not np.isfinite(matrix).all() or not math.isclose(determinant, 1.0, abs_tol=1.0e-12):
            raise RuntimeError(f"invalid proper transform for {suffix}")
        local_vertices = _transform(vertices, matrix)
        output_obj = geometry_root / f"{suffix}.obj"
        _write_obj(output_obj, local_vertices, faces)
        if suffix == "gripper_link":
            reference_vertices, reference_faces, reference_paths = _gripper_fixed_group_reference(
                root=urdf_root,
                zero_pose=zero_pose,
                mesh_root=mesh_root,
            )
        else:
            reference_vertices, reference_faces, reference_path = _urdf_visual_reference(
                root=urdf_root,
                link_name=f"follower_left_{suffix}",
                mesh_root=mesh_root,
            )
            reference_paths = [reference_path]
        suffix_results[suffix] = {
            **left,
            "run_a_matches_run_b": True,
            "output_obj": {
                "absolute_path": str(output_obj),
                "sha256": _sha256(output_obj),
            },
            "registration_method": method,
            "cad_to_link_matrix": matrix.tolist(),
            "transform_determinant": determinant,
            "mirror_used": False,
            "approximation": "convexHull",
            "convex_piece_count": left["connected_components"],
            "vertex_count": len(local_vertices),
            "triangle_count": len(faces),
            "aabb_link_local_m": _aabb(local_vertices),
            "canonical_geometry_sha256": _canonical_signature(local_vertices, faces),
            "output_canonical_matches_second_run": (
                left["canonical_geometry_sha256"] == right["canonical_geometry_sha256"]
            ),
            "urdf_visual_references": [
                {
                    "absolute_path": str(path.resolve(strict=True)),
                    "sha256": _sha256(path),
                }
                for path in reference_paths
            ],
            "surface_deviation": _surface_deviation(local_vertices, faces, reference_vertices, reference_faces),
        }

    semantics_path = root_dir / "reports/aloha1_mapping/aloha1_cad_link_collision_semantics.json"
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    physical_records: list[dict[str, Any]] = []
    for robot in ("follower_left", "follower_right"):
        for suffix in MAIN_LINK_SUFFIXES:
            source = suffix_results[suffix]
            physical_records.append(
                {
                    "robot": robot,
                    "urdf_link_name": f"{robot}_{suffix}",
                    "link_suffix": suffix,
                    "owner_count": 1,
                    "kind": "CAD_CANDIDATE",
                    "status": source["status"],
                    "source_object": source["object_name"],
                    "source_solid_count": source["source_solid_count"],
                    "convex_piece_count": source["convex_piece_count"],
                    "approximation": source["approximation"],
                    "output_obj": source["output_obj"],
                    "vertex_count": source["vertex_count"],
                    "triangle_count": source["triangle_count"],
                    "connected_components": source["connected_components"],
                    "degenerate_triangle_count": source["degenerate_triangle_count"],
                    "brep_volume_mm3": source["brep_volume_mm3"],
                    "aabb_link_local_m": source.get("aabb_link_local_m"),
                    "cad_to_link_matrix": source["cad_to_link_matrix"],
                    "registration_method": source["registration_method"],
                    "transform_determinant": source["transform_determinant"],
                    "unit_conversion_mm_to_m": 0.001,
                    "mirror_used": False,
                    "run_a_matches_run_b": source["run_a_matches_run_b"],
                    "canonical_geometry_sha256": source["canonical_geometry_sha256"],
                    "surface_deviation": source["surface_deviation"],
                    "fixed_group_members": (
                        [
                            f"{robot}_gripper_link",
                            f"{robot}_gripper_bar_link",
                        ]
                        if suffix == "gripper_link"
                        else [f"{robot}_{suffix}"]
                    ),
                    "downstream_authoring_rule": (
                        "AUTHOR_CAD_SHELL_ON_GRIPPER_LINK_AND_DISABLE_DUPLICATE_"
                        "BASELINE_GRIPPER_BAR_COLLIDER_IN_DIAGNOSTIC_LAYER_ONLY"
                        if suffix == "gripper_link"
                        else "AUTHOR_ON_OWNING_LINK_ONLY"
                    ),
                }
            )

    finger_root = (
        root_dir / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
        "viper_gripper/tessellation_angular_controlled/run_a"
    )
    gripper_mapping = json.loads(
        (root_dir / "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json").read_text(encoding="utf-8")
    )
    for robot in ("follower_left", "follower_right"):
        for suffix, (filename, expected_hash) in FINGER_INPUTS.items():
            path = finger_root / filename
            if _sha256(path) != expected_hash:
                raise RuntimeError(f"accepted finger hash drift: {path}")
            side = "left" if suffix.startswith("left") else "right"
            matrix = gripper_mapping["cad_to_finger_link_mapping"][f"{side}_matrix"]
            physical_records.append(
                {
                    "robot": robot,
                    "urdf_link_name": f"{robot}_{suffix}",
                    "link_suffix": suffix,
                    "owner_count": 1,
                    "kind": "ACCEPTED_FINGER",
                    "status": "PASS",
                    "source_object": "Part__Feature007" if side == "left" else "Part__Feature008",
                    "source_solid_count": 1,
                    "convex_piece_count": 1,
                    "approximation": "UNCHANGED_ACCEPTED_BASELINE",
                    "output_obj": {"absolute_path": str(path.resolve(strict=True)), "sha256": expected_hash},
                    "vertex_count": 831,
                    "triangle_count": 1662,
                    "connected_components": 1,
                    "degenerate_triangle_count": 0,
                    "brep_volume_mm3": 35160.658046734075,
                    "aabb_link_local_m": None,
                    "cad_to_link_matrix": matrix,
                    "registration_method": "VERIFIED_EXISTING_SUPPLIER_CAD_REGISTRATION",
                    "transform_determinant": 1.0,
                    "unit_conversion_mm_to_m": 0.001,
                    "mirror_used": False,
                    "run_a_matches_run_b": True,
                    "canonical_geometry_sha256": None,
                    "surface_deviation": None,
                }
            )

    virtual = [
        {
            "robot": item["robot"],
            "urdf_link_name": item["urdf_link_name"],
            "link_suffix": item["link_suffix"],
            "collider_authored": False,
            "reason": "VIRTUAL_FRAME_NO_COLLIDER",
        }
        for item in semantics["links"]
        if item["classification"] == "VIRTUAL_FRAME_NO_COLLIDER"
    ]
    blockers = [
        {
            "robot": item["robot"],
            "urdf_link_name": item["urdf_link_name"],
            "link_suffix": item["link_suffix"],
            "collider_authored": False,
            "reason": "HARD_BLOCKER_CAD_TO_LINK_IDENTITY",
        }
        for item in semantics["links"]
        if item["classification"] == "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"
    ]
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "profile": profile,
        "source_cad": {"absolute_path": str(source_step.resolve()), "sha256": SOURCE_SHA256},
        "source_stage": {"absolute_path": str(source_stage.resolve()), "sha256": STAGE_SHA256},
        "toolchain": toolchain,
        "two_fresh_directory_determinism": "PASS",
        "freecad_runs": {
            "run_a_manifest": str(run_a_path.resolve()),
            "run_a_manifest_sha256": _sha256(run_a_path),
            "run_a_log": str(run_a_log.resolve()),
            "run_a_log_sha256": _sha256(run_a_log),
            "run_b_manifest": str(run_b_path.resolve()),
            "run_b_manifest_sha256": _sha256(run_b_path),
            "run_b_log": str(run_b_log.resolve()),
            "run_b_log_sha256": _sha256(run_b_log),
        },
        "physical_link_records": physical_records,
        "virtual_frame_records": virtual,
        "identity_blockers": blockers,
        "multi_link_source_groupings": [
            {
                "source_object": "Part__Feature006",
                "cad_label": "Aloha VX Gripper 2024-4-19 v4",
                "owner_link_suffix": "gripper_link",
                "fixed_member_link_suffixes": [
                    "gripper_link",
                    "gripper_bar_link",
                ],
                "moving_gripper_prop_included": False,
                "evidence": (
                    "aloha_viper_cad_mount_registration.json compares the "
                    "supplier shell against the fixed gripper+bar URDF group"
                ),
                "diagnostic_authoring_constraint": (
                    "disable the duplicate baseline gripper_bar collider only inside the isolated diagnostic layer"
                ),
            }
        ],
        "invalid_brep_blockers": [
            record["urdf_link_name"] for record in physical_records if record["status"] == "HARD_BLOCKER_INVALID_BREP"
        ],
        "collision_mesh_policy": "ONE_CONVEX_HULL_PER_SOURCE_CAD_SOLID_GROUPED_BY_OWNING_LINK",
        "accepted_finger_colliders_modified": False,
        "source_or_imported_asset_modified": False,
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }

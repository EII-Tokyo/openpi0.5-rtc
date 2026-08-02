"""Numerically compare authoritative link meshes with per-component convex hulls."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

PHYSICAL_LINK_SUFFIXES = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
    "gripper_link",
    "gripper_prop_link",
    "gripper_bar_link",
    "left_finger_link",
    "right_finger_link",
]
FINGER_PATHS = {
    "left_finger_link": (
        ".codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/"
        "tessellation_angular_controlled/run_a/left_finger.obj"
    ),
    "right_finger_link": (
        ".codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/"
        "tessellation_angular_controlled/run_a/right_finger.obj"
    ),
}
FINGER_HASHES = {
    "left_finger_link": "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
    "right_finger_link": "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_triangles(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.float64)
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + 50 * count == len(raw):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)[
                "vertices"
            ].astype(np.float64)
            return _deduplicate_triangles(triangles)
    vertices = []
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.strip().split()
        if fields[:1] == ["vertex"] and len(fields) == 4:
            vertices.append([float(value) for value in fields[1:]])
    if len(vertices) % 3:
        raise ValueError(f"invalid ASCII STL triangle count: {path}")
    return _deduplicate_triangles(np.asarray(vertices).reshape(-1, 3, 3))


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]]
            if len(indices) < 3:
                continue
            faces.extend([indices[0], indices[index], indices[index + 1]] for index in range(1, len(indices) - 1))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".stl":
        vertices, faces = _load_stl(path)
    elif path.suffix.lower() == ".obj":
        vertices, faces = _load_obj(path)
    else:
        raise ValueError(f"unsupported mesh type: {path}")
    return vertices * scale, faces


def _face_components(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    parent = np.arange(vertex_count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[0]), int(face[2]))
    groups: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        groups.setdefault(find(int(face[0])), []).append(face_index)
    return [np.asarray(indices, dtype=np.int64) for _, indices in sorted(groups.items())]


def _surface_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    edge_midpoints = edges.mean(axis=1)
    centroids = triangles.mean(axis=1)
    return np.unique(np.concatenate((vertices, edge_midpoints, centroids)), axis=0)


def _signed_volume_abs(vertices: np.ndarray, faces: np.ndarray) -> float:
    centered = vertices - vertices.mean(axis=0)
    triangles = centered[faces]
    return abs(
        float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6.0
        )
    )


def _maximum_nearest_distance(source: np.ndarray, target: np.ndarray) -> float:
    distances, _ = cKDTree(target).query(source, workers=1)
    return float(np.max(distances))


def _certificate_for_mesh(
    *,
    suffix: str,
    path: Path,
    scale: float,
    authority: str,
) -> dict[str, Any]:
    vertices, faces = _load_mesh(path, scale)
    triangles = vertices[faces]
    extent = np.ptp(vertices, axis=0)
    numeric_scale = max(1.0, float(np.max(np.abs(vertices))), float(np.max(extent)))
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    degenerate_threshold = 256.0 * np.finfo(float).eps * numeric_scale**2
    degenerate_count = int(np.count_nonzero(area2 <= degenerate_threshold))

    source_samples = _surface_samples(vertices, faces)
    all_hull_samples = []
    hull_volume = 0.0
    source_volume = 0.0
    containment_margin = -np.inf
    piece_records = []
    components = _face_components(faces, len(vertices))
    for component_index, component_faces_index in enumerate(components):
        component_faces_global = faces[component_faces_index]
        component_vertex_global = np.unique(component_faces_global)
        component_vertices = vertices[component_vertex_global]
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[component_vertex_global] = np.arange(len(component_vertex_global))
        component_faces = remap[component_faces_global]
        hull = ConvexHull(component_vertices)
        hull_faces = np.asarray(hull.simplices, dtype=np.int64)
        hull_samples = _surface_samples(component_vertices, hull_faces)
        all_hull_samples.append(hull_samples)
        hull_volume += float(hull.volume)
        component_source_volume = _signed_volume_abs(
            component_vertices, component_faces
        )
        source_volume += component_source_volume
        equation_values = (
            component_vertices @ hull.equations[:, :3].T
            + hull.equations[:, 3]
        )
        component_margin = float(np.max(equation_values))
        containment_margin = max(containment_margin, component_margin)
        piece_records.append(
            {
                "piece_index": component_index,
                "source_vertex_count": len(component_vertices),
                "source_face_count": len(component_faces),
                "hull_vertex_count": len(hull.vertices),
                "hull_face_count": len(hull.simplices),
                "source_signed_volume_abs_m3": component_source_volume,
                "hull_volume_m3": float(hull.volume),
                "containment_max_halfspace_value_m": component_margin,
            }
        )
    hull_samples = np.unique(np.concatenate(all_hull_samples), axis=0)
    containment_tolerance = 128.0 * np.finfo(float).eps * numeric_scale
    return {
        "link_suffix": suffix,
        "source_authority": authority,
        "source_path": str(path.resolve()),
        "source_sha256": _sha256(path),
        "source_scale_to_m": scale,
        "mirror_used": False,
        "convex_policy": "ONE_CONVEX_HULL_PER_CONNECTED_SOURCE_COMPONENT",
        "connected_component_count": len(components),
        "convex_piece_count": len(components),
        "source_vertex_count": len(vertices),
        "source_face_count": len(faces),
        "source_sample_count": len(source_samples),
        "hull_sample_count": len(hull_samples),
        "degenerate_triangle_count": degenerate_count,
        "source_mesh_quality_status": (
            "SOURCE_HAS_DEGENERATE_TRIANGLES_RECORDED_NOT_REPAIRED"
            if degenerate_count
            else "PASS_NO_DEGENERATE_TRIANGLES"
        ),
        "aabb_min_m": vertices.min(axis=0).tolist(),
        "aabb_max_m": vertices.max(axis=0).tolist(),
        "source_signed_volume_abs_m3": source_volume,
        "hull_volume_m3": hull_volume,
        "hull_to_source_volume_ratio": hull_volume / source_volume,
        "source_to_hull_sample_max_m": _maximum_nearest_distance(
            source_samples, hull_samples
        ),
        "hull_to_source_sample_max_m": _maximum_nearest_distance(
            hull_samples, source_samples
        ),
        "source_contained_by_hulls": bool(
            containment_margin <= containment_tolerance
        ),
        "containment_max_halfspace_value_m": containment_margin,
        "containment_numeric_tolerance_m": containment_tolerance,
        "surface_distance_method": (
            "DETERMINISTIC_VERTICES_EDGE_MIDPOINTS_CENTROIDS_CKDTREE_APPROXIMATION"
        ),
        "pieces": piece_records,
    }


def _finger_contact_surface_audit(
    path: Path, face: dict[str, Any]
) -> dict[str, Any]:
    vertices, faces = _load_obj(path)
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
        raise ValueError(f"no contact-surface samples selected from {path}")
    hull = ConvexHull(vertices)
    normals = hull.equations[:, :3]
    signed_halfspaces = (
        contact_samples @ normals.T + hull.equations[:, 3]
    )
    boundary_distances = np.min(
        -signed_halfspaces / np.linalg.norm(normals, axis=1), axis=1
    )
    boundary_distances = np.maximum(boundary_distances, 0.0)
    maximum = float(np.max(boundary_distances))
    return {
        "cad_face_index": int(face["face_index"]),
        "cad_face_normal": normal.tolist(),
        "cad_face_center_m": center.tolist(),
        "sample_selection": (
            "authoritative mesh vertices, edge midpoints and triangle centroids within "
            "the FreeCAD face AABB and 0.20 mm of its plane"
        ),
        "sample_count": len(contact_samples),
        "tessellation_error_budget_m": budget,
        "source_to_hull_boundary_min_m": float(np.min(boundary_distances)),
        "source_to_hull_boundary_mean_m": float(np.mean(boundary_distances)),
        "source_to_hull_boundary_max_m": maximum,
        "status": (
            "FAIL_SINGLE_HULL_RECESSES_CONTACT_SURFACE"
            if maximum > budget
            else "PASS_WITHIN_TESSELLATION_ERROR_BUDGET"
        ),
    }


def _mesh_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"only frozen file mesh URIs are accepted: {uri}")
    return Path(unquote(parsed.path)).resolve(strict=True)


def _official_sources_from_urdf(root: Path) -> dict[str, tuple[Path, float, str]]:
    urdf = ET.parse(root / "generated/urdf/follower_left.urdf").getroot()
    records: dict[str, tuple[Path, float, str]] = {}
    prefix = "follower_left_"
    for link in urdf.findall("link"):
        name = link.attrib["name"]
        suffix = name.removeprefix(prefix)
        mesh = link.find("./collision/geometry/mesh")
        if mesh is None or suffix in FINGER_PATHS:
            continue
        scale_values = [float(value) for value in mesh.attrib.get("scale", "1 1 1").split()]
        if len(set(scale_values)) != 1:
            raise ValueError(f"non-uniform mesh scale for {suffix}")
        records[suffix] = (
            _mesh_uri_to_path(mesh.attrib["filename"]),
            scale_values[0],
            "PINNED_INTERBOTIX_ALOHA_VX300S_URDF_COLLISION_MESH",
        )
    for suffix, relative in FINGER_PATHS.items():
        path = (root / relative).resolve(strict=True)
        if _sha256(path) != FINGER_HASHES[suffix]:
            raise ValueError(f"supplier CAD handed finger hash mismatch: {suffix}")
        records[suffix] = (
            path,
            1.0,
            "SUPPLIER_ASSEMBLY_EMBEDDED_HANDED_FINGER_BREP_TESSELLATION",
        )
    return records


def build_certificate(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    sources = _official_sources_from_urdf(root)
    missing = [suffix for suffix in PHYSICAL_LINK_SUFFIXES if suffix not in sources]
    if missing:
        raise ValueError(f"physical link source mesh missing: {missing}")
    records = [
        _certificate_for_mesh(
            suffix=suffix,
            path=sources[suffix][0],
            scale=sources[suffix][1],
            authority=sources[suffix][2],
        )
        for suffix in PHYSICAL_LINK_SUFFIXES
    ]
    geometry_probe_path = (
        root / "reports/aloha1_mapping/aloha1_cad_source_geometry_probe.json"
    )
    geometry_probe = json.loads(geometry_probe_path.read_text(encoding="utf-8"))
    records_by_suffix = {record["link_suffix"]: record for record in records}
    for side in ("left", "right"):
        suffix = f"{side}_finger_link"
        records_by_suffix[suffix]["inward_contact_surface"] = (
            _finger_contact_surface_audit(
                sources[suffix][0],
                geometry_probe["finger_contact_surfaces"][side],
            )
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "ROBOT_LOCAL_OFFLINE_COLLIDER_SURFACE_CERTIFICATE_NO_USD_AUTHORING",
        "link_suffixes": PHYSICAL_LINK_SUFFIXES,
        "source_completeness": "PASS",
        "surface_error_certificate": "COMPLETE_NUMERICAL",
        "convex_policy": "ONE_CONVEX_HULL_PER_CONNECTED_SOURCE_COMPONENT",
        "acceptance_tolerance": None,
        "acceptance_status": "HARD_BLOCKER_ERROR_BUDGET_NOT_DEFINED",
        "records": records,
        "summary": {
            "link_count": len(records),
            "convex_piece_count": sum(item["convex_piece_count"] for item in records),
            "maximum_hull_to_source_sample_distance_m": max(
                item["hull_to_source_sample_max_m"] for item in records
            ),
            "maximum_volume_ratio": max(
                item["hull_to_source_volume_ratio"] for item in records
            ),
        },
        "method_limitations": [
            "surface distances use deterministic finite samples, not an exact analytic Hausdorff distance",
            "the report certifies source-to-hull geometry only; PhysX cooked readback remains a separate runtime gate",
            "no millimetre tolerance is invented without an official or task-derived numerical error budget",
        ],
        "finger_contact_surface_summary": {
            suffix: records_by_suffix[suffix]["inward_contact_surface"]
            for suffix in ("left_finger_link", "right_finger_link")
        },
        "final_or_default_asset_modified": False,
        "isaac_runtime_started": False,
        "real_robot_accessed": False,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["deterministic_signature"] = hashlib.sha256(payload).hexdigest()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows: Iterable[str] = (
        f"| `{item['link_suffix']}` | {item['connected_component_count']} | "
        f"{item['hull_to_source_volume_ratio']:.6g} | "
        f"{item['source_to_hull_sample_max_m']:.6g} | "
        f"{item['hull_to_source_sample_max_m']:.6g} |"
        for item in report["records"]
    )
    return "\n".join(
        [
            "# ALOHA1 official-source collider surface certificate",
            "",
            f"- Status: **{report['status']}**",
            f"- Numerical coverage: **{report['surface_error_certificate']}**",
            f"- Acceptance: **{report['acceptance_status']}**",
            "- Isaac runtime started: `false`",
            "- Final/default asset modified: `false`",
            "",
            "| Link | Components/hulls | Hull/source volume | source→hull max (m) | hull→source max (m) |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "Every physical link now has a deterministic numerical source-to-convex-hull "
            "record. This does not automatically make every hull acceptable: the finite-sample "
            "surface errors and volume growth require a task-derived or official error budget. "
            "No tolerance was fitted from successful grasp videos.",
            "",
        ]
    )

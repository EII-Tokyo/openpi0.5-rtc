#!/usr/bin/env python3
"""Utilities for the Iteration 000 ALOHA reference model.

This module intentionally treats the current Isaac ALOHA source assets as
read-only reference geometry. It extracts the zero-joint visual mesh placements
needed to create a FreeCAD reference file and repeatable review renders.
"""

from __future__ import annotations

import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
MJCF_DIR = REPO_ROOT / "external" / "mujoco_menagerie" / "aloha"
ALOHA_XML = MJCF_DIR / "aloha.xml"
SCENE_XML = MJCF_DIR / "scene.xml"
ASSET_DIR = MJCF_DIR / "assets"


@dataclass(frozen=True)
class MeshReference:
    name: str
    mesh_name: str
    mesh_path: Path
    matrix_m: tuple[tuple[float, float, float, float], ...]
    source_xml: Path
    category: str = "robot"


def _vec(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(x) for x in value.split())


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def _translate(xyz: Iterable[float]) -> list[list[float]]:
    m = _identity()
    vals = list(xyz)
    m[0][3], m[1][3], m[2][3] = vals[0], vals[1], vals[2]
    return m


def _scale(xyz: Iterable[float]) -> list[list[float]]:
    vals = list(xyz)
    m = _identity()
    m[0][0], m[1][1], m[2][2] = vals[0], vals[1], vals[2]
    return m


def _quat_to_matrix(wxyz: Iterable[float]) -> list[list[float]]:
    w, x, y, z = list(wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return _identity()
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _euler_xyz_to_matrix(rpy: Iterable[float]) -> list[list[float]]:
    rx, ry, rz = list(rpy)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]]
    my = [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]]
    mz = [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return _matmul(_matmul(mx, my), mz)


def _element_transform(elem: ET.Element) -> list[list[float]]:
    transform = _translate(_vec(elem.get("pos"), (0.0, 0.0, 0.0)))
    if elem.get("quat") is not None:
        transform = _matmul(transform, _quat_to_matrix(_vec(elem.get("quat"), (1.0, 0.0, 0.0, 0.0))))
    if elem.get("euler") is not None:
        transform = _matmul(transform, _euler_xyz_to_matrix(_vec(elem.get("euler"), (0.0, 0.0, 0.0))))
    return transform


def _sanitize_name(text: str) -> str:
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "unnamed"


def _mesh_asset_scales(root: ET.Element) -> dict[str, tuple[Path, tuple[float, float, float]]]:
    out: dict[str, tuple[Path, tuple[float, float, float]]] = {}
    for mesh in root.findall(".//asset/mesh"):
        file_name = mesh.get("file")
        if not file_name:
            continue
        name = mesh.get("name") or Path(file_name).stem
        scale = _vec(mesh.get("scale"), (1.0, 1.0, 1.0))
        out[name] = (ASSET_DIR / file_name, scale)  # MJCF scale converts raw mesh units to meters.
    return out


def load_aloha_visual_meshes() -> list[MeshReference]:
    root = ET.parse(ALOHA_XML).getroot()
    assets = _mesh_asset_scales(root)
    refs: list[MeshReference] = []

    def walk_body(body: ET.Element, parent_matrix: list[list[float]], body_path: list[str]) -> None:
        body_name = body.get("name") or "body"
        current_path = [*body_path, body_name]
        body_matrix = _matmul(parent_matrix, _element_transform(body))
        geom_index = 0
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue
            geom_class = geom.get("class", "")
            if "visual" not in geom_class:
                continue
            if mesh_name not in assets:
                continue
            mesh_path, asset_scale = assets[mesh_name]
            geom_matrix = _matmul(body_matrix, _element_transform(geom))
            matrix_m = _matmul(geom_matrix, _scale(asset_scale))
            clean = _sanitize_name("_".join(current_path + [mesh_name, str(geom_index)]))
            refs.append(
                MeshReference(
                    name=f"REF_ALOHA_{clean}",
                    mesh_name=mesh_name,
                    mesh_path=mesh_path,
                    matrix_m=tuple(tuple(row) for row in matrix_m),
                    source_xml=ALOHA_XML,
                    category="robot",
                )
            )
            geom_index += 1
        for child in body.findall("body"):
            walk_body(child, body_matrix, current_path)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody in {ALOHA_XML}")
    for body in worldbody.findall("body"):
        walk_body(body, _identity(), [])
    return refs


def load_scene_static_meshes() -> list[MeshReference]:
    root = ET.parse(SCENE_XML).getroot()
    assets = _mesh_asset_scales(root)
    refs: list[MeshReference] = []
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody in {SCENE_XML}")
    geom_index = 0
    for geom in worldbody.findall("geom"):
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_name not in assets:
            continue
        mesh_path, asset_scale = assets[mesh_name]
        matrix_m = _matmul(_element_transform(geom), _scale(asset_scale))
        category = "scene"
        if mesh_name.startswith("table"):
            category = "table"
        elif "extrusion" in mesh_name or "bracket" in mesh_name or "mount" in mesh_name:
            category = "frame"
        elif "d405" in mesh_name:
            category = "camera"
        name = _sanitize_name(f"REF_SCENE_{category}_{mesh_name}_{geom_index}")
        refs.append(
            MeshReference(
                name=name,
                mesh_name=mesh_name,
                mesh_path=mesh_path,
                matrix_m=tuple(tuple(row) for row in matrix_m),
                source_xml=SCENE_XML,
                category=category,
            )
        )
        geom_index += 1
    return refs


def load_current_isaac_reference_meshes() -> list[MeshReference]:
    # Current Isaac stage is generated from scene.xml; combine the included robot
    # model with the scene-level table/frame/camera meshes.
    return load_aloha_visual_meshes() + load_scene_static_meshes()


def matrix_m_to_mm(matrix_m: tuple[tuple[float, float, float, float], ...]) -> list[list[float]]:
    return [[float(value) * 1000.0 for value in row] for row in matrix_m]


def transform_points(points, matrix_m):
    import numpy as np

    pts = np.asarray(points, dtype=float)
    linear = np.asarray([row[:3] for row in matrix_m[:3]], dtype=float)
    trans = np.asarray([row[3] for row in matrix_m[:3]], dtype=float)
    return pts @ linear.T + trans


def read_stl_triangles(path: Path):
    import numpy as np

    data = path.read_bytes()
    if len(data) > 84:
        n = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + n * 50
        if expected == len(data):
            tris = np.empty((n, 3, 3), dtype=float)
            offset = 84
            for idx in range(n):
                values = struct.unpack_from("<12fH", data, offset)
                tris[idx, :, :] = [values[3:6], values[6:9], values[9:12]]
                offset += 50
            return tris
    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            vertices.append([float(x) for x in line.split()[1:4]])
    arr = np.asarray(vertices, dtype=float)
    if len(arr) % 3 != 0:
        raise RuntimeError(f"Cannot parse STL triangles: {path}")
    return arr.reshape((-1, 3, 3))


def read_obj_triangles(path: Path):
    import numpy as np

    vertices: list[list[float]] = []
    triangles: list[list[list[float]]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            indices = []
            for item in parts[1:]:
                head = item.split("/")[0]
                idx = int(head)
                if idx < 0:
                    idx = len(vertices) + idx + 1
                indices.append(idx - 1)
            for i in range(1, len(indices) - 1):
                triangles.append([vertices[indices[0]], vertices[indices[i]], vertices[indices[i + 1]]])
    if not triangles:
        raise RuntimeError(f"Cannot parse OBJ triangles: {path}")
    return np.asarray(triangles, dtype=float)


def read_mesh_triangles(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return read_stl_triangles(path)
    if suffix == ".obj":
        return read_obj_triangles(path)
    raise RuntimeError(f"Unsupported mesh type for render: {path}")


def table_reference_box_mm() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # min/max corners in millimeters for the transparent desktop reference plane.
    return (-605.0, -380.0, -18.0), (605.0, 380.0, 0.0)


def known_dimensions() -> dict[str, object]:
    return {
        "model": "Current Isaac ALOHA dual-arm table/frame reference",
        "current_isaac_stage": "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd",
        "source_mjcf": str(ALOHA_XML.relative_to(REPO_ROOT)),
        "source_scene": str(SCENE_XML.relative_to(REPO_ROOT)),
        "table_length_width_height_mm": [1210.0, 760.0, 750.0],
        "left_base_pos_mm": [-469.0, -19.0, 20.0],
        "right_base_pos_mm": [469.0, -19.0, 20.0],
        "official_reach_mm": 750.0,
    }

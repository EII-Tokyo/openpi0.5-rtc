"""Export the embedded Simple Viper handed finger B-Reps deterministically.

Required environment variables:
ALOHA_VIPER_STEP
ALOHA_VIPER_TESSELLATION_OUTPUT_DIR

This local FreeCAD snap only exposes linear deflection through
Part.Shape.tessellate.  The output is therefore a diagnostic visual mesh, not
an angular-deflection-controlled production mesh and not a collision mesh.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import FreeCAD as App
import Import
import Part

LINEAR_DEFLECTION_MM = 0.20
EXPECTED_SOURCE_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
FINGERS = {
    "left_finger": {
        "object_name": "Part__Feature007",
        "label": "Aloha VX Fingers 2024-4-21 v2",
        "cad_side": "+X",
    },
    "right_finger": {
        "object_name": "Part__Feature008",
        "label": "Aloha VX Fingers 2024-4-21 v001",
        "cad_side": "-X",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(placement: object) -> list[list[float]]:
    values = [float(value) for value in placement.toMatrix().A]
    return [values[index : index + 4] for index in range(0, 16, 4)]


def _bbox(shape: object) -> dict[str, float]:
    box = shape.BoundBox
    return {
        name: float(getattr(box, name))
        for name in (
            "XMin",
            "YMin",
            "ZMin",
            "XMax",
            "YMax",
            "ZMax",
            "XLength",
            "YLength",
            "ZLength",
        )
    }


def _triangle_area(
    points: list[list[float]],
    triangle: list[int],
) -> float:
    a, b, c = (points[index] for index in triangle)
    ab = [b[index] - a[index] for index in range(3)]
    ac = [c[index] - a[index] for index in range(3)]
    cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def _connected_components(
    vertex_count: int,
    triangles: list[list[int]],
) -> int:
    parent = list(range(vertex_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    used: set[int] = set()
    for triangle in triangles:
        used.update(triangle)
        union(triangle[0], triangle[1])
        union(triangle[1], triangle[2])
    return len({find(index) for index in used})


def _canonical_signature(
    points: list[list[float]],
    triangles: list[list[int]],
) -> str:
    canonical_triangles = []
    for triangle in triangles:
        coordinates = [
            tuple(round(value, 9) for value in points[index])
            for index in triangle
        ]
        canonical_triangles.append(tuple(sorted(coordinates)))
    payload = json.dumps(
        sorted(canonical_triangles),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_obj(
    path: Path,
    points_mm: list[list[float]],
    triangles: list[list[int]],
) -> None:
    lines = [
        "# ALOHA Simple Viper embedded finger diagnostic visual mesh",
        "# source unit: millimetre; OBJ coordinate unit: metre",
        "# scale: 0.001 m/mm",
    ]
    lines.extend(
        (
            "v " + " ".join(format(value * 0.001, ".17g") for value in point)
        )
        for point in points_mm
    )
    lines.extend(
        (
            "f " + " ".join(str(index + 1) for index in triangle)
        )
        for triangle in triangles
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


source_text = os.environ.get("ALOHA_VIPER_STEP")
output_text = os.environ.get("ALOHA_VIPER_TESSELLATION_OUTPUT_DIR")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_VIPER_STEP and ALOHA_VIPER_TESSELLATION_OUTPUT_DIR are required"
    )

source = Path(source_text).resolve(strict=True)
source_sha256 = _sha256(source)
if source_sha256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(f"unexpected Simple Viper source hash: {source_sha256}")
output_dir = Path(output_text).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

imported = Import.open(str(source))
document = (
    getattr(imported[0], "Document", None)
    if isinstance(imported, list | tuple) and imported
    else imported
)
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError(f"FreeCAD did not create a document for {source}")
    document = documents[-1]

objects_by_name = {obj.Name: obj for obj in document.Objects}
records = {}
for joint_name, expected in FINGERS.items():
    obj = objects_by_name.get(expected["object_name"])
    if obj is None or obj.Label != expected["label"]:
        raise RuntimeError(
            f"{joint_name} source instance mismatch: "
            f"{expected['object_name']} / {expected['label']}"
        )
    shape = obj.Shape
    vertices, facets = shape.tessellate(LINEAR_DEFLECTION_MM)
    points = [
        [float(point.x), float(point.y), float(point.z)]
        for point in vertices
    ]
    triangles = [[int(index) for index in facet] for facet in facets]
    obj_path = output_dir / f"{joint_name}.obj"
    _write_obj(obj_path, points, triangles)
    degenerate_count = sum(
        _triangle_area(points, triangle) <= 1.0e-18
        for triangle in triangles
    )
    placement_matrix = _matrix(obj.Placement)
    records[joint_name] = {
        **expected,
        "assembly_path": [
            "Dummy_Aloha_VX_v3",
            "Aloha_VX_Fingers_2024_4_21_v2",
            expected["object_name"],
        ],
        "source_placement_matrix_mm": placement_matrix,
        "source_placement_determinant": float(
            App.Matrix(obj.Placement.toMatrix()).determinant()
        ),
        "obj_path": str(obj_path),
        "obj_sha256": _sha256(obj_path),
        "canonical_geometry_sha256": _canonical_signature(
            points,
            triangles,
        ),
        "vertex_count": len(points),
        "triangle_count": len(triangles),
        "aabb_mm": _bbox(shape),
        "brep_volume_mm3": float(shape.Volume),
        "connected_components": _connected_components(
            len(points),
            triangles,
        ),
        "degenerate_triangle_count": degenerate_count,
        "normal_winding_policy": (
            "preserve Part.Shape.tessellate vertex and triangle winding"
        ),
    }

manifest_path = output_dir / "manifest.json"
manifest = {
    "schema_version": 1,
    "status": "PARTIAL",
    "scope": (
        "LINEAR_DEFLECTION_ONLY_DIAGNOSTIC_VISUAL_MESH; "
        "NOT_COLLISION_MESH; NOT_FINAL_ASSET"
    ),
    "source_path": str(source),
    "source_sha256": source_sha256,
    "freecad_version": list(App.Version()),
    "opencascade_version": str(Part.OCC_VERSION),
    "linear_deflection_mm": LINEAR_DEFLECTION_MM,
    "angular_deflection": "NOT_APPLIED_HARD_BLOCKER",
    "unit_scale_m_per_mm": 0.001,
    "weld_sew_policy": "NONE",
    "instance_merge_policy": "NONE",
    "finger_shape_or_handedness_modified": False,
    "meshes": records,
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("ALOHA_VIPER_FINGER_TESSELLATION", manifest_path)
for name, record in records.items():
    print(
        name,
        record["vertex_count"],
        record["triangle_count"],
        record["obj_sha256"],
    )
App.closeDocument(document.Name)

"""Extract ALOHA Viper link B-Reps with the pinned FreeCAD mesher.

Required environment variables:
ALOHA_VIPER_STEP
ALOHA_CAD_LINK_OUTPUT_DIR

The emitted OBJ files remain in the supplier STEP global coordinate frame,
with millimetres converted to metres.  Link-local registration is deliberately
performed by the outer project runner, where URDF zero-pose transforms can be
audited independently of FreeCAD.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import FreeCAD as App
import Import
import MeshPart
import Part

LINEAR_DEFLECTION_MM = 0.20
ANGULAR_DEFLECTION_DEG = 20.0
ANGULAR_DEFLECTION_RAD = math.radians(ANGULAR_DEFLECTION_DEG)
SOURCE_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
CAD_LINKS = {
    "base_link": ("Part__Feature", "Simple VX Base v3"),
    "shoulder_link": ("Part__Feature001", "Simple VX Shoulder v3"),
    "upper_arm_link": ("Part__Feature002", "Simple VX Upper Arm v3"),
    "upper_forearm_link": ("Part__Feature003", "Simple VX Forearm v3"),
    "lower_forearm_link": ("Part__Feature004", "Simple VX Wrist Twist v3"),
    "wrist_link": ("Part__Feature005", "Simple VX Wrist Tilt v3"),
    "gripper_link": ("Part__Feature006", "Aloha VX Gripper 2024-4-19 v4"),
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


def _triangle_area(points: list[list[float]], triangle: list[int]) -> float:
    a, b, c = (points[index] for index in triangle)
    ab = [b[index] - a[index] for index in range(3)]
    ac = [c[index] - a[index] for index in range(3)]
    cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def _connected_components(vertex_count: int, triangles: list[list[int]]) -> int:
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
    points_m: list[list[float]], triangles: list[list[int]]
) -> str:
    canonical = []
    for triangle in triangles:
        coordinates = [
            tuple(round(value, 9) for value in points_m[index])
            for index in triangle
        ]
        canonical.append(tuple(sorted(coordinates)))
    payload = json.dumps(sorted(canonical), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_obj(
    path: Path, points_m: list[list[float]], triangles: list[list[int]]
) -> None:
    lines = [
        "# ALOHA supplier-CAD diagnostic link mesh",
        "# source frame: STEP global; unit: metre",
    ]
    lines.extend(
        "v " + " ".join(format(value, ".17g") for value in point)
        for point in points_m
    )
    lines.extend(
        "f " + " ".join(str(index + 1) for index in triangle)
        for triangle in triangles
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


source_text = os.environ.get("ALOHA_VIPER_STEP")
output_text = os.environ.get("ALOHA_CAD_LINK_OUTPUT_DIR")
if not source_text or not output_text:
    raise RuntimeError("ALOHA_VIPER_STEP and ALOHA_CAD_LINK_OUTPUT_DIR are required")

source = Path(source_text).resolve(strict=True)
if _sha256(source) != SOURCE_SHA256:
    raise RuntimeError("supplier CAD hash does not match the frozen input")
output_dir = Path(output_text).resolve()
output_dir.mkdir(parents=True, exist_ok=False)

imported = Import.open(str(source))
document = (
    getattr(imported[0], "Document", None)
    if isinstance(imported, list | tuple) and imported
    else imported
)
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError("FreeCAD did not create a STEP document")
    document = documents[-1]

objects = {obj.Name: obj for obj in document.Objects}
records: dict[str, object] = {}
for link_suffix, (object_name, expected_label) in CAD_LINKS.items():
    obj = objects.get(object_name)
    if obj is None or obj.Label != expected_label:
        raise RuntimeError(
            f"CAD identity mismatch for {link_suffix}: {object_name} / {expected_label}"
        )
    shape = obj.Shape
    valid = bool(not shape.isNull() and shape.isValid())
    record = {
        "link_suffix": link_suffix,
        "object_name": object_name,
        "label": obj.Label,
        "shape_type": str(shape.ShapeType),
        "brep_is_valid": valid,
        "source_solid_count": len(shape.Solids),
        "brep_volume_mm3": float(shape.Volume),
        "brep_area_mm2": float(shape.Area),
        "aabb_mm": _bbox(shape),
        "source_placement_matrix_mm": _matrix(obj.Placement),
        "source_placement_determinant": float(
            App.Matrix(obj.Placement.toMatrix()).determinant()
        ),
    }
    if not valid:
        record.update(
            {
                "status": "HARD_BLOCKER_INVALID_BREP",
                "obj_path": None,
                "obj_sha256": None,
                "canonical_geometry_sha256": None,
                "vertex_count": 0,
                "triangle_count": 0,
                "connected_components": 0,
                "degenerate_triangle_count": 0,
            }
        )
        records[link_suffix] = record
        continue

    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=LINEAR_DEFLECTION_MM,
        AngularDeflection=ANGULAR_DEFLECTION_RAD,
        Relative=False,
    )
    vertices, facets = mesh.Topology
    points_m = [
        [float(point.x) * 0.001, float(point.y) * 0.001, float(point.z) * 0.001]
        for point in vertices
    ]
    triangles = [[int(index) for index in facet] for facet in facets]
    obj_path = output_dir / f"{link_suffix}.obj"
    _write_obj(obj_path, points_m, triangles)
    record.update(
        {
            "status": "PASS",
            "obj_path": str(obj_path),
            "obj_sha256": _sha256(obj_path),
            "canonical_geometry_sha256": _canonical_signature(points_m, triangles),
            "vertex_count": len(points_m),
            "triangle_count": len(triangles),
            "connected_components": _connected_components(len(points_m), triangles),
            "degenerate_triangle_count": sum(
                _triangle_area(points_m, triangle) <= 1.0e-18
                for triangle in triangles
            ),
        }
    )
    records[link_suffix] = record

manifest = {
    "schema_version": 1,
    "status": "PARTIAL"
    if any(record["status"] != "PASS" for record in records.values())
    else "PASS",
    "scope": "SUPPLIER_CAD_GLOBAL_LINK_MESH_EXTRACTION_NOT_FINAL_COLLIDER",
    "source_path": str(source),
    "source_sha256": SOURCE_SHA256,
    "freecad_version": list(App.Version()),
    "opencascade_version": str(Part.OCC_VERSION),
    "mesher_api": "MeshPart.meshFromShape",
    "meshpart_module_path": str(Path(MeshPart.__file__).resolve()),
    "linear_deflection_mm": LINEAR_DEFLECTION_MM,
    "angular_deflection_deg": ANGULAR_DEFLECTION_DEG,
    "angular_deflection_rad": ANGULAR_DEFLECTION_RAD,
    "relative_deflection": False,
    "unit_scale_m_per_mm": 0.001,
    "normal_winding_policy": "preserve MeshPart.meshFromShape topology winding",
    "weld_sew_policy": "NONE",
    "instance_merge_policy": "NONE",
    "records": records,
}
(output_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("ALOHA_CAD_LINK_EXTRACTION", output_dir / "manifest.json")
App.closeDocument(document.Name)

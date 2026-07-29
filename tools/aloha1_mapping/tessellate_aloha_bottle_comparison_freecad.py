"""Tessellate the project bottle and downloaded reference for visual review.

Required environment variables:
  ALOHA_PROJECT_BOTTLE_FCSTD
  ALOHA_REFERENCE_BOTTLE_STEP
  ALOHA_BOTTLE_COMPARISON_MESH_DIR

Meshes are diagnostic visual evidence only. They are not collision meshes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import FreeCAD as App
import Import
import MeshPart
import Part

LINEAR_DEFLECTION_MM = 0.20
ANGULAR_DEFLECTION_DEG = 20.0
ANGULAR_DEFLECTION_RAD = math.radians(ANGULAR_DEFLECTION_DEG)
MM_TO_M = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _canonical_signature(
    points: list[list[float]],
    triangles: list[list[int]],
) -> str:
    canonical_triangles = []
    for triangle in triangles:
        coordinates = [tuple(round(value, 9) for value in points[index]) for index in triangle]
        canonical_triangles.append(tuple(sorted(coordinates)))
    payload = json.dumps(
        sorted(canonical_triangles),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def _write_obj(
    path: Path,
    points_mm: list[list[float]],
    triangles: list[list[int]],
) -> None:
    lines = [
        "# ALOHA bottle CAD comparison diagnostic visual mesh",
        "# source unit: millimetre; OBJ coordinate unit: metre",
        "# collision use prohibited",
    ]
    lines.extend("v " + " ".join(format(value * MM_TO_M, ".17g") for value in point) for point in points_mm)
    lines.extend("f " + " ".join(str(index + 1) for index in triangle) for triangle in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _mesh_record(
    *,
    asset_id: str,
    shape: Any,
    source: Path,
    output_dir: Path,
    source_axis: str,
) -> dict[str, Any]:
    brep_bounds = shape.BoundBox
    brep_aabb = {
        "x_min": float(brep_bounds.XMin),
        "x_max": float(brep_bounds.XMax),
        "x_length": float(brep_bounds.XLength),
        "y_min": float(brep_bounds.YMin),
        "y_max": float(brep_bounds.YMax),
        "y_length": float(brep_bounds.YLength),
        "z_min": float(brep_bounds.ZMin),
        "z_max": float(brep_bounds.ZMax),
        "z_length": float(brep_bounds.ZLength),
    }
    optimal_bounds = shape.optimalBoundingBox()
    optimal_brep_aabb = {
        "call": "Part.Shape.optimalBoundingBox()",
        "is_valid": bool(optimal_bounds.isValid()),
        "x_min": float(optimal_bounds.XMin),
        "x_max": float(optimal_bounds.XMax),
        "x_length": float(optimal_bounds.XLength),
        "y_min": float(optimal_bounds.YMin),
        "y_max": float(optimal_bounds.YMax),
        "y_length": float(optimal_bounds.YLength),
        "z_min": float(optimal_bounds.ZMin),
        "z_max": float(optimal_bounds.ZMax),
        "z_length": float(optimal_bounds.ZLength),
    }
    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=LINEAR_DEFLECTION_MM,
        AngularDeflection=ANGULAR_DEFLECTION_RAD,
        Relative=False,
    )
    vertices, facets = mesh.Topology
    points = [[float(point.x), float(point.y), float(point.z)] for point in vertices]
    triangles = [[int(index) for index in facet] for facet in facets]
    mesh_aabb = {
        "x_min": min(point[0] for point in points),
        "x_max": max(point[0] for point in points),
        "x_length": (max(point[0] for point in points) - min(point[0] for point in points)),
        "y_min": min(point[1] for point in points),
        "y_max": max(point[1] for point in points),
        "y_length": (max(point[1] for point in points) - min(point[1] for point in points)),
        "z_min": min(point[2] for point in points),
        "z_max": max(point[2] for point in points),
        "z_length": (max(point[2] for point in points) - min(point[2] for point in points)),
    }
    obj_path = output_dir / f"{asset_id}.obj"
    _write_obj(obj_path, points, triangles)
    return {
        "asset_id": asset_id,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "source_long_axis": source_axis,
        "obj_path": str(obj_path),
        "obj_sha256": _sha256(obj_path),
        "canonical_geometry_sha256": _canonical_signature(
            points,
            triangles,
        ),
        "vertex_count": len(points),
        "triangle_count": len(triangles),
        "connected_components": _connected_components(
            len(points),
            triangles,
        ),
        "degenerate_triangle_count": sum(_triangle_area(points, triangle) <= 1.0e-18 for triangle in triangles),
        "brep_aabb_mm_before_tessellation": brep_aabb,
        "optimal_brep_aabb_mm_before_tessellation": optimal_brep_aabb,
        "mesh_aabb_mm": mesh_aabb,
        "mesh_to_brep_extent_delta_mm": {
            axis: mesh_aabb[f"{axis}_length"] - brep_aabb[f"{axis}_length"] for axis in ("x", "y", "z")
        },
        "brep_volume_mm3": float(shape.Volume),
        "brep_area_mm2": float(shape.Area),
        "normal_winding_policy": ("preserve MeshPart.meshFromShape topology winding"),
    }


project_fcstd = Path(os.environ["ALOHA_PROJECT_BOTTLE_FCSTD"]).resolve(strict=True)
reference_step = Path(os.environ["ALOHA_REFERENCE_BOTTLE_STEP"]).resolve(strict=True)
output_dir = Path(os.environ["ALOHA_BOTTLE_COMPARISON_MESH_DIR"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

project_document = App.openDocument(str(project_fcstd))
project_master = project_document.getObject("BottleMaster")
if project_master is None or project_master.Shape.isNull():
    raise RuntimeError("project FCStd does not contain a valid BottleMaster")
project_record = _mesh_record(
    asset_id="project_main_bottle",
    shape=project_master.Shape,
    source=project_fcstd,
    output_dir=output_dir,
    source_axis="+Z",
)
App.closeDocument(project_document.Name)

Import.open(str(reference_step))
reference_documents = list(App.listDocuments().values())
reference_candidates = [
    obj
    for document in reference_documents
    for obj in document.Objects
    if getattr(obj, "Shape", None) is not None and not obj.Shape.isNull()
]
if not reference_candidates:
    raise RuntimeError("reference STEP produced no physical shape")
reference_root = max(
    reference_candidates,
    key=lambda obj: (float(obj.Shape.Volume), float(obj.Shape.Area)),
)
reference_record = _mesh_record(
    asset_id="downloaded_reference_bottle",
    shape=reference_root.Shape,
    source=reference_step,
    output_dir=output_dir,
    source_axis="+Y",
)

manifest = {
    "schema_version": 1,
    "status": "PASS",
    "scope": ("CAD_COMPARISON_DIAGNOSTIC_VISUAL_MESH; NOT_COLLISION_MESH; NOT_FINAL_ASSET"),
    "runtime": {
        "freecad_version": ".".join(App.Version()[:3]),
        "freecad_build": App.Version()[3],
        "opencascade_version": Part.OCC_VERSION,
        "python_version": platform.python_version(),
    },
    "mesher_api": "MeshPart.meshFromShape",
    "meshpart_module_path": str(Path(MeshPart.__file__).resolve()),
    "linear_deflection_mm": LINEAR_DEFLECTION_MM,
    "angular_deflection_deg": ANGULAR_DEFLECTION_DEG,
    "angular_deflection_rad": ANGULAR_DEFLECTION_RAD,
    "relative_deflection": False,
    "unit_scale_m_per_mm": MM_TO_M,
    "weld_sew_policy": "NONE",
    "instance_merge_policy": "NONE",
    "collision_use": "PROHIBITED_UNTIL_SEPARATE_VALIDATION",
    "assets": {
        "project_main_bottle": project_record,
        "downloaded_reference_bottle": reference_record,
    },
}
manifest_path = output_dir / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"status={manifest['status']}")
print(f"manifest={manifest_path}")
for asset_id, record in manifest["assets"].items():
    print(
        asset_id,
        record["vertex_count"],
        record["triangle_count"],
        record["obj_sha256"],
    )

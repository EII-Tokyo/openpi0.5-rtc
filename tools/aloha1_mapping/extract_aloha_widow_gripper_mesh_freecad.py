"""Extract deterministic tessellations from the authoritative Widow gripper.

Required environment variables:
ALOHA_WIDOW_GRIPPER_STEP
ALOHA_WIDOW_GRIPPER_MESH_OUTPUT
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import FreeCAD as App
import Import


DEFLECTION_MM = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(placement: object) -> list[list[float]]:
    values = [float(value) for value in placement.toMatrix().A]
    return [values[index : index + 4] for index in range(0, 16, 4)]


def _mesh_record(obj: object) -> dict[str, object]:
    shape = obj.Shape
    vertices, triangles = shape.tessellate(DEFLECTION_MM)
    points = [[float(point.x), float(point.y), float(point.z)] for point in vertices]
    faces = [[int(index) for index in triangle] for triangle in triangles]
    return {
        "object_name": obj.Name,
        "label": obj.Label,
        "placement_matrix": _matrix(obj.Placement),
        "shape_placement_matrix": _matrix(shape.Placement),
        "vertex_count": len(points),
        "triangle_count": len(faces),
        "vertices_mm": points,
        "triangles": faces,
    }


def _relationship(left: object, right: object) -> dict[str, object]:
    distance = left.Shape.distToShape(right.Shape)
    common = left.Shape.common(right.Shape)
    return {
        "left_object": left.Name,
        "right_object": right.Name,
        "minimum_shape_distance_mm": float(distance[0]),
        "closest_point_pair_count": len(distance[1]),
        "common_volume_mm3": float(common.Volume),
        "common_solid_count": len(common.Solids),
        "common_face_count": len(common.Faces),
    }


source_text = os.environ.get("ALOHA_WIDOW_GRIPPER_STEP")
output_text = os.environ.get("ALOHA_WIDOW_GRIPPER_MESH_OUTPUT")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_WIDOW_GRIPPER_STEP and "
        "ALOHA_WIDOW_GRIPPER_MESH_OUTPUT are required"
    )

source = Path(source_text).resolve(strict=True)
imported = Import.open(str(source))
if isinstance(imported, (list, tuple)) and imported:
    document = getattr(imported[0], "Document", None)
else:
    document = imported
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError(f"FreeCAD did not create a document for {source}")
    document = documents[-1]

by_label = {obj.Label: obj for obj in document.Objects}
required = {
    "gripper_shell": "Dummy Aloha WX Gripper 2024-5-13 v2",
    "cad_positive_x_finger": "Aloha VX Fingers 2024-4-21 v2",
    "cad_negative_x_finger": "Aloha VX Fingers 2024-4-21 v001",
}
missing = sorted(label for label in required.values() if label not in by_label)
if missing:
    raise RuntimeError(f"authoritative gripper objects are missing: {missing}")

report = {
    "schema_version": 1,
    "status": "PASS",
    "source_path": str(source),
    "source_sha256": _sha256(source),
    "freecad_version": list(App.Version()),
    "deflection_mm": DEFLECTION_MM,
    "meshes": {
        role: _mesh_record(by_label[label]) for role, label in required.items()
    },
    "relationships": {
        "gripper_to_positive_x_finger": _relationship(
            by_label[required["gripper_shell"]],
            by_label[required["cad_positive_x_finger"]],
        ),
        "gripper_to_negative_x_finger": _relationship(
            by_label[required["gripper_shell"]],
            by_label[required["cad_negative_x_finger"]],
        ),
        "finger_to_finger": _relationship(
            by_label[required["cad_positive_x_finger"]],
            by_label[required["cad_negative_x_finger"]],
        ),
    },
}
output = Path(output_text)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, separators=(",", ":"), ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("ALOHA_WIDOW_GRIPPER_MESH", output)
for role, mesh in report["meshes"].items():
    print(role, mesh["object_name"], mesh["vertex_count"], mesh["triangle_count"])
App.closeDocument(document.Name)

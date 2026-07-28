"""Probe closed/open B-Rep geometry for the purchase-confirmed Viper follower.

Required environment variables:
ALOHA_VIPER_STEP
ALOHA_VIPER_GRIPPER_STATE_OUTPUT
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import FreeCAD as App
import Import

DEFLECTION_MM = 0.20
OPEN_DELTA_MM = 36.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_translated(shape: object, translation: list[float]) -> object:
    result = shape.copy()
    result.translate(App.Vector(*translation))
    return result


def _bbox(shape: object) -> dict[str, float]:
    bounds = shape.BoundBox
    return {
        name: float(getattr(bounds, name))
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


def _mesh_record(shape: object, *, source_object: object) -> dict[str, object]:
    vertices, triangles = shape.tessellate(DEFLECTION_MM)
    return {
        "source_object_name": source_object.Name,
        "source_label": source_object.Label,
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "bbox_mm": _bbox(shape),
        "volume_mm3": float(shape.Volume),
        "vertices_mm": [
            [float(point.x), float(point.y), float(point.z)]
            for point in vertices
        ],
        "triangles": [
            [int(index) for index in triangle] for triangle in triangles
        ],
    }


def _relationship(
    left: object,
    right: object,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    distance = left.distToShape(right)
    common = left.common(right)
    return {
        "left": left_name,
        "right": right_name,
        "minimum_shape_distance_mm": float(distance[0]),
        "closest_point_pair_count": len(distance[1]),
        "common_volume_mm3": float(common.Volume),
        "common_solid_count": len(common.Solids),
        "common_face_count": len(common.Faces),
    }


source_text = os.environ.get("ALOHA_VIPER_STEP")
output_text = os.environ.get("ALOHA_VIPER_GRIPPER_STATE_OUTPUT")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_VIPER_STEP and ALOHA_VIPER_GRIPPER_STATE_OUTPUT are required"
    )

source = Path(source_text).resolve(strict=True)
imported = Import.open(str(source))
document = None
if isinstance(imported, list | tuple) and imported:
    document = getattr(imported[0], "Document", None)
elif imported is not None:
    document = imported
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError(f"FreeCAD did not create a document for {source}")
    document = documents[-1]

by_label = {obj.Label: obj for obj in document.Objects}
labels = {
    "gripper_shell": "Aloha VX Gripper 2024-4-19 v4",
    "cad_positive_x_finger": "Aloha VX Fingers 2024-4-21 v2",
    "cad_negative_x_finger": "Aloha VX Fingers 2024-4-21 v001",
}
missing = sorted(label for label in labels.values() if label not in by_label)
if missing:
    raise RuntimeError(f"Viper gripper objects are missing: {missing}")
objects = {role: by_label[label] for role, label in labels.items()}

translations = {
    "closed": {
        "cad_positive_x_finger": [0.0, 0.0, 0.0],
        "cad_negative_x_finger": [0.0, 0.0, 0.0],
    },
    "open": {
        "cad_positive_x_finger": [OPEN_DELTA_MM, 0.0, 0.0],
        "cad_negative_x_finger": [-OPEN_DELTA_MM, 0.0, 0.0],
    },
}
states = {}
gripper_shape = objects["gripper_shell"].Shape.copy()
for state_name, state_translations in translations.items():
    positive_shape = _copy_translated(
        objects["cad_positive_x_finger"].Shape,
        state_translations["cad_positive_x_finger"],
    )
    negative_shape = _copy_translated(
        objects["cad_negative_x_finger"].Shape,
        state_translations["cad_negative_x_finger"],
    )
    states[state_name] = {
        "translations_mm": state_translations,
        "meshes": {
            "gripper_shell": _mesh_record(
                gripper_shape,
                source_object=objects["gripper_shell"],
            ),
            "cad_positive_x_finger": _mesh_record(
                positive_shape,
                source_object=objects["cad_positive_x_finger"],
            ),
            "cad_negative_x_finger": _mesh_record(
                negative_shape,
                source_object=objects["cad_negative_x_finger"],
            ),
        },
        "relationships": {
            "finger_to_finger": _relationship(
                positive_shape,
                negative_shape,
                left_name="cad_positive_x_finger",
                right_name="cad_negative_x_finger",
            ),
            "gripper_to_positive_x_finger": _relationship(
                gripper_shape,
                positive_shape,
                left_name="gripper_shell",
                right_name="cad_positive_x_finger",
            ),
            "gripper_to_negative_x_finger": _relationship(
                gripper_shape,
                negative_shape,
                left_name="gripper_shell",
                right_name="cad_negative_x_finger",
            ),
        },
    }

report = {
    "schema_version": 1,
    "status": "PASS",
    "source_path": str(source),
    "source_sha256": _sha256(source),
    "freecad_version": list(App.Version()),
    "deflection_mm": DEFLECTION_MM,
    "static_cad_state": "CLOSED_REFERENCE",
    "open_delta_mm": OPEN_DELTA_MM,
    "states": states,
}
output = Path(output_text)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, separators=(",", ":"), ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("ALOHA_VIPER_GRIPPER_STATES", output)
for state_name, state in states.items():
    relation = state["relationships"]["finger_to_finger"]
    print(
        state_name,
        "distance_mm=",
        relation["minimum_shape_distance_mm"],
        "common_volume_mm3=",
        relation["common_volume_mm3"],
    )
App.closeDocument(document.Name)

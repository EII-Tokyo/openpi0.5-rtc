"""Audit the installed supplier-CAD inner finger pad faces with FreeCAD.

Required environment variables:
  ALOHA_VIPER_STEP
  ALOHA_VIPER_GRASP_FACE_REPORT

This script must run with the project-pinned FreeCAD 1.1.1 runtime.  It is a
read-only B-Rep cross-check for the deterministic OBJ-derived grasp frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import FreeCAD as App
import Import
import Part

EXPECTED_STEP_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
CAD_TO_FINGER_LINK = {
    "left": (
        (0.0, -1.0, 0.0, -0.49899999973392),
        (1.0, 0.0, 0.0, -0.021099890257662776),
        (0.0, 0.0, 1.0, -0.42680133373174),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "right": (
        (0.0, -1.0, 0.0, -0.49899999973392),
        (1.0, 0.0, 0.0, 0.020900109742337226),
        (0.0, 0.0, 1.0, -0.42680133373174),
        (0.0, 0.0, 0.0, 1.0),
    ),
}
FINGERS = {
    "left": {
        "object_name": "Part__Feature007",
        "expected_label": "Aloha VX Fingers 2024-4-21 v2",
        "inward_axis": (0.0, -1.0, 0.0),
    },
    "right": {
        "object_name": "Part__Feature008",
        "expected_label": "Aloha VX Fingers 2024-4-21 v001",
        "inward_axis": (0.0, 1.0, 0.0),
    },
}
MINIMUM_INWARD_NORMAL_DOT = math.cos(math.radians(12.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point_to_finger_link(
    side: str,
    point_mm: tuple[float, float, float],
) -> list[float]:
    point_m = [value * 0.001 for value in point_mm] + [1.0]
    return [
        float(
            sum(
                CAD_TO_FINGER_LINK[side][row][column] * point_m[column]
                for column in range(4)
            )
        )
        for row in range(3)
    ]


def _normal_to_finger_link(
    side: str,
    normal: tuple[float, float, float],
) -> list[float]:
    transformed = [
        float(
            sum(
                CAD_TO_FINGER_LINK[side][row][column] * normal[column]
                for column in range(3)
            )
        )
        for row in range(3)
    ]
    length = math.sqrt(sum(value * value for value in transformed))
    return [value / length for value in transformed]


def _dot(left: list[float], right: tuple[float, float, float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _face_record(side: str, index: int, face: Any) -> dict[str, Any]:
    center = face.CenterOfMass
    parameter_source = "SURFACE_PARAMETER_AT_FACE_CENTER_OF_MASS"
    try:
        u_value, v_value = face.Surface.parameter(center)
    except Exception:
        u_min, u_max, v_min, v_max = face.ParameterRange
        u_value = (float(u_min) + float(u_max)) / 2.0
        v_value = (float(v_min) + float(v_max)) / 2.0
        parameter_source = "PARAMETER_RANGE_MIDPOINT_FALLBACK"
    normal = face.normalAt(float(u_value), float(v_value))
    center_global_mm = (
        float(center.x),
        float(center.y),
        float(center.z),
    )
    normal_global = (
        float(normal.x),
        float(normal.y),
        float(normal.z),
    )
    normal_link = _normal_to_finger_link(side, normal_global)
    surface_type = type(face.Surface).__name__
    return {
        "face_index_1_based": index,
        "surface_type": surface_type,
        "is_planar": surface_type.lower() == "plane",
        "area_mm2": float(face.Area),
        "center_global_mm": list(center_global_mm),
        "center_finger_link_m": _point_to_finger_link(
            side,
            center_global_mm,
        ),
        "normal_global": list(normal_global),
        "normal_finger_link": normal_link,
        "inward_normal_dot": _dot(
            normal_link,
            FINGERS[side]["inward_axis"],
        ),
        "normal_parameter_source": parameter_source,
        "orientation": str(face.Orientation),
    }


source_text = os.environ.get("ALOHA_VIPER_STEP")
output_text = os.environ.get("ALOHA_VIPER_GRASP_FACE_REPORT")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_VIPER_STEP and ALOHA_VIPER_GRASP_FACE_REPORT are required"
    )

source = Path(source_text).resolve(strict=True)
output = Path(output_text).resolve()
source_sha256 = _sha256(source)
if source_sha256 != EXPECTED_STEP_SHA256:
    raise RuntimeError(f"unexpected Simple Viper STEP hash: {source_sha256}")

imported = Import.open(str(source))
document = (
    getattr(imported[0], "Document", None)
    if isinstance(imported, list | tuple) and imported
    else imported
)
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError("FreeCAD did not create a document")
    document = documents[-1]

records: dict[str, Any] = {}
for side, contract in FINGERS.items():
    obj = document.getObject(contract["object_name"])
    if obj is None:
        raise RuntimeError(f"missing CAD object {contract['object_name']}")
    if str(obj.Label) != contract["expected_label"]:
        raise RuntimeError(
            f"unexpected label for {contract['object_name']}: {obj.Label}"
        )
    if obj.Shape.isNull() or not obj.Shape.isValid():
        raise RuntimeError(f"invalid B-Rep for {contract['object_name']}")
    faces = [
        _face_record(side, index, face)
        for index, face in enumerate(obj.Shape.Faces, start=1)
    ]
    candidates = sorted(
        (
            face
            for face in faces
            if face["is_planar"]
            and face["inward_normal_dot"] >= MINIMUM_INWARD_NORMAL_DOT
        ),
        key=lambda face: float(face["area_mm2"]),
        reverse=True,
    )
    if len(candidates) < 2:
        raise RuntimeError(f"insufficient inner face candidates: {side}")
    selected = dict(candidates[0])
    selected["area_ratio_to_next_candidate"] = float(
        selected["area_mm2"] / candidates[1]["area_mm2"]
    )
    selected["selection_rule"] = (
        "LARGEST_PLANAR_INWARD_FACING_BREP_FACE"
    )
    records[side] = {
        "object_name": str(obj.Name),
        "label": str(obj.Label),
        "brep_valid": bool(obj.Shape.isValid()),
        "face_count": len(faces),
        "selected_inner_pad": selected,
        "candidate_faces": candidates,
    }

report = {
    "schema_version": 1,
    "status": "PASS",
    "classification": "SUPPLIER_STEP_BREP_INNER_PAD_FACE_CROSSCHECK",
    "source": {
        "absolute_path": str(source),
        "sha256": source_sha256,
        "read_only": True,
    },
    "toolchain": {
        "freecad": App.Version(),
        "opencascade": str(Part.OCC_VERSION),
        "required_freecad": "1.1.1",
        "required_opencascade": "7.8.1",
    },
    "selection": {
        "minimum_inward_normal_dot": MINIMUM_INWARD_NORMAL_DOT,
        "no_mirror_applied": True,
        "no_shape_change": True,
    },
    "fingers": records,
    "task8": "NOT_RUN",
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))

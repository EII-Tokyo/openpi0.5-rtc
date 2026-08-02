"""Sample the audited supplier-CAD finger contact faces from their B-Reps.

Run this file only with the project-pinned FreeCAD 1.1.1 / OCCT 7.8.1
``freecadcmd`` wrapper.  The STEP input is opened read-only and no CAD file is
saved.  Points come from exact OCCT face/edge evaluation, not from an OBJ mesh.

Required environment variables:
  ALOHA_VIPER_STEP
  ALOHA_VIPER_BREP_CONTACT_SAMPLES
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
FINGERS = {
    "left": {
        "object_name": "Part__Feature007",
        "expected_label": "Aloha VX Fingers 2024-4-21 v2",
        "face_index_1_based": 117,
        "expected_inward_normal": (-0.9945218953682733, 0.10452846326765405, 0.0),
    },
    "right": {
        "object_name": "Part__Feature008",
        "expected_label": "Aloha VX Fingers 2024-4-21 v001",
        "face_index_1_based": 128,
        "expected_inward_normal": (0.9945218953682733, 0.10452846326765403, 0.0),
    },
}
UV_SAMPLES_PER_AXIS = 41
EDGE_SAMPLES_PER_EDGE = 65
FACE_MEMBERSHIP_TOLERANCE_MM = 1.0e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(point: Any) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def _linspace(lower: float, upper: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("sample count must be at least two")
    return [
        lower + (upper - lower) * index / (count - 1)
        for index in range(count)
    ]


def _unique_points(points: list[list[float]]) -> list[list[float]]:
    unique: dict[tuple[float, float, float], list[float]] = {}
    for point in points:
        key = tuple(round(value, 12) for value in point)
        unique.setdefault(key, point)
    return [unique[key] for key in sorted(unique)]


def _sample_face(face: Any) -> dict[str, Any]:
    u_min, u_max, v_min, v_max = [float(value) for value in face.ParameterRange]
    uv_points: list[list[float]] = []
    rejected_uv_count = 0
    for u_value in _linspace(u_min, u_max, UV_SAMPLES_PER_AXIS):
        for v_value in _linspace(v_min, v_max, UV_SAMPLES_PER_AXIS):
            point = face.valueAt(u_value, v_value)
            if face.isInside(
                point,
                FACE_MEMBERSHIP_TOLERANCE_MM,
                True,  # noqa: FBT003 - FreeCAD binding is positional-only
            ):
                uv_points.append(_vector(point))
            else:
                rejected_uv_count += 1

    boundary_points: list[list[float]] = []
    for edge in face.Edges:
        parameter_min, parameter_max = [
            float(value) for value in edge.ParameterRange
        ]
        boundary_points.extend(
            _vector(edge.valueAt(parameter))
            for parameter in _linspace(
                parameter_min, parameter_max, EDGE_SAMPLES_PER_EDGE
            )
        )
    vertex_points = [_vector(vertex.Point) for vertex in face.Vertexes]
    center = face.CenterOfMass
    try:
        center_u, center_v = face.Surface.parameter(center)
    except Exception:
        center_u = 0.5 * (u_min + u_max)
        center_v = 0.5 * (v_min + v_max)
    normal = face.normalAt(float(center_u), float(center_v))
    all_points = _unique_points(
        uv_points + boundary_points + vertex_points + [_vector(center)]
    )
    return {
        "surface_type": type(face.Surface).__name__,
        "orientation": str(face.Orientation),
        "area_mm2": float(face.Area),
        "parameter_range": [u_min, u_max, v_min, v_max],
        "center_mm": _vector(center),
        "normal": _vector(normal),
        "uv_grid": {
            "samples_per_axis": UV_SAMPLES_PER_AXIS,
            "accepted_count": len(uv_points),
            "rejected_outside_trimmed_face_count": rejected_uv_count,
            "membership_api": "Part.Face.isInside(point, tolerance, True)",
            "membership_tolerance_mm": FACE_MEMBERSHIP_TOLERANCE_MM,
        },
        "boundary_sampling": {
            "edge_count": len(face.Edges),
            "samples_per_edge": EDGE_SAMPLES_PER_EDGE,
            "raw_sample_count": len(boundary_points),
            "evaluation_api": "Part.Edge.valueAt(parameter)",
        },
        "vertex_count": len(face.Vertexes),
        "sample_count": len(all_points),
        "samples_mm": all_points,
    }


source_text = os.environ.get("ALOHA_VIPER_STEP")
output_text = os.environ.get("ALOHA_VIPER_BREP_CONTACT_SAMPLES")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_VIPER_STEP and ALOHA_VIPER_BREP_CONTACT_SAMPLES are required"
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
        raise RuntimeError(f"missing CAD object: {contract['object_name']}")
    if str(obj.Label) != contract["expected_label"]:
        raise RuntimeError(f"unexpected label: {obj.Label}")
    if obj.Shape.isNull() or not obj.Shape.isValid():
        raise RuntimeError(f"invalid finger B-Rep: {side}")
    face_index = int(contract["face_index_1_based"])
    face = obj.Shape.Faces[face_index - 1]
    record = _sample_face(face)
    if record["surface_type"].lower() != "plane":
        raise RuntimeError(f"audited contact face is no longer planar: {side}")
    normal = record["normal"]
    expected = contract["expected_inward_normal"]
    normal_length = math.sqrt(sum(value * value for value in normal))
    expected_length = math.sqrt(sum(value * value for value in expected))
    normal_dot = sum(
        value * expected_value
        for value, expected_value in zip(normal, expected, strict=True)
    ) / (normal_length * expected_length)
    if normal_dot < 1.0 - 1.0e-10:
        raise RuntimeError(f"audited contact normal changed: {side}")
    record.update(
        {
            "object_name": str(obj.Name),
            "label": str(obj.Label),
            "face_index_1_based": face_index,
            "brep_valid": bool(obj.Shape.isValid()),
            "normal_dot_expected": float(normal_dot),
        }
    )
    records[side] = record

report = {
    "schema_version": 1,
    "status": "PASS",
    "classification": "EXACT_OCCT_BREP_CONTACT_FACE_SAMPLES",
    "process_id": os.getpid(),
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
    "sampling": {
        "coordinate_frame": "supplier STEP assembly",
        "length_unit": "millimetre",
        "source_geometry": "trimmed OCCT B-Rep face",
        "no_tessellation_used_for_points": True,
        "no_mirror_applied": True,
        "no_shape_change": True,
    },
    "fingers": records,
}
signature_payload = dict(report)
signature_payload.pop("process_id")
payload = json.dumps(
    signature_payload, sort_keys=True, separators=(",", ":")
).encode()
report["deterministic_signature"] = hashlib.sha256(payload).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"status": report["status"], "output": str(output)}))

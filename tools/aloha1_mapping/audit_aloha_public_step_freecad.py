"""Read-only FreeCAD audit of public ALOHA STEP/AP214 sources.

This file is executed by the installed FreeCAD interpreter. Inputs and output
are passed through task-specific environment variables so the FreeCAD launcher
does not need to parse project arguments:

ALOHA_FREECAD_STEP_INPUTS_JSON
    JSON object mapping stable source labels to absolute STEP paths.
ALOHA_FREECAD_STEP_AUDIT_OUTPUT
    Absolute path for the machine-readable audit.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    components = [_number(getattr(value, axis, None)) for axis in "xyz"]
    if any(component is None for component in components):
        return None
    return components


def _matrix(placement: Any) -> list[list[float]] | None:
    if placement is None:
        return None
    try:
        matrix = placement.toMatrix()
    except AttributeError:
        matrix = getattr(placement, "Matrix", None)
    if matrix is None:
        return None
    values = getattr(matrix, "A", None)
    if values is None:
        return None
    values = [_number(value) for value in values]
    if len(values) != 16 or any(value is None for value in values):
        return None
    return [values[index : index + 4] for index in range(0, 16, 4)]


def _placement(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "base_mm": _vector(getattr(value, "Base", None)),
        "rotation_xyzw": list(getattr(getattr(value, "Rotation", None), "Q", [])),
        "matrix": _matrix(value),
    }


def _bbox(value: Any) -> dict[str, float | None] | None:
    if value is None:
        return None
    names = (
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
    return {name: _number(getattr(value, name, None)) for name in names}


def _object_names(values: Any) -> list[str]:
    if not values:
        return []
    return [str(getattr(value, "Name", value)) for value in values]


def _shape_record(shape: Any) -> dict[str, Any] | None:
    if shape is None:
        return None
    try:
        is_null = bool(shape.isNull())
    except Exception:
        is_null = False
    if is_null:
        return {"is_null": True}

    def count(name: str) -> int | None:
        value = getattr(shape, name, None)
        try:
            return len(value)
        except TypeError:
            return None

    try:
        valid = bool(shape.isValid())
    except Exception:
        valid = None
    try:
        hash_code = int(shape.hashCode())
    except Exception:
        hash_code = None
    return {
        "is_null": False,
        "shape_type": str(getattr(shape, "ShapeType", "")),
        "topology_counts": {
            "vertexes": count("Vertexes"),
            "edges": count("Edges"),
            "wires": count("Wires"),
            "faces": count("Faces"),
            "shells": count("Shells"),
            "solids": count("Solids"),
            "compsolids": count("CompSolids"),
            "compounds": count("Compounds"),
        },
        "bound_box_mm": _bbox(getattr(shape, "BoundBox", None)),
        "volume_mm3": _number(getattr(shape, "Volume", None)),
        "area_mm2": _number(getattr(shape, "Area", None)),
        "center_of_mass_mm": _vector(getattr(shape, "CenterOfMass", None)),
        "placement": _placement(getattr(shape, "Placement", None)),
        "hash_code_runtime_only": hash_code,
        "is_valid": valid,
    }


def _property_summary(obj: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in getattr(obj, "PropertiesList", []):
        if name not in {
            "Label",
            "Label2",
            "LinkTransform",
            "LinkedObject",
            "Placement",
            "ProductName",
            "SourceFile",
            "StepId",
            "TypeId",
        }:
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            summary[name] = {"read_error": repr(exc)}
            continue
        if hasattr(value, "Name"):
            summary[name] = value.Name
        elif isinstance(value, str | int | float | bool) or value is None:
            summary[name] = value
        else:
            summary[name] = str(value)
    return summary


def _object_record(obj: Any) -> dict[str, Any]:
    try:
        global_placement = obj.getGlobalPlacement()
    except Exception:
        global_placement = None
    shape = getattr(obj, "Shape", None)
    return {
        "name": str(getattr(obj, "Name", "")),
        "label": str(getattr(obj, "Label", "")),
        "type_id": str(getattr(obj, "TypeId", "")),
        "properties": list(getattr(obj, "PropertiesList", [])),
        "selected_property_values": _property_summary(obj),
        "in_list": _object_names(getattr(obj, "InList", [])),
        "out_list": _object_names(getattr(obj, "OutList", [])),
        "local_placement": _placement(getattr(obj, "Placement", None)),
        "global_placement": _placement(global_placement),
        "shape": _shape_record(shape),
    }


def _audit_source(label: str, path: Path) -> dict[str, Any]:
    imported = Import.open(str(path))
    document = (
        getattr(imported[0], "Document", None)
        if isinstance(imported, list | tuple) and imported
        else imported
    )
    if document is None:
        documents = list(App.listDocuments().values())
        if not documents:
            raise RuntimeError(f"Import.open created no document for {path}")
        document = documents[-1]
    objects = [_object_record(obj) for obj in document.Objects]
    record = {
        "source_label": label,
        "path": str(path),
        "sha256": _sha256(path),
        "document_name": document.Name,
        "document_label": document.Label,
        "object_count": len(objects),
        "shape_object_count": sum(
            1
            for item in objects
            if item["shape"] is not None and not item["shape"].get("is_null")
        ),
        "root_objects": [
            item["name"] for item in objects if not item["in_list"]
        ],
        "objects": objects,
    }
    App.closeDocument(document.Name)
    return record


inputs_text = os.environ.get("ALOHA_FREECAD_STEP_INPUTS_JSON")
output_text = os.environ.get("ALOHA_FREECAD_STEP_AUDIT_OUTPUT")
if not inputs_text or not output_text:
    raise RuntimeError(
        "ALOHA_FREECAD_STEP_INPUTS_JSON and "
        "ALOHA_FREECAD_STEP_AUDIT_OUTPUT are required"
    )

inputs = json.loads(inputs_text)
report = {
    "schema_version": 1,
    "status": "PASS",
    "freecad_version": list(App.Version()),
    "unit_interpretation": "FreeCAD geometry readback recorded in millimetres",
    "sources": [],
}
for source_label, source_path in inputs.items():
    report["sources"].append(
        _audit_source(source_label, Path(source_path).resolve(strict=True))
    )

output = Path(output_text)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("ALOHA_FREECAD_STEP_AUDIT", output)
for source in report["sources"]:
    print(
        source["source_label"],
        "objects=",
        source["object_count"],
        "shapes=",
        source["shape_object_count"],
        "roots=",
        source["root_objects"],
    )

"""Read-only FreeCAD audit for the canonical ALOHA 500 mL bottle STEP.

Environment:
  ALOHA_BOTTLE_STEP: absolute source STEP path
  ALOHA_BOTTLE_AUDIT_OUTPUT: absolute JSON output path
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

import FreeCAD as App
import Import
import Part


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(shape: Any) -> dict[str, float]:
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


def _optimal_bbox(shape: Any) -> dict[str, Any]:
    bounds = shape.optimalBoundingBox()
    return {
        "call": "Part.Shape.optimalBoundingBox()",
        "is_valid": bool(bounds.isValid()),
        **{
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
        },
    }


def _topology(shape: Any) -> dict[str, int]:
    return {
        name.lower(): len(getattr(shape, name))
        for name in (
            "Vertexes",
            "Edges",
            "Wires",
            "Faces",
            "Shells",
            "Solids",
            "CompSolids",
            "Compounds",
        )
    }


def _shape_record(shape: Any) -> dict[str, Any]:
    brep = shape.exportBrepToString()
    brep_bytes = brep.encode("utf-8") if isinstance(brep, str) else bytes(brep)
    center = getattr(shape, "CenterOfMass", None)
    return {
        "shape_type": str(shape.ShapeType),
        "is_null": bool(shape.isNull()),
        "is_valid": bool(shape.isValid()),
        "topology": _topology(shape),
        "bbox_mm": _bbox(shape),
        "optimal_bbox_mm": _optimal_bbox(shape),
        "volume_mm3": float(shape.Volume),
        "area_mm2": float(shape.Area),
        "center_of_mass_mm": (
            [
                float(center.x),
                float(center.y),
                float(center.z),
            ]
            if center is not None
            else None
        ),
        "brep_sha256": hashlib.sha256(brep_bytes).hexdigest(),
    }


def _step_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="latin-1", errors="replace")
    file_name = re.search(
        r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'",
        text,
    )
    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(([^)]+)\)", text)
    products = re.findall(
        r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'",
        text,
    )
    return {
        "embedded_file_name": file_name.group(1) if file_name else None,
        "embedded_timestamp": file_name.group(2) if file_name else None,
        "file_schema_raw": schema.group(1) if schema else None,
        "ap_standard": ("AP214" if schema and "214" in schema.group(1) else "UNKNOWN"),
        "length_unit": ("millimetre" if "SI_UNIT(.MILLI.,.METRE.)" in text else "UNRESOLVED"),
        "angle_unit": ("radian" if "PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.)" in text else "UNRESOLVED"),
        "product_record_count": len(products),
        "product_names": sorted({name or description for name, description in products}),
        "next_assembly_usage_occurrence_count": len(re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(", text)),
    }


def _physical_shape_objects(document: Any) -> list[Any]:
    return [
        obj
        for obj in document.Objects
        if obj.TypeId in {"Part::Feature", "App::Part"}
        and getattr(obj, "Shape", None) is not None
        and not obj.Shape.isNull()
    ]


def _root_shape_objects(document: Any) -> list[Any]:
    physical = _physical_shape_objects(document)
    physical_names = {obj.Name for obj in physical}
    return [obj for obj in physical if not any(parent.Name in physical_names for parent in obj.InList)]


def main() -> None:
    source = Path(os.environ["ALOHA_BOTTLE_STEP"]).resolve(strict=True)
    output = Path(os.environ["ALOHA_BOTTLE_AUDIT_OUTPUT"]).resolve()

    Import.open(str(source))
    documents = list(App.listDocuments().values())
    shape_documents = [document for document in documents if _root_shape_objects(document)]
    if not shape_documents:
        raise RuntimeError("STEP import produced no document with a physical root shape")
    document_root_records = [
        [
            {
                "name": obj.Name,
                "label": obj.Label,
                "shape": _shape_record(obj.Shape),
            }
            for obj in _root_shape_objects(document)
        ]
        for document in shape_documents
    ]
    document_root_signatures = [[item["shape"]["brep_sha256"] for item in records] for records in document_root_records]
    if any(signature != document_root_signatures[0] for signature in document_root_signatures[1:]):
        raise RuntimeError(
            f"duplicate imported documents have different root B-Rep signatures: {document_root_signatures}"
        )
    document = shape_documents[0]
    object_records = []
    for obj in _physical_shape_objects(document):
        shape = obj.Shape
        object_records.append(
            {
                "name": str(obj.Name),
                "label": str(obj.Label),
                "type_id": str(obj.TypeId),
                "shape": _shape_record(shape),
            }
        )
    root_shapes = [obj.Shape for obj in _root_shape_objects(document)]
    aggregate = root_shapes[0] if len(root_shapes) == 1 else Part.makeCompound(root_shapes)
    report = {
        "schema_version": 1,
        "status": ("PASS" if all(item["shape"]["is_valid"] for item in object_records) else "PARTIAL"),
        "source": {
            "absolute_path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "step_header": _step_metadata(source),
        "runtime": {
            "freecad_version": ".".join(App.Version()[:3]),
            "freecad_build": App.Version()[3],
            "python_version": platform.python_version(),
            "opencascade_version": Part.OCC_VERSION,
        },
        "document": {
            "name": str(document.Name),
            "label": str(document.Label),
            "all_open_document_names": [str(item.Name) for item in documents],
            "identical_import_document_count": len(shape_documents),
            "duplicate_import_status": (
                "IDENTICAL_DUPLICATE_IMPORT_DOCUMENTS" if len(shape_documents) > 1 else "SINGLE_IMPORT_DOCUMENT"
            ),
            "root_shape_signatures": document_root_signatures[0],
            "object_count": len(document.Objects),
            "shape_object_count": len(object_records),
        },
        "objects": object_records,
        "aggregate_shape": _shape_record(aggregate),
        "evidence_boundary": {
            "cad_geometry_confirmed": True,
            "nominal_capacity_ml": 500,
            "nominal_capacity_source": "USER_DESIGNATION_FROM_FILENAME",
            "mass_confirmed": False,
            "material_confirmed": False,
            "wall_thickness_confirmed": False,
            "collision_accepted": False,
            "isaac_runtime_validated": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"shape_objects={len(object_records)}")
    print(f"output={output}")


main()

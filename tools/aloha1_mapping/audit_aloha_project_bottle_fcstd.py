"""Audit the project-authored Bottle500 FCStd and its exported STEP.

Environment:
  ALOHA_PROJECT_BOTTLE_FCSTD: absolute FCStd path
  ALOHA_PROJECT_BOTTLE_STEP: absolute exported STEP path
  ALOHA_PROJECT_BOTTLE_BUILD_SCRIPT: absolute generating script path
  ALOHA_PROJECT_BOTTLE_AUDIT_OUTPUT: absolute JSON output path
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any
import zipfile

import FreeCAD as App
import Import
import Part


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_record(shape: Any) -> dict[str, Any]:
    bounds = shape.BoundBox
    optimal_bounds = shape.optimalBoundingBox()
    brep = shape.exportBrepToString()
    brep_bytes = brep.encode("utf-8") if isinstance(brep, str) else bytes(brep)
    center = getattr(shape, "CenterOfMass", None)
    return {
        "shape_type": str(shape.ShapeType),
        "is_null": bool(shape.isNull()),
        "is_valid": bool(shape.isValid()),
        "solid_count": len(shape.Solids),
        "shell_count": len(shape.Shells),
        "face_count": len(shape.Faces),
        "edge_count": len(shape.Edges),
        "vertex_count": len(shape.Vertexes),
        "bbox_mm": {
            "x_min": float(bounds.XMin),
            "x_max": float(bounds.XMax),
            "x_length": float(bounds.XLength),
            "y_min": float(bounds.YMin),
            "y_max": float(bounds.YMax),
            "y_length": float(bounds.YLength),
            "z_min": float(bounds.ZMin),
            "z_max": float(bounds.ZMax),
            "z_length": float(bounds.ZLength),
        },
        "optimal_bbox_mm": {
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
        },
        "volume_mm3": float(shape.Volume),
        "area_mm2": float(shape.Area),
        "center_of_mass_mm": ([float(center.x), float(center.y), float(center.z)] if center is not None else None),
        "brep_sha256": hashlib.sha256(brep_bytes).hexdigest(),
    }


def _parameters(sheet: Any) -> list[dict[str, str]]:
    records = []
    for row in range(2, 100):
        try:
            name = str(sheet.get(f"A{row}") or "").strip()
            value = str(sheet.get(f"B{row}") or "").strip()
            unit = str(sheet.get(f"C{row}") or "").strip()
        except ValueError:
            break
        if not any((name, value, unit)):
            break
        records.append({"name": name, "value": value, "unit": unit})
    return records


def _shape_objects(document: Any) -> list[Any]:
    return [obj for obj in document.Objects if getattr(obj, "Shape", None) is not None and not obj.Shape.isNull()]


def _close_all_documents() -> None:
    for name in list(App.listDocuments()):
        App.closeDocument(name)


def main() -> None:
    fcstd = Path(os.environ["ALOHA_PROJECT_BOTTLE_FCSTD"]).resolve(strict=True)
    step = Path(os.environ["ALOHA_PROJECT_BOTTLE_STEP"]).resolve(strict=True)
    build_script = Path(os.environ["ALOHA_PROJECT_BOTTLE_BUILD_SCRIPT"]).resolve(strict=True)
    output = Path(os.environ["ALOHA_PROJECT_BOTTLE_AUDIT_OUTPUT"]).resolve()

    document = App.openDocument(str(fcstd))
    document_name = str(document.Name)
    document_label = str(document.Label)
    object_records = []
    for obj in document.Objects:
        record: dict[str, Any] = {
            "name": str(obj.Name),
            "label": str(obj.Label),
            "type_id": str(obj.TypeId),
        }
        if getattr(obj, "Shape", None) is not None and not obj.Shape.isNull():
            record["shape"] = _shape_record(obj.Shape)
        if obj.TypeId == "Spreadsheet::Sheet":
            record["parameters"] = _parameters(obj)
        object_records.append(record)

    master = document.getObject("BottleMaster")
    if master is None or getattr(master, "Shape", None) is None:
        raise RuntimeError("BottleMaster shape is missing from FCStd")
    master_shape = _shape_record(master.Shape)

    with zipfile.ZipFile(fcstd) as archive:
        fcstd_members = sorted(archive.namelist())

    _close_all_documents()
    Import.open(str(step))
    step_documents = list(App.listDocuments().values())
    step_shapes = [obj.Shape for step_document in step_documents for obj in _shape_objects(step_document)]
    if len(step_shapes) != 1:
        raise RuntimeError(f"expected one exported STEP shape, found {len(step_shapes)}")
    step_shape = _shape_record(step_shapes[0])

    tolerances = {
        "bbox_length_mm": 1.0e-6,
        "volume_mm3": 1.0e-4,
        "area_mm2": 1.0e-4,
    }
    bbox_keys = ("x_length", "y_length", "z_length")
    bbox_match = all(
        abs(master_shape["bbox_mm"][key] - step_shape["bbox_mm"][key]) <= tolerances["bbox_length_mm"]
        for key in bbox_keys
    )
    volume_match = abs(master_shape["volume_mm3"] - step_shape["volume_mm3"]) <= tolerances["volume_mm3"]
    area_match = abs(master_shape["area_mm2"] - step_shape["area_mm2"]) <= tolerances["area_mm2"]
    topology_match = all(
        master_shape[key] == step_shape[key]
        for key in (
            "solid_count",
            "shell_count",
            "face_count",
            "edge_count",
            "vertex_count",
        )
    )
    export_geometry_match = bbox_match and volume_match and area_match and topology_match

    report = {
        "schema_version": 1,
        "status": ("PASS" if master_shape["is_valid"] and export_geometry_match else "FAIL"),
        "runtime": {
            "freecad_version": ".".join(App.Version()[:3]),
            "freecad_build": App.Version()[3],
            "opencascade_version": Part.OCC_VERSION,
            "python_version": platform.python_version(),
        },
        "sources": {
            "fcstd": {
                "absolute_path": str(fcstd),
                "sha256": _sha256(fcstd),
                "size_bytes": fcstd.stat().st_size,
                "archive_members": fcstd_members,
            },
            "step": {
                "absolute_path": str(step),
                "sha256": _sha256(step),
                "size_bytes": step.stat().st_size,
            },
            "build_script": {
                "absolute_path": str(build_script),
                "sha256": _sha256(build_script),
                "size_bytes": build_script.stat().st_size,
            },
        },
        "fcstd": {
            "document_name": document_name,
            "document_label": document_label,
            "object_count": len(object_records),
            "objects": object_records,
            "bottle_master": master_shape,
        },
        "exported_step": {
            "shape": step_shape,
        },
        "export_comparison": {
            "status": ("PASS" if export_geometry_match else "FAIL"),
            "tolerances": tolerances,
            "bbox_length_match": bbox_match,
            "volume_match": volume_match,
            "area_match": area_match,
            "topology_match": topology_match,
            "brep_hash_match_required": False,
            "brep_hash_match": (master_shape["brep_sha256"] == step_shape["brep_sha256"]),
        },
        "evidence_boundary": {
            "parametric_source_confirmed": True,
            "fcstd_to_step_geometry_match_confirmed": export_geometry_match,
            "visual_tessellation_revalidated": False,
            "collision_revalidated": False,
            "mass_measured": False,
            "isaac_runtime_revalidated": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"objects={len(object_records)}")
    print(f"output={output}")


main()

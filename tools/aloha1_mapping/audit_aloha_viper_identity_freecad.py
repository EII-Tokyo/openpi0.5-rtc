"""Read-only FreeCAD audit for ALOHA Viper follower product identity.

Environment:
  ALOHA_VIPER_IDENTITY_STEP: absolute supplier STEP path
  ALOHA_VIPER_IDENTITY_OUTPUT: absolute JSON output path
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
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


def _matrix(placement: Any) -> list[list[float]]:
    values = [float(value) for value in placement.toMatrix().A]
    return [values[index : index + 4] for index in range(0, 16, 4)]


def _determinant(matrix: list[list[float]]) -> float:
    a, b, c = (matrix[index][:3] for index in range(3))
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _scale(matrix: list[list[float]]) -> list[float]:
    return [
        math.sqrt(sum(matrix[row][column] ** 2 for column in range(3)))
        for row in range(3)
    ]


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


def _shape_signature(shape: Any) -> dict[str, Any] | None:
    if shape is None or shape.isNull():
        return None
    brep = shape.exportBrepToString()
    brep_bytes = (
        brep.encode("utf-8") if isinstance(brep, str) else bytes(brep)
    )
    topology = {
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
    stable = {
        "shape_type": str(shape.ShapeType),
        "topology": topology,
        "bbox_mm": _bbox(shape),
        "volume_mm3": float(shape.Volume),
        "area_mm2": float(shape.Area),
        "brep_sha256": hashlib.sha256(brep_bytes).hexdigest(),
    }
    stable["geometry_signature_sha256"] = hashlib.sha256(
        json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    center = getattr(shape, "CenterOfMass", None)
    return {
        **stable,
        "is_valid": bool(shape.isValid()),
        "center_of_mass_mm": (
            [
                float(center.x),
                float(center.y),
                float(center.z),
            ]
            if center is not None
            else None
        ),
    }


def _record_object(obj: Any) -> dict[str, Any]:
    local = obj.Placement
    try:
        world = obj.getGlobalPlacement()
    except Exception:
        world = local
    local_matrix = _matrix(local)
    world_matrix = _matrix(world)
    linked = getattr(obj, "LinkedObject", None)
    return {
        "name": str(obj.Name),
        "label": str(obj.Label),
        "type_id": str(obj.TypeId),
        "parent_names": sorted(parent.Name for parent in obj.InList),
        "child_names": sorted(child.Name for child in obj.OutList),
        "linked_object": (
            str(linked.Name) if linked is not None else None
        ),
        "local_placement": {
            "matrix_mm": local_matrix,
            "determinant": _determinant(local_matrix),
            "scale": _scale(local_matrix),
            "mirror": _determinant(local_matrix) < 0.0,
        },
        "world_placement": {
            "matrix_mm": world_matrix,
            "determinant": _determinant(world_matrix),
            "scale": _scale(world_matrix),
            "mirror": _determinant(world_matrix) < 0.0,
        },
        "shape": _shape_signature(getattr(obj, "Shape", None)),
    }


def _step_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="latin-1", errors="replace")
    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(([^)]+)\)", text)
    products = re.findall(
        r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'",
        text,
    )
    return {
        "file_schema_raw": schema.group(1) if schema else None,
        "product_record_count": len(products),
        "product_names": sorted(
            {name or description for name, description in products}
        ),
        "next_assembly_usage_occurrence_count": len(
            re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(", text)
        ),
        "item_defined_transformation_count": len(
            re.findall(r"ITEM_DEFINED_TRANSFORMATION\s*\(", text)
        ),
        "context_dependent_shape_representation_count": len(
            re.findall(
                r"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\s*\(",
                text,
            )
        ),
    }


source_text = os.environ.get("ALOHA_VIPER_IDENTITY_STEP")
output_text = os.environ.get("ALOHA_VIPER_IDENTITY_OUTPUT")
if not source_text or not output_text:
    raise RuntimeError(
        "ALOHA_VIPER_IDENTITY_STEP and ALOHA_VIPER_IDENTITY_OUTPUT required"
    )

source = Path(source_text).resolve(strict=True)
output = Path(output_text).resolve()
source_hash_before = _sha256(source)
imported = Import.open(str(source))
document = (
    getattr(imported[0], "Document", None)
    if isinstance(imported, list | tuple) and imported
    else imported
)
if document is None:
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError("Import.open created no FreeCAD document")
    document = documents[-1]

objects = [_record_object(obj) for obj in document.Objects]
by_name = {item["name"]: item for item in objects}
roots = [
    item
    for item in objects
    if not item["parent_names"] and item["type_id"] == "App::Part"
]
links = [item for item in objects if item["type_id"] == "App::Link"]
required_labels = {
    "Simple VX Base v3",
    "Simple VX Shoulder v3",
    "Simple VX Upper Arm v3",
    "Simple VX Forearm v3",
    "Simple VX Wrist Twist v3",
    "Simple VX Wrist Tilt v3",
    "Aloha VX Gripper 2024-4-19 v4",
    "Aloha VX Fingers 2024-4-21 v2",
    "Aloha VX Fingers 2024-4-21 v001",
}
labels = {item["label"] for item in objects}
root_products = [
        {
            "name": root["name"],
            "label": root["label"],
            "child_names": root["child_names"],
            "complete_viper_product": required_labels.issubset(labels),
            "required_labels_present": sorted(required_labels & labels),
            "required_labels_missing": sorted(required_labels - labels),
            "shape_valid": bool(
                root["shape"] and root["shape"]["is_valid"]
            ),
            "geometry_signature": (
                root["shape"]["geometry_signature_sha256"]
                if root["shape"]
                else None
            ),
            "placement_matrix": root["world_placement"]["matrix_mm"],
            "placement_determinant": root["world_placement"][
                "determinant"
            ],
            "scale": root["world_placement"]["scale"],
            "mirror": root["world_placement"]["mirror"],
        }
    for root in roots
]

left_finger = by_name.get("Part__Feature007")
right_finger = by_name.get("Part__Feature008")
finger_container = by_name.get("Aloha_VX_Fingers_2024_4_21_v2")
gripper = by_name.get("Part__Feature006")
finger_pair_verified = bool(
    left_finger
    and right_finger
    and finger_container
    and set(finger_container["child_names"])
    >= {"Part__Feature007", "Part__Feature008"}
    and left_finger["shape"]
    and right_finger["shape"]
    and left_finger["shape"]["is_valid"]
    and right_finger["shape"]["is_valid"]
)
gripper_semantics_verified = bool(
    gripper
    and finger_container
    and roots
    and gripper["name"] in roots[0]["child_names"]
    and finger_container["name"] in roots[0]["child_names"]
)

report = {
    "schema_version": 1,
    "status": (
        "PASS"
        if roots
        and finger_pair_verified
        and gripper_semantics_verified
        and source_hash_before == _sha256(source)
        else "FAIL"
    ),
    "scope": "READ_ONLY_SUPPLIER_STEP_PRODUCT_IDENTITY_AUDIT",
    "source": {
        "absolute_path": str(source),
        "sha256_before": source_hash_before,
        "sha256_after": _sha256(source),
        "size_bytes": source.stat().st_size,
        "read_only": True,
    },
    "toolchain": {
        "freecad_version": list(App.Version()),
        "opencascade_version": str(Part.OCC_VERSION),
        "freecad_executable_contract": (
            "project-local local_tools/freecad-tessellation/freecadcmd"
        ),
        "linear_deflection_mm": 0.20,
        "angular_deflection_deg": 20.0,
        "tessellation_note": (
            "identity audit uses exact B-Rep; pinned tessellation parameters "
            "are recorded but no tessellation is performed"
        ),
    },
    "step_metadata": _step_metadata(source),
    "document": {
        "name": str(document.Name),
        "label": str(document.Label),
        "object_count": len(objects),
        "root_object_names": [item["name"] for item in roots],
        "app_link_count": len(links),
        "shape_object_count": sum(
            1 for item in objects if item["shape"] is not None
        ),
    },
    "brep_validity": {
        "status": (
            "PASS"
            if all(
                item["shape"] is None or item["shape"]["is_valid"]
                for item in objects
            )
            else "PARTIAL"
        ),
        "invalid_objects": [
            {
                "name": item["name"],
                "label": item["label"],
                "type_id": item["type_id"],
                "shape_type": item["shape"]["shape_type"],
                "geometry_signature_sha256": item["shape"][
                    "geometry_signature_sha256"
                ],
            }
            for item in objects
            if item["shape"] is not None and not item["shape"]["is_valid"]
        ],
        "identity_boundary": (
            "B-Rep validity is recorded separately from product identity; "
            "invalid source solids are not silently healed."
        ),
    },
    "root_products": root_products,
    "product_instances": [
        {
            "name": item["name"],
            "label": item["label"],
            "source_product": item["linked_object"],
            "geometry_signature": (
                item["shape"]["geometry_signature_sha256"]
                if item["shape"]
                else None
            ),
            "placement_matrix": item["world_placement"]["matrix_mm"],
            "placement_determinant": item["world_placement"][
                "determinant"
            ],
            "scale": item["world_placement"]["scale"],
            "mirror": item["world_placement"]["mirror"],
        }
        for item in links
    ],
    "objects": objects,
    "handed_finger_pair_verified": finger_pair_verified,
    "gripper_assembly_semantics_verified": gripper_semantics_verified,
    "gripper_semantics": {
        "gripper_shell": gripper,
        "finger_container": finger_container,
        "left_finger": left_finger,
        "right_finger": right_finger,
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    "ALOHA_VIPER_IDENTITY_AUDIT",
    report["status"],
    "roots=",
    len(root_products),
    "instances=",
    len(links),
    "objects=",
    len(objects),
    "output=",
    output,
)
App.closeDocument(document.Name)

"""Audit the complete supplier gripper and project Bottle500 clearance.

Required environment variables:
  ALOHA_VIPER_STEP
  ALOHA_PROJECT_BOTTLE_FCSTD
  ALOHA_RUNTIME_GRIPPER_BAR_STL
  ALOHA_GRASP_CLEARANCE_OUTPUT

Run only with the project-pinned FreeCAD 1.1.1 / OCCT 7.8.1 wrapper.
The script reads immutable geometry and writes one raw semantic report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any

import FreeCAD as App
import Import
import Part

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.aloha1_mapping.supplier_cad_grasp_clearance import build_right_handed_grasp_frame  # noqa: E402
from tools.aloha1_mapping.supplier_cad_grasp_clearance import select_chebyshev_grasp_station  # noqa: E402

EXPECTED = {
    "supplier_step": (
        "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
    ),
    "project_bottle_fcstd": (
        "3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a"
    ),
    "runtime_gripper_bar_stl": (
        "a4de62c9a2ed2c78433010e4c05530a1254b1774a7651967f406120c9bf8973e"
    ),
}
GRIPPER_REFERENCE_FROM_CAD = {
    "x_from_global_y_offset_m": -0.43029999973392,
    "y_from_global_x_offset_m": -0.0000998902576627736,
    "z_from_global_z_offset_m": -0.42680133373174,
}
FINGER_OBJECTS = {
    "left": ("Part__Feature007", "Aloha VX Fingers 2024-4-21 v2", 117),
    "right": ("Part__Feature008", "Aloha VX Fingers 2024-4-21 v001", 128),
}
SHELL_OBJECT = ("Part__Feature006", "Aloha VX Gripper 2024-4-19 v4")
BOTTLE_OBJECT = "BottleMaster"
BOTTLE_AXIAL_STATION_MM = 69.0
CLOSED_FINGER_Q_M = 0.021
LINEAR_DEFLECTION_MM = 0.2
ANGULAR_DEFLECTION_DEG = 20.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_import_document(path: Path) -> Any:
    imported = Import.open(str(path))
    document = (
        getattr(imported[0], "Document", None)
        if isinstance(imported, list | tuple) and imported
        else imported
    )
    if document is not None:
        return document
    documents = list(App.listDocuments().values())
    if not documents:
        raise RuntimeError(f"FreeCAD did not open {path}")
    return documents[-1]


def _point_cad_mm_to_reference_m(point: Any) -> list[float]:
    return [
        -float(point.y) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["x_from_global_y_offset_m"],
        float(point.x) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["y_from_global_x_offset_m"],
        float(point.z) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["z_from_global_z_offset_m"],
    ]


def _normal_cad_to_reference(normal: Any) -> list[float]:
    values = [-float(normal.y), float(normal.x), float(normal.z)]
    length = math.sqrt(sum(value * value for value in values))
    return [value / length for value in values]


def _face_normal(face: Any) -> Any:
    center = face.CenterOfMass
    try:
        u_value, v_value = face.Surface.parameter(center)
    except Exception:
        u_min, u_max, v_min, v_max = face.ParameterRange
        u_value = (float(u_min) + float(u_max)) / 2.0
        v_value = (float(v_min) + float(v_max)) / 2.0
    return face.normalAt(float(u_value), float(v_value))


def _face_record(face: Any, index: int) -> dict[str, Any]:
    bounds = face.BoundBox
    center = face.CenterOfMass
    normal = _face_normal(face)
    x_values = [
        -float(bounds.YMin) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["x_from_global_y_offset_m"],
        -float(bounds.YMax) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["x_from_global_y_offset_m"],
    ]
    return {
        "face_index_1_based": index,
        "surface_type": type(face.Surface).__name__,
        "area_mm2": float(face.Area),
        "center_cad_global_mm": [
            float(center.x),
            float(center.y),
            float(center.z),
        ],
        "center_gripper_reference_m": _point_cad_mm_to_reference_m(center),
        "normal_cad_global": [
            float(normal.x),
            float(normal.y),
            float(normal.z),
        ],
        "normal_gripper_reference": _normal_cad_to_reference(normal),
        "approach_interval_gripper_reference_m": [
            min(x_values),
            max(x_values),
        ],
    }


def _binary_stl_vertices_mm(path: Path) -> list[tuple[float, float, float]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise RuntimeError("runtime gripper bar STL is truncated")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise RuntimeError("runtime gripper bar STL is not expected binary STL")
    vertices = []
    for triangle_index in range(triangle_count):
        vertex_offset = 84 + triangle_index * 50 + 12
        vertices.extend(
            [
                struct.unpack_from(
                    "<fff",
                    data,
                    vertex_offset + vertex_index * 12,
                )
                for vertex_index in range(3)
            ]
        )
    return vertices


def _runtime_bar_record(path: Path) -> dict[str, Any]:
    vertices_mm = _binary_stl_vertices_mm(path)
    # URDF: gripper_link -> ee_arm_link +42.825 mm, then the collision
    # mesh has translation -63 mm and Rz(+90 deg).
    points_reference_m = [
        (
            (-vertex[1] - 63.0 + 42.825) * 0.001,
            vertex[0] * 0.001,
            vertex[2] * 0.001,
        )
        for vertex in vertices_mm
    ]
    minimum = [
        min(point[axis] for point in points_reference_m)
        for axis in range(3)
    ]
    maximum = [
        max(point[axis] for point in points_reference_m)
        for axis in range(3)
    ]
    return {
        "source_type": "URDF_COLLISION_STL_CONSERVATIVE_AABB",
        "triangle_count": len(vertices_mm) // 3,
        "gripper_reference_aabb_m": {
            "min": minimum,
            "max": maximum,
        },
        "maximum_approach_x_m": maximum[0],
        "urdf_transform": {
            "gripper_link_to_ee_arm_translation_m": [0.042825, 0.0, 0.0],
            "collision_origin_translation_m": [-0.063, 0.0, 0.0],
            "collision_origin_rpy_rad": [0.0, 0.0, math.pi / 2.0],
            "mesh_scale": [0.001, 0.001, 0.001],
        },
    }


def _bottle_section_record(shape: Any) -> dict[str, Any]:
    plane = Part.makePlane(
        200.0,
        200.0,
        App.Vector(-100.0, -100.0, BOTTLE_AXIAL_STATION_MM),
        App.Vector(0.0, 0.0, 1.0),
    )
    section = shape.section(plane)
    if section.isNull() or not section.Edges:
        raise RuntimeError("Bottle500 B-Rep section is empty")
    bounds = section.BoundBox
    radius = max(
        abs(float(bounds.XMin)),
        abs(float(bounds.XMax)),
        abs(float(bounds.YMin)),
        abs(float(bounds.YMax)),
    )
    return {
        "evidence": "PROJECT_BOTTLE_BREP_SECTION_READBACK",
        "axial_station_mm": BOTTLE_AXIAL_STATION_MM,
        "cad_long_axis": "+Z",
        "section_edge_count": len(section.Edges),
        "section_bbox_mm": {
            "x_min": float(bounds.XMin),
            "x_max": float(bounds.XMax),
            "y_min": float(bounds.YMin),
            "y_max": float(bounds.YMax),
        },
        "outer_radius_mm": radius,
        "outer_diameter_mm": radius * 2.0,
    }


def _bottle_at_reference_center(
    bottle_shape: Any,
    *,
    center_x_m: float,
) -> Any:
    result = bottle_shape.copy()
    result.Placement = App.Placement(
        App.Vector(
            (
                -GRIPPER_REFERENCE_FROM_CAD[
                    "y_from_global_x_offset_m"
                ]
                * 1000.0
            ),
            -(
                center_x_m
                - GRIPPER_REFERENCE_FROM_CAD[
                    "x_from_global_y_offset_m"
                ]
            )
            * 1000.0,
            (
                -GRIPPER_REFERENCE_FROM_CAD[
                    "z_from_global_z_offset_m"
                ]
                * 1000.0
                - BOTTLE_AXIAL_STATION_MM
            ),
        ),
        App.Rotation(),
    )
    return result


def _shell_clearance_record(
    shell_shape: Any,
    bottle_shape: Any,
    *,
    center_x_m: float,
) -> dict[str, Any]:
    placed_bottle = _bottle_at_reference_center(
        bottle_shape,
        center_x_m=center_x_m,
    )
    common = shell_shape.common(placed_bottle)
    distance = shell_shape.distToShape(placed_bottle)
    return {
        "bottle_center_x_m": center_x_m,
        "common_volume_mm3": float(common.Volume),
        "minimum_distance_mm": float(distance[0]),
        "intersects": float(common.Volume) > 1.0e-9,
    }


def _plane_contact_at_bottle_center(
    *,
    face_record: dict[str, Any],
    bottle_center_x_m: float,
    radius_m: float,
) -> tuple[list[float], float]:
    normal = face_record["normal_gripper_reference"]
    center = [
        bottle_center_x_m,
        0.0,
        0.0,
    ]
    contact = [
        center[axis] - radius_m * normal[axis]
        for axis in range(3)
    ]
    face_center = face_record["center_gripper_reference_m"]
    side = (
        "left"
        if float(normal[1]) < 0.0
        else "right"
    )
    translation_sign = 1.0 if side == "left" else -1.0
    numerator = sum(
        normal[axis] * (contact[axis] - face_center[axis])
        for axis in range(3)
    )
    delta_m = numerator / (normal[1] * translation_sign)
    return contact, delta_m


def _semantic_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    supplier_step = Path(os.environ["ALOHA_VIPER_STEP"]).resolve(strict=True)
    bottle_fcstd = Path(
        os.environ["ALOHA_PROJECT_BOTTLE_FCSTD"]
    ).resolve(strict=True)
    runtime_bar_stl = Path(
        os.environ["ALOHA_RUNTIME_GRIPPER_BAR_STL"]
    ).resolve(strict=True)
    output = Path(os.environ["ALOHA_GRASP_CLEARANCE_OUTPUT"]).resolve()
    inputs = {
        "supplier_step": supplier_step,
        "project_bottle_fcstd": bottle_fcstd,
        "runtime_gripper_bar_stl": runtime_bar_stl,
    }
    for name, path in inputs.items():
        actual = _sha256(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(
                f"{name} SHA-256 mismatch: {actual} != {EXPECTED[name]}"
            )

    supplier_document = _open_import_document(supplier_step)
    shell_name, shell_label = SHELL_OBJECT
    shell = supplier_document.getObject(shell_name)
    if shell is None or str(shell.Label) != shell_label:
        raise RuntimeError("supplier gripper shell identity mismatch")
    if shell.Shape.isNull() or not shell.Shape.isValid():
        raise RuntimeError("supplier gripper shell B-Rep is invalid")

    fingers: dict[str, dict[str, Any]] = {}
    for side, (object_name, expected_label, face_index) in (
        FINGER_OBJECTS.items()
    ):
        obj = supplier_document.getObject(object_name)
        if obj is None or str(obj.Label) != expected_label:
            raise RuntimeError(f"supplier {side} finger identity mismatch")
        if obj.Shape.isNull() or not obj.Shape.isValid():
            raise RuntimeError(f"supplier {side} finger B-Rep is invalid")
        face = obj.Shape.Faces[face_index - 1]
        record = _face_record(face, face_index)
        record.update(
            {
                "object_name": object_name,
                "label": expected_label,
                "brep_valid": True,
            }
        )
        fingers[side] = record

    bottle_document = App.openDocument(str(bottle_fcstd))
    bottle = bottle_document.getObject(BOTTLE_OBJECT)
    if bottle is None or bottle.Shape.isNull() or not bottle.Shape.isValid():
        raise RuntimeError("project Bottle500 B-Rep is invalid")
    bottle_section = _bottle_section_record(bottle.Shape)
    radius_m = float(bottle_section["outer_radius_mm"]) * 0.001

    left_interval = fingers["left"][
        "approach_interval_gripper_reference_m"
    ]
    right_interval = fingers["right"][
        "approach_interval_gripper_reference_m"
    ]
    pad_interval = (
        max(float(left_interval[0]), float(right_interval[0])),
        min(float(left_interval[1]), float(right_interval[1])),
    )
    normal_x = (
        float(fingers["left"]["normal_gripper_reference"][0])
        + float(fingers["right"]["normal_gripper_reference"][0])
    ) / 2.0

    shell_bounds = shell.Shape.BoundBox
    shell_max_x = (
        -float(shell_bounds.YMin) * 0.001
        + GRIPPER_REFERENCE_FROM_CAD["x_from_global_y_offset_m"]
    )
    runtime_bar = _runtime_bar_record(runtime_bar_stl)
    selection = select_chebyshev_grasp_station(
        pad_interval_m=pad_interval,
        forbidden_max_x_m={
            "supplier_gripper_shell": shell_max_x,
            "runtime_urdf_gripper_bar": float(
                runtime_bar["maximum_approach_x_m"]
            ),
        },
        bottle_radius_m=radius_m,
        pad_inward_normal_x=normal_x,
        rejected_station_m=0.11127188479610935,
    )
    if selection["status"] != "PASS":
        raise RuntimeError("complete gripper has no legal Bottle500 station")

    selected_center_x = float(selection["selected_station_m"])
    left_contact, left_delta = _plane_contact_at_bottle_center(
        face_record=fingers["left"],
        bottle_center_x_m=selected_center_x,
        radius_m=radius_m,
    )
    right_contact, right_delta = _plane_contact_at_bottle_center(
        face_record=fingers["right"],
        bottle_center_x_m=selected_center_x,
        radius_m=radius_m,
    )
    if abs(left_delta - right_delta) > 1.0e-8:
        raise RuntimeError("left/right contact solutions are asymmetric")
    q_contact = CLOSED_FINGER_Q_M + (left_delta + right_delta) / 2.0
    if not CLOSED_FINGER_Q_M <= q_contact <= 0.057:
        raise RuntimeError(f"contact q is outside legal range: {q_contact}")
    grasp_frame = build_right_handed_grasp_frame(
        left_contact_reference_m=left_contact,
        right_contact_reference_m=right_contact,
        approach_axis_reference=(1.0, 0.0, 0.0),
        bottle_axis_reference=(0.0, 0.0, 1.0),
    )
    grasp_origin_x = float(grasp_frame["origin_reference_m"][0])
    bottle_center_from_grasp = [
        selected_center_x - grasp_origin_x,
        0.0,
        0.0,
    ]

    source_records = {
        name: {
            "absolute_path": str(path),
            "sha256": EXPECTED[name],
            "size_bytes": path.stat().st_size,
            "read_only": True,
        }
        for name, path in inputs.items()
    }
    core: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "classification": (
            "COMPLETE_SUPPLIER_GRIPPER_PROJECT_BOTTLE_CLEARANCE"
        ),
        "task8": "NOT_RUN",
        "toolchain": {
            "freecad": ".".join(App.Version()[:3]),
            "freecad_build": App.Version()[3],
            "opencascade": str(Part.OCC_VERSION),
            "linear_deflection_mm": LINEAR_DEFLECTION_MM,
            "angular_deflection_deg": ANGULAR_DEFLECTION_DEG,
            "relative": False,
        },
        "sources": {
            "supplier_viper_step": source_records["supplier_step"],
            "project_bottle_fcstd": source_records[
                "project_bottle_fcstd"
            ],
            "runtime_gripper_bar_stl": source_records[
                "runtime_gripper_bar_stl"
            ],
        },
        "coordinate_contract": {
            "gripper_reference": "follower_left_gripper_link",
            "approach_axis": "+X",
            "closing_axis": "+Y toward left_finger",
            "bottle_axis": "+Z",
            "cad_to_reference": GRIPPER_REFERENCE_FROM_CAD,
            "mirror_used": False,
        },
        "fingers": fingers,
        "pad_interval_gripper_reference_m": list(pad_interval),
        "bottle_section": bottle_section,
        "forbidden_envelopes": {
            "supplier_gripper_shell": {
                "source_type": "SUPPLIER_STEP_BREP",
                "object_name": shell_name,
                "label": shell_label,
                "brep_valid": True,
                "maximum_approach_x_m": shell_max_x,
                "old_station_clearance": _shell_clearance_record(
                    shell.Shape,
                    bottle.Shape,
                    center_x_m=0.11127188479610935,
                ),
                "selected_station_clearance": _shell_clearance_record(
                    shell.Shape,
                    bottle.Shape,
                    center_x_m=selected_center_x,
                ),
            },
            "runtime_urdf_gripper_bar": runtime_bar,
        },
        "station_selection": selection,
        "contact_solution": {
            "left_contact_reference_m": left_contact,
            "right_contact_reference_m": right_contact,
            "left_translation_from_closed_m": left_delta,
            "right_translation_from_closed_m": right_delta,
            "left_finger_q_m": q_contact,
            "right_finger_q_m": -q_contact,
            "legal_range_m": [0.021, 0.057],
            "symmetric": abs(left_delta - right_delta) <= 1.0e-8,
        },
        "grasp_frame": {
            **grasp_frame,
            "official_ee_helper": (
                "follower_left_ee_gripper_link_NOT_GRASP_CENTER"
            ),
            "bottle_axis_center_from_grasp_m": bottle_center_from_grasp,
            "object_placement_must_apply_normal_offset": True,
        },
        "rejected_run13": {
            "report": str(
                (
                    ROOT
                    / "reports/aloha1_mapping/"
                    "aloha1_grasp_frame_run13_rejection.json"
                ).resolve()
            ),
            "classification": (
                "REJECTED_WHOLE_PAD_FACE_CENTROID_NOT_EFFECTIVE_GRASP_CENTER"
            ),
        },
    }
    core["semantic_signature"] = _semantic_signature(core)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(core, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": core["status"],
                "output": str(output),
                "semantic_signature": core["semantic_signature"],
                "selected_bottle_center_x_m": selected_center_x,
                "selected_pad_contact_x_m": grasp_origin_x,
                "contact_q_m": q_contact,
            },
            sort_keys=True,
        )
    )


main()

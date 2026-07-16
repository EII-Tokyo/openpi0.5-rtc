"""Create Iteration 001 with measured table/rack footprint.

Iteration 000 remains the read-only Isaac/MJCF reference. This script creates a
separate review FCStd where the scene table and rack/camera support structure
are adjusted to the user's measured tabletop footprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import FreeCAD
from FreeCAD import Base


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER0_DIR = WORKDIR / "iterations" / "iter_000_reference"
ITER1_DIR = WORKDIR / "iterations" / "iter_001_measured_table_rack"
INPUT_FCSTD = ITER0_DIR / "iter_000_reference.FCStd"
OUTPUT_FCSTD = ITER1_DIR / "iter_001_measured_table_rack.FCStd"

ORIGINAL_TABLE_LENGTH_MM = 1210.0
ORIGINAL_TABLE_WIDTH_MM = 760.0
MEASURED_TABLE_LENGTH_MM = 1220.0
MEASURED_TABLE_WIDTH_MM = 625.0
DESKTOP_THICKNESS_MM = 18.0
TABLE_TOP_Z_MM = 0.0

SCALE_X = MEASURED_TABLE_LENGTH_MM / ORIGINAL_TABLE_LENGTH_MM
SCALE_Y = MEASURED_TABLE_WIDTH_MM / ORIGINAL_TABLE_WIDTH_MM


def _scale_about_origin_matrix(sx: float, sy: float, sz: float = 1.0) -> Base.Matrix:
    matrix = Base.Matrix()
    matrix.A11 = sx
    matrix.A22 = sy
    matrix.A33 = sz
    return matrix


def _document_bbox(doc):
    boxes = []
    for obj in doc.Objects:
        box = None
        if hasattr(obj, "Mesh"):
            box = obj.Mesh.BoundBox
        elif hasattr(obj, "Shape"):
            box = obj.Shape.BoundBox
        if box is not None and box.isValid():
            boxes.append(box)
    if not boxes:
        raise RuntimeError("No valid object bounding boxes in document")
    out = boxes[0]
    for box in boxes[1:]:
        out.add(box)
    return out


def _is_table_or_rack_scene_mesh(name: str) -> bool:
    return (
        name.startswith("REF_SCENE_table_")
        or name.startswith("REF_SCENE_frame_")
        or name.startswith("REF_SCENE_camera_")
    )


def _set_desktop_plane(obj) -> None:
    obj.Length = MEASURED_TABLE_LENGTH_MM
    obj.Width = MEASURED_TABLE_WIDTH_MM
    obj.Height = DESKTOP_THICKNESS_MM
    obj.Placement.Base = Base.Vector(
        -MEASURED_TABLE_LENGTH_MM / 2.0,
        -MEASURED_TABLE_WIDTH_MM / 2.0,
        TABLE_TOP_Z_MM - DESKTOP_THICKNESS_MM,
    )
    try:
        obj.SourceAsset = "measured correction: tabletop footprint 1220 x 625 mm"
    except Exception:
        pass


def main() -> None:
    ITER1_DIR.mkdir(parents=True, exist_ok=True)
    doc = FreeCAD.openDocument(str(INPUT_FCSTD))
    doc.Label = "iter_001_measured_table_rack"

    scale_matrix = _scale_about_origin_matrix(SCALE_X, SCALE_Y, 1.0)
    changed_meshes: list[str] = []
    skipped_robot: list[str] = []

    for obj in doc.Objects:
        if obj.Name.startswith("REF_ALOHA_"):
            skipped_robot.append(obj.Name)
            continue
        if obj.Name == "REF_TABLE_DESKTOP_PLANE":
            _set_desktop_plane(obj)
            continue
        if _is_table_or_rack_scene_mesh(obj.Name) and hasattr(obj, "Mesh"):
            scaled_mesh = obj.Mesh.copy()
            scaled_mesh.transform(scale_matrix)
            obj.Mesh = scaled_mesh
            changed_meshes.append(obj.Name)
            try:
                obj.addProperty("App::PropertyString", "MeasuredAdjustment", "Reference")
            except Exception:
                pass
            try:
                obj.MeasuredAdjustment = (
                    f"Scaled about world origin: sx={SCALE_X:.9f}, sy={SCALE_Y:.9f}, sz=1.0; "
                    "source measurement tabletop footprint 1220 x 625 mm"
                )
            except Exception:
                pass

    doc.recompute()
    doc.saveAs(str(OUTPUT_FCSTD))

    bbox = _document_bbox(doc)
    metadata = {
        "iteration": "iter_001_measured_table_rack",
        "units": "mm",
        "source_iteration": str((ITER0_DIR / "iter_000_reference.FCStd").relative_to(ROOT)),
        "output_freecad_file": str(OUTPUT_FCSTD.relative_to(ROOT)),
        "measured_table": {
            "length_mm": MEASURED_TABLE_LENGTH_MM,
            "width_mm": MEASURED_TABLE_WIDTH_MM,
            "source": "user physical measurement",
            "status": "measured",
        },
        "original_reference_table": {
            "length_mm": ORIGINAL_TABLE_LENGTH_MM,
            "width_mm": ORIGINAL_TABLE_WIDTH_MM,
            "source": "external/mujoco_menagerie/aloha/scene.xml",
            "status": "from_cad",
        },
        "transform": {
            "center": "world origin / tabletop center",
            "scale_x": SCALE_X,
            "scale_y": SCALE_Y,
            "scale_z": 1.0,
            "affected_prefixes": ["REF_SCENE_table_", "REF_SCENE_frame_", "REF_SCENE_camera_"],
            "not_scaled_prefixes": ["REF_ALOHA_", "REF_AXIS_"],
        },
        "changed_mesh_count": len(changed_meshes),
        "changed_meshes": changed_meshes,
        "skipped_robot_mesh_count": len(skipped_robot),
        "document_bbox_mm": {
            "xmin": bbox.XMin,
            "xmax": bbox.XMax,
            "ymin": bbox.YMin,
            "ymax": bbox.YMax,
            "zmin": bbox.ZMin,
            "zmax": bbox.ZMax,
            "x_length": bbox.XLength,
            "y_length": bbox.YLength,
            "z_length": bbox.ZLength,
        },
    }
    (ITER1_DIR / "bbox_and_dimensions.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (ITER1_DIR / "changes.md").write_text(
        "\n".join(
            [
                "# Iteration 001 Measured Table/Rack",
                "",
                "## What Changed",
                "",
                "- Created a separate FreeCAD review model from `iter_000_reference.FCStd`.",
                "- Applied the user's measured tabletop footprint: `1220 mm x 625 mm`.",
                f"- Scaled table/rack/camera-support scene meshes about the tabletop center: X scale `{SCALE_X:.6f}`, Y scale `{SCALE_Y:.6f}`, Z scale `1.0`.",
                "- Set `REF_TABLE_DESKTOP_PLANE` directly to `Length=1220 mm`, `Width=625 mm`, `Height=18 mm`.",
                "",
                "## What Did Not Change",
                "",
                "- The original Isaac/MJCF/USD assets were not modified.",
                "- `iter_000_reference.FCStd` was not overwritten.",
                "- `REF_ALOHA_*` robot meshes were not scaled.",
                "- World axes were not scaled.",
                "",
                "## Assumption",
                "",
                "The measured tabletop center remains at the existing world origin. The rack and scene camera support structure follows the tabletop X/Y footprint, while the robot base geometry remains unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUTPUT_FCSTD)
    print(f"changed_mesh_count={len(changed_meshes)} scale_x={SCALE_X:.9f} scale_y={SCALE_Y:.9f}")


main()

"""Create Iteration 002 with measured ALOHA base Y offset.

This starts from Iteration 001, keeps the measured table/rack footprint, and
translates the ALOHA robot meshes in Y to match the user's physical margins.
"""

from __future__ import annotations

import json
from pathlib import Path

import FreeCAD
from FreeCAD import Base


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
WORKDIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental"
ITER1_DIR = WORKDIR / "iterations" / "iter_001_measured_table_rack"
ITER2_DIR = WORKDIR / "iterations" / "iter_002_measured_aloha_y_offset"
INPUT_FCSTD = ITER1_DIR / "iter_001_measured_table_rack.FCStd"
OUTPUT_FCSTD = ITER2_DIR / "iter_002_measured_aloha_y_offset.FCStd"

TABLE_WIDTH_MM = 625.0
MEASURED_TOP_MARGIN_MM = 180.0
MEASURED_BOTTOM_MARGIN_MM = 235.0


def _bbox_for_prefix(doc, prefix: str):
    boxes = []
    for obj in doc.Objects:
        if not obj.Name.startswith(prefix):
            continue
        box = None
        if hasattr(obj, "Mesh"):
            box = obj.Mesh.BoundBox
        elif hasattr(obj, "Shape"):
            box = obj.Shape.BoundBox
        if box is not None and box.isValid():
            boxes.append(box)
    if not boxes:
        raise RuntimeError(f"No objects with prefix {prefix}")
    out = boxes[0]
    for box in boxes[1:]:
        out.add(box)
    return out


def _translate_mesh(obj, dx: float, dy: float, dz: float) -> None:
    matrix = Base.Matrix()
    matrix.A14 = dx
    matrix.A24 = dy
    matrix.A34 = dz
    moved_mesh = obj.Mesh.copy()
    moved_mesh.transform(matrix)
    obj.Mesh = moved_mesh


def main() -> None:
    ITER2_DIR.mkdir(parents=True, exist_ok=True)
    doc = FreeCAD.openDocument(str(INPUT_FCSTD))
    doc.Label = "iter_002_measured_aloha_y_offset"

    right_base = doc.getObject("REF_ALOHA_right_base_link_vx300s_1_base_0")
    if right_base is None:
        raise RuntimeError("Missing right base object")
    before = right_base.Mesh.BoundBox

    table_y_min = -TABLE_WIDTH_MM / 2.0
    table_y_max = TABLE_WIDTH_MM / 2.0
    target_base_y_max = table_y_max - MEASURED_TOP_MARGIN_MM
    target_base_y_min = table_y_min + MEASURED_BOTTOM_MARGIN_MM
    target_center_y = (target_base_y_min + target_base_y_max) / 2.0
    current_center_y = (before.YMin + before.YMax) / 2.0
    dy = target_center_y - current_center_y

    moved = []
    for obj in doc.Objects:
        if obj.Name.startswith("REF_ALOHA_") and hasattr(obj, "Mesh"):
            _translate_mesh(obj, 0.0, dy, 0.0)
            moved.append(obj.Name)
            try:
                obj.addProperty("App::PropertyString", "MeasuredAdjustment", "Reference")
            except Exception:
                pass
            try:
                obj.MeasuredAdjustment = f"Translated ALOHA robot mesh by dy={dy:.3f} mm from physical base margin measurement."
            except Exception:
                pass

    doc.recompute()
    doc.saveAs(str(OUTPUT_FCSTD))

    after = doc.getObject("REF_ALOHA_right_base_link_vx300s_1_base_0").Mesh.BoundBox
    actual_top_margin = table_y_max - after.YMax
    actual_bottom_margin = after.YMin - table_y_min
    metadata = {
        "iteration": "iter_002_measured_aloha_y_offset",
        "units": "mm",
        "source_iteration": str(INPUT_FCSTD.relative_to(ROOT)),
        "output_freecad_file": str(OUTPUT_FCSTD.relative_to(ROOT)),
        "measured_margins": {
            "top_margin_mm": MEASURED_TOP_MARGIN_MM,
            "bottom_margin_mm": MEASURED_BOTTOM_MARGIN_MM,
            "source": "user physical measurement from annotated screenshot on 2026-07-15",
            "status": "measured",
        },
        "table_y_range_mm": [table_y_min, table_y_max],
        "right_base_bbox_before_mm": {
            "y_min": before.YMin,
            "y_max": before.YMax,
            "y_length": before.YLength,
            "center_y": current_center_y,
        },
        "right_base_bbox_after_mm": {
            "y_min": after.YMin,
            "y_max": after.YMax,
            "y_length": after.YLength,
            "center_y": (after.YMin + after.YMax) / 2.0,
        },
        "translation": {
            "dx_mm": 0.0,
            "dy_mm": dy,
            "dz_mm": 0.0,
            "affected_prefix": "REF_ALOHA_",
        },
        "actual_margins_after_translation": {
            "top_margin_mm": actual_top_margin,
            "bottom_margin_mm": actual_bottom_margin,
            "note": "The CAD robot base mesh is 204 mm wide in Y, while the measured margins imply 210 mm. The model keeps the robot mesh unscaled, so both margins are off by about 3 mm rather than scaling the robot.",
        },
        "moved_robot_mesh_count": len(moved),
    }
    (ITER2_DIR / "bbox_and_dimensions.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (ITER2_DIR / "changes.md").write_text(
        "\n".join(
            [
                "# Iteration 002 Measured ALOHA Y Offset",
                "",
                "## What Changed",
                "",
                f"- Started from `{INPUT_FCSTD.relative_to(ROOT)}`.",
                f"- Translated all `REF_ALOHA_*` robot meshes by `dy={dy:.3f} mm`.",
                "- Kept the measured table/rack footprint from Iteration 001.",
                "",
                "## Measurement Used",
                "",
                "- Table width: `625 mm`.",
                "- Physical top margin to robot base: `180 mm`.",
                "- Physical bottom margin to robot base: `235 mm`.",
                "",
                "## Important Note",
                "",
                "The CAD robot base mesh is `204 mm` wide in Y, while the two measured margins imply a `210 mm` base footprint. This iteration does not scale the robot; it aligns the base center, giving approximately `183 mm` and `238 mm` margins.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUTPUT_FCSTD)
    print(f"dy_mm={dy:.3f} moved_robot_mesh_count={len(moved)}")
    print(f"actual_top_margin_mm={actual_top_margin:.3f} actual_bottom_margin_mm={actual_bottom_margin:.3f}")


main()

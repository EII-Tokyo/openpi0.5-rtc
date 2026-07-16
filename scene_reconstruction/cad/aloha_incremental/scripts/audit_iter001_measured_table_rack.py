"""Audit Iteration 001 measured table/rack FreeCAD output."""

from __future__ import annotations

import json
from pathlib import Path

import FreeCAD


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
ITER_DIR = ROOT / "scene_reconstruction" / "cad" / "aloha_incremental" / "iterations" / "iter_001_measured_table_rack"
FCSTD = ITER_DIR / "iter_001_measured_table_rack.FCStd"


def _get_object(doc, name: str):
    obj = doc.getObject(name)
    if obj is None:
        raise RuntimeError(f"Missing object: {name}")
    return obj


def main() -> int:
    doc = FreeCAD.openDocument(str(FCSTD))
    plane = _get_object(doc, "REF_TABLE_DESKTOP_PLANE")
    adjusted = [
        obj.Name
        for obj in doc.Objects
        if (
            obj.Name.startswith("REF_SCENE_table_")
            or obj.Name.startswith("REF_SCENE_frame_")
            or obj.Name.startswith("REF_SCENE_camera_")
        )
    ]
    robot = [obj.Name for obj in doc.Objects if obj.Name.startswith("REF_ALOHA_")]
    report = {
        "iteration": "iter_001_measured_table_rack",
        "freecad_file": str(FCSTD.relative_to(ROOT)),
        "desktop_plane": {
            "length_mm": float(plane.Length.Value),
            "width_mm": float(plane.Width.Value),
            "height_mm": float(plane.Height.Value),
            "base_mm": [float(plane.Placement.Base.x), float(plane.Placement.Base.y), float(plane.Placement.Base.z)],
        },
        "expected_desktop_plane": {"length_mm": 1220.0, "width_mm": 625.0, "height_mm": 18.0},
        "adjusted_table_rack_camera_mesh_count": len(adjusted),
        "robot_mesh_count_not_scaled": len(robot),
        "ok": abs(float(plane.Length.Value) - 1220.0) < 1e-6
        and abs(float(plane.Width.Value) - 625.0) < 1e-6
        and len(adjusted) == 31
        and len(robot) > 0,
    }
    (ITER_DIR / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


raise SystemExit(main())

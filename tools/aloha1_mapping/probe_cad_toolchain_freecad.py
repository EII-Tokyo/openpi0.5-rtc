"""Probe the exact local FreeCAD/OCC tessellation API without GUI mutation.

Required environment variable:
ALOHA_CAD_TOOLCHAIN_PROBE_OUTPUT
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import FreeCAD as App
import Part


def _occ_version() -> str | None:
    for name in ("OCC_VERSION", "OCC_VERSION_STRING"):
        value = getattr(Part, name, None)
        if value:
            return str(value)
    getter = getattr(Part, "getOCCVersion", None)
    if callable(getter):
        return str(getter())
    return None


output_text = os.environ.get("ALOHA_CAD_TOOLCHAIN_PROBE_OUTPUT")
if not output_text:
    raise RuntimeError("ALOHA_CAD_TOOLCHAIN_PROBE_OUTPUT is required")

shape = Part.makeBox(10.0, 20.0, 30.0)
linear_deflection_mm = 0.20
angular_deflection_rad = 0.3490658503988659
part_vertices, part_facets = shape.tessellate(linear_deflection_mm)
two_argument_error = None
try:
    shape.tessellate(linear_deflection_mm, angular_deflection_rad)
    part_two_argument_supported = True
except Exception as exc:
    part_two_argument_supported = False
    two_argument_error = f"{type(exc).__name__}: {exc}"
meshpart_error = None
meshpart_record = {
    "available": False,
    "linear_deflection_mm": linear_deflection_mm,
    "angular_deflection_rad": angular_deflection_rad,
    "relative": False,
}
try:
    import MeshPart

    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=linear_deflection_mm,
        AngularDeflection=angular_deflection_rad,
        Relative=False,
    )
    topology = mesh.Topology
    meshpart_record.update(
        {
            "available": True,
            "point_count": len(topology[0]),
            "facet_count": len(topology[1]),
        }
    )
except Exception as exc:
    meshpart_error = f"{type(exc).__name__}: {exc}"
    meshpart_record["error"] = meshpart_error
report = {
    "schema_version": 1,
    "status": "PASS",
    "freecad_version": list(App.Version()),
    "freecad_executable": "/snap/bin/freecad.cmd",
    "opencascade_version": _occ_version(),
    "part_shape_tessellate": {
        "probe_shape": "Part.makeBox(10,20,30) millimetres",
        "linear_deflection_mm": linear_deflection_mm,
        "point_count": len(part_vertices),
        "facet_count": len(part_facets),
        "two_argument_angular_supported": part_two_argument_supported,
        "two_argument_error": two_argument_error,
        "doc": Part.Shape.tessellate.__doc__,
    },
    "meshpart_mesh_from_shape": meshpart_record,
    "part_shape_tessellate_note": (
        "Part.Shape.tessellate exposes only linear deflection. "
        + (
            "MeshPart.meshFromShape is available, so a later production "
            "tessellation may explicitly control angular deflection and "
            "Relative=False."
            if meshpart_error is None
            else
            "MeshPart.meshFromShape is unavailable in this local snap "
            "runtime, so angular-deflection-controlled production "
            "tessellation is HARD_BLOCKED. Part.Shape.tessellate may still "
            "be used for a clearly labelled linear-deflection-only "
            "determinism diagnostic."
        )
    ),
    "production_tessellation_gate": (
        "PASS" if meshpart_error is None else "HARD_BLOCKER"
    ),
}
if report["opencascade_version"] is None or meshpart_error is not None:
    report["status"] = "PARTIAL"
output = Path(output_text).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(output)

"""Runtime probe for the installed FreeCAD STEP/AP214 Python interface.

Run with:
    /snap/bin/freecad.cmd -c tools/aloha1_mapping/probe_freecad_step_api.py
"""

from __future__ import annotations

import json
import os

import FreeCAD as App
import Import
import Part


def _public_names(value: object) -> list[str]:
    return sorted(name for name in dir(value) if not name.startswith("_"))


shape_members = set(_public_names(Part.Shape))
step_preferences = App.ParamGet(
    "User parameter:BaseApp/Preferences/Mod/Part/STEP"
)
probe = {
    "freecad_version": list(App.Version()),
    "modules": {
        "FreeCAD": getattr(App, "__file__", None),
        "Import": getattr(Import, "__file__", None),
        "Part": getattr(Part, "__file__", None),
    },
    "import_api": _public_names(Import),
    "step_preferences": {
        "contents": [list(item) for item in step_preferences.GetContents()],
        "groups": list(step_preferences.GetGroups()),
    },
    "part_shape_capabilities": {
        name: name in shape_members
        for name in (
            "Area",
            "BoundBox",
            "CenterOfMass",
            "CompSolids",
            "Compounds",
            "Edges",
            "Faces",
            "Placement",
            "ShapeType",
            "Shells",
            "Solids",
            "Vertexes",
            "Volume",
            "check",
            "exportBrep",
            "hashCode",
            "isNull",
            "isSame",
            "isValid",
            "read",
            "tessellate",
        )
    },
}

payload = json.dumps(probe, indent=2, ensure_ascii=False)
print("ALOHA_FREECAD_STEP_API_PROBE_BEGIN")
print(payload)
print("ALOHA_FREECAD_STEP_API_PROBE_END")

output = os.environ.get("ALOHA_FREECAD_PROBE_OUTPUT")
if output:
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(payload + "\n")

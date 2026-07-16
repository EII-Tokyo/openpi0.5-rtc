"""Open Iteration 004 mesh-repaired model in FreeCAD."""

from pathlib import Path

import FreeCAD
import FreeCADGui as Gui


ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
FCSTD = (
    ROOT
    / "scene_reconstruction"
    / "cad"
    / "aloha_incremental"
    / "iterations"
    / "iter_004_mesh_repaired_lower_camera_top_position"
    / "iter_004_mesh_repaired_lower_camera_top_position.FCStd"
)


doc = FreeCAD.openDocument(str(FCSTD))
Gui.activateWorkbench("MeshWorkbench")
Gui.ActiveDocument = Gui.getDocument(doc.Name)

for obj in doc.Objects:
    try:
        obj.Visibility = True
    except Exception:
        pass
    try:
        obj.ViewObject.Visibility = True
    except Exception:
        pass

try:
    Gui.Selection.clearSelection()
    marker = doc.getObject("NEW_LOWER_CAMERA_POSITION_GREEN")
    if marker is not None:
        Gui.Selection.addSelection(marker)
except Exception:
    pass

try:
    view = Gui.ActiveDocument.ActiveView
    view.viewFront()
    view.fitAll()
except Exception as exc:
    print("VIEW_SETUP_FAILED", repr(exc))

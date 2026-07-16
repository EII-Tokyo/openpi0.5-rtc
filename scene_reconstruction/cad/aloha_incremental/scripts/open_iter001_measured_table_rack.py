"""Open Iteration 001 measured table/rack model in FreeCAD."""

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
    / "iter_001_measured_table_rack"
    / "iter_001_measured_table_rack.FCStd"
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
    view = Gui.ActiveDocument.ActiveView
    view.viewTop()
    view.fitAll()
except Exception as exc:
    print("VIEW_SETUP_FAILED", repr(exc))

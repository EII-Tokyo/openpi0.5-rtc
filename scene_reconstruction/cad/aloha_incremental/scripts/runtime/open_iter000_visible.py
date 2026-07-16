import os
import FreeCAD
import FreeCADGui as Gui

fcstd = "/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_000_reference/iter_000_reference.FCStd"
doc = FreeCAD.openDocument(fcstd)
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
    view.viewAxometric()
    view.fitAll()
except Exception as exc:
    print("VIEW_SETUP_FAILED", repr(exc))

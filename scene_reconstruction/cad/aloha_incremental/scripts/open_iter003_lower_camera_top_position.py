"""Open Iteration 003 lower camera top-position model in FreeCAD."""

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
    / "iter_003_lower_camera_top_position"
    / "iter_003_lower_camera_top_position.FCStd"
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
    for name in (
        "REF_SCENE_frame_wormseye_mount_30",
        "MEASURED_CAMERA_SUPPORT_PIPE_260MM_1",
        "MEASURED_CAMERA_SUPPORT_PIPE_260MM_2",
        "MEASURED_CAMERA_SUPPORT_PIPE_260MM_3",
        "MEASURED_CAMERA_SUPPORT_PIPE_260MM_4",
    ):
        obj = doc.getObject(name)
        if obj is not None:
            Gui.Selection.addSelection(obj)
except Exception:
    pass

try:
    view = Gui.ActiveDocument.ActiveView
    view.viewTop()
    view.fitAll()
except Exception as exc:
    print("VIEW_SETUP_FAILED", repr(exc))

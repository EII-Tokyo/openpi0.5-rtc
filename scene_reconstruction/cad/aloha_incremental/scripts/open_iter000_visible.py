"""Open Iteration 000 in FreeCAD with a deterministic visible 3D view.

Run with the FreeCAD GUI executable, for example:

    /snap/bin/freecad /path/to/open_iter000_visible.py

The script does not modify source assets. It opens the generated FCStd
reference file, makes imported reference objects visible, and frames the view.
"""

from pathlib import Path

import FreeCAD
import FreeCADGui as Gui


ROOT = Path(__file__).resolve().parents[4]
FCSTD = (
    ROOT
    / "scene_reconstruction"
    / "cad"
    / "aloha_incremental"
    / "iterations"
    / "iter_000_reference"
    / "iter_000_reference.FCStd"
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
    view.viewAxometric()
    view.fitAll()
except Exception as exc:
    print("VIEW_SETUP_FAILED", repr(exc))

"""Open the local ALOHA sandpaper review with deterministic GUI visibility."""

from __future__ import annotations

import os
from pathlib import Path

import FreeCAD as App
import FreeCADGui as Gui

review_dir_text = os.environ.get("ALOHA_SANDPAPER_REVIEW_DIR")
if not review_dir_text:
    raise RuntimeError("ALOHA_SANDPAPER_REVIEW_DIR is required")
review_dir = Path(review_dir_text).resolve(strict=True)

documents = {}
for side in ("right", "left"):
    path = review_dir / f"aloha_sandpaper_{side}_zero_thickness_review.FCStd"
    if not path.is_file():
        raise RuntimeError(f"missing review FCStd: {path}")
    document = App.openDocument(str(path))
    documents[side] = document
    for obj in document.Objects:
        if obj.ViewObject is not None:
            obj.ViewObject.Visibility = False
    for group_name in ("SourceFinger", "WrappedCoverage"):
        group = document.getObject(group_name)
        if group is None:
            raise RuntimeError(f"{side}: missing GUI group {group_name}")
        group.ViewObject.Visibility = True
        for obj in group.Group:
            obj.ViewObject.Visibility = True
            if hasattr(obj, "ReviewColor"):
                obj.ViewObject.ShapeColor = obj.ReviewColor
                obj.ViewObject.LineColor = obj.ReviewColor
    source = document.getObject("InstalledFingerBRep")
    if source is None:
        raise RuntimeError(f"{side}: missing InstalledFingerBRep")
    source.ViewObject.Transparency = 72
    for obj in document.getObject("WrappedCoverage").Group:
        obj.ViewObject.Transparency = 5
        obj.ViewObject.LineWidth = 2.0

left = documents["left"]
App.setActiveDocument(left.Name)
Gui.activeDocument().activeView().viewAxonometric()
Gui.activeDocument().activeView().fitAll()
main_window = Gui.getMainWindow()
main_window.showMaximized()
main_window.raise_()
main_window.activateWindow()

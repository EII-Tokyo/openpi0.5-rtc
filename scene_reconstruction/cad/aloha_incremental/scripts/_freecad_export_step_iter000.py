from __future__ import annotations

import Import
import FreeCAD

ROOT = "/home/eii/project/openpi0.5-rtc-reward-learning"
fcstd = f"{ROOT}/scene_reconstruction/cad/aloha_incremental/iterations/iter_000_reference/iter_000_reference.FCStd"
step = f"{ROOT}/scene_reconstruction/cad/aloha_incremental/exports/iter_000_reference_editable_solids.step"
doc = FreeCAD.openDocument(fcstd)
objects = [obj for obj in doc.Objects if obj.Name.startswith("REF_TABLE") or obj.Name.startswith("REF_AXIS")]
Import.export(objects, step)
print(step)

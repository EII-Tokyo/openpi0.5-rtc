#!/usr/bin/env python3
"""Create the ALOHA1 clean Isaac USD skeleton.

This A0/A1 stage is only a namespaced asset scaffold. It does not reference
legacy ALOHA USDs or Trossen USDs and it does not define sim-ready semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


ROOT = "/aloha"

ROOT_SCOPES = [
    "/Render",
    "/meshes",
    "/visuals",
    "/colliders",
]

ROOT_CHILDREN = {
    "Looks": "Scope",
    "joints": "Scope",
    "table_link": "Xform",
    "support_frame": "Xform",
    "cam_high_link": "Xform",
    "cam_low_link": "Xform",
    "left_base_link": "Xform",
    "right_base_link": "Xform",
    "left_camera_link": "Xform",
    "right_camera_link": "Xform",
}


def create_stage(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)

    for path in ROOT_SCOPES:
        UsdGeom.Scope.Define(stage, path)

    for name, prim_type in ROOT_CHILDREN.items():
        path = f"{ROOT}/{name}"
        if prim_type == "Scope":
            UsdGeom.Scope.Define(stage, path)
        elif prim_type == "Xform":
            UsdGeom.Xform.Define(stage, path)
        else:
            raise ValueError(f"Unsupported prim type for {path}: {prim_type}")

    root.SetCustomDataByKey("aloha1_rebuild_stage", "A0_A1_skeleton_only")
    root.SetCustomDataByKey("source_policy", "structure_only_no_numeric_parameters")

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aloha_isaac_rebuild/scenes/aloha_skeleton.usda"),
        help="Output USDA path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_stage(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

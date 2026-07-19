from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_STAGE = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda"
)
DEFAULT_BOTTLE_USD = REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/bottle_tabletop_debug_stage_20260719"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _bbox(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {"path": prim_path, "bbox_valid": False}
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    return {
        "path": prim_path,
        "bbox_valid": True,
        "min": [float(mn[i]) for i in range(3)],
        "max": [float(mx[i]) for i in range(3)],
        "center": [float((mn[i] + mx[i]) * 0.5) for i in range(3)],
        "size": [float(mx[i] - mn[i]) for i in range(3)],
    }


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bottle_tabletop_debug_stage.json"
    md_path = output_dir / "bottle_tabletop_debug_stage.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Bottle Tabletop Debug Stage",
        "",
        f"- status: `{payload['status']}`",
        f"- generated stage: `{payload['debug_stage_usda']}`",
        f"- base stage: `{payload['base_stage']}`",
        f"- bottle runtime path: `{payload['bottle_path']}`",
        f"- table path: `{payload['table_path']}`",
        f"- table top z: `{payload.get('table_top_z_m')}` m",
        f"- bottle bottom z: `{payload.get('bottle_bottom_z_m')}` m",
        f"- tabletop gap: `{payload.get('tabletop_gap_m')}` m",
        f"- bbox pass: `{payload.get('bbox_pass')}`",
        f"- tabletop pass: `{payload.get('tabletop_pass')}`",
        "",
        "This is a simulation-only visual/debug overlay. It does not overwrite the confirmed ALOHA startup stage.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standalone debug USD that overlays a visible Bottle500 on the measured ALOHA tabletop. "
            "The base stage and BottleUSD are referenced read-only; the output stage is safe to open in Isaac."
        )
    )
    parser.add_argument("--base-stage", type=Path, default=DEFAULT_BASE_STAGE)
    parser.add_argument("--bottle-usd", type=Path, default=DEFAULT_BOTTLE_USD)
    parser.add_argument("--bottle-usd-prim-path", default="/Bottle500")
    parser.add_argument("--table-path", default="/scene/worldBody/table")
    parser.add_argument("--bottle-path", default="/World/Debug/Bottle500TabletopProbe")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--axis", choices=("X", "Y"), default="X")
    args = parser.parse_args()

    app = SimulationApp({"headless": True, "width": 640, "height": 480})
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stage = output_dir / "bottle_tabletop_debug_stage.usda"

    stage = Usd.Stage.CreateNew(str(output_stage))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(str(args.base_stage.resolve()))

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Debug")

    table_box = _bbox(stage, args.table_path)
    if not table_box["bbox_valid"]:
        raise RuntimeError(f"Cannot place bottle: missing table prim {args.table_path}")
    table_center = np.asarray(table_box["center"], dtype=np.float64)
    table_top_z = float(table_box["max"][2])

    bottle = UsdGeom.Xform.Define(stage, args.bottle_path)
    bottle.GetPrim().GetReferences().AddReference(str(args.bottle_usd.resolve()), args.bottle_usd_prim_path)
    xform = UsdGeom.Xformable(bottle.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
    # Bottle500 local +Z is the long axis. Rotate it onto the tabletop.
    rotate_op.Set(Gf.Vec3d(0.0, 90.0, 0.0) if args.axis == "X" else Gf.Vec3d(-90.0, 0.0, 0.0))
    translate_op.Set(Gf.Vec3d(float(table_center[0]), float(table_center[1]), table_top_z + 0.04))

    first_box = _bbox(stage, args.bottle_path)
    if not first_box["bbox_valid"]:
        raise RuntimeError(f"Cannot compute bottle bbox at {args.bottle_path}")
    z_correction = table_top_z - float(first_box["min"][2])
    translate = np.asarray([table_center[0], table_center[1], table_top_z + 0.04 + z_correction], dtype=np.float64)
    translate_op.Set(Gf.Vec3d(*[float(v) for v in translate]))

    light = UsdLux.DistantLight.Define(stage, "/World/Debug/BottleTabletopKeyLight")
    light.CreateIntensityAttr(1500.0)
    camera = UsdGeom.Camera.Define(stage, "/World/Debug/BottleTabletopCamera")
    camera_xf = UsdGeom.Xformable(camera.GetPrim())
    camera_xf.ClearXformOpOrder()
    camera_xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(table_center[0]) - 0.55, float(table_center[1]) - 0.75, table_top_z + 0.55)
    )
    camera_xf.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(62.0, 0.0, -35.0))
    camera.CreateFocalLengthAttr(28.0)

    bottle_box = _bbox(stage, args.bottle_path)
    bottle_size = bottle_box["size"]
    longest_axis = max(bottle_size)
    bbox_pass = bool(0.18 <= longest_axis <= 0.24 and min(bottle_size) >= 0.04)
    tabletop_gap = float(bottle_box["min"][2] - table_top_z)
    tabletop_pass = bool(abs(tabletop_gap) <= 1e-4)
    payload = {
        "status": "PASS" if bbox_pass and tabletop_pass else "FAILED_GATE",
        "base_stage": _rel(args.base_stage),
        "debug_stage_usda": _rel(output_stage),
        "bottle_usd": _rel(args.bottle_usd),
        "bottle_path": args.bottle_path,
        "table_path": args.table_path,
        "table_bbox": table_box,
        "bottle_bbox": bottle_box,
        "table_top_z_m": table_top_z,
        "bottle_bottom_z_m": float(bottle_box["min"][2]),
        "tabletop_gap_m": tabletop_gap,
        "bbox_pass": bbox_pass,
        "tabletop_pass": tabletop_pass,
        "camera_path": "/World/Debug/BottleTabletopCamera",
        "notes": "Debug overlay only: generated stage references the base ALOHA scene and Bottle500 without modifying them.",
    }
    stage.GetRootLayer().Save()
    _write_report(output_dir, payload)
    print(json.dumps({"status": payload["status"], "debug_stage_usda": payload["debug_stage_usda"]}))
    app.close()
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

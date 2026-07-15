#!/usr/bin/env python3
"""Read-only OpenUSD stage audit using Isaac Sim's bundled USD libraries.

This script avoids launching SimulationApp. It bootstraps the local Isaac
OpenUSD Python bindings only for this process and writes deterministic audit
files into scene_reconstruction/audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "scene_reconstruction/audit"


def _bootstrap_pxr() -> None:
    try:
        from pxr import Usd  # noqa: F401

        return
    except Exception:
        pass
    if os.environ.get("SCENE_RECONSTRUCTION_PXR_BOOTSTRAPPED") == "1":
        return

    exts = REPO_ROOT / ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache"
    usd_libs = sorted(exts.glob("omni.usd.libs-*"))
    py_libs = sorted(Path("/home/eii/.local/share/uv/python").glob("cpython-3.11*/lib/libpython3.11.so.1.0"))
    if not usd_libs or not py_libs:
        raise RuntimeError("Cannot locate Isaac OpenUSD libraries or uv Python libpython.")

    usd_py_dir = usd_libs[0]
    usd_bin_dir = usd_py_dir / "bin"
    py_lib_dir = py_libs[0].parent
    env = dict(os.environ)
    env["SCENE_RECONSTRUCTION_PXR_BOOTSTRAPPED"] = "1"
    env["PYTHONPATH"] = f"{usd_py_dir}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{usd_bin_dir}:{py_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    os.execvpe(os.environ.get("PYTHON", ".venv_issac/bin/python"), [os.environ.get("PYTHON", ".venv_issac/bin/python"), *os.sys.argv], env)


def _vec(value: Any) -> list[float]:
    try:
        return [float(x) for x in value]
    except Exception:
        return []


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def _quat_from_matrix(matrix: Any) -> list[float]:
    quat = matrix.ExtractRotationQuat().GetNormalized()
    return [float(quat.GetReal()), float(quat.GetImaginary()[0]), float(quat.GetImaginary()[1]), float(quat.GetImaginary()[2])]


def audit_stage(stage_path: Path, max_prims: int) -> dict[str, Any]:
    _bootstrap_pxr()
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise FileNotFoundError(stage_path)

    cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    transforms: dict[str, Any] = {}
    cameras: dict[str, Any] = {}
    prim_tree: list[str] = []
    for idx, prim in enumerate(stage.Traverse()):
        path = str(prim.GetPath())
        if idx < max_prims:
            prim_tree.append(f"{path} [{prim.GetTypeName()}]")
        if prim.IsA(UsdGeom.Xformable):
            matrix = cache.GetLocalToWorldTransform(prim)
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            transforms[path] = {
                "type": prim.GetTypeName(),
                "world_matrix": _matrix_rows(matrix),
                "translation": _vec(matrix.ExtractTranslation()),
                "quaternion_wxyz": _quat_from_matrix(matrix),
                "bbox_min": _vec(bbox.GetMin()),
                "bbox_max": _vec(bbox.GetMax()),
            }
        if prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)
            matrix = cache.GetLocalToWorldTransform(prim)
            cameras[path] = {
                "translation": _vec(matrix.ExtractTranslation()),
                "quaternion_wxyz": _quat_from_matrix(matrix),
                "focal_length": float(camera.GetFocalLengthAttr().Get() or 0.0),
                "horizontal_aperture": float(camera.GetHorizontalApertureAttr().Get() or 0.0),
                "vertical_aperture": float(camera.GetVerticalApertureAttr().Get() or 0.0),
                "clipping_range": _vec(camera.GetClippingRangeAttr().Get() or Gf.Vec2f(0.0, 0.0)),
                "note": "USD Camera Prim, not GUI viewport.",
            }

    return {
        "stage_path": str(stage_path),
        "stage_units": {
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        },
        "layers": {
            "root_identifier": stage.GetRootLayer().identifier,
            "root_real_path": stage.GetRootLayer().realPath,
            "root_sub_layers": list(stage.GetRootLayer().subLayerPaths),
        },
        "prim_count": sum(1 for _ in stage.Traverse()),
        "prim_tree_sample": prim_tree,
        "transforms": transforms,
        "cameras": cameras,
        "render_products": {},
    }


def write_outputs(data: dict[str, Any], prefix: str) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / f"{prefix}_stage_audit.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (AUDIT / f"{prefix}_transforms.json").write_text(json.dumps(data["transforms"], indent=2), encoding="utf-8")
    (AUDIT / f"{prefix}_cameras.json").write_text(json.dumps(data["cameras"], indent=2), encoding="utf-8")
    (AUDIT / f"{prefix}_prim_tree.txt").write_text("\n".join(data["prim_tree_sample"]) + "\n", encoding="utf-8")
    (AUDIT / f"{prefix}_stage_layers.json").write_text(json.dumps(data["layers"], indent=2), encoding="utf-8")
    (AUDIT / f"{prefix}_render_products.json").write_text(json.dumps(data["render_products"], indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--prefix", default="openusd")
    parser.add_argument("--max-prims", type=int, default=1000)
    args = parser.parse_args()
    data = audit_stage(args.stage.resolve(), args.max_prims)
    write_outputs(data, args.prefix)
    print(json.dumps({"stage": data["stage_path"], "prim_count": data["prim_count"], "cameras": list(data["cameras"])}, indent=2))


if __name__ == "__main__":
    main()

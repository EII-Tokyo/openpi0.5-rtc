from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ITER003_JSON = (
    REPO_ROOT
    / "scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/bbox_and_dimensions.json"
)
DEFAULT_BASE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_OUTPUT_USD = (
    REPO_ROOT
    / "scene_reconstruction/usd/iter003_on_original/aloha_original_with_iter003_table_rack.usda"
)


def _bootstrap_pxr() -> None:
    try:
        from pxr import Usd  # noqa: F401

        return
    except Exception:
        pass
    if os.environ.get("ITER003_OVERLAY_PXR_BOOTSTRAPPED") == "1":
        return

    exts = REPO_ROOT / ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache"
    usd_libs = sorted(exts.glob("omni.usd.libs-*"))
    py_libs = sorted(Path("/home/eii/.local/share/uv/python").glob("cpython-3.11*/lib/libpython3.11.so.1.0"))
    if not usd_libs or not py_libs:
        raise RuntimeError("Cannot locate Isaac OpenUSD libraries or uv Python libpython.")

    env = dict(os.environ)
    env["ITER003_OVERLAY_PXR_BOOTSTRAPPED"] = "1"
    env["PYTHONPATH"] = f"{usd_libs[0]}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{usd_libs[0] / 'bin'}:{py_libs[0].parent}:{env.get('LD_LIBRARY_PATH', '')}"
    python = os.environ.get("PYTHON", str(REPO_ROOT / ".venv_issac/bin/python"))
    os.execvpe(python, [python, *os.sys.argv], env)


def _mm_to_m(value: float) -> float:
    return float(value) / 1000.0


def _vec_mm_to_m(values: list[float]) -> list[float]:
    return [_mm_to_m(v) for v in values]


def _add_cube(
    stage: Any,
    path: str,
    center_m: list[float],
    size_m: list[float],
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if opacity < 1.0:
        cube.CreateDisplayOpacityAttr([float(opacity)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_m))


def _add_cylinder_between(
    stage: Any,
    path: str,
    start_m: list[float],
    end_m: list[float],
    radius_m: float,
    color: tuple[float, float, float],
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    start = Gf.Vec3d(*start_m)
    end = Gf.Vec3d(*end_m)
    midpoint = (start + end) * 0.5
    direction = end - start
    height = direction.GetLength()
    if height <= 1e-9:
        raise ValueError(f"zero-length cylinder: {path}")

    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cyl.CreateRadiusAttr(float(radius_m))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), direction.GetNormalized())
    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(midpoint)
    xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))


def _bbox_center_size_m(bbox_mm: list[float]) -> tuple[list[float], list[float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bbox_mm]
    center = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0]
    size = [xmax - xmin, ymax - ymin, zmax - zmin]
    return _vec_mm_to_m(center), _vec_mm_to_m(size)


def build_stage(iter003_json: Path, base_usd: Path, output_usd: Path) -> Path:
    _bootstrap_pxr()
    from pxr import Sdf, Usd, UsdGeom

    if not iter003_json.exists():
        raise FileNotFoundError(iter003_json)
    if not base_usd.exists():
        raise FileNotFoundError(base_usd)

    data = json.loads(iter003_json.read_text(encoding="utf-8"))
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()

    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(base_usd, output_usd.parent))

    root_path = "/World/Iter003FreeCADOverlay"
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path)).GetPrim()
    root.SetCustomDataByKey("source_iteration", data["iteration"])
    root.SetCustomDataByKey("source_json", str(iter003_json.resolve()))
    root.SetCustomDataByKey("base_usd", str(base_usd.resolve()))
    root.SetCustomDataByKey("intent", "FreeCAD iter_003 proxy geometry over original ALOHA only; lighting overlay untouched")

    edges = data["reference_edges"]
    outer_width = _mm_to_m(edges["outer_width_mm"])
    outer_depth = _mm_to_m(edges["y_outer_edges_mm"][1] - edges["y_outer_edges_mm"][0])
    _add_cube(
        stage,
        f"{root_path}/table_footprint_from_iter003",
        [0.0, 0.0, 0.006],
        [outer_width, outer_depth, 0.006],
        (0.78, 0.65, 0.42),
        opacity=0.58,
    )

    rail = data["top_steel_rail_proxy"]
    _add_cube(
        stage,
        f"{root_path}/{rail['object']}",
        _vec_mm_to_m(rail["center_mm"]),
        _vec_mm_to_m(rail["size_mm"]),
        (0.16, 0.16, 0.16),
    )

    for pipe in data["support_pipes_260mm"]:
        _add_cube(
            stage,
            f"{root_path}/{pipe['object']}",
            _vec_mm_to_m(pipe["center_mm"]),
            _vec_mm_to_m(pipe["size_mm"]),
            (0.18, 0.18, 0.18),
        )

    mount = data["moved_mount"]
    mount_center, mount_size = _bbox_center_size_m(mount["bbox_after_mm"])
    _add_cube(
        stage,
        f"{root_path}/MOVED_LOWER_CAMERA_MOUNT_BBOX",
        mount_center,
        mount_size,
        (0.0, 0.86, 0.10),
        opacity=0.86,
    )
    camera_center = _vec_mm_to_m(data["new_camera_position"]["center_mm"])
    _add_cube(
        stage,
        f"{root_path}/LOWER_CAMERA_TARGET_CENTER",
        camera_center,
        [0.035, 0.035, 0.035],
        (0.0, 1.0, 0.0),
    )
    _add_cylinder_between(
        stage,
        f"{root_path}/LOWER_CAMERA_NEGATIVE_Y_DIRECTION",
        camera_center,
        [camera_center[0], camera_center[1] - 0.16, camera_center[2]],
        0.006,
        (0.0, 0.85, 0.0),
    )

    left_edge, right_edge = [_mm_to_m(v) for v in edges["x_outer_edges_mm"]]
    front_edge, back_edge = [_mm_to_m(v) for v in edges["y_outer_edges_mm"]]
    for name, center in {
        "outer_left_edge_marker": [left_edge, 0.0, 0.035],
        "outer_right_edge_marker": [right_edge, 0.0, 0.035],
        "outer_front_edge_marker": [0.0, front_edge, 0.035],
        "outer_back_edge_marker": [0.0, back_edge, 0.035],
    }.items():
        _add_cube(stage, f"{root_path}/{name}", center, [0.018, 0.018, 0.05], (0.1, 0.35, 1.0))

    stage.GetRootLayer().Save()
    return output_usd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build original ALOHA + FreeCAD iter_003 proxy overlay USD.")
    parser.add_argument("--iter003-json", type=Path, default=DEFAULT_ITER003_JSON)
    parser.add_argument("--base-usd", type=Path, default=DEFAULT_BASE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    args = parser.parse_args()

    output = build_stage(args.iter003_json.resolve(), args.base_usd.resolve(), args.output_usd.resolve())
    print(f"usd={output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only audit of the confirmed ALOHA Isaac USD stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
OUT = REPO_ROOT / "scene_reconstruction"
AUDIT = OUT / "audit"
REPORTS = OUT / "reports"


def _vec(value: Any) -> list[float]:
    try:
        return [float(x) for x in value]
    except Exception:
        return []


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def _quat_from_matrix(matrix: Any) -> list[float]:
    q = matrix.ExtractRotationQuat().GetNormalized()
    return [float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2])]


def audit_stage(stage_path: Path, max_prims: int) -> dict[str, Any]:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(stage_path))
        if stage is None:
            raise FileNotFoundError(stage_path)

        cache = UsdGeom.XformCache()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()

        prim_tree: list[str] = []
        transforms: dict[str, Any] = {}
        cameras: dict[str, Any] = {}
        robot_candidates: list[str] = []
        table_candidates: list[str] = []
        pipe_candidates: list[str] = []
        for idx, prim in enumerate(stage.Traverse()):
            if idx < max_prims:
                prim_tree.append(f"{prim.GetPath()} [{prim.GetTypeName()}]")
            path = str(prim.GetPath())
            lower = path.lower()
            if "aloha" in lower or "left" in lower or "right" in lower or "vx300" in lower:
                robot_candidates.append(path)
            if "table" in lower:
                table_candidates.append(path)
            if "pipe" in lower:
                pipe_candidates.append(path)

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
                    "world_matrix": _matrix_rows(matrix),
                    "translation": _vec(matrix.ExtractTranslation()),
                    "quaternion_wxyz": _quat_from_matrix(matrix),
                    "focal_length": float(camera.GetFocalLengthAttr().Get() or 0.0),
                    "horizontal_aperture": float(camera.GetHorizontalApertureAttr().Get() or 0.0),
                    "vertical_aperture": float(camera.GetVerticalApertureAttr().Get() or 0.0),
                    "clipping_range": _vec(camera.GetClippingRangeAttr().Get() or Gf.Vec2f(0.0, 0.0)),
                    "note": "USD Camera Prim, not GUI viewport.",
                }

        data = {
            "stage_path": str(stage_path),
            "stage_units": {
                "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            },
            "layers": {
                "root_identifier": root_layer.identifier,
                "root_real_path": root_layer.realPath,
                "root_sub_layers": list(root_layer.subLayerPaths),
                "session_identifier": session_layer.identifier if session_layer else "",
            },
            "prim_count": sum(1 for _ in stage.Traverse()),
            "prim_tree_sample": prim_tree,
            "transforms": transforms,
            "cameras": cameras,
            "render_products": {},
            "candidates": {
                "robot": robot_candidates[:80],
                "table": table_candidates[:40],
                "pipe": pipe_candidates[:40],
            },
        }
        write_outputs(data, stage_path)
        return data
    finally:
        app.close()


def write_outputs(data: dict[str, Any], stage_path: Path) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (AUDIT / "stage_layers.json").write_text(json.dumps(data["layers"], indent=2), encoding="utf-8")
    (AUDIT / "prim_tree.txt").write_text("\n".join(data["prim_tree_sample"]) + "\n", encoding="utf-8")
    (AUDIT / "transforms.json").write_text(json.dumps(data["transforms"], indent=2), encoding="utf-8")
    (AUDIT / "cameras.json").write_text(json.dumps(data["cameras"], indent=2), encoding="utf-8")
    (AUDIT / "render_products.json").write_text(json.dumps(data["render_products"], indent=2), encoding="utf-8")
    (AUDIT / "isaac_stage_audit.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    camera_lines = [f"- `{path}` focal={info['focal_length']} translation={info['translation']}" for path, info in data["cameras"].items()]
    if not camera_lines:
        camera_lines = ["- No `UsdGeom.Camera` prim found; GUI viewport camera must not be treated as a real sensor camera."]
    md = [
        "# Isaac Stage Audit",
        "",
        f"- Stage: `{stage_path}`",
        "- Original stage modified: `no`",
        f"- Meters per unit: `{data['stage_units']['meters_per_unit']}`",
        f"- Up axis: `{data['stage_units']['up_axis']}`",
        f"- Prim count: `{data['prim_count']}`",
        "",
        "## Camera Prim Audit",
        "",
        *camera_lines,
        "",
        "## Important Distinction",
        "",
        "- GUI viewport: an editor view used by the human. It is not automatically a simulated sensor.",
        "- `UsdGeom.Camera`: a stage prim with transform and optical parameters. Only these are recorded in `cameras.json`.",
        "- Isaac sensor camera / render product: runtime sensor pipeline. None was assumed unless found in the stage.",
        "",
        "## Candidate Prim Paths",
        "",
        f"- Robot candidates: `{len(data['candidates']['robot'])}` sample paths in JSON",
        f"- Table candidates: `{len(data['candidates']['table'])}` sample paths in JSON",
        f"- Pipe candidates: `{len(data['candidates']['pipe'])}` sample paths in JSON",
    ]
    (REPORTS / "isaac_stage_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=None)
    parser.add_argument("--max-prims", type=int, default=500)
    args = parser.parse_args()
    stage_arg = args.stage or Path(os.environ.get("SCENE_RECONSTRUCTION_STAGE", DEFAULT_STAGE))
    stage_path = stage_arg.resolve()
    data = audit_stage(stage_path, args.max_prims)
    print(json.dumps({"stage": str(stage_path), "cameras": len(data["cameras"]), "prim_count": data["prim_count"]}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()

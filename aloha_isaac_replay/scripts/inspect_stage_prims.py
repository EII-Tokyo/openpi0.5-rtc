from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/prim_audit"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return list(value)
    except Exception:
        return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _applied(prim: Any) -> list[str]:
    return [str(item) for item in prim.GetAppliedSchemas()]


def _has_schema(prim: Any, schema_name: str) -> bool:
    return schema_name in _applied(prim)


def _collision_enabled(prim: Any) -> bool | None:
    if not _has_schema(prim, "PhysicsCollisionAPI"):
        return None
    from pxr import UsdPhysics

    attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
    if not attr:
        return None
    value = attr.Get()
    return None if value is None else bool(value)


def _bbox_row(cache: Any, prim: Any) -> dict[str, Any]:
    # ComputeWorldBound returns an oriented GfBBox3d. GetBox() alone can expose
    # the unaligned local range for scaled/rotated Cube/Cylinder prims. Use the
    # aligned world range for geometry audits and contact policy decisions.
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return {
            "bbox_valid": False,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_center": None,
            "bbox_size": None,
        }
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_center": [float((min_pt[i] + max_pt[i]) * 0.5) for i in range(3)],
        "bbox_size": [float(max_pt[i] - min_pt[i]) for i in range(3)],
    }


def _prim_row(stage: Any, cache: Any, path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    row: dict[str, Any] = {"path": path, "exists": bool(prim and prim.IsValid())}
    if not prim or not prim.IsValid():
        return row
    row.update(
        {
            "type_name": prim.GetTypeName(),
            "is_instance": bool(prim.IsInstance()),
            "is_instanceable": bool(prim.IsInstanceable()),
            "applied_schemas": _applied(prim),
            "has_collision_api": _has_schema(prim, "PhysicsCollisionAPI"),
            "collision_enabled": _collision_enabled(prim),
            "has_rigid_body_api": _has_schema(prim, "PhysicsRigidBodyAPI"),
        }
    )
    row.update(_bbox_row(cache, prim))
    return row


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage Prim Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- prim count: `{len(payload.get('rows', []))}`",
        "",
        "| path | exists | type | collision | rigid body | bbox center | bbox size |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("rows", []):
        collision_state = row.get("has_collision_api", "")
        if row.get("collision_enabled") is not None:
            collision_state = f"{collision_state} / enabled={row.get('collision_enabled')}"
        lines.append(
            "| "
            f"`{row['path']}` | "
            f"{row.get('exists')} | "
            f"`{row.get('type_name', '')}` | "
            f"{collision_state} | "
            f"{row.get('has_rigid_body_api', '')} | "
            f"`{row.get('bbox_center')}` | "
            f"`{row.get('bbox_size')}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect exact composed USD prims and their world-space bboxes.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--path", action="append", default=[], help="Prim path to inspect. Can be repeated.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "stage_prim_audit.json"
    md_path = output_dir / "stage_prim_audit.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {"stage_usd": _rel(args.stage_usd), "paths": list(args.path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)

        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd
        from pxr import UsdGeom

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        stage = stage_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        rows = [_prim_row(stage, cache, path) for path in args.path]
        payload.update({"status": "PASS", "rows": rows, "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)}})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "PASS", "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": repr(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "error": repr(exc)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())

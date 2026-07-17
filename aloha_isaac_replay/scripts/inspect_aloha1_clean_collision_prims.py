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
DEFAULT_STAGE_USD = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase37_clean_collision_prim_audit_20260718"


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
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _applied(prim: Any) -> list[str]:
    return [str(item) for item in prim.GetAppliedSchemas()]


def _has_schema(prim: Any, schema_name: str) -> bool:
    return schema_name in _applied(prim)


def _nearest_schema_ancestor(prim: Any, schema_name: str) -> str | None:
    current = prim.GetParent()
    while current and current.IsValid():
        if _has_schema(current, schema_name):
            return str(current.GetPath())
        current = current.GetParent()
    return None


def _bbox_row(cache: Any, prim: Any) -> dict[str, Any]:
    box = cache.ComputeWorldBound(prim).GetBox()
    if box.IsEmpty():
        return {
            "bbox_valid": False,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_center": None,
            "bbox_size": None,
            "bbox_max_dim": None,
        }
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    size = [float(max_pt[i] - min_pt[i]) for i in range(3)]
    if any(item < 0 for item in size):
        return {
            "bbox_valid": False,
            "bbox_min": [float(min_pt[i]) for i in range(3)],
            "bbox_max": [float(max_pt[i]) for i in range(3)],
            "bbox_center": None,
            "bbox_size": size,
            "bbox_max_dim": None,
        }
    center = [float((max_pt[i] + min_pt[i]) * 0.5) for i in range(3)]
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_center": center,
        "bbox_size": size,
        "bbox_max_dim": max(size) if size else 0.0,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 37 Clean Collision Prim Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- meters per unit: `{payload['stage']['meters_per_unit']}`",
        f"- up axis: `{payload['stage']['up_axis']}`",
        f"- collision prim count: `{payload['summary']['collision_prim_count']}`",
        f"- collision prims under articulation roots: `{payload['summary']['under_articulation_count']}`",
        f"- collision prims with RigidBodyAPI: `{payload['summary']['with_rigid_body_api_count']}`",
        f"- collision prims with rigid-body ancestor: `{payload['summary']['with_rigid_body_ancestor_count']}`",
        "",
        "## Collision Rows",
        "",
        "| path | under articulation | rigid body API | rigid ancestor | max dim |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in payload["collision_rows"]:
        lines.append(
            f"| `{row['path']}` | {row['under_articulation_root']} | {row['has_rigid_body_api']} | "
            f"`{row['rigid_body_ancestor'] or ''}` | {row['bbox_max_dim'] if row['bbox_max_dim'] is not None else 'invalid'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Collision prims that are outside the articulation roots and have no rigid-body ancestor are suspicious for robot self-collision geometry: PhysX may treat them as static world obstacles rather than robot-link collision shapes.",
            "Phase 36 showed that disabling all collision prims makes the one-joint dynamic gate pass, so these rows are the next repair target.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect clean ALOHA1 runtime stage collision prim hierarchy and bounding boxes.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "clean_collision_prim_audit.json"
    md_path = output_dir / "clean_collision_prim_audit.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {"stage_usd": _rel(args.stage_usd)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd, UsdGeom

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        stage = stage_utils.get_current_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        articulation_roots = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if _has_schema(prim, "ArticulationRootAPI") or _has_schema(prim, "PhysicsArticulationRootAPI")
        ]

        collision_rows: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            if not _has_schema(prim, "PhysicsCollisionAPI"):
                continue
            path = str(prim.GetPath())
            row = {
                "path": path,
                "type_name": prim.GetTypeName(),
                "applied_schemas": _applied(prim),
                "under_articulation_root": any(path.startswith(root + "/") or path == root for root in articulation_roots),
                "has_rigid_body_api": _has_schema(prim, "PhysicsRigidBodyAPI"),
                "rigid_body_ancestor": _nearest_schema_ancestor(prim, "PhysicsRigidBodyAPI"),
            }
            row.update(_bbox_row(bbox_cache, prim))
            collision_rows.append(row)

        summary = {
            "collision_prim_count": len(collision_rows),
            "under_articulation_count": sum(1 for row in collision_rows if row["under_articulation_root"]),
            "with_rigid_body_api_count": sum(1 for row in collision_rows if row["has_rigid_body_api"]),
            "with_rigid_body_ancestor_count": sum(1 for row in collision_rows if row["rigid_body_ancestor"]),
            "valid_bbox_count": sum(1 for row in collision_rows if row["bbox_valid"]),
            "suspicious_static_collision_count": sum(
                1
                for row in collision_rows
                if not row["under_articulation_root"] and not row["has_rigid_body_api"] and not row["rigid_body_ancestor"]
            ),
        }
        payload.update(
            {
                "status": "PASS",
                "stage": {
                    "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                    "articulation_roots": articulation_roots,
                },
                "summary": summary,
                "collision_rows": collision_rows,
                "outputs": {"json": _rel(json_path), "markdown": _rel(md_path)},
            }
        )
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)))
        print(json.dumps({"status": "PASS", "json": _rel(json_path), "markdown": _rel(md_path), "summary": summary}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())

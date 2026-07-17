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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase39_link_visual_proxy_candidates_20260718"


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
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_center": [float((max_pt[i] + min_pt[i]) * 0.5) for i in range(3)],
        "bbox_size": size,
        "bbox_max_dim": max(size) if size else 0.0,
    }


def _mesh_descendants(prim: Any) -> list[str]:
    from pxr import Usd

    meshes: list[str] = []
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        if child.GetTypeName() == "Mesh":
            meshes.append(str(child.GetPath()))
    return meshes


def _collision_descendants(prim: Any) -> list[str]:
    from pxr import Usd

    rows: list[str] = []
    for child in Usd.PrimRange(prim):
        if child == prim:
            continue
        if _has_schema(child, "PhysicsCollisionAPI"):
            rows.append(str(child.GetPath()))
    return rows


def _side_from_path(path: str) -> str:
    if "puppet_left" in path:
        return "left"
    if "puppet_right" in path:
        return "right"
    return "unknown"


def _robot_root_from_side(side: str) -> str | None:
    if side == "left":
        return "/puppet_left_vx300s"
    if side == "right":
        return "/puppet_right_vx300s"
    return None


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Phase 39 Link Visual Proxy Candidate Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- rigid body rows: `{summary['rigid_body_count']}`",
        f"- rows with valid bbox: `{summary['valid_bbox_count']}`",
        f"- rows with mesh descendants: `{summary['with_mesh_descendants_count']}`",
        f"- rows with link-owned collision descendants: `{summary['with_collision_descendants_count']}`",
        f"- mesh-owned proxy candidate rows: `{summary['mesh_owned_proxy_candidate_count']}`",
        f"- bbox-only proxy candidate rows: `{summary['bbox_proxy_candidate_count']}`",
        f"- articulation root API paths: `{summary['articulation_roots']}`",
        f"- robot roots used for ownership: `{summary['robot_roots']}`",
        "",
        "## Rows",
        "",
        "| side | rigid body | robot root | bbox valid | bbox max dim | mesh descendants | collision descendants | bbox proxy | mesh-owned proxy |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["rigid_body_rows"]:
        lines.append(
            f"| {row['side']} | `{row['path']}` | `{row['robot_root'] or ''}` | {row['bbox_valid']} | "
            f"{row['bbox_max_dim'] if row['bbox_max_dim'] is not None else 'invalid'} | "
            f"{len(row['mesh_descendants'])} | {len(row['collision_descendants'])} | "
            f"{row['bbox_proxy_candidate']} | {row['mesh_owned_proxy_candidate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current ALOHA1 clean stage does not expose Mesh prims directly under each rigid body, so mesh-owned collider generation is not available from simple link traversal.",
            "The Isaac importer applies ArticulationRootAPI on the side-specific `root_joint` prims, while rigid bodies are siblings under `/puppet_left_vx300s` and `/puppet_right_vx300s`. Candidate ownership is therefore checked against those robot roots, not by requiring rigid bodies to be descendants of `root_joint`.",
            "However, many rigid bodies have valid composed bounding boxes. That means a conservative bbox-only collision proxy can be generated as an approximation in a later phase.",
            "This audit does not create collision geometry. It separates high-confidence mesh-owned candidates from lower-confidence bbox-only candidates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ALOHA1 rigid bodies for link-owned visual geometry usable as simplified collision proxy candidates.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "link_visual_proxy_candidates.json"
    md_path = output_dir / "link_visual_proxy_candidates.md"
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

        rows: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            if not _has_schema(prim, "PhysicsRigidBodyAPI"):
                continue
            path = str(prim.GetPath())
            mesh_descendants = _mesh_descendants(prim)
            collision_descendants = _collision_descendants(prim)
            side = _side_from_path(path)
            robot_root = _robot_root_from_side(side)
            row: dict[str, Any] = {
                "path": path,
                "side": side,
                "robot_root": robot_root,
                "type_name": prim.GetTypeName(),
                "under_articulation_root": any(path.startswith(root + "/") or path == root for root in articulation_roots),
                "under_robot_root": bool(robot_root and (path.startswith(robot_root + "/") or path == robot_root)),
                "applied_schemas": _applied(prim),
                "mesh_descendants": mesh_descendants,
                "collision_descendants": collision_descendants,
            }
            row.update(_bbox_row(bbox_cache, prim))
            row["bbox_proxy_candidate"] = bool(
                row["under_robot_root"]
                and row["bbox_valid"]
                and not collision_descendants
            )
            row["mesh_owned_proxy_candidate"] = bool(
                row["under_robot_root"]
                and row["bbox_valid"]
                and mesh_descendants
                and not collision_descendants
            )
            rows.append(row)

        summary = {
            "rigid_body_count": len(rows),
            "valid_bbox_count": sum(1 for row in rows if row["bbox_valid"]),
            "with_mesh_descendants_count": sum(1 for row in rows if row["mesh_descendants"]),
            "with_collision_descendants_count": sum(1 for row in rows if row["collision_descendants"]),
            "bbox_proxy_candidate_count": sum(1 for row in rows if row["bbox_proxy_candidate"]),
            "mesh_owned_proxy_candidate_count": sum(1 for row in rows if row["mesh_owned_proxy_candidate"]),
            "articulation_roots": articulation_roots,
            "robot_roots": sorted({row["robot_root"] for row in rows if row["robot_root"]}),
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
                "rigid_body_rows": rows,
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

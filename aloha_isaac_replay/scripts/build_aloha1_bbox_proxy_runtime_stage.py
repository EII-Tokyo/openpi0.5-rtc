from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.validation.contact_proxy_profiles import contact_proxy_profile_names
from aloha_isaac_replay.validation.contact_proxy_profiles import proxy_path_for_rigid_body
from aloha_isaac_replay.validation.contact_proxy_profiles import robot_root_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import side_from_rigid_body_path
from aloha_isaac_replay.validation.contact_proxy_profiles import stage_units_in_meters_for_profile
from aloha_isaac_replay.validation.contact_proxy_profiles import stage_up_axis_for_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_clean_runtime.usda"
DEFAULT_OUTPUT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase40_bbox_proxy_runtime_build_20260718"


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


def _side_from_path(path: str, contact_proxy_profile: str = "legacy_puppet") -> str:
    return side_from_rigid_body_path(contact_proxy_profile, path)


def _robot_root_from_side(side: str, contact_proxy_profile: str = "legacy_puppet") -> str | None:
    return robot_root_for_side(contact_proxy_profile, side)


def _box_row(box: Any, bbox_scale: float, axis_scale: list[float] | None, min_extent: float) -> dict[str, Any]:
    if box.IsEmpty():
        return {
            "bbox_valid": False,
            "bbox_min": None,
            "bbox_max": None,
            "center": None,
            "size": None,
            "scaled_size": None,
        }
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    size = [float(max_pt[i] - min_pt[i]) for i in range(3)]
    if any(item <= 0 for item in size):
        return {
            "bbox_valid": False,
            "bbox_min": [float(min_pt[i]) for i in range(3)],
            "bbox_max": [float(max_pt[i]) for i in range(3)],
            "center": None,
            "size": size,
            "scaled_size": None,
        }
    center = [float((max_pt[i] + min_pt[i]) * 0.5) for i in range(3)]
    scale = axis_scale if axis_scale is not None else [bbox_scale, bbox_scale, bbox_scale]
    scaled_size = [max(float(item) * float(scale[index]), min_extent) for index, item in enumerate(size)]
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "center": center,
        "size": size,
        "scaled_size": scaled_size,
    }


def _matches_filters(path: str, include_regex: list[str], exclude_regex: list[str]) -> bool:
    included = True if not include_regex else any(re.search(pattern, path) for pattern in include_regex)
    excluded = any(re.search(pattern, path) for pattern in exclude_regex)
    return included and not excluded


def _collect_candidates(
    stage: Any,
    bbox_scale: float,
    axis_scale: list[float] | None,
    min_extent: float,
    include_regex: list[str],
    exclude_regex: list[str],
    contact_proxy_profile: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    from pxr import Usd
    from pxr import UsdGeom

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    disabled_root_collisions = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if _has_schema(prim, "PhysicsCollisionAPI") and str(prim.GetPath()).startswith("/colliders/")
    ]
    rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not _has_schema(prim, "PhysicsRigidBodyAPI"):
            continue
        path = str(prim.GetPath())
        side = _side_from_path(path, contact_proxy_profile)
        robot_root = _robot_root_from_side(side, contact_proxy_profile)
        under_robot_root = bool(robot_root and (path.startswith(robot_root + "/") or path == robot_root))
        local_box = bbox_cache.ComputeLocalBound(prim).GetBox()
        row: dict[str, Any] = {
            "path": path,
            "side": side,
            "robot_root": robot_root,
            "under_robot_root": under_robot_root,
            "filter_match": _matches_filters(path, include_regex=include_regex, exclude_regex=exclude_regex),
            "applied_schemas": _applied(prim),
            "proxy_path": proxy_path_for_rigid_body(contact_proxy_profile, path),
        }
        row.update(_box_row(local_box, bbox_scale=bbox_scale, axis_scale=axis_scale, min_extent=min_extent))
        row["selected"] = bool(under_robot_root and row["bbox_valid"] and row["filter_match"])
        rows.append(row)
    return rows, disabled_root_collisions


def _create_proxy_stage(
    input_stage_path: Path,
    output_stage_path: Path,
    candidates: list[dict[str, Any]],
    disabled_root_collisions: list[str],
    proxy_contact_offset: float | None,
    proxy_rest_offset: float | None,
    proxy_static_friction: float | None,
    proxy_dynamic_friction: float | None,
    proxy_restitution: float | None,
    contact_proxy_profile: str,
) -> None:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade

    output_stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_stage_path))
    root = stage.GetRootLayer()
    root.subLayerPaths.append(str(input_stage_path.resolve()))
    UsdGeom.SetStageMetersPerUnit(stage, stage_units_in_meters_for_profile(contact_proxy_profile))
    up_axis = stage_up_axis_for_profile(contact_proxy_profile)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    world = stage.GetPrimAtPath("/World")
    if not world:
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    for prim_path in disabled_root_collisions:
        prim = stage.OverridePrim(prim_path)
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(False)

    material = None
    if proxy_static_friction is not None or proxy_dynamic_friction is not None or proxy_restitution is not None:
        material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/FingertipProxyMaterial")
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        if proxy_static_friction is not None:
            material_api.CreateStaticFrictionAttr(float(proxy_static_friction))
        if proxy_dynamic_friction is not None:
            material_api.CreateDynamicFrictionAttr(float(proxy_dynamic_friction))
        if proxy_restitution is not None:
            material_api.CreateRestitutionAttr(float(proxy_restitution))

    for row in candidates:
        if not row["selected"]:
            continue
        proxy = UsdGeom.Cube.Define(stage, row["proxy_path"])
        proxy.CreateSizeAttr(1.0)
        proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
        proxy.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 1.0)])
        xform = UsdGeom.Xformable(proxy.GetPrim())
        xform.ClearXformOpOrder()
        translate = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        scale = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate.Set(Gf.Vec3d(*row["center"]))
        scale.Set(Gf.Vec3d(*row["scaled_size"]))
        collision = UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)
        if proxy_contact_offset is not None or proxy_rest_offset is not None:
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(proxy.GetPrim())
            if proxy_contact_offset is not None:
                physx_collision.CreateContactOffsetAttr(float(proxy_contact_offset))
            if proxy_rest_offset is not None:
                physx_collision.CreateRestOffsetAttr(float(proxy_rest_offset))
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(proxy.GetPrim()).Bind(material)

    stage.Save()


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Phase 40 Bbox Proxy Runtime Stage Build",
        "",
        f"- status: `{payload['status']}`",
        f"- input stage: `{payload['inputs']['stage_usd']}`",
        f"- output stage: `{payload['outputs']['stage_usd']}`",
        f"- bbox scale: `{payload['inputs']['bbox_scale']}`",
        f"- min extent: `{payload['inputs']['min_extent']}`",
        f"- include regex: `{payload['inputs']['include_regex']}`",
        f"- exclude regex: `{payload['inputs']['exclude_regex']}`",
        f"- disabled root collision prims: `{summary['disabled_root_collision_count']}`",
        f"- selected bbox proxies: `{summary['selected_proxy_count']}`",
        f"- skipped rigid bodies: `{summary['skipped_rigid_body_count']}`",
        "",
        "## Selected Proxies",
        "",
        "| side | rigid body | proxy | local center | scaled size |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["candidate_rows"]:
        if not row["selected"]:
            continue
        lines.append(
            f"| {row['side']} | `{row['path']}` | `{row['proxy_path']}` | "
            f"`{[round(x, 5) for x in row['center']]}` | `{[round(x, 5) for x in row['scaled_size']]}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This stage is an experimental collision-repair candidate. It disables the root-level imported `/colliders` layer and adds reduced-size box colliders under rigid-body links.",
            "The selected proxies are bbox-only approximations. They are intended for free-space stability gates first, not for final grasp/contact realism.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an experimental ALOHA1 runtime stage with link-owned bbox collision proxies."
    )
    parser.add_argument("--stage-usd", default=str(DEFAULT_INPUT_STAGE))
    parser.add_argument("--output-usd", default=str(DEFAULT_OUTPUT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bbox-scale", type=float, default=0.6)
    parser.add_argument(
        "--axis-scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional per-axis bbox scale. If omitted, --bbox-scale is used for all axes.",
    )
    parser.add_argument("--min-extent", type=float, default=0.005)
    parser.add_argument("--proxy-contact-offset", type=float, default=None)
    parser.add_argument("--proxy-rest-offset", type=float, default=None)
    parser.add_argument("--proxy-static-friction", type=float, default=None)
    parser.add_argument("--proxy-dynamic-friction", type=float, default=None)
    parser.add_argument("--proxy-restitution", type=float, default=None)
    parser.add_argument(
        "--contact-proxy-profile",
        choices=contact_proxy_profile_names(),
        default="legacy_puppet",
        help=(
            "Namespace profile used to select robot rigid bodies and author bbox_collision_proxy paths. "
            "Use scene_base_link with Trossen/Menagerie /scene stages."
        ),
    )
    parser.add_argument("--include-regex", action="append", default=[])
    parser.add_argument("--exclude-regex", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "bbox_proxy_runtime_build.json"
    md_path = output_dir / "bbox_proxy_runtime_build.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "bbox_scale": args.bbox_scale,
            "axis_scale": args.axis_scale,
            "min_extent": args.min_extent,
            "proxy_contact_offset": args.proxy_contact_offset,
            "proxy_rest_offset": args.proxy_rest_offset,
            "proxy_static_friction": args.proxy_static_friction,
            "proxy_dynamic_friction": args.proxy_dynamic_friction,
            "proxy_restitution": args.proxy_restitution,
            "contact_proxy_profile": args.contact_proxy_profile,
            "stage_units_in_meters": stage_units_in_meters_for_profile(args.contact_proxy_profile),
            "stage_up_axis": stage_up_axis_for_profile(args.contact_proxy_profile),
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
        },
        "outputs": {"stage_usd": _rel(args.output_usd), "json": _rel(json_path), "markdown": _rel(md_path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils

        input_stage_path = Path(args.stage_usd).resolve()
        output_stage_path = Path(args.output_usd).resolve()
        stage_utils.open_stage(str(input_stage_path))
        stage = stage_utils.get_current_stage()
        candidates, disabled_root_collisions = _collect_candidates(
            stage,
            bbox_scale=args.bbox_scale,
            axis_scale=args.axis_scale,
            min_extent=args.min_extent,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            contact_proxy_profile=args.contact_proxy_profile,
        )
        _create_proxy_stage(
            input_stage_path=input_stage_path,
            output_stage_path=output_stage_path,
            candidates=candidates,
            disabled_root_collisions=disabled_root_collisions,
            proxy_contact_offset=args.proxy_contact_offset,
            proxy_rest_offset=args.proxy_rest_offset,
            proxy_static_friction=args.proxy_static_friction,
            proxy_dynamic_friction=args.proxy_dynamic_friction,
            proxy_restitution=args.proxy_restitution,
            contact_proxy_profile=args.contact_proxy_profile,
        )
        summary = {
            "rigid_body_count": len(candidates),
            "selected_proxy_count": sum(1 for row in candidates if row["selected"]),
            "skipped_rigid_body_count": sum(1 for row in candidates if not row["selected"]),
            "disabled_root_collision_count": len(disabled_root_collisions),
        }
        payload.update({"status": "PASS", "summary": summary, "candidate_rows": candidates})
        _write_json(json_path, payload)
        md_path.write_text(_render_markdown(_json_safe(payload)))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "json": _rel(json_path),
                    "markdown": _rel(md_path),
                    "stage": _rel(output_stage_path),
                    "summary": summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
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
        print(
            json.dumps(
                {"status": "EXCEPTION", "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())

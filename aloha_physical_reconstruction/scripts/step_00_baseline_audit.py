from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_BOTTLE_USD = REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "aloha_physical_reconstruction"


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


def _attr_value(prim: Any, attr_name: str) -> Any:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return None
    return attr.Get()


def _authored_attrs(prim: Any, contains: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    needles = tuple(item.lower() for item in contains)
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if any(needle in name.lower() for needle in needles):
            try:
                out[name] = attr.Get()
            except Exception as exc:
                out[name] = f"<read failed: {exc}>"
    return out


def _bbox_row(cache: Any, prim: Any) -> dict[str, Any]:
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return {"bbox_valid": False, "bbox_min": None, "bbox_max": None, "bbox_center": None, "bbox_size": None}
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return {
        "bbox_valid": True,
        "bbox_min": [float(min_pt[i]) for i in range(3)],
        "bbox_max": [float(max_pt[i]) for i in range(3)],
        "bbox_center": [float((min_pt[i] + max_pt[i]) * 0.5) for i in range(3)],
        "bbox_size": [float(max_pt[i] - min_pt[i]) for i in range(3)],
    }


def _prim_summary(stage: Any, cache: Any, path: str) -> dict[str, Any]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    row: dict[str, Any] = {"path": path, "exists": bool(prim and prim.IsValid())}
    if not prim or not prim.IsValid():
        return row
    row.update(
        {
            "type_name": prim.GetTypeName(),
            "applied_schemas": _applied(prim),
            "is_instance": bool(prim.IsInstance()),
            "is_instanceable": bool(prim.IsInstanceable()),
            "xform_ops": [str(op.GetOpName()) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
            if prim.IsA(UsdGeom.Xformable)
            else [],
            "physics_attrs": _authored_attrs(prim, ("physics:", "physx", "mass", "inertia", "centerOfMass")),
        }
    )
    row.update(_bbox_row(cache, prim))
    return row


def _root_layers(stage: Any) -> dict[str, Any]:
    root = stage.GetRootLayer()
    session = stage.GetSessionLayer()
    return {
        "root_identifier": root.identifier,
        "root_real_path": getattr(root, "realPath", ""),
        "session_identifier": session.identifier if session else None,
        "subLayerPaths": list(root.subLayerPaths),
        "defaultPrim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "startTimeCode": stage.GetStartTimeCode(),
        "endTimeCode": stage.GetEndTimeCode(),
        "timeCodesPerSecond": stage.GetTimeCodesPerSecond(),
        "framesPerSecond": stage.GetFramesPerSecond(),
    }


def _stage_units(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    kilograms_per_unit = None
    try:
        from pxr import UsdPhysics

        kilograms_per_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    except Exception as exc:
        kilograms_per_unit = f"<unavailable: {exc}>"
    return {
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
        "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "kilogramsPerUnit": kilograms_per_unit,
    }


def _physics_scenes(stage: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        applied = _applied(prim)
        if type_name == "PhysicsScene" or "PhysxSceneAPI" in applied:
            rows.append(
                {
                    "path": str(prim.GetPath()),
                    "type_name": type_name,
                    "applied_schemas": applied,
                    "attrs": _authored_attrs(
                        prim,
                        (
                            "timeStepsPerSecond",
                            "physics:",
                            "physxScene:",
                            "gravity",
                            "solver",
                            "broadphase",
                        ),
                    ),
                }
            )
    return rows


def _find_references(stage: Any, limit: int = 80) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        stack = prim.GetPrimStack()
        for spec in stack:
            ref_list = getattr(spec, "referenceList", None)
            if not ref_list:
                continue
            items = list(ref_list.prependedItems) + list(ref_list.appendedItems) + list(ref_list.explicitItems)
            for ref in items:
                refs.append({"prim": str(prim.GetPath()), "assetPath": ref.assetPath, "primPath": str(ref.primPath)})
                if len(refs) >= limit:
                    return refs
    return refs


def _robot_audit(stage: Any, cache: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    articulations = []
    joints = []
    bodies = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = str(prim.GetTypeName())
        applied = _applied(prim)
        if "ArticulationRootAPI" in " ".join(applied) or "PhysicsArticulationRootAPI" in applied:
            articulations.append(_prim_summary(stage, cache, path))
        if type_name.endswith("Joint") or "Joint" in type_name:
            joints.append(
                {
                    "path": path,
                    "type_name": type_name,
                    "applied_schemas": applied,
                    "drive_attrs": _authored_attrs(prim, ("drive", "limit", "axis", "physics:")),
                }
            )
        if path.endswith("_link") or "gripper" in path.lower() or "finger" in path.lower():
            if prim.IsA(UsdGeom.Boundable) or applied:
                bodies.append(_prim_summary(stage, cache, path))
    left_joints = [row for row in joints if "/left_" in row["path"] or row["path"].startswith("/scene/left")]
    left_arm_joints = [
        row
        for row in left_joints
        if not any(token in row["path"].lower() for token in ("finger", "gripper"))
    ]
    left_gripper_joints = [
        row
        for row in left_joints
        if any(token in row["path"].lower() for token in ("finger", "gripper"))
    ]
    return {
        "articulation_roots": articulations,
        "base_link_candidates": [
            _prim_summary(stage, cache, path)
            for path in ("/scene/left_base_link", "/scene/left_base_link/left_base_link", "/scene/right_base_link", "/scene/right_base_link/right_base_link")
        ],
        "left_arm_dof_candidates": left_arm_joints,
        "left_gripper_dof_candidates": left_gripper_joints,
        "joint_count_total": len(joints),
        "body_candidates_sample": bodies[:80],
    }


def _bottle_audit(stage: Any, cache: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lowered = path.lower()
        if "bottle" not in lowered:
            continue
        rows.append(_prim_summary(stage, cache, path))
    return {"bottle_named_prims": rows, "bottle_named_prim_count": len(rows)}


def _open_stage(stage_utils: Any, usd_path: Path) -> Any:
    import omni.kit.app

    stage_utils.open_stage(str(usd_path.resolve()))
    for _ in range(10):
        omni.kit.app.get_app().update()
    return stage_utils.get_current_stage()


def _make_camera(stage: Any, cache: Any, target_prim_path: str, camera_path: str, distance_scale: float = 2.2) -> str | None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(target_prim_path)
    if not prim or not prim.IsValid():
        return None
    bbox = _bbox_row(cache, prim)
    if not bbox["bbox_valid"]:
        return None
    center = Gf.Vec3d(*bbox["bbox_center"])
    size = bbox["bbox_size"]
    radius = max(size) if size else 1.0
    eye = center + Gf.Vec3d(-distance_scale * radius, distance_scale * radius, max(0.35, 0.8 * radius))
    cam = UsdGeom.Camera.Define(stage, camera_path)
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(eye)
    direction = (center - eye).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(direction, up).GetNormalized()
    true_up = Gf.Cross(right, direction).GetNormalized()
    # USD camera looks down -Z with +Y up.
    rot = Gf.Matrix3d(
        right[0], right[1], right[2],
        true_up[0], true_up[1], true_up[2],
        -direction[0], -direction[1], -direction[2],
    ).GetTranspose()
    xform.AddOrientOp().Set(Gf.Quatd(rot))
    cam.GetFocalLengthAttr().Set(28.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return camera_path


async def _capture_next_frame(path: Path, camera_path: str | None) -> str:
    import omni.kit.app
    from pxr import Sdf

    app = omni.kit.app.get_app()
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport and camera_path:
            viewport.camera_path = Sdf.Path(camera_path)
    except Exception:
        pass
    for _ in range(25):
        await app.next_update_async()
    try:
        import omni.kit.viewport_legacy
        import omni.renderer_capture

        renderer = omni.renderer_capture.acquire_renderer_capture_interface()
        viewport_interface = omni.kit.viewport_legacy.acquire_viewport_interface()
        viewport_window = viewport_interface.get_viewport_window(None)
        resource = viewport_window.get_drawable_ldr_resource() if viewport_window else None
        for _ in range(80):
            if resource is not None:
                break
            await app.next_update_async()
            resource = viewport_window.get_drawable_ldr_resource() if viewport_window else None
        if resource is None:
            return "FAILED: viewport LDR resource unavailable"
        renderer.capture_next_frame_rp_resource(str(path), resource)
        for _ in range(120):
            await app.next_update_async()
            if path.exists() and path.stat().st_size > 0:
                return "PASS"
        return "PASS" if path.exists() and path.stat().st_size > 0 else "FAILED: output file missing or empty"
    except Exception as exc:
        return f"FAILED: {exc}"


def _draw_audit_visualization(path: Path, title: str, rows: list[dict[str, Any]], subtitle: str) -> str:
    """Write a static PNG from audited USD bbox data without saving the USD stage."""
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1400, 900), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        title_font = body_font = small_font = None

    draw.text((36, 28), title, fill=(24, 32, 40), font=title_font)
    draw.text((36, 78), subtitle, fill=(74, 85, 104), font=body_font)
    valid = [row for row in rows if row.get("bbox_valid")]
    if not valid:
        draw.text((36, 150), "No valid bounding boxes found.", fill=(160, 40, 40), font=body_font)
        image.save(path)
        return "PASS: static audit visualization generated; no valid bbox rows"

    mins = [row["bbox_min"] for row in valid]
    maxs = [row["bbox_max"] for row in valid]
    min_x = min(float(item[0]) for item in mins)
    max_x = max(float(item[0]) for item in maxs)
    min_y = min(float(item[1]) for item in mins)
    max_y = max(float(item[1]) for item in maxs)
    if math.isclose(max_x, min_x):
        min_x -= 0.5
        max_x += 0.5
    if math.isclose(max_y, min_y):
        min_y -= 0.5
        max_y += 0.5

    plot = (80, 145, 980, 825)
    left, top, right, bottom = plot
    draw.rectangle(plot, outline=(180, 186, 196), width=2)
    draw.text((left, bottom + 16), "Top-down bbox view: X horizontal, Y vertical", fill=(74, 85, 104), font=small_font)

    colors = [
        (45, 118, 191),
        (203, 74, 62),
        (54, 151, 92),
        (154, 94, 188),
        (235, 150, 45),
        (73, 85, 99),
        (16, 130, 140),
    ]

    def project(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - min_x) / (max_x - min_x) * (right - left))
        py = bottom - int((y - min_y) / (max_y - min_y) * (bottom - top))
        return px, py

    legend_y = 150
    for idx, row in enumerate(valid[:24]):
        color = colors[idx % len(colors)]
        x0, y0 = project(float(row["bbox_min"][0]), float(row["bbox_min"][1]))
        x1, y1 = project(float(row["bbox_max"][0]), float(row["bbox_max"][1]))
        rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        draw.rectangle(rect, outline=color, width=3)
        cx, cy = project(float(row["bbox_center"][0]), float(row["bbox_center"][1]))
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
        label_box = (cx + 6, cy - 15, cx + 34, cy + 13)
        draw.rectangle(label_box, fill=(248, 248, 244), outline=color)
        draw.text((cx + 12, cy - 13), str(idx + 1), fill=color, font=small_font)

        lx = 1030
        draw.rectangle((lx, legend_y + 3, lx + 20, legend_y + 23), fill=color)
        draw.text((lx + 30, legend_y), f"{idx + 1}. {row.get('path', '')[:44]}", fill=(32, 41, 51), font=small_font)
        size = row.get("bbox_size")
        if size:
            draw.text((lx + 30, legend_y + 22), f"size m: [{size[0]:.3g}, {size[1]:.3g}, {size[2]:.3g}]", fill=(94, 103, 117), font=small_font)
        legend_y += 58
        if legend_y > 800:
            break

    image.save(path)
    return "PASS: static audit visualization generated from USD bbox data"


def _render_markdown(payload: dict[str, Any]) -> str:
    main = payload["main_scene"]
    bottle_asset = payload["bottle_asset"]
    lines = [
        "# Step 00 Baseline Audit",
        "",
        "本报告只审计当前 ALOHA 基础环境，不修改物理参数，不创建新瓶子，不运行抓取。",
        "",
        "## Scope",
        "",
        f"- canonical main scene USD: `{main['path']}`",
        f"- existing bottle USD: `{bottle_asset['path']}`",
        f"- real robot touched: `{payload['real_robot_touched']}`",
        f"- stage saved: `{payload['stage_saved']}`",
        "",
        "## Stage Units And Timing",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| upAxis | `{main.get('units', {}).get('upAxis')}` |",
        f"| metersPerUnit | `{main.get('units', {}).get('metersPerUnit')}` |",
        f"| kilogramsPerUnit | `{main.get('units', {}).get('kilogramsPerUnit')}` |",
        f"| timeCodesPerSecond | `{main.get('layers', {}).get('timeCodesPerSecond')}` |",
        f"| framesPerSecond | `{main.get('layers', {}).get('framesPerSecond')}` |",
        "",
        "## PhysicsScene",
        "",
    ]
    if main.get("physics_scenes"):
        lines.extend(["| path | attrs |", "| --- | --- |"])
        for row in main["physics_scenes"]:
            lines.append(f"| `{row['path']}` | `{row.get('attrs')}` |")
    else:
        lines.append("- No PhysicsScene prim found in the composed stage.")
    robot = main.get("robot", {})
    lines.extend(
        [
            "",
            "## ALOHA Robot Audit",
            "",
            f"- articulation root count: `{len(robot.get('articulation_roots', []))}`",
            f"- joint count total: `{robot.get('joint_count_total')}`",
            f"- left arm DOF candidate count: `{len(robot.get('left_arm_dof_candidates', []))}`",
            f"- left gripper DOF candidate count: `{len(robot.get('left_gripper_dof_candidates', []))}`",
            "",
            "### Articulation Roots",
            "",
            "| path | type | bbox size | schemas |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in robot.get("articulation_roots", []):
        lines.append(f"| `{row['path']}` | `{row.get('type_name')}` | `{row.get('bbox_size')}` | `{row.get('applied_schemas')}` |")
    lines.extend(["", "### Left DOF Candidates", "", "| category | path | type | attrs |", "| --- | --- | --- | --- |"])
    for row in robot.get("left_arm_dof_candidates", []):
        lines.append(f"| arm | `{row['path']}` | `{row.get('type_name')}` | `{row.get('drive_attrs')}` |")
    for row in robot.get("left_gripper_dof_candidates", []):
        lines.append(f"| gripper | `{row['path']}` | `{row.get('type_name')}` | `{row.get('drive_attrs')}` |")
    lines.extend(["", "## Bottle In Current Main Scene", ""])
    main_bottle = main.get("bottle", {})
    lines.append(f"- bottle-named prim count in main scene: `{main_bottle.get('bottle_named_prim_count')}`")
    if main_bottle.get("bottle_named_prims"):
        lines.extend(["", "| path | type | bbox size | schemas |", "| --- | --- | --- | --- |"])
        for row in main_bottle["bottle_named_prims"]:
            lines.append(f"| `{row['path']}` | `{row.get('type_name')}` | `{row.get('bbox_size')}` | `{row.get('applied_schemas')}` |")
    else:
        lines.append("- 当前 canonical main scene 中未发现名称包含 `bottle` 的 Prim。")
    lines.extend(["", "## Existing Bottle Asset Audit", "", "| path | type | bbox size | physics attrs | schemas |", "| --- | --- | --- | --- | --- |"])
    for row in bottle_asset.get("bottle", {}).get("bottle_named_prims", []):
        lines.append(
            f"| `{row['path']}` | `{row.get('type_name')}` | `{row.get('bbox_size')}` | "
            f"`{row.get('physics_attrs')}` | `{row.get('applied_schemas')}` |"
        )
    lines.extend(["", "## Screenshots", "", "| screenshot | status | path |", "| --- | --- | --- |"])
    for name, row in payload.get("screenshots", {}).items():
        lines.append(f"| `{name}` | `{row.get('status')}` | `{row.get('path')}` |")
    lines.extend(
        [
            "",
            "## Step 00 Conclusion",
            "",
            "- 本步没有修改 stage、物理参数、瓶子、碰撞体、质量或机器人控制。",
            "- 当前找到的 canonical ALOHA 场景来自项目 AGENTS 中确认的启动 USD。",
            "- 如果你确认这是正确 ALOHA 场景，下一步才进入真实瓶子测量。",
            "",
            "HUMAN CONFIRMATION REQUIRED",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 00 read-only baseline audit for ALOHA physical reconstruction.")
    parser.add_argument("--stage-usd", type=Path, default=DEFAULT_STAGE_USD)
    parser.add_argument("--bottle-usd", type=Path, default=DEFAULT_BOTTLE_USD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    out_root = args.output_root
    reports = out_root / "reports"
    screenshots = out_root / "artifacts" / "screenshots"
    raw = out_root / "artifacts" / "raw"
    for path in (reports, screenshots, raw):
        path.mkdir(parents=True, exist_ok=True)

    report_md = reports / "step_00_baseline_audit.md"
    report_json = raw / "step_00_baseline_audit.json"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "real_robot_touched": False,
        "stage_saved": False,
        "main_scene": {"path": _rel(args.stage_usd), "exists": args.stage_usd.exists()},
        "bottle_asset": {"path": _rel(args.bottle_usd), "exists": args.bottle_usd.exists()},
        "screenshots": {},
    }
    _write_json(report_json, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config.update({"headless": bool(args.headless), "disable_viewport_updates": False, "width": 1280, "height": 720})
        app = SimulationApp(app_config)

        import isaacsim.core.utils.stage as stage_utils
        import omni.kit.app
        from pxr import Usd
        from pxr import UsdGeom

        stage = _open_stage(stage_utils, args.stage_usd)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        payload["main_scene"].update(
            {
                "layers": _root_layers(stage),
                "units": _stage_units(stage),
                "physics_scenes": _physics_scenes(stage),
                "references": _find_references(stage),
                "robot": _robot_audit(stage, cache),
                "bottle": _bottle_audit(stage, cache),
                "stage_overview_paths": [
                    _prim_summary(stage, cache, path)
                    for path in (
                        "/scene",
                        "/scene/worldBody",
                        "/scene/left_base_link",
                        "/scene/right_base_link",
                        "/scene/StartupViewCamera",
                    )
                ],
            }
        )
        overview_path = screenshots / "step_00_stage_overview.png"
        overview_rows = (
            payload["main_scene"]["stage_overview_paths"]
            + payload["main_scene"]["robot"].get("articulation_roots", [])
            + payload["main_scene"]["robot"].get("base_link_candidates", [])
        )
        status = _draw_audit_visualization(
            overview_path,
            "Step 00: Current ALOHA Stage Overview",
            overview_rows,
            "Static audit visualization generated from existing USD bounding boxes; no stage edits.",
        )
        payload["screenshots"]["stage_overview"] = {"path": _rel(overview_path), "status": status}

        bottle_stage = _open_stage(stage_utils, args.bottle_usd)
        bottle_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        bottle_default = bottle_stage.GetDefaultPrim()
        bottle_root = str(bottle_default.GetPath()) if bottle_default else "/"
        payload["bottle_asset"].update(
            {
                "layers": _root_layers(bottle_stage),
                "units": _stage_units(bottle_stage),
                "physics_scenes": _physics_scenes(bottle_stage),
                "bottle": _bottle_audit(bottle_stage, bottle_cache),
                "root_summary": _prim_summary(bottle_stage, bottle_cache, bottle_root),
            }
        )
        bottle_path = screenshots / "step_00_bottle_current.png"
        bottle_rows = [payload["bottle_asset"]["root_summary"]] + payload["bottle_asset"].get("bottle", {}).get("bottle_named_prims", [])
        bottle_status = _draw_audit_visualization(
            bottle_path,
            "Step 00: Existing Bottle Asset",
            bottle_rows,
            "Static audit visualization generated from current bottle USD bounding boxes; no bottle edits.",
        )
        payload["screenshots"]["bottle_current"] = {"path": _rel(bottle_path), "status": bottle_status}

        payload["status"] = "PASS"
        _write_json(report_json, payload)
        report_md.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "PASS", "report": _rel(report_md), "json": _rel(report_json)}, ensure_ascii=False), flush=True)
        app.close(skip_cleanup=True)
        return 0
    except BaseException as exc:
        payload.update({"status": "EXCEPTION", "exception": repr(exc), "traceback": traceback.format_exc(limit=30)})
        _write_json(report_json, payload)
        report_md.write_text(_render_markdown(_json_safe(payload)), encoding="utf-8")
        print(json.dumps({"status": "EXCEPTION", "report": _rel(report_md), "json": _rel(report_json), "error": repr(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

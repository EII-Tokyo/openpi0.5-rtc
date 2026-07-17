from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase2_runtime_inspection_20260717"
DEFAULT_ALOHA1_LEFT_USD = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/vx300s_left.usd"
DEFAULT_ALOHA1_RIGHT_USD = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/vx300s_right.usd"
DEFAULT_ALOHA1_WRAPPER_USD = REPO_ROOT / "assets/isaac/original_stationary_aloha/generated/original_stationary_aloha.usd"
DEFAULT_TROSSEN_USD = REPO_ROOT / "external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__iter__"):
        try:
            return [_json_safe(item) for item in value]
        except Exception:
            pass
    return str(value)


def _call_optional(obj: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)(*args, **kwargs)
            except Exception:
                continue
    return None


def _as_flat_optional(value: Any, n: int | None = None) -> list[float | None]:
    if value is None:
        return [] if n is None else [None] * n
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return [] if n is None else [None] * n
    arr = np.squeeze(arr)
    vals = [float(arr)] if arr.ndim == 0 else [float(x) for x in arr.reshape(-1).tolist()]
    if n is not None:
        vals = vals[:n] + [None] * max(0, n - len(vals))
    return vals


def _get_limits(art: Any) -> list[list[float | None]]:
    n = int(art.num_dof)
    view = art._articulation_view
    value = _call_optional(view, ("get_dof_limits", "get_joint_limits"))
    if value is None and hasattr(view, "_physics_view"):
        value = _call_optional(view._physics_view, ("get_dof_limits", "get_joint_limits"))
    if value is None:
        return [[None, None] for _ in range(n)]
    arr = np.squeeze(np.asarray(value, dtype=np.float64))
    if arr.shape == (n, 2):
        return [[float(x), float(y)] for x, y in arr.tolist()]
    if arr.shape == (2, n):
        return [[float(x), float(y)] for x, y in arr.T.tolist()]
    arr = arr.reshape((-1, 2))
    rows = [[float(x), float(y)] for x, y in arr[:n].tolist()]
    rows.extend([[None, None]] * max(0, n - len(rows)))
    return rows


def _get_gains(art: Any) -> tuple[list[float | None], list[float | None]]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_gains",))
    if isinstance(value, tuple) and len(value) >= 2:
        return _as_flat_optional(value[0], n), _as_flat_optional(value[1], n)
    return [None] * n, [None] * n


def _get_max_efforts(art: Any) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_efforts", "get_dof_max_efforts"))
    if value is None and hasattr(art._articulation_view, "_physics_view"):
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_forces", "get_dof_max_efforts"))
    return _as_flat_optional(value, n)


def _get_max_velocities(art: Any) -> list[float | None]:
    n = int(art.num_dof)
    value = _call_optional(art._articulation_view, ("get_max_velocities", "get_dof_max_velocities"))
    if value is None and hasattr(art._articulation_view, "_physics_view"):
        value = _call_optional(art._articulation_view._physics_view, ("get_dof_max_velocities",))
    return _as_flat_optional(value, n)


def _is_collision_prim(prim: Any) -> bool:
    from pxr import UsdPhysics

    try:
        return bool(UsdPhysics.CollisionAPI(prim))
    except Exception:
        return False


def _is_rigid_body_prim(prim: Any) -> bool:
    from pxr import UsdPhysics

    try:
        return bool(UsdPhysics.RigidBodyAPI(prim))
    except Exception:
        return False


def _world_transform(prim: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    try:
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        translation = matrix.ExtractTranslation()
        return {
            "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
            "matrix": [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _stage_static_summary(stage: Any, root_prefix: str) -> dict[str, Any]:
    from pxr import UsdGeom

    prims = []
    joints = []
    cameras = []
    colliders = []
    rigid_bodies = []
    meshes = []
    materials = []
    articulation_roots = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if root_prefix and not path.startswith(root_prefix):
            continue
        type_name = prim.GetTypeName()
        applied = [str(item) for item in prim.GetAppliedSchemas()]
        if len(prims) < 300:
            prims.append({"path": path, "type": type_name, "applied_schemas": applied})
        if "ArticulationRootAPI" in applied or "PhysicsArticulationRootAPI" in applied:
            articulation_roots.append(path)
        if "Joint" in type_name:
            joints.append(_joint_prim_info(prim))
        if type_name == "Camera":
            cameras.append(_camera_info(prim))
        if type_name == "Mesh":
            meshes.append({"path": path, "world_transform": _world_transform(prim)})
        if type_name == "Material":
            materials.append({"path": path})
        if _is_collision_prim(prim):
            colliders.append(
                {
                    "path": path,
                    "type": type_name,
                    "applied_schemas": applied,
                    "world_transform": _world_transform(prim),
                }
            )
        if _is_rigid_body_prim(prim):
            rigid_bodies.append({"path": path, "type": type_name, "applied_schemas": applied})

    return {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim().IsValid() else None,
        "prim_count_under_root": len([1 for prim in stage.Traverse() if not root_prefix or str(prim.GetPath()).startswith(root_prefix)]),
        "prim_tree_sample": prims,
        "articulation_root_candidates": articulation_roots,
        "joint_count": len(joints),
        "joints": joints,
        "camera_count": len(cameras),
        "cameras": cameras,
        "collider_count": len(colliders),
        "colliders_sample": colliders[:120],
        "rigid_body_count": len(rigid_bodies),
        "rigid_bodies_sample": rigid_bodies[:120],
        "mesh_count": len(meshes),
        "meshes_sample": meshes[:120],
        "material_count": len(materials),
        "materials_sample": materials[:120],
    }


def _joint_prim_info(prim: Any) -> dict[str, Any]:
    from pxr import UsdPhysics

    type_name = prim.GetTypeName()
    info: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type": type_name,
        "applied_schemas": [str(item) for item in prim.GetAppliedSchemas()],
    }
    try:
        joint = UsdPhysics.Joint(prim)
        info["body0"] = [str(x) for x in joint.GetBody0Rel().GetTargets()]
        info["body1"] = [str(x) for x in joint.GetBody1Rel().GetTargets()]
    except Exception:
        info["body0"] = []
        info["body1"] = []
    try:
        if type_name == "PhysicsRevoluteJoint":
            joint = UsdPhysics.RevoluteJoint(prim)
            info["axis"] = str(joint.GetAxisAttr().Get())
            info["lower"] = joint.GetLowerLimitAttr().Get()
            info["upper"] = joint.GetUpperLimitAttr().Get()
        elif type_name == "PhysicsPrismaticJoint":
            joint = UsdPhysics.PrismaticJoint(prim)
            info["axis"] = str(joint.GetAxisAttr().Get())
            info["lower"] = joint.GetLowerLimitAttr().Get()
            info["upper"] = joint.GetUpperLimitAttr().Get()
    except Exception as exc:
        info["limit_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _camera_info(prim: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    camera = UsdGeom.Camera(prim)
    attrs = {}
    for key, getter in {
        "focal_length": camera.GetFocalLengthAttr,
        "horizontal_aperture": camera.GetHorizontalApertureAttr,
        "vertical_aperture": camera.GetVerticalApertureAttr,
        "clipping_range": camera.GetClippingRangeAttr,
    }.items():
        try:
            value = getter().Get()
            attrs[key] = list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else value
        except Exception:
            attrs[key] = None
    return {
        "path": str(prim.GetPath()),
        "attributes": attrs,
        "world_transform": _world_transform(prim),
        "frame_convention_note": "USD camera convention; compare carefully with Isaac/ROS optical frames before using for real image alignment.",
    }


def _inspect_articulation(world: Any, prim_path: str, name: str) -> dict[str, Any]:
    from isaacsim.core.prims import SingleArticulation

    art = world.scene.add(SingleArticulation(prim_path=prim_path, name=name))
    world.reset()
    n = int(art.num_dof)
    limits = _get_limits(art)
    kps, kds = _get_gains(art)
    efforts = _get_max_efforts(art)
    velocities = _get_max_velocities(art)
    qpos = _as_flat_optional(art.get_joint_positions(), n)
    qvel = _as_flat_optional(art.get_joint_velocities(), n)
    dofs = []
    for idx, dof_name in enumerate(list(art.dof_names)):
        lower, upper = limits[idx] if idx < len(limits) else (None, None)
        dofs.append(
            {
                "index": idx,
                "name": dof_name,
                "runtime_lower": lower,
                "runtime_upper": upper,
                "stiffness": kps[idx],
                "damping": kds[idx],
                "effort_limit": efforts[idx],
                "velocity_limit": velocities[idx],
                "current_qpos": qpos[idx],
                "current_qvel": qvel[idx],
                "semantic_flags": {
                    "arm_candidate": any(token in dof_name for token in ("waist", "shoulder", "elbow", "forearm", "wrist", "joint_")),
                    "gripper_candidate": any(token in dof_name.lower() for token in ("gripper", "finger", "carriage")),
                },
            }
        )
    view = art._articulation_view
    return {
        "status": "PASS",
        "prim_path": prim_path,
        "num_dof": n,
        "num_bodies": int(art.num_bodies),
        "dof_names": list(art.dof_names),
        "body_names": list(view.body_names),
        "ee_body_candidates": [name for name in view.body_names if "ee" in name or "gripper" in name or "link_6" in name],
        "dofs": dofs,
    }


def _safe_inspect_articulations(world: Any, candidates: list[str], name_prefix: str) -> list[dict[str, Any]]:
    results = []
    used = set()
    for index, prim_path in enumerate(candidates):
        if prim_path in used:
            continue
        used.add(prim_path)
        try:
            results.append(_inspect_articulation(world, prim_path, f"{name_prefix}_{index}"))
        except Exception as exc:
            results.append({"status": "FAIL", "prim_path": prim_path, "error": f"{type(exc).__name__}: {exc}"})
    return results


def _inspect_reference_asset(stage_utils: Any, World: Any, *, label: str, usd_path: Path, root_prim: str, known_articulation_roots: list[str]) -> dict[str, Any]:
    stage_utils.create_new_stage()
    world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
    stage_utils.add_reference_to_stage(usd_path=str(usd_path.resolve()), prim_path=root_prim)
    stage = world.stage
    static = _stage_static_summary(stage, root_prim)
    candidates = known_articulation_roots or static["articulation_root_candidates"]
    runtime = _safe_inspect_articulations(world, candidates, label)
    return {
        "label": label,
        "usd_path": _rel(usd_path),
        "root_prim": root_prim,
        "known_articulation_roots": known_articulation_roots,
        "stage_static": static,
        "runtime_articulations": runtime,
    }


def _failed_asset(label: str, usd_path: Path, root_prim: str, exc: BaseException) -> dict[str, Any]:
    return {
        "label": label,
        "usd_path": _rel(usd_path),
        "root_prim": root_prim,
        "known_articulation_roots": [],
        "stage_static": {
            "meters_per_unit": None,
            "up_axis": None,
            "default_prim": None,
            "prim_count_under_root": 0,
            "prim_tree_sample": [],
            "articulation_root_candidates": [],
            "joint_count": 0,
            "joints": [],
            "camera_count": 0,
            "cameras": [],
            "collider_count": 0,
            "colliders_sample": [],
            "rigid_body_count": 0,
            "rigid_bodies_sample": [],
            "mesh_count": 0,
            "meshes_sample": [],
            "material_count": 0,
            "materials_sample": [],
        },
        "runtime_articulations": [],
        "status": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback_tail": traceback.format_exc(limit=8),
    }


def _write_report_files(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase2_runtime_inspection.json"
    md_path = output_dir / "phase2_runtime_inspection.md"
    safe_payload = _json_safe(payload)
    json_path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(build_report(safe_payload))
    return json_path, md_path


def build_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Isaac Runtime Asset Inspection",
        "",
        f"- Isaac runtime started: `{payload['isaac_runtime_started']}`",
        f"- Real robot touched: `{payload['real_robot_touched']}`",
        f"- Stage saved: `{payload['stage_saved']}`",
        "",
        "## Assets",
        "",
    ]
    for label, asset in payload["assets"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- USD: `{asset['usd_path']}`",
                f"- Root prim: `{asset['root_prim']}`",
                f"- asset status: `{asset.get('status', 'PASS')}`",
                f"- error: `{asset.get('error', '')}`",
                f"- metersPerUnit: `{asset['stage_static']['meters_per_unit']}`",
                f"- upAxis: `{asset['stage_static']['up_axis']}`",
                f"- articulation candidates: `{asset['stage_static']['articulation_root_candidates']}`",
                f"- joint count: `{asset['stage_static']['joint_count']}`",
                f"- collider count: `{asset['stage_static']['collider_count']}`",
                f"- mesh count: `{asset['stage_static']['mesh_count']}`",
                f"- camera count: `{asset['stage_static']['camera_count']}`",
                "",
                "Runtime articulations:",
            ]
        )
        for art in asset["runtime_articulations"]:
            lines.append(
                f"- `{art.get('prim_path')}` status `{art.get('status')}`, "
                f"num_dof `{art.get('num_dof')}`, dofs `{art.get('dof_names')}`"
            )
        lines.append("")
    lines.extend(["## Gates", ""])
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Isaac runtime/articulation inspection for ALOHA1 adaptation.")
    parser.add_argument("--aloha1-left-usd", default=str(DEFAULT_ALOHA1_LEFT_USD))
    parser.add_argument("--aloha1-right-usd", default=str(DEFAULT_ALOHA1_RIGHT_USD))
    parser.add_argument("--aloha1-wrapper-usd", default=str(DEFAULT_ALOHA1_WRAPPER_USD))
    parser.add_argument("--trossen-usd", default=str(DEFAULT_TROSSEN_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--normal-close",
        action="store_true",
        help="Call SimulationApp.close() before exit. Disabled by default because Isaac 5.1 headless teardown can hang or segfault after diagnostics are already flushed.",
    )
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    # Prefer a normal close for diagnostic scripts. Fast shutdown can hide Python
    # exceptions before reports are flushed when Kit terminates aggressively.
    app_config["fast_shutdown"] = False
    app = SimulationApp(app_config)
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World

        World.clear_instance()
        assets: dict[str, Any] = {}
        output_dir = Path(args.output_dir)

        def inspect_one(label: str, usd_path: str, root_prim: str, known_roots: list[str]) -> None:
            print(f"[phase2] inspecting {label}: {usd_path}", flush=True)
            World.clear_instance()
            try:
                assets[label] = _inspect_reference_asset(
                    stage_utils,
                    World,
                    label=label,
                    usd_path=Path(usd_path),
                    root_prim=root_prim,
                    known_articulation_roots=known_roots,
                )
                assets[label]["status"] = "PASS"
            except BaseException as exc:
                assets[label] = _failed_asset(label, Path(usd_path), root_prim, exc)
            # Persist partial evidence after every asset so a later Kit/runtime
            # failure does not erase the completed inspection evidence.
            partial_payload = {
                "isaac_runtime_started": True,
                "real_robot_touched": False,
                "stage_saved": False,
                "process_pid": os.getpid(),
                "assets": assets,
                "gates": {"partial": "IN_PROGRESS"},
            }
            _write_report_files(output_dir, partial_payload)

        inspect_one(
            "aloha1_left_side",
            args.aloha1_left_usd,
            "/World/aloha1_left",
            ["/World/aloha1_left/root_joint/root_joint"],
        )
        inspect_one(
            "aloha1_right_side",
            args.aloha1_right_usd,
            "/World/aloha1_right",
            ["/World/aloha1_right/root_joint/root_joint"],
        )
        inspect_one(
            "aloha1_wrapper",
            args.aloha1_wrapper_usd,
            "/World/aloha1_wrapper",
            [],
        )
        inspect_one(
            "trossen_stationary_ai",
            args.trossen_usd,
            "/World/trossen_stationary_ai",
            [],
        )

        gates = {
            "runtime_started": "PASS",
            "real_robot_touched": "PASS_FALSE",
            "stage_saved": "PASS_FALSE",
            "aloha1_side_dof_runtime": "PASS"
            if all(
                art.get("status") == "PASS"
                for key in ("aloha1_left_side", "aloha1_right_side")
                for art in assets[key]["runtime_articulations"]
            )
            else "FAIL",
            "trossen_runtime_dof_order": "PASS"
            if any(art.get("status") == "PASS" for art in assets["trossen_stationary_ai"]["runtime_articulations"])
            else "BLOCKED_RUNTIME_ARTICULATION_NOT_INITIALIZED",
            "contact_rl": "BLOCKED_UNTIL_COLLIDER_AND_MATERIAL_REVIEW",
            "camera_validation": "BLOCKED_UNTIL_EXTRINSIC_PROJECTION_TEST",
            "gripper_mapping": "BLOCKED_UNTIL_OPEN_CLOSE_CALIBRATION",
        }
        payload = {
            "isaac_runtime_started": True,
            "real_robot_touched": False,
            "stage_saved": False,
            "process_pid": os.getpid(),
            "assets": assets,
            "gates": gates,
        }
        json_path, md_path = _write_report_files(output_dir, payload)
        print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "gates": gates}, ensure_ascii=False, indent=2), flush=True)
        if not args.normal_close:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0
    finally:
        # Isaac/Kit cleanup can hang in headless diagnostic runs after the report
        # has already been flushed. All asset-level failures are captured above,
        # so fast shutdown keeps the diagnostic reproducible without changing
        # simulation state or saving stages.
        if args.normal_close:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())

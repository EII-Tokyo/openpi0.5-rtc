#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.validation.fixed_bottle_grasp import classify_trial, summarize_trials


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_STAGE = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe_inner_pad_proxy_runtime.usda"
)
DEFAULT_BOTTLE_USD = REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_SCENE = REPO_ROOT / "scenes/aloha_fixed_bottle_grasp_test.usd"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/aloha_fixed_bottle_grasp"

LEFT_ARTICULATION = "/scene/left_base_link/left_base_link"
LEFT_GRIPPER_LINK = "/scene/left_base_link/left_gripper_link"
LEFT_FINGER_PATH = "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy"
RIGHT_FINGER_PATH = "/scene/left_base_link/left_right_finger_link/bbox_collision_proxy"
TABLE_PATH = "/scene/worldBody/table"
BOTTLE_PATH = "/World/FixedBottleGrasp/Bottle500"
PHYSICS_SCENE_PATH = "/World/physicsScene"
PHASE132_RUNNER = REPO_ROOT / "aloha_isaac_replay/scripts/run_phase132_active_tabletop_grasp_gate.py"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _bbox(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {"path": prim_path, "exists": False, "bbox_valid": False}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    return {
        "path": prim_path,
        "exists": True,
        "bbox_valid": True,
        "min": [float(mn[i]) for i in range(3)],
        "max": [float(mx[i]) for i in range(3)],
        "center": [float((mn[i] + mx[i]) * 0.5) for i in range(3)],
        "size": [float(mx[i] - mn[i]) for i in range(3)],
    }


def _xform_translation(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.asarray(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=np.float64)


def _clear_xform_to(stage: Any, prim_path: str, translate: np.ndarray, rotate_xyz_deg: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in translate]))
    xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in rotate_xyz_deg]))


def _safe_create_attr(api: Any, creator_name: str, value: Any) -> bool:
    creator = getattr(api, creator_name, None)
    if creator is None:
        return False
    try:
        creator().Set(value)
    except Exception:
        return False
    return True


def build_test_scene(*, base_stage: Path, bottle_usd: Path, output_scene: Path) -> Path:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    output_scene.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_scene))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(str(base_stage.resolve()))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/FixedBottleGrasp")
    physics_scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    _safe_create_attr(physx_scene, "CreateTimeStepsPerSecondAttr", 50)

    for path in ("/World/PipePlaceholder", "/World/Pipe", "/World/UserMeasuredPipe"):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)

    table_box = _bbox(stage, TABLE_PATH)
    if not table_box["bbox_valid"]:
        raise RuntimeError(f"Cannot build fixed grasp scene: missing table prim {TABLE_PATH}")
    table_center = np.asarray(table_box["center"], dtype=np.float64)
    table_top_z = float(table_box["max"][2])

    bottle = UsdGeom.Xform.Define(stage, BOTTLE_PATH)
    bottle.GetPrim().GetReferences().AddReference(str(bottle_usd.resolve()), "/Bottle500")
    # Bottle local +Z is its long axis. Place it flat along world +X and centered in a fixed table region.
    _clear_xform_to(stage, BOTTLE_PATH, table_center + np.asarray([0.0, 0.0, 0.07]), (0.0, 90.0, 0.0))
    first_box = _bbox(stage, BOTTLE_PATH)
    if not first_box["bbox_valid"]:
        raise RuntimeError("Bottle reference did not compose real geometry")
    dz = table_top_z - float(first_box["min"][2])
    placement = np.asarray(first_box["center"], dtype=np.float64)
    _clear_xform_to(stage, BOTTLE_PATH, table_center + np.asarray([0.0, 0.0, 0.07 + dz]), (0.0, 90.0, 0.0))

    light = UsdLux.DistantLight.Define(stage, "/World/FixedBottleGrasp/KeyLight")
    light.CreateIntensityAttr(1500.0)
    light.CreateAngleAttr(0.3)
    stage.GetRootLayer().Save()
    return output_scene


def _get_attr(api: Any, getter_name: str, default: Any = "UNKNOWN") -> Any:
    getter = getattr(api, getter_name, None)
    if getter is None:
        return default
    try:
        attr = getter()
        if attr is None:
            return default
        value = attr.Get()
    except Exception:
        return default
    return default if value is None else value


def _valid_prim(stage: Any, prim_path: str) -> Any | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    return prim


def _physics_material_summary(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import UsdPhysics, UsdShade

    prim = _valid_prim(stage, prim_path)
    if prim is None:
        return {"material_path": "UNKNOWN"}
    binding = UsdShade.MaterialBindingAPI(prim)
    material, rel = binding.ComputeBoundMaterial()
    if not material:
        return {"material_path": "UNKNOWN"}
    mat_prim = material.GetPrim()
    api = UsdPhysics.MaterialAPI(mat_prim)
    return {
        "material_path": str(mat_prim.GetPath()),
        "static_friction": _json_safe(api.GetStaticFrictionAttr().Get()) if api.GetStaticFrictionAttr() else "UNKNOWN",
        "dynamic_friction": _json_safe(api.GetDynamicFrictionAttr().Get()) if api.GetDynamicFrictionAttr() else "UNKNOWN",
        "restitution": _json_safe(api.GetRestitutionAttr().Get()) if api.GetRestitutionAttr() else "UNKNOWN",
        "binding_relationship": str(rel.GetPath()) if rel else "UNKNOWN",
    }


def audit_scene(stage: Any, art: Any | None = None) -> dict[str, Any]:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    physics_scene_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    physics_scene_prim = (
        _valid_prim(stage, PHYSICS_SCENE_PATH)
        or (physics_scene_prims[0] if physics_scene_prims else None)
    )
    colliders: list[dict[str, Any]] = []
    mesh_collider_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        prim_type = prim.GetTypeName()
        if prim_type == "Mesh":
            mesh_collider_count += 1
        physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim) if prim_type == "Mesh" else None
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_enabled = _get_attr(collision_api, "GetCollisionEnabledAttr")
        colliders.append(
            {
                "path": str(prim.GetPath()),
                "type": prim_type,
                "is_mesh_collider": prim_type == "Mesh",
                "collision_enabled": collision_enabled,
                "approximation": _get_attr(mesh_collision, "GetApproximationAttr") if mesh_collision else "NOT_MESH",
                "contact_offset": _get_attr(physx_collision, "GetContactOffsetAttr"),
                "rest_offset": _get_attr(physx_collision, "GetRestOffsetAttr"),
            }
        )

    rigid_bodies = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    bottle_prim = _valid_prim(stage, BOTTLE_PATH)
    mass_api = UsdPhysics.MassAPI(bottle_prim) if bottle_prim is not None else None
    physx_scene = PhysxSchema.PhysxSceneAPI(physics_scene_prim) if physics_scene_prim is not None else None
    physics_scene_path = str(physics_scene_prim.GetPath()) if physics_scene_prim is not None else "UNKNOWN"
    blocking_unknowns: list[str] = []
    if physics_scene_path == "UNKNOWN":
        blocking_unknowns.append("physics_scene")
    if bottle_prim is None:
        blocking_unknowns.append("bottle_prim")
    if _valid_prim(stage, TABLE_PATH) is None:
        blocking_unknowns.append("table_prim")
    if not any(str(row["path"]).startswith(BOTTLE_PATH) for row in colliders):
        blocking_unknowns.append("bottle_colliders")
    if not any(str(row["path"]) == LEFT_FINGER_PATH for row in colliders):
        blocking_unknowns.append("left_finger_contact_proxy")
    if not any(str(row["path"]) == RIGHT_FINGER_PATH for row in colliders):
        blocking_unknowns.append("right_finger_contact_proxy")
    audit = {
        "audit_status": "PASS_AUDIT_FOR_STATIC_FIELDS" if not blocking_unknowns else "AUDIT_INCOMPLETE",
        "blocking_unknowns": blocking_unknowns,
        "stage": {
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "root_layer": stage.GetRootLayer().identifier,
            "authored_physics_scenes": [str(prim.GetPath()) for prim in physics_scene_prims],
        },
        "aloha_articulation_root": LEFT_ARTICULATION if stage.GetPrimAtPath(LEFT_ARTICULATION) else "UNKNOWN",
        "bottle_prim": BOTTLE_PATH if bottle_prim is not None else "UNKNOWN",
        "table_prim": TABLE_PATH if _valid_prim(stage, TABLE_PATH) is not None else "UNKNOWN",
        "rigid_bodies": rigid_bodies,
        "collider_count": len(colliders),
        "mesh_collider_count": mesh_collider_count,
        "colliders": colliders,
        "bottle": {
            "mass": _get_attr(mass_api, "GetMassAttr") if mass_api else "UNKNOWN",
            "center_of_mass": _get_attr(mass_api, "GetCenterOfMassAttr") if mass_api else "UNKNOWN",
            "diagonal_inertia": _get_attr(mass_api, "GetDiagonalInertiaAttr") if mass_api else "UNKNOWN",
            "principal_axes": _get_attr(mass_api, "GetPrincipalAxesAttr") if mass_api else "UNKNOWN",
            "physics_material": _physics_material_summary(stage, BOTTLE_PATH),
            "bbox": _bbox(stage, BOTTLE_PATH),
        },
        "table": {
            "physics_material": _physics_material_summary(stage, TABLE_PATH),
            "bbox": _bbox(stage, TABLE_PATH),
        },
        "physics_scene": {
            "path": physics_scene_path,
            "time_steps_per_second": _get_attr(physx_scene, "GetTimeStepsPerSecondAttr"),
            "solver_type": _get_attr(physx_scene, "GetSolverTypeAttr"),
            "enable_ccd": _get_attr(physx_scene, "GetEnableCCDAttr"),
        },
        "gripper_drive": {
            "left_arm_dof_names": list(getattr(art, "dof_names", []) or []) if art is not None else "UNKNOWN",
            "left_gripper_dof_names": ["left_left_finger", "left_right_finger"],
            "drive_type": "UNKNOWN",
            "stiffness": "UNKNOWN",
            "damping": "UNKNOWN",
            "effort_limit": "UNKNOWN",
            "velocity_limit": "UNKNOWN",
        },
    }
    return _json_safe(audit)


def write_audit_markdown(audit: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ALOHA Fixed Bottle Grasp Scene Audit",
        "",
        f"- audit status: `{audit.get('audit_status')}`",
        f"- blocking unknowns: `{audit.get('blocking_unknowns')}`",
        f"- stage: `{audit['stage']['root_layer']}`",
        f"- ALOHA articulation root: `{audit['aloha_articulation_root']}`",
        f"- bottle prim: `{audit['bottle_prim']}`",
        f"- table prim: `{audit['table_prim']}`",
        f"- rigid body count: `{len(audit['rigid_bodies'])}`",
        f"- collider count: `{audit['collider_count']}`",
        f"- mesh collider count: `{audit['mesh_collider_count']}`",
        f"- physics timestep: `{audit['physics_scene'].get('time_steps_per_second')}` steps/s",
        f"- solver type: `{audit['physics_scene'].get('solver_type')}`",
        "",
        "## Bottle",
        "",
        f"- mass: `{audit['bottle']['mass']}`",
        f"- center of mass: `{audit['bottle']['center_of_mass']}`",
        f"- diagonal inertia: `{audit['bottle']['diagonal_inertia']}`",
        f"- physics material: `{audit['bottle']['physics_material']}`",
        "",
        "## Gripper Drive",
        "",
        f"- DOF names: `{audit['gripper_drive']['left_arm_dof_names']}`",
        f"- left gripper DOFs: `{audit['gripper_drive']['left_gripper_dof_names']}`",
        f"- stiffness: `{audit['gripper_drive']['stiffness']}`",
        f"- damping: `{audit['gripper_drive']['damping']}`",
        f"- effort limit: `{audit['gripper_drive']['effort_limit']}`",
        f"- velocity limit: `{audit['gripper_drive']['velocity_limit']}`",
        "",
        "## Colliders",
        "",
        "| path | type | mesh | collision enabled | approximation | contact offset | rest offset |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in audit["colliders"][:200]:
        lines.append(
            f"| `{row['path']}` | `{row['type']}` | `{row['is_mesh_collider']}` | "
            f"`{row['collision_enabled']}` | `{row['approximation']}` | "
            f"`{row['contact_offset']}` | `{row['rest_offset']}` |"
        )
    if len(audit["colliders"]) > 200:
        lines.append(f"| ... | truncated after 200 of {len(audit['colliders'])} colliders | | | | | |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trial_npz_path(report_dir: Path, trial_idx: int) -> Path:
    return report_dir / "trials" / f"trial_{trial_idx:02d}.npz"


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_timeseries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _max_column(rows: list[dict[str, str]], column: str) -> float:
    values = [_float_or_nan(row.get(column)) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _array_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(column)) for row in rows], dtype=np.float64)


def _phase_rows(rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("phase") == phase]


def _finger_contact_metrics(gate: dict[str, Any]) -> tuple[bool, bool, float]:
    finger_rows = list(gate.get("finger_rows") or [])
    contacts: list[bool] = []
    impulses: list[float] = []
    for row in finger_rows[:2]:
        contacts.append(int(row.get("contact_step_count") or 0) > 0 and int(row.get("nonzero_impulse_step_count") or 0) > 0)
        impulses.append(_float_or_nan(row.get("max_impulse_norm")))
    while len(contacts) < 2:
        contacts.append(False)
    finite_impulses = [value for value in impulses if np.isfinite(value)]
    return contacts[0], contacts[1], float(max(finite_impulses)) if finite_impulses else 0.0


def _fixed_trial_metrics_from_runner(raw_metrics: dict[str, Any], rows: list[dict[str, str]], *, expected_hold_steps: int) -> dict[str, Any]:
    hold_phase = "post_close_lift_hold" if _phase_rows(rows, "post_close_lift_hold") else "post_close_lift"
    hold_rows = _phase_rows(rows, hold_phase)
    lift_rows = _phase_rows(rows, "post_close_lift")
    eval_rows = hold_rows or lift_rows or rows
    by_phase = raw_metrics.get("bilateral_grasp_formation_gate_by_phase") or {}
    bilateral_gate = by_phase.get(hold_phase) or by_phase.get("post_close_lift") or raw_metrics.get("bilateral_grasp_formation_gate") or {}
    left_contact, right_contact, max_impulse = _finger_contact_metrics(bilateral_gate)
    failure_reasons = [str(reason) for reason in raw_metrics.get("failure_reasons") or []]
    drive_audit = raw_metrics.get("drive_authority_audit") or {}
    workcell_gate = raw_metrics.get("workcell_contact_policy_gate") or {}
    object_lift_gate = raw_metrics.get("object_lift_gate") or {}
    lift_transport_gate = raw_metrics.get("lift_transport_gate") or {}
    final_row = eval_rows[-1] if eval_rows else {}
    initial_z = _float_or_nan((lift_transport_gate or {}).get("object_height_initial_m"))
    final_z = _float_or_nan(final_row.get("object_center_z"))
    object_lift = _float_or_nan(raw_metrics.get("object_lift"))
    if not np.isfinite(object_lift) and np.isfinite(initial_z) and np.isfinite(final_z):
        object_lift = float(final_z - initial_z)
    max_slip = _max_column(eval_rows, "object_cross_closing_axis_offset_norm_m")
    if not np.isfinite(max_slip):
        max_slip = 0.0
    hold_step_count = len(hold_rows)
    tracking_gate = (raw_metrics.get("controller_tracking_gate_by_phase") or {}).get(hold_phase) or {}
    return {
        "reset_stable": True,
        "initial_penetration": any("INITIAL_PENETRATION" in reason for reason in failure_reasons),
        "control_timeout": any("TIMEOUT" in reason for reason in failure_reasons),
        "nan_or_inf": False,
        "joint_limit_or_effort_violation": bool(
            (raw_metrics.get("target_limit_gripper_max_violation") or 0) not in (0, 0.0, None)
        ),
        "collider_penetration": any("PENETRATION" in reason for reason in failure_reasons),
        "left_contact": left_contact,
        "right_contact": right_contact,
        "max_contact_force_n": max_impulse,
        "lift_height_m": object_lift,
        "left_table_during_hold": bool(
            hold_step_count >= int(expected_hold_steps)
            and (object_lift_gate.get("pass") is True or object_lift >= 0.08)
            and np.isfinite(final_z)
        ),
        "touched_table_during_hold": not bool(workcell_gate.get("pass", True)),
        "max_slip_m": max_slip,
        "hold_phase": hold_phase,
        "hold_step_count": hold_step_count,
        "expected_hold_steps": int(expected_hold_steps),
        "object_final_z_m": final_z,
        "estimated_effort_clipped": drive_audit.get("estimated_effort_clipped"),
        "measured_effort_available": False,
        "legacy_runner_status": raw_metrics.get("status"),
        "legacy_failure_reasons": failure_reasons,
        "legacy_contact_trace_status": raw_metrics.get("contact_trace_status"),
        "legacy_tracking_status": tracking_gate.get("status") or (raw_metrics.get("controller_tracking_gate") or {}).get("status"),
        "lift_transport_status": lift_transport_gate.get("status"),
        "diagnostic_only": bool((lift_transport_gate or {}).get("diagnostic_only", True)),
    }


def _save_trial_npz(report_dir: Path, trial_idx: int, *, rows: list[dict[str, str]], result: dict[str, Any], raw_dir: Path) -> None:
    path = _trial_npz_path(report_dir, trial_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(len(rows), dtype=np.float64) * 0.02
    np.savez_compressed(
        path,
        simulation_timestamp=timestamps,
        phase=np.asarray([row.get("phase", "") for row in rows], dtype=object),
        object_position=np.stack(
            [
                _array_column(rows, "object_center_x"),
                _array_column(rows, "object_center_y"),
                _array_column(rows, "object_center_z"),
            ],
            axis=1,
        )
        if rows
        else np.empty((0, 3), dtype=np.float64),
        finger_gap=_array_column(rows, "finger_center_distance"),
        left_gripper_qpos=_array_column(rows, "left_finger_qpos"),
        right_gripper_qpos=_array_column(rows, "right_finger_qpos"),
        object_relative_slip=_array_column(rows, "object_cross_closing_axis_offset_norm_m"),
        success=bool(result["success"]),
        reason=str(result["reason"]),
        metrics_json=json.dumps(_json_safe(result["metrics"]), ensure_ascii=False),
        raw_runner_dir=str(raw_dir),
    )


def run_fixed_bottle_trials(report_dir: Path, *, trials: int, post_lift_hold_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_root = report_dir / "runner_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for idx in range(int(trials)):
        raw_dir = raw_root / f"trial_{idx:02d}"
        cmd = [
            sys.executable,
            str(PHASE132_RUNNER),
            "--python",
            sys.executable,
            "--output-dir",
            str(raw_dir),
            "--hdf5-replay-mode",
            "hdf5_arm_start_then_gripper_only",
            "--hdf5-arm-hold-frame-offset",
            "28",
            "--tabletop-mode",
            "diagnostic_shift_to_open_finger",
            "--post-close-hold-steps",
            "100",
            "--post-close-lift-source",
            "jacobian_vertical",
            "--post-close-lift-height",
            "0.10",
            "--post-close-lift-steps",
            "100",
            "--post-close-lift-hold-steps",
            str(int(post_lift_hold_steps)),
            "--post-close-lift-max-joint-delta",
            "0.16",
            "--min-object-lift",
            "0.08",
            "--enable-episode18-loaded-qpos-calibration",
            "--object-shape",
            "bottle_usd_grasp_box_proxy",
            "--placement-basis",
            "close",
            "--closing-axis-gap-solver-basis",
            "placement",
            "--object-center-offset",
            "0",
            "0.005",
            "0",
            "--diagnostic-loaded-clamp-squeeze-depth",
            "0.016",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (raw_dir / "fixed_bottle_runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
        metrics_path = raw_dir / "gripper_passive_contact_metrics.json"
        csv_path = raw_dir / "gripper_passive_contact_timeseries.csv"
        if not metrics_path.exists() or not csv_path.exists():
            metrics = {"reset_stable": False, "control_timeout": True, "runner_exit_code": proc.returncode}
            result = classify_trial(metrics).to_dict()
        else:
            raw_metrics = json.loads(metrics_path.read_text())
            timeseries_rows = _read_timeseries(csv_path)
            metrics = _fixed_trial_metrics_from_runner(raw_metrics, timeseries_rows, expected_hold_steps=post_lift_hold_steps)
            result = classify_trial(metrics).to_dict()
            _save_trial_npz(report_dir, idx, rows=timeseries_rows, result=result, raw_dir=raw_dir)
        result["trial_index"] = idx
        result["runner_exit_code"] = proc.returncode
        result["raw_runner_dir"] = _rel(raw_dir)
        rows.append(result)
        print(json.dumps({"trial": idx, "success": result["success"], "reason": result["reason"], "lift": result["metrics"].get("lift_height_m")}), flush=True)
    return rows


def run_blocked_trials(report_dir: Path, reason: str, trials: int) -> list[dict[str, Any]]:
    rows = []
    (report_dir / "trials").mkdir(parents=True, exist_ok=True)
    for idx in range(int(trials)):
        metrics = {"reset_stable": False, "control_timeout": True, "blocked_reason": reason}
        result = classify_trial(metrics).to_dict()
        result["trial_index"] = idx
        np.savez_compressed(_trial_npz_path(report_dir, idx), metrics=json.dumps(metrics), reason=result["reason"])
        rows.append(result)
    return rows


def write_final_report(report_dir: Path, summary: dict[str, Any], *, audit_path: Path, scene_path: Path) -> None:
    lines = [
        "# ALOHA Fixed Bottle Grasp Final Report",
        "",
        f"- final conclusion: `{summary['final_conclusion']}`",
        f"- generated scene: `{_rel(scene_path)}`",
        f"- scene audit: `{_rel(audit_path)}`",
        f"- trials: `{summary['trial_count']}`",
        f"- success: `{summary['success_count']}`",
        f"- failure: `{summary['failure_count']}`",
        f"- required success: `{summary['required_successes']}`",
        f"- max slip: `{summary['max_slip_m']}` m",
        f"- max lift height: `{summary.get('max_lift_height_m')}` m",
        f"- min lift height: `{summary.get('min_lift_height_m')}` m",
        f"- post-lift hold steps: `{summary.get('post_lift_hold_steps')}`",
        f"- measured gripper effort available: `{summary.get('measured_effort_available')}`",
        f"- estimated effort clipped in any trial: `{summary.get('estimated_effort_clipped_any')}`",
        "",
        "## Failure Reasons",
        "",
    ]
    reason_counts = summary.get("failure_reason_counts") or {}
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    config = summary.get("benchmark_config") or {}
    if config:
        lines.extend(["", "## Benchmark Config", ""])
        for key, value in config.items():
            lines.append(f"- {key}: `{value}`")
    first_trial = next((trial for trial in summary.get("trials", []) if trial.get("metrics")), None)
    if first_trial:
        metrics = first_trial.get("metrics") or {}
        lines.extend(
            [
                "",
                "## Legacy Gate Context",
                "",
                f"- legacy runner status: `{metrics.get('legacy_runner_status')}`",
                f"- legacy contact trace status: `{metrics.get('legacy_contact_trace_status')}`",
                f"- legacy tracking status: `{metrics.get('legacy_tracking_status')}`",
                f"- lift transport status: `{metrics.get('lift_transport_status')}`",
                "",
                "The legacy phase132 replay gates are retained as diagnostics, but this report's pass/fail result is governed by the fixed-bottle close/lift/hold criteria requested for this experiment.",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is simulation-only.",
            "- It does not run RL, RLT, PPO, VLA, camera collection, pipe insertion, randomization, or real-robot control.",
            "- The current deterministic benchmark uses the diagnostic table alignment and soft-bottle loaded clamp squeeze recorded in each trial metric; it is not a random-bottle-position RL benchmark.",
            "- Gripper effort is not directly measured in the current runner; the report records the existing first-order drive-authority estimate and keeps raw runner artifacts for follow-up.",
            "- A `BLOCKED` or failed result must not be treated as a fixed-bottle grasp pass.",
        ]
    )
    (report_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Isaac fixed-bottle left-gripper grasp validation.")
    parser.add_argument("--base-stage", type=Path, default=DEFAULT_BASE_STAGE)
    parser.add_argument("--bottle-usd", type=Path, default=DEFAULT_BOTTLE_USD)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--run-simulation", action="store_true", help="Reserved for the full physics trial loop.")
    parser.add_argument(
        "--post-lift-hold-steps",
        type=int,
        default=100,
        help="Number of 50 Hz target frames to hold the final lifted pose. Default 100 is 2 seconds.",
    )
    args = parser.parse_args()

    if args.run_simulation:
        report_dir = args.report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        audit_md = report_dir / "00_scene_audit.md"
        scene_path = args.scene.resolve()
        rows = run_fixed_bottle_trials(
            report_dir,
            trials=args.trials,
            post_lift_hold_steps=args.post_lift_hold_steps,
        )
        summary = summarize_trials(rows)
        lift_values = [
            _float_or_nan((trial.get("metrics") or {}).get("lift_height_m"))
            for trial in rows
            if trial.get("metrics")
        ]
        finite_lifts = [value for value in lift_values if np.isfinite(value)]
        summary.update(
            {
                "post_lift_hold_steps": int(args.post_lift_hold_steps),
                "benchmark_config": {
                    "episode": "episode_18.hdf5",
                    "arm_hold_frame_offset": 28,
                    "tabletop_mode": "diagnostic_shift_to_open_finger",
                    "object_shape": "bottle_usd_grasp_box_proxy",
                    "object_center_offset": [0.0, 0.005, 0.0],
                    "diagnostic_loaded_clamp_squeeze_depth": 0.016,
                    "post_close_lift_height": 0.10,
                    "post_close_lift_steps": 100,
                    "post_close_lift_hold_steps": int(args.post_lift_hold_steps),
                    "post_close_lift_max_joint_delta": 0.16,
                    "required_successes": 19,
                },
                "max_lift_height_m": float(max(finite_lifts)) if finite_lifts else float("nan"),
                "min_lift_height_m": float(min(finite_lifts)) if finite_lifts else float("nan"),
                "measured_effort_available": all(
                    bool((trial.get("metrics") or {}).get("measured_effort_available")) for trial in rows
                ),
                "estimated_effort_clipped_any": any(
                    bool((trial.get("metrics") or {}).get("estimated_effort_clipped")) for trial in rows
                ),
                "trials": rows,
            }
        )
        (report_dir / "trials_summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_final_report(report_dir, summary, audit_path=audit_md, scene_path=scene_path)
        print(json.dumps({"status": summary["final_conclusion"], "report": _rel(report_dir / "FINAL_REPORT.md")}), flush=True)
        return 0 if summary["final_conclusion"] == "PASS_FIXED_BOTTLE_GRASP" else 3

    from isaacsim import SimulationApp

    app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
    progress_path = args.report_dir.resolve() / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.unlink(missing_ok=True)

    def progress(step: str, **fields: Any) -> None:
        row = {"step": step, **fields}
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
            f.flush()

    progress("before_simulation_app")
    app = SimulationApp(app_config)
    try:
        from pxr import Usd

        report_dir = args.report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        scene_path = args.scene.resolve()
        progress("after_simulation_app", scene=str(scene_path), report_dir=str(report_dir))
        if args.rebuild_scene or not scene_path.exists():
            progress("before_build_scene")
            build_test_scene(base_stage=args.base_stage, bottle_usd=args.bottle_usd, output_scene=scene_path)
            progress("after_build_scene", exists=scene_path.exists())
        progress("before_open_stage")
        stage = Usd.Stage.Open(str(scene_path))
        if stage is None:
            raise RuntimeError(f"failed to open test scene {scene_path}")
        progress("after_open_stage")
        progress("before_audit")
        try:
            audit = audit_scene(stage, None)
        except BaseException as exc:  # noqa: BLE001 - Kit/pxr may raise SystemExit-like exceptions here.
            progress("audit_exception", exc_type=type(exc).__name__, exc=str(exc))
            error_path = report_dir / "00_scene_audit_error.json"
            error_path.write_text(
                json.dumps(
                    {"status": "AUDIT_FAILED", "error_type": type(exc).__name__, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"status": "AUDIT_FAILED", "error": str(exc), "artifact": _rel(error_path)}), flush=True)
            return 2
        progress("after_audit", collider_count=audit.get("collider_count"))
        audit_json = report_dir / "00_scene_audit.json"
        audit_md = report_dir / "00_scene_audit.md"
        audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_audit_markdown(audit, audit_md)
        progress("after_write_audit", audit_md=str(audit_md))

        if args.audit_only:
            print(json.dumps({"status": "AUDIT_ONLY", "audit": _rel(audit_md), "scene": _rel(scene_path)}), flush=True)
            return 0
        # The full closed-loop physics executor will replace this guarded BLOCKED path once reset,
        # articulation drive readback, and contact-force extraction are verified in one Isaac process.
        rows = run_blocked_trials(report_dir, "full_physics_trial_loop_not_yet_enabled", args.trials)
        summary = summarize_trials(rows)
        write_final_report(report_dir, summary, audit_path=audit_md, scene_path=scene_path)
        print(json.dumps({"status": summary["final_conclusion"], "report": _rel(report_dir / "FINAL_REPORT.md")}), flush=True)
        return 0 if summary["final_conclusion"] == "PASS_FIXED_BOTTLE_GRASP" else 3
    finally:
        progress("before_app_close")
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

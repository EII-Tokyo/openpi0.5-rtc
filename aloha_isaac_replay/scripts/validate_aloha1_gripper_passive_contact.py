from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG
from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_gravity
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _get_limits
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _json_safe
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_state
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _set_full_target
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import FINGER_PROXY_PATHS
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _bbox_row
from aloha_isaac_replay.scripts.validate_aloha1_gripper_proxy_gap import _gap_metrics
from aloha_isaac_replay.scripts.validate_aloha1_native_single_joint_response import _safe_target


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = REPO_ROOT / "local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase43_gripper_passive_contact_20260718"
DEFAULT_BOTTLE_USD = REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_MAPPING = REPO_ROOT / "configs/aloha/original_stationary_aloha_mapping.yaml"


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "step",
        "object_center_x",
        "object_center_y",
        "object_center_z",
        "object_displacement",
        "left_finger_qpos",
        "right_finger_qpos",
        "finger_center_distance",
        "left_axis_min",
        "left_axis_max",
        "left_axis_center",
        "right_axis_min",
        "right_axis_max",
        "right_axis_center",
        "object_axis_min",
        "object_axis_max",
        "object_axis_center",
        "target_finger_object_surface_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    unique_pairs = payload.get("unique_contact_pairs") or []
    pair_lines = [f"- `{pair}`" for pair in unique_pairs[:12]]
    if len(unique_pairs) > 12:
        pair_lines.append(f"- ... {len(unique_pairs) - 12} more unique pairs")
    finger_hits = payload.get("target_contact_finger_hits") or {}
    finger_hit_lines = [f"- `{path}`: `{hit}`" for path, hit in finger_hits.items()]
    cross_overlap = payload.get("cross_side_proxy_overlap") or {}
    lines = [
        "# Gripper Passive Contact Smoke",
        "",
        f"- status: `{payload['status']}`",
        f"- contact trace status: `{payload.get('contact_trace_status')}`",
        f"- stage: `{payload['inputs']['stage_usd']}`",
        f"- control mode: `{payload['inputs']['control_mode']}`",
        f"- moving fingers: `{payload['inputs'].get('moving_fingers')}`",
        f"- object side length: `{payload.get('object_side_length_stage_units')}` stage units",
        f"- object settle displacement: `{payload.get('object_settle_displacement')}` stage units",
        f"- object close displacement: `{payload.get('object_displacement')}` stage units",
        f"- object total displacement: `{payload.get('total_object_displacement')}` stage units",
        f"- max object displacement: `{payload.get('max_object_displacement')}` stage units",
        f"- finite object motion: `{payload.get('object_motion_finite')}`",
        f"- contact motion lower bound ok: `{payload.get('contact_motion_ok')}`",
        f"- no explosion upper bound ok: `{payload.get('no_explosion_ok')}`",
        f"- contact pair trace enabled: `{payload.get('contact_pair_trace_enabled')}`",
        f"- contact pair count: `{payload.get('contact_pair_count')}`",
        f"- target contact pair found: `{payload.get('target_contact_pair_found')}`",
        f"- all expected fingers contacted object: `{payload.get('all_expected_fingers_target_contact_pair_found')}`",
        f"- cross-side proxy overlap detected: `{cross_overlap.get('overlap_detected')}`",
        f"- first contact pair: `{payload.get('first_contact_pair')}`",
        f"- first target contact pair: `{payload.get('first_target_contact_pair')}`",
        f"- first target contact step: `{payload.get('first_target_contact_step')}`",
        f"- target contact persistence steps: `{payload.get('target_contact_persistence_steps')}`",
        "",
        "## Expected Finger Coverage",
        "",
        *(finger_hit_lines or ["- none"]),
        "",
        "## Unique Contact Pairs",
        "",
        *(pair_lines or ["- none"]),
        "",
        "## Interpretation",
        "",
        "This is a local contact smoke test. It only checks whether a small passive cube between the gripper proxies remains numerically stable and moves within a bounded range during finger closure.",
        "A non-zero contact count is not a success condition. The trace must show that the target fingertip proxy contacts the target object collider, and object motion must remain bounded.",
        "It does not validate grasp success, bottle geometry, friction realism, or full-arm task behavior.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _finger_targets(art: Any, offset: float, limit_margin: float) -> tuple[np.ndarray, dict[str, float]]:
    dof_names = list(art.dof_names)
    limits = _get_limits(art)
    qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
    target = qpos.copy()
    target_values: dict[str, float] = {}
    for name, sign in [("left_finger", 1.0), ("right_finger", -1.0)]:
        idx = dof_names.index(name)
        lower, upper = [float(x) for x in limits[idx]]
        origin = (lower + upper) * 0.5
        target_value, _clipped = _safe_target(origin, offset * sign, lower, upper, limit_margin)
        target[idx] = target_value
        target_values[name] = target_value
    return target, target_values


def _load_hdf5_qpos(path: str | Path, *, start: int | None, end: int | None, max_frames: int | None) -> np.ndarray:
    import h5py

    episode = Path(path)
    with h5py.File(episode, "r") as h5:
        qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        raise ValueError(f"Expected observations/qpos shape (T, >=14), got {qpos.shape} in {episode}")
    lo = 0 if start is None else int(start)
    hi = len(qpos) if end is None else int(end)
    seq = qpos[lo:hi]
    if max_frames is not None:
        seq = seq[: int(max_frames)]
    if seq.shape[0] < 2:
        raise ValueError(f"Need at least two HDF5 qpos samples, got {seq.shape[0]} from {episode}")
    if not np.isfinite(seq).all():
        raise ValueError(f"HDF5 qpos contains NaN/Inf: {episode}")
    return np.asarray(seq, dtype=np.float64)


def _target_from_standard_qpos(
    *,
    art: Any,
    side: str,
    qpos_frame: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
) -> np.ndarray:
    dof_names = list(art.dof_names)
    target = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1).copy()
    if replay_mode == "left_arm_and_gripper":
        if mapping is None:
            raise ValueError("left_arm_and_gripper replay requires a mapping")
        side_prefix = f"{side}/"
        for arm_target in arm_only_targets_from_standard_qpos(qpos_frame, mapping):
            if not arm_target.isaac_dof_name.startswith(side_prefix):
                continue
            dof_name = arm_target.isaac_dof_name[len(side_prefix):]
            target[dof_names.index(dof_name)] = float(arm_target.value)
    channel = 6 if side == "left" else 13
    fingers = standard_gripper_qpos_to_isaac_fingers(float(qpos_frame[channel]), side=side)
    target[dof_names.index("left_finger")] = float(fingers[f"{side}/left_finger"])
    target[dof_names.index("right_finger")] = float(fingers[f"{side}/right_finger"])
    return target


def _targets_from_hdf5_qpos(
    *,
    art: Any,
    side: str,
    qpos: np.ndarray,
    mapping: dict[str, Any] | None,
    replay_mode: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    dof_names = list(art.dof_names)
    left_idx = dof_names.index("left_finger")
    right_idx = dof_names.index("right_finger")
    channel = 6 if side == "left" else 13
    gripper_qpos = np.asarray(qpos[:, channel], dtype=np.float64)
    targets: list[np.ndarray] = []
    for frame in qpos:
        targets.append(
            _target_from_standard_qpos(
                art=art,
                side=side,
                qpos_frame=frame,
                mapping=mapping,
                replay_mode=replay_mode,
            )
        )
    arm_delta = None
    if replay_mode == "left_arm_and_gripper":
        indices = slice(0, 6) if side == "left" else slice(7, 13)
        arm_qpos = np.asarray(qpos[:, indices], dtype=np.float64)
        arm_delta = {
            "max_abs_frame_delta": float(np.max(np.abs(np.diff(arm_qpos, axis=0)))) if len(arm_qpos) > 1 else 0.0,
            "max_abs_net_delta": float(np.max(np.abs(arm_qpos[-1] - arm_qpos[0]))),
        }
    return targets, {
        "source": "observations/qpos",
        "side": side,
        "replay_mode": replay_mode,
        "sample_count": int(gripper_qpos.size),
        "raw_start": float(gripper_qpos[0]),
        "raw_end": float(gripper_qpos[-1]),
        "raw_min": float(np.min(gripper_qpos)),
        "raw_max": float(np.max(gripper_qpos)),
        "raw_range": float(np.max(gripper_qpos) - np.min(gripper_qpos)),
        "raw_net": float(gripper_qpos[-1] - gripper_qpos[0]),
        "first_target_values": {
            "left_finger": float(targets[0][left_idx]),
            "right_finger": float(targets[0][right_idx]),
        },
        "last_target_values": {
            "left_finger": float(targets[-1][left_idx]),
            "right_finger": float(targets[-1][right_idx]),
        },
        "arm_qpos_delta": arm_delta,
    }


def _set_finger_target_and_step(world: Any, art: Any, target: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        _set_full_target(art, target)
        world.step(render=False)


def _surface_gap(left_box: dict[str, Any], right_box: dict[str, Any], axis: int) -> float:
    left_min = float(left_box["min"][axis])
    left_max = float(left_box["max"][axis])
    right_min = float(right_box["min"][axis])
    right_max = float(right_box["max"][axis])
    if left_max <= right_min:
        return right_min - left_max
    if right_max <= left_min:
        return left_min - right_max
    return 0.0


def _axis_probe_row(
    *,
    axis: int,
    left_box: dict[str, Any],
    right_box: dict[str, Any],
    object_box: dict[str, Any],
    target_finger_box: dict[str, Any],
) -> dict[str, float | None]:
    def pick(box: dict[str, Any], key: str) -> float | None:
        values = box.get(key)
        if values is None:
            return None
        return float(values[axis])

    return {
        "left_axis_min": pick(left_box, "min"),
        "left_axis_max": pick(left_box, "max"),
        "left_axis_center": pick(left_box, "center"),
        "right_axis_min": pick(right_box, "min"),
        "right_axis_max": pick(right_box, "max"),
        "right_axis_center": pick(right_box, "center"),
        "object_axis_min": pick(object_box, "min"),
        "object_axis_max": pick(object_box, "max"),
        "object_axis_center": pick(object_box, "center"),
        "target_finger_object_surface_gap": _surface_gap(target_finger_box, object_box, axis)
        if target_finger_box.get("bbox_valid") and object_box.get("bbox_valid")
        else None,
    }


def _axis_rotation_xyz(axis: str) -> tuple[float, float, float]:
    """Rotate Bottle500 local +Z long axis onto the requested world axis."""
    normalized_axis = axis.upper()
    if normalized_axis == "X":
        return (0.0, 90.0, 0.0)
    if normalized_axis == "Y":
        return (-90.0, 0.0, 0.0)
    if normalized_axis == "Z":
        return (0.0, 0.0, 0.0)
    raise ValueError(f"Unsupported object axis: {axis}")


def _bbox_center(stage: Any, path: str) -> np.ndarray:
    box = _bbox_row(stage, path)
    if not box.get("bbox_valid"):
        raise RuntimeError(f"Cannot compute bbox center for {path}")
    return np.asarray(box["center"], dtype=np.float64)


def _create_passive_cube(
    *,
    world: Any,
    stage: Any,
    path: str,
    center: np.ndarray,
    side_length: float,
    mass: float,
    creation_mode: str,
    shape: str = "cube",
    axis: str = "X",
    length_multiplier: float = 4.0,
    usd_path: str | Path | None = None,
    usd_prim_path: str = "/Bottle500",
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    if shape != "cube" and creation_mode != "raw_usd":
        raise ValueError(f"{shape} object shape requires raw_usd creation; got {creation_mode}")
    if creation_mode == "dynamic_cuboid":
        from isaacsim.core.api.objects import DynamicCuboid

        world.scene.add(
            DynamicCuboid(
                prim_path=path,
                name="phase43_passive_contact_cube",
                position=np.asarray(center, dtype=np.float64),
                scale=np.asarray([side_length, side_length, side_length], dtype=np.float64),
                size=1.0,
                mass=float(mass),
                color=np.asarray([0.9, 0.2, 0.1], dtype=np.float64),
            )
        )
        return
    if creation_mode != "raw_usd":
        raise ValueError(f"Unsupported object creation mode: {creation_mode}")
    normalized_axis = axis.upper()
    if normalized_axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported object axis: {axis}")
    if shape == "cube":
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        scale = Gf.Vec3d(side_length, side_length, side_length)
    elif shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateAxisAttr(normalized_axis)
        geom.CreateRadiusAttr(side_length * 0.5)
        geom.CreateHeightAttr(side_length * length_multiplier)
        scale = Gf.Vec3d(1.0, 1.0, 1.0)
    elif shape == "capsule":
        geom = UsdGeom.Capsule.Define(stage, path)
        geom.CreateAxisAttr(normalized_axis)
        geom.CreateRadiusAttr(side_length * 0.5)
        geom.CreateHeightAttr(side_length * length_multiplier)
        scale = Gf.Vec3d(1.0, 1.0, 1.0)
    elif shape == "bottle_proxy":
        root = UsdGeom.Xform.Define(stage, path)
        root_xform = UsdGeom.Xformable(root.GetPrim())
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))

        body_length = side_length * length_multiplier
        neck_length = side_length * max(length_multiplier * 0.35, 1.0)
        body_radius = side_length * 0.5
        neck_radius = side_length * 0.18
        mouth_radius = side_length * 0.22

        body = UsdGeom.Cylinder.Define(stage, f"{path}/body")
        body.CreateAxisAttr(normalized_axis)
        body.CreateRadiusAttr(body_radius)
        body.CreateHeightAttr(body_length)
        body.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.95)])

        neck = UsdGeom.Cylinder.Define(stage, f"{path}/neck")
        neck.CreateAxisAttr(normalized_axis)
        neck.CreateRadiusAttr(neck_radius)
        neck.CreateHeightAttr(neck_length)
        neck.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.9, 1.0)])

        mouth = UsdGeom.Sphere.Define(stage, f"{path}/mouth")
        mouth.CreateRadiusAttr(mouth_radius)
        mouth.CreateDisplayColorAttr([Gf.Vec3f(0.02, 0.04, 0.1)])

        axis_index = {"X": 0, "Y": 1, "Z": 2}[normalized_axis]

        def offset_vec(distance: float) -> Gf.Vec3d:
            values = [0.0, 0.0, 0.0]
            values[axis_index] = distance
            return Gf.Vec3d(*values)

        neck_distance = body_length * 0.5 + neck_length * 0.5
        mouth_distance = body_length * 0.5 + neck_length + mouth_radius
        UsdGeom.Xformable(neck.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(offset_vec(neck_distance))
        UsdGeom.Xformable(mouth.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(offset_vec(mouth_distance))

        for child in (body.GetPrim(), neck.GetPrim(), mouth.GetPrim()):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
        return
    elif shape == "bottle_usd":
        if usd_path is None:
            raise ValueError("bottle_usd requires a USD asset path")
        asset_path = Path(usd_path).expanduser().resolve()
        if not asset_path.exists():
            raise FileNotFoundError(f"bottle_usd asset does not exist: {asset_path}")
        root = UsdGeom.Xform.Define(stage, path)
        root.GetPrim().GetReferences().AddReference(str(asset_path), usd_prim_path)
        root_xform = UsdGeom.Xformable(root.GetPrim())
        root_xform.ClearXformOpOrder()
        translate_op = root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        rotate_op = root_xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*[float(x) for x in center]))
        rotate_op.Set(Gf.Vec3d(*_axis_rotation_xyz(normalized_axis)))

        # The referenced asset origin is semantic, not necessarily its collision
        # bbox center. Move once more after composition so the actual object used
        # by PhysX is centered between the fingertips.
        composed_center = _bbox_center(stage, path)
        correction = np.asarray(center, dtype=np.float64) - composed_center
        translate_op.Set(Gf.Vec3d(*[float(x) for x in np.asarray(center, dtype=np.float64) + correction]))

        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
        return
    else:
        raise ValueError(f"Unsupported object shape: {shape}")
    geom.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.2, 0.1)])
    xform = UsdGeom.Xformable(geom.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(x) for x in center]))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(scale)
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
    UsdPhysics.MassAPI.Apply(geom.GetPrim()).CreateMassAttr(float(mass))


def _set_collision_offsets(stage: Any, prim_path: str, contact_offset: float | None, rest_offset: float | None) -> dict[str, Any]:
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return {"path": prim_path, "exists": False, "applied": False}
    author_offsets = contact_offset is not None or rest_offset is not None
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim) if author_offsets else PhysxSchema.PhysxCollisionAPI(prim)
    if contact_offset is not None:
        api.CreateContactOffsetAttr(float(contact_offset)).Set(float(contact_offset))
    if rest_offset is not None:
        api.CreateRestOffsetAttr(float(rest_offset)).Set(float(rest_offset))
    return {
        "path": prim_path,
        "exists": True,
        "applied": author_offsets,
        "contact_offset": api.GetContactOffsetAttr().Get() if api.GetContactOffsetAttr() else None,
        "rest_offset": api.GetRestOffsetAttr().Get() if api.GetRestOffsetAttr() else None,
    }


def _set_object_collision_offsets(stage: Any, prim_path: str, contact_offset: float | None, rest_offset: float | None) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return {"path": prim_path, "exists": False, "targets": []}
    targets = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not targets:
        targets = [prim_path]
    return {
        "path": prim_path,
        "exists": True,
        "targets": [
            _set_collision_offsets(stage, target, contact_offset, rest_offset)
            for target in targets
        ],
    }


def _begin_contact_pair_trace(stage: Any, *, disable_usd_updates: bool) -> dict[str, Any]:
    import carb
    import usdrt
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
    from pxr import PhysxSchema, Sdf, Usd, UsdUtils

    session_sub_layer = Sdf.Layer.CreateAnonymous()
    stage.GetSessionLayer().subLayerPaths.append(session_sub_layer.identifier)
    old_layer = stage.GetEditTarget().GetLayer()
    stage.SetEditTarget(Usd.EditTarget(session_sub_layer))

    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Insert(stage)
    stage_id = stage_cache.GetId(stage).ToLongInt()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    rigid_body_paths = [str(path) for path in usdrt_stage.GetPrimsWithAppliedAPIName("PhysicsRigidBodyAPI")]
    for prim_path in rigid_body_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            contact_report_api.CreateThresholdAttr().Set(0)

    settings = carb.settings.get_settings()
    write_usd = settings.get_as_bool(SETTING_UPDATE_TO_USD)
    write_fabric = settings.get_as_bool("/physics/fabricEnabled")
    if disable_usd_updates:
        settings.set(SETTING_UPDATE_TO_USD, False)
        settings.set("/physics/fabricEnabled", False)
    return {
        "enabled": True,
        "stage_id": stage_id,
        "session_sub_layer": session_sub_layer,
        "old_layer": old_layer,
        "settings": settings,
        "write_usd": write_usd,
        "write_fabric": write_fabric,
        "disable_usd_updates": disable_usd_updates,
        "rigid_body_paths": rigid_body_paths,
        "physx_interface": get_physx_simulation_interface(),
    }


def _finish_contact_pair_trace(stage: Any, trace_state: dict[str, Any] | None) -> None:
    if not trace_state:
        return
    from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD

    settings = trace_state["settings"]
    if trace_state.get("disable_usd_updates"):
        settings.set(SETTING_UPDATE_TO_USD, trace_state["write_usd"])
        settings.set("/physics/fabricEnabled", trace_state["write_fabric"])
    stage.SetEditTarget(trace_state["old_layer"])
    layer_id = trace_state["session_sub_layer"].identifier
    if layer_id in stage.GetSessionLayer().subLayerPaths:
        stage.GetSessionLayer().subLayerPaths.remove(layer_id)


def _read_contact_pairs(trace_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_state:
        return []
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools

    contact_headers, _contact_data = trace_state["physx_interface"].get_contact_report()
    rows: list[dict[str, Any]] = []
    for contact_header in contact_headers:
        collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
        collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
        rows.append(
            {
                "type": int(contact_header.type),
                "type_name": "CONTACT_FOUND" if contact_header.type == ContactEventType.CONTACT_FOUND else str(contact_header.type),
                "collider0": collider0,
                "collider1": collider1,
                "sorted_pair": sorted([collider0, collider1]),
            }
        )
    return rows


def _path_matches(path: str, target: str) -> bool:
    return path == target or path.startswith(f"{target}/")


def _pair_touches_targets(pair: dict[str, Any], object_path: str, finger_paths: list[str]) -> bool:
    collider0 = str(pair["collider0"])
    collider1 = str(pair["collider1"])
    touches_object = _path_matches(collider0, object_path) or _path_matches(collider1, object_path)
    touches_finger = any(_path_matches(collider0, finger_path) or _path_matches(collider1, finger_path) for finger_path in finger_paths)
    return bool(touches_object and touches_finger)


def _pair_touches_finger(pair: dict[str, Any], object_path: str, finger_path: str) -> bool:
    return _pair_touches_targets(pair, object_path, [finger_path])


def _summarize_contact_pairs(
    *,
    contact_pair_rows: list[dict[str, Any]],
    object_path: str,
    expected_finger_paths: list[str],
    sample_limit: int = 80,
) -> dict[str, Any]:
    unique_pairs = sorted({tuple(row["sorted_pair"]) for row in contact_pair_rows})
    target_rows = [
        row
        for row in contact_pair_rows
        if _pair_touches_targets(row, object_path, expected_finger_paths)
    ]
    wrong_rows = [
        row
        for row in contact_pair_rows
        if not _pair_touches_targets(row, object_path, expected_finger_paths)
    ]
    target_found_rows = [row for row in target_rows if row.get("type_name") == "CONTACT_FOUND"]
    target_steps = sorted({int(row["step"]) for row in target_rows})
    wrong_pairs = sorted({tuple(row["sorted_pair"]) for row in wrong_rows})
    finger_target_rows = {
        finger_path: [row for row in contact_pair_rows if _pair_touches_finger(row, object_path, finger_path)]
        for finger_path in expected_finger_paths
    }
    finger_target_found_rows = {
        finger_path: [row for row in rows if row.get("type_name") == "CONTACT_FOUND"]
        for finger_path, rows in finger_target_rows.items()
    }
    return {
        "contact_pair_count": len(contact_pair_rows),
        "unique_contact_pairs": [list(pair) for pair in unique_pairs],
        "contact_pairs_sample": contact_pair_rows[:sample_limit],
        "expected_contact_object": object_path,
        "expected_contact_fingers": expected_finger_paths,
        "target_contact_pair_found": bool(target_rows),
        "target_contact_found_event": bool(target_found_rows),
        "target_contact_finger_hits": {
            finger_path: bool(rows)
            for finger_path, rows in finger_target_rows.items()
        },
        "target_contact_found_finger_hits": {
            finger_path: bool(rows)
            for finger_path, rows in finger_target_found_rows.items()
        },
        "all_expected_fingers_target_contact_pair_found": all(bool(rows) for rows in finger_target_rows.values())
        if expected_finger_paths
        else False,
        "all_expected_fingers_target_contact_found_event": all(bool(rows) for rows in finger_target_found_rows.values())
        if expected_finger_paths
        else False,
        "first_target_contact_pair": target_rows[0] if target_rows else None,
        "first_target_contact_found_pair": target_found_rows[0] if target_found_rows else None,
        "first_target_contact_step": target_steps[0] if target_steps else None,
        "target_contact_steps": target_steps,
        "target_contact_persistence_steps": len(target_steps),
        "wrong_contact_pairs": [list(pair) for pair in wrong_pairs],
    }


def _cross_side_proxy_overlap_summary(stage: Any, side: str, tolerance: float = 1e-8) -> dict[str, Any]:
    other_side = "right" if side == "left" else "left"
    rows: list[dict[str, Any]] = []
    for finger_name in ("left_finger", "right_finger"):
        current_path = FINGER_PROXY_PATHS[side][finger_name]
        other_path = FINGER_PROXY_PATHS[other_side][finger_name]
        current_box = _bbox_row(stage, current_path)
        other_box = _bbox_row(stage, other_path)
        center_distance = None
        size_delta = None
        overlaps = False
        if current_box.get("bbox_valid") and other_box.get("bbox_valid"):
            current_center = np.asarray(current_box["center"], dtype=np.float64)
            other_center = np.asarray(other_box["center"], dtype=np.float64)
            current_size = np.asarray(current_box["size"], dtype=np.float64)
            other_size = np.asarray(other_box["size"], dtype=np.float64)
            center_distance = float(np.linalg.norm(current_center - other_center))
            size_delta = float(np.linalg.norm(current_size - other_size))
            overlaps = bool(center_distance <= tolerance and size_delta <= tolerance)
        rows.append(
            {
                "finger": finger_name,
                "current_path": current_path,
                "other_path": other_path,
                "current_bbox_valid": bool(current_box.get("bbox_valid")),
                "other_bbox_valid": bool(other_box.get("bbox_valid")),
                "center_distance": center_distance,
                "size_delta": size_delta,
                "overlaps_with_other_side": overlaps,
            }
        )
    return {
        "side": side,
        "other_side": other_side,
        "tolerance": tolerance,
        "overlap_detected": any(row["overlaps_with_other_side"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local passive-object contact smoke test for ALOHA1 gripper proxies.")
    parser.add_argument("--stage-usd", default=str(DEFAULT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--open-offset", type=float, default=0.006)
    parser.add_argument("--close-offset", type=float, default=-0.006)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--close-steps", type=int, default=180)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--limit-margin", type=float, default=0.001)
    parser.add_argument("--object-fill-fraction", type=float, default=0.6)
    parser.add_argument("--object-placement", choices=("gap_center", "moving_finger_surface"), default="gap_center")
    parser.add_argument("--object-clearance", type=float, default=0.001)
    parser.add_argument("--object-creation", choices=("dynamic_cuboid", "raw_usd"), default="raw_usd")
    parser.add_argument("--object-shape", choices=("cube", "cylinder", "capsule", "bottle_proxy", "bottle_usd"), default="cube")
    parser.add_argument("--object-axis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument("--object-length-multiplier", type=float, default=4.0)
    parser.add_argument("--object-usd", default=str(DEFAULT_BOTTLE_USD))
    parser.add_argument("--object-usd-prim-path", default="/Bottle500")
    parser.add_argument("--object-mass", type=float, default=0.01)
    parser.add_argument("--object-contact-offset", type=float, default=None)
    parser.add_argument("--object-rest-offset", type=float, default=None)
    parser.add_argument("--proxy-contact-offset", type=float, default=None)
    parser.add_argument("--proxy-rest-offset", type=float, default=None)
    parser.add_argument("--closure-profile", choices=("abrupt", "linear"), default="abrupt")
    parser.add_argument("--moving-fingers", choices=("both", "left", "right"), default="both")
    parser.add_argument("--hdf5-gripper-episode", default=None)
    parser.add_argument("--hdf5-replay-mode", choices=("gripper_only", "left_arm_and_gripper"), default="gripper_only")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--hdf5-gripper-start-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-end-frame", type=int, default=None)
    parser.add_argument("--hdf5-gripper-max-frames", type=int, default=None)
    parser.add_argument("--trace-contact-pairs", action="store_true")
    parser.add_argument(
        "--trace-disable-usd-updates",
        action="store_true",
        help="Match Isaac asset-validator style contact probing. Off by default because this script needs live USD bbox readback.",
    )
    parser.add_argument("--min-contact-motion", type=float, default=1e-5)
    parser.add_argument("--max-object-displacement", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "gripper_passive_contact_metrics.json"
    csv_path = output_dir / "gripper_passive_contact_timeseries.csv"
    md_path = output_dir / "gripper_passive_contact_metrics.md"
    payload: dict[str, Any] = {
        "status": "STARTED",
        "overall_pass": False,
        "real_robot_touched": False,
        "stage_saved": False,
        "inputs": {
            "stage_usd": _rel(args.stage_usd),
            "side": args.side,
            "control_mode": "opposed_fingers",
            "open_offset": args.open_offset,
            "close_offset": args.close_offset,
            "settle_steps": args.settle_steps,
            "close_steps": args.close_steps,
            "physics_dt": args.physics_dt,
            "gravity": args.gravity,
            "object_fill_fraction": args.object_fill_fraction,
            "object_placement": args.object_placement,
            "object_clearance": args.object_clearance,
            "object_creation": args.object_creation,
            "object_shape": args.object_shape,
            "object_axis": args.object_axis,
            "object_length_multiplier": args.object_length_multiplier,
            "object_usd": _rel(args.object_usd),
            "object_usd_prim_path": args.object_usd_prim_path,
            "object_contact_offset": args.object_contact_offset,
            "object_rest_offset": args.object_rest_offset,
            "proxy_contact_offset": args.proxy_contact_offset,
            "proxy_rest_offset": args.proxy_rest_offset,
            "closure_profile": args.closure_profile,
            "moving_fingers": args.moving_fingers,
            "hdf5_gripper_episode": _rel(args.hdf5_gripper_episode) if args.hdf5_gripper_episode else None,
            "hdf5_replay_mode": args.hdf5_replay_mode,
            "mapping": _rel(args.mapping),
            "hdf5_gripper_start_frame": args.hdf5_gripper_start_frame,
            "hdf5_gripper_end_frame": args.hdf5_gripper_end_frame,
            "hdf5_gripper_max_frames": args.hdf5_gripper_max_frames,
            "trace_contact_pairs": args.trace_contact_pairs,
            "trace_disable_usd_updates": args.trace_disable_usd_updates,
            "reset_after_object_creation": False,
            "min_contact_motion": args.min_contact_motion,
            "max_object_displacement": args.max_object_displacement,
        },
        "outputs": {"json": _rel(json_path), "csv": _rel(csv_path), "markdown": _rel(md_path)},
    }
    _write_json(json_path, payload)

    try:
        from isaacsim import SimulationApp

        app_config = dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG)
        app_config["fast_shutdown"] = False
        _app = SimulationApp(app_config)
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        import omni.usd

        stage_utils.open_stage(str(Path(args.stage_usd).resolve()))
        World.clear_instance()
        world = World(stage_units_in_meters=0.01, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
        paths = FINGER_PROXY_PATHS[args.side]
        art = world.scene.add(SingleArticulation(prim_path=paths["articulation"], name=f"{args.side}_vx300s"))
        world.reset()
        _apply_gravity(world, args.gravity)
        _apply_arm_gains(art, None, None)

        hdf5_target_sequence: list[np.ndarray] | None = None
        hdf5_gripper_summary: dict[str, Any] | None = None
        if args.hdf5_gripper_episode:
            qpos = _load_hdf5_qpos(
                args.hdf5_gripper_episode,
                start=args.hdf5_gripper_start_frame,
                end=args.hdf5_gripper_end_frame,
                max_frames=args.hdf5_gripper_max_frames,
            )
            mapping = load_mapping(args.mapping) if args.hdf5_replay_mode == "left_arm_and_gripper" else None
            hdf5_target_sequence, hdf5_gripper_summary = _targets_from_hdf5_qpos(
                art=art,
                side=args.side,
                qpos=qpos,
                mapping=mapping,
                replay_mode=args.hdf5_replay_mode,
            )
            open_target = hdf5_target_sequence[0]
            open_values = hdf5_gripper_summary["first_target_values"]
            payload["inputs"]["control_mode"] = f"hdf5_{args.hdf5_replay_mode}_qpos_replay"
            payload["inputs"]["hdf5_gripper_summary"] = hdf5_gripper_summary
        else:
            open_target, open_values = _finger_targets(art, args.open_offset, args.limit_margin)
        _set_full_state(art, open_target)
        _set_full_target(art, open_target)
        pre_object_update_steps = max(args.settle_steps, 1)
        payload["inputs"]["pre_object_update_steps"] = pre_object_update_steps
        _set_finger_target_and_step(world, art, open_target, pre_object_update_steps)

        left_box = _bbox_row(stage, paths["left_finger"])
        right_box = _bbox_row(stage, paths["right_finger"])
        placement_left_box = dict(left_box)
        placement_right_box = dict(right_box)
        cross_side_proxy_overlap = _cross_side_proxy_overlap_summary(stage, args.side)
        gap = _gap_metrics(left_box, right_box)
        if not gap.get("bbox_pair_valid"):
            raise RuntimeError("Finger proxy bbox pair is invalid; cannot place contact object.")
        axis_name = str(gap["dominant_axis"])
        axis = {"x": 0, "y": 1, "z": 2}[axis_name]
        center = (np.asarray(left_box["center"], dtype=np.float64) + np.asarray(right_box["center"], dtype=np.float64)) * 0.5
        surface_gap = _surface_gap(left_box, right_box, axis)
        side_length = max(surface_gap * args.object_fill_fraction, 1e-4)
        object_placement_row: dict[str, Any] = {
            "mode": args.object_placement,
            "axis": axis_name,
            "clearance": args.object_clearance,
            "base_center": center.tolist(),
        }
        if args.object_placement == "moving_finger_surface" and args.moving_fingers != "both":
            moving_box = left_box if args.moving_fingers == "left" else right_box
            other_box = right_box if args.moving_fingers == "left" else left_box
            moving_center = np.asarray(moving_box["center"], dtype=np.float64)
            other_center = np.asarray(other_box["center"], dtype=np.float64)
            direction = 1.0 if other_center[axis] >= moving_center[axis] else -1.0
            moving_surface = float(moving_box["max"][axis] if direction > 0 else moving_box["min"][axis])
            center = np.asarray(moving_box["center"], dtype=np.float64)
            center[axis] = moving_surface + direction * (side_length * 0.5 + args.object_clearance)
            object_placement_row.update(
                {
                    "moving_finger": args.moving_fingers,
                    "other_finger": "right" if args.moving_fingers == "left" else "left",
                    "direction_toward_other_finger": direction,
                    "moving_surface": moving_surface,
                    "placed_center": center.tolist(),
                }
            )
        elif args.object_placement == "moving_finger_surface":
            object_placement_row["warning"] = "moving_finger_surface requires --moving-fingers left or right; used gap_center."
        object_path = "/World/phase43_passive_contact_cube"
        proxy_offset_rows = [
            _set_collision_offsets(stage, paths["left_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
            _set_collision_offsets(stage, paths["right_finger"], args.proxy_contact_offset, args.proxy_rest_offset),
        ]
        _create_passive_cube(
            world=world,
            stage=stage,
            path=object_path,
            center=center,
            side_length=side_length,
            mass=args.object_mass,
            creation_mode=args.object_creation,
            shape=args.object_shape,
            axis=args.object_axis,
            length_multiplier=args.object_length_multiplier,
            usd_path=args.object_usd,
            usd_prim_path=args.object_usd_prim_path,
        )
        object_offset_row = _set_object_collision_offsets(stage, object_path, args.object_contact_offset, args.object_rest_offset)
        trace_state = None
        first_contact_row: dict[str, Any] | None = None
        contact_pair_rows: list[dict[str, Any]] = []
        try:
            if args.trace_contact_pairs:
                trace_state = _begin_contact_pair_trace(stage, disable_usd_updates=args.trace_disable_usd_updates)
            # Do not reset after object creation: object placement is computed from
            # the current open-pose fingertip bboxes, and a later reset can move
            # the articulation back under the already-placed object.
            _apply_gravity(world, args.gravity)
            _set_full_state(art, open_target)
            _set_full_target(art, open_target)
            dof_names = list(art.dof_names)
            object_reset_box = _bbox_row(stage, object_path)
            object_reset_center = np.asarray(object_reset_box["center"], dtype=np.float64)
            rows: list[dict[str, Any]] = []
            max_displacement = 0.0
            finite_motion = True
            for step in range(args.settle_steps):
                _set_full_target(art, open_target)
                world.step(render=False)
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
                object_center = np.asarray(object_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
                displacement_from_reset = float(np.linalg.norm(object_center - object_reset_center))
                finite_motion = bool(
                    finite_motion and np.all(np.isfinite(object_center)) and np.isfinite(displacement_from_reset)
                )
                max_displacement = max(
                    max_displacement,
                    displacement_from_reset if np.isfinite(displacement_from_reset) else float("inf"),
                )
                step_contact_pairs = _read_contact_pairs(trace_state)
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": "settle", "step": step, **pair}
                        contact_pair_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                rows.append(
                    {
                        "phase": "settle",
                        "step": step,
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
                        "object_displacement": displacement_from_reset,
                        "left_finger_qpos": float(qpos[dof_names.index("left_finger")]),
                        "right_finger_qpos": float(qpos[dof_names.index("right_finger")]),
                        "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
                        **_axis_probe_row(
                            axis=axis,
                            left_box=left_box,
                            right_box=right_box,
                            object_box=object_box,
                            target_finger_box=left_box if args.moving_fingers != "right" else right_box,
                        ),
                    }
                )

            object_initial_box = _bbox_row(stage, object_path)
            object_initial_center = np.asarray(object_initial_box["center"], dtype=np.float64)
            object_latest_box = dict(object_initial_box)
            object_latest_center = object_initial_center.copy()
            object_settle_displacement = float(np.linalg.norm(object_initial_center - object_reset_center))
            if hdf5_target_sequence is not None:
                close_target = hdf5_target_sequence[-1]
                close_values = hdf5_gripper_summary["last_target_values"] if hdf5_gripper_summary else {}
                close_sequence = hdf5_target_sequence[1:]
                if args.close_steps is not None:
                    close_sequence = close_sequence[: args.close_steps]
            else:
                close_target, close_values = _finger_targets(art, args.close_offset, args.limit_margin)
                close_sequence = []
            if args.moving_fingers != "both" and hdf5_target_sequence is None:
                isolated_target = open_target.copy()
                isolated_target[dof_names.index(f"{args.moving_fingers}_finger")] = close_target[
                    dof_names.index(f"{args.moving_fingers}_finger")
                ]
                close_target = isolated_target
                close_values = {
                    "left_finger": float(close_target[dof_names.index("left_finger")]),
                    "right_finger": float(close_target[dof_names.index("right_finger")]),
                }
            close_step_count = len(close_sequence) if hdf5_target_sequence is not None else args.close_steps
            for step in range(close_step_count):
                if hdf5_target_sequence is not None:
                    step_target = close_sequence[step]
                elif args.closure_profile == "linear":
                    alpha = float(step + 1) / float(max(args.close_steps, 1))
                    step_target = open_target + alpha * (close_target - open_target)
                else:
                    step_target = close_target
                _set_full_target(art, step_target)
                world.step(render=False)
                qpos = np.asarray(art.get_joint_positions(), dtype=np.float64).reshape(-1)
                left_box = _bbox_row(stage, paths["left_finger"])
                right_box = _bbox_row(stage, paths["right_finger"])
                object_box = _bbox_row(stage, object_path)
                object_center = np.asarray(object_box.get("center", [np.nan, np.nan, np.nan]), dtype=np.float64)
                object_latest_box = dict(object_box)
                object_latest_center = object_center.copy()
                displacement = float(np.linalg.norm(object_center - object_initial_center))
                displacement_from_reset = float(np.linalg.norm(object_center - object_reset_center))
                finite_motion = bool(
                    finite_motion
                    and np.all(np.isfinite(object_center))
                    and np.isfinite(displacement)
                    and np.isfinite(displacement_from_reset)
                )
                max_displacement = max(
                    max_displacement,
                    displacement_from_reset if np.isfinite(displacement_from_reset) else float("inf"),
                )
                step_contact_pairs = _read_contact_pairs(trace_state)
                if step_contact_pairs:
                    for pair in step_contact_pairs:
                        contact_row = {"phase": "close", "step": step, **pair}
                        contact_pair_rows.append(contact_row)
                    if first_contact_row is None:
                        first_contact_row = dict(contact_pair_rows[-len(step_contact_pairs)])
                rows.append(
                    {
                        "phase": "close",
                        "step": step,
                        "object_center_x": float(object_center[0]),
                        "object_center_y": float(object_center[1]),
                        "object_center_z": float(object_center[2]),
                        "object_displacement": displacement,
                        "left_finger_qpos": float(qpos[dof_names.index("left_finger")]),
                        "right_finger_qpos": float(qpos[dof_names.index("right_finger")]),
                        "finger_center_distance": _gap_metrics(left_box, right_box).get("center_distance"),
                        **_axis_probe_row(
                            axis=axis,
                            left_box=left_box,
                            right_box=right_box,
                            object_box=object_box,
                            target_finger_box=left_box if args.moving_fingers != "right" else right_box,
                        ),
                    }
                )
        finally:
            _finish_contact_pair_trace(stage, trace_state)

        object_final_box = object_latest_box
        object_final_center = object_latest_center
        object_displacement = float(np.linalg.norm(object_final_center - object_initial_center))
        total_object_displacement = float(np.linalg.norm(object_final_center - object_reset_center))
        contact_motion_policy = (
            "not_required_for_bilateral_closure"
            if args.moving_fingers == "both"
            else "single_finger_push_requires_minimum_motion"
        )
        contact_motion_ok = bool(args.moving_fingers == "both" or object_displacement >= args.min_contact_motion)
        no_explosion_ok = bool(finite_motion and max_displacement <= args.max_object_displacement)
        overall_pass = bool(contact_motion_ok and no_explosion_ok)
        if args.moving_fingers == "both":
            expected_finger_paths = [paths["left_finger"], paths["right_finger"]]
        else:
            expected_finger_paths = [paths[f"{args.moving_fingers}_finger"]]
        contact_summary = _summarize_contact_pairs(
            contact_pair_rows=contact_pair_rows,
            object_path=object_path,
            expected_finger_paths=expected_finger_paths,
        )
        if args.moving_fingers == "both":
            target_contact_ok = bool(contact_summary["all_expected_fingers_target_contact_pair_found"])
        else:
            target_contact_ok = bool(contact_summary["target_contact_pair_found"])
        cross_side_overlap_blocks_gate = bool(args.moving_fingers == "both" and cross_side_proxy_overlap["overlap_detected"])
        trace_pair_ok = bool((not args.trace_contact_pairs) or (target_contact_ok and not cross_side_overlap_blocks_gate))
        overall_pass = bool(overall_pass and trace_pair_ok)
        if args.trace_contact_pairs:
            if cross_side_overlap_blocks_gate:
                contact_trace_status = "FAIL_CROSS_SIDE_PROXY_OVERLAP"
            elif not target_contact_ok:
                contact_trace_status = "FAIL_NO_TARGET_CONTACT"
            elif not no_explosion_ok:
                contact_trace_status = "FAIL_OBJECT_EJECTION"
            else:
                contact_trace_status = (
                    "PASS_SINGLE_FINGER_CONTACT_ISOLATION"
                    if args.moving_fingers != "both"
                    else "PASS_BILATERAL_CONTACT_CANDIDATE"
                )
        else:
            contact_trace_status = "NOT_TRACED"
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED_GATE",
                "overall_pass": overall_pass,
                "contact_trace_status": contact_trace_status,
                "open_target_values": open_values,
                "close_target_values": close_values,
                "hdf5_gripper_summary": hdf5_gripper_summary,
                "hdf5_gripper_replay_steps": len(close_sequence) if hdf5_target_sequence is not None else None,
                "finger_gap_axis": axis_name,
                "finger_surface_gap_open": surface_gap,
                "left_finger_placement_box": placement_left_box,
                "right_finger_placement_box": placement_right_box,
                "cross_side_proxy_overlap": cross_side_proxy_overlap,
                "left_finger_final_box": left_box,
                "right_finger_final_box": right_box,
                "object_path": object_path,
                "object_shape": args.object_shape,
                "object_axis": args.object_axis,
                "object_length_multiplier": args.object_length_multiplier,
                "object_usd": _rel(args.object_usd),
                "object_usd_prim_path": args.object_usd_prim_path,
                "object_placement": object_placement_row,
                "object_side_length_stage_units": side_length,
                "proxy_collision_offsets": proxy_offset_rows,
                "object_collision_offsets": object_offset_row,
                "object_reset_box": object_reset_box,
                "object_initial_box": object_initial_box,
                "object_final_box": object_final_box,
                "object_reset_center": object_reset_center.tolist(),
                "object_initial_center": object_initial_center.tolist(),
                "object_final_center": object_final_center.tolist(),
                "object_settle_displacement": object_settle_displacement,
                "object_displacement": object_displacement,
                "total_object_displacement": total_object_displacement,
                "max_object_displacement": max_displacement,
                "object_motion_finite": finite_motion,
                "contact_motion_policy": contact_motion_policy,
                "contact_motion_ok": contact_motion_ok,
                "no_explosion_ok": no_explosion_ok,
                "contact_pair_trace_enabled": bool(args.trace_contact_pairs),
                "contact_trace_disable_usd_updates": bool(args.trace_disable_usd_updates),
                "contact_trace_rigid_body_paths": trace_state["rigid_body_paths"] if trace_state else [],
                "first_contact_pair": first_contact_row,
                **contact_summary,
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "next_gate": "gripper_contact_with_task_shape" if overall_pass else "inspect_contact_geometry_or_finger_control",
            }
        )
        _write_csv(csv_path, rows)
        _write_json(json_path, payload)
        _write_markdown(md_path, _json_safe(payload))
        print(json.dumps({"status": payload["status"], "json": _rel(json_path), "markdown": _rel(md_path)}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if overall_pass else 3)
    except BaseException as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-25:],
            }
        )
        _write_json(json_path, payload)
        print(json.dumps({"status": payload["status"], "json": _rel(json_path), "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())

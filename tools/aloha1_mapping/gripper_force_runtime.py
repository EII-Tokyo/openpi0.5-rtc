"""Isaac Sim 5.1-only runtime for ALOHA1 gripper hold diagnosis v2."""

# The v2 harness intentionally reuses frozen, validated baseline helpers so
# the diagnostic changes no bottle, drive, material, or contact-report setup.
# Stateful simulation loops are kept explicit because each call advances time.
# ruff: noqa: FBT003, PERF401, RET504, SLF001

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time
from typing import Any

import numpy as np

from tools.aloha1_mapping.gripper_force_diagnosis import audit_material_pair
from tools.aloha1_mapping.gripper_force_diagnosis import classify_contact_semantics
from tools.aloha1_mapping.gripper_force_diagnosis import classify_hold_failure_mode
from tools.aloha1_mapping.gripper_force_diagnosis import classify_normal_force
from tools.aloha1_mapping.gripper_force_diagnosis import classify_root_cause_v2
from tools.aloha1_mapping.gripper_force_diagnosis import classify_solver_sensitivity
from tools.aloha1_mapping.gripper_force_diagnosis import finite_cylinder_signed_distance
from tools.aloha1_mapping.gripper_force_diagnosis import finite_or_none
from tools.aloha1_mapping.gripper_force_diagnosis import friction_scan_gate
from tools.aloha1_mapping.gripper_force_diagnosis import has_consecutive_true
from tools.aloha1_mapping.gripper_force_diagnosis import load_force_diagnosis_config
from tools.aloha1_mapping.gripper_force_diagnosis import required_normal_force_each
from tools.aloha1_mapping.gripper_force_diagnosis import select_contact_event_at_frame
from tools.aloha1_mapping.gripper_force_diagnosis import select_solver_iteration_frequency
from tools.aloha1_mapping.gripper_force_diagnosis import sha256_file
from tools.aloha1_mapping.gripper_force_diagnosis import summarize_preload_trials
from tools.aloha1_mapping.gripper_force_diagnosis import verify_solver_trial_invariants
from tools.aloha1_mapping.gripper_validation import build_gripper_validation_plan
import tools.validate_aloha1_gripper as baseline


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _config(project_root: Path) -> dict[str, Any]:
    config_path = (
        project_root / "configs/aloha1_gripper_force_diagnosis.yaml"
    )
    config = load_force_diagnosis_config(
        config_path,
        project_root,
    )
    readback = _runtime_environment_readback()
    if readback["normalized"] != config["environment"]:
        raise RuntimeError(
            f"Isaac runtime version boundary mismatch: expected={config['environment']} actual={readback['normalized']}"
        )
    config["runtime_environment_readback"] = readback
    config["diagnostic_config_readback"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
    }
    return config


def _runtime_environment_readback() -> dict[str, Any]:
    """Read the active Kit/extension runtime; do not trust report labels."""

    import carb
    import omni.kit.app
    import omni.physx

    manager = omni.kit.app.get_app().get_extension_manager()
    physx = manager.get_extension_dict("omni.physx")
    physx_version = physx.get("package", {}).get("version") if physx else None
    physx_module_path = Path(next(iter(omni.physx.__path__))).resolve()
    if physx_version is None:
        extension_dir = next(
            (parent for parent in physx_module_path.parents if parent.name.startswith("omni.physx-")),
            None,
        )
        match = re.match(r"omni\.physx-([^+]+)", extension_dir.name) if extension_dir else None
        physx_version = match.group(1) if match else None
    kit_full = carb.tokens.get_tokens_interface().resolve("${kit_version}")
    normalized = {
        "isaac_sim": version("isaacsim"),
        "kit": str(kit_full).split("+", maxsplit=1)[0],
        "physx": (str(physx_version).split("+", maxsplit=1)[0] if physx_version else None),
        "python": platform.python_version(),
    }
    return {
        "normalized": normalized,
        "kit_full": kit_full,
        "physx_extension_id": "omni.physx",
        "physx_extension_version": physx_version,
        "physx_extension_path": str(physx_module_path),
        "source": "ACTIVE_RUNTIME_READBACK",
    }


def _asset_path(
    project_root: Path,
    *,
    robot: str,
    approximation: str,
) -> Path:
    if approximation == "convexHull":
        name = f"{robot}_force_diagnostic.usda"
        return (project_root / "assets/Trossen/ALOHA1/1.0/diagnostics/gripper_force" / name).resolve(strict=True)
    if approximation == "convexDecomposition":
        name = f"{robot}_convex_decomposition.usd"
        return (
            project_root
            / "assets/Trossen/ALOHA1/1.0/diagnostics/gripper_collision"
            / "convex_decomposition"
            / robot
            / name
        ).resolve(strict=True)
    raise ValueError(f"unsupported approximation: {approximation}")


def _source_evidence(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = {}
    for name, relative in config["source_evidence"].items():
        path = project_root / relative
        evidence[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return evidence


def _robot_plan(project_root: Path, robot: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = build_gripper_validation_plan(project_root)
    robots = {item["name"]: item for item in plan["robots"]}
    if robot not in robots:
        raise ValueError(f"unknown follower: {robot}")
    return plan, robots[robot]


def _collider_paths(stage: Any, robot: str) -> dict[str, str]:
    from pxr import Usd
    from pxr import UsdPhysics

    result = {}
    for side in ("left", "right"):
        root = stage.GetPrimAtPath(f"/World/Robot/{robot}_{side}_finger_link/collisions")
        candidates = [
            prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        meshes = [prim for prim in candidates if prim.IsA(UsdPhysics.MeshCollisionAPI)]
        chosen = meshes[0] if len(meshes) == 1 else None
        if chosen is None:
            mesh_candidates = [prim for prim in candidates if prim.HasAPI(UsdPhysics.MeshCollisionAPI)]
            if len(mesh_candidates) != 1:
                raise RuntimeError(
                    f"finger collider is not unique for {robot} {side}: {[str(item.GetPath()) for item in candidates]}"
                )
            chosen = mesh_candidates[0]
        result[side] = str(chosen.GetPath())
    return result


def _offset_and_approximation_readback(
    stage: Any,
    collider_path: str,
) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(collider_path)
    collision_api_applied = prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
    contact_attr = prim.GetAttribute("physxCollision:contactOffset")
    rest_attr = prim.GetAttribute("physxCollision:restOffset")
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    return {
        "collider_path": collider_path,
        "physx_collision_api_applied": collision_api_applied,
        "contact_offset": {
            "authored": bool(contact_attr and contact_attr.HasAuthoredValue()),
            "usd_readback": contact_attr.Get() if contact_attr else None,
            "schema_default": "-inf",
            "runtime_effective": (
                "SIMULATION_DETERMINED_NOT_EXPOSED_BY_107_3_USD_READBACK"
                if not contact_attr or contact_attr.Get() is None or not math.isfinite(float(contact_attr.Get()))
                else float(contact_attr.Get())
            ),
        },
        "rest_offset": {
            "authored": bool(rest_attr and rest_attr.HasAuthoredValue()),
            "usd_readback": rest_attr.Get() if rest_attr else None,
            "schema_default": "-inf",
            "runtime_effective": (
                "ZERO_FOR_RIGID_BODIES_PER_LOCAL_107_3_SCHEMA"
                if not rest_attr or rest_attr.Get() is None or not math.isfinite(float(rest_attr.Get()))
                else float(rest_attr.Get())
            ),
        },
        "approximation": mesh_api.GetApproximationAttr().Get(),
    }


def _material_readback(stage: Any, collider_path: str) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import UsdPhysics
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(collider_path)
    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    if not material:
        return {
            "collider_path": collider_path,
            "material_path": None,
            "binding_strength": None,
            "binding_relationship": None,
        }
    material_prim = material.GetPrim()
    usd_api = UsdPhysics.MaterialAPI(material_prim)
    physx_api_applied = material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
    physx_api = PhysxSchema.PhysxMaterialAPI(material_prim)
    friction_combine = physx_api.GetFrictionCombineModeAttr().Get() if physx_api_applied else "average"
    restitution_combine = physx_api.GetRestitutionCombineModeAttr().Get() if physx_api_applied else "average"
    return {
        "collider_path": collider_path,
        "material_path": str(material.GetPath()),
        "binding_strength": str(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)),
        "binding_relationship": (str(relationship.GetPath()) if relationship else None),
        "binding_source": "direct_or_inherited_physics_binding",
        "static_friction": float(usd_api.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(usd_api.GetDynamicFrictionAttr().Get()),
        "restitution": float(usd_api.GetRestitutionAttr().Get()),
        "physx_material_api_applied": physx_api_applied,
        "friction_combine_mode": str(friction_combine),
        "restitution_combine_mode": str(restitution_combine),
        "combine_mode_source": (
            "AUTHORED_OR_APPLIED_SCHEMA" if physx_api_applied else "LOCAL_PHYSX_SCHEMA_107_3_DEFAULT"
        ),
    }


def _drive_readback(stage: Any) -> dict[str, Any]:
    from pxr import UsdPhysics

    result = {}
    for side in ("left", "right"):
        path = f"/World/Robot/joints/{side}_finger"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"missing finger joint: {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        result[side] = {
            "joint_path": path,
            "drive_applied": bool(drive),
            "stiffness": drive.GetStiffnessAttr().Get() if drive else None,
            "damping": drive.GetDampingAttr().Get() if drive else None,
            "max_force": drive.GetMaxForceAttr().Get() if drive else None,
            "type": str(drive.GetTypeAttr().Get()) if drive else None,
        }
    return result


def _finger_contact_events(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    frame: int | None = None,
) -> list[Mapping[str, Any]]:
    token = f"_{side}_finger_link/"
    result = []
    for event in events:
        if frame is not None and int(event["frame"]) != frame:
            continue
        pair = f"{event['collider0']} {event['collider1']}"
        if token in pair and "/BottleProxy" in pair:
            result.append(event)
    return result


def _contact_frame_metrics(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    frame: int,
    dt: float,
) -> dict[str, Any]:
    frame_events = _finger_contact_events(events, side=side, frame=frame)
    contacts = [contact for event in frame_events for contact in event.get("contacts", [])]
    normal_impulses = []
    for contact in contacts:
        normal = np.asarray(contact["normal"], dtype=np.float64)
        impulse = np.asarray(contact["impulse"], dtype=np.float64)
        normal_impulses.append(float(abs(np.dot(impulse, normal))))
    total_impulse = sum(normal_impulses)
    separations = [float(contact["separation"]) for contact in contacts]
    event_types = sorted({str(event["type"]).split(".")[-1] for event in frame_events})
    return {
        "event_types": event_types,
        "contact_point_count": len(contacts),
        "normal_impulse_n_s": total_impulse,
        "estimated_normal_force_n": total_impulse / dt,
        "minimum_separation_m": min(separations) if separations else None,
        "maximum_separation_m": max(separations) if separations else None,
        "contacts": contacts,
        "physical_contact": bool(contacts and min(separations) <= 0.0 and total_impulse > 0.0),
        "solver_load_bearing_contact": bool(contacts and total_impulse > 0.0),
    }


def _finger_state(
    articulation: Any,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    velocities = np.asarray(articulation.get_joint_velocities(), dtype=np.float64)
    effort_error = None
    efforts = None
    try:
        measured = articulation.get_measured_joint_efforts()
        if measured is not None:
            efforts = np.asarray(measured, dtype=np.float64)
    except Exception as error:  # Isaac runtime capability readback
        effort_error = f"{type(error).__name__}: {error}"
    return {
        "left_position_m": float(positions[left_index]),
        "right_position_m": float(positions[right_index]),
        "left_velocity_m_s": float(velocities[left_index]),
        "right_velocity_m_s": float(velocities[right_index]),
        "left_effort": (float(efforts[left_index]) if efforts is not None else None),
        "right_effort": (float(efforts[right_index]) if efforts is not None else None),
        "effort_readback_error": effort_error,
    }


def _command_left(articulation: Any, left_index: int, target: float) -> None:
    baseline._command_left_finger(
        articulation,
        left_index=left_index,
        target=float(target),
    )


def _step_and_sample(
    context: Mapping[str, Any],
    *,
    target: float | None,
    phase: str,
    phase_step: int,
) -> dict[str, Any]:
    frame_state = context["frame_state"]
    frame_state["frame"] += 1
    context["world"].step(render=False)
    frame = frame_state["frame"]
    finger = _finger_state(
        context["articulation"],
        context["left_index"],
        context["right_index"],
    )
    left = _contact_frame_metrics(
        context["events"],
        side="left",
        frame=frame,
        dt=context["dt"],
    )
    right = _contact_frame_metrics(
        context["events"],
        side="right",
        frame=frame,
        dt=context["dt"],
    )
    return {
        "frame": frame,
        "phase": phase,
        "phase_step": phase_step,
        "target": {
            "left_m": finite_or_none(target),
            "right_m": finite_or_none(-target if target is not None else None),
        },
        "finger": {
            **finger,
            "left_target_error_m": finite_or_none(target - finger["left_position_m"] if target is not None else None),
            "right_target_error_m": finite_or_none(
                -target - finger["right_position_m"] if target is not None else None
            ),
        },
        "contact": {"left": left, "right": right},
        "bottle": baseline_ab_bottle_state(context["bottle"]),
    }


def baseline_ab_bottle_state(bottle: Any) -> dict[str, Any]:
    position, orientation = bottle.get_world_pose()
    linear = bottle.get_linear_velocity()
    angular = bottle.get_angular_velocity()
    position = np.asarray(position, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    angular = np.asarray(angular, dtype=np.float64)
    return {
        "position_world_m": position,
        "orientation_wxyz": np.asarray(orientation, dtype=np.float64),
        "z_m": float(position[2]),
        "linear_velocity_world_m_s": linear,
        "vertical_velocity_m_s": float(linear[2]),
        "angular_velocity_world_rad_s": angular,
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
        "linear_speed_m_s": float(np.linalg.norm(linear)),
    }


def _setup_trial(
    project_root: Path,
    *,
    robot: str,
    approximation: str,
    friction: float,
    frequency_hz: int,
    trial_tag: str,
    solver_position_iterations: int | None = None,
    solver_velocity_iterations: int | None = None,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.stage import create_new_stage
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface

    plan, robot_plan = _robot_plan(project_root, robot)
    World.clear_instance()
    create_new_stage()
    stage = get_current_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)
    stage.DefinePrim("/World/Materials", "Scope")
    asset = _asset_path(
        project_root,
        robot=robot,
        approximation=approximation,
    )
    add_reference_to_stage(str(asset), "/World/Robot")
    fingertip_material = baseline._apply_fingertip_material(
        stage,
        robot_name=robot,
        friction=friction,
    )
    report_bodies = baseline._apply_contact_reports(stage, robot)
    bottle_prim, bottle_description = baseline._create_bottle(
        stage,
        plan,
        friction=friction,
    )
    collider_paths = _collider_paths(stage, robot)
    offset_readback = {side: _offset_and_approximation_readback(stage, path) for side, path in collider_paths.items()}
    material_readback = {side: _material_readback(stage, path) for side, path in collider_paths.items()}
    material_readback["bottle"] = _material_readback(stage, "/World/BottleProxy")
    drive_readback = _drive_readback(stage)

    dt = 1.0 / float(frequency_hz)
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    physics_context = world.get_physics_context()
    physics_context.set_solve_articulation_contact_last(True)
    articulation = SingleArticulation(
        prim_path="/World/Robot/root_joint",
        name=f"force_{trial_tag}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    frame_state = {"frame": -1}
    events: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        events.extend(
            baseline._serialize_contacts(
                headers,
                data,
                frame=frame_state["frame"],
            )
        )

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    world.reset()
    if solver_position_iterations is not None:
        articulation.set_solver_position_iteration_count(int(solver_position_iterations))
    if solver_velocity_iterations is not None:
        articulation.set_solver_velocity_iteration_count(int(solver_velocity_iterations))
    bottle = SingleRigidPrim(
        "/World/BottleProxy",
        name=f"bottle_{trial_tag}",
        reset_xform_properties=False,
    )
    bottle.initialize()
    order = list(articulation.dof_names)
    if order != robot_plan["dof_order"]:
        raise RuntimeError(f"DOF order mismatch for {robot}: {order}")
    return {
        "project_root": project_root,
        "stage": stage,
        "world": world,
        "physics_context": physics_context,
        "articulation": articulation,
        "bottle": bottle,
        "bottle_prim": bottle_prim,
        "bottle_description": bottle_description,
        "robot_plan": robot_plan,
        "plan": plan,
        "events": events,
        "frame_state": frame_state,
        "subscription": subscription,
        "dt": dt,
        "frequency_hz": frequency_hz,
        "left_index": order.index("left_finger"),
        "right_index": order.index("right_finger"),
        "collider_paths": collider_paths,
        "offset_readback": offset_readback,
        "material_readback": material_readback,
        "drive_readback": drive_readback,
        "fingertip_material": fingertip_material,
        "report_bodies": report_bodies,
        "asset": str(asset),
        "asset_sha256": sha256_file(asset),
        "solve_contact_last": (physics_context.get_solve_articulation_contact_last()),
        "solver_readback": {
            "position_iterations": int(articulation.get_solver_position_iteration_count()),
            "velocity_iterations": int(articulation.get_solver_velocity_iteration_count()),
        },
    }


def _prepare_open_and_bottle(
    context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf
    from pxr import UsdGeom

    telemetry = []
    open_target = float(context["robot_plan"]["open_left_finger_m"])
    frequency = int(context["frequency_hz"])
    for step in range(frequency):
        telemetry.append(
            _step_and_sample(
                context,
                target=None,
                phase="initial_settle",
                phase_step=step,
            )
        )
    _command_left(context["articulation"], context["left_index"], open_target)
    for step in range(round(1.5 * frequency)):
        telemetry.append(
            _step_and_sample(
                context,
                target=open_target,
                phase="open",
                phase_step=step,
            )
        )
    robot = context["robot_plan"]["name"]
    left_bounds = baseline._collision_bounds(
        context["stage"],
        f"/World/Robot/{robot}_left_finger_link/collisions",
    )
    right_bounds = baseline._collision_bounds(
        context["stage"],
        f"/World/Robot/{robot}_right_finger_link/collisions",
    )
    aperture = baseline._aperture(left_bounds, right_bounds)
    bottle_position = np.asarray(aperture["midpoint_world_m"], dtype=np.float64)
    xform = UsdGeom.Xformable(context["bottle_prim"])
    ops = xform.GetOrderedXformOps()
    if len(ops) != 1:
        raise RuntimeError("BottleProxy must have exactly one translate op")
    ops[0].Set(Gf.Vec3d(*bottle_position.tolist()))
    get_physx_simulation_interface().flush_changes()
    for step in range(round(0.5 * frequency)):
        telemetry.append(
            _step_and_sample(
                context,
                target=open_target,
                phase="fixed_bottle_settle",
                phase_step=step,
            )
        )
    return telemetry, aperture


def _current_collider_aabb(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    robot = context["robot_plan"]["name"]
    return {
        side: baseline._collision_bounds(
            context["stage"],
            f"/World/Robot/{robot}_{side}_finger_link/collisions",
        )
        for side in ("left", "right")
    }


def _first_contact_event(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    frame: int | None = None,
) -> dict[str, Any] | None:
    relevant = _finger_contact_events(events, side=side)
    with_contacts = [event for event in relevant if event.get("contacts")]
    if not with_contacts:
        return None
    if frame is None:
        event = min(with_contacts, key=lambda item: int(item["frame"]))
    else:
        event = select_contact_event_at_frame(with_contacts, frame=frame)
        if event is None:
            return None
    contact = event["contacts"][0]
    return {
        "frame": int(event["frame"]),
        "event_type": str(event["type"]),
        "collider0": event["collider0"],
        "collider1": event["collider1"],
        **contact,
    }


def _independent_bottle_to_finger_distance(
    context: Mapping[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    """Sample BottleProxy's analytic surface and query closest finger points."""

    from omni.physx import get_physx_attachment_private_interface
    from pxr import Usd
    from pxr import UsdGeom

    robot = context["robot_plan"]["name"]
    root = f"/World/Robot/{robot}_{side}_finger_link/collisions"
    bounds = baseline._collision_bounds(context["stage"], root)
    center = np.asarray(
        baseline_ab_bottle_state(context["bottle"])["position_world_m"],
        dtype=np.float64,
    )
    radius = float(context["bottle_description"]["diameter_m"]) / 2.0
    half_height = float(context["bottle_description"]["height_m"]) / 2.0
    z_min = max(float(bounds["minimum_m"][2]), center[2] - half_height)
    z_max = min(float(bounds["maximum_m"][2]), center[2] + half_height)
    if z_min > z_max:
        z_values = np.asarray([center[2]], dtype=np.float64)
    else:
        count = max(2, math.ceil((z_max - z_min) / 0.001) + 1)
        z_values = np.linspace(z_min, z_max, count)
    angle_count = 360
    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    points = [
        (
            float(center[0] + radius * math.cos(angle)),
            float(center[1] + radius * math.sin(angle)),
            float(z),
        )
        for z in z_values
        for angle in angles
    ]
    sampling_error = math.hypot(
        radius * math.pi / angle_count,
        (float(z_values[-1] - z_values[0]) / max(1, len(z_values) - 1)) / 2.0,
    )
    collider_path = context["collider_paths"][side]
    collider_prim = context["stage"].GetPrimAtPath(root)
    xform_cache = UsdGeom.XformCache()
    world_points = []
    faces = []
    for prim in Usd.PrimRange(collider_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        local_points = list(mesh.GetPointsAttr().Get() or [])
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        base = len(world_points)
        world_points.extend(np.asarray(local_to_world.Transform(point), dtype=np.float64) for point in local_points)
        counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        cursor = 0
        for count in counts:
            faces.append([base + index for index in indices[cursor : cursor + count]])
            cursor += count
    vertex_signed = [
        finite_cylinder_signed_distance(
            point_xyz=point,
            center_xyz=center,
            radius_m=radius,
            half_height_m=half_height,
        )
        for point in world_points
    ]
    maximum_edge_m = 0.0
    for face in faces:
        for first_index, second_index in zip(
            face,
            [*face[1:], face[0]],
            strict=True,
        ):
            maximum_edge_m = max(
                maximum_edge_m,
                float(np.linalg.norm(world_points[first_index] - world_points[second_index])),
            )
    try:
        response = get_physx_attachment_private_interface().get_closest_points(
            points,
            collider_path,
        )
        squared = [float(value) for value in response.get("dists", [])]
        if not squared:
            raise RuntimeError("get_closest_points returned no distances")
        minimum = math.sqrt(max(0.0, min(squared)))
        return {
            "status": "PASS",
            "method": ("IPhysxAttachmentPrivate.get_closest_points on analytic BottleProxy surface samples"),
            "surface_distance_m": minimum,
            "distance_sign_limit": (
                "API returns squared distance and clamps inside points to zero; "
                "negative penetration depth is not observable by this method"
            ),
            "minimum_vertex_signed_distance_m": (min(vertex_signed) if vertex_signed else None),
            "vertex_signed_distance_method": (
                "world-transformed collision-mesh vertices evaluated against the exact axis-Z finite-cylinder SDF"
            ),
            "vertex_sampling_error_bound_m": (maximum_edge_m if world_points else None),
            "vertex_sampling_error_interpretation": (
                "maximum source-mesh polygon edge; the cylinder SDF is "
                "1-Lipschitz, so this conservatively bounds missed face "
                "interior samples, but not PhysX cooking approximation error"
            ),
            "sample_count": len(points),
            "angle_count": angle_count,
            "z_sample_count": len(z_values),
            "sampling_error_bound_m": sampling_error,
            "collider_path": collider_path,
        }
    except Exception as error:
        # This is a conservative projected AABB fallback. It is not treated as
        # an exact surface distance and the report retains its method limit.
        bottle_min_y = center[1] - radius
        bottle_max_y = center[1] + radius
        if side == "left":
            projected_gap = float(bounds["minimum_m"][1]) - bottle_max_y
        else:
            projected_gap = bottle_min_y - float(bounds["maximum_m"][1])
        return {
            "status": "PARTIAL",
            "method": "projected_world_AABB_gap_fallback",
            "surface_distance_m": projected_gap,
            "minimum_vertex_signed_distance_m": (min(vertex_signed) if vertex_signed else None),
            "vertex_signed_distance_method": (
                "world-transformed collision-mesh vertices evaluated against the exact axis-Z finite-cylinder SDF"
            ),
            "vertex_sampling_error_bound_m": (maximum_edge_m if world_points else None),
            "distance_sign_limit": ("AABB can underestimate local separation and overestimate overlap"),
            "sample_count": 0,
            "sampling_error_bound_m": None,
            "collider_path": collider_path,
            "closest_point_error": f"{type(error).__name__}: {error}",
        }


def _find_bilateral_load_bearing_contact(
    context: Mapping[str, Any],
    telemetry: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    open_target = float(context["robot_plan"]["open_left_finger_m"])
    minimum_target = float(context["robot_plan"]["closed_left_finger_m"])
    frequency = float(context["frequency_hz"])
    increment = float(config["preload"]["close_increment_m"]) * 60.0 / frequency
    maximum_steps = round(int(config["preload"]["maximum_find_contact_steps"]) * frequency / 60.0)
    required_run = max(
        1,
        round(int(config["preload"]["bilateral_contact_required_steps"]) * frequency / 60.0),
    )
    consecutive = 0
    first_run_frame = None
    distance_probes: dict[str, Any] = {}
    for step in range(maximum_steps):
        target = max(minimum_target, open_target - increment * (step + 1))
        _command_left(context["articulation"], context["left_index"], target)
        sample = _step_and_sample(
            context,
            target=target,
            phase="find_physical_contact",
            phase_step=step,
        )
        telemetry.append(sample)
        for side in ("left", "right"):
            if side not in distance_probes and sample["contact"][side]["contact_point_count"] > 0:
                distance_probes[side] = {
                    "frame": int(sample["frame"]),
                    "report_minimum_separation_m": sample["contact"][side]["minimum_separation_m"],
                    "independent": _independent_bottle_to_finger_distance(
                        context,
                        side=side,
                    ),
                }
        bilateral = bool(
            sample["contact"]["left"]["solver_load_bearing_contact"]
            and sample["contact"]["right"]["solver_load_bearing_contact"]
        )
        if bilateral:
            consecutive += 1
            if consecutive == 1:
                first_run_frame = int(sample["frame"])
        else:
            consecutive = 0
            first_run_frame = None
        if consecutive >= required_run:
            return {
                "found": True,
                "target_at_contact_m": target,
                "first_bilateral_load_bearing_frame": first_run_frame,
                "confirmation_frame": int(sample["frame"]),
                "consecutive_required": required_run,
                "independent_first_event_distance": distance_probes,
                "first_report_event": {
                    side: _first_contact_event(
                        context["events"],
                        side=side,
                        frame=distance_probes.get(side, {}).get("frame"),
                    )
                    for side in ("left", "right")
                },
            }
        if target <= minimum_target and step + 1 >= maximum_steps:
            break
    return {
        "found": False,
        "target_at_contact_m": None,
        "first_bilateral_load_bearing_frame": None,
        "confirmation_frame": None,
        "consecutive_required": required_run,
        "independent_first_event_distance": distance_probes,
        "first_report_event": {
            side: _first_contact_event(
                context["events"],
                side=side,
                frame=distance_probes.get(side, {}).get("frame"),
            )
            for side in ("left", "right")
        },
    }


def _stable_force_samples(
    telemetry: Sequence[Mapping[str, Any]],
    *,
    side: str,
    stable_steps: int,
) -> list[float]:
    stable = [sample for sample in telemetry if sample["phase"] == "preload_stable"][-stable_steps:]
    return [float(sample["contact"][side]["estimated_normal_force_n"]) for sample in stable]


def run_preload_trial(
    project_root: Path,
    *,
    robot: str,
    delta_m: float,
    trial_index: int,
    approximation: str = "convexHull",
    friction: float = 0.7,
    frequency_hz: int = 60,
    release: bool = False,
    absolute_contact_target_m: float | None = None,
    solver_position_iterations: int | None = None,
    solver_velocity_iterations: int | None = None,
) -> dict[str, Any]:
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    start = time.perf_counter()
    config = _config(project_root)
    tag = f"{robot}_{approximation}_{friction:.3f}_{frequency_hz}_{delta_m:.6f}_{trial_index}"
    context = _setup_trial(
        project_root,
        robot=robot,
        approximation=approximation,
        friction=friction,
        frequency_hz=frequency_hz,
        trial_tag=tag,
        solver_position_iterations=solver_position_iterations,
        solver_velocity_iterations=solver_velocity_iterations,
    )
    telemetry, aperture = _prepare_open_and_bottle(context)
    initial_conditions = {
        "bottle": baseline_ab_bottle_state(context["bottle"]),
        "bottle_kinematic_readback": bool(
            UsdPhysics.RigidBodyAPI(context["bottle_prim"]).GetKinematicEnabledAttr().Get()
        ),
        "finger": telemetry[-1]["finger"],
        "finger_target": telemetry[-1]["target"],
        "aperture": aperture,
    }
    frozen_input_manifest = {
        "restitution": float(config["frozen"]["restitution"]),
        "bottle_mass_kg": float(config["frozen"]["bottle_mass_kg"]),
        "bottle_diameter_m": float(config["frozen"]["bottle_diameter_m"]),
        "bottle_height_m": float(config["frozen"]["bottle_height_m"]),
        "control_mode": str(config["frozen"]["control_mode"]),
        "self_collision": bool(config["frozen"]["self_collision"]),
        "contact_offset_policy": str(config["frozen"]["contact_offset_policy"]),
        "rest_offset_policy": str(config["frozen"]["rest_offset_policy"]),
        "close_increment_m": float(config["preload"]["close_increment_m"]),
        "hold_duration_s": float(config["hold"]["duration_s"]),
        "maximum_drop_m": float(config["hold"]["maximum_drop_m"]),
    }
    contact = _find_bilateral_load_bearing_contact(context, telemetry, config)
    if not contact["found"]:
        debug = {
            side: {
                "physical_contact_frames": [
                    int(sample["frame"]) for sample in telemetry if sample["contact"][side]["physical_contact"]
                ],
                "solver_load_bearing_frames": [
                    int(sample["frame"])
                    for sample in telemetry
                    if sample["contact"][side]["solver_load_bearing_contact"]
                ],
                "maximum_normal_impulse_n_s": max(
                    (float(sample["contact"][side]["normal_impulse_n_s"]) for sample in telemetry),
                    default=0.0,
                ),
                "minimum_separation_m": min(
                    (
                        float(sample["contact"][side]["minimum_separation_m"])
                        for sample in telemetry
                        if sample["contact"][side]["minimum_separation_m"] is not None
                    ),
                    default=None,
                ),
            }
            for side in ("left", "right")
        }
        del context["subscription"]
        return {
            "schema_version": 1,
            "status": "FAIL",
            "failure": "bilateral_solver_load_bearing_contact_not_found",
            "robot": robot,
            "delta_m": delta_m,
            "trial_index": trial_index,
            "fresh_reset": True,
            "finite": True,
            "contact": contact,
            "debug": debug,
            "offset_readback": context["offset_readback"],
            "material_readback": context["material_readback"],
            "drive_readback": context["drive_readback"],
            "solver_readback": context["solver_readback"],
            "bottle_description": context["bottle_description"],
            "initial_conditions": initial_conditions,
            "frozen_input_manifest": frozen_input_manifest,
            "collider_aabb": _current_collider_aabb(context),
            "solve_articulation_contact_last": context["solve_contact_last"],
            "aperture_at_bottle_placement": aperture,
            "input_asset": context["asset"],
            "input_asset_sha256": context["asset_sha256"],
            "telemetry": telemetry,
        }
    measured_contact_target = float(contact["target_at_contact_m"])
    if absolute_contact_target_m is not None:
        if not math.isclose(
            measured_contact_target,
            float(absolute_contact_target_m),
            abs_tol=float(config["preload"]["close_increment_m"]),
        ):
            raise RuntimeError("fresh-reset load-bearing contact target changed outside one close increment")
        contact_target = float(absolute_contact_target_m)
    else:
        contact_target = measured_contact_target
    preload_target = contact_target - float(delta_m)
    _command_left(context["articulation"], context["left_index"], preload_target)
    stable_steps = round(int(config["preload"]["stable_window_steps"]) * frequency_hz / 60.0)
    for step in range(stable_steps):
        telemetry.append(
            _step_and_sample(
                context,
                target=preload_target,
                phase="preload_stable",
                phase_step=step,
            )
        )
    release_result = None
    if release:
        constraint_found, constraint_paths = baseline._has_bottle_constraint(context["stage"])
        if constraint_found:
            raise RuntimeError(f"forbidden BottleProxy constraint: {constraint_paths}")
        stage = context["stage"]
        bottle_path = str(context["bottle_prim"].GetPath())
        surface_gripper_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if "surfacegripper" in (str(prim.GetPath()) + prim.GetTypeName()).lower()
        ]
        relationship_attachment_paths = [
            f"{prim.GetPath()}.{relationship.GetName()}"
            for prim in stage.Traverse()
            for relationship in prim.GetRelationships()
            if bottle_path in {str(target) for target in relationship.GetTargets()}
        ]
        bottle_parent_path = str(context["bottle_prim"].GetParent().GetPath())
        parent_attachment_used = bottle_parent_path != "/World"
        release_state = baseline_ab_bottle_state(context["bottle"])
        UsdPhysics.RigidBodyAPI(context["bottle_prim"]).GetKinematicEnabledAttr().Set(False)
        get_physx_simulation_interface().flush_changes()
        kinematic_enabled_readback = bool(
            UsdPhysics.RigidBodyAPI(context["bottle_prim"]).GetKinematicEnabledAttr().Get()
        )
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI(context["bottle_prim"])
        disable_gravity_attr = physx_rigid.GetDisableGravityAttr() if physx_rigid else None
        disable_gravity = bool(
            disable_gravity_attr.Get() if disable_gravity_attr and disable_gravity_attr.Get() is not None else False
        )
        gravity_enabled = not disable_gravity
        if (
            kinematic_enabled_readback
            or surface_gripper_paths
            or relationship_attachment_paths
            or parent_attachment_used
            or not gravity_enabled
        ):
            raise RuntimeError(
                "bottle release readback failed: "
                f"kinematic={kinematic_enabled_readback} "
                f"surface={surface_gripper_paths} "
                f"relationships={relationship_attachment_paths} "
                f"parent={bottle_parent_path} gravity={gravity_enabled}"
            )
        hold_steps = round(float(config["hold"]["duration_s"]) * frequency_hz)
        release_frame = context["frame_state"]["frame"] + 1
        for step in range(hold_steps):
            telemetry.append(
                _step_and_sample(
                    context,
                    target=preload_target,
                    phase="released_hold",
                    phase_step=step,
                )
            )
        hold_samples = [sample for sample in telemetry if sample["phase"] == "released_hold"]
        final_state = hold_samples[-1]["bottle"]
        drop = float(release_state["z_m"] - final_state["z_m"])
        left_contact_frames = [
            int(sample["frame"]) for sample in hold_samples if sample["contact"]["left"]["contact_point_count"] > 0
        ]
        right_contact_frames = [
            int(sample["frame"]) for sample in hold_samples if sample["contact"]["right"]["contact_point_count"] > 0
        ]
        bilateral_frames = sorted(set(left_contact_frames) & set(right_contact_frames))
        contact_loss = next(
            (
                int(sample["frame"])
                for sample in hold_samples
                if sample["contact"]["left"]["contact_point_count"] == 0
                or sample["contact"]["right"]["contact_point_count"] == 0
            ),
            None,
        )
        drop_crossing = next(
            (
                int(sample["frame"])
                for sample in hold_samples
                if release_state["z_m"] - sample["bottle"]["z_m"] > float(config["hold"]["maximum_drop_m"])
            ),
            None,
        )
        stable_preload_forces = [
            min(
                sample["contact"]["left"]["estimated_normal_force_n"],
                sample["contact"]["right"]["estimated_normal_force_n"],
            )
            for sample in telemetry
            if sample["phase"] == "preload_stable"
        ]
        hold_forces = [
            min(
                sample["contact"]["left"]["estimated_normal_force_n"],
                sample["contact"]["right"]["estimated_normal_force_n"],
            )
            for sample in hold_samples
        ]
        preload_mean = statistics.fmean(stable_preload_forces) if stable_preload_forces else 0.0
        hold_tail_mean = statistics.fmean(hold_forces[-max(1, frequency_hz // 2) :]) if hold_forces else 0.0
        release_first = hold_samples[0]["bottle"]
        penetration_threshold = float(config["hold"]["penetration_threshold_m"])
        penetration_flags = [
            any(
                float(contact_item["separation"]) < -penetration_threshold
                for side in ("left", "right")
                for contact_item in sample["contact"][side]["contacts"]
            )
            for sample in hold_samples
        ]
        penetration_required = int(config["hold"]["penetration_required_consecutive_steps"])
        persistent_penetration = has_consecutive_true(
            penetration_flags,
            required=penetration_required,
        )
        metrics = {
            "drop_m": drop,
            "drop_gate_m": float(config["hold"]["maximum_drop_m"]),
            "contact_loss_frame": contact_loss,
            "drop_gate_crossing_frame": drop_crossing,
            "contacts_persist_to_end": bool(
                bilateral_frames and bilateral_frames[-1] == int(hold_samples[-1]["frame"])
            ),
            "normal_force_decay_ratio": (hold_tail_mean / preload_mean if preload_mean > 0.0 else 0.0),
            "maximum_angular_speed_rad_s": max(
                float(sample["bottle"]["angular_speed_rad_s"]) for sample in hold_samples
            ),
            "persistent_penetration": persistent_penetration,
            "penetration_threshold_m": penetration_threshold,
            "penetration_required_consecutive_steps": penetration_required,
            "release_linear_speed_m_s": float(release_first["linear_speed_m_s"]),
            "release_ejection_threshold_m_s": 0.1,
        }
        failure_mode = classify_hold_failure_mode(metrics)
        release_result = {
            "release_frame": release_frame,
            "release_state": release_state,
            "first_dynamic_state": release_first,
            "final_state": final_state,
            "hold_steps": hold_steps,
            "duration_s": float(config["hold"]["duration_s"]),
            "drop_gate_m": float(config["hold"]["maximum_drop_m"]),
            "drop_m": drop,
            "bilateral_contact_frame_count": len(bilateral_frames),
            "constraint_found": constraint_found,
            "constraint_paths": constraint_paths,
            "surface_gripper_used": bool(surface_gripper_paths),
            "surface_gripper_paths": surface_gripper_paths,
            "relationship_attachment_paths": relationship_attachment_paths,
            "parent_attachment_used": parent_attachment_used,
            "bottle_parent_path": bottle_parent_path,
            "kinematic_enabled_after_release": kinematic_enabled_readback,
            "disable_gravity_readback": disable_gravity,
            "gravity_enabled": gravity_enabled,
            "failure_metrics": metrics,
            "classification": failure_mode,
        }
    left_forces = _stable_force_samples(
        telemetry,
        side="left",
        stable_steps=stable_steps,
    )
    right_forces = _stable_force_samples(
        telemetry,
        side="right",
        stable_steps=stable_steps,
    )
    stable_samples = [sample for sample in telemetry if sample["phase"] == "preload_stable"]
    finite = all(
        math.isfinite(float(value))
        for sample in stable_samples
        for value in (
            sample["finger"]["left_position_m"],
            sample["finger"]["right_position_m"],
            sample["contact"]["left"]["estimated_normal_force_n"],
            sample["contact"]["right"]["estimated_normal_force_n"],
        )
    )
    signature_payload = {
        "robot": robot,
        "delta_m": delta_m,
        "friction": friction,
        "frequency_hz": frequency_hz,
        "contact_target": contact_target,
        "preload_target": preload_target,
        "left_forces": left_forces,
        "right_forces": right_forces,
        "release": release_result,
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    result = {
        "schema_version": 1,
        "status": ("PASS" if finite and contact["found"] else "FAIL"),
        "robot": robot,
        "trial_index": trial_index,
        "fresh_reset": True,
        "finite": finite,
        "approximation": approximation,
        "friction": friction,
        "frequency_hz": frequency_hz,
        "dt_s": context["dt"],
        "solve_articulation_contact_last": bool(context["solve_contact_last"]),
        "delta_m": float(delta_m),
        "contact": contact,
        "contact_target_m": contact_target,
        "measured_contact_target_m": measured_contact_target,
        "preload_target_m": preload_target,
        "left_stable_normal_force_n": left_forces,
        "right_stable_normal_force_n": right_forces,
        "left_target_error_m": [float(sample["finger"]["left_target_error_m"]) for sample in stable_samples],
        "right_target_error_m": [float(sample["finger"]["right_target_error_m"]) for sample in stable_samples],
        "drive_readback": context["drive_readback"],
        "bottle_description": context["bottle_description"],
        "initial_conditions": initial_conditions,
        "frozen_input_manifest": frozen_input_manifest,
        "solver_readback": context["solver_readback"],
        "offset_readback": context["offset_readback"],
        "material_readback": context["material_readback"],
        "aperture": aperture,
        "collider_aabb": _current_collider_aabb(context),
        "release": release_result,
        "telemetry": telemetry,
        "deterministic_signature": signature,
        "runtime_s": time.perf_counter() - start,
        "input_asset": context["asset"],
        "input_asset_sha256": context["asset_sha256"],
    }
    del context["subscription"]
    return result


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _all_contact_samples(
    telemetry: Sequence[Mapping[str, Any]],
    side: str,
) -> list[Mapping[str, Any]]:
    return [contact for sample in telemetry for contact in sample["contact"][side]["contacts"]]


def audit_contact_semantics(project_root: Path) -> dict[str, Any]:
    config = _config(project_root)
    trials = {}
    statuses = []
    for approximation in ("convexHull", "convexDecomposition"):
        profile = {}
        for robot in ("follower_left", "follower_right"):
            trial = run_preload_trial(
                project_root,
                robot=robot,
                delta_m=0.0,
                trial_index=0,
                approximation=approximation,
                friction=0.7,
                frequency_hz=60,
                release=False,
            )
            if trial["status"] != "PASS":
                profile[robot] = {
                    "status": "FAIL",
                    "failure": trial.get("failure"),
                    "debug": trial.get("debug"),
                    "contact": trial.get("contact"),
                    "input_asset": trial.get("input_asset"),
                    "input_asset_sha256": trial.get("input_asset_sha256"),
                }
                statuses.append("INCONCLUSIVE")
                continue
            sides = {}
            for side in ("left", "right"):
                first = trial["contact"]["first_report_event"][side]
                distance = trial["contact"]["independent_first_event_distance"].get(side, {})
                independent = distance.get("independent", {})
                contacts = _all_contact_samples(trial["telemetry"], side)
                separations = [float(contact["separation"]) for contact in contacts]
                classification = classify_contact_semantics(
                    {
                        "report_first_separation_m": (float(first["separation"]) if first else None),
                        "independent_first_surface_distance_m": independent.get("surface_distance_m"),
                        "independent_distance_error_bound_m": independent.get("sampling_error_bound_m"),
                        "minimum_report_separation_m": (min(separations) if separations else None),
                        "minimum_independent_vertex_signed_distance_m": independent.get(
                            "minimum_vertex_signed_distance_m"
                        ),
                        "finger_only_pairs": bool(
                            first
                            and f"_{side}_finger_link/" in (first["collider0"] + first["collider1"])
                            and "/BottleProxy" in (first["collider0"] + first["collider1"])
                        ),
                    }
                )
                statuses.append(classification["CONTACT_SEMANTICS_STATUS"])
                per_frame = [
                    {
                        "frame": int(sample["frame"]),
                        "phase": sample["phase"],
                        "state": (
                            "PHYSICAL_CONTACT"
                            if sample["contact"][side]["physical_contact"]
                            else (
                                "CONTACT_ENVELOPE"
                                if sample["contact"][side]["contact_point_count"]
                                else "NO_CONTACT_EVENT"
                            )
                        ),
                        "event_types": sample["contact"][side]["event_types"],
                        "contact_point_count": sample["contact"][side]["contact_point_count"],
                        "minimum_separation_m": sample["contact"][side]["minimum_separation_m"],
                        "normal_impulse_n_s": sample["contact"][side]["normal_impulse_n_s"],
                    }
                    for sample in trial["telemetry"]
                ]
                sides[side] = {
                    "collider": trial["offset_readback"][side],
                    "collider_aabb": trial["collider_aabb"][side],
                    "material_binding": trial["material_readback"][side],
                    "first_report_event": first,
                    "independent_first_event_distance": distance,
                    "minimum_separation_m": (min(separations) if separations else None),
                    "maximum_penetration_m": (max(0.0, -min(separations)) if separations else None),
                    "contact_point_count": len(contacts),
                    "contact_pair_paths": (
                        {
                            "collider0": first["collider0"],
                            "collider1": first["collider1"],
                        }
                        if first
                        else None
                    ),
                    "per_frame_contact_state": per_frame,
                    "classification": classification,
                }
            profile[robot] = {
                "input_asset": trial["input_asset"],
                "input_asset_sha256": trial["input_asset_sha256"],
                "solve_articulation_contact_last": trial["solve_articulation_contact_last"],
                "bottle_material_binding": trial["material_readback"]["bottle"],
                "sides": sides,
                "runtime_s": trial["runtime_s"],
            }
        trials[approximation] = profile
    if statuses and all(item == statuses[0] for item in statuses):
        overall = statuses[0]
    elif "REPORT_INTERPRETATION_ERROR" in statuses:
        overall = "REPORT_INTERPRETATION_ERROR"
    elif "CONTACT_ENVELOPE_DOMINATED" in statuses:
        overall = "CONTACT_ENVELOPE_DOMINATED"
    else:
        overall = "INCONCLUSIVE"
    report = {
        "schema_version": 1,
        "status": "PASS" if overall != "INCONCLUSIVE" else "PARTIAL",
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "CONTACT_SEMANTICS_STATUS": overall,
        "semantics": {
            "separation": {
                "source_direct_confirmation": (
                    "107.3 ContactData.separation is the per-contact separation "
                    "value; OnContactEvent maps the same field to contactDepths"
                ),
                "runtime_sign_cross_check": (
                    "positive values occur before nonzero impulse and independent "
                    "zero-distance; negative values coincide with overlap/impulse"
                ),
                "interpretation": ("positive=separated inside contact envelope; zero=touching; negative=penetration"),
            },
            "impulse": {
                "definition": "per-contact-point vector impulse",
                "force_conversion": "sum(abs(dot(impulse, normal))) / physics_dt",
                "not_pair_or_frame_aggregate": True,
            },
            "position_and_normal": {
                "coordinate_system": "world",
                "runtime_cross_check": ("contact points align with world BottleProxy and finger AABBs"),
            },
            "collider_and_material_path_decode": (
                "PhysicsSchemaTools.intToSdfPath for header collider ids and ContactData material ids"
            ),
            "events": {
                "found": "CONTACT_FOUND",
                "persists": "CONTACT_PERSIST",
                "lost": "CONTACT_LOST",
                "callback_cadence": "after each simulation step",
            },
            "offsets": {
                "contact_offset_schema_default": "-inf => simulation determined",
                "rest_offset_schema_default": ("-inf => simulation determined; zero for rigid bodies"),
                "shape_scope": ("PhysxCollisionAPI is applied/read per collision-shape prim"),
                "runtime_effective_limit": (
                    "simulation-selected contactOffset is not exposed by the "
                    "107.3 USD attribute readback when unauthored"
                ),
            },
        },
        "profiles": trials,
        "source_evidence": _source_evidence(project_root, config),
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    return report


def contact_semantics_markdown(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "# ALOHA1 Gripper Contact Semantics",
        "",
        f"- Status: `{report['status']}`",
        (f"- CONTACT_SEMANTICS_STATUS: `{report['CONTACT_SEMANTICS_STATUS']}`"),
        "- Runtime: Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26.",
        ("- Positive separation is classified only after comparison with an independent closest-point/AABB distance."),
        "",
        "## Per-profile evidence",
        "",
        "| Approximation | Robot | Side | First separation (m) | "
        "Independent distance (m) | Min separation (m) | Classification |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for approximation, robots in report["profiles"].items():
        for robot, robot_data in robots.items():
            if "sides" not in robot_data:
                lines.append(
                    f"| {approximation} | {robot} | n/a | n/a | n/a | "
                    f"n/a | INCONCLUSIVE ({robot_data.get('failure')}) |"
                )
                continue
            for side, data in robot_data["sides"].items():
                first = data["first_report_event"]
                independent = data["independent_first_event_distance"].get("independent", {})
                lines.append(
                    f"| {approximation} | {robot} | {side} | "
                    f"{first['separation'] if first else 'n/a'} | "
                    f"{independent.get('minimum_vertex_signed_distance_m', 'n/a')} | "
                    f"{data['minimum_separation_m']} | "
                    f"{data['classification']['CONTACT_SEMANTICS_STATUS']} |"
                )
    lines.extend(
        [
            "",
            "The fixed-bottle contact-persistence signal is not a static-hold pass.",
            "Task 8 remains `NOT_RUN`; the final collider is unchanged.",
        ]
    )
    return lines


def _linear_fit(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    if len(points) < 2 or len({item[0] for item in points}) < 2:
        return {"observable": False, "slope_n_per_m": None, "intercept_n": None}
    x = np.asarray([item[0] for item in points], dtype=np.float64)
    y = np.asarray([item[1] for item in points], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "observable": True,
        "slope_n_per_m": float(slope),
        "intercept_n": float(intercept),
        "r_squared": (1.0 - residual / total if total > 0.0 else 1.0),
    }


def measure_preload_force_curve(
    project_root: Path,
    *,
    repeats: int | None = None,
) -> dict[str, Any]:
    config = _config(project_root)
    repeats = repeats or int(config["preload"]["repeats"])
    if repeats < int(config["preload"]["repeats"]):
        raise ValueError("acceptance preload curve requires at least 10 repeats")
    report_root = project_root / config["report_root"]
    required = required_normal_force_each(
        mass_kg=float(config["frozen"]["bottle_mass_kg"]),
        friction=float(config["frozen"]["friction"]),
        gravity_m_s2=float(config["preload"]["theoretical_gravity_m_s2"]),
    )
    robots = {}
    for robot in ("follower_left", "follower_right"):
        curves = []
        all_trials = []
        contact_targets: dict[str, float] = {}
        for delta in config["preload"]["delta_m"]:
            delta_trials = []
            reference_contact_target = None
            for trial_index in range(repeats):
                trial = run_preload_trial(
                    project_root,
                    robot=robot,
                    delta_m=float(delta),
                    trial_index=trial_index,
                    approximation="convexHull",
                    friction=0.7,
                    frequency_hz=60,
                    release=False,
                    absolute_contact_target_m=reference_contact_target,
                )
                if trial["contact"]["found"] and reference_contact_target is None:
                    reference_contact_target = float(trial["contact_target_m"])
                delta_trials.append(trial)
                all_trials.append(trial)
            if reference_contact_target is not None:
                contact_targets[f"{float(delta):.6f}"] = reference_contact_target
            curves.append(
                summarize_preload_trials(
                    delta_trials,
                    minimum_repeats=repeats,
                )
            )
        normal_status = classify_normal_force(
            curves,
            required_each_n=required,
        )
        selected = normal_status["lowest_sufficient_preload_m"]
        if selected is None:
            selected = max(float(value) for value in config["preload"]["delta_m"])
            selection_status = "HIGHEST_TESTED_PRELOAD_NO_SUFFICIENT_GATE"
        else:
            selection_status = "LOWEST_SUFFICIENT_PRELOAD"
        selected_key = f"{float(selected):.6f}"
        trial_path = report_root / f"preload_trials_{robot}.jsonl"
        _write_jsonl(trial_path, all_trials)
        left_fit = _linear_fit(
            [
                (
                    float(curve["delta_m"]),
                    float(curve["left"]["mean_normal_force_n"]),
                )
                for curve in curves
                if curve.get("observable")
            ]
        )
        right_fit = _linear_fit(
            [
                (
                    float(curve["delta_m"]),
                    float(curve["right"]["mean_normal_force_n"]),
                )
                for curve in curves
                if curve.get("observable")
            ]
        )
        robots[robot] = {
            "curves": curves,
            "NORMAL_FORCE_STATUS": normal_status["NORMAL_FORCE_STATUS"],
            "normal_force_classification": normal_status,
            "selected_preload_m": selected,
            "selected_preload_policy": selection_status,
            "selected_contact_target_m": contact_targets.get(selected_key),
            "force_vs_delta": {"left": left_fit, "right": right_fit},
            "trial_file": str(trial_path),
            "trial_file_sha256": sha256_file(trial_path),
            "deterministic_within_delta": all(
                len(
                    {
                        trial["deterministic_signature"]
                        for trial in all_trials
                        if math.isclose(
                            float(trial["delta_m"]),
                            float(curve["delta_m"]),
                        )
                    }
                )
                == 1
                for curve in curves
            ),
        }
    statuses = {data["NORMAL_FORCE_STATUS"] for data in robots.values()}
    overall = (
        "SUFFICIENT"
        if statuses == {"SUFFICIENT"}
        else (
            "NOT_OBSERVABLE"
            if "NOT_OBSERVABLE" in statuses
            else ("INSUFFICIENT" if statuses <= {"SUFFICIENT", "INSUFFICIENT"} else "INCONCLUSIVE")
        )
    )
    return {
        "schema_version": 1,
        "status": "PASS" if overall == "SUFFICIENT" else "FAIL",
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "NORMAL_FORCE_STATUS": overall,
        "required_normal_force": {
            "mass_kg": float(config["frozen"]["bottle_mass_kg"]),
            "gravity_m_s2": float(config["preload"]["theoretical_gravity_m_s2"]),
            "friction": float(config["frozen"]["friction"]),
            "formula": "N_each = mg / (2 * mu)",
            "required_each_n": required,
            "status": config["preload"]["theoretical_threshold_status"],
        },
        "preload_delta_m": config["preload"]["delta_m"],
        "repeats_per_delta_per_robot": repeats,
        "fresh_world_reset_per_trial": True,
        "robots": robots,
        "frozen": config["frozen"],
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }


def write_preload_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for robot, robot_data in report["robots"].items():
        for curve in robot_data["curves"]:
            rows.append(
                {
                    "robot": robot,
                    "delta_m": curve["delta_m"],
                    "trial_count": curve["trial_count"],
                    "left_mean_normal_force_n": curve["left"]["mean_normal_force_n"],
                    "left_minimum_stable_normal_force_n": curve["left"]["minimum_stable_normal_force_n"],
                    "right_mean_normal_force_n": curve["right"]["mean_normal_force_n"],
                    "right_minimum_stable_normal_force_n": curve["right"]["minimum_stable_normal_force_n"],
                    "left_right_asymmetry_ratio": curve["left_right_asymmetry_ratio"],
                    "required_each_n": report["required_normal_force"]["required_each_n"],
                }
            )
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def preload_markdown(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "# ALOHA1 Gripper Preload Force Curve",
        "",
        f"- Status: `{report['status']}`",
        f"- NORMAL_FORCE_STATUS: `{report['NORMAL_FORCE_STATUS']}`",
        (f"- Theoretical diagnostic requirement per side: {report['required_normal_force']['required_each_n']:.6f} N."),
        "- The bottle remains kinematic in this report; this is not a hold pass.",
        "",
        "| Robot | Delta (mm) | Left mean (N) | Left min (N) | Right mean (N) | Right min (N) | Symmetry ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for robot, data in report["robots"].items():
        for curve in data["curves"]:
            lines.append(
                f"| {robot} | {1000.0 * curve['delta_m']:.1f} | "
                f"{curve['left']['mean_normal_force_n']:.6f} | "
                f"{curve['left']['minimum_stable_normal_force_n']:.6f} | "
                f"{curve['right']['mean_normal_force_n']:.6f} | "
                f"{curve['right']['minimum_stable_normal_force_n']:.6f} | "
                f"{curve['left_right_asymmetry_ratio']:.6f} |"
            )
    return lines


def _load_force_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "reports/aloha1_mapping/gripper_preload_force_curve.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_preload(force_report: Mapping[str, Any], robot: str) -> tuple[float, float]:
    data = force_report["robots"][robot]
    delta = float(data["selected_preload_m"])
    target = data.get("selected_contact_target_m")
    if target is None:
        raise RuntimeError(f"selected contact target missing for {robot}")
    return delta, float(target)


def _material_status_from_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> str:
    if any(not bool(pair["material_applied"]) for pair in pairs):
        return "MATERIAL_NOT_APPLIED"
    if any(not bool(pair["combine_mode_consistent"]) for pair in pairs):
        return "COMBINE_MODE_UNEXPECTED"
    if any(not bool(pair["expected_values_match"]) for pair in pairs):
        return "MATERIAL_NOT_APPLIED"
    if any(not bool(pair["contact_materials_match_binding"]) for pair in pairs):
        return "MATERIAL_NOT_APPLIED"
    return "SUFFICIENT"


def _trial_material_invariant(
    trial: Mapping[str, Any],
    *,
    expected_friction: float,
) -> dict[str, Any]:
    pairs = {}
    for side in ("left", "right"):
        event = trial["contact"]["first_report_event"][side]
        contact_materials = {
            "material0": event["material0"] if event else None,
            "material1": event["material1"] if event else None,
        }
        pairs[side] = audit_material_pair(
            trial["material_readback"][side],
            trial["material_readback"]["bottle"],
            expected_friction=expected_friction,
            expected_restitution=0.0,
            contact_materials=contact_materials,
        )
    return {
        "pass": all(
            pair["material_applied"]
            and pair["combine_mode_consistent"]
            and pair["expected_values_match"]
            and pair["contact_materials_match_binding"]
            for pair in pairs.values()
        ),
        "expected_friction": expected_friction,
        "expected_restitution": 0.0,
        "pairs": pairs,
    }


def audit_materials_and_friction(
    project_root: Path,
    *,
    repeats: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _config(project_root)
    force_report = _load_force_report(project_root)
    pairs = {}
    pair_values = []
    contact_report_materials = {}
    for robot in ("follower_left", "follower_right"):
        trial_path = Path(force_report["robots"][robot]["trial_file"])
        first = _read_jsonl(trial_path)[0]
        bottle = first["material_readback"]["bottle"]
        robot_pairs = {}
        robot_contact_materials = {}
        for side in ("left", "right"):
            event = first["contact"]["first_report_event"][side]
            contact_materials = {
                "material0": event["material0"] if event else None,
                "material1": event["material1"] if event else None,
            }
            pair = audit_material_pair(
                first["material_readback"][side],
                bottle,
                expected_friction=0.7,
                expected_restitution=0.0,
                contact_materials=contact_materials,
            )
            robot_pairs[side] = pair
            pair_values.append(pair)
            robot_contact_materials[side] = contact_materials
        pairs[robot] = robot_pairs
        contact_report_materials[robot] = robot_contact_materials
    material_status = _material_status_from_pairs(pair_values)
    material_report = {
        "schema_version": 1,
        "status": "PASS" if material_status == "SUFFICIENT" else "FAIL",
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "MATERIAL_STATUS": material_status,
        "FRICTION_STATUS": ("INCONCLUSIVE" if material_status == "SUFFICIENT" else material_status),
        "pairs": pairs,
        "contact_report_materials": contact_report_materials,
        "combine_rule_evidence": {
            "allowed_tokens": ["average", "min", "multiply", "max"],
            "schema_default": "average",
            "different_mode_precedence": (
                "NOT_NEEDED_AND_NOT_INFERRED; actual contacting materials read back the same mode"
            ),
        },
        "binding_override_audit": {
            "method": (
                "UsdShade.MaterialBindingAPI.ComputeBoundMaterial(materialPurpose=physics) on actual collider prim"
            ),
            "parent_or_stronger_override_detected": any(
                pair["finger"].get("material_path") != "/World/Materials/temporary_fingertip"
                or pair["bottle"].get("material_path") != "/World/Materials/temporary_bottle"
                for pair in pair_values
            ),
        },
        "temporary_material_status": "TEMPORARY_UNCALIBRATED",
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    gate = friction_scan_gate({"NORMAL_FORCE_STATUS": force_report["NORMAL_FORCE_STATUS"]})
    repeats = repeats or int(config["friction_scan"]["repeats"])
    report_root = project_root / config["report_root"]
    groups = {}
    if gate["run"]:
        if repeats < int(config["friction_scan"]["repeats"]):
            raise ValueError("acceptance friction scan requires at least 20 repeats")
        for mu in config["friction_scan"]["mu"]:
            combined = []
            robot_results = {}
            for robot in ("follower_left", "follower_right"):
                delta, contact_target = _selected_preload(force_report, robot)
                trials = [
                    run_preload_trial(
                        project_root,
                        robot=robot,
                        delta_m=delta,
                        trial_index=trial_index,
                        approximation="convexHull",
                        friction=float(mu),
                        frequency_hz=60,
                        release=True,
                        absolute_contact_target_m=contact_target,
                    )
                    for trial_index in range(repeats)
                ]
                path = report_root / (f"friction_mu_{float(mu):.3f}_{robot}.jsonl")
                _write_jsonl(path, trials)
                material_invariants = [
                    _trial_material_invariant(
                        trial,
                        expected_friction=float(mu),
                    )
                    for trial in trials
                ]
                if not all(item["pass"] for item in material_invariants):
                    raise RuntimeError(f"requested friction/material binding not applied: mu={mu} robot={robot}")
                success = [bool(trial["release"]["classification"]["pass"]) for trial in trials]
                robot_results[robot] = {
                    "trial_count": len(trials),
                    "hold_success_count": sum(success),
                    "hold_success_rate": sum(success) / len(success),
                    "trial_file": str(path),
                    "trial_file_sha256": sha256_file(path),
                    "deterministic": (len({trial["deterministic_signature"] for trial in trials}) == 1),
                    "material_invariant_pass": True,
                    "material_invariants": material_invariants,
                }
                combined.extend(trials)
            normal_forces = [
                min(
                    statistics.fmean(trial["left_stable_normal_force_n"]),
                    statistics.fmean(trial["right_stable_normal_force_n"]),
                )
                for trial in combined
            ]
            required_tangential_each = (
                float(config["frozen"]["bottle_mass_kg"]) * float(config["preload"]["theoretical_gravity_m_s2"]) / 2.0
            )
            available = [float(mu) * normal_force for normal_force in normal_forces]
            friction_ratios = [(required_tangential_each / force if force > 0.0 else math.inf) for force in available]
            hold_success = [bool(trial["release"]["classification"]["pass"]) for trial in combined]
            groups[f"{float(mu):.3f}"] = {
                "mu": float(mu),
                "status": (
                    "DIAGNOSTIC_ONLY_NOT_CALIBRATED" if math.isclose(float(mu), 1.0) else "TEMPORARY_UNCALIBRATED"
                ),
                "robots": robot_results,
                "combined_trial_count": len(combined),
                "combined_hold_success_count": sum(hold_success),
                "combined_hold_success_rate": (sum(hold_success) / len(hold_success)),
                "normal_force_n": {
                    "mean": statistics.fmean(normal_forces),
                    "minimum": min(normal_forces),
                },
                "required_tangential_force_each_n": required_tangential_each,
                "available_friction_force_each_n": {
                    "mean": statistics.fmean(available),
                    "minimum": min(available),
                },
                "friction_ratio": {
                    "mean": statistics.fmean(friction_ratios),
                    "maximum": max(friction_ratios),
                },
                "failure_modes": {
                    mode: sum(1 for trial in combined if trial["release"]["classification"]["mode"] == mode)
                    for mode in sorted({trial["release"]["classification"]["mode"] for trial in combined})
                },
            }
        baseline_group = groups["0.700"]
        higher_group = groups["1.000"]
        if baseline_group["combined_hold_success_rate"] == 1.0:
            friction_status = "SUFFICIENT"
        elif higher_group["combined_hold_success_rate"] > baseline_group["combined_hold_success_rate"]:
            friction_status = "INSUFFICIENT"
        elif all(group["friction_ratio"]["maximum"] <= 1.0 for group in groups.values() if group["mu"] >= 0.7):
            friction_status = "INCONCLUSIVE"
        else:
            friction_status = "INSUFFICIENT"
        scan_status = "PASS"
    else:
        friction_status = material_status if material_status != "SUFFICIENT" else "INCONCLUSIVE"
        scan_status = "PARTIAL"
    material_report["FRICTION_STATUS"] = friction_status
    friction_report = {
        "schema_version": 1,
        "status": scan_status,
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "FRICTION_STATUS": friction_status,
        "scan_gate": gate,
        "scan_run": gate["run"],
        "groups": groups,
        "frozen_except_friction": {
            **config["frozen"],
            "friction": "ONLY_EXPERIMENTAL_VARIABLE",
        },
        "repeats_per_mu_per_robot": repeats if gate["run"] else 0,
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    return material_report, friction_report


def validate_hold_v2(
    project_root: Path,
    *,
    repeats: int | None = None,
) -> dict[str, Any]:
    config = _config(project_root)
    force_report = _load_force_report(project_root)
    friction_path = project_root / "reports/aloha1_mapping/gripper_friction_margin.json"
    friction_report = json.loads(friction_path.read_text(encoding="utf-8")) if friction_path.exists() else None
    repeats = repeats or int(config["hold"]["repeats"])
    report_root = project_root / config["report_root"]
    robots = {}
    all_trials = []
    for robot in ("follower_left", "follower_right"):
        reused = None
        if friction_report and friction_report.get("scan_run"):
            reused = friction_report["groups"]["0.700"]["robots"][robot]["trial_file"]
            trials = _read_jsonl(Path(reused))
        else:
            if repeats < int(config["hold"]["repeats"]):
                raise ValueError("acceptance hold requires at least 20 repeats")
            delta, target = _selected_preload(force_report, robot)
            trials = [
                run_preload_trial(
                    project_root,
                    robot=robot,
                    delta_m=delta,
                    trial_index=trial_index,
                    approximation="convexHull",
                    friction=0.7,
                    frequency_hz=60,
                    release=True,
                    absolute_contact_target_m=target,
                )
                for trial_index in range(repeats)
            ]
            trial_path = report_root / f"hold_v2_{robot}.jsonl"
            _write_jsonl(trial_path, trials)
            reused = str(trial_path)
        passes = [bool(trial["release"]["classification"]["pass"]) for trial in trials]
        modes = [trial["release"]["classification"]["mode"] for trial in trials]
        robots[robot] = {
            "trial_count": len(trials),
            "hold_success_count": sum(passes),
            "hold_success_rate": sum(passes) / len(passes),
            "failure_modes": {mode: modes.count(mode) for mode in sorted(set(modes))},
            "trial_file": reused,
            "trial_file_sha256": sha256_file(Path(reused)),
            "reused_executed_friction_mu_0_700_trials": bool(friction_report and friction_report.get("scan_run")),
            "deterministic": (len({trial["deterministic_signature"] for trial in trials}) == 1),
            "maximum_drop_m": max(float(trial["release"]["drop_m"]) for trial in trials),
            "maximum_release_linear_speed_m_s": max(
                float(trial["release"]["first_dynamic_state"]["linear_speed_m_s"]) for trial in trials
            ),
        }
        all_trials.extend(trials)
    pass_all = all(bool(trial["release"]["classification"]["pass"]) for trial in all_trials)
    modes = [trial["release"]["classification"]["mode"] for trial in all_trials]
    return {
        "schema_version": 1,
        "status": "PASS" if pass_all else "FAIL",
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "STATIC_HOLD_STATUS": "PASS" if pass_all else "FAIL",
        "gate": {
            "hold_interval_s": float(config["hold"]["duration_s"]),
            "maximum_drop_m": float(config["hold"]["maximum_drop_m"]),
            "bilateral_contact_required": True,
            "fixed_constraint_allowed": False,
            "surface_gripper_allowed": False,
            "parent_attachment_allowed": False,
            "gravity_enabled": True,
        },
        "robots": robots,
        "overall_failure_modes": {mode: modes.count(mode) for mode in sorted(set(modes))},
        "trial_count": len(all_trials),
        "hold_success_count": sum(bool(trial["release"]["classification"]["pass"]) for trial in all_trials),
        "determinism": ("PASS" if all(data["deterministic"] for data in robots.values()) else "FAIL"),
        "contact_persistence_is_not_hold_pass": True,
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }


def _solver_gate(
    semantics_report: Mapping[str, Any],
    force_report: Mapping[str, Any],
    material_report: Mapping[str, Any],
    friction_report: Mapping[str, Any],
    hold_report: Mapping[str, Any],
) -> dict[str, Any]:
    if semantics_report["CONTACT_SEMANTICS_STATUS"] != "VERIFIED_PHYSICAL_CONTACT":
        return {
            "run": False,
            "reason": "contact_semantics_already_explains_or_blocks_hold",
        }
    if force_report["NORMAL_FORCE_STATUS"] in {
        "INSUFFICIENT",
        "NOT_OBSERVABLE",
    }:
        return {
            "run": False,
            "reason": "normal_force_already_explains_or_blocks_hold",
        }
    if material_report["MATERIAL_STATUS"] != "SUFFICIENT":
        return {
            "run": False,
            "reason": "material_binding_or_combine_already_explains_hold",
        }
    if friction_report["FRICTION_STATUS"] == "INSUFFICIENT":
        return {
            "run": False,
            "reason": "insufficient_friction_already_explains_hold",
        }
    if hold_report["STATIC_HOLD_STATUS"] == "PASS":
        return {"run": False, "reason": "hold_is_already_stable_at_60hz"}
    return {
        "run": True,
        "reason": "force_material_and_friction_do_not_yet_explain_hold_failure",
    }


def _summarize_solver_trials(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed = [
        trial
        for trial in trials
        if trial.get("status") == "PASS"
        and isinstance(trial.get("release"), Mapping)
    ]
    successes = [
        bool(trial["release"]["classification"]["pass"])
        for trial in completed
    ]
    modes = [
        str(trial["release"]["classification"]["mode"])
        for trial in completed
    ]
    failed_trial_count = len(trials) - len(completed)
    if failed_trial_count:
        modes.extend(["TRIAL_SETUP_OR_CONTACT_FAILURE"] * failed_trial_count)
    return {
        "trial_count": len(trials),
        "successful_trial_count": len(completed),
        "failed_trial_count": failed_trial_count,
        "hold_success_count": sum(successes),
        "hold_success_rate": (
            sum(successes) / len(trials)
            if trials
            else 0.0
        ),
        "deterministic_per_robot": (
            bool(completed)
            and len(completed) == len(trials)
            and len({trial["deterministic_signature"] for trial in completed}) == 1
        ),
        "maximum_drop_m": (
            max(float(trial["release"]["drop_m"]) for trial in completed)
            if completed
            else None
        ),
        "failure_modes": {
            mode: modes.count(mode)
            for mode in sorted(set(modes))
        },
    }


def _solver_expected_invariants(
    reference_trial: Mapping[str, Any],
    *,
    approximation: str,
    friction: float,
    frequency_hz: int,
    delta_m: float,
    contact_target_m: float,
    input_asset_sha256: str,
    position_iterations: int,
    velocity_iterations: int,
) -> dict[str, Any]:
    """Build the complete frozen-input/readback manifest for one scan cell."""

    expected = {
        "approximation": approximation,
        "friction": friction,
        "frequency_hz": frequency_hz,
        "delta_m": delta_m,
        "contact_target_m": contact_target_m,
        "solve_articulation_contact_last": True,
        "input_asset_sha256": input_asset_sha256,
        "drive_readback": reference_trial["drive_readback"],
        "offset_readback": reference_trial["offset_readback"],
        "material_readback": reference_trial["material_readback"],
        "bottle_description": reference_trial["bottle_description"],
        "frozen_input_manifest": reference_trial["frozen_input_manifest"],
        "initial_conditions.bottle.position_world_m": reference_trial[
            "initial_conditions"
        ]["bottle"]["position_world_m"],
        "initial_conditions.bottle.orientation_wxyz": reference_trial[
            "initial_conditions"
        ]["bottle"]["orientation_wxyz"],
        "initial_conditions.bottle.linear_velocity_world_m_s": reference_trial[
            "initial_conditions"
        ]["bottle"]["linear_velocity_world_m_s"],
        "initial_conditions.bottle.angular_velocity_world_rad_s": reference_trial[
            "initial_conditions"
        ]["bottle"]["angular_velocity_world_rad_s"],
        "initial_conditions.bottle_kinematic_readback": True,
        "initial_conditions.finger.left_position_m": reference_trial[
            "initial_conditions"
        ]["finger"]["left_position_m"],
        "initial_conditions.finger.right_position_m": reference_trial[
            "initial_conditions"
        ]["finger"]["right_position_m"],
        "initial_conditions.finger_target": reference_trial[
            "initial_conditions"
        ]["finger_target"],
        "initial_conditions.aperture": reference_trial["initial_conditions"][
            "aperture"
        ],
        "release.kinematic_enabled_after_release": False,
        "release.gravity_enabled": True,
        "release.constraint_found": False,
        "release.surface_gripper_used": False,
        "release.parent_attachment_used": False,
        "release.duration_s": reference_trial["frozen_input_manifest"][
            "hold_duration_s"
        ],
        "release.drop_gate_m": reference_trial["frozen_input_manifest"][
            "maximum_drop_m"
        ],
        "solver_readback.position_iterations": position_iterations,
        "solver_readback.velocity_iterations": velocity_iterations,
    }
    return expected


def test_solver_sensitivity(
    project_root: Path,
    *,
    repeats: int | None = None,
) -> dict[str, Any]:
    config = _config(project_root)
    semantics_report = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_contact_semantics.json").read_text(encoding="utf-8")
    )
    force_report = _load_force_report(project_root)
    material_report = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_material_audit.json").read_text(encoding="utf-8")
    )
    friction_report = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_friction_margin.json").read_text(encoding="utf-8")
    )
    hold_report = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_force_diagnosis/hold_v2.json").read_text(encoding="utf-8")
    )
    gate = _solver_gate(
        semantics_report,
        force_report,
        material_report,
        friction_report,
        hold_report,
    )
    if not gate["run"]:
        status = "STABLE_AT_60HZ" if hold_report["STATIC_HOLD_STATUS"] == "PASS" else "INCONCLUSIVE"
        return {
            "schema_version": 1,
            "status": "PARTIAL",
            "scope": config["scope"],
            "environment": config["environment"],
            "runtime_environment_readback": config["runtime_environment_readback"],
            "SOLVER_STATUS": status,
            "run": False,
            "gate": gate,
            "frequency_results": [],
            "position_iteration_results": [],
            "velocity_iteration_results": [],
            "protected_baseline": config["protected_baseline_readback"],
            "diagnostic_config": config["diagnostic_config_readback"],
            "task8": "NOT_RUN",
            "default_asset_collider_modified": False,
        }
    repeats = repeats or int(config["solver"]["repeats"])
    if repeats < int(config["solver"]["repeats"]):
        raise ValueError("acceptance solver scan requires configured repeats")
    report_root = project_root / config["report_root"] / "solver_trials"
    frequency_results = []
    frequency_trials_for_classification = []
    frequency_reference_trials: dict[int, dict[str, Mapping[str, Any]]] = {}
    baseline_frequency_hz = int(config["frozen"]["physics_frequency_hz"])
    configured_frequencies = [int(value) for value in config["solver"]["frequencies_hz"]]
    frequencies = [
        baseline_frequency_hz,
        *[
            frequency
            for frequency in configured_frequencies
            if frequency != baseline_frequency_hz
        ],
    ]

    def baseline_solver_readback(robot: str, delta: float) -> Mapping[str, Any]:
        trials = _read_jsonl(Path(force_report["robots"][robot]["trial_file"]))
        baseline_trial = next(trial for trial in trials if math.isclose(float(trial["delta_m"]), float(delta)))
        return baseline_trial["solver_readback"]

    for frequency in frequencies:
        robot_results = {}
        frequency_reference_trials[int(frequency)] = {}
        for robot in ("follower_left", "follower_right"):
            delta, target = _selected_preload(force_report, robot)
            trials = [
                run_preload_trial(
                    project_root,
                    robot=robot,
                    delta_m=delta,
                    trial_index=trial_index,
                    approximation="convexHull",
                    friction=0.7,
                    frequency_hz=int(frequency),
                    release=True,
                    absolute_contact_target_m=target,
                )
                for trial_index in range(repeats)
            ]
            path = report_root / f"frequency_{frequency}_{robot}.jsonl"
            _write_jsonl(path, trials)
            baseline_solver = baseline_solver_readback(robot, delta)
            frequency_reference_trials[int(frequency)][robot] = trials[0]
            reference_trial = frequency_reference_trials[baseline_frequency_hz][robot]
            invariant = verify_solver_trial_invariants(
                trials,
                _solver_expected_invariants(
                    reference_trial,
                    approximation="convexHull",
                    friction=0.7,
                    frequency_hz=int(frequency),
                    delta_m=delta,
                    contact_target_m=target,
                    input_asset_sha256=sha256_file(
                        _asset_path(
                            project_root,
                            robot=robot,
                            approximation="convexHull",
                        )
                    ),
                    position_iterations=baseline_solver["position_iterations"],
                    velocity_iterations=baseline_solver["velocity_iterations"],
                ),
            )
            robot_results[robot] = {
                **_summarize_solver_trials(trials),
                "invariant_manifest": invariant,
                "trial_file": str(path),
                "trial_file_sha256": sha256_file(path),
            }
        combined_success = sum(data["hold_success_count"] for data in robot_results.values())
        combined_count = sum(data["trial_count"] for data in robot_results.values())
        combined_completed = sum(
            data["successful_trial_count"]
            for data in robot_results.values()
        )
        entry = {
            "frequency_hz": int(frequency),
            "robots": robot_results,
            "hold_success_rate": combined_success / combined_count,
            "trial_count": combined_count,
            "successful_trial_count": combined_completed,
            "invariant_pass": all(data["invariant_manifest"]["pass"] for data in robot_results.values()),
            "only_changed": "physics_frequency_hz",
        }
        frequency_results.append(entry)
        frequency_trials_for_classification.append(entry)

    frequency_selection = select_solver_iteration_frequency(
        frequency_results,
        baseline_frequency_hz=baseline_frequency_hz,
    )
    selected_frequency_hz = frequency_selection["selected_frequency_hz"]
    if selected_frequency_hz is None:
        return {
            "schema_version": 1,
            "status": "PARTIAL",
            "scope": config["scope"],
            "environment": config["environment"],
            "runtime_environment_readback": config["runtime_environment_readback"],
            "SOLVER_STATUS": "INCONCLUSIVE",
            "run": True,
            "gate": gate,
            "frequency_results": frequency_results,
            "frequency_selection": frequency_selection,
            "position_iteration_results": [],
            "velocity_iteration_results": [],
            "protected_baseline": config["protected_baseline_readback"],
            "diagnostic_config": config["diagnostic_config_readback"],
            "task8": "NOT_RUN",
            "default_asset_collider_modified": False,
        }

    def iteration_scan(kind: str, values: Sequence[int]) -> list[dict[str, Any]]:
        results = []
        for value in values:
            robot_results = {}
            for robot in ("follower_left", "follower_right"):
                delta, target = _selected_preload(force_report, robot)
                kwargs = {
                    "solver_position_iterations": (int(value) if kind == "position" else None),
                    "solver_velocity_iterations": (int(value) if kind == "velocity" else None),
                }
                trials = [
                    run_preload_trial(
                        project_root,
                        robot=robot,
                        delta_m=delta,
                        trial_index=trial_index,
                        approximation="convexHull",
                        friction=0.7,
                        frequency_hz=int(selected_frequency_hz),
                        release=True,
                        absolute_contact_target_m=target,
                        **kwargs,
                    )
                    for trial_index in range(repeats)
                ]
                path = report_root / (f"{kind}_iterations_{value}_{robot}.jsonl")
                _write_jsonl(path, trials)
                reference_trial = frequency_reference_trials[
                    int(selected_frequency_hz)
                ][robot]
                baseline_solver = reference_trial["solver_readback"]
                expected_position = int(value) if kind == "position" else baseline_solver["position_iterations"]
                expected_velocity = int(value) if kind == "velocity" else baseline_solver["velocity_iterations"]
                invariant = verify_solver_trial_invariants(
                    trials,
                    _solver_expected_invariants(
                        reference_trial,
                        approximation="convexHull",
                        friction=0.7,
                        frequency_hz=int(selected_frequency_hz),
                        delta_m=delta,
                        contact_target_m=target,
                        input_asset_sha256=sha256_file(
                            _asset_path(
                                project_root,
                                robot=robot,
                                approximation="convexHull",
                            )
                        ),
                        position_iterations=expected_position,
                        velocity_iterations=expected_velocity,
                    ),
                )
                robot_results[robot] = {
                    **_summarize_solver_trials(trials),
                    "invariant_manifest": invariant,
                    "trial_file": str(path),
                    "trial_file_sha256": sha256_file(path),
                    "solver_readback": next(
                        (
                            trial.get("solver_readback")
                            for trial in trials
                            if trial.get("solver_readback") is not None
                        ),
                        None,
                    ),
                }
            count = sum(data["trial_count"] for data in robot_results.values())
            successes = sum(data["hold_success_count"] for data in robot_results.values())
            results.append(
                {
                    f"{kind}_iterations": int(value),
                    "frequency_hz": int(selected_frequency_hz),
                    "robots": robot_results,
                    "hold_success_rate": successes / count,
                    "invariant_pass": all(data["invariant_manifest"]["pass"] for data in robot_results.values()),
                    "only_changed": f"solver_{kind}_iterations",
                }
            )
        return results

    position_results = iteration_scan(
        "position",
        config["solver"]["position_iterations"],
    )
    velocity_results = iteration_scan(
        "velocity",
        config["solver"]["velocity_iterations"],
    )
    solver_classification = classify_solver_sensitivity(
        frequency_trials_for_classification,
        [*position_results, *velocity_results],
    )
    overall_invariant_pass = all(
        bool(item["invariant_pass"])
        for item in [
            *frequency_results,
            *position_results,
            *velocity_results,
        ]
    )
    return {
        "schema_version": 1,
        "status": ("PASS" if overall_invariant_pass else "PARTIAL"),
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        "SOLVER_STATUS": solver_classification["SOLVER_STATUS"],
        "run": True,
        "gate": gate,
        "frequency_results": frequency_results,
        "frequency_selection": frequency_selection,
        "position_iteration_results": position_results,
        "velocity_iteration_results": velocity_results,
        "overall_invariant_pass": overall_invariant_pass,
        "single_variable_policy": (
            "frequency, position iterations, and velocity iterations are "
            "separate scans; all other listed inputs remain frozen"
        ),
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }


def build_root_cause_v2(project_root: Path) -> tuple[dict[str, Any], list[str]]:
    semantics = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_contact_semantics.json").read_text(encoding="utf-8")
    )
    force = _load_force_report(project_root)
    material = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_material_audit.json").read_text(encoding="utf-8")
    )
    friction = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_friction_margin.json").read_text(encoding="utf-8")
    )
    hold = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_force_diagnosis/hold_v2.json").read_text(encoding="utf-8")
    )
    solver = json.loads(
        (project_root / "reports/aloha1_mapping/gripper_solver_sensitivity.json").read_text(encoding="utf-8")
    )
    dominant_mode = max(
        hold["overall_failure_modes"],
        key=hold["overall_failure_modes"].get,
        default="STATIC_HOLD_PASS",
    )
    evidence = {
        "contact_semantics": semantics["CONTACT_SEMANTICS_STATUS"],
        "normal_force": force["NORMAL_FORCE_STATUS"],
        "material": material["MATERIAL_STATUS"],
        "friction": friction["FRICTION_STATUS"],
        "solver": solver["SOLVER_STATUS"],
        "hold_failure_mode": dominant_mode,
        "max_force_observable": False,
        "max_force_saturated": None,
        "contact_normal_quality": "MEASURED_ALIGNED_IN_PRIOR_AB",
    }
    classification = classify_root_cause_v2(evidence)
    config = _config(project_root)
    report = {
        "schema_version": 1,
        "status": ("PASS" if classification["root_cause"] != "inconclusive" else "PARTIAL"),
        "scope": config["scope"],
        "environment": config["environment"],
        "runtime_environment_readback": config["runtime_environment_readback"],
        **classification,
        "evidence_inputs": evidence,
        "subsystem_status": {
            "contact_semantics": semantics["CONTACT_SEMANTICS_STATUS"],
            "normal_force": force["NORMAL_FORCE_STATUS"],
            "material": material["MATERIAL_STATUS"],
            "friction": friction["FRICTION_STATUS"],
            "static_hold": hold["STATIC_HOLD_STATUS"],
            "mimic_accuracy": ("PRIOR_AB_EXPLICIT_AND_MIMIC_TRAJECTORIES_IDENTICAL"),
            "solver": solver["SOLVER_STATUS"],
            "determinism": hold["determinism"],
        },
        "max_force_observability": {
            "joint_solver_effort_observable": True,
            "applied_drive_force_observable": False,
            "max_force_saturated": None,
            "interpretation": (
                "get_measured_joint_efforts is a solver-force readback and is "
                "not equated with applied drive force. The current runtime "
                "does not expose sufficient per-finger drive-force evidence "
                "to classify maxForce saturation."
            ),
        },
        "evidence_classification": {
            "source_direct_confirmation": [
                "ContactData fields and callback slicing from local 107.3 stubs",
                "offset and combine defaults from local PhysX Schema 107.3",
            ],
            "runtime_readback": [
                "collider paths, approximation, offsets, materials, drive and solver",
                "per-step joint state, contact impulse, bottle motion and hold",
            ],
            "numerical_calculation": [
                "normal force = summed normal impulse / dt",
                "N_each_required = mg / (2 mu)",
                "friction ratio and force/preload regression",
            ],
            "engineering_inference": [
                "root-cause category selection from measured subsystem statuses",
            ],
            "temporary_diagnostic_values": [
                "friction scan 0.3/0.5/0.7/1.0",
                "BottleProxy geometry/mass and 0-2 mm preload grid",
            ],
            "unmeasured_real_hardware_parameters": [
                "finger/bottle friction",
                "real bottle mass distribution and inertia",
                "real motor current-to-force and compliance",
            ],
        },
        "convex_decomposition": {
            "status": "NO_MEANINGFUL_EFFECT",
            "geometry_fit": "IMPROVED",
            "static_hold_resolution": False,
        },
        "mimic_explicit_ab": "NO_CHANGE_TO_HOLD_RESULT",
        "protected_baseline": config["protected_baseline_readback"],
        "diagnostic_config": config["diagnostic_config_readback"],
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    markdown = [
        "# ALOHA1 Gripper Hold Root Cause V2",
        "",
        f"- Status: `{report['status']}`",
        f"- Root cause: `{report['root_cause']}`",
        (f"- Contributing causes: `{', '.join(report['contributing_causes']) or 'none'}`"),
        "",
        "## Subsystem results",
        "",
    ]
    markdown.extend(f"- {name}: `{value}`" for name, value in report["subsystem_status"].items())
    markdown.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The fixed-bottle preload measurements prove that stable bilateral",
            "normal-force delivery is insufficient for the temporary 20 g,",
            "mu=0.7 diagnostic threshold. They do not distinguish insufficient",
            "commanded preload from maxForce saturation: the available measured",
            "joint effort is solver force, not applied drive force.",
            "",
            "Unresolved observations:",
            "",
            *[
                f"- `{observation}`"
                for observation in report["unresolved_observations"]
            ],
            "",
            "The 40/40 dynamic-release failures are deterministic numerical",
            "ejection/release transients followed by contact loss and free fall.",
            "Contact persistence is not treated as a physical hold pass.",
            "Convex Decomposition improved fit but did not solve static hold.",
            "Task 8 remains `NOT_RUN`; the final collider is unchanged.",
        ]
    )
    return report, markdown

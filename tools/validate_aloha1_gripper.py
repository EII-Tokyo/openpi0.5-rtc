#!/usr/bin/env python3
"""Headless contact and hold validation for Stationary ALOHA 1 grippers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.gripper_validation import build_gripper_validation_plan
from tools.aloha1_mapping.gripper_validation import canonicalize_contact_events
from tools.aloha1_mapping.gripper_validation import classify_gripper_trial
from tools.aloha1_mapping.gripper_validation import classify_repeat_determinism
from tools.aloha1_mapping.gripper_validation import summarize_contact_events


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(event: str, **details: Any) -> None:
    print(
        json.dumps(
            {"gripper_validation_event": event, **details},
            sort_keys=True,
            default=_json_default,
        ),
        flush=True,
    )


def _vector(value: Any) -> list[float]:
    return [float(component) for component in value]


def _path_from_id(value: Any) -> str:
    from pxr import PhysicsSchemaTools

    return str(PhysicsSchemaTools.intToSdfPath(value))


def _serialize_contacts(
    headers: Sequence[Any],
    data: Sequence[Any],
    *,
    frame: int,
) -> list[dict[str, Any]]:
    serialized = []
    for header in headers:
        contacts = []
        begin = int(header.contact_data_offset)
        end = begin + int(header.num_contact_data)
        for index in range(begin, end):
            contact = data[index]
            contacts.append(
                {
                    "position": _vector(contact.position),
                    "normal": _vector(contact.normal),
                    "impulse": _vector(contact.impulse),
                    "separation": float(contact.separation),
                    "face_index0": int(contact.face_index0),
                    "face_index1": int(contact.face_index1),
                    "material0": _path_from_id(contact.material0),
                    "material1": _path_from_id(contact.material1),
                }
            )
        serialized.append(
            {
                "frame": frame,
                "type": str(header.type),
                "actor0": _path_from_id(header.actor0),
                "actor1": _path_from_id(header.actor1),
                "collider0": _path_from_id(header.collider0),
                "collider1": _path_from_id(header.collider1),
                "stage_id": int(header.stage_id),
                "contacts": contacts,
            }
        )
    return serialized


def _collision_bounds(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"collision prim is unavailable: {prim_path}")
    colliders = [
        item for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if item.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders and prim.HasAPI(UsdPhysics.CollisionAPI):
        colliders = [prim]
    if not colliders:
        raise RuntimeError(f"no collider below prim: {prim_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.guide,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
    maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
    if not (np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)) and np.all(maximum > minimum)):
        raise RuntimeError(
            f"invalid world bounds for {prim_path}: minimum={minimum.tolist()}, maximum={maximum.tolist()}"
        )
    return {
        "root": prim_path,
        "colliders": [str(item.GetPath()) for item in colliders],
        "minimum_m": minimum,
        "maximum_m": maximum,
        "center_m": (minimum + maximum) / 2.0,
        "half_extent_m": (maximum - minimum) / 2.0,
    }


def _aperture(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_center = np.asarray(left["center_m"], dtype=np.float64)
    right_center = np.asarray(right["center_m"], dtype=np.float64)
    delta = right_center - left_center
    center_distance = float(np.linalg.norm(delta))
    if not math.isfinite(center_distance) or center_distance <= 1.0e-9:
        raise RuntimeError("finger collider centers do not define an axis")
    axis = delta / center_distance
    left_support = float(np.dot(np.abs(axis), np.asarray(left["half_extent_m"])))
    right_support = float(np.dot(np.abs(axis), np.asarray(right["half_extent_m"])))
    return {
        "closing_axis_world": axis,
        "center_distance_m": center_distance,
        "surface_gap_m": center_distance - left_support - right_support,
        "midpoint_world_m": (left_center + right_center) / 2.0,
    }


def _apply_fingertip_material(
    stage: Any,
    *,
    robot_name: str,
    friction: float,
) -> dict[str, Any]:
    from pxr import UsdPhysics
    from pxr import UsdShade

    material_path = "/World/Materials/temporary_fingertip"
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(friction))
    material_api.CreateDynamicFrictionAttr(float(friction))
    material_api.CreateRestitutionAttr(0.0)
    bound = []
    for side in ("left", "right"):
        path = f"/World/Robot/{robot_name}_{side}_finger_link/collisions"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"finger collision prim is unavailable: {path}")
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        bound.append(path)
    return {
        "path": material_path,
        "status": "TEMPORARY_UNCALIBRATED",
        "static_friction": friction,
        "dynamic_friction": friction,
        "restitution": 0.0,
        "bound_to": bound,
    }


def _create_bottle(
    stage: Any,
    plan: Mapping[str, Any],
    *,
    friction: float,
) -> tuple[Any, dict[str, Any]]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade

    bottle_plan = plan["bottle_proxy"]
    path = "/World/BottleProxy"
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(bottle_plan["diameter_m"]) / 2.0)
    cylinder.CreateHeightAttr(float(bottle_plan["height_m"]))
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)  # noqa: FBT003
    mass = UsdPhysics.MassAPI.Apply(cylinder.GetPrim())
    mass.CreateMassAttr(float(bottle_plan["mass_kg"]))
    report_api = PhysxSchema.PhysxContactReportAPI.Apply(cylinder.GetPrim())
    report_api.CreateThresholdAttr().Set(0.0)

    material_path = "/World/Materials/temporary_bottle"
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(friction))
    material_api.CreateDynamicFrictionAttr(float(friction))
    material_api.CreateRestitutionAttr(0.0)
    binding = UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim())
    binding.Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )
    return cylinder.GetPrim(), {
        "path": path,
        "shape": "cylinder",
        "axis": "Z",
        "diameter_m": float(bottle_plan["diameter_m"]),
        "height_m": float(bottle_plan["height_m"]),
        "mass_kg": float(bottle_plan["mass_kg"]),
        "status": bottle_plan["status"],
        "inertia_status": bottle_plan["inertia_status"],
        "material": {
            "path": material_path,
            "status": "TEMPORARY_UNCALIBRATED",
            "static_friction": friction,
            "dynamic_friction": friction,
            "restitution": 0.0,
        },
    }


def _apply_contact_reports(stage: Any, robot_name: str) -> list[str]:
    from pxr import PhysxSchema

    paths = []
    for side in ("left", "right"):
        path = f"/World/Robot/{robot_name}_{side}_finger_link"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"finger rigid body is unavailable: {path}")
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        paths.append(path)
    return paths


def _has_bottle_constraint(stage: Any) -> tuple[bool, list[str]]:
    from pxr import UsdPhysics

    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = [
            str(path)
            for relationship in (
                joint.GetBody0Rel(),
                joint.GetBody1Rel(),
            )
            for path in relationship.GetTargets()
        ]
        if any("/BottleProxy" in target for target in targets):
            joints.append(str(prim.GetPath()))
    return bool(joints), joints


def _finger_state(
    articulation: Any,
    left_index: int,
    right_index: int,
) -> dict[str, float]:
    positions = articulation.get_joint_positions()
    return {
        "left_finger_m": float(positions[left_index]),
        "right_finger_m": float(positions[right_index]),
    }


def _rigid_body_position(path: str) -> np.ndarray:
    from omni.physx import get_physx_interface

    transformation = get_physx_interface().get_rigidbody_transformation(path)
    if not transformation.get("ret_val", False):
        raise RuntimeError(f"PhysX rigid body transform unavailable: {path}")
    return np.asarray(transformation["position"], dtype=np.float64)


def _command_left_finger(
    articulation: Any,
    *,
    left_index: int,
    target: float,
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray([target], dtype=np.float32),
            joint_indices=np.asarray([left_index], dtype=np.int32),
        )
    )


def _step(
    world: Any,
    *,
    steps: int,
    frame_state: dict[str, int],
    articulation: Any,
    left_index: int,
    right_index: int,
    robot_name: str,
    friction: float,
    phase: str,
    curves: list[dict[str, Any]],
) -> None:
    for phase_step in range(steps):
        frame_state["frame"] += 1
        world.step(render=False)
        state = _finger_state(articulation, left_index, right_index)
        curves.append(
            {
                "robot": robot_name,
                "friction": friction,
                "phase": phase,
                "phase_step": phase_step,
                "frame": frame_state["frame"],
                **state,
            }
        )


def _run_trial(
    robot_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    friction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.stage import create_new_stage
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics

    World.clear_instance()
    _progress("trial_start", robot=robot_plan["name"], friction=friction)
    create_new_stage()
    stage = get_current_stage()
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)
    stage.DefinePrim("/World/Materials", "Scope")
    add_reference_to_stage(
        usd_path=robot_plan["asset"],
        prim_path="/World/Robot",
    )
    material = _apply_fingertip_material(
        stage,
        robot_name=robot_plan["name"],
        friction=friction,
    )
    report_bodies = _apply_contact_reports(stage, robot_plan["name"])
    bottle_prim, bottle = _create_bottle(
        stage,
        plan,
        friction=friction,
    )

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=plan["physics"]["physics_dt_s"],
        rendering_dt=plan["physics"]["physics_dt_s"],
    )
    physics_context = world.get_physics_context()
    physics_context.set_solve_articulation_contact_last(True)
    solve_contact_last = physics_context.get_solve_articulation_contact_last()

    articulation = SingleArticulation(
        prim_path="/World/Robot/root_joint",
        name=f"{robot_plan['name']}_{friction:g}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)

    frame_state = {"frame": -1}
    events: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        events.extend(
            _serialize_contacts(
                headers,
                data,
                frame=frame_state["frame"],
            )
        )

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    world.reset()
    _progress("world_reset", robot=robot_plan["name"], friction=friction)
    order = list(articulation.dof_names)
    if order != robot_plan["dof_order"]:
        raise RuntimeError(f"runtime DOF order mismatch for {robot_plan['name']}: {order}")
    left_index = order.index("left_finger")
    right_index = order.index("right_finger")
    motion = plan["motion"]
    curves: list[dict[str, Any]] = []

    _step(
        world,
        steps=motion["settle_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="settle",
        curves=curves,
    )
    start = _finger_state(articulation, left_index, right_index)
    _command_left_finger(
        articulation,
        left_index=left_index,
        target=robot_plan["open_left_finger_m"],
    )
    _step(
        world,
        steps=motion["open_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="open",
        curves=curves,
    )
    opened = _finger_state(articulation, left_index, right_index)
    _progress(
        "gripper_opened",
        robot=robot_plan["name"],
        friction=friction,
        state=opened,
    )
    left_path = f"/World/Robot/{robot_plan['name']}_left_finger_link/collisions"
    right_path = f"/World/Robot/{robot_plan['name']}_right_finger_link/collisions"
    open_left_bounds = _collision_bounds(stage, left_path)
    open_right_bounds = _collision_bounds(stage, right_path)
    _progress(
        "open_collider_bounds",
        robot=robot_plan["name"],
        friction=friction,
        left=open_left_bounds,
        right=open_right_bounds,
    )
    open_aperture = _aperture(open_left_bounds, open_right_bounds)

    bottle_position = np.asarray(
        open_aperture["midpoint_world_m"],
        dtype=np.float64,
    )
    bottle_xform = UsdGeom.Xformable(bottle_prim)
    bottle_ops = bottle_xform.GetOrderedXformOps()
    if len(bottle_ops) != 1:
        raise RuntimeError(f"unexpected bottle xform op count: {len(bottle_ops)}")
    bottle_ops[0].Set(Gf.Vec3d(*bottle_position.tolist()))
    _progress(
        "bottle_placed",
        robot=robot_plan["name"],
        friction=friction,
        position=bottle_position,
    )
    get_physx_simulation_interface().flush_changes()
    _step(
        world,
        steps=motion["settle_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="fixed_bottle_settle",
        curves=curves,
    )
    placement_frame = frame_state["frame"]

    _command_left_finger(
        articulation,
        left_index=left_index,
        target=robot_plan["closed_left_finger_m"],
    )
    _step(
        world,
        steps=motion["close_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="close_fixed_bottle",
        curves=curves,
    )
    closed = _finger_state(articulation, left_index, right_index)
    _step(
        world,
        steps=motion["fixed_contact_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="fixed_contact",
        curves=curves,
    )
    fixed_contact_end_frame = frame_state["frame"]
    close_left_bounds = _collision_bounds(stage, left_path)
    close_right_bounds = _collision_bounds(stage, right_path)
    close_aperture = _aperture(close_left_bounds, close_right_bounds)
    fixed_events = [event for event in events if placement_frame <= int(event["frame"]) <= fixed_contact_end_frame]
    fixed_contact = summarize_contact_events(
        fixed_events,
        bottle_path_token="/BottleProxy",
        penetration_limit_m=plan["penetration"]["maximum_persistent_depth_m"],
        persistence_steps=plan["penetration"]["persistence_steps"],
    )
    _progress(
        "fixed_contact_complete",
        robot=robot_plan["name"],
        friction=friction,
        summary=fixed_contact,
    )

    constraint_found, constraint_paths = _has_bottle_constraint(stage)
    release_position = _rigid_body_position("/World/BottleProxy")
    UsdPhysics.RigidBodyAPI(bottle_prim).GetKinematicEnabledAttr().Set(False)  # noqa: FBT003
    get_physx_simulation_interface().flush_changes()
    release_frame = frame_state["frame"] + 1
    _step(
        world,
        steps=plan["released_hold"]["hold_steps"],
        frame_state=frame_state,
        articulation=articulation,
        left_index=left_index,
        right_index=right_index,
        robot_name=robot_plan["name"],
        friction=friction,
        phase="released_hold",
        curves=curves,
    )
    final_position = _rigid_body_position("/World/BottleProxy")
    _progress(
        "released_hold_complete",
        robot=robot_plan["name"],
        friction=friction,
        final_position=final_position,
    )
    release_position = np.asarray(release_position, dtype=np.float64)
    final_position = np.asarray(final_position, dtype=np.float64)
    drop_m = float(release_position[2] - final_position[2])
    displacement_m = float(np.linalg.norm(final_position - release_position))
    hold_events = [event for event in events if int(event["frame"]) >= release_frame]
    all_contact = summarize_contact_events(
        events,
        bottle_path_token="/BottleProxy",
        penetration_limit_m=plan["penetration"]["maximum_persistent_depth_m"],
        persistence_steps=plan["penetration"]["persistence_steps"],
    )

    open_target = float(robot_plan["open_left_finger_m"])
    closed_target = float(robot_plan["closed_left_finger_m"])
    readback_tolerance = float(motion["readback_tolerance_m"])
    mimic_tolerance = float(motion["mimic_tolerance_m"])
    sampled_states = (start, opened, closed)
    finite_state = all(math.isfinite(value) for state in sampled_states for value in state.values()) and all(
        math.isfinite(value) for value in final_position
    )
    limits_ok = all(
        closed_target - 1.0e-5 <= state["left_finger_m"] <= open_target + 1.0e-5
        and -open_target - 1.0e-5 <= state["right_finger_m"] <= -closed_target + 1.0e-5
        for state in sampled_states
    )
    mimic_ok = all(
        abs(
            state["right_finger_m"]
            - (robot_plan["mimic"]["multiplier"] * state["left_finger_m"] + robot_plan["mimic"]["offset"])
        )
        <= mimic_tolerance
        for state in sampled_states
    )
    metrics = {
        "solve_articulation_contact_last_ok": bool(solve_contact_last),
        "open_direction_ok": (
            opened["left_finger_m"] > start["left_finger_m"] and opened["right_finger_m"] < start["right_finger_m"]
        ),
        "close_direction_ok": (
            closed["left_finger_m"] < opened["left_finger_m"] and closed["right_finger_m"] > opened["right_finger_m"]
        ),
        "limits_ok": limits_ok,
        "readback_ok": (
            abs(opened["left_finger_m"] - open_target) <= readback_tolerance
            and opened["left_finger_m"] - closed["left_finger_m"] >= 0.001
        ),
        "mimic_ok": mimic_ok,
        "aperture_monotonic": (
            math.isfinite(open_aperture["surface_gap_m"])
            and math.isfinite(close_aperture["surface_gap_m"])
            and open_aperture["surface_gap_m"] > close_aperture["surface_gap_m"]
        ),
        "left_finger_contact": fixed_contact["left_finger_contact"],
        "right_finger_contact": fixed_contact["right_finger_contact"],
        "bilateral_contact_before_release": (
            fixed_contact["left_finger_contact"] and fixed_contact["right_finger_contact"]
        ),
        "impulses_finite": all_contact["impulses_finite"],
        "persistent_penetration": all_contact["persistent_penetration"],
        "unexpected_gripper_collision": all_contact["unexpected_gripper_collision"],
        "released_without_constraint": not constraint_found,
        "held_for_required_steps": (math.isfinite(drop_m) and drop_m <= plan["released_hold"]["max_drop_m"]),
        "finite_state": finite_state,
    }
    classification = classify_gripper_trial(
        metrics,
        hard_blockers=plan["hard_blockers"],
    )
    result = {
        **classification,
        "robot": robot_plan["name"],
        "friction": friction,
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "asset": robot_plan["asset"],
        "physics": {
            "solve_articulation_contact_last_requested": True,
            "solve_articulation_contact_last_readback": bool(solve_contact_last),
            "self_collision": False,
            "contact_rest_offsets_authored": False,
        },
        "material": material,
        "bottle": bottle,
        "contact_report_bodies": report_bodies,
        "states": {
            "start": start,
            "open": opened,
            "closed_against_fixed_bottle": closed,
        },
        "aperture": {
            "measurement": "world AABB support distance along measured closing axis",
            "open": open_aperture,
            "closed_against_fixed_bottle": close_aperture,
            "left_open_bounds": open_left_bounds,
            "right_open_bounds": open_right_bounds,
            "left_closed_bounds": close_left_bounds,
            "right_closed_bounds": close_right_bounds,
        },
        "fixed_bottle_contact": fixed_contact,
        "released_hold": {
            "release_frame": release_frame,
            "required_steps": plan["released_hold"]["hold_steps"],
            "required_time_s": plan["released_hold"]["hold_time_s"],
            "maximum_drop_m": plan["released_hold"]["max_drop_m"],
            "release_position_world_m": release_position,
            "final_position_world_m": final_position,
            "drop_m": drop_m,
            "total_displacement_m": displacement_m,
            "contact_event_count": len(hold_events),
            "constraint_found": constraint_found,
            "constraint_paths": constraint_paths,
            "support_surface": "NONE_BOTTLE_STARTED_SUSPENDED",
        },
        "contact_summary": all_contact,
        "event_count": len(events),
        "contact_subscription_active_during_trial": subscription is not None,
    }
    del subscription
    return result, events, curves


def _write_curves(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "robot",
        "friction",
        "phase",
        "phase_step",
        "frame",
        "left_finger_m",
        "right_finger_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curves)


def run(
    project_root: Path,
    *,
    report_dir: Path,
    friction_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    plan = build_gripper_validation_plan(project_root)
    report_path = report_dir / "gripper_validation.json"
    previous_signature = None
    if report_path.is_file():
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        previous_signature = previous_report.get("determinism", {}).get("current_signature")
    selected_friction = list(
        friction_values if friction_values is not None else plan["fingertip_material"]["friction_scan"]
    )
    if not selected_friction or any(not math.isfinite(value) or value < 0.0 for value in selected_friction):
        raise ValueError("friction values must be finite and non-negative")

    trials = []
    all_curves = []
    for robot in plan["robots"]:
        for friction in selected_friction:
            trial, events, curves = _run_trial(
                robot,
                plan,
                friction=float(friction),
            )
            trials.append(trial)
            all_curves.extend(curves)
            event_path = report_dir / (f"gripper_contact_events_{robot['name']}_mu_{float(friction):.3f}.json")
            _write_json(
                event_path,
                {
                    "schema_version": 1,
                    "robot": robot["name"],
                    "friction": float(friction),
                    "events": canonicalize_contact_events(events),
                },
            )
            trial["contact_event_report"] = str(event_path.resolve())

    per_robot = []
    for robot in plan["robots"]:
        robot_trials = [trial for trial in trials if trial["robot"] == robot["name"]]
        passing_interface = [trial for trial in robot_trials if trial["passed_interface_gate"]]
        per_robot.append(
            {
                "robot": robot["name"],
                "status": ("PARTIAL" if passing_interface else "FAIL"),
                "passing_interface_friction_values": [trial["friction"] for trial in passing_interface],
                "trial_statuses": [
                    {
                        "friction": trial["friction"],
                        "status": trial["status"],
                        "failed_checks": trial["failed_checks"],
                    }
                    for trial in robot_trials
                ],
            }
        )
    overall_status = "PARTIAL" if all(item["status"] == "PARTIAL" for item in per_robot) else "FAIL"
    curve_path = report_dir / "gripper_curves.csv"
    _write_curves(curve_path, all_curves)
    artifact_paths = sorted(
        [curve_path, *report_dir.glob("gripper_contact_events_*.json")],
        key=lambda path: str(path),
    )
    artifact_hashes = {str(path.resolve()): _sha256(path) for path in artifact_paths}
    signature_payload = {
        "trials": trials,
        "artifact_hashes": artifact_hashes,
    }
    current_signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()
    determinism = classify_repeat_determinism(
        previous_signature,
        current_signature,
    )
    determinism["artifact_hashes"] = artifact_hashes
    report = {
        "schema_version": 1,
        "status": overall_status,
        "scope": "Stationary ALOHA 1 follower grippers",
        "collider_variant": "A_CURRENT_STL_CONVEX_HULL_BASELINE",
        "plan": plan,
        "friction_values": selected_friction,
        "robots": per_robot,
        "trials": trials,
        "determinism": determinism,
        "acceptance_semantics": {
            "PASS": "interface and calibrated physics gates pass",
            "PARTIAL": (
                "machine interface/contact/hold gate passes for at least one "
                "temporary friction value, but measured calibration is blocked"
            ),
            "FAIL": "no temporary-friction trial passes the interface gate",
        },
        "optimization_gate": "BLOCKED",
    }
    _write_json(report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--friction",
        action="append",
        type=float,
        help="temporary coefficient; repeat to override the default scan",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve(strict=True)
    report_dir = args.report_dir.resolve() if args.report_dir is not None else project_root / "reports/aloha1_mapping"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        try:
            report = run(
                project_root,
                report_dir=report_dir,
                friction_values=args.friction,
            )
        except BaseException as error:
            _write_json(
                report_dir / "gripper_validation_failure.json",
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            traceback.print_exc()
            raise
    finally:
        app.close()
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run supplier-CAD follower-left fixed-contact and 20 g bottle hold Task 5."""

# Local Isaac Sim 5.1.0 / Kit 107.3.3 / PhysX 107.3.26 only.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import yaml

from tools.aloha1_mapping.cad_finger_task5_bottle import classify_hold_failure_mode
from tools.aloha1_mapping.cad_finger_task5_bottle import compute_hold_kinematics
from tools.aloha1_mapping.cad_finger_task5_bottle import evaluate_bottle_trial
from tools.aloha1_mapping.cad_finger_task5_bottle import summarize_bottle_trials
from tools.aloha1_mapping.gripper_collider_ab import canonical_signature
from tools.aloha1_mapping.gripper_validation import summarize_contact_events
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot
import tools.validate_aloha1_gripper as baseline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/aloha1_cad_finger_task5_bottle.yaml"
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle.json"
)
TRIAL_OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_bottle_trials.jsonl"
)
SCREENSHOT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_bottle/screenshots_raw"
)

ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
FINGER_DOF_NAMES = (
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
)
FINGER_LINKS = {
    side: f"/workcell/vx300s_left/vx300s_left_{side}_finger_link"
    for side in ("left", "right")
}
FINGER_COLLISION_ROOTS = {
    side: (
        f"{FINGER_LINKS[side]}/collisions/"
        f"diagnostic_supplier_cad_{side}_finger"
    )
    for side in ("left", "right")
}
FINGER_COLLIDER_MESHES = {
    side: f"{root}/mesh"
    for side, root in FINGER_COLLISION_ROOTS.items()
}
BOTTLE_PATH = "/workcell/Task5BottleSession/BottleProxy"
MATERIAL_ROOT = "/workcell/Task5BottleSession/Materials"
PHASES = ("open", "bilateral_contact", "release", "hold_end")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                )
                + "\n"
            )
    temporary.replace(path)


def _resolve_profile(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = (ROOT / config["diagnostic_stage"]["path"]).resolve(strict=True)
    parent = (ROOT / config["parent_stage"]["path"]).resolve(strict=True)
    source = (
        ROOT / config["approved_source_stage"]["path"]
    ).resolve(strict=True)
    if _sha256(parent) != config["parent_stage"]["sha256"]:
        raise RuntimeError("frozen parent diagnostic hash mismatch")
    if _sha256(source) != config["approved_source_stage"]["sha256"]:
        raise RuntimeError("approved supplier review Stage hash mismatch")
    return {
        "document": config,
        "stage": stage,
        "parent": parent,
        "source": source,
        "hashes": {
            "config": _sha256(config_path),
            "diagnostic_stage": _sha256(stage),
            "parent_stage": _sha256(parent),
            "approved_source_stage": _sha256(source),
        },
    }


def _material_readback(stage: Any, path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    api = UsdPhysics.MaterialAPI(prim)
    return {
        "path": path,
        "static_friction": float(api.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
        "restitution": float(api.GetRestitutionAttr().Get()),
        "status": "TEMPORARY_UNCALIBRATED",
    }


def _create_material(
    stage: Any,
    path: str,
    *,
    friction: float,
    restitution: float,
) -> Any:
    from pxr import UsdPhysics
    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(restitution)
    return material


def _bind_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _create_session_physics(
    stage: Any,
    frozen: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage.DefinePrim(MATERIAL_ROOT, "Scope")
    finger_material_path = f"{MATERIAL_ROOT}/TemporaryFingertip"
    bottle_material_path = f"{MATERIAL_ROOT}/TemporaryBottle"
    finger_material = _create_material(
        stage,
        finger_material_path,
        friction=float(frozen["friction"]),
        restitution=float(frozen["restitution"]),
    )
    bottle_material = _create_material(
        stage,
        bottle_material_path,
        friction=float(frozen["friction"]),
        restitution=float(frozen["restitution"]),
    )
    bound = []
    approximation = {}
    for side, path in FINGER_COLLIDER_MESHES.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"finger collider is missing: {path}")
        _bind_material(prim, finger_material)
        approximation[side] = prim.GetAttribute(
            "physics:approximation"
        ).Get()
        bound.append(path)
    report_bodies = []
    for path in FINGER_LINKS.values():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"finger rigid body is missing: {path}")
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        report_bodies.append(path)

    bottle = UsdGeom.Cylinder.Define(stage, BOTTLE_PATH)
    bottle.CreateAxisAttr(UsdGeom.Tokens.z)
    bottle.CreateRadiusAttr(float(frozen["bottle_diameter_m"]) / 2.0)
    bottle.CreateHeightAttr(float(frozen["bottle_height_m"]))
    bottle.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.85, 0.30)])
    bottle.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
    UsdPhysics.CollisionAPI.Apply(bottle.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(bottle.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)  # noqa: FBT003
    mass = UsdPhysics.MassAPI.Apply(bottle.GetPrim())
    mass.CreateMassAttr(float(frozen["bottle_mass_kg"]))
    bottle_report = PhysxSchema.PhysxContactReportAPI.Apply(bottle.GetPrim())
    bottle_report.CreateThresholdAttr().Set(0.0)
    _bind_material(bottle.GetPrim(), bottle_material)
    return bottle.GetPrim(), {
        "bottle": {
            "path": BOTTLE_PATH,
            "shape": "cylinder",
            "axis": "Z",
            "diameter_m": float(frozen["bottle_diameter_m"]),
            "height_m": float(frozen["bottle_height_m"]),
            "mass_kg": float(frozen["bottle_mass_kg"]),
            "kinematic_initial": True,
        },
        "materials": {
            "finger": _material_readback(stage, finger_material_path),
            "bottle": _material_readback(stage, bottle_material_path),
            "binding_strength": "weakerThanDescendants",
            "combine_mode": "SCHEMA_DEFAULT_UNAUTHORED",
        },
        "finger_material_bound_to": bound,
        "contact_report_bodies": [*report_bodies, BOTTLE_PATH],
        "approximation_readback": approximation,
        "contact_rest_offsets_authored": False,
    }


def _self_collision_readback(stage: Any) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(ARTICULATION_PATH)
    attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
    authored = attr.IsValid() and attr.HasAuthoredValueOpinion()
    value = attr.Get() if attr.IsValid() else False
    return {
        "attribute": "physxArticulation:enabledSelfCollisions",
        "authored": authored,
        "value": bool(value) if value is not None else False,
    }


def _finger_state(
    articulation: Any,
    left_index: int,
    right_index: int,
) -> dict[str, float]:
    positions = np.asarray(
        articulation.get_joint_positions(), dtype=np.float64
    )
    velocities = np.asarray(
        articulation.get_joint_velocities(), dtype=np.float64
    )
    return {
        "left_target_readback_m": float(positions[left_index]),
        "right_target_readback_m": float(positions[right_index]),
        "left_velocity_m_s": float(velocities[left_index]),
        "right_velocity_m_s": float(velocities[right_index]),
        "symmetric_residual_m": float(
            abs(positions[left_index] + positions[right_index])
        ),
    }


def _bottle_state(bottle: Any) -> dict[str, Any]:
    position, orientation = bottle.get_world_pose()
    linear = bottle.get_linear_velocity()
    angular = bottle.get_angular_velocity()
    position = np.asarray(position, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    angular = np.asarray(angular, dtype=np.float64)
    return {
        "position_world_m": position.tolist(),
        "orientation_wxyz": np.asarray(
            orientation, dtype=np.float64
        ).tolist(),
        "z_m": float(position[2]),
        "linear_velocity_world_m_s": linear.tolist(),
        "vertical_velocity_m_s": float(linear[2]),
        "angular_velocity_world_rad_s": angular.tolist(),
        "angular_speed_rad_s": float(np.linalg.norm(angular)),
    }


def _mesh_bounds(stage: Any, mesh_path: str) -> dict[str, Any]:
    """Read collision-mesh bounds independent of viewport visibility."""

    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"collision mesh is missing: {mesh_path}")
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    world_points = np.asarray(
        [list(transform.Transform(point)) for point in points],
        dtype=np.float64,
    )
    if (
        world_points.ndim != 2
        or world_points.shape[0] == 0
        or world_points.shape[1] != 3
        or not np.all(np.isfinite(world_points))
    ):
        raise RuntimeError(f"invalid collision mesh points: {mesh_path}")
    minimum = world_points.min(axis=0)
    maximum = world_points.max(axis=0)
    return {
        "root": str(prim.GetParent().GetPath()),
        "colliders": [mesh_path],
        "minimum_m": minimum,
        "maximum_m": maximum,
        "center_m": (minimum + maximum) / 2.0,
        "half_extent_m": (maximum - minimum) / 2.0,
        "method": "composed_mesh_points_times_local_to_world_transform",
        "point_count": int(world_points.shape[0]),
    }


def _command_all(
    articulation: Any,
    qpos: np.ndarray,
) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    articulation.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=qpos.astype(np.float32))
    )


def _step(
    world: Any,
    *,
    steps: int,
    phase: str,
    frame_state: dict[str, int],
    articulation: Any,
    bottle: Any,
    left_index: int,
    right_index: int,
    telemetry: list[dict[str, Any]],
) -> None:
    for phase_step in range(steps):
        frame_state["frame"] += 1
        world.step(render=False)
        telemetry.append(
            {
                "frame": frame_state["frame"],
                "time_s": float(frame_state["frame"] / 60.0),
                "phase": phase,
                "phase_step": phase_step,
                "finger": _finger_state(
                    articulation, left_index, right_index
                ),
                "bottle": _bottle_state(bottle),
            }
        )


def _side_contact_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    minimum_frame: int,
    dt: float,
) -> dict[str, Any]:
    finger_token = f"vx300s_left_{side}_finger_link"
    relevant = [
        event
        for event in events
        if int(event["frame"]) >= minimum_frame
        and BOTTLE_PATH
        in f"{event.get('collider0', '')}\n{event.get('collider1', '')}"
        and finger_token
        in f"{event.get('collider0', '')}\n{event.get('collider1', '')}"
    ]
    with_contacts = [event for event in relevant if event.get("contacts")]
    samples = [
        contact
        for event in with_contacts
        for contact in event.get("contacts", [])
    ]
    if not samples:
        return {
            "contact": False,
            "first_contact_frame": None,
            "contact_frame_count": 0,
            "contact_duration_s": 0.0,
            "contact_loss_frame": None,
            "contact_samples": 0,
            "physical_surface_contact": False,
        }
    first_reported_event = min(
        with_contacts, key=lambda item: int(item["frame"])
    )
    first_reported = first_reported_event["contacts"][0]
    physical_records = [
        (event, contact)
        for event in with_contacts
        for contact in event.get("contacts", [])
        if float(contact["separation"]) <= 0.0
    ]
    first_event, first = (
        min(physical_records, key=lambda item: int(item[0]["frame"]))
        if physical_records
        else (first_reported_event, first_reported)
    )
    last_event, last = (
        max(physical_records, key=lambda item: int(item[0]["frame"]))
        if physical_records
        else (first_reported_event, first_reported)
    )
    frames = sorted(
        {
            int(event["frame"])
            for event, _contact in physical_records
        }
    )
    lost = [
        int(item["frame"])
        for item in relevant
        if "LOST" in str(item.get("type", "")).upper()
        and int(item["frame"]) >= int(first_event["frame"])
    ]
    impulses = []
    separations = []
    for contact in samples:
        normal = np.asarray(contact["normal"], dtype=np.float64)
        impulse = np.asarray(contact["impulse"], dtype=np.float64)
        impulses.append(float(abs(np.dot(normal, impulse))))
        separations.append(float(contact["separation"]))
    first_normal = np.asarray(first["normal"], dtype=np.float64)
    first_impulse = np.asarray(first["impulse"], dtype=np.float64)
    first_normal_impulse = float(abs(np.dot(first_normal, first_impulse)))
    return {
        "contact": True,
        "physical_surface_contact": bool(physical_records),
        "first_reported_event": {
            "frame": int(first_reported_event["frame"]),
            "separation_m": float(first_reported["separation"]),
            "position_world_m": first_reported["position"],
            "normal": first_reported["normal"],
        },
        "first_contact_frame": int(first_event["frame"]),
        "first_contact_time_s": float(int(first_event["frame"]) * dt),
        "first_contact_paths": {
            "actor0": first_event["actor0"],
            "actor1": first_event["actor1"],
            "collider0": first_event["collider0"],
            "collider1": first_event["collider1"],
        },
        "first_contact": {
            "position_world_m": first["position"],
            "normal": first["normal"],
            "impulse_n_s": first["impulse"],
            "normal_impulse_n_s": first_normal_impulse,
            "estimated_normal_force_n": first_normal_impulse / dt,
            "separation_m": float(first["separation"]),
            "material0": first["material0"],
            "material1": first["material1"],
        },
        "last_contact": {
            "frame": int(last_event["frame"]),
            "position_world_m": last["position"],
            "normal": last["normal"],
            "impulse_n_s": last["impulse"],
            "separation_m": float(last["separation"]),
            "material0": last["material0"],
            "material1": last["material1"],
        },
        "contact_frame_count": len(frames),
        "contact_duration_s": float(len(frames) * dt),
        "contact_loss_frame": min(lost) if lost else None,
        "contact_samples": len(samples),
        "normal_impulse_n_s": {
            "minimum": min(impulses),
            "maximum": max(impulses),
            "mean": float(np.mean(impulses)),
            "finite": bool(np.all(np.isfinite(impulses))),
        },
        "estimated_normal_force_n": {
            "minimum": min(impulses) / dt,
            "maximum": max(impulses) / dt,
            "mean": float(np.mean(impulses)) / dt,
        },
        "separation_m": {
            "minimum": min(separations),
            "maximum": max(separations),
            "maximum_penetration_depth_m": max(
                0.0, -min(separations)
            ),
        },
    }


def _normal_quality(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    closing_axis: Sequence[float],
) -> dict[str, Any]:
    if not left.get("physical_surface_contact") or not right.get(
        "physical_surface_contact"
    ):
        return {"status": "FAIL", "reason": "bilateral_contact_missing"}
    axis = np.asarray(closing_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    left_normal = np.asarray(
        left["first_contact"]["normal"], dtype=np.float64
    )
    right_normal = np.asarray(
        right["first_contact"]["normal"], dtype=np.float64
    )
    return {
        "status": "MEASURED_NO_CALIBRATED_THRESHOLD",
        "left_abs_alignment_with_closing_axis": float(
            abs(np.dot(left_normal, axis))
        ),
        "right_abs_alignment_with_closing_axis": float(
            abs(np.dot(right_normal, axis))
        ),
        "left_right_normal_dot": float(
            np.dot(left_normal, right_normal)
        ),
    }


def _capture_viewport_to_file(
    app: Any,
    viewport: Any,
    destination: Path,
) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file

    helper = capture_viewport_to_file(
        viewport,
        file_path=str(destination),
    )
    previous_size = -1
    stable_updates = 0
    for _ in range(300):
        app.update()
        if not destination.exists():
            continue
        size = destination.stat().st_size
        if size > 0 and size == previous_size:
            stable_updates += 1
        else:
            stable_updates = 0
        previous_size = size
        if stable_updates >= 2:
            break
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"viewport capture did not create {destination}"
        )
    del helper


def _capture_phase(
    *,
    app: Any,
    viewport: Any,
    camera: Any,
    world: Any,
    destination: Path,
    phase: str,
    frame: int,
    stage_path: Path,
    stage_hash: str,
    camera_pose: Mapping[str, Any],
    articulation: Any,
    bottle: Any,
    contact_state: Mapping[str, Any],
    contact_annotations_world: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    from pxr import Sdf

    bottle_before = _bottle_state(bottle)
    joints_before = np.asarray(
        articulation.get_joint_positions(), dtype=np.float64
    )
    world.pause()
    camera.set_world_pose(
        position=np.asarray(camera_pose["position_world_m"]),
        orientation=np.asarray(camera_pose["orientation_wxyz"]),
        camera_axes="usd",
    )
    contact_projection = {}
    for side, contact in (contact_annotations_world or {}).items():
        point = np.asarray(
            contact["position_world_m"], dtype=np.float64
        )
        normal = np.asarray(contact["normal"], dtype=np.float64)
        normal_endpoint = point + 0.02 * normal
        pixels = np.asarray(
            camera.get_image_coords_from_world_points(
                np.asarray([point, normal_endpoint], dtype=np.float64)
            ),
            dtype=np.float64,
        )
        if pixels.shape != (2, 2) or not np.all(np.isfinite(pixels)):
            raise RuntimeError(
                f"invalid contact projection for {phase}/{side}: "
                f"{pixels.tolist()}"
            )
        contact_projection[side] = {
            "position_world_m": point.tolist(),
            "normal_world": normal.tolist(),
            "normal_endpoint_world_m": normal_endpoint.tolist(),
            "contact_pixel_xy": pixels[0].tolist(),
            "normal_endpoint_pixel_xy": pixels[1].tolist(),
            "normal_arrow_length_world_m": 0.02,
            "projection_method": (
                "isaacsim.sensors.camera.Camera."
                "get_image_coords_from_world_points"
            ),
        }
    viewport.camera_path = Sdf.Path(camera.prim_path)
    for _ in range(20):
        app.update()
    _capture_viewport_to_file(app, viewport, destination)
    bottle_after = _bottle_state(bottle)
    joints_after = np.asarray(
        articulation.get_joint_positions(), dtype=np.float64
    )
    capture_physics_state_unchanged = bool(
        np.allclose(joints_before, joints_after, atol=1.0e-12, rtol=0.0)
        and np.allclose(
            bottle_before["position_world_m"],
            bottle_after["position_world_m"],
            atol=1.0e-12,
            rtol=0.0,
        )
        and np.allclose(
            bottle_before["linear_velocity_world_m_s"],
            bottle_after["linear_velocity_world_m_s"],
            atol=1.0e-12,
            rtol=0.0,
        )
        and np.allclose(
            bottle_before["angular_velocity_world_rad_s"],
            bottle_after["angular_velocity_world_rad_s"],
            atol=1.0e-12,
            rtol=0.0,
        )
    )
    world.play()
    if not capture_physics_state_unchanged:
        raise RuntimeError(
            f"viewport capture advanced physics during {phase}"
        )
    actual_position, actual_orientation = camera.get_world_pose(
        camera_axes="usd"
    )
    return validate_screenshot(
        destination.resolve(strict=True),
        artifact_root=destination.parent,
        phase="supplier_cad_task5_bottle",
        capture_name=phase,
        gate_status="PASS",
        camera={
            "view": "fixed_tip_end_contact",
            "resolution": [1280, 900],
            "capture_backend": (
                "omni.kit.viewport.utility.capture_viewport_to_file"
            ),
            "fixed_camera_for_all_phases": True,
            "position_world_m": np.asarray(actual_position).tolist(),
            "orientation_wxyz": np.asarray(
                actual_orientation
            ).tolist(),
            "target_world_m": list(camera_pose["target_world_m"]),
            "contact_projection": contact_projection,
        },
        simulation={
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "stage_absolute_path": str(stage_path),
            "stage_sha256": stage_hash,
            "robot": "follower_left",
            "collider_type": (
                "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC"
            ),
            "phase": phase,
            "frame": frame,
            "time_s": float(frame / 60.0),
            "joint_readback": np.asarray(
                articulation.get_joint_positions(), dtype=np.float64
            ).tolist(),
            "bottle": _bottle_state(bottle),
            "contact_state": dict(contact_state),
            "capture_physics_steps_added": 0,
            "capture_physics_state_unchanged": (
                capture_physics_state_unchanged
            ),
            "acceptance_boundary": (
                "AUXILIARY PHYSICS SCREENSHOT; MACHINE CONTACT, POSE, "
                "VELOCITY, DROP AND PENETRATION DATA ARE AUTHORITATIVE"
            ),
        },
    )


def _signature_payload(trial: Mapping[str, Any]) -> dict[str, Any]:
    contacts = {
        key: value
        for key, value in trial["contacts"].items()
        if key != "raw_events"
    }
    return {
        "metrics": trial["metrics"],
        "failure_mode": trial["failure_mode"],
        "states": trial["states"],
        "contacts": contacts,
        "released_hold": trial["released_hold"],
        "telemetry": trial["telemetry"],
    }


def _run_trial(
    *,
    app: Any,
    profile: Mapping[str, Any],
    trial_index: int,
    screenshot_root: Path | None,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_interface
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux
    from pxr import UsdPhysics

    from tools.validate_aloha_viper_cad_finger_task5_structure import _hide_non_target_visuals
    from tools.validate_aloha_viper_cad_finger_task5_structure import _set_view_visibility
    from tools.validate_aloha_viper_cad_finger_task5_structure import _world_points

    started = time.perf_counter()
    config = profile["document"]
    frozen = config["frozen"]
    motion = config["motion"]
    penetration = config["penetration"]
    stage_path = profile["stage"]
    stage_hash = profile["hashes"]["diagnostic_stage"]
    World.clear_instance()
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    stage = get_current_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        hidden_visuals = _hide_non_target_visuals(stage)
        hidden_visuals.extend(
            _set_view_visibility(stage, "base_oblique")
        )
        bottle_prim, session = _create_session_physics(stage, frozen)
        dome = UsdLux.DomeLight.Define(
            stage, "/workcell/Task5BottleSession/Dome"
        )
        dome.CreateIntensityAttr(700.0)
        dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
        key = UsdLux.DistantLight.Define(
            stage, "/workcell/Task5BottleSession/Key"
        )
        key.CreateIntensityAttr(1100.0)

    dt = 1.0 / float(frozen["physics_frequency_hz"])
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
        prim_path=ARTICULATION_PATH,
        name=f"supplier_cad_task5_bottle_{trial_index}",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    camera = None
    viewport = None
    if screenshot_root is not None:
        camera = Camera(
            prim_path="/workcell/Task5BottleSession/Camera",
            name=f"supplier_cad_task5_bottle_camera_{trial_index}",
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active Isaac viewport")

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

    physx_sim = get_physx_simulation_interface()
    subscription = physx_sim.subscribe_contact_report_events(on_contact)
    world.reset()
    if camera is not None:
        camera.initialize()
        camera.set_clipping_range(0.01, 10.0)
    bottle = SingleRigidPrim(
        BOTTLE_PATH,
        name=f"task5_bottle_{trial_index}",
        reset_xform_properties=False,
    )
    bottle.initialize()
    order = list(articulation.dof_names)
    if any(order.count(name) != 1 for name in FINGER_DOF_NAMES):
        raise RuntimeError(f"finger DOF identity mismatch: {order}")
    left_index = order.index(FINGER_DOF_NAMES[0])
    right_index = order.index(FINGER_DOF_NAMES[1])
    home = {
        "vx300s_left_waist": 0.0,
        "vx300s_left_shoulder": -0.96,
        "vx300s_left_elbow": 1.16,
        "vx300s_left_forearm_roll": 0.0,
        "vx300s_left_wrist_angle": -0.3,
        "vx300s_left_wrist_rotate": 0.0,
    }
    qpos = np.asarray(
        [home.get(name, 0.0) for name in order],
        dtype=np.float32,
    )
    qpos[left_index], qpos[right_index] = motion["open_targets_m"]
    articulation.set_joint_positions(qpos)
    _command_all(articulation, qpos)
    telemetry: list[dict[str, Any]] = []
    _step(
        world,
        steps=int(motion["settle_steps"]),
        phase="settle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    _command_all(articulation, qpos)
    _step(
        world,
        steps=int(motion["open_steps"]),
        phase="open",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    open_state = _finger_state(
        articulation, left_index, right_index
    )
    get_physx_interface().update_transformations(
        True, True, False, False  # noqa: FBT003
    )
    left_open = _mesh_bounds(stage, FINGER_COLLIDER_MESHES["left"])
    right_open = _mesh_bounds(stage, FINGER_COLLIDER_MESHES["right"])
    open_aperture = baseline._aperture(left_open, right_open)
    bottle_position = np.asarray(
        open_aperture["midpoint_world_m"], dtype=np.float64
    )
    bottle_xform = UsdGeom.Xformable(bottle_prim)
    ops = bottle_xform.GetOrderedXformOps()
    if len(ops) != 1:
        raise RuntimeError(f"unexpected bottle xform ops: {len(ops)}")
    ops[0].Set(Gf.Vec3d(*bottle_position.tolist()))
    physx_sim.flush_changes()
    _step(
        world,
        steps=int(motion["settle_steps"]),
        phase="fixed_bottle_settle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    placement_frame = frame_state["frame"]
    captures = []
    camera_pose = None
    if camera is not None and viewport is not None:
        points = np.concatenate(
            [
                _world_points(stage, FINGER_COLLIDER_MESHES[side])
                for side in ("left", "right")
            ]
        )
        target = np.mean(points, axis=0)
        source_structure = json.loads(
            (
                ROOT
                / "reports/aloha1_mapping/"
                "aloha_viper_cad_finger_task5_structure.json"
            ).read_text(encoding="utf-8")
        )
        source_pose = source_structure["camera_poses"]["tip_end"]
        source_offset = np.asarray(
            source_pose["position_world_m"], dtype=np.float64
        ) - np.asarray(source_pose["target_world_m"], dtype=np.float64)
        camera_pose = {
            "position_world_m": (target + source_offset).tolist(),
            "orientation_wxyz": source_pose["orientation_wxyz"],
            "target_world_m": target.tolist(),
        }
        captures.append(
            _capture_phase(
                app=app,
                viewport=viewport,
                camera=camera,
                world=world,
                destination=screenshot_root / "open_raw.png",
                phase="open",
                frame=frame_state["frame"],
                stage_path=stage_path,
                stage_hash=stage_hash,
                camera_pose=camera_pose,
                articulation=articulation,
                bottle=bottle,
                contact_state={
                    "bottle_kinematic": True,
                    "left": False,
                    "right": False,
                },
                contact_annotations_world=None,
            )
        )

    closed_qpos = qpos.copy()
    closed_qpos[left_index], closed_qpos[right_index] = (
        motion["closed_targets_m"]
    )
    _command_all(articulation, closed_qpos)
    _step(
        world,
        steps=int(motion["close_steps"]),
        phase="close_fixed_bottle",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    _step(
        world,
        steps=int(motion["fixed_contact_steps"]),
        phase="fixed_contact",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    closed_state = _finger_state(
        articulation, left_index, right_index
    )
    fixed_contact_end = frame_state["frame"]
    fixed_events = [
        event
        for event in events
        if placement_frame <= int(event["frame"]) <= fixed_contact_end
    ]
    fixed_summary = summarize_contact_events(
        fixed_events,
        bottle_path_token=BOTTLE_PATH,
        penetration_limit_m=float(
            penetration["maximum_persistent_depth_m"]
        ),
        persistence_steps=int(penetration["persistence_steps"]),
    )
    left_fixed = _side_contact_summary(
        fixed_events,
        side="left",
        minimum_frame=placement_frame,
        dt=dt,
    )
    right_fixed = _side_contact_summary(
        fixed_events,
        side="right",
        minimum_frame=placement_frame,
        dt=dt,
    )
    if camera is not None and viewport is not None:
        captures.append(
            _capture_phase(
                app=app,
                viewport=viewport,
                camera=camera,
                world=world,
                destination=screenshot_root
                / "bilateral_contact_raw.png",
                phase="bilateral_contact",
                frame=frame_state["frame"],
                stage_path=stage_path,
                stage_hash=stage_hash,
                camera_pose=camera_pose,
                articulation=articulation,
                bottle=bottle,
                contact_state={
                    "bottle_kinematic": True,
                    "left": left_fixed["physical_surface_contact"],
                    "right": right_fixed["physical_surface_contact"],
                    "fixed_stage_not_hold_pass": True,
                },
                contact_annotations_world={
                    "left": left_fixed["first_contact"],
                    "right": right_fixed["first_contact"],
                },
            )
        )

    constraint_found, constraint_paths = baseline._has_bottle_constraint(stage)
    release_state = _bottle_state(bottle)
    UsdPhysics.RigidBodyAPI(
        bottle_prim
    ).GetKinematicEnabledAttr().Set(False)  # noqa: FBT003
    physx_sim.flush_changes()
    release_frame = frame_state["frame"] + 1
    if camera is not None and viewport is not None:
        captures.append(
            _capture_phase(
                app=app,
                viewport=viewport,
                camera=camera,
                world=world,
                destination=screenshot_root / "release_raw.png",
                phase="release",
                frame=frame_state["frame"],
                stage_path=stage_path,
                stage_hash=stage_hash,
                camera_pose=camera_pose,
                articulation=articulation,
                bottle=bottle,
                contact_state={
                    "bottle_kinematic": False,
                    "left": left_fixed["physical_surface_contact"],
                    "right": right_fixed["physical_surface_contact"],
                    "constraint_found": constraint_found,
                },
                contact_annotations_world={
                    "left": left_fixed["first_contact"],
                    "right": right_fixed["first_contact"],
                },
            )
        )
    _step(
        world,
        steps=int(frozen["hold_steps"]),
        phase="released_static_hold",
        frame_state=frame_state,
        articulation=articulation,
        bottle=bottle,
        left_index=left_index,
        right_index=right_index,
        telemetry=telemetry,
    )
    final_state = _bottle_state(bottle)
    final_finger_state = _finger_state(
        articulation, left_index, right_index
    )
    release_telemetry = [
        item
        for item in telemetry
        if item["phase"] == "released_static_hold"
    ]
    hold_kinematics = compute_hold_kinematics(
        release_z_m=float(release_state["z_m"]),
        z_samples_m=[
            float(item["bottle"]["z_m"])
            for item in release_telemetry
        ],
        dt_s=dt,
    )
    drop_m = float(hold_kinematics["maximum_drop_m"])
    all_summary = summarize_contact_events(
        events,
        bottle_path_token=BOTTLE_PATH,
        penetration_limit_m=float(
            penetration["maximum_persistent_depth_m"]
        ),
        persistence_steps=int(penetration["persistence_steps"]),
    )
    left_all = _side_contact_summary(
        events,
        side="left",
        minimum_frame=placement_frame,
        dt=dt,
    )
    right_all = _side_contact_summary(
        events,
        side="right",
        minimum_frame=placement_frame,
        dt=dt,
    )
    if camera is not None and viewport is not None:
        captures.append(
            _capture_phase(
                app=app,
                viewport=viewport,
                camera=camera,
                world=world,
                destination=screenshot_root / "hold_end_raw.png",
                phase="hold_end",
                frame=frame_state["frame"],
                stage_path=stage_path,
                stage_hash=stage_hash,
                camera_pose=camera_pose,
                articulation=articulation,
                bottle=bottle,
                contact_state={
                    "bottle_kinematic": False,
                    "left": left_all["physical_surface_contact"],
                    "right": right_all["physical_surface_contact"],
                    "drop_m": drop_m,
                    "drop_gate_m": float(frozen["drop_gate_m"]),
                },
                contact_annotations_world={
                    "left": left_all["last_contact"],
                    "right": right_all["last_contact"],
                },
            )
        )

    post_release_event_frames = {
        int(event["frame"])
        for event in events
        if int(event["frame"]) >= release_frame
        and event.get("contacts")
        and BOTTLE_PATH
        in f"{event.get('collider0', '')}\n{event.get('collider1', '')}"
    }
    last_quarter_start = release_frame + int(frozen["hold_steps"]) * 3 // 4
    bilateral_pre_release = bool(
        left_fixed["physical_surface_contact"]
        and right_fixed["physical_surface_contact"]
    )
    contact_lost = bool(
        bilateral_pre_release
        and not any(frame >= last_quarter_start for frame in post_release_event_frames)
    )
    vertical_speeds = [
        abs(float(item["bottle"]["vertical_velocity_m_s"]))
        for item in release_telemetry
    ]
    api_vertical_speeds = [
        float(item["bottle"]["vertical_velocity_m_s"])
        for item in release_telemetry
    ]
    pose_vertical_speeds = [
        float(value)
        for value in hold_kinematics[
            "pose_derived_vertical_velocity_m_s"
        ]
    ]
    velocity_readback_disagreement = [
        api_value - pose_value
        for api_value, pose_value in zip(
            api_vertical_speeds,
            pose_vertical_speeds,
            strict=True,
        )
    ]
    angular_speeds = [
        float(item["bottle"]["angular_speed_rad_s"])
        for item in release_telemetry
    ]
    continuous_slip = bool(
        bilateral_pre_release
        and not contact_lost
        and drop_m > float(frozen["drop_gate_m"])
    )
    rotation_escape = bool(
        drop_m > float(frozen["drop_gate_m"])
        and max(angular_speeds, default=0.0) > 1.0
    )
    max_penetration = float(
        all_summary["maximum_penetration_depth_m"]
    )
    numerical_ejection = bool(
        all_summary["persistent_penetration"]
        or (
            max(vertical_speeds, default=0.0) > 2.0
            and drop_m < -float(frozen["drop_gate_m"])
        )
    )
    metrics = {
        "solve_articulation_contact_last_ok": bool(
            physics_context.get_solve_articulation_contact_last()
        ),
        "left_finger_contact": bool(
            left_fixed["physical_surface_contact"]
        ),
        "right_finger_contact": bool(
            right_fixed["physical_surface_contact"]
        ),
        "bilateral_contact_before_release": bilateral_pre_release,
        "impulses_finite": bool(all_summary["impulses_finite"]),
        "persistent_penetration": bool(
            all_summary["persistent_penetration"]
        ),
        "unexpected_gripper_collision": bool(
            all_summary["unexpected_gripper_collision"]
        ),
        "released_without_constraint": not constraint_found,
        "gravity_enabled_after_release": True,
        "held_for_required_time": len(release_telemetry)
        == int(frozen["hold_steps"]),
        "drop_within_gate": bool(
            math.isfinite(drop_m)
            and drop_m <= float(frozen["drop_gate_m"])
        ),
        "finite_state": bool(
            math.isfinite(drop_m)
            and all(
                math.isfinite(float(item["bottle"]["z_m"]))
                and math.isfinite(
                    float(item["bottle"]["vertical_velocity_m_s"])
                )
                and math.isfinite(
                    float(item["bottle"]["angular_speed_rad_s"])
                )
                for item in release_telemetry
            )
        ),
        "contact_lost_after_release": contact_lost,
        "continuous_slip_with_bilateral_contact": continuous_slip,
        "rotation_induced_escape": rotation_escape,
        "normal_force_decay": False,
        "numerical_penetration_or_ejection": numerical_ejection,
    }
    evaluation = evaluate_bottle_trial(metrics)
    failure_mode = classify_hold_failure_mode(metrics)
    trial = {
        "schema_version": 1,
        "status": evaluation["status"],
        "trial_index": trial_index,
        "robot": "follower_left",
        "profile": "supplier_cad_v2_convex_hull_arm_max_force",
        "fresh_world_reset": True,
        "metrics": metrics,
        "failed_checks": evaluation["failed_checks"],
        "failure_mode": failure_mode,
        "states": {
            "dof_order": order,
            "open": open_state,
            "closed_before_release": closed_state,
            "hold_end": final_finger_state,
            "closed_targets_m": list(motion["closed_targets_m"]),
            "self_collision_readback": _self_collision_readback(stage),
        },
        "aperture": {
            "open": open_aperture,
        },
        "contacts": {
            "left_fixed": left_fixed,
            "right_fixed": right_fixed,
            "left_all": left_all,
            "right_all": right_all,
            "normal_quality": _normal_quality(
                left_fixed,
                right_fixed,
                open_aperture["closing_axis_world"],
            ),
            "fixed_summary": fixed_summary,
            "all_summary": all_summary,
            "raw_event_count": len(events),
            "maximum_penetration_depth_m": max_penetration,
            "raw_events": events,
        },
        "released_hold": {
            "release_frame": release_frame,
            "required_steps": int(frozen["hold_steps"]),
            "required_time_s": float(frozen["hold_interval_s"]),
            "drop_gate_m": float(frozen["drop_gate_m"]),
            "release_state": release_state,
            "final_state": final_state,
            "drop_m": drop_m,
            "drop_semantics": (
                "MAXIMUM_RELEASE_Z_MINUS_MINIMUM_HOLD_Z_OVER_120_FRAMES"
            ),
            "final_drop_m": float(
                hold_kinematics["final_drop_m"]
            ),
            "maximum_rise_m": float(
                hold_kinematics["maximum_rise_m"]
            ),
            "pose_derived_vertical_velocity": {
                "maximum_abs_m_s": float(
                    hold_kinematics[
                        "maximum_abs_pose_derived_vertical_velocity_m_s"
                    ]
                ),
                "final_m_s": float(
                    hold_kinematics[
                        "final_pose_derived_vertical_velocity_m_s"
                    ]
                ),
                "samples_m_s": pose_vertical_speeds,
            },
            "api_velocity_vs_pose_difference": {
                "maximum_abs_m_s": max(
                    map(abs, velocity_readback_disagreement),
                    default=0.0,
                ),
                "final_m_s": (
                    velocity_readback_disagreement[-1]
                    if velocity_readback_disagreement
                    else 0.0
                ),
                "status": (
                    "RUNTIME_READBACK_DISAGREEMENT_RECORDED_"
                    "NOT_USED_TO_OVERRIDE_POSITION_DROP_GATE"
                ),
            },
            "maximum_abs_vertical_velocity_m_s": max(
                vertical_speeds, default=0.0
            ),
            "maximum_angular_speed_rad_s": max(
                angular_speeds, default=0.0
            ),
            "constraint_found": constraint_found,
            "constraint_paths": constraint_paths,
            "support_surface": "NONE_STATIC_SUSPENDED_HOLD",
            "lift_trajectory": (
                "NOT_RUN_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY"
            ),
        },
        "session": session,
        "hidden_visuals_session_only": hidden_visuals,
        "telemetry": telemetry,
        "screenshots": captures,
        "runtime_s": time.perf_counter() - started,
        "contact_subscription_active": subscription is not None,
    }
    trial["deterministic_signature"] = canonical_signature(
        json.loads(
            json.dumps(
                _signature_payload(trial),
                default=_json_default,
                allow_nan=False,
            )
        )
    )
    if camera is not None:
        camera.destroy()
    del subscription
    return trial


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--report", type=Path, default=OUTPUT)
    parser.add_argument("--trials", type=Path, default=TRIAL_OUTPUT)
    parser.add_argument("--screenshot-root", type=Path, default=SCREENSHOT_ROOT)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve(strict=True)
    profile = _resolve_profile(config_path)
    config = profile["document"]
    required = int(
        config["experiment"][
            "smoke_repeats" if args.smoke else "acceptance_repeats"
        ]
    )
    repeats = args.repeats if args.repeats is not None else required
    if not args.smoke and repeats < required:
        raise ValueError(
            f"acceptance requires at least {required} repeats"
        )
    screenshot_root = args.screenshot_root.resolve()
    if screenshot_root.exists():
        raise FileExistsError(
            f"screenshot root already exists: {screenshot_root}"
        )
    screenshot_root.mkdir(parents=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 900})
    exit_code = 1
    try:
        trials = []
        for trial_index in range(repeats):
            print(
                json.dumps(
                    {
                        "task5_bottle_event": "trial_start",
                        "trial_index": trial_index,
                        "repeats": repeats,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            trial = _run_trial(
                app=app,
                profile=profile,
                trial_index=trial_index,
                screenshot_root=(
                    screenshot_root if trial_index == 0 else None
                ),
            )
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "task5_bottle_event": "trial_complete",
                        "trial_index": trial_index,
                        "status": trial["status"],
                        "failure_mode": trial["failure_mode"],
                        "drop_m": trial["released_hold"]["drop_m"],
                        "signature": trial["deterministic_signature"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        summary = summarize_bottle_trials(
            trials, required_repeats=repeats
        )
        captures = trials[0]["screenshots"] if trials else []
        screenshot_manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={
                "supplier_cad_task5_bottle": list(PHASES)
            },
            artifact_root=screenshot_root,
        )
        hashes_after = {
            "diagnostic_stage": _sha256(profile["stage"]),
            "parent_stage": _sha256(profile["parent"]),
            "approved_source_stage": _sha256(profile["source"]),
        }
        protected_immutable = all(
            hashes_after[name] == profile["hashes"][name]
            for name in hashes_after
        )
        report = {
            "schema_version": 1,
            "status": (
                (
                    "PARTIAL"
                    if all(trial["status"] == "PASS" for trial in trials)
                    else "FAIL"
                )
                if args.smoke
                else summary["status"]
            ),
            "scope": "supplier-CAD follower-left 20 g bottle Task 5",
            "run_mode": "NON_ACCEPTANCE_SMOKE" if args.smoke else "ACCEPTANCE",
            "config": {
                "absolute_path": str(config_path),
                "sha256": profile["hashes"]["config"],
            },
            "stages": {
                "approved_source": {
                    "absolute_path": str(profile["source"]),
                    "sha256": profile["hashes"][
                        "approved_source_stage"
                    ],
                },
                "parent_diagnostic": {
                    "absolute_path": str(profile["parent"]),
                    "sha256": profile["hashes"]["parent_stage"],
                },
                "bottle_diagnostic": {
                    "absolute_path": str(profile["stage"]),
                    "sha256": profile["hashes"]["diagnostic_stage"],
                },
            },
            "frozen": config["frozen"],
            "summary": summary,
            "first_trial": trials[0] if trials else None,
            "trial_file": str(args.trials.resolve()),
            "screenshots": screenshot_manifest,
            "visual_model_review": "PENDING_VISUAL_MODEL_REVIEW",
            "baseline_protection": {
                "hashes_before": profile["hashes"],
                "hashes_after": hashes_after,
                "protected_assets_immutable": protected_immutable,
                "source_stage_modified": False,
                "default_configuration_modified": False,
                "final_collider_modified": False,
            },
            "boundaries": {
                "fixed_bottle_not_counted_as_hold_pass": True,
                "surface_gripper_used": False,
                "fixed_joint_used": False,
                "parent_attachment_used": False,
                "follower_right": (
                    "NOT_RUN_APPROVED_STAGE_CONTAINS_FOLLOWER_LEFT_ONLY"
                ),
                "lift_trajectory": (
                    "HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_"
                    "LIFT_TRAJECTORY"
                ),
                "task7": "NOT_RUN",
                "task8": "NOT_RUN",
            },
        }
        _write_jsonl(args.trials.resolve(), trials)
        _write_json(args.report.resolve(), report)
        print(f"status={report['status']}")
        print(f"report={args.report.resolve()}")
        print(f"trials={args.trials.resolve()}")
        print(f"raw_screenshots={screenshot_root}")
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

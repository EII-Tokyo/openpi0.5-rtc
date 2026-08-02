"""Run the frozen three-trial follower-left/table collision gate in Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from isaacsim import SimulationApp


TRIAL_COUNT = 3
STRESS_DT = 1.0 / 60.0
SHOULDER_START_DEG = -55.00394821166992
SHOULDER_END_DEG = 20.0
SHOULDER_STEP_DEG = 0.5
HOLD_STEPS = 180

PROJECT_ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
STAGE_PATH = PROJECT_ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)
EXPECTED_STAGE_SHA256 = (
    "165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2"
)
ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
TABLE_PATH = "/World/environment/worldBody/user_confirmed_table"
LEFT_BODY_NAMES = (
    "follower_left_base_link",
    "follower_left_shoulder_link",
    "follower_left_upper_arm_link",
    "follower_left_upper_forearm_link",
    "follower_left_lower_forearm_link",
    "follower_left_wrist_link",
    "follower_left_gripper_link",
    "follower_left_ee_arm_link",
    "follower_left_gripper_prop_link",
    "follower_left_gripper_bar_link",
    "follower_left_fingers_link",
    "follower_left_ee_gripper_link",
    "follower_left_left_finger_link",
    "follower_left_right_finger_link",
)
LEFT_BODY_PATHS = tuple(
    f"/World/follower_left/vx300s_left/{name}" for name in LEFT_BODY_NAMES
)
FINGER_MESH_PAIRS = (
    (
        "/World/follower_left/vx300s_left/follower_left_left_finger_link/"
        "visuals/diagnostic_supplier_cad_left_finger/mesh",
        "/World/follower_left/vx300s_left/follower_left_left_finger_link/"
        "collisions/diagnostic_supplier_cad_left_finger/mesh",
    ),
    (
        "/World/follower_left/vx300s_left/follower_left_right_finger_link/"
        "visuals/diagnostic_supplier_cad_right_finger/mesh",
        "/World/follower_left/vx300s_left/follower_left_right_finger_link/"
        "collisions/diagnostic_supplier_cad_right_finger/mesh",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_matches(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _target_pair(pair: tuple[str, str], allowed_roots: tuple[str, ...]) -> bool:
    first, second = pair
    return (
        _path_matches(first, TABLE_PATH)
        and any(_path_matches(second, root) for root in allowed_roots)
    ) or (
        _path_matches(second, TABLE_PATH)
        and any(_path_matches(first, root) for root in allowed_roots)
    )


def _disallowed_environment_pairs(
    pairs: list[tuple[str, str]], allowed_roots: tuple[str, ...]
) -> list[tuple[str, str]]:
    result = []
    for first, second in pairs:
        first_tip = any(_path_matches(first, root) for root in allowed_roots)
        second_tip = any(_path_matches(second, root) for root in allowed_roots)
        other = second if first_tip else first if second_tip else ""
        if (
            other
            and _path_matches(other, "/World/environment")
            and not _path_matches(other, TABLE_PATH)
        ):
            result.append((first, second))
    return sorted(set(result))


def _open_stage_and_wait(app: Any, stage_utils: Any, omni_usd: Any) -> Any:
    stage_utils.open_stage(str(STAGE_PATH))
    deadline = time.monotonic() + 30.0
    context = omni_usd.get_context()
    while time.monotonic() < deadline:
        app.update()
        stage = context.get_stage()
        if stage is not None and context.get_stage_loading_status()[2] == 0:
            return stage
    raise TimeoutError(f"Stage loading did not finish: {context.get_stage_loading_status()}")


def _preflight(stage: Any) -> dict[str, Any]:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    if _sha256(STAGE_PATH) != EXPECTED_STAGE_SHA256:
        raise RuntimeError("frozen Stage hash changed")
    if str(stage.GetDefaultPrim().GetPath()) != "/World":
        raise RuntimeError("unexpected default prim")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Stage is not Z-up")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise RuntimeError("Stage is not meter-scaled")
    if not stage.GetPrimAtPath(ARTICULATION_PATH).HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        raise RuntimeError("left articulation root missing")

    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    if not scene_prim.IsA(UsdPhysics.Scene):
        raise RuntimeError("PhysicsScene missing")
    scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
    scene = {
        "path": "/World/PhysicsScene",
        "time_steps_per_second": scene_api.GetTimeStepsPerSecondAttr().Get(),
        "enable_ccd": scene_api.GetEnableCCDAttr().Get(),
        "enable_gpu_dynamics": scene_api.GetEnableGPUDynamicsAttr().Get(),
        "broadphase_type": str(scene_api.GetBroadphaseTypeAttr().Get()),
        "min_position_iteration_count": scene_api.GetMinPositionIterationCountAttr().Get(),
    }
    if scene != {
        "path": "/World/PhysicsScene",
        "time_steps_per_second": 240,
        "enable_ccd": True,
        "enable_gpu_dynamics": False,
        "broadphase_type": "SAP",
        "min_position_iteration_count": 64,
    }:
        raise RuntimeError(f"unexpected PhysicsScene: {scene}")

    ccd_paths = []
    for path in LEFT_BODY_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"left rigid body missing: {path}")
        if PhysxSchema.PhysxRigidBodyAPI(prim).GetEnableCCDAttr().Get() is not True:
            raise RuntimeError(f"CCD not enabled: {path}")
        ccd_paths.append(path)

    table = stage.GetPrimAtPath(TABLE_PATH)
    if not table.IsValid() or not table.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("confirmed table collider missing")
    collision_enabled = UsdPhysics.CollisionAPI(table).GetCollisionEnabledAttr().Get()
    if collision_enabled is False:
        raise RuntimeError("confirmed table collider disabled")
    return {
        "stage_sha256": EXPECTED_STAGE_SHA256,
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "scene": scene,
        "left_rigid_body_count": len(LEFT_BODY_PATHS),
        "left_ccd_body_count": len(ccd_paths),
        "table_path": TABLE_PATH,
        "table_collision_enabled": collision_enabled is not False,
    }


def _begin_contact_reporting(stage: Any) -> dict[str, Any]:
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysxSchema, Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous("left_table_contact_report")
    stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
    old_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(layer))
    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    quasistatic_api = PhysxSchema.PhysxSceneQuasistaticAPI.Apply(scene_prim)
    quasistatic_api.CreateEnableQuasistaticAttr().Set(True)
    applied = []
    for path in LEFT_BODY_PATHS:
        prim = stage.GetPrimAtPath(path)
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(0)
        applied.append(path)
    return {
        "layer": layer,
        "old_target": old_target,
        "paths": applied,
        "quasistatic_enabled": True,
        "interface": get_physx_simulation_interface(),
    }


def _finish_contact_reporting(stage: Any, state: dict[str, Any]) -> None:
    stage.SetEditTarget(state["old_target"])
    identifier = state["layer"].identifier
    if identifier in stage.GetSessionLayer().subLayerPaths:
        stage.GetSessionLayer().subLayerPaths.remove(identifier)


def _contact_report(state: dict[str, Any]) -> list[dict[str, Any]]:
    from pxr import PhysicsSchemaTools

    headers, data = state["interface"].get_contact_report()
    result: list[dict[str, Any]] = []
    for header in headers:
        first = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
        second = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        separations = [float(datum.separation) for datum in list(data)[start:stop]]
        result.append(
            {
                "pair": (first, second),
                "minimum_separation_m": min(separations, default=math.inf),
                "contact_data_count": len(separations),
            }
        )
    return result


def _world_points(cache: Any, prim: Any) -> list[tuple[float, float, float]]:
    from pxr import UsdGeom

    points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
    if not points:
        raise RuntimeError(f"mesh has no points: {prim.GetPath()}")
    matrix = cache.GetLocalToWorldTransform(prim)
    return [tuple(float(value) for value in matrix.Transform(point)) for point in points]


def _live_finger_geometry(stage: Any) -> dict[str, Any]:
    from pxr import Gf, UsdGeom, UsdPhysics

    from tools.isaac_sim.left_table_geometry import (
        maximum_point_error,
        minimum_local_z,
        points_in_table_footprint,
    )

    cache = UsdGeom.XformCache()
    table = stage.GetPrimAtPath(TABLE_PATH)
    if not table.IsA(UsdGeom.Cube):
        raise RuntimeError("confirmed table is not a Cube")
    table_size = float(UsdGeom.Cube(table).GetSizeAttr().Get())
    half_size = table_size / 2.0
    world_from_table = cache.GetLocalToWorldTransform(table)
    table_from_world = world_from_table.GetInverse()
    table_from_world_rows = [
        [float(table_from_world[column][row]) for column in range(4)]
        for row in range(4)
    ]
    table_scale_z_m = float(
        world_from_table.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetLength()
    )
    if not math.isfinite(table_scale_z_m) or table_scale_z_m <= 0:
        raise RuntimeError(f"invalid table Z scale: {table_scale_z_m}")

    pair_rows = []
    minimum_table_local_finger_z_m = math.inf
    maximum_visual_collision_error_m = 0.0
    for visual_path, collision_path in FINGER_MESH_PAIRS:
        visual = stage.GetPrimAtPath(visual_path)
        collision = stage.GetPrimAtPath(collision_path)
        if not visual.IsA(UsdGeom.Mesh) or not collision.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"supplier-CAD finger mesh missing: {visual_path}, {collision_path}"
            )
        if not collision.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"finger collider missing CollisionAPI: {collision_path}")
        visual_world = _world_points(cache, visual)
        collision_world = _world_points(cache, collision)
        correspondence_error = maximum_point_error(visual_world, collision_world)
        local_collision = points_in_table_footprint(
            collision_world,
            table_from_world_rows,
            (half_size, half_size),
        )
        local_visual = points_in_table_footprint(
            visual_world,
            table_from_world_rows,
            (half_size, half_size),
        )
        local_points = local_collision + local_visual
        minimum_relative_z_m = (
            minimum_local_z(local_points) - half_size
        ) * table_scale_z_m
        minimum_table_local_finger_z_m = min(
            minimum_table_local_finger_z_m, minimum_relative_z_m
        )
        maximum_visual_collision_error_m = max(
            maximum_visual_collision_error_m, correspondence_error
        )
        pair_rows.append(
            {
                "visual_path": visual_path,
                "collision_path": collision_path,
                "point_count": len(visual_world),
                "inside_footprint_point_count": len(local_points),
                "minimum_relative_z_m": minimum_relative_z_m,
                "maximum_visual_collision_error_m": correspondence_error,
            }
        )
    return {
        "table_from_world": table_from_world_rows,
        "table_half_size_local": half_size,
        "table_scale_z_m": table_scale_z_m,
        "minimum_table_local_finger_z_m": minimum_table_local_finger_z_m,
        "maximum_visual_collision_error_m": maximum_visual_collision_error_m,
        "finger_pairs": pair_rows,
    }


def _normalized_limits(articulation: Any) -> np.ndarray:
    limits = np.asarray(
        articulation._articulation_view.get_dof_limits(), dtype=np.float64
    )
    if limits.ndim == 3 and limits.shape[0] == 1:
        limits = limits[0]
    if limits.shape != (int(articulation.num_dof), 2):
        raise RuntimeError(f"unexpected limit shape: {limits.shape}")
    return limits


def _within_limits(qpos: np.ndarray, limits: np.ndarray) -> bool:
    finite = np.isfinite(limits)
    lower_ok = np.logical_or(~finite[:, 0], qpos >= limits[:, 0] - 1e-6)
    upper_ok = np.logical_or(~finite[:, 1], qpos <= limits[:, 1] + 1e-6)
    return bool(np.all(lower_ok & upper_ok))


def _set_full_target(articulation: Any, target: np.ndarray) -> None:
    indices = np.arange(int(articulation.num_dof), dtype=np.int64)
    articulation._articulation_view.set_joint_position_targets(
        np.asarray(target, dtype=np.float64), joint_indices=indices
    )


def _capture_verified_contact(app: Any, output_path: Path) -> None:
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Sdf

    from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz

    camera = Camera(
        prim_path="/World/CollisionGateSessionCamera",
        name="collision_gate_session_camera",
        resolution=(1280, 720),
        frequency=60,
    )
    camera.initialize()
    camera.set_clipping_range(0.01, 10.0)
    position = np.asarray((1.1, -1.1, 0.8), dtype=np.float64)
    target = np.asarray((0.0, 0.0, 0.1), dtype=np.float64)
    camera.set_world_pose(
        position=position,
        orientation=look_at_orientation_wxyz(position, target),
        camera_axes="usd",
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport unavailable for evidence capture")
    viewport.camera_path = Sdf.Path(camera.prim_path)
    for _ in range(30):
        app.update()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    helper = capture_viewport_to_file(viewport, file_path=str(output_path))
    previous_size = -1
    stable_updates = 0
    for _ in range(360):
        app.update()
        if not output_path.exists():
            continue
        size = output_path.stat().st_size
        stable_updates = stable_updates + 1 if size > 0 and size == previous_size else 0
        previous_size = size
        if stable_updates >= 3:
            break
    del helper
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("verified contact screenshot was not created")


def _run_trial(
    *,
    app: Any,
    trial_index: int,
    screenshot_path: Path,
    capture: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    import isaacsim.core.utils.stage as stage_utils
    import omni.usd

    from tools.isaac_sim.left_table_collision_gate import (
        ALLOWED_TIP_ROOTS,
        MAX_CONTACT_SEPARATION_M,
        MAX_TABLE_TOP_PENETRATION_M,
        TrialMetrics,
        evaluate_trial,
    )

    World.clear_instance()
    stage = _open_stage_and_wait(app, stage_utils, omni.usd)
    preflight = _preflight(stage)
    contact_state = _begin_contact_reporting(stage)
    telemetry = []
    all_pairs: set[tuple[str, str]] = set()
    disallowed: set[tuple[str, str]] = set()
    persistent_contact_steps = 0
    minimum_target_separation_m = math.inf
    minimum_table_local_finger_z_m = math.inf
    maximum_visual_collision_error_m = 0.0
    finite = True
    within_limits = True
    physx_errors: list[str] = []
    world = None
    try:
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=STRESS_DT,
            rendering_dt=STRESS_DT,
        )
        articulation = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name=f"follower_left_collision_gate_{trial_index}",
                reset_xform_properties=False,
            )
        )
        world.reset()
        names = list(articulation.dof_names)
        if "shoulder" not in names:
            raise RuntimeError(f"shoulder DOF missing: {names}")
        shoulder_index = names.index("shoulder")
        limits = _normalized_limits(articulation)
        initial = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        initial_deg = math.degrees(float(initial[shoulder_index]))
        if abs(initial_deg - SHOULDER_START_DEG) > 1e-4:
            raise RuntimeError(
                f"reset shoulder mismatch: {initial_deg} vs {SHOULDER_START_DEG}"
            )
        held_target = initial.copy()
        contact_found = False

        targets = np.arange(
            SHOULDER_START_DEG + SHOULDER_STEP_DEG,
            SHOULDER_END_DEG + SHOULDER_STEP_DEG / 2.0,
            SHOULDER_STEP_DEG,
        )
        phases = [("sweep", float(value)) for value in targets]
        for phase, target_deg in phases:
            held_target[shoulder_index] = math.radians(target_deg)
            _set_full_target(articulation, held_target)
            world.step(render=False)

            qpos = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
            qvel = np.asarray(articulation.get_joint_velocities(), dtype=np.float64)
            contacts = _contact_report(contact_state)
            pairs = [row["pair"] for row in contacts]
            all_pairs.update(pairs)
            disallowed.update(_disallowed_environment_pairs(pairs, ALLOWED_TIP_ROOTS))
            target_separations = [
                row["minimum_separation_m"]
                for row in contacts
                if _target_pair(row["pair"], ALLOWED_TIP_ROOTS)
            ]
            step_separation = min(target_separations, default=math.inf)
            minimum_target_separation_m = min(
                minimum_target_separation_m, step_separation
            )
            step_contact = (
                math.isfinite(step_separation)
                and step_separation <= MAX_CONTACT_SEPARATION_M
            )
            geometry = _live_finger_geometry(stage)
            step_local_z = geometry["minimum_table_local_finger_z_m"]
            step_correspondence = geometry["maximum_visual_collision_error_m"]
            minimum_table_local_finger_z_m = min(
                minimum_table_local_finger_z_m, step_local_z
            )
            maximum_visual_collision_error_m = max(
                maximum_visual_collision_error_m, step_correspondence
            )
            step_finite = bool(
                np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
                and math.isfinite(step_local_z)
                and math.isfinite(step_correspondence)
            )
            step_limits = _within_limits(qpos, limits)
            finite = finite and step_finite
            within_limits = within_limits and step_limits
            telemetry.append(
                {
                    "phase": phase,
                    "target_deg": target_deg,
                    "shoulder_deg": math.degrees(float(qpos[shoulder_index])),
                    "shoulder_velocity_rad_s": float(qvel[shoulder_index]),
                    "minimum_target_separation_m": step_separation,
                    "minimum_table_local_finger_z_m": step_local_z,
                    "maximum_visual_collision_error_m": step_correspondence,
                    "finger_geometry": geometry["finger_pairs"],
                    "target_contact": step_contact,
                    "contact_report": contacts,
                    "contact_pairs": pairs,
                    "finite": step_finite,
                    "within_joint_limits": step_limits,
                }
            )
            if not step_finite:
                physx_errors.append("non_finite_runtime_state")
                break
            if not step_limits:
                break
            if step_local_z < -MAX_TABLE_TOP_PENETRATION_M:
                break
            if disallowed:
                break
            if step_contact:
                contact_found = True
                break

        if contact_found and finite and within_limits and not disallowed:
            held_target[shoulder_index] = math.radians(SHOULDER_END_DEG)
            for hold_step in range(HOLD_STEPS):
                _set_full_target(articulation, held_target)
                world.step(render=capture and hold_step == HOLD_STEPS - 1)
                qpos = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
                qvel = np.asarray(articulation.get_joint_velocities(), dtype=np.float64)
                contacts = _contact_report(contact_state)
                pairs = [row["pair"] for row in contacts]
                all_pairs.update(pairs)
                disallowed.update(
                    _disallowed_environment_pairs(pairs, ALLOWED_TIP_ROOTS)
                )
                target_separations = [
                    row["minimum_separation_m"]
                    for row in contacts
                    if _target_pair(row["pair"], ALLOWED_TIP_ROOTS)
                ]
                step_separation = min(target_separations, default=math.inf)
                minimum_target_separation_m = min(
                    minimum_target_separation_m, step_separation
                )
                step_contact = (
                    math.isfinite(step_separation)
                    and step_separation <= MAX_CONTACT_SEPARATION_M
                )
                persistent_contact_steps += int(step_contact)
                geometry = _live_finger_geometry(stage)
                step_local_z = geometry["minimum_table_local_finger_z_m"]
                step_correspondence = geometry["maximum_visual_collision_error_m"]
                minimum_table_local_finger_z_m = min(
                    minimum_table_local_finger_z_m, step_local_z
                )
                maximum_visual_collision_error_m = max(
                    maximum_visual_collision_error_m, step_correspondence
                )
                step_finite = bool(
                    np.all(np.isfinite(qpos))
                    and np.all(np.isfinite(qvel))
                    and math.isfinite(step_local_z)
                    and math.isfinite(step_correspondence)
                )
                step_limits = _within_limits(qpos, limits)
                finite = finite and step_finite
                within_limits = within_limits and step_limits
                telemetry.append(
                    {
                        "phase": "hold",
                        "hold_step": hold_step + 1,
                        "target_deg": SHOULDER_END_DEG,
                        "shoulder_deg": math.degrees(float(qpos[shoulder_index])),
                        "shoulder_velocity_rad_s": float(qvel[shoulder_index]),
                        "minimum_target_separation_m": step_separation,
                        "minimum_table_local_finger_z_m": step_local_z,
                        "maximum_visual_collision_error_m": step_correspondence,
                        "finger_geometry": geometry["finger_pairs"],
                        "target_contact": step_contact,
                        "contact_report": contacts,
                        "contact_pairs": pairs,
                        "finite": step_finite,
                        "within_joint_limits": step_limits,
                    }
                )
                if (
                    not step_finite
                    or not step_limits
                    or step_local_z < -MAX_TABLE_TOP_PENETRATION_M
                    or disallowed
                ):
                    break

        final_qpos = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        final_error = math.radians(SHOULDER_END_DEG) - float(
            final_qpos[shoulder_index]
        )
        metrics = TrialMetrics(
            contact_pairs=sorted(all_pairs),
            minimum_target_separation_m=minimum_target_separation_m,
            minimum_table_local_finger_z_m=minimum_table_local_finger_z_m,
            maximum_visual_collision_error_m=maximum_visual_collision_error_m,
            final_target_error_rad=final_error,
            persistent_contact_steps=persistent_contact_steps,
            finite=finite,
            within_joint_limits=within_limits,
            ccd_effective=(
                preflight["scene"]["enable_ccd"]
                and not preflight["scene"]["enable_gpu_dynamics"]
                and preflight["left_ccd_body_count"] == len(LEFT_BODY_PATHS)
            ),
            disallowed_tip_contacts=sorted(disallowed),
            physx_errors=physx_errors,
        )
        decision = evaluate_trial(metrics)
        if capture and decision["status"] == "PASS":
            _capture_verified_contact(app, screenshot_path)
        record = {
            "trial_index": trial_index,
            **decision,
            "initial_shoulder_deg": initial_deg,
            "commanded_shoulder_end_deg": SHOULDER_END_DEG,
            "final_shoulder_deg": math.degrees(float(final_qpos[shoulder_index])),
            "telemetry": telemetry,
            "preflight": preflight,
        }
        return record, preflight
    finally:
        if world is not None:
            world.stop()
        _finish_contact_reporting(stage, contact_state)
        World.clear_instance()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "collision_gate_report.json"
    screenshot_path = args.output_dir / "verified_contact.png"
    if report_path.exists() or screenshot_path.exists():
        raise FileExistsError(
            "collision-gate outputs already exist; select a non-overwriting directory"
        )
    app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    report: dict[str, Any] = {
        "status": "FAIL",
        "stage_path": str(STAGE_PATH),
        "stage_sha256": EXPECTED_STAGE_SHA256,
        "real_robot_touched": False,
        "stage_saved": False,
        "configuration": {
            "trial_count": TRIAL_COUNT,
            "stress_dt_s": STRESS_DT,
            "shoulder_start_deg": SHOULDER_START_DEG,
            "shoulder_end_deg": SHOULDER_END_DEG,
            "shoulder_step_deg": SHOULDER_STEP_DEG,
            "hold_steps": HOLD_STEPS,
            "quasistatic_enabled": True,
        },
        "trials": [],
        "failure_reasons": [],
    }
    exit_code = 1
    try:
        from tools.isaac_sim.left_table_collision_gate import aggregate_trials

        preflight = None
        for trial_index in range(1, TRIAL_COUNT + 1):
            trial, preflight = _run_trial(
                app=app,
                trial_index=trial_index,
                screenshot_path=screenshot_path,
                capture=trial_index == 1,
            )
            report["trials"].append(trial)
            print(
                f"CODEX_COLLISION_TRIAL index={trial_index} status={trial['status']} "
                "minimum_table_local_finger_z_m="
                f"{trial['metrics']['minimum_table_local_finger_z_m']} "
                "minimum_target_separation_m="
                f"{trial['metrics']['minimum_target_separation_m']} "
                f"persistent={trial['metrics']['persistent_contact_steps']}",
                flush=True,
            )
            if trial["status"] != "PASS":
                break
        aggregate = aggregate_trials(report["trials"])
        report.update(
            {
                "status": aggregate["status"],
                "failure_reasons": aggregate["failure_reasons"],
                "preflight": preflight,
                "screenshot": str(screenshot_path),
                "screenshot_nonempty": screenshot_path.exists()
                and screenshot_path.stat().st_size > 0,
            }
        )
        if report["status"] == "PASS" and not report["screenshot_nonempty"]:
            report["status"] = "FAIL"
            report["failure_reasons"].append("missing_verified_contact_screenshot")
        exit_code = 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        report["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        report["stage_sha256_after"] = _sha256(STAGE_PATH)
        if report["stage_sha256_after"] != EXPECTED_STAGE_SHA256:
            report["status"] = "FAIL"
            report["failure_reasons"].append("stage_hash_changed")
            exit_code = 1
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"CODEX_COLLISION_GATE_{report['status']} report={report_path}", flush=True)
        app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

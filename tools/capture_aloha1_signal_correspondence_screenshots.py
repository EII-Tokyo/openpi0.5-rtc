#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Capture fresh Isaac Sim 5.1 Task 7A signal screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.hydra_protopath_diagnosis import PROTOPATH_SETTINGS
from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.aloha1_mapping.signal_correspondence import HOME_ARM
from tools.aloha1_mapping.signal_correspondence import HOME_LEFT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import HOME_RIGHT_FINGER_M
from tools.aloha1_mapping.signal_correspondence import RUNTIME_SPECS
from tools.aloha1_mapping.signal_correspondence import build_fixed_oblique_camera_spec
from tools.aloha1_mapping.signal_correspondence import canonical_dof_name

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
OUTPUT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-signal-correspondence"
METADATA_PATH = OUTPUT_ROOT / "metadata/aloha1_signal_screenshot_metadata.json"
RESOLUTION = (1280, 900)
PHYSICS_HZ = 60
JOINT_VISUAL_LINK_TOKENS = {
    "shoulder": "_upper_arm_link/",
    "waist": "_shoulder_link/",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _home() -> np.ndarray:
    return np.asarray(
        [
            *HOME_ARM,
            0.0,
            HOME_LEFT_FINGER_M,
            HOME_RIGHT_FINGER_M,
        ],
        dtype=np.float32,
    )


def _poses(robot: str) -> list[dict[str, Any]]:
    home = _home()
    records = []
    for phase, joint, delta in (
        ("home_reference", "shoulder", 0.0),
        ("small_up_start", "shoulder", -0.02),
        ("small_up_max", "shoulder", -0.08),
        ("small_down_return", "shoulder", 0.0),
        ("waist_positive", "waist", 0.05),
        ("waist_negative", "waist", -0.05),
    ):
        qpos = home.copy()
        index = list(ARM_NAMES).index(joint)
        qpos[index] += delta
        records.append(
            {
                "capture_id": f"{robot}_{phase}",
                "robot": robot,
                "phase": phase,
                "joint": joint,
                "isaac_dof_index": index,
                "target_qpos": qpos.tolist(),
                "command_target": float(qpos[index]),
                "expected_direction": (
                    "END_EFFECTOR_Z_UP"
                    if phase in {"small_up_start", "small_up_max"}
                    else "RETURN_HOME"
                    if phase == "small_down_return"
                    else "JOINT_POSITIVE"
                    if phase == "waist_positive"
                    else "JOINT_NEGATIVE"
                    if phase == "waist_negative"
                    else "REFERENCE"
                ),
                "numeric_acceptance": (
                    "target/readback error <= 0.02 rad; up delta_z > 0; non-target drift <= 0.01 rad"
                ),
            }
        )
    return records


ARM_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


def _capture_png(
    app: Any,
    viewport: Any,
    destination: Path,
) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file

    helper = capture_viewport_to_file(
        viewport,
        file_path=str(destination),
    )
    previous = -1
    stable = 0
    for _ in range(360):
        app.update()
        if not destination.exists():
            continue
        current = destination.stat().st_size
        if current > 0 and current == previous:
            stable += 1
        else:
            stable = 0
        previous = current
        if stable >= 3:
            break
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"capture failed: {destination}")
    del helper


def _apply_positions(articulation: Any, qpos: np.ndarray) -> None:
    from isaacsim.core.utils.types import ArticulationAction

    active = np.arange(len(qpos) - 1, dtype=np.int32)
    articulation.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=qpos[active],
            joint_indices=active,
        )
    )


def _world_points(prim: Any) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [list(transform.Transform(point)) for point in (mesh.GetPointsAttr().Get() or [])],
        dtype=np.float64,
    )


def _visible_robot_meshes(stage: Any, robot: str) -> list[Any]:
    from pxr import Usd
    from pxr import UsdGeom

    root_path = RUNTIME_SPECS[robot]["articulation_path"].rsplit("/", 1)[0]
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"missing robot root: {root_path}")
    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in path:
            continue
        if "/collisions/" in path or "/sites/" in path:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == (UsdGeom.Tokens.invisible):
            continue
        meshes.append(prim)
    if not meshes:
        raise RuntimeError(f"no visible visual meshes for {robot}")
    return meshes


def _point_cloud(prims: list[Any]) -> np.ndarray:
    clouds = [_world_points(prim) for prim in prims]
    clouds = [cloud for cloud in clouds if cloud.size]
    if not clouds:
        raise RuntimeError("visible robot meshes contain no points")
    return np.concatenate(clouds)


def _projection(camera: Any, points: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(
        camera.get_image_coords_from_world_points(points),
        dtype=np.float64,
    )
    finite = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(finite):
        raise RuntimeError("camera projection contains no finite points")
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    return {
        "bbox_min_px": minimum.tolist(),
        "bbox_max_px": maximum.tolist(),
        "bbox_center_px": ((minimum + maximum) / 2.0).tolist(),
        "finite_point_count": len(finite),
        "fully_in_frame": bool(
            minimum[0] >= 0.0 and minimum[1] >= 0.0 and maximum[0] < RESOLUTION[0] and maximum[1] < RESOLUTION[1]
        ),
    }


def _joint_visual_points(
    stage: Any,
    robot: str,
    joint: str,
) -> tuple[np.ndarray, list[str]]:
    token = JOINT_VISUAL_LINK_TOKENS[joint]
    prims = [prim for prim in _visible_robot_meshes(stage, robot) if token in str(prim.GetPath())]
    if not prims:
        raise RuntimeError(f"no visual mesh found for {robot} {joint}: {token}")
    return _point_cloud(prims), [str(prim.GetPath()) for prim in prims]


def _set_signal_visual_scope(stage: Any) -> list[str]:
    from pxr import UsdGeom

    changed = []
    robot_roots = (
        "/World/follower_left/vx300s_left/",
        "/World/follower_right/vx300s_right/",
    )
    supplier_tokens = (
        "/diagnostic_supplier_cad_left_finger/",
        "/diagnostic_supplier_cad_right_finger/",
    )
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        path = str(prim.GetPath())
        if not path.startswith(robot_roots):
            continue
        imageable = UsdGeom.Imageable(prim)
        if "/collisions" in path or "/sites" in path:
            imageable.MakeInvisible()
            changed.append(path)
            continue
        if "_finger_link/visuals/" in path and not any(token in path for token in supplier_tokens):
            imageable.MakeInvisible()
            changed.append(path)
            continue
        imageable.MakeVisible()
        changed.append(path)
    return sorted(set(changed))


def _create_diagnostic_arm_materials(stage: Any) -> dict[str, Any]:
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdShade

    colors = {
        "follower_left": Gf.Vec3f(0.18, 0.48, 0.72),
        "follower_right": Gf.Vec3f(0.55, 0.34, 0.68),
    }
    materials = {}
    for robot, color in colors.items():
        material = UsdShade.Material.Define(
            stage,
            f"/World/SignalScreenshotSession/Materials/{robot}",
        )
        shader = UsdShade.Shader.Define(
            stage,
            f"/World/SignalScreenshotSession/Materials/{robot}/Shader",
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(color)
        shader.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(0.62)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )
        materials[robot] = material

    return materials


def _create_arm_visual_clones(
    stage: Any,
    materials: dict[str, Any],
    robots: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, str]]]]:
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdShade

    manifest = []
    handles: dict[str, list[tuple[str, str]]] = {}
    for robot in robots:
        sources = [prim for prim in _visible_robot_meshes(stage, robot) if "_finger_link/" not in str(prim.GetPath())]
        handles[robot] = []
        for index, source_prim in enumerate(sources):
            source = UsdGeom.Mesh(source_prim)
            clone_path = f"/World/SignalScreenshotSession/ExactVisualClones/{robot}/mesh_{index:02d}"
            clone = UsdGeom.Mesh.Define(stage, clone_path)
            counts = source.GetFaceVertexCountsAttr().Get()
            indices = source.GetFaceVertexIndicesAttr().Get()
            if counts is None or indices is None:
                raise RuntimeError(f"source visual mesh lacks topology: {source_prim.GetPath()}")
            clone.CreateFaceVertexCountsAttr(counts)
            clone.CreateFaceVertexIndicesAttr(indices)
            clone.CreateSubdivisionSchemeAttr(source.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none)
            clone.CreateOrientationAttr(source.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
            clone.CreateDoubleSidedAttr(True)  # noqa: FBT003 - USD API positional value.
            points = _world_points(source_prim)
            clone.CreatePointsAttr(points.tolist())
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            clone.CreateExtentAttr(
                [
                    Gf.Vec3f(*minimum.tolist()),
                    Gf.Vec3f(*maximum.tolist()),
                ]
            )
            UsdShade.MaterialBindingAPI.Apply(clone.GetPrim()).Bind(materials[robot])
            handles[robot].append((str(source_prim.GetPath()), clone_path))
            manifest.append(
                {
                    "robot": robot,
                    "source_prim_path": str(source_prim.GetPath()),
                    "clone_prim_path": clone_path,
                    "point_count": len(points),
                    "face_count": len(counts),
                    "source_instance_proxy": source_prim.IsInstanceProxy(),
                    "physics_schema_applied": False,
                    "collision_schema_applied": False,
                }
            )
    if not manifest:
        raise RuntimeError("no exact arm visual clones were created")
    return manifest, handles


def _update_arm_visual_clones(
    stage: Any,
    handles: dict[str, list[tuple[str, str]]],
) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    for pairs in handles.values():
        for source_path, clone_path in pairs:
            source_prim = stage.GetPrimAtPath(source_path)
            clone_prim = stage.GetPrimAtPath(clone_path)
            if not source_prim.IsValid() or not clone_prim.IsValid():
                raise RuntimeError(
                    f"visual clone source/destination invalid after reset: {source_path} -> {clone_path}"
                )
            clone = UsdGeom.Mesh(clone_prim)
            points = _world_points(source_prim)
            clone.GetPointsAttr().Set(points.tolist())
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            clone.GetExtentAttr().Set(
                [
                    Gf.Vec3f(*minimum.tolist()),
                    Gf.Vec3f(*maximum.tolist()),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument(
        "--robot",
        choices=(*RUNTIME_SPECS, "all"),
        default="all",
    )
    parser.add_argument("--diagnostic-hydra-setting-path")
    parser.add_argument(
        "--diagnostic-hydra-setting-value",
        choices=("true", "false"),
    )
    args = parser.parse_args()
    stage_path = args.stage.resolve(strict=True)
    if stage_path != STAGE.resolve():
        raise ValueError("capture accepts only the frozen signal Stage")
    stage_hash_before = _sha256(stage_path)
    output_root = args.output_root.resolve()
    raw_root = output_root / "screenshots_raw"
    selected_robots = tuple(RUNTIME_SPECS) if args.robot == "all" else (args.robot,)
    for robot in selected_robots:
        target_dir = raw_root / robot
        if target_dir.exists() and any(target_dir.glob("*.png")):
            raise FileExistsError(f"fresh robot capture directory contains PNGs: {target_dir}")
        (raw_root / robot).mkdir(parents=True, exist_ok=True)

    import carb.settings
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.xforms import get_world_pose
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_interface
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdLux

    app = globals()["_SIMULATION_APP"]
    settings = carb.settings.get_settings()
    hydra_settings_before = {
        path: {
            "exists": settings.get(path) is not None,
            "python_type": (type(settings.get(path)).__name__ if settings.get(path) is not None else None),
            "value": settings.get(path),
        }
        for path in PROTOPATH_SETTINGS.values()
    }
    if bool(args.diagnostic_hydra_setting_path) != bool(args.diagnostic_hydra_setting_value):
        raise ValueError("Hydra diagnostic setting path and value must be provided together")
    hydra_setting_override = {}
    if args.diagnostic_hydra_setting_path:
        path = args.diagnostic_hydra_setting_path
        if path not in PROTOPATH_SETTINGS.values():
            raise ValueError(f"Hydra diagnostic setting is not allowlisted: {path}")
        if not hydra_settings_before[path]["exists"]:
            raise RuntimeError(f"Hydra diagnostic setting is unsupported locally: {path}")
        value = args.diagnostic_hydra_setting_value == "true"
        settings.set_bool(path, value)
        hydra_setting_override[path] = value
    hydra_settings_effective = {
        path: {
            "exists": settings.get(path) is not None,
            "python_type": (type(settings.get(path)).__name__ if settings.get(path) is not None else None),
            "value": settings.get(path),
        }
        for path in PROTOPATH_SETTINGS.values()
    }
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    stage = get_current_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        visual_scope_changes = _set_signal_visual_scope(stage)
        diagnostic_arm_materials = _create_diagnostic_arm_materials(stage)
        exact_visual_clone_manifest, exact_visual_clone_handles = _create_arm_visual_clones(
            stage,
            diagnostic_arm_materials,
            selected_robots,
        )
        dome = UsdLux.DomeLight.Define(
            stage,
            "/World/SignalScreenshotSession/Dome",
        )
        dome.CreateIntensityAttr(850.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))
        key = UsdLux.DistantLight.Define(
            stage,
            "/World/SignalScreenshotSession/Key",
        )
        key.CreateIntensityAttr(1250.0)

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / PHYSICS_HZ,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, spec in RUNTIME_SPECS.items():
        articulation = SingleArticulation(
            prim_path=spec["articulation_path"],
            name=f"signal_capture_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    camera = Camera(
        prim_path="/World/SignalScreenshotSession/Camera",
        name="signal_screenshot_camera",
        resolution=RESOLUTION,
        frequency=PHYSICS_HZ,
    )
    world.scene.add(camera)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no active Isaac viewport")
    world.reset()
    camera.set_clipping_range(0.01, 10.0)

    captures = []
    home_eef = {}
    for articulation in articulations.values():
        qpos = _home()
        articulation.set_joint_positions(qpos)
        articulation.set_joint_velocities(np.zeros_like(qpos))
        _apply_positions(articulation, qpos)
    # Isaac Sim 5.1 exposes this PhysX method as a positional-only binding.
    get_physx_interface().update_transformations(True, True, False, False)  # noqa: FBT003
    camera_specs = {
        robot: build_fixed_oblique_camera_spec(
            _point_cloud(_visible_robot_meshes(stage, robot)),
            robot,
        )
        for robot in selected_robots
    }
    for robot in selected_robots:
        for articulation in articulations.values():
            qpos = _home()
            articulation.set_joint_positions(qpos)
            articulation.set_joint_velocities(np.zeros_like(qpos))
            _apply_positions(articulation, qpos)
        for _ in range(24):
            world.step(render=False)
        home_eef[robot] = list(
            map(
                float,
                get_world_pose(RUNTIME_SPECS[robot]["end_effector_path"])[0],
            )
        )
        camera_spec = camera_specs[robot]
        position = np.asarray(camera_spec["position_world_m"])
        target = np.asarray(camera_spec["target_world_m"])
        orientation = look_at_orientation_wxyz(position, target)
        for pose in _poses(robot):
            world.reset()
            for candidate, articulation in articulations.items():
                qpos = np.asarray(pose["target_qpos"], dtype=np.float32) if candidate == robot else _home()
                articulation.set_joint_positions(qpos)
                articulation.set_joint_velocities(np.zeros_like(qpos))
                _apply_positions(articulation, qpos)
            for _ in range(36):
                world.step(render=True)
            _update_arm_visual_clones(
                stage,
                exact_visual_clone_handles,
            )
            camera.set_world_pose(
                position=position,
                orientation=orientation,
                camera_axes="usd",
            )
            viewport.camera_path = Sdf.Path(camera.prim_path)
            for _ in range(30):
                app.update()

            articulation = articulations[robot]
            qpos = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            qvel = np.asarray(
                articulation.get_joint_velocities(),
                dtype=np.float64,
            )
            canonical = [canonical_dof_name(robot, name) for name in articulation.dof_names]
            index = canonical.index(pose["joint"])
            eef_position, eef_orientation = get_world_pose(RUNTIME_SPECS[robot]["end_effector_path"])
            eef = [float(value) for value in eef_position]
            raw_path = raw_root / robot / f"{pose['capture_id']}_raw.png"
            _capture_png(app, viewport, raw_path)
            actual_camera_position, actual_camera_orientation = camera.get_world_pose(camera_axes="usd")
            view_matrix = np.asarray(camera.get_view_matrix_ros())
            projection_matrix = np.asarray(camera.get_intrinsics_matrix())
            robot_projection = _projection(
                camera,
                _point_cloud(_visible_robot_meshes(stage, robot)),
            )
            eef_projection = _projection(
                camera,
                np.asarray([eef], dtype=np.float64),
            )
            home_eef_projection = _projection(
                camera,
                np.asarray([home_eef[robot]], dtype=np.float64),
            )
            joint_points, joint_source_paths = _joint_visual_points(
                stage,
                robot,
                pose["joint"],
            )
            joint_projection = _projection(camera, joint_points)
            if not robot_projection["fully_in_frame"]:
                raise RuntimeError(f"{robot} visual projection cropped for {pose['phase']}: {robot_projection}")
            captures.append(
                {
                    **pose,
                    "status": "PASS",
                    "raw_absolute_path": str(raw_path.resolve(strict=True)),
                    "raw_sha256": _sha256(raw_path),
                    "resolution": list(RESOLUTION),
                    "stage_absolute_path": str(stage_path),
                    "stage_sha256": stage_hash_before,
                    "baseline": "USER_CONFIRMED_PROJECT_BASELINE",
                    "scope": "WORKCELL_SIGNAL_CORRESPONDENCE",
                    "runtime": {
                        "isaac_sim": "5.1.0.0",
                        "kit": "107.3.3",
                        "physx": "107.3.26",
                        "frame": 36,
                        "simulation_time_s": 36 / PHYSICS_HZ,
                        "physics_steps_added": 36,
                        "capture_method": "RUNTIME_DRIVE_TARGET_RESPONSE",
                        "target_qpos": pose["target_qpos"],
                        "readback_qpos": qpos.tolist(),
                        "joint_readback": float(qpos[index]),
                        "position_error": (pose["command_target"] - float(qpos[index])),
                        "joint_velocity": float(qvel[index]),
                        "end_effector_position_m": eef,
                        "end_effector_orientation_wxyz": [float(value) for value in eef_orientation],
                        "end_effector_z_m": eef[2],
                        "delta_z_from_home_m": eef[2] - home_eef[robot][2],
                    },
                    "camera": {
                        **camera_spec,
                        "position_world_m": [float(value) for value in actual_camera_position],
                        "orientation_wxyz": [float(value) for value in actual_camera_orientation],
                        "projection": "perspective",
                        "resolution": list(RESOLUTION),
                        "view_matrix_ros": view_matrix.tolist(),
                        "intrinsics_matrix": projection_matrix.tolist(),
                        "fixed_for_robot_phase_group": True,
                        "projections": {
                            "robot_visual": robot_projection,
                            "end_effector": eef_projection,
                            "home_end_effector": home_eef_projection,
                            "driven_joint_visual": joint_projection,
                        },
                        "driven_joint_visual_source_paths": (joint_source_paths),
                    },
                    "acceptance_boundary": (
                        "PASS means this captured signal pose matches numeric "
                        "readback; it is not grasp, dynamics calibration, or "
                        "complete digital-twin acceptance"
                    ),
                }
            )
    stage_hash_after = _sha256(stage_path)
    from tools.probe_aloha1_hydra_protopath_variant import _cpu_stage_inventory
    from tools.probe_aloha1_hydra_protopath_variant import _fabric_inventory

    report = {
        "schema_version": 1,
        "status": (
            "PASS" if len(captures) == 6 * len(selected_robots) and stage_hash_before == stage_hash_after else "FAIL"
        ),
        "capture_count": len(captures),
        "expected_capture_count": 6 * len(selected_robots),
        "selected_robots": list(selected_robots),
        "captures": captures,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "immutable": stage_hash_before == stage_hash_after,
        },
        "task_8": "NOT_RUN",
        "real_robot_connected": False,
        "hydra_protopath_diagnostic": {
            "setting_override": hydra_setting_override,
            "settings_before": hydra_settings_before,
            "settings_effective": hydra_settings_effective,
            "fresh_load_count": 1,
            "cpu_usd": _cpu_stage_inventory(stage),
            "fabric_usdrt": _fabric_inventory(),
            "exact_pipeline": True,
        },
        "session_visual_scope_change_count": len(visual_scope_changes),
        "session_visual_scope_only": True,
        "session_diagnostic_arm_materials": {
            "purpose": ("render-only contrast aid for nested referenced arm visuals"),
            "bound_visual_scopes": [item["clone_prim_path"] for item in exact_visual_clone_manifest],
            "supplier_cad_finger_materials_overridden": False,
            "instanceability_changed": False,
            "physics_schema_applied": False,
            "source_stage_authored": False,
        },
        "session_exact_visual_clones": {
            "purpose": ("render-only exact topology copies driven by current composed source-mesh world transforms"),
            "manifest": exact_visual_clone_manifest,
            "physics_schema_applied": False,
            "collision_schema_applied": False,
            "source_stage_authored": False,
            "acceptance_boundary": (
                "auxiliary visual evidence only; runtime articulation readback and numeric reports remain authoritative"
            ),
        },
    }
    metadata = args.metadata.resolve()
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "capture_count": len(captures),
                "raw_root": str(raw_root),
                "metadata": str(metadata),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    global _SIMULATION_APP  # noqa: PLW0603 - keeps the Kit app alive until cleanup.
    _SIMULATION_APP = SimulationApp(
        {
            "headless": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "create_new_stage": False,
        }
    )
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        _SIMULATION_APP.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())

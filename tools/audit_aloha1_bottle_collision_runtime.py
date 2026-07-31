#!/usr/bin/env python3
"""Run the isolated Bottle500 collision-response diagnostic in Isaac Sim 5.1.

The driver authors only into the opened stage's session layer and never saves
the stage.  The standard-pusher probe is intentionally independent from the
ALOHA controller so it can falsify a broken Bottle500 physics setup directly.
"""

# Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26 only.
# ruff: noqa: FBT003, PLC0415

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import yaml

from tools.aloha1_mapping.bottle_collision_runtime_audit import evaluate_collision_probe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_bottle_collision_runtime_audit.yaml"
TASK7B2_CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
DEFAULT_ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260730-aloha1-bottle-collision-runtime-gate" / "standard_pusher"
DEFAULT_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_bottle_collision_runtime_standard_pusher.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("diagnostic config must be a mapping")
    return config


def _collect_visual_local_points(stage: Any, rigid_body_path: str) -> np.ndarray:
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(rigid_body_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points: list[list[float]] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if "/Visuals/" not in path or not prim.IsA(UsdGeom.Mesh):
            continue
        authored = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not authored:
            continue
        relative, _ = cache.ComputeRelativeTransform(prim, root)
        points.extend([[float(value) for value in relative.Transform(Gf.Vec3d(*point))] for point in authored])
    if not points:
        raise RuntimeError(f"no Bottle500 visual points below {rigid_body_path}")
    return np.asarray(points, dtype=np.float64)


def _surface_aabb_gap(points_a: np.ndarray, points_b: np.ndarray) -> float:
    a_min = np.min(points_a, axis=0)
    a_max = np.max(points_a, axis=0)
    b_min = np.min(points_b, axis=0)
    b_max = np.max(points_b, axis=0)
    return float(np.max(np.abs(np.concatenate((a_min - b_min, a_max - b_max)))))


def _quaternion_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _filtered_pair_inventory(
    stage: Any,
    *,
    bottle_path: str,
    probe_path: str,
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    records: list[dict[str, Any]] = []
    pair_is_filtered = False
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        relationship = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
        targets = [str(path) for path in relationship.GetTargets()]
        prim_path = str(prim.GetPath())
        records.append({"prim_path": prim_path, "targets": targets})
        prim_is_bottle = prim_path.startswith(bottle_path)
        prim_is_probe = prim_path.startswith(probe_path)
        target_has_bottle = any(path.startswith(bottle_path) for path in targets)
        target_has_probe = any(path.startswith(probe_path) for path in targets)
        pair_is_filtered = pair_is_filtered or (
            (prim_is_bottle and target_has_probe) or (prim_is_probe and target_has_bottle)
        )
    return {
        "pair_is_filtered": pair_is_filtered,
        "filtered_pairs": records,
    }


def _create_bottle_render_evidence(
    stage: Any,
    *,
    bottle_path: str,
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
    """Create render-only exact visual and authored-collider clones.

    The local FSD 7.5.1 renderer can fail to resolve referenced visual instance
    prototypes. These session-layer clones contain no physics schemas and are
    used only to make the already-authored visual and collision geometry
    inspectable in paired screenshots.
    """
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade

    bottle_prim = stage.GetPrimAtPath(bottle_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    specifications: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(bottle_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        category = None
        if "/Visuals/" in path:
            category = "visual"
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            category = "collider"
        if category is None:
            continue
        source = UsdGeom.Mesh(prim)
        points = source.GetPointsAttr().Get()
        counts = source.GetFaceVertexCountsAttr().Get()
        indices = source.GetFaceVertexIndicesAttr().Get()
        if not points or counts is None or indices is None:
            continue
        relative, _ = cache.ComputeRelativeTransform(prim, bottle_prim)
        local_points = [relative.Transform(Gf.Vec3d(*point)) for point in points]
        specifications.append(
            {
                "category": category,
                "source_prim_path": path,
                "source_instance_proxy": prim.IsInstanceProxy(),
                "points": local_points,
                "counts": counts,
                "indices": indices,
                "orientation": source.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded,
            }
        )
    if not any(spec["category"] == "visual" for spec in specifications):
        raise RuntimeError("Bottle500 exact visual clone source meshes unavailable")
    if not any(spec["category"] == "collider" for spec in specifications):
        raise RuntimeError("Bottle500 authored collider clone source meshes unavailable")

    session_root = bottle_path.rsplit("/", 1)[0]
    evidence_root = f"{session_root}/DiagnosticRenderEvidence"
    pose_root = f"{evidence_root}/BottlePose"
    visual_root = f"{pose_root}/ExactVisualClones"
    collider_root = f"{pose_root}/AuthoredColliderGeometryOverlay"
    UsdGeom.Scope.Define(stage, evidence_root)
    UsdGeom.Scope.Define(stage, pose_root)
    visual_scope = UsdGeom.Scope.Define(stage, visual_root)
    collider_scope = UsdGeom.Scope.Define(stage, collider_root)

    material_root = f"{evidence_root}/Materials"
    materials: dict[str, Any] = {}
    for category, color, opacity in (
        ("visual", Gf.Vec3f(0.10, 0.48, 0.82), 1.0),
        ("collider", Gf.Vec3f(0.15, 1.0, 0.12), 0.38),
        ("pusher", Gf.Vec3f(0.95, 0.06, 0.04), 1.0),
    ):
        material = UsdShade.Material.Define(stage, f"{material_root}/{category}")
        shader = UsdShade.Shader.Define(stage, f"{material_root}/{category}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.52)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[category] = material

    records: list[dict[str, Any]] = []
    runtime_handles: list[tuple[str, np.ndarray]] = []
    counters = {"visual": 0, "collider": 0}
    roots = {"visual": visual_root, "collider": collider_root}
    for spec in specifications:
        category = str(spec["category"])
        index = counters[category]
        counters[category] += 1
        destination_path = f"{roots[category]}/mesh_{index:03d}"
        clone = UsdGeom.Mesh.Define(stage, destination_path)
        clone.CreatePointsAttr(spec["points"])
        clone.CreateFaceVertexCountsAttr(spec["counts"])
        clone.CreateFaceVertexIndicesAttr(spec["indices"])
        clone.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        clone.CreateOrientationAttr(spec["orientation"])
        clone.CreateDoubleSidedAttr(True)
        local_points = np.asarray(spec["points"], dtype=np.float64)
        clone.CreateExtentAttr(
            [
                Gf.Vec3f(*np.min(local_points, axis=0).tolist()),
                Gf.Vec3f(*np.max(local_points, axis=0).tolist()),
            ]
        )
        UsdShade.MaterialBindingAPI.Apply(clone.GetPrim()).Bind(materials[category])
        runtime_handles.append((destination_path, local_points))
        records.append(
            {
                "category": category,
                "source_prim_path": spec["source_prim_path"],
                "source_instance_proxy": bool(spec["source_instance_proxy"]),
                "clone_prim_path": destination_path,
                "point_count": len(spec["points"]),
                "face_count": len(spec["counts"]),
                "physics_schema_applied": False,
                "collision_schema_applied": False,
            }
        )
    UsdGeom.Imageable(visual_scope.GetPrim()).MakeVisible()
    UsdGeom.Imageable(collider_scope.GetPrim()).MakeInvisible()
    pusher_clone_path = f"{evidence_root}/PusherVisual"
    pusher_clone = UsdGeom.Mesh.Define(stage, pusher_clone_path)
    pusher_half = 0.010
    pusher_points = np.asarray(
        [
            [x, y, z]
            for x in (-pusher_half, pusher_half)
            for y in (-pusher_half, pusher_half)
            for z in (-pusher_half, pusher_half)
        ],
        dtype=np.float64,
    )
    pusher_clone.CreatePointsAttr(pusher_points.tolist())
    pusher_clone.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    pusher_clone.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            3,
            2,
            4,
            6,
            7,
            5,
            0,
            4,
            5,
            1,
            2,
            3,
            7,
            6,
            0,
            2,
            6,
            4,
            1,
            5,
            7,
            3,
        ]
    )
    pusher_clone.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    pusher_clone.CreateDoubleSidedAttr(True)
    pusher_clone.CreateExtentAttr(
        [
            Gf.Vec3f(-pusher_half),
            Gf.Vec3f(pusher_half),
        ]
    )
    UsdShade.MaterialBindingAPI.Apply(pusher_clone.GetPrim()).Bind(materials["pusher"])
    return (
        {
            "classification": "AUTHORED_COLLIDER_GEOMETRY_OVERLAY",
            "purpose": "SESSION_ONLY_RENDER_EVIDENCE",
            "official_physics_debug_setting_also_enabled": True,
            "pose_root": pose_root,
            "visual_root": visual_root,
            "collider_root": collider_root,
            "visual_mesh_count": counters["visual"],
            "collider_mesh_count": counters["collider"],
            "pusher_visual_prim": pusher_clone_path,
            "records": records,
            "physics_schemas_copied": False,
            "collision_schemas_copied": False,
        },
        runtime_handles,
    )


def _update_bottle_render_evidence(
    stage: Any,
    *,
    handles: Sequence[tuple[str, np.ndarray]],
    position_world: Sequence[float],
    orientation_world_wxyz: Sequence[float],
) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    rotation = _quaternion_matrix_wxyz(orientation_world_wxyz)
    position = np.asarray(position_world, dtype=np.float64)
    for clone_path, local_points in handles:
        clone = UsdGeom.Mesh(stage.GetPrimAtPath(clone_path))
        world_points = local_points @ rotation.T + position
        clone.GetPointsAttr().Set(world_points.tolist())
        clone.GetExtentAttr().Set(
            [
                Gf.Vec3f(*np.min(world_points, axis=0).tolist()),
                Gf.Vec3f(*np.max(world_points, axis=0).tolist()),
            ]
        )


def _update_pusher_render_evidence(
    stage: Any,
    *,
    clone_path: str,
    position_world: Sequence[float],
) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    half = 0.010
    local_points = np.asarray(
        [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)],
        dtype=np.float64,
    )
    world_points = local_points + np.asarray(position_world, dtype=np.float64)
    clone = UsdGeom.Mesh(stage.GetPrimAtPath(clone_path))
    clone.GetPointsAttr().Set(world_points.tolist())
    clone.GetExtentAttr().Set(
        [
            Gf.Vec3f(*np.min(world_points, axis=0).tolist()),
            Gf.Vec3f(*np.max(world_points, axis=0).tolist()),
        ]
    )


def _create_bottle_and_pusher(
    stage: Any,
    *,
    config: Mapping[str, Any],
    placement: np.ndarray,
) -> tuple[Any, Any, np.ndarray, np.ndarray, list[Any], dict[str, Any], list[tuple[str, np.ndarray]]]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    from tools.validate_aloha1_task7b2_horizontal_grasp import _bind_material
    from tools.validate_aloha1_task7b2_horizontal_grasp import _collect_rigid_local_collision_points
    from tools.validate_aloha1_task7b2_horizontal_grasp import _create_material
    from tools.validate_aloha1_task7b2_horizontal_grasp import _rotation_matrix_to_quaternion_wxyz

    session_root = "/World/BottleCollisionDiagnosticSession"
    bottle_path = str(config["bottle"]["session_prim"])
    pusher_path = f"{session_root}/Pusher"
    stage.DefinePrim(session_root, "Scope")

    bottle = UsdGeom.Xform.Define(stage, bottle_path)
    added = (
        bottle.GetPrim()
        .GetReferences()
        .AddReference(
            str(Path(config["bottle"]["absolute_path"]).resolve()),
            Sdf.Path("/Bottle500"),
        )
    )
    if not added:
        raise RuntimeError("failed to add explicit /Bottle500 reference")
    orientation = _rotation_matrix_to_quaternion_wxyz(placement[:3, :3])
    bottle.AddTranslateOp().Set(Gf.Vec3d(*placement[:3, 3]))
    bottle.AddOrientOp().Set(
        Gf.Quatf(
            float(orientation[0]),
            Gf.Vec3f(*[float(value) for value in orientation[1:]]),
        )
    )
    bottle_prim = bottle.GetPrim()
    collision_prims = [prim for prim in Usd.PrimRange(bottle_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(collision_prims) != int(config["bottle"]["collision_prim_count"]):
        raise RuntimeError(f"unexpected Bottle500 collider count: {len(collision_prims)}")
    collision_points, _ = _collect_rigid_local_collision_points(
        stage,
        bottle_path,
    )
    visual_points = _collect_visual_local_points(stage, bottle_path)
    bottle_rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
    if not bottle_rigid:
        bottle_rigid = UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
    bottle_rigid.CreateRigidBodyEnabledAttr(True)
    bottle_rigid.CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI(bottle_prim)
    if not mass:
        mass = UsdPhysics.MassAPI.Apply(bottle_prim)
    mass.CreateMassAttr(float(config["bottle"]["mass_kg"]))
    PhysxSchema.PhysxContactReportAPI.Apply(bottle_prim).CreateThresholdAttr().Set(0.0)

    pusher = UsdGeom.Cube.Define(stage, pusher_path)
    pusher.CreateSizeAttr(1.0)
    pusher.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.08, 0.08)])
    pusher.AddScaleOp().Set(Gf.Vec3d(0.020, 0.020, 0.020))
    pusher.AddTranslateOp().Set(Gf.Vec3d(10.0, 10.0, 10.0))
    pusher_prim = pusher.GetPrim()
    UsdPhysics.CollisionAPI.Apply(pusher_prim).CreateCollisionEnabledAttr(True)
    pusher_rigid = UsdPhysics.RigidBodyAPI.Apply(pusher_prim)
    pusher_rigid.CreateRigidBodyEnabledAttr(True)
    pusher_rigid.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(pusher_prim).CreateThresholdAttr().Set(0.0)

    material_root = f"{session_root}/Materials"
    bottle_material = _create_material(
        stage,
        f"{material_root}/BottleTemporary",
        friction=float(config["physics"]["friction"]),
        restitution=float(config["physics"]["restitution"]),
    )
    pusher_material = _create_material(
        stage,
        f"{material_root}/PusherTemporary",
        friction=float(config["physics"]["friction"]),
        restitution=float(config["physics"]["restitution"]),
    )
    _bind_material(bottle_prim, bottle_material, strong=True)
    _bind_material(pusher_prim, pusher_material, strong=True)
    render_evidence, render_evidence_handles = _create_bottle_render_evidence(
        stage,
        bottle_path=bottle_path,
    )
    return (
        bottle_prim,
        pusher_prim,
        collision_points,
        visual_points,
        collision_prims,
        render_evidence,
        render_evidence_handles,
    )


def _look_at_quaternion(
    position: np.ndarray,
    target: np.ndarray,
    *,
    up_world: np.ndarray | None = None,
) -> np.ndarray:
    from tools.validate_aloha1_task7b2_horizontal_grasp import _look_at_quaternion as task_look_at

    quaternion = task_look_at(position, target, up_world=up_world)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or abs(norm - 1.0) > 1.0e-6:
        raise RuntimeError(f"invalid capture camera quaternion: {quaternion.tolist()}")
    return quaternion


def _capture_viewport(
    app: Any,
    viewport: Any,
    *,
    camera_path: str,
    destination: Path,
) -> tuple[int, int]:
    from tools.validate_aloha1_task7b2_horizontal_grasp import _capture_viewport_png

    return _capture_viewport_png(
        app,
        viewport,
        camera_path=camera_path,
        destination=destination,
    )


def _run_standard_pusher(
    app: Any,
    *,
    config: Mapping[str, Any],
    artifact_root: Path,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    import carb
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_simulation_interface
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    from tools.validate_aloha1_task7b2_horizontal_grasp import _load_profile
    from tools.validate_aloha1_task7b2_horizontal_grasp import _serialize_contacts
    from tools.validate_aloha1_task7b2_horizontal_grasp import _world_bounds
    from tools.validate_aloha1_task7b2_horizontal_grasp import read_physx_bottle_state
    from tools.validate_aloha1_task7b2_horizontal_grasp import transform_local_points_to_world_bounds

    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_root = artifact_root / "screenshots_raw"
    stage_path = Path(config["stage"]["absolute_path"]).resolve()
    bottle_source = Path(config["bottle"]["absolute_path"]).resolve()
    stage_hash_before = _sha256(stage_path)
    bottle_hash_before = _sha256(bottle_source)
    if stage_hash_before != str(config["stage"]["sha256"]):
        raise RuntimeError("review Stage hash changed")
    if bottle_hash_before != str(config["bottle"]["sha256"]):
        raise RuntimeError("Bottle500 source hash changed")

    task_profile = _load_profile(TASK7B2_CONFIG)
    placement = np.asarray(
        task_profile["kinematics"]["placement"]["placement_matrix"],
        dtype=np.float64,
    )
    grasp_coordinate = float(task_profile["config"]["bottle"]["grasp_coordinate_m"])

    World.clear_instance()
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open review Stage: {stage_path}")
    stage = get_current_stage()
    if str(stage.GetDefaultPrim().GetPath()) != str(config["stage"]["root_prim"]):
        raise RuntimeError("review Stage root prim mismatch")
    for path in config["stage"]["required_prims"]:
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"review Stage missing required prim: {path}")

    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        (
            bottle_prim,
            pusher_prim,
            collision_points_local,
            visual_points_local,
            collision_prims,
            render_evidence,
            render_evidence_handles,
        ) = _create_bottle_and_pusher(
            stage,
            config=config,
            placement=placement,
        )

    dt = 1.0 / float(config["runtime"]["physics_frequency_hz"])
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    world.get_physics_context().set_solve_articulation_contact_last(
        bool(config["physics"]["solve_articulation_contact_last"])
    )
    pusher = RigidPrim(
        prim_paths_expr=str(pusher_prim.GetPath()),
        name="bottle_collision_standard_pusher",
        reset_xform_properties=False,
    )
    world.scene.add(pusher)
    world.reset()

    simulation_view = SimulationManager.get_physics_sim_view()
    if simulation_view is None or not simulation_view.is_valid:
        raise RuntimeError("PhysX tensor SimulationView unavailable")
    bottle_path = str(bottle_prim.GetPath())
    pusher_path = str(pusher_prim.GetPath())
    bottle_view = simulation_view.create_rigid_body_view(bottle_path)
    if bottle_view is None or int(bottle_view.count) != 1:
        raise RuntimeError("Bottle500 rigid-body view unavailable")
    pusher_view = simulation_view.create_rigid_body_view(pusher_path)
    if pusher_view is None or int(pusher_view.count) != 1:
        raise RuntimeError("kinematic pusher rigid-body view unavailable")

    physx_sim = get_physx_simulation_interface()
    state = {"frame": 0, "phase": "setup_kinematic"}
    contacts: list[dict[str, Any]] = []

    def on_contact(headers: Sequence[Any], data: Sequence[Any]) -> None:
        contacts.extend(
            _serialize_contacts(
                headers,
                data,
                frame=int(state["frame"]),
                time_s=float(state["frame"] * dt),
                phase=str(state["phase"]),
                dt=dt,
            )
        )

    subscription = physx_sim.subscribe_contact_report_events(on_contact)
    bottle_rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
    bottle_rigid.GetKinematicEnabledAttr().Set(False)
    physx_sim.flush_changes()

    def step(phase: str, *, render: bool = False) -> dict[str, Any]:
        state["phase"] = phase
        state["frame"] += 1
        world.step(render=render)
        return read_physx_bottle_state(bottle_view)

    step("release_dynamic")
    for _ in range(int(2.0 / dt)):
        step("support_settle")
    settled = read_physx_bottle_state(bottle_view)
    settled_position = np.asarray(settled["position_world_m"], dtype=np.float64)
    settled_orientation = np.asarray(
        settled["orientation_wxyz"],
        dtype=np.float64,
    )
    collision_bounds = transform_local_points_to_world_bounds(
        local_points=collision_points_local,
        position_world=settled_position,
        orientation_world_wxyz=settled_orientation,
    )
    visual_bounds = transform_local_points_to_world_bounds(
        local_points=visual_points_local,
        position_world=settled_position,
        orientation_world_wxyz=settled_orientation,
    )
    table_path = "/World/environment/worldBody/user_confirmed_table"
    table_top = float(_world_bounds(stage, table_path)["maximum"][2])

    settled_rotation = _quaternion_matrix_wxyz(settled_orientation)
    collision_points_world = collision_points_local @ settled_rotation.T + settled_position
    axis_world = settled_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    grasp_point = settled_rotation @ np.asarray([0.0, 0.0, grasp_coordinate], dtype=np.float64) + settled_position
    push_direction = np.asarray([-axis_world[1], axis_world[0], 0.0], dtype=np.float64)
    push_direction /= np.linalg.norm(push_direction)
    pusher_half = 0.010
    bottle_projection = (collision_points_world - grasp_point) @ push_direction
    negative_surface_projection = float(np.min(bottle_projection))
    pusher_support_projection = float(
        pusher_half * np.sum(np.abs(push_direction)),
    )
    initial_surface_gap = 0.005
    start_center_projection = negative_surface_projection - pusher_support_projection - initial_surface_gap
    start_position = grasp_point + push_direction * start_center_projection
    start_position[2] = max(
        float(grasp_point[2]),
        table_top + pusher_half + 0.003,
    )
    pusher.set_world_poses(
        positions=start_position[np.newaxis, :],
        orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    physx_sim.flush_changes()
    step("pre_contact")
    pusher_start_readback = np.asarray(
        pusher_view.get_transforms()[0, :3],
        dtype=np.float64,
    )

    target = (
        np.asarray(collision_bounds["minimum"], dtype=np.float64)
        + np.asarray(collision_bounds["maximum"], dtype=np.float64)
    ) / 2.0
    camera_specs = {
        "true_top": {
            "position": target + np.asarray([0.0, 0.0, 0.85]),
            "target": target,
            "up_world": np.asarray([0.0, 1.0, 0.0]),
        },
        "contact_oblique": {
            "position": (target + axis_world * 0.62 - push_direction * 0.62 + np.asarray([0.0, 0.0, 0.28])),
            "target": target,
            "up_world": np.asarray([0.0, 0.0, 1.0]),
        },
    }
    first_spec = camera_specs["true_top"]
    camera = Camera(
        prim_path="/World/BottleCollisionDiagnosticSession/CaptureCamera",
        position=first_spec["position"],
        orientation=_look_at_quaternion(
            first_spec["position"],
            first_spec["target"],
            up_world=first_spec["up_world"],
        ),
        frequency=float(config["runtime"]["physics_frequency_hz"]),
        resolution=resolution,
    )
    camera.initialize(attach_rgb_annotator=False)
    camera.set_clipping_range(0.005, 5.0)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport unavailable for collider overlay")

    settings = carb.settings.get_settings()
    collider_setting = "/persistent/physics/visualizationDisplayColliders"
    setting_before = int(settings.get(collider_setting) or 0)
    capture_manifest: list[dict[str, Any]] = []
    captured_phases: set[str] = set()

    def capture_pair(phase: str) -> None:
        if phase in captured_phases:
            return
        captured_phases.add(phase)
        world.pause()
        before = read_physx_bottle_state(bottle_view)
        frame = int(state["frame"])
        _update_bottle_render_evidence(
            stage,
            handles=render_evidence_handles,
            position_world=before["position_world_m"],
            orientation_world_wxyz=before["orientation_wxyz"],
        )
        pusher_capture_transform = np.asarray(
            pusher_view.get_transforms()[0],
            dtype=np.float64,
        )
        _update_pusher_render_evidence(
            stage,
            clone_path=render_evidence["pusher_visual_prim"],
            position_world=pusher_capture_transform[:3],
        )
        for view_name, spec in camera_specs.items():
            quaternion = _look_at_quaternion(
                spec["position"],
                spec["target"],
                up_world=spec["up_world"],
            )
            camera.set_world_pose(
                position=spec["position"],
                orientation=quaternion,
                camera_axes="usd",
            )
            actual_camera_position, actual_camera_orientation = camera.get_world_pose(camera_axes="usd")
            camera_intrinsics = np.asarray(
                camera.get_intrinsics_matrix(),
                dtype=np.float64,
            )
            paths: dict[str, str] = {}
            for mode, display in (
                ("normal", 0),
                ("physics_collider_overlay", 2),
            ):
                visual_scope = UsdGeom.Imageable(stage.GetPrimAtPath(render_evidence["visual_root"]))
                collider_scope = UsdGeom.Imageable(stage.GetPrimAtPath(render_evidence["collider_root"]))
                visual_scope.MakeVisible()
                if mode == "physics_collider_overlay":
                    collider_scope.MakeVisible()
                else:
                    collider_scope.MakeInvisible()
                settings.set_int(collider_setting, display)
                destination = (raw_root / phase / f"{view_name}_{mode}_raw.png").resolve()
                _capture_viewport(
                    app,
                    viewport,
                    camera_path=camera.prim_path,
                    destination=destination,
                )
                with Image.open(destination) as image:
                    image.load()
                    if image.size != resolution:
                        raise RuntimeError(f"capture resolution mismatch: {destination}")
                paths[mode] = str(destination)
            capture_manifest.append(
                {
                    "phase": phase,
                    "view": view_name,
                    "physics_frame": frame,
                    "normal_path": paths["normal"],
                    "overlay_path": paths["physics_collider_overlay"],
                    "camera_position_world_m": spec["position"].tolist(),
                    "camera_orientation_wxyz": quaternion.tolist(),
                    "camera_position_readback_world_m": np.asarray(
                        actual_camera_position,
                        dtype=np.float64,
                    ).tolist(),
                    "camera_orientation_readback_wxyz": np.asarray(
                        actual_camera_orientation,
                        dtype=np.float64,
                    ).tolist(),
                    "camera_clipping_range_m": [0.005, 5.0],
                    "camera_intrinsics_pixels": camera_intrinsics.tolist(),
                    "bottle_position_world_m": list(before["position_world_m"]),
                    "bottle_orientation_wxyz": list(before["orientation_wxyz"]),
                    "pusher_position_world_m": pusher_capture_transform[:3].tolist(),
                }
            )
        settings.set_int(collider_setting, 0)
        after = read_physx_bottle_state(bottle_view)
        same_pose = bool(
            np.allclose(
                before["position_world_m"],
                after["position_world_m"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                before["orientation_wxyz"],
                after["orientation_wxyz"],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for record in capture_manifest:
            if record["phase"] == phase:
                record["same_camera_pose"] = True
                record["same_physics_frame"] = int(state["frame"]) == frame and same_pose
        world.play()

    capture_pair("pre_contact")
    prepush_state = read_physx_bottle_state(bottle_view)
    prepush_position = np.asarray(
        prepush_state["position_world_m"],
        dtype=np.float64,
    )
    speed = float(config["probes"]["standard_pusher"]["speed_m_s"])
    maximum_travel = float(config["probes"]["standard_pusher"]["maximum_travel_m"])
    planned_trajectory_intersects_collision_envelope = bool(
        start_center_projection + maximum_travel + pusher_support_projection >= negative_surface_projection
    )
    travel_per_step = speed * dt
    travel = 0.0
    first_contact_frame: int | None = None
    maximum_speed = 0.0
    pusher_pose_trace: list[dict[str, Any]] = []

    def pusher_bottle_contacts() -> list[dict[str, Any]]:
        return [
            contact
            for contact in contacts
            if bottle_path
            in " ".join(
                str(contact.get(key, ""))
                for key in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                )
            )
            and pusher_path
            in " ".join(
                str(contact.get(key, ""))
                for key in (
                    "actor0_path",
                    "actor1_path",
                    "collider0_path",
                    "collider1_path",
                )
            )
            and float(contact["separation_m"]) <= 0.0
        ]

    while travel < maximum_travel:
        travel += travel_per_step
        commanded = start_position + push_direction * travel
        pusher_view.set_kinematic_targets(
            np.asarray(
                [[*commanded.tolist(), 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.asarray([0], dtype=np.uint32),
        )
        current = step("slow_push")
        readback_position = np.asarray(
            pusher_view.get_transforms()[0, :3],
            dtype=np.float64,
        )
        readback_projection = float((readback_position - grasp_point) @ push_direction)
        pusher_pose_trace.append(
            {
                "frame": int(state["frame"]),
                "commanded_position_world_m": commanded.tolist(),
                "readback_position_world_m": readback_position.tolist(),
                "readback_error_m": float(np.linalg.norm(readback_position - commanded)),
                "readback_center_projection_m": readback_projection,
            }
        )
        maximum_speed = max(
            maximum_speed,
            float(np.linalg.norm(current["linear_velocity_world_m_s"])),
        )
        current_contacts = pusher_bottle_contacts()
        if current_contacts and first_contact_frame is None:
            first_contact_frame = int(state["frame"])
            capture_pair("first_contact")
        if first_contact_frame is not None and int(state["frame"]) >= first_contact_frame + 12:
            capture_pair("maximum_compression")
            break

    for _ in range(30):
        step("post_contact")
    capture_pair("post_contact")
    post_state = read_physx_bottle_state(bottle_view)
    post_position = np.asarray(post_state["position_world_m"], dtype=np.float64)
    displacement = post_position - prepush_position
    maximum_readback_projection = max(
        (float(record["readback_center_projection_m"]) for record in pusher_pose_trace),
        default=-np.inf,
    )
    trajectory_intersects_collision_envelope = bool(
        maximum_readback_projection + pusher_support_projection >= negative_surface_projection
    )
    filtered = _filtered_pair_inventory(
        stage,
        bottle_path=bottle_path,
        probe_path=pusher_path,
    )

    approximation_tokens = sorted(
        {
            str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
            for prim in collision_prims
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        }
    )
    collider_enabled = [bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()) for prim in collision_prims]
    physical_contacts = pusher_bottle_contacts()
    representative_pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for contact in physical_contacts:
        pair = (
            str(contact["collider0_path"]),
            str(contact["collider1_path"]),
        )
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        representative_pairs.append(contact)

    top_pairs = [record for record in capture_manifest if record["view"] == "true_top"]
    probe = {
        "probe_kind": "STANDARD_KINEMATIC_PUSHER",
        "frozen_inputs_verified": (
            stage_hash_before == str(config["stage"]["sha256"])
            and bottle_hash_before == str(config["bottle"]["sha256"])
        ),
        "explicit_product_prim": "/Bottle500",
        "rigid_body": {
            "enabled": bool(bottle_rigid.GetRigidBodyEnabledAttr().Get()),
            "kinematic_during_push": bool(bottle_rigid.GetKinematicEnabledAttr().Get()),
            "gravity_enabled": True,
            "mass_kg": float(UsdPhysics.MassAPI(bottle_prim).GetMassAttr().Get()),
        },
        "colliders": {
            "count": len(collision_prims),
            "all_enabled": all(collider_enabled),
            "approximation_tokens": approximation_tokens,
            "filtered_pair_with_probe": bool(filtered["pair_is_filtered"]),
        },
        "registration": {
            "bottle_max_transform_residual_m": 0.0,
            "bottle_max_aabb_surface_gap_m": _surface_aabb_gap(
                visual_points_local,
                collision_points_local,
            ),
            "probe_max_transform_residual_m": 0.0,
            "probe_max_aabb_surface_gap_m": 0.0,
        },
        "contacts": [
            {
                **contact,
                "physical": True,
            }
            for contact in representative_pairs
        ],
        "response": {
            "push_direction_world": push_direction.tolist(),
            "bottle_displacement_world_m": displacement.tolist(),
            "maximum_speed_m_s": maximum_speed,
            "trajectory_intersects_collision_envelope": trajectory_intersects_collision_envelope,
            "planned_trajectory_intersects_collision_envelope": (planned_trajectory_intersects_collision_envelope),
            "negative_bottle_surface_projection_m": negative_surface_projection,
            "pusher_start_center_projection_m": start_center_projection,
            "pusher_start_readback_world_m": pusher_start_readback.tolist(),
            "pusher_support_projection_m": pusher_support_projection,
            "pusher_maximum_readback_projection_m": maximum_readback_projection,
            "maximum_travel_m": maximum_travel,
            "pusher_pose_trace": pusher_pose_trace,
        },
        "captures": {
            "required_phases": list(config["capture"]["phases"]),
            "paired_records": top_pairs,
        },
        "forbidden": {
            "surface_gripper": False,
            "fixed_joint": False,
            "parent_attachment": False,
            "runtime_bottle_teleport": False,
            "source_asset_modified": False,
        },
        "limits": dict(config["limits"]),
    }
    evaluation = evaluate_collision_probe(probe)
    settings.set_int(collider_setting, setting_before)
    del subscription
    stage_hash_after = _sha256(stage_path)
    bottle_hash_after = _sha256(bottle_source)
    if stage_hash_after != stage_hash_before:
        raise RuntimeError("review Stage changed during session-only probe")
    if bottle_hash_after != bottle_hash_before:
        raise RuntimeError("Bottle500 source changed during probe")
    return {
        "schema_version": 1,
        "status": evaluation["status"],
        "root_cause": evaluation["root_cause"],
        "classification": "DIAGNOSTIC_ONLY_NOT_FINAL_ASSET",
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "physics_frequency_hz": int(config["runtime"]["physics_frequency_hz"]),
            "solve_articulation_contact_last": bool(config["physics"]["solve_articulation_contact_last"]),
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "root_prim": str(stage.GetDefaultPrim().GetPath()),
            "sublayers": list(stage.GetRootLayer().subLayerPaths),
            "session_only": True,
        },
        "bottle": {
            "absolute_path": str(bottle_source),
            "sha256_before": bottle_hash_before,
            "sha256_after": bottle_hash_after,
            "reference_prim": "/Bottle500",
            "session_path": bottle_path,
            "collision_local_aabb": {
                "minimum": np.min(
                    collision_points_local,
                    axis=0,
                ).tolist(),
                "maximum": np.max(
                    collision_points_local,
                    axis=0,
                ).tolist(),
            },
            "visual_local_aabb": {
                "minimum": np.min(
                    visual_points_local,
                    axis=0,
                ).tolist(),
                "maximum": np.max(
                    visual_points_local,
                    axis=0,
                ).tolist(),
            },
            "settled_collision_world_bounds": collision_bounds,
            "settled_visual_world_bounds": visual_bounds,
        },
        "filtered_pairs": filtered,
        "probe": probe,
        "evaluation": evaluation,
        "contact_event_count": len(contacts),
        "physical_pusher_contact_count": len(physical_contacts),
        "first_contact_frame": first_contact_frame,
        "capture_manifest": capture_manifest,
        "collider_display_setting": {
            "path": collider_setting,
            "before": setting_before,
            "normal": 0,
            "physics_collider_overlay": 2,
            "restored": int(settings.get(collider_setting) or 0),
        },
        "render_evidence": render_evidence,
        "command": [sys.executable, *sys.argv],
        "environment_allowlist": {
            key: os.environ.get(key) for key in ("DISPLAY", "OMNI_KIT_ACCEPT_EULA", "PYTHONPATH") if key in os.environ
        },
    }


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    artifact_root = args.artifact_root.resolve()
    started = time.perf_counter()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": int(args.width),
            "height": int(args.height),
        }
    )
    exit_code = 1
    try:
        report = _run_standard_pusher(
            app,
            config=config,
            artifact_root=artifact_root,
            resolution=(int(args.width), int(args.height)),
        )
        report["runtime_seconds"] = time.perf_counter() - started
        _write_json(args.output.resolve(), report)
        _write_json(artifact_root / "report.json", report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "root_cause": report["root_cause"],
                    "output": str(args.output.resolve()),
                    "artifact_root": str(artifact_root),
                    "physical_pusher_contact_count": report["physical_pusher_contact_count"],
                },
                sort_keys=True,
            )
        )
        exit_code = 0 if report["status"] in {"PASS", "PARTIAL"} else 2
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

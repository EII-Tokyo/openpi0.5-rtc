#!/usr/bin/env python3
"""Replay the follower-left Bottle500 trajectory with collider evidence.

The diagnostic uses a fresh Isaac Sim 5.1 process and authors only into the
opened Stage session layer. It replays the previously recorded joint targets,
records actual supplier-CAD finger/Bottle500 contact pairs, and captures normal
and physics-collider-overlay images from the same paused physics frame.
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
import traceback
from typing import Any

import numpy as np
import yaml

from tools.aloha1_mapping.bottle_collision_runtime_audit import evaluate_follower_finger_collision_probe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_follower_finger_collision_runtime_audit.yaml"
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_follower_finger_collision_runtime.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/aloha1_mapping/aloha1_follower_finger_collision_runtime.md"
)
DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260730-aloha1-bottle-collision-runtime-gate"
    / "follower_finger_runtime"
)
TASK7_CONFIG = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
STATIC_REGISTRATION_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_follower_finger_collision_registration.json"
)
EXPECTED_DOF_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return document


def _pair_text(contact: Mapping[str, Any]) -> str:
    return "\n".join(
        str(contact.get(key, ""))
        for key in (
            "actor0_path",
            "actor1_path",
            "collider0_path",
            "collider1_path",
        )
    )


def _physical_pair_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    bottle_path: str,
    finger_collider_path: str,
) -> list[dict[str, Any]]:
    return [
        {**contact, "physical": True}
        for contact in contacts
        if bottle_path in _pair_text(contact)
        and finger_collider_path in _pair_text(contact)
        and float(contact["separation_m"]) <= 0.0
        and float(contact["impulse_ns"]) >= 0.0
    ]


def _create_material(stage: Any, path: str, color: Sequence[float], opacity: float) -> Any:
    from pxr import Sdf
    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _collect_link_local_mesh(stage: Any, *, mesh_path: str, link_path: str) -> dict[str, Any]:
    from pxr import Gf
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    link = stage.GetPrimAtPath(link_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing supplier-CAD mesh: {mesh_path}")
    if not link.IsValid():
        raise RuntimeError(f"missing finger link: {link_path}")
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or counts is None or indices is None:
        raise RuntimeError(f"incomplete mesh topology: {mesh_path}")
    cache = UsdGeom.XformCache()
    relative, _ = cache.ComputeRelativeTransform(prim, link)
    local_points = np.asarray(
        [
            [float(value) for value in relative.Transform(Gf.Vec3d(*point))]
            for point in points
        ],
        dtype=np.float64,
    )
    return {
        "source_path": mesh_path,
        "points": local_points,
        "face_vertex_counts": list(counts),
        "face_vertex_indices": list(indices),
        "orientation": mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded,
    }


def _create_finger_render_evidence(
    stage: Any,
    *,
    finger_paths: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdShade

    root = "/World/Task7B2HorizontalSession/FingerCollisionRenderEvidence"
    UsdGeom.Scope.Define(stage, root)
    materials = {
        "left_visual": _create_material(
            stage,
            f"{root}/Materials/LeftVisual",
            (0.05, 0.25, 0.95),
            1.0,
        ),
        "right_visual": _create_material(
            stage,
            f"{root}/Materials/RightVisual",
            (1.0, 0.30, 0.02),
            1.0,
        ),
        "collider": _create_material(
            stage,
            f"{root}/Materials/Collider",
            (0.10, 1.0, 0.10),
            0.42,
        ),
    }
    handles: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for side in ("left", "right"):
        paths = finger_paths[side]
        side_root = f"{root}/{side}"
        visual_root = f"{side_root}/ExactVisualAtPhysxPose"
        collider_root = f"{side_root}/AuthoredColliderAtPhysxPose"
        UsdGeom.Scope.Define(stage, side_root)
        visual_scope = UsdGeom.Scope.Define(stage, visual_root)
        collider_scope = UsdGeom.Scope.Define(stage, collider_root)
        source_visual = _collect_link_local_mesh(
            stage,
            mesh_path=paths["visual"],
            link_path=paths["link"],
        )
        source_collider = _collect_link_local_mesh(
            stage,
            mesh_path=paths["collider"],
            link_path=paths["link"],
        )
        for category, source, destination_root, material in (
            (
                "visual",
                source_visual,
                visual_root,
                materials[f"{side}_visual"],
            ),
            ("collider", source_collider, collider_root, materials["collider"]),
        ):
            destination = f"{destination_root}/mesh"
            clone = UsdGeom.Mesh.Define(stage, destination)
            clone.CreatePointsAttr(source["points"].tolist())
            clone.CreateFaceVertexCountsAttr(source["face_vertex_counts"])
            clone.CreateFaceVertexIndicesAttr(source["face_vertex_indices"])
            clone.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            clone.CreateOrientationAttr(source["orientation"])
            clone.CreateDoubleSidedAttr(True)
            clone.CreateExtentAttr(
                [
                    Gf.Vec3f(*np.min(source["points"], axis=0).tolist()),
                    Gf.Vec3f(*np.max(source["points"], axis=0).tolist()),
                ]
            )
            UsdShade.MaterialBindingAPI.Apply(clone.GetPrim()).Bind(material)
            handles.append(
                {
                    "side": side,
                    "category": category,
                    "clone_path": destination,
                    "local_points": source["points"],
                }
            )
            records.append(
                {
                    "side": side,
                    "category": category,
                    "source_prim_path": source["source_path"],
                    "clone_prim_path": destination,
                    "point_count": int(source["points"].shape[0]),
                    "face_count": len(source["face_vertex_counts"]),
                    "physics_schema_applied": False,
                    "collision_schema_applied": False,
                }
            )
        UsdGeom.Imageable(visual_scope.GetPrim()).MakeVisible()
        UsdGeom.Imageable(collider_scope.GetPrim()).MakeInvisible()
        source_visual_root = stage.GetPrimAtPath(paths["visual"]).GetParent()
        if source_visual_root.IsValid():
            UsdGeom.Imageable(source_visual_root).MakeInvisible()
    return {
        "root": root,
        "records": records,
        "classification": "SESSION_ONLY_EXACT_MESH_RENDER_EVIDENCE",
        "physics_schemas_copied": False,
        "collision_schemas_copied": False,
    }, handles


def _update_finger_render_evidence(
    stage: Any,
    *,
    handles: Sequence[Mapping[str, Any]],
    link_transforms: Mapping[str, Sequence[float]],
) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    from tools.audit_aloha1_bottle_collision_runtime import _quaternion_matrix_wxyz

    for handle in handles:
        transform = np.asarray(link_transforms[str(handle["side"])], dtype=np.float64)
        position = transform[:3]
        quaternion_wxyz = [
            float(transform[6]),
            float(transform[3]),
            float(transform[4]),
            float(transform[5]),
        ]
        rotation = _quaternion_matrix_wxyz(quaternion_wxyz)
        world_points = np.asarray(handle["local_points"], dtype=np.float64) @ rotation.T + position
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(str(handle["clone_path"])))
        mesh.GetPointsAttr().Set(world_points.tolist())
        mesh.GetExtentAttr().Set(
            [
                Gf.Vec3f(*np.min(world_points, axis=0).tolist()),
                Gf.Vec3f(*np.max(world_points, axis=0).tolist()),
            ]
        )


def _filtered_pair_inventory(
    stage: Any,
    *,
    bottle_path: str,
    finger_paths: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    records = []
    pair_filtered = False
    tokens = [paths["link"] for paths in finger_paths.values()]
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        prim_path = str(prim.GetPath())
        targets = [
            str(path)
            for path in UsdPhysics.FilteredPairsAPI(prim)
            .GetFilteredPairsRel()
            .GetTargets()
        ]
        records.append({"prim_path": prim_path, "targets": targets})
        for token in tokens:
            pair_filtered = pair_filtered or (
                prim_path.startswith(bottle_path)
                and any(target.startswith(token) for target in targets)
            ) or (
                prim_path.startswith(token)
                and any(target.startswith(bottle_path) for target in targets)
            )
    groups = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPseudoRoot())
        if prim.IsA(UsdPhysics.CollisionGroup)
    ]
    return {
        "pair_is_filtered": pair_filtered,
        "filtered_pairs": records,
        "collision_group_paths": groups,
        "collision_group_pair_resolution": (
            "NO_COLLISION_GROUPS_AUTHORED"
            if not groups
            else "INCONCLUSIVE_REQUIRES_COLLECTION_MEMBERSHIP_RESOLUTION"
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    evaluation = report["evaluation"]
    lines = [
        "# ALOHA follower finger collision runtime diagnosis",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Stage: `{report['stage']['absolute_path']}`",
        f"- Stage SHA-256: `{report['stage']['sha256_before']}`",
        f"- Left physical contacts: `{evaluation['metrics']['left_physical_contact_count']}`",
        f"- Right physical contacts: `{evaluation['metrics']['right_physical_contact_count']}`",
        (
            "- Maximum bottle displacement during replay: "
            f"`{evaluation['metrics']['maximum_bottle_displacement_m']:.9f} m`"
        ),
        "",
        "## Collider screenshot semantics",
        "",
        "- Every required phase has a normal image and a collision-overlay image from the same paused physics frame.",
        "- The Isaac 5.1 setting `/persistent/physics/visualizationDisplayColliders` is read back and set to `2` for overlay captures.",
        "- Green render evidence is the exact authored CollisionAPI mesh synchronized to PhysX body poses.",
        "- Green geometry is not a cooked PhysX convex-hull readback.",
        "- Blue is `left_finger`; orange is `right_finger`.",
        "",
        "## Boundary",
        "",
        "This report validates the finger/Bottle500 collision pipeline only. It does not promote the collider,",
        "does not replace the final asset, and does not by itself prove a five-position grasp acceptance run.",
        "Task 8 remains `NOT_RUN`.",
        "",
    ]
    return "\n".join(lines)


def _run(
    app: Any,
    *,
    config: Mapping[str, Any],
    artifact_root: Path,
    resolution: tuple[int, int],
) -> dict[str, Any]:
    import carb
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from omni.physx import get_physx_interface
    from omni.physx import get_physx_simulation_interface
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    from tools.audit_aloha1_bottle_collision_runtime import _create_bottle_render_evidence
    from tools.audit_aloha1_bottle_collision_runtime import _look_at_quaternion
    from tools.audit_aloha1_bottle_collision_runtime import _update_bottle_render_evidence
    from tools.validate_aloha1_gripper_coupling_ab import author_coupling_variant
    from tools.validate_aloha1_task7b2_horizontal_grasp import _author_session_finger_drive_type
    from tools.validate_aloha1_task7b2_horizontal_grasp import _capture_viewport_png
    from tools.validate_aloha1_task7b2_horizontal_grasp import _command_positions
    from tools.validate_aloha1_task7b2_horizontal_grasp import _create_session_bottle
    from tools.validate_aloha1_task7b2_horizontal_grasp import _load_profile
    from tools.validate_aloha1_task7b2_horizontal_grasp import _serialize_contacts
    from tools.validate_aloha1_task7b2_horizontal_grasp import read_physx_bottle_state

    stage_path = Path(config["stage"]["absolute_path"]).resolve()
    bottle_path_source = Path(config["bottle"]["absolute_path"]).resolve()
    replay_path = Path(config["replay"]["source_report"]).resolve()
    stage_hash_before = _sha256(stage_path)
    bottle_hash_before = _sha256(bottle_path_source)
    replay_hash_before = _sha256(replay_path)
    if stage_hash_before != str(config["stage"]["sha256"]):
        raise RuntimeError("frozen Stage hash changed")
    if bottle_hash_before != str(config["bottle"]["sha256"]):
        raise RuntimeError("Bottle500 source hash changed")
    if replay_hash_before != str(config["replay"]["source_report_sha256"]):
        raise RuntimeError("replay report hash changed")

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    source_trial = replay["trials"][int(config["replay"]["source_trial_index"])]
    if source_trial["runtime_trial_signature"] != str(
        config["replay"]["source_runtime_signature"]
    ):
        raise RuntimeError("source runtime signature mismatch")
    telemetry_source = source_trial["telemetry"]
    if not telemetry_source:
        raise RuntimeError("source telemetry is empty")

    task_profile = _load_profile(TASK7_CONFIG)
    task_profile["diagnostic_preload_delta_m"] = 0.0
    task_profile["diagnostic_finger_drive_type"] = "force"
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_root = artifact_root / "screenshots_raw"

    World.clear_instance()
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open frozen Stage: {stage_path}")
    stage = get_current_stage()
    if str(stage.GetDefaultPrim().GetPath()) != str(config["stage"]["root_prim"]):
        raise RuntimeError("frozen Stage root prim mismatch")
    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        coupling = author_coupling_variant(
            stage=stage,
            variant="official_symmetric_adapter",
            physx_schema=__import__("pxr", fromlist=["PhysxSchema"]).PhysxSchema,
            usd_physics=UsdPhysics,
        )
        drive_readback = _author_session_finger_drive_type(
            stage=stage,
            usd_physics=UsdPhysics,
            requested_type="force",
        )
        bottle_prim, bottle_session, _ = _create_session_bottle(stage, task_profile)
        bottle_render, bottle_handles = _create_bottle_render_evidence(
            stage,
            bottle_path=str(bottle_prim.GetPath()),
        )
        UsdGeom.Imageable(stage.GetPrimAtPath(bottle_render["pusher_visual_prim"])).MakeInvisible()
        finger_render, finger_handles = _create_finger_render_evidence(
            stage,
            finger_paths=config["finger_colliders"],
        )

    dt = 1.0 / float(config["runtime"]["physics_frequency_hz"])
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulation = SingleArticulation(
        prim_path="/World/follower_left/vx300s_left/root_joint",
        name="follower_left_collision_replay",
        reset_xform_properties=False,
    )
    world.scene.add(articulation)
    world.reset()
    if list(articulation.dof_names) != EXPECTED_DOF_ORDER:
        raise RuntimeError(f"unexpected DOF order: {list(articulation.dof_names)}")
    initial_target = np.asarray(telemetry_source[0]["joint_target"], dtype=np.float64)
    articulation.set_joint_positions(initial_target)
    articulation.set_joint_velocities(np.zeros_like(initial_target))
    _command_positions(articulation, initial_target)
    world.step(render=False)

    simulation_view = SimulationManager.get_physics_sim_view()
    if simulation_view is None or not simulation_view.is_valid:
        raise RuntimeError("PhysX tensor SimulationView unavailable")
    bottle_path = str(bottle_prim.GetPath())
    bottle_view = simulation_view.create_rigid_body_view(bottle_path)
    link_views = {
        side: simulation_view.create_rigid_body_view(paths["link"])
        for side, paths in config["finger_colliders"].items()
    }
    if bottle_view is None or int(bottle_view.count) != 1:
        raise RuntimeError("Bottle500 rigid-body view unavailable")
    if any(view is None or int(view.count) != 1 for view in link_views.values()):
        raise RuntimeError("finger rigid-body view unavailable")

    physx = get_physx_interface()
    physx_sim = get_physx_simulation_interface()
    state = {"frame": -1, "phase": "setup_kinematic"}
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
    rigid = UsdPhysics.RigidBodyAPI(bottle_prim)
    source_contact = source_trial["metrics"]["contact_geometry"]
    target = (
        np.asarray(source_contact["left_impulse_weighted_center_world_m"], dtype=np.float64)
        + np.asarray(source_contact["right_impulse_weighted_center_world_m"], dtype=np.float64)
    ) / 2.0
    capture_target = target + np.asarray([0.0, 0.0, 0.055])
    camera_specs = {
        "left_contact_oblique": {
            "position": capture_target + np.asarray([0.42, 0.44, 0.28]),
            "target": capture_target,
            "up": np.asarray([0.0, 0.0, 1.0]),
        },
        "right_contact_oblique": {
            "position": capture_target + np.asarray([0.42, -0.44, 0.28]),
            "target": capture_target,
            "up": np.asarray([0.0, 0.0, 1.0]),
        },
    }
    first_camera = camera_specs["left_contact_oblique"]
    camera = Camera(
        prim_path="/World/Task7B2HorizontalSession/FingerCollisionCaptureCamera",
        position=first_camera["position"],
        orientation=_look_at_quaternion(
            first_camera["position"],
            first_camera["target"],
            up_world=first_camera["up"],
        ),
        frequency=float(config["runtime"]["physics_frequency_hz"]),
        resolution=resolution,
    )
    camera.initialize(attach_rgb_annotator=False)
    camera.set_clipping_range(0.005, 5.0)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport unavailable")
    settings = carb.settings.get_settings()
    collider_setting = str(config["capture"]["display_colliders_setting"])
    setting_before = int(settings.get(collider_setting) or 0)
    capture_manifest: list[dict[str, Any]] = []
    captured_phases: set[str] = set()
    bottle_trace: list[dict[str, Any]] = []
    joint_trace: list[dict[str, Any]] = []

    def link_transforms() -> dict[str, list[float]]:
        return {
            side: np.asarray(view.get_transforms()[0], dtype=np.float64).tolist()
            for side, view in link_views.items()
        }

    def set_render_mode(mode: str) -> None:
        overlay = mode == "physics_collider_overlay"
        UsdGeom.Imageable(stage.GetPrimAtPath(bottle_render["visual_root"])).MakeVisible()
        if overlay:
            UsdGeom.Imageable(stage.GetPrimAtPath(bottle_render["collider_root"])).MakeVisible()
        else:
            UsdGeom.Imageable(stage.GetPrimAtPath(bottle_render["collider_root"])).MakeInvisible()
        for side in ("left", "right"):
            visual_root = f"{finger_render['root']}/{side}/ExactVisualAtPhysxPose"
            collider_root = f"{finger_render['root']}/{side}/AuthoredColliderAtPhysxPose"
            UsdGeom.Imageable(stage.GetPrimAtPath(visual_root)).MakeVisible()
            if overlay:
                UsdGeom.Imageable(stage.GetPrimAtPath(collider_root)).MakeVisible()
            else:
                UsdGeom.Imageable(stage.GetPrimAtPath(collider_root)).MakeInvisible()
        settings.set_int(
            collider_setting,
            int(config["capture"]["display_colliders_values"][mode]),
        )

    def capture_pair(phase: str) -> None:
        if phase in captured_phases:
            return
        captured_phases.add(phase)
        world.pause()
        frame = int(state["frame"])
        bottle_before = read_physx_bottle_state(bottle_view)
        links_before = link_transforms()
        contact_evidence: dict[str, dict[str, Any] | None] = {}
        for side in ("left", "right"):
            physical = _physical_pair_contacts(
                contacts,
                bottle_path=bottle_path,
                finger_collider_path=config["finger_colliders"][side]["collider"],
            )
            latest_frame = max(
                (int(contact["frame"]) for contact in physical),
                default=None,
            )
            latest = [
                contact
                for contact in physical
                if latest_frame is not None
                and int(contact["frame"]) == latest_frame
            ]
            contact_evidence[side] = (
                max(latest, key=lambda contact: float(contact["impulse_ns"]))
                if latest
                else None
            )
        _update_bottle_render_evidence(
            stage,
            handles=bottle_handles,
            position_world=bottle_before["position_world_m"],
            orientation_world_wxyz=bottle_before["orientation_wxyz"],
        )
        _update_finger_render_evidence(
            stage,
            handles=finger_handles,
            link_transforms=links_before,
        )
        for view_name, spec in camera_specs.items():
            quaternion = _look_at_quaternion(
                spec["position"],
                spec["target"],
                up_world=spec["up"],
            )
            camera.set_world_pose(
                position=spec["position"],
                orientation=quaternion,
                camera_axes="usd",
            )
            actual_position, actual_orientation = camera.get_world_pose(
                camera_axes="usd"
            )
            camera_intrinsics = np.asarray(
                camera.get_intrinsics_matrix(),
                dtype=np.float64,
            )
            paths = {}
            for mode in ("normal", "physics_collider_overlay"):
                set_render_mode(mode)
                destination = raw_root / phase / f"{view_name}_{mode}_raw.png"
                width, height = _capture_viewport_png(
                    app,
                    viewport,
                    camera_path=camera.prim_path,
                    destination=destination,
                )
                paths[mode] = str(destination.resolve())
                capture_manifest.append(
                    {
                        "phase": phase,
                        "view": view_name,
                        "mode": mode,
                        "physics_frame": frame,
                        "time_s": frame * dt,
                        "absolute_path": str(destination.resolve()),
                        "sha256": _sha256(destination),
                        "resolution": [width, height],
                        "camera_position_world_m": [
                            float(value) for value in actual_position
                        ],
                        "camera_orientation_wxyz": [
                            float(value) for value in actual_orientation
                        ],
                        "camera_intrinsics_pixels": camera_intrinsics.tolist(),
                        "camera_target_world_m": [
                            float(value) for value in spec["target"]
                        ],
                        "camera_clipping_range_m": [0.005, 5.0],
                        "display_colliders_readback": int(
                            settings.get(collider_setting) or 0
                        ),
                        "bottle_position_world_m": bottle_before[
                            "position_world_m"
                        ],
                        "left_finger_physx_transform_xyzw": links_before["left"],
                        "right_finger_physx_transform_xyzw": links_before["right"],
                        "contact_evidence": contact_evidence,
                        "overlay_semantics": config["capture"]["overlay_semantics"],
                    }
                )
            for record in capture_manifest:
                if record["phase"] == phase and record["view"] == view_name:
                    record["normal_path"] = paths["normal"]
                    record["overlay_path"] = paths[
                        "physics_collider_overlay"
                    ]
        set_render_mode("normal")
        bottle_after = read_physx_bottle_state(bottle_view)
        links_after = link_transforms()
        same_physics = bool(
            np.allclose(
                bottle_before["position_world_m"],
                bottle_after["position_world_m"],
                rtol=0.0,
                atol=1.0e-12,
            )
            and all(
                np.allclose(
                    links_before[side],
                    links_after[side],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for side in ("left", "right")
            )
            and int(state["frame"]) == frame
        )
        for record in capture_manifest:
            if record["phase"] == phase:
                record["same_camera_pose"] = True
                record["same_physics_frame"] = same_physics
        world.play()

    phase_end = source_trial["runtime"]["phase_end_frames"]
    bottle_dynamic = False
    bilateral_captured = False
    for source_record in telemetry_source:
        frame = int(source_record["frame"])
        phase = str(source_record["phase"])
        target_command = np.asarray(source_record["joint_target"], dtype=np.float64)
        if phase == "release_dynamic" and not bottle_dynamic:
            rigid.GetKinematicEnabledAttr().Set(False)
            physx_sim.flush_changes()
            bottle_dynamic = True
        state["frame"] = frame
        state["phase"] = phase
        _command_positions(articulation, target_command)
        world.play()
        world.step(render=False)
        physx.update_transformations(True, True, False, False)
        bottle_state = read_physx_bottle_state(bottle_view)
        bottle_trace.append(
            {
                "frame": frame,
                "phase": phase,
                "position_world_m": bottle_state["position_world_m"],
                "linear_velocity_world_m_s": bottle_state[
                    "linear_velocity_world_m_s"
                ],
                "angular_velocity_world_rad_s": bottle_state[
                    "angular_velocity_world_rad_s"
                ],
            }
        )
        joint_trace.append(
            {
                "frame": frame,
                "phase": phase,
                "target": target_command.tolist(),
                "readback": np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                ).tolist(),
            }
        )
        left_now = _physical_pair_contacts(
            contacts,
            bottle_path=bottle_path,
            finger_collider_path=config["finger_colliders"]["left"]["collider"],
        )
        right_now = _physical_pair_contacts(
            contacts,
            bottle_path=bottle_path,
            finger_collider_path=config["finger_colliders"]["right"]["collider"],
        )
        if frame == int(phase_end["open_pregrasp"]):
            capture_pair("open_pregrasp")
        if left_now and right_now and not bilateral_captured:
            capture_pair("bilateral_contact")
            bilateral_captured = True
        if frame == int(phase_end["bilateral_contact"]):
            capture_pair("maximum_closure")
        if frame == int(phase_end["hold_end"]):
            capture_pair("hold_end")

    settings.set_int(collider_setting, setting_before)
    left_contacts = _physical_pair_contacts(
        contacts,
        bottle_path=bottle_path,
        finger_collider_path=config["finger_colliders"]["left"]["collider"],
    )
    right_contacts = _physical_pair_contacts(
        contacts,
        bottle_path=bottle_path,
        finger_collider_path=config["finger_colliders"]["right"]["collider"],
    )
    static_registration = json.loads(
        STATIC_REGISTRATION_REPORT.read_text(encoding="utf-8")
    )
    filtering = _filtered_pair_inventory(
        stage,
        bottle_path=bottle_path,
        finger_paths=config["finger_colliders"],
    )
    positions = np.asarray(
        [record["position_world_m"] for record in bottle_trace],
        dtype=np.float64,
    )
    contact_start_frame = min(
        [int(record["frame"]) for record in left_contacts + right_contacts],
        default=0,
    )
    before_candidates = [
        record
        for record in bottle_trace
        if int(record["frame"]) < contact_start_frame
    ]
    reference_position = np.asarray(
        (before_candidates[-1] if before_candidates else bottle_trace[0])[
            "position_world_m"
        ],
        dtype=np.float64,
    )
    maximum_displacement = float(
        np.max(np.linalg.norm(positions - reference_position, axis=1))
    )
    representative = {
        "left": left_contacts[:8],
        "right": right_contacts[:8],
    }
    paired_capture_records = [
        record
        for record in capture_manifest
        if record["view"] == "left_contact_oblique" and record["mode"] == "normal"
    ]
    probe = {
        "frozen_inputs_verified": True,
        "finger_colliders": {
            side: {
                "enabled": bool(
                    static_registration["fingers"][side]["collider"][
                        "collision_enabled"
                    ]
                ),
                "approximation": static_registration["fingers"][side][
                    "collider"
                ]["approximation"],
                "maximum_registration_gap_m": static_registration["fingers"][
                    side
                ]["registration"]["maximum_surface_gap_m"],
            }
            for side in ("left", "right")
        },
        "filtered_pair_with_bottle": bool(filtering["pair_is_filtered"])
        or bool(filtering["collision_group_paths"]),
        "contacts": representative,
        "bottle_response": {
            "maximum_displacement_m": maximum_displacement,
            "minimum_required_displacement_m": float(
                config["limits"]["minimum_bottle_response_m"]
            ),
        },
        "captures": {
            "required_phases": list(config["capture"]["phases"]),
            "paired_records": paired_capture_records,
        },
        "forbidden_helpers_absent": True,
        "maximum_registration_gap_m": float(
            config["limits"]["maximum_registration_gap_m"]
        ),
    }
    evaluation = evaluate_follower_finger_collision_probe(probe)
    del subscription
    stage_hash_after = _sha256(stage_path)
    bottle_hash_after = _sha256(bottle_path_source)
    replay_hash_after = _sha256(replay_path)
    if stage_hash_after != stage_hash_before:
        raise RuntimeError("frozen Stage changed during session-only replay")
    if bottle_hash_after != bottle_hash_before:
        raise RuntimeError("Bottle500 source changed during session-only replay")
    if replay_hash_after != replay_hash_before:
        raise RuntimeError("replay report changed during session-only replay")
    return {
        "schema_version": 1,
        "status": evaluation["status"],
        "classification": evaluation["classification"],
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "physics_frequency_hz": int(
                config["runtime"]["physics_frequency_hz"]
            ),
            "solve_articulation_contact_last": True,
            "diagnostic_coupling": coupling,
            "diagnostic_finger_drive_type": drive_readback,
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
            **bottle_session,
            "sha256_before": bottle_hash_before,
            "sha256_after": bottle_hash_after,
        },
        "replay": {
            "absolute_path": str(replay_path),
            "sha256_before": replay_hash_before,
            "sha256_after": replay_hash_after,
            "source_runtime_signature": source_trial[
                "runtime_trial_signature"
            ],
            "frame_count": len(telemetry_source),
        },
        "static_registration_report": {
            "absolute_path": str(STATIC_REGISTRATION_REPORT.resolve()),
            "sha256": _sha256(STATIC_REGISTRATION_REPORT),
            "status": static_registration["status"],
        },
        "filtering": filtering,
        "contacts": {
            "left_physical_count": len(left_contacts),
            "right_physical_count": len(right_contacts),
            "left_first_frame": min(
                (int(record["frame"]) for record in left_contacts),
                default=None,
            ),
            "right_first_frame": min(
                (int(record["frame"]) for record in right_contacts),
                default=None,
            ),
            "maximum_penetration_m": min(
                (
                    float(record["separation_m"])
                    for record in left_contacts + right_contacts
                ),
                default=None,
            ),
            "representative": representative,
        },
        "capture_manifest": capture_manifest,
        "collider_display_setting": {
            "path": collider_setting,
            "before": setting_before,
            "normal": int(config["capture"]["display_colliders_values"]["normal"]),
            "physics_collider_overlay": int(
                config["capture"]["display_colliders_values"][
                    "physics_collider_overlay"
                ]
            ),
            "restored": int(settings.get(collider_setting) or 0),
        },
        "render_evidence": {
            "bottle": bottle_render,
            "fingers": finger_render,
            "overlay_semantics": config["capture"]["overlay_semantics"],
        },
        "bottle_trace": bottle_trace,
        "joint_trace": joint_trace,
        "probe": probe,
        "evaluation": evaluation,
        "boundaries": {
            "source_stage_modified": False,
            "bottle_source_modified": False,
            "replay_source_modified": False,
            "default_final_collider_modified": False,
            "grasp_acceptance": "NOT_EVALUATED_BY_COLLISION_REPLAY",
            "task8": "NOT_RUN",
        },
    }


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config.resolve())
    artifact_root = args.artifact_root.resolve()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": int(args.width),
            "height": int(args.height),
        }
    )
    exit_code = 1
    started = time.perf_counter()
    try:
        report = _run(
            app,
            config=config,
            artifact_root=artifact_root,
            resolution=(int(args.width), int(args.height)),
        )
        report["runtime_seconds"] = time.perf_counter() - started
        report["command"] = [sys.executable, *sys.argv]
        report["environment_allowlist"] = {
            key: os.environ.get(key)
            for key in ("DISPLAY", "OMNI_KIT_ACCEPT_EULA", "PYTHONPATH")
            if key in os.environ
        }
        _write_json(args.output.resolve(), report)
        _write_json(artifact_root / "report.json", report)
        _write_text(args.markdown.resolve(), _render_markdown(report))
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "classification": report["classification"],
                    "output": str(args.output.resolve()),
                    "artifact_report": str((artifact_root / "report.json").resolve()),
                    "screenshot_count": len(report["capture_manifest"]),
                },
                sort_keys=True,
            )
        )
        exit_code = 0 if report["status"] == "PASS" else 2
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the supplier-CAD follower-left gripper without a bottle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_finger_task5_structure import FINGER_DOF_NAMES
from tools.aloha1_mapping.cad_finger_task5_structure import GLOBAL_SESSION_HIDDEN_VISUALS
from tools.aloha1_mapping.cad_finger_task5_structure import LEGAL_POSES_M
from tools.aloha1_mapping.cad_finger_task5_structure import POSE_ALIASES
from tools.aloha1_mapping.cad_finger_task5_structure import VIEW_HIDDEN_VISUALS
from tools.aloha1_mapping.cad_finger_task5_structure import VIEW_RADII_M
from tools.aloha1_mapping.cad_finger_task5_structure import drive_mimic_status
from tools.aloha1_mapping.cad_finger_task5_structure import hide_non_target_robot_gprim
from tools.aloha1_mapping.cad_finger_task5_structure import hide_non_target_robot_visual
from tools.aloha1_mapping.cad_finger_task5_structure import hide_robot_debug_container
from tools.aloha1_mapping.cad_finger_task5_structure import summarize_image_projection
from tools.aloha1_mapping.cad_finger_task5_structure import validate_pose_records
from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png
from tools.aloha1_mapping.screenshot_manifest import build_screenshot_manifest
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot

ROOT = Path(__file__).resolve().parents[1]
ASSET_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha_viper_cad_finger_task5_asset.json"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha_viper_cad_finger_task5_structure.json"
)
OUTPUT_MARKDOWN = OUTPUT_REPORT.with_suffix(".md")
SCREENSHOT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/task5_structure/screenshots_raw"
)
ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
FINGER_LINKS = {
    side: (
        f"/workcell/vx300s_left/vx300s_left_{side}_finger_link"
    )
    for side in ("left", "right")
}
FINGER_MESHES = {
    side: (
        f"{FINGER_LINKS[side]}/visuals/"
        f"diagnostic_supplier_cad_{side}_finger/mesh"
    )
    for side in ("left", "right")
}
VIEWS = ("true_top", "true_bottom", "tip_end", "base_oblique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _world_points(stage: Any, path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"required supplier-CAD mesh missing: {path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [
            list(transform.Transform(point))
            for point in (UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])
        ],
        dtype=np.float64,
    )


def _geometry_state(stage: Any) -> dict[str, Any]:
    from pxr import Gf
    from pxr import UsdGeom

    clouds = {
        side: _world_points(stage, path)
        for side, path in FINGER_MESHES.items()
    }
    centers = {side: cloud.mean(axis=0) for side, cloud in clouds.items()}
    closing = centers["right"] - centers["left"]
    closing /= np.linalg.norm(closing)
    ordered = sorted(
        clouds,
        key=lambda side: float(centers[side] @ closing),
    )
    low = clouds[ordered[0]] @ closing
    high = clouds[ordered[1]] @ closing
    combined = np.concatenate(list(clouds.values()))
    link = stage.GetPrimAtPath(FINGER_LINKS["left"])
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(link)
    tool = np.asarray(
        list(transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))),
        dtype=np.float64,
    )
    tool /= np.linalg.norm(tool)
    return {
        "centers_world_m": {
            side: centers[side].tolist() for side in ("left", "right")
        },
        "closing_axis_world": closing.tolist(),
        "tool_axis_world": tool.tolist(),
        "surface_gap_m": float(high.min() - low.max()),
        "combined_aabb_min_m": combined.min(axis=0).tolist(),
        "combined_aabb_max_m": combined.max(axis=0).tolist(),
        "combined_center_world_m": (
            (combined.min(axis=0) + combined.max(axis=0)) / 2.0
        ).tolist(),
    }


def _camera_poses(
    geometries: Sequence[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    lower = np.min(
        [item["combined_aabb_min_m"] for item in geometries],
        axis=0,
    )
    upper = np.max(
        [item["combined_aabb_max_m"] for item in geometries],
        axis=0,
    )
    target = (lower + upper) / 2.0
    closing = np.asarray(
        geometries[0]["closing_axis_world"],
        dtype=np.float64,
    )
    tool = np.asarray(
        geometries[0]["tool_axis_world"],
        dtype=np.float64,
    )
    top = np.cross(closing, tool)
    top /= np.linalg.norm(top)
    if float(top @ np.asarray([0.0, 0.0, 1.0])) < 0.0:
        top = -top
    positions = {
        "true_top": target + VIEW_RADII_M["true_top"] * top,
        "true_bottom": target - VIEW_RADII_M["true_bottom"] * top,
        "tip_end": target + VIEW_RADII_M["tip_end"] * tool + 0.05 * top,
        "base_oblique": (
            target
            - VIEW_RADII_M["base_oblique_tool"] * tool
            + VIEW_RADII_M["base_oblique_top"] * top
            + VIEW_RADII_M["base_oblique_closing"] * closing
        ),
    }
    up_by_view = {
        "true_top": tool,
        "true_bottom": tool,
        "tip_end": top,
        "base_oblique": top,
    }
    return {
        view: {
            "position_world_m": positions[view].tolist(),
            "orientation_wxyz": look_at_orientation_wxyz(
                positions[view],
                target,
                up_world=up_by_view[view],
            ).tolist(),
            "target_world_m": target.tolist(),
            "up_world": up_by_view[view].tolist(),
        }
        for view in VIEWS
    }


def _recenter_camera_pose_from_projection(
    world: Any,
    camera: Any,
    pose: dict[str, Any],
    points_world: np.ndarray,
    *,
    width: int,
    height: int,
    maximum_iterations: int = 3,
) -> dict[str, Any]:
    """Aim one shared camera pose at the projected CAD-state envelope."""

    position = np.asarray(pose["position_world_m"], dtype=np.float64)
    target = np.asarray(pose["target_world_m"], dtype=np.float64)
    up_world = np.asarray(pose["up_world"], dtype=np.float64)
    history = []
    for iteration in range(maximum_iterations):
        orientation = look_at_orientation_wxyz(
            position,
            target,
            up_world=up_world,
        )
        camera.set_world_pose(
            position=position,
            orientation=orientation,
            camera_axes="usd",
        )
        for _ in range(3):
            world.render()
        summary = summarize_image_projection(
            camera.get_image_coords_from_world_points(
                points_world
            ).tolist(),
            width=width,
            height=height,
        )
        history.append(
            {
                "iteration": iteration,
                "target_world_m": target.tolist(),
                "projection": summary,
            }
        )
        center = np.asarray(summary["bbox_center_px"], dtype=np.float64)
        image_center = np.asarray(
            [width / 2.0, height / 2.0],
            dtype=np.float64,
        )
        if float(np.linalg.norm(center - image_center)) <= 1.0:
            break
        depth = float(np.linalg.norm(target - position))
        target = np.asarray(
            camera.get_world_points_from_image_coords(
                center.reshape(1, 2),
                np.asarray([depth], dtype=np.float64),
            )[0],
            dtype=np.float64,
        )
    final_orientation = look_at_orientation_wxyz(
        position,
        target,
        up_world=up_world,
    )
    return {
        **pose,
        "nominal_target_world_m": pose["target_world_m"],
        "target_world_m": target.tolist(),
        "orientation_wxyz": final_orientation.tolist(),
        "recenter_method": (
            "UNION_OF_THREE_LEGAL_CAD_FINGER_STATES_"
            "ISAAC51_PINHOLE_PROJECTION"
        ),
        "recenter_history": history,
    }


def _read_nonblank_rgba(
    world: Any,
    camera: Any,
    *,
    maximum_render_updates: int = 60,
) -> np.ndarray:
    """Wait for a non-constant Isaac camera frame without physics stepping."""

    last = None
    for _ in range(maximum_render_updates):
        world.render()
        candidate = camera.get_rgba()
        if candidate is None:
            continue
        pixels = np.asarray(candidate)
        last = pixels
        if pixels.size == 0:
            continue
        display_pixels = pixels
        if np.issubdtype(display_pixels.dtype, np.floating):
            scale = (
                255.0
                if float(np.nanmax(display_pixels)) <= 1.0
                else 1.0
            )
            display_pixels = np.clip(
                display_pixels * scale,
                0.0,
                255.0,
            )
        if (
            pixels.ndim == 3
            and pixels.shape[2] == 4
            and pixels.size
            and max(
                float(np.ptp(display_pixels[..., channel]))
                for channel in range(3)
            )
            > 2.0
        ):
            return np.array(pixels, copy=True)
    shape = None if last is None else list(last.shape)
    spread = (
        None
        if last is None or last.size == 0
        else float(np.ptp(last[..., :3]))
    )
    raise RuntimeError(
        "Isaac camera remained blank after render polling: "
        f"shape={shape}, rgb_spread={spread}"
    )


def _sync_physx_transforms_to_usd(physx_interface: Any) -> None:
    """Read PhysX body transforms back into the unsaved runtime Stage."""

    physx_interface.update_transformations(
        True, True, False, False  # noqa: FBT003
    )


def _hide_non_target_visuals(stage: Any) -> list[str]:
    from pxr import UsdGeom

    hidden = []
    for path in (
        "/workcell/table",
        "/workcell/table_frame_T",
        "/workcell/midair",
        "/workcell/placeholder_pipe",
        "/workcell/worldBody",
        *GLOBAL_SESSION_HIDDEN_VISUALS,
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(path)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            (
                hide_non_target_robot_visual(path, prim.GetName())
                or hide_robot_debug_container(path, prim.GetName())
            )
            and prim.IsA(UsdGeom.Imageable)
        ) or (
            prim.IsA(UsdGeom.Gprim)
            and hide_non_target_robot_gprim(path)
        ):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(path)
    return hidden


def _set_view_visibility(stage: Any, view: str) -> list[str]:
    """Expose the finger evidence in the unsaved session layer only."""

    from pxr import UsdGeom

    requested = VIEW_HIDDEN_VISUALS[view]
    all_paths = {
        path
        for paths in VIEW_HIDDEN_VISUALS.values()
        for path in paths
    }
    hidden = []
    for path in sorted(all_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            raise RuntimeError(
                f"required evidence-view visual path missing: {path}"
            )
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        if path in requested:
            imageable.MakeInvisible()
            hidden.append(path)
    return hidden


def _serialize_contacts(
    headers: Sequence[Any],
    data: Sequence[Any],
    *,
    frame: int,
) -> list[dict[str, Any]]:
    from pxr import PhysicsSchemaTools

    def path(value: Any) -> str:
        return str(PhysicsSchemaTools.intToSdfPath(value))

    records = []
    for header in headers:
        contacts = []
        begin = int(header.contact_data_offset)
        end = begin + int(header.num_contact_data)
        for index in range(begin, end):
            contact = data[index]
            contacts.append(
                {
                    "position_world_m": [
                        float(value) for value in contact.position
                    ],
                    "normal": [float(value) for value in contact.normal],
                    "impulse_n_s": [
                        float(value) for value in contact.impulse
                    ],
                    "separation_m": float(contact.separation),
                    "material0": path(contact.material0),
                    "material1": path(contact.material1),
                }
            )
        records.append(
            {
                "frame": frame,
                "type": str(header.type),
                "actor0": path(header.actor0),
                "actor1": path(header.actor1),
                "collider0": path(header.collider0),
                "collider1": path(header.collider1),
                "contacts": contacts,
            }
        )
    return records


def _drive_snapshot(stage: Any) -> dict[str, Any]:
    snapshot = {}
    for side in ("left", "right"):
        path = f"/workcell/joints/vx300s_left_{side}_finger"
        prim = stage.GetPrimAtPath(path)
        schemas = list(prim.GetAppliedSchemas())
        values = {}
        for name in (
            "drive:linear:physics:targetPosition",
            "drive:linear:physics:stiffness",
            "drive:linear:physics:damping",
            "drive:linear:physics:maxForce",
            "drive:linear:physics:type",
        ):
            attribute = prim.GetAttribute(name)
            values[name] = attribute.Get() if attribute.IsValid() else None
        snapshot[side] = {
            "path": path,
            "applied_schemas": schemas,
            "physx_mimic_api_present": any(
                "MimicJointAPI" in schema for schema in schemas
            ),
            "drive": values,
        }
    return snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-report", type=Path, default=ASSET_REPORT)
    parser.add_argument("--report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--markdown", type=Path, default=OUTPUT_MARKDOWN)
    parser.add_argument(
        "--screenshot-root",
        type=Path,
        default=SCREENSHOT_ROOT,
    )
    parser.add_argument("--settle-steps", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    asset_report_path = args.asset_report.resolve(strict=True)
    asset_report = json.loads(asset_report_path.read_text(encoding="utf-8"))
    if asset_report["status"] != "PARTIAL":
        raise RuntimeError("expected v2 asset report status PARTIAL")
    root_asset = Path(
        asset_report["outputs"]["root_usd"]["absolute_path"]
    ).resolve(strict=True)
    protected = {
        name: {
            "path": Path(record["absolute_path"]).resolve(strict=True),
            "expected_sha256": record["sha256"],
        }
        for name, record in asset_report["outputs"].items()
    }
    protected["approved_source_stage"] = {
        "path": Path(
            asset_report["source_stage"]["absolute_path"]
        ).resolve(strict=True),
        "expected_sha256": asset_report["source_stage"]["sha256_after"],
    }
    hashes_before = {
        name: _sha256(record["path"])
        for name, record in protected.items()
    }
    if any(
        hashes_before[name] != record["expected_sha256"]
        for name, record in protected.items()
    ):
        raise RuntimeError("protected source/diagnostic hash mismatch")
    screenshot_root = args.screenshot_root.resolve()
    if screenshot_root.exists():
        raise FileExistsError(
            f"screenshot root already exists: {screenshot_root}"
        )
    screenshot_root.mkdir(parents=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "width": 1280, "height": 900}
    )
    report: dict[str, Any] = {}
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from omni.physx import get_physx_interface
        from omni.physx import get_physx_simulation_interface
        from pxr import Gf
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdLux

        if not open_stage(str(root_asset)):
            raise RuntimeError(f"failed to open {root_asset}")
        stage = get_current_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            hidden = _hide_non_target_visuals(stage)
            for link_path in FINGER_LINKS.values():
                api = PhysxSchema.PhysxContactReportAPI.Apply(
                    stage.GetPrimAtPath(link_path)
                )
                api.CreateThresholdAttr().Set(0.0)
            dome = UsdLux.DomeLight.Define(
                stage, "/workcell/Task5StructureSession/Dome"
            )
            dome.CreateIntensityAttr(700.0)
            dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
            key = UsdLux.DistantLight.Define(
                stage, "/workcell/Task5StructureSession/Key"
            )
            key.CreateIntensityAttr(1100.0)
            key.CreateAngleAttr(1.0)

        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        physics_context = world.get_physics_context()
        physics_context.set_solve_articulation_contact_last(True)
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="supplier_cad_task5_structure",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        camera = Camera(
            prim_path="/workcell/Task5StructureSession/Camera",
            name="supplier_cad_task5_structure_camera",
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)
        frame = {"value": 0}
        current_events: list[dict[str, Any]] = []

        def on_contact(
            headers: Sequence[Any],
            data: Sequence[Any],
        ) -> None:
            current_events.extend(
                _serialize_contacts(
                    headers,
                    data,
                    frame=frame["value"],
                )
            )

        physx_simulation = get_physx_simulation_interface()
        physx_interface = get_physx_interface()
        subscription = (
            physx_simulation.subscribe_contact_report_events(on_contact)
        )
        world.reset()
        camera.initialize()
        camera.set_clipping_range(0.01, 10.0)
        order = list(articulation.dof_names)
        if any(order.count(name) != 1 for name in FINGER_DOF_NAMES):
            raise RuntimeError(f"finger DOF identity mismatch: {order}")
        left_index = order.index(FINGER_DOF_NAMES[0])
        right_index = order.index(FINGER_DOF_NAMES[1])
        limits = np.asarray(articulation.dof_properties["lower"], dtype=float)
        upper = np.asarray(articulation.dof_properties["upper"], dtype=float)
        limits_by_side = {
            "left": [float(limits[left_index]), float(upper[left_index])],
            "right": [
                float(limits[right_index]),
                float(upper[right_index]),
            ],
        }
        home_by_name = {
            "vx300s_left_waist": 0.0,
            "vx300s_left_shoulder": -0.96,
            "vx300s_left_elbow": 1.16,
            "vx300s_left_forearm_roll": 0.0,
            "vx300s_left_wrist_angle": -0.3,
            "vx300s_left_wrist_rotate": 0.0,
        }
        base = np.asarray(
            [home_by_name.get(name, 0.0) for name in order],
            dtype=np.float32,
        )
        prepass = []
        prepass_points = []
        for targets in LEGAL_POSES_M.values():
            world.reset()
            qpos = base.copy()
            qpos[left_index], qpos[right_index] = targets
            articulation.set_joint_positions(qpos)
            _sync_physx_transforms_to_usd(physx_interface)
            prepass.append(_geometry_state(stage))
            prepass_points.append(
                np.concatenate(
                    [
                        _world_points(stage, FINGER_MESHES[side])
                        for side in ("left", "right")
                    ]
                )
            )
        camera_poses = _camera_poses(prepass)
        camera_poses["base_oblique"] = (
            _recenter_camera_pose_from_projection(
                world,
                camera,
                camera_poses["base_oblique"],
                np.concatenate(prepass_points),
                width=1280,
                height=900,
            )
        )
        print(
            "TASK5_STRUCTURE_CAMERAS "
            + json.dumps(camera_poses, sort_keys=True),
            flush=True,
        )
        states = []
        captures = []
        for state, targets in LEGAL_POSES_M.items():
            current_events.clear()
            frame["value"] = 0
            world.reset()
            camera.initialize()
            camera.set_clipping_range(0.01, 10.0)
            qpos = base.copy()
            qpos[left_index], qpos[right_index] = targets
            articulation.set_joint_positions(qpos)
            _sync_physx_transforms_to_usd(physx_interface)
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=qpos)
            )
            injected = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            geometry_injected = _geometry_state(stage)
            trace = []
            for _ in range(args.settle_steps):
                frame["value"] += 1
                world.step(render=False)
                current = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )
                trace.append(
                    {
                        "frame": frame["value"],
                        "left_m": float(current[left_index]),
                        "right_m": float(current[right_index]),
                    }
                )
            final = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            geometry_final = _geometry_state(stage)
            print(
                "TASK5_STRUCTURE_STATE "
                + json.dumps(
                    {
                        "state": state,
                        "target_m": list(targets),
                        "injected_center_world_m": (
                            geometry_injected["combined_center_world_m"]
                        ),
                        "post_step_center_world_m": (
                            geometry_final["combined_center_world_m"]
                        ),
                        "post_step_aabb_min_m": (
                            geometry_final["combined_aabb_min_m"]
                        ),
                        "post_step_aabb_max_m": (
                            geometry_final["combined_aabb_max_m"]
                        ),
                        "camera_target_world_m": (
                            camera_poses["true_top"]["target_world_m"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            state_record = {
                "state": state,
                "semantic_aliases": [
                    alias
                    for alias, target_state in POSE_ALIASES.items()
                    if target_state == state
                ],
                "target_m": list(targets),
                "readback_m": [
                    float(injected[left_index]),
                    float(injected[right_index]),
                ],
                "limits_m": limits_by_side,
                "surface_gap_m": geometry_injected["surface_gap_m"],
                "geometry_at_injection": geometry_injected,
                "post_step_readback_m": [
                    float(final[left_index]),
                    float(final[right_index]),
                ],
                "post_step_geometry": geometry_final,
                "post_step_trace": trace,
                "contact_events": list(current_events),
                "maximum_post_step_target_error_m": float(
                    max(
                        abs(final[left_index] - targets[0]),
                        abs(final[right_index] - targets[1]),
                    )
                ),
            }
            states.append(state_record)
            # Use a fresh World reset for the structure screenshot. Reinjecting
            # only qpos after the failed dynamic trace does not restore body
            # transforms displaced by the source Stage's disjoint joint frames.
            # The failed dynamic trace remains separately recorded above.
            world.reset()
            camera.initialize()
            camera.set_clipping_range(0.01, 10.0)
            world.pause()
            articulation.set_joint_positions(qpos)
            _sync_physx_transforms_to_usd(physx_interface)
            capture_readback = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            geometry_capture = _geometry_state(stage)
            for view in VIEWS:
                view_hidden_visuals = _set_view_visibility(stage, view)
                pose = camera_poses[view]
                camera.set_world_pose(
                    position=np.asarray(pose["position_world_m"]),
                    orientation=np.asarray(pose["orientation_wxyz"]),
                    camera_axes="usd",
                )
                for _ in range(8):
                    world.render()
                pixels = _read_nonblank_rgba(world, camera)
                actual_position, actual_orientation = (
                    camera.get_world_pose(camera_axes="usd")
                )
                target_projection = (
                    camera.get_image_coords_from_world_points(
                        np.asarray(
                            [pose["target_world_m"]],
                            dtype=np.float64,
                        )
                    )
                )
                finger_points = np.concatenate(
                    [
                        _world_points(stage, FINGER_MESHES[side])
                        for side in ("left", "right")
                    ]
                )
                finger_projection = summarize_image_projection(
                    camera.get_image_coords_from_world_points(
                        finger_points
                    ).tolist(),
                    width=1280,
                    height=900,
                )
                output = screenshot_root / f"{state}_{view}_raw.png"
                render_readback = save_camera_rgba_png(
                    camera,
                    output,
                    rgba=pixels,
                )
                captures.append(
                    validate_screenshot(
                        output.resolve(strict=True),
                        artifact_root=screenshot_root,
                        phase="task5_no_bottle_structure",
                        capture_name=f"{state}_{view}",
                        gate_status="PASS",
                        camera={
                            **pose,
                            "view": view,
                            "resolution": [1280, 900],
                            "runtime": "isaacsim.sensors.camera.Camera",
                            "render_readback": render_readback,
                            "actual_position_world_m": (
                                np.asarray(actual_position).tolist()
                            ),
                            "actual_orientation_wxyz": (
                                np.asarray(actual_orientation).tolist()
                            ),
                            "target_projection_px": (
                                np.asarray(target_projection).tolist()
                            ),
                            "finger_projection": finger_projection,
                        },
                        simulation={
                            "isaac_sim": "5.1.0.0",
                            "kit": "107.3.3",
                            "physx": "107.3.26",
                            "stage_absolute_path": str(root_asset),
                            "stage_sha256": hashes_before["root_usd"],
                            "frame": int(
                                world.current_time_step_index
                            ),
                            "time_s": float(world.current_time),
                            "robot": "follower_left",
                            "collider_type": (
                                "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC"
                            ),
                            "state": state,
                            "finger_targets_m": list(targets),
                            "finger_readback_m": (
                                [
                                    float(capture_readback[left_index]),
                                    float(capture_readback[right_index]),
                                ]
                            ),
                            "surface_gap_m": (
                                geometry_capture["surface_gap_m"]
                            ),
                            "bottle_present": False,
                            "capture_physics_stage": (
                                "FRESH_WORLD_RESET_REINJECTED_FULL_"
                                "LEGAL_POSE_NO_PHYSICS_STEP"
                            ),
                            "capture_world_reset": True,
                            "dynamic_trace_last_frame": frame["value"],
                            "dynamic_post_step_failure_preserved_in_report": (
                                True
                            ),
                            "acceptance_boundary": (
                                "STRUCTURE SCREENSHOT; NO CONTACT OR GRASP "
                                "PASS CLAIM"
                            ),
                            "session_only_hidden_visuals": (
                                view_hidden_visuals
                            ),
                        },
                    )
                )

        pose_gate = validate_pose_records(states)
        drives = _drive_snapshot(stage)
        mimic_present = any(
            record["physx_mimic_api_present"]
            for record in drives.values()
        )
        mimic_drive = drive_mimic_status(
            physx_mimic_api_present=mimic_present,
            left_max_force=float(
                drives["left"]["drive"][
                    "drive:linear:physics:maxForce"
                ]
            ),
            right_max_force=float(
                drives["right"]["drive"][
                    "drive:linear:physics:maxForce"
                ]
            ),
        )
        tracking_pass = all(
            record["maximum_post_step_target_error_m"] <= 1.0e-3
            for record in states
        )
        unexpected_contacts = [
            event
            for record in states
            for event in record["contact_events"]
            if any(
                FINGER_LINKS[side] in (
                    event["actor0"] + event["actor1"]
                )
                for side in ("left", "right")
            )
        ]
        bottle_prims = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if "bottle" in str(prim.GetPath()).lower()
        ]
        manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={
                "task5_no_bottle_structure": [
                    f"{state}_{view}"
                    for state in LEGAL_POSES_M
                    for view in VIEWS
                ]
            },
            artifact_root=screenshot_root,
        )
        hashes_after = {
            name: _sha256(record["path"])
            for name, record in protected.items()
        }
        gates = {
            "approved_and_diagnostic_assets_immutable": (
                hashes_before == hashes_after
            ),
            "runtime_articulation_handle": (
                articulation.num_dof == len(order) == 8
            ),
            "finger_dof_names_unique": all(
                order.count(name) == 1 for name in FINGER_DOF_NAMES
            ),
            "legal_pose_injection": pose_gate["status"] == "PASS",
            "post_step_drive_tracking": tracking_pass,
            "physx_mimic_or_controller_coupling": (
                mimic_drive["status"] == "PASS"
            ),
            "no_bottle_or_test_object": not bottle_prims,
            "no_reported_unexpected_internal_contact": (
                not unexpected_contacts
            ),
            "solve_articulation_contact_last": bool(
                physics_context.get_solve_articulation_contact_last()
            ),
            "raw_screenshot_manifest": manifest["status"] == "PASS",
        }
        report = {
            "schema_version": 1,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "scope": (
                "FOLLOWER_LEFT_SUPPLIER_CAD_TASK5_NO_BOTTLE_STRUCTURE; "
                "DIAGNOSTIC_ONLY_NOT_FINAL"
            ),
            "asset_report_absolute_path": str(asset_report_path),
            "stage_absolute_path": str(root_asset),
            "protected_hashes_before": hashes_before,
            "protected_hashes_after": hashes_after,
            "hard_blockers": asset_report["hard_blockers"],
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "physics_frequency_hz": 60,
                "settle_steps": args.settle_steps,
                "solve_articulation_contact_last": bool(
                    physics_context.get_solve_articulation_contact_last()
                ),
                "contact_subscription_created": subscription is not None,
            },
            "articulation": {
                "path": ARTICULATION_PATH,
                "num_dof": int(articulation.num_dof),
                "num_bodies": int(articulation.num_bodies),
                "dof_order": order,
                "finger_limits_m": limits_by_side,
            },
            "drive_and_mimic": drives,
            "drive_mimic_gate": mimic_drive,
            "pose_injection_gate": pose_gate,
            "states": states,
            "bottle_named_prims": bottle_prims,
            "unexpected_internal_contact_events": unexpected_contacts,
            "camera_poses": camera_poses,
            "captures": captures,
            "screenshot_manifest": manifest,
            "screenshot_visual_review_status": "NOT_RUN",
            "hidden_session_visuals": hidden,
            "gates": gates,
            "known_failed_probe": {
                "api": "PhysXSceneQuery.overlap_shape_any",
                "status": "UNSAFE_LOCAL_NATIVE_CRASH",
                "exit_code": 139,
                "log": str(
                    (
                        ROOT
                        / ".codex/artifacts/"
                        "20260729-aloha-finger-palm-orientation/"
                        "isaac_cad_finger/"
                        "task5_runtime_collision_inventory_probe.log"
                    ).resolve()
                ),
                "retry": False,
            },
            "task8": "NOT_RUN",
        }
        _write_json(args.report.resolve(), report)
        lines = [
            "# Supplier-CAD follower-left Task 5 structure",
            "",
            f"- Status: `{report['status']}`",
            f"- Stage: `{root_asset}`",
            f"- Raw screenshots: `{screenshot_root}`",
            f"- Pose injection: `{pose_gate['status']}`",
            f"- Dynamic drive/mimic: `{mimic_drive['status']}`",
            f"- Post-step tracking: `{'PASS' if tracking_pass else 'FAIL'}`",
            "- Bottle/contact/grasp acceptance: `NOT_RUN`",
            "- Follower-right: `HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT`",
            "- Task 8: `NOT_RUN`",
            "",
            "Pose injection is not counted as dynamic drive or mimic PASS.",
            "",
        ]
        args.markdown.resolve().write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report.resolve()),
                    "screenshots": str(screenshot_root),
                    "capture_count": len(captures),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        world.stop()
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "task8": "NOT_RUN",
        }
        _write_json(args.report.resolve(), failure)
        return 1
    finally:
        app.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

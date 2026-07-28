#!/usr/bin/env python3
"""Capture Isaac Sim 5.1 open/closed supplier-CAD finger evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.isaac_screenshot import (
    look_at_orientation_wxyz,
)
from tools.aloha1_mapping.isaac_screenshot import save_camera_rgba_png
from tools.aloha1_mapping.screenshot_manifest import (
    build_screenshot_manifest,
)
from tools.aloha1_mapping.screenshot_manifest import validate_screenshot

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_installation_v2/aloha_viperx_cad_finger_diagnostic.usda"
)
EXPECTED_DIAGNOSTIC_STAGE_SHA256 = (
    "9f64f2ef6e280d3505c900a7b13e649331cf8bb227d910928647762ef4a5edc3"
)
EXPECTED_DIAGNOSTIC_LAYER_SHA256 = {
    "configuration/supplier_cad_finger_installation.usda": (
        "990f8e9a2c32b401d20b2df8e3c1313b34b67365ad87a001b20451c08a13731e"
    ),
    "geometry/supplier_cad_finger_visual.usda": (
        "781613d408843737b17d9f9a75e8c1b037ecc45749358d4b34ab48a8a7e98d4f"
    ),
}
SCREENSHOT_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "isaac_cad_finger/screenshots_raw"
)
REPORT_PATH = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_isaac_screenshots_raw.json"
)
ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
MESH_PATHS = {
    side: (
        "/workcell/vx300s_left/"
        f"vx300s_left_{side}_finger_link/visuals/"
        f"diagnostic_supplier_cad_{side}_finger/mesh"
    )
    for side in ("left", "right")
}
REQUIRED_VIEWS = (
    "true_top",
    "true_bottom",
    "tip_end",
    "base_oblique",
)


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
        raise RuntimeError(f"supplier-CAD visual Mesh missing: {path}")
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
        for side, path in MESH_PATHS.items()
    }
    centers = {side: cloud.mean(axis=0) for side, cloud in clouds.items()}
    closing = centers["right"] - centers["left"]
    closing /= np.linalg.norm(closing)
    left_projection = clouds["left"] @ closing
    right_projection = clouds["right"] @ closing
    combined = np.concatenate(list(clouds.values()), axis=0)
    target = (combined.min(axis=0) + combined.max(axis=0)) / 2.0
    link = stage.GetPrimAtPath(
        "/workcell/vx300s_left/vx300s_left_left_finger_link"
    )
    link_transform = UsdGeom.XformCache().GetLocalToWorldTransform(link)
    tool = np.asarray(
        list(
            link_transform.TransformDir(
                Gf.Vec3d(1.0, 0.0, 0.0)
            )
        ),
        dtype=np.float64,
    )
    tool /= np.linalg.norm(tool)
    return {
        "centers_world_m": {
            side: centers[side].tolist() for side in ("left", "right")
        },
        "closing_axis_world": closing.tolist(),
        "tool_axis_world": tool.tolist(),
        "target_world_m": target.tolist(),
        "combined_aabb_min_m": combined.min(axis=0).tolist(),
        "combined_aabb_max_m": combined.max(axis=0).tolist(),
        "surface_gap_m": float(
            right_projection.min() - left_projection.max()
        ),
    }


def _set_visual_q_projection(
    stage: Any,
    side: str,
    qpos_m: float,
) -> None:
    """Project legal qpos into the Stage gripper frame for visual review.

    The approved source Stage has authored rigid-body transforms that are
    disjoint from its joint frames and are snapped by PhysX at initialization.
    This visual-only session transform compensates that authored link-frame
    offset while retaining the replacement under its actual finger link.
    """
    from pxr import Gf
    from pxr import UsdGeom

    replacement_path = MESH_PATHS[side].rsplit("/mesh", 1)[0]
    prim = stage.GetPrimAtPath(replacement_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"diagnostic replacement Xform missing: {replacement_path}"
        )
    link_path = (
        "/workcell/vx300s_left/"
        f"vx300s_left_{side}_finger_link"
    )
    gripper_path = (
        "/workcell/vx300s_left/vx300s_left_gripper_link"
    )
    cache = UsdGeom.XformCache()
    link_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(link_path)
    )
    gripper_world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(gripper_path)
    )
    qpos_in_gripper = Gf.Matrix4d(1.0)
    qpos_in_gripper.SetTranslate(
        Gf.Vec3d(0.0687, float(qpos_m), 0.0)
    )
    replacement_local = (
        qpos_in_gripper
        * gripper_world
        * link_world.GetInverse()
    )
    xformable = UsdGeom.Xformable(prim)
    matching = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpName()
        == "xformOp:transform:diagnosticQposProjection"
    ]
    if len(matching) > 1:
        raise RuntimeError(
            f"duplicate visual qpos projection op: {replacement_path}"
        )
    operation = (
        matching[0]
        if matching
        else xformable.AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble,
            opSuffix="diagnosticQposProjection",
        )
    )
    operation.Set(replacement_local)


def _camera_poses(
    geometry: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    target = np.asarray(geometry["target_world_m"], dtype=np.float64)
    tool = np.asarray(geometry["tool_axis_world"], dtype=np.float64)
    closing = np.asarray(
        geometry["closing_axis_world"],
        dtype=np.float64,
    )
    top = np.cross(closing, tool)
    top /= np.linalg.norm(top)
    if float(top @ np.asarray([0.0, 0.0, 1.0])) < 0.0:
        top = -top
    positions = {
        "true_top": target + 0.55 * top,
        "true_bottom": target - 0.55 * top,
        "tip_end": target + 0.42 * tool + 0.035 * top,
        "base_oblique": (
            target - 0.38 * tool + 0.25 * top + 0.14 * closing
        ),
    }
    orientations = {
        "true_top": look_at_orientation_wxyz(
            positions["true_top"],
            target,
            up_world=tool,
        ),
        "true_bottom": look_at_orientation_wxyz(
            positions["true_bottom"],
            target,
            up_world=tool,
        ),
        "tip_end": look_at_orientation_wxyz(
            positions["tip_end"],
            target,
            up_world=top,
        ),
        "base_oblique": look_at_orientation_wxyz(
            positions["base_oblique"],
            target,
            up_world=top,
        ),
    }
    return {
        view: {
            "position_world_m": positions[view].tolist(),
            "orientation_wxyz": orientations[view].tolist(),
            "target_world_m": target.tolist(),
        }
        for view in REQUIRED_VIEWS
    }


def _hide_non_target_workcell(stage: Any) -> list[str]:
    from pxr import UsdGeom

    hidden = []
    for path in (
        "/workcell/table",
        "/workcell/table_frame_T",
        "/workcell/midair",
        "/workcell/placeholder_pipe",
        "/workcell/worldBody",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(path)
    keep_robot_visuals = (
        "/vx300s_left_gripper_link/visuals",
        "/vx300s_left_gripper_prop_link/visuals",
        "/vx300s_left_left_finger_link/visuals",
        "/vx300s_left_right_finger_link/visuals",
    )
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            prim.GetName() == "visuals"
            and path.startswith("/workcell/vx300s_left/")
            and not any(token in path for token in keep_robot_visuals)
            and prim.IsA(UsdGeom.Imageable)
        ):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(path)
    return hidden


def _finger_index(order: list[str], side: str) -> int:
    matches = [
        index
        for index, name in enumerate(order)
        if name == f"vx300s_left_{side}_finger"
        or name == f"{side}_finger"
        or name.endswith(f"_{side}_finger")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {side} finger DOF, found {matches}: {order}"
        )
    return matches[0]


def _set_view_visibility(stage: Any, view: str) -> list[str]:
    """Hide the source gripper shell only where it blocks base-end review."""
    from pxr import UsdGeom

    shell_visual_path = (
        "/workcell/vx300s_left/"
        "vx300s_left_gripper_link/visuals"
    )
    prim = stage.GetPrimAtPath(shell_visual_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
        raise RuntimeError(
            f"gripper shell visual root missing: {shell_visual_path}"
        )
    imageable = UsdGeom.Imageable(prim)
    if view == "base_oblique":
        imageable.MakeInvisible()
        return [shell_visual_path]
    imageable.MakeVisible()
    return []


def _paired_pose_signature(capture: dict[str, Any]) -> tuple[str, ...]:
    camera = capture["camera"]
    return tuple(
        json.dumps(camera[field], sort_keys=True)
        for field in (
            "position_world_m",
            "orientation_wxyz",
            "target_world_m",
            "view",
            "resolution",
            "runtime",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DIAGNOSTIC_STAGE)
    parser.add_argument(
        "--screenshot-root",
        type=Path,
        default=SCREENSHOT_ROOT,
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage.resolve(strict=True)
    screenshot_root = args.screenshot_root.resolve()
    report_path = args.report.resolve()
    stage_sha256_before = _sha256(stage_path)
    if stage_sha256_before != EXPECTED_DIAGNOSTIC_STAGE_SHA256:
        raise RuntimeError(
            f"unexpected diagnostic Stage hash: {stage_sha256_before}"
        )
    layer_sha256_before = {
        relative: _sha256(stage_path.parent / relative)
        for relative in EXPECTED_DIAGNOSTIC_LAYER_SHA256
    }
    if layer_sha256_before != EXPECTED_DIAGNOSTIC_LAYER_SHA256:
        raise RuntimeError(
            "unexpected diagnostic composition layer hash: "
            f"{layer_sha256_before}"
        )
    if screenshot_root.exists():
        raise FileExistsError(
            f"screenshot output already exists: {screenshot_root}"
        )
    screenshot_root.mkdir(parents=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 900,
        }
    )
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf
        from pxr import Usd
        from pxr import UsdLux

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open diagnostic Stage: {stage_path}")
        stage = get_current_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            hidden_workcell_paths = _hide_non_target_workcell(stage)
            dome = UsdLux.DomeLight.Define(
                stage,
                "/workcell/DiagnosticSession/Dome",
            )
            dome.CreateIntensityAttr(700.0)
            dome.CreateColorAttr(Gf.Vec3f(0.9, 0.92, 1.0))
            key = UsdLux.DistantLight.Define(
                stage,
                "/workcell/DiagnosticSession/Key",
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
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="supplier_cad_finger_visual_diagnostic",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        camera = Camera(
            prim_path="/workcell/DiagnosticSession/Camera",
            name="supplier_cad_finger_camera",
            resolution=(1280, 900),
            frequency=60,
        )
        world.scene.add(camera)
        world.reset()
        camera.initialize()
        camera.set_clipping_range(0.01, 10.0)

        order = list(articulation.dof_names)
        left_index = _finger_index(order, "left")
        right_index = _finger_index(order, "right")
        base_qpos = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        )
        captures = []
        states = {}
        paired_camera_poses = None
        for state_name, targets in (
            ("closed", (0.021, -0.021)),
            ("open", (0.057, -0.057)),
        ):
            world.reset()
            qpos = base_qpos.copy()
            qpos[left_index] = targets[0]
            qpos[right_index] = targets[1]
            articulation.set_joint_positions(qpos)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                _set_visual_q_projection(stage, "left", targets[0])
                _set_visual_q_projection(stage, "right", targets[1])
            for _ in range(3):
                world.render()
            readback = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            geometry = _geometry_state(stage)
            if paired_camera_poses is None:
                paired_camera_poses = _camera_poses(geometry)
            state_gate = bool(
                abs(readback[left_index] - targets[0]) <= 1.0e-6
                and abs(readback[right_index] - targets[1]) <= 1.0e-6
            )
            states[state_name] = {
                **geometry,
                "targets_m": list(targets),
                "readback_m": [
                    float(readback[left_index]),
                    float(readback[right_index]),
                ],
                "readback_gate": "PASS" if state_gate else "FAIL",
            }
            for view in REQUIRED_VIEWS:
                pose = paired_camera_poses[view]
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    view_hidden_visuals = _set_view_visibility(
                        stage,
                        view,
                    )
                camera.set_world_pose(
                    position=np.asarray(pose["position_world_m"]),
                    orientation=np.asarray(pose["orientation_wxyz"]),
                    camera_axes="usd",
                )
                for _ in range(8):
                    world.render()
                output = screenshot_root / f"{state_name}_{view}_raw.png"
                render_readback = save_camera_rgba_png(camera, output)
                capture = validate_screenshot(
                    output.resolve(strict=True),
                    artifact_root=screenshot_root,
                    phase="supplier_cad_finger_installation",
                    capture_name=f"{state_name}_{view}",
                    gate_status="PASS" if state_gate else "FAIL",
                    camera={
                        **pose,
                        "view": view,
                        "runtime": "isaacsim.sensors.camera.Camera",
                        "resolution": [1280, 900],
                        "render_readback": render_readback,
                    },
                    simulation={
                        "isaac_sim": "5.1.0.0",
                        "kit": "107.3.3",
                        "physx": "107.3.26",
                        "stage_absolute_path": str(stage_path),
                        "stage_sha256": stage_sha256_before,
                        "robot": "follower_left",
                        "visual_type": (
                            "SUPPLIER_CAD_V2_VISUAL_ONLY_DIAGNOSTIC"
                        ),
                        "collider_type": "SOURCE_COLLIDER_UNCHANGED",
                        "state": state_name,
                        "frame": int(world.current_time_step_index),
                        "time_s": float(world.current_time),
                        "dof_order": order,
                        "finger_targets_m": list(targets),
                        "finger_readback_m": [
                            float(readback[left_index]),
                            float(readback[right_index]),
                        ],
                        "surface_gap_m": geometry["surface_gap_m"],
                        "visual_state_method": (
                            "VISUAL_SESSION_QPOS_PROJECTION_WITH_"
                            "USD_LINK_FRAME_COMPENSATION"
                        ),
                        "visual_state_boundary": (
                            "Diagnostic render only. The approved source "
                            "Stage has disjoint authored body/joint frames; "
                            "this compensates only the added visual Xform "
                            "in the anonymous session layer. It is not "
                            "physics motion, collision, contact, or grasp "
                            "evidence."
                        ),
                        "hidden_session_visuals": hidden_workcell_paths,
                        "view_hidden_visuals": view_hidden_visuals,
                    },
                )
                captures.append(capture)

        required_capture_count = 8
        manifest = build_screenshot_manifest(
            captures=captures,
            required_captures={
                "supplier_cad_finger_installation": [
                    f"{state}_{view}"
                    for state in ("closed", "open")
                    for view in REQUIRED_VIEWS
                ]
            },
            artifact_root=screenshot_root,
        )
        paired_pose_gate = all(
            _paired_pose_signature(
                next(
                    capture
                    for capture in captures
                    if capture["capture_name"] == f"closed_{view}"
                )
            )
            == _paired_pose_signature(
                next(
                    capture
                    for capture in captures
                    if capture["capture_name"] == f"open_{view}"
                )
            )
            for view in REQUIRED_VIEWS
        )
        aperture_gate = (
            states["open"]["surface_gap_m"]
            > states["closed"]["surface_gap_m"]
        )
        stage_sha256_after = _sha256(stage_path)
        layer_sha256_after = {
            relative: _sha256(stage_path.parent / relative)
            for relative in EXPECTED_DIAGNOSTIC_LAYER_SHA256
        }
        gates = {
            "capture_count": len(captures) == required_capture_count,
            "manifest": manifest["status"] == "PASS",
            "paired_camera_pose_exact": paired_pose_gate,
            "open_aperture_exceeds_closed": aperture_gate,
            "state_readback": all(
                record["readback_gate"] == "PASS"
                for record in states.values()
            ),
            "stage_immutable": (
                stage_sha256_before
                == stage_sha256_after
                == EXPECTED_DIAGNOSTIC_STAGE_SHA256
            ),
            "composition_layers_immutable": (
                layer_sha256_before
                == layer_sha256_after
                == EXPECTED_DIAGNOSTIC_LAYER_SHA256
            ),
        }
        report = {
            "schema_version": 1,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "required_capture_count": required_capture_count,
            "stage_absolute_path": str(stage_path),
            "stage_sha256_before": stage_sha256_before,
            "stage_sha256_after": stage_sha256_after,
            "composition_layer_sha256_before": layer_sha256_before,
            "composition_layer_sha256_after": layer_sha256_after,
            "camera_views": list(REQUIRED_VIEWS),
            "paired_camera_poses": paired_camera_poses,
            "states": states,
            "captures": captures,
            "manifest": manifest,
            "gates": gates,
            "visual_review_status": "NOT_RUN",
            "acceptance_boundary": (
                "Raw screenshot acquisition and numeric geometry state only; "
                "visual-model review and annotation are separate required "
                "gates. No collision or grasp claim."
            ),
            "task8": "NOT_RUN",
        }
        _write_json(report_path, report)
        print(f"status={report['status']}")
        print(f"captures={len(captures)}")
        print(f"report={report_path}")
        print(f"screenshot_root={screenshot_root}")
        return 0 if report["status"] == "PASS" else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

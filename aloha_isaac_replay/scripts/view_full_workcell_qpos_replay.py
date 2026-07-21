from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from aloha_isaac_replay.adapters.gripper_mapping import standard_gripper_qpos_to_isaac_fingers


DEFAULT_STAGE_USD = (
    "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
    "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda"
)
DEFAULT_EPISODE = (
    "/home/eii/data/openpi0.5-rtc-reward-learning/from_103/2026-07-07_morning_strict6000/"
    "rollouts/key_regions/twist_off_the_bottle_cap/2026-07-07/rl/"
    "key_region_00590092c6824332a8770a49ffc6dc31/episode.hdf5"
)
DEFAULT_BOTTLE_USD = "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
DEFAULT_BOTTLE_USD_PRIM_PATH = "/Bottle500"
DEFAULT_GRASP_YAML = "assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml"
DEFAULT_GRASP_NAME = "grasp_rear_quarter"
DEFAULT_TABLETOP_TOP_Z = 0.004086510930165169
DEFAULT_TABLETOP_CLEARANCE = 0.001
DEFAULT_TABLETOP_PLACEMENT_FRAME = 326
DEFAULT_REAR_QUARTER_FRACTION = 0.25
LEFT_ROOT = "/scene/left_base_link/left_base_link"
RIGHT_ROOT = "/scene/right_base_link/right_base_link"
LEFT_EE_BODY = "left_gripper_link"
LEFT_FINGER_BODIES = ("left_left_finger_link", "left_right_finger_link")

LEFT_DOF_ORDER = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger",
    "left_right_finger",
)
RIGHT_DOF_ORDER = (
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_left_finger",
    "right_right_finger",
)


def _load_qpos(path: Path, max_frames: int | None) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        qpos = h5["observations/qpos"][:]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected observations/qpos shape (T, 14), got {qpos.shape}")
    if max_frames is not None:
        qpos = qpos[:max_frames]
    if len(qpos) == 0:
        raise ValueError(f"Episode has no qpos frames: {path}")
    return qpos.astype(np.float64, copy=False)


def _resolve_indices(actual_names: list[str], expected_names: tuple[str, ...], side: str) -> np.ndarray:
    missing = [name for name in expected_names if name not in actual_names]
    if missing:
        raise ValueError(f"{side} articulation missing DOFs {missing}; actual={actual_names}")
    indices = np.asarray([actual_names.index(name) for name in expected_names], dtype=np.int64)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"{side} duplicate DOF indices: {indices.tolist()}")
    return indices


def _qpos_to_side_targets(qpos_frame: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        arm = qpos_frame[0:6]
        gripper_qpos = float(np.clip(qpos_frame[6], 0.0, 1.0))
        gripper = standard_gripper_qpos_to_isaac_fingers(gripper_qpos, "left")
        fingers = (gripper["left/left_finger"], gripper["left/right_finger"])
    elif side == "right":
        arm = qpos_frame[7:13]
        gripper_qpos = float(np.clip(qpos_frame[13], 0.0, 1.0))
        gripper = standard_gripper_qpos_to_isaac_fingers(gripper_qpos, "right")
        fingers = (gripper["right/left_finger"], gripper["right/right_finger"])
    else:
        raise ValueError(f"side must be left or right, got {side!r}")
    return np.asarray([*arm, *fingers], dtype=np.float64)


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return _quat_xyzw_to_matrix(np.asarray([x, y, z, w], dtype=np.float64))


def _matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _transform_from_pose(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _pose_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(transform, dtype=np.float64)
    return t[:3, 3].copy(), _matrix_to_quat_wxyz(t[:3, :3])


def _set_bottle_transform(translate_op, orient_op, Gf, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
    translate_op.Set(Gf.Vec3d(*[float(v) for v in position]))
    orient_op.Set(
        Gf.Quatd(
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        )
    )


def _load_grasp_transform(grasp_yaml: Path, grasp_name: str) -> dict[str, object]:
    data = yaml.safe_load(grasp_yaml.read_text(encoding="utf-8")) or {}
    grasps = data.get("grasps") or {}
    grasp = grasps.get(grasp_name) if isinstance(grasps, dict) else None
    if grasp is None:
        raise ValueError(f"Cannot find grasp {grasp_name!r} in {grasp_yaml}")
    quat = np.asarray([grasp["orientation"]["w"], *grasp["orientation"]["xyz"]], dtype=np.float64)
    position = np.asarray(grasp["position"], dtype=np.float64)
    return {
        "path": str(grasp_yaml),
        "name": grasp_name,
        "object_frame": data.get("object_frame"),
        "gripper_frame": data.get("gripper_frame"),
        "t_object_gripper": _transform_from_pose(position, quat),
        "position": position.tolist(),
        "quat_wxyz": quat.tolist(),
    }


def _set_articulation_qpos(left, right, left_indices, right_indices, qpos_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values = _qpos_to_side_targets(qpos_frame, "left")
    right_values = _qpos_to_side_targets(qpos_frame, "right")
    left.set_joint_positions(left_values, joint_indices=left_indices)
    right.set_joint_positions(right_values, joint_indices=right_indices)
    left.set_joint_velocities(np.zeros_like(left_values), joint_indices=left_indices)
    right.set_joint_velocities(np.zeros_like(right_values), joint_indices=right_indices)
    return left_values, right_values


def _move_window_to_workspace(workspace: int, timeout_sec: float = 8.0) -> dict[str, object]:
    if workspace < 0:
        return {"attempted": False, "reason": "disabled"}
    if not os.environ.get("DISPLAY"):
        return {"attempted": False, "reason": "no DISPLAY"}
    if subprocess.run(["which", "xdotool"], capture_output=True, text=True).returncode != 0:
        return {"attempted": False, "reason": "xdotool unavailable"}
    pid = str(os.getpid())
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = subprocess.run(["xdotool", "search", "--pid", pid], capture_output=True, text=True)
        windows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if windows:
            for window in windows:
                subprocess.run(["xdotool", "set_desktop_for_window", window, str(workspace)], check=False)
            return {"attempted": True, "workspace": workspace, "windows": windows}
        time.sleep(0.2)
    return {"attempted": True, "workspace": workspace, "windows": [], "reason": "no window found before timeout"}


def _create_bottle_proxy(stage, UsdGeom, Gf, root_path: str):
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetDisplayName("Replay bottle proxy follows left gripper link")
    translate_op = root.AddTranslateOp()
    orient_op = root.AddOrientOp()

    body = UsdGeom.Cylinder.Define(stage, f"{root_path}/body")
    body.CreateAxisAttr("X")
    body.CreateHeightAttr(0.16)
    body.CreateRadiusAttr(0.025)
    body.GetDisplayColorAttr().Set([(0.1, 0.32, 0.92)])

    body_xform = UsdGeom.Xformable(body.GetPrim())
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0.07, 0.0, 0.0))

    neck = UsdGeom.Cylinder.Define(stage, f"{root_path}/neck")
    neck.CreateAxisAttr("X")
    neck.CreateHeightAttr(0.07)
    neck.CreateRadiusAttr(0.008)
    neck.GetDisplayColorAttr().Set([(0.82, 0.9, 1.0)])
    neck_xform = UsdGeom.Xformable(neck.GetPrim())
    neck_xform.AddTranslateOp().Set(Gf.Vec3d(0.185, 0.0, 0.0))

    mouth = UsdGeom.Sphere.Define(stage, f"{root_path}/mouth")
    mouth.CreateRadiusAttr(0.012)
    mouth.GetDisplayColorAttr().Set([(0.02, 0.04, 0.1)])
    mouth_xform = UsdGeom.Xformable(mouth.GetPrim())
    mouth_xform.AddTranslateOp().Set(Gf.Vec3d(0.225, 0.0, 0.0))
    return translate_op, orient_op


def _create_bottle_usd_reference(stage, Usd, UsdGeom, Gf, root_path: str, usd_path: Path, usd_prim_path: str):
    if not usd_path.exists():
        raise FileNotFoundError(f"Bottle USD does not exist: {usd_path}")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetDisplayName("Replay Bottle500 follows left gripper grasp transform")
    root.GetPrim().GetReferences().AddReference(str(usd_path), usd_prim_path)
    xform = UsdGeom.Xformable(root.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    return translate_op, orient_op


def _inspect_bottle_composition(stage, Usd, root_path: str) -> dict[str, object]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return {"runtime_object_path": root_path, "pass": False, "status": "FAIL_BOTTLE_ROOT_MISSING"}
    mesh_paths = []
    visual_meshes = []
    collision_prims = []
    frame_paths = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.GetTypeName() == "Mesh":
            mesh_paths.append(path)
            if "/Visuals/" in path:
                visual_meshes.append(path)
        if "/Collisions/" in path:
            collision_prims.append(path)
        if "/Frames/" in path:
            frame_paths.append(path)
    mouth_frame = f"{root_path}/Frames/MouthFrame"
    inner_bottom_frame = f"{root_path}/Frames/InnerBottomFrame"
    ok = bool(visual_meshes and stage.GetPrimAtPath(mouth_frame).IsValid() and stage.GetPrimAtPath(inner_bottom_frame).IsValid())
    return {
        "runtime_object_path": root_path,
        "pass": ok,
        "status": "PASS_BOTTLE_USD_RUNTIME_COMPOSITION" if ok else "FAIL_BOTTLE_USD_RUNTIME_COMPOSITION",
        "mesh_count": len(mesh_paths),
        "visual_mesh_count": len(visual_meshes),
        "visual_mesh_sample": visual_meshes[:8],
        "collision_prim_count": len(collision_prims),
        "collision_prim_sample": collision_prims[:8],
        "mouth_frame_path": mouth_frame,
        "mouth_frame_exists": stage.GetPrimAtPath(mouth_frame).IsValid(),
        "inner_bottom_frame_path": inner_bottom_frame,
        "inner_bottom_frame_exists": stage.GetPrimAtPath(inner_bottom_frame).IsValid(),
        "frame_sample": frame_paths[:8],
    }


def _make_bottle_kinematic(stage, Usd, UsdPhysics, root_path: str) -> dict[str, object]:
    """Keep the visual replay bottle controlled by this script, not by gravity/contact physics."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return {"status": "FAIL_BOTTLE_ROOT_MISSING", "kinematic_prim_count": 0, "paths": []}
    paths = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            rigid_body.CreateKinematicEnabledAttr(True)
            rigid_body.CreateRigidBodyEnabledAttr(True)
            paths.append(str(prim.GetPath()))
    return {
        "status": "PASS_BOTTLE_KINEMATIC_RUNTIME_CONTROL" if paths else "NO_RIGID_BODY_API_FOUND",
        "kinematic_prim_count": len(paths),
        "paths": paths[:12],
    }


def _make_invisible_if_present(stage, UsdGeom, prim_path: str) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    return True


def _set_camera_look_at(stage, UsdGeom, Gf, camera_path: str, eye: tuple[float, float, float], target: tuple[float, float, float]) -> str:
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(28.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    eye_vec = Gf.Vec3d(*eye)
    target_vec = Gf.Vec3d(*target)
    camera_to_world = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(camera_to_world)
    return camera_path


def _set_active_viewport_camera(camera_path: str) -> bool:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.camera_path = Sdf.Path(camera_path)
        return True
    except Exception:
        return False


def _body_pose_xyzw(articulation, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_names = list(articulation._articulation_view.body_names)
    if body_name not in body_names:
        raise ValueError(f"Body {body_name!r} not found; available={body_names}")
    index = body_names.index(body_name)
    transforms = np.asarray(articulation._articulation_view._physics_view.get_link_transforms(), dtype=np.float64)
    transforms = transforms.reshape((-1, 7))
    pose = transforms[index]
    return pose[:3].copy(), pose[3:7].copy()


def _body_position(articulation, body_name: str) -> np.ndarray:
    return _body_pose_xyzw(articulation, body_name)[0]


def _finger_gap_center(articulation) -> tuple[np.ndarray, dict[str, object]]:
    left_pos = _body_position(articulation, LEFT_FINGER_BODIES[0])
    right_pos = _body_position(articulation, LEFT_FINGER_BODIES[1])
    gap_center = 0.5 * (left_pos + right_pos)
    gap_vector = right_pos - left_pos
    gap_norm = float(np.linalg.norm(gap_vector))
    return gap_center, {
        "left_finger_body": LEFT_FINGER_BODIES[0],
        "right_finger_body": LEFT_FINGER_BODIES[1],
        "left_finger_position": left_pos.tolist(),
        "right_finger_position": right_pos.tolist(),
        "finger_gap_center": gap_center.tolist(),
        "finger_gap_vector": gap_vector.tolist(),
        "finger_gap_norm": gap_norm,
    }


def _bbox_dict(stage, UsdGeom, root_path: str) -> dict[str, object]:
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        return {"path": root_path, "exists": False, "bbox_valid": False}
    cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    min_array = np.asarray([minimum[0], minimum[1], minimum[2]], dtype=np.float64)
    max_array = np.asarray([maximum[0], maximum[1], maximum[2]], dtype=np.float64)
    if not np.all(np.isfinite(min_array)) or not np.all(np.isfinite(max_array)):
        return {"path": root_path, "exists": True, "bbox_valid": False}
    center = 0.5 * (min_array + max_array)
    size = max_array - min_array
    return {
        "path": root_path,
        "exists": True,
        "bbox_valid": bool(np.all(size >= -1e-9)),
        "min": min_array.tolist(),
        "max": max_array.tolist(),
        "center": center.tolist(),
        "size": size.tolist(),
    }


def _place_bottle_tabletop_rear_quarter(
    *,
    stage,
    UsdGeom,
    Gf,
    app,
    translate_op,
    orient_op,
    bottle_root: str,
    finger_gap_center: np.ndarray,
    table_top_z: float,
    clearance: float,
    rear_fraction: float,
) -> tuple[list[float], list[float], dict[str, object]]:
    # Bottle500 local +Z is the bottle long axis. Rotate local +Z onto world +X so it lies flat.
    quat_wxyz = np.asarray([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64)
    zero_position = np.zeros(3, dtype=np.float64)
    _set_bottle_transform(translate_op, orient_op, Gf, zero_position, quat_wxyz)
    for _ in range(3):
        app.update()
    bbox_before = _bbox_dict(stage, UsdGeom, bottle_root)
    if not bbox_before.get("bbox_valid"):
        raise RuntimeError(f"Cannot place Bottle500 on tabletop; invalid bbox: {bbox_before}")

    bbox_min = np.asarray(bbox_before["min"], dtype=np.float64)
    bbox_center = np.asarray(bbox_before["center"], dtype=np.float64)
    bbox_size = np.asarray(bbox_before["size"], dtype=np.float64)
    rear_target_x = float(bbox_min[0] + float(rear_fraction) * bbox_size[0])
    target_bottom_z = float(table_top_z) + float(clearance)
    position = np.asarray(
        [
            float(finger_gap_center[0] - rear_target_x),
            float(finger_gap_center[1] - bbox_center[1]),
            float(target_bottom_z - bbox_min[2]),
        ],
        dtype=np.float64,
    )
    _set_bottle_transform(translate_op, orient_op, Gf, position, quat_wxyz)
    for _ in range(3):
        app.update()
    bbox_after = _bbox_dict(stage, UsdGeom, bottle_root)
    if not bbox_after.get("bbox_valid"):
        raise RuntimeError(f"Cannot verify Bottle500 tabletop placement; invalid bbox: {bbox_after}")

    after_min = np.asarray(bbox_after["min"], dtype=np.float64)
    after_size = np.asarray(bbox_after["size"], dtype=np.float64)
    after_rear_x = float(after_min[0] + float(rear_fraction) * after_size[0])
    gap_error = float(after_min[2] - target_bottom_z)
    rear_quarter_error = float(after_rear_x - finger_gap_center[0])
    long_axis_index = int(np.argmax(after_size))
    placement = {
        "mode": "tabletop_fixed",
        "description": "Bottle500 is flat on the tabletop; local +Z long axis is rotated onto world +X.",
        "long_axis_world": "X",
        "long_axis_index": long_axis_index,
        "long_axis_size_m": float(after_size[long_axis_index]),
        "table_top_z_m": float(table_top_z),
        "tabletop_clearance_m": float(clearance),
        "target_bottom_z_m": target_bottom_z,
        "bbox_before": bbox_before,
        "bbox_after": bbox_after,
        "finger_gap_center": np.asarray(finger_gap_center, dtype=np.float64).tolist(),
        "rear_fraction_target": float(rear_fraction),
        "rear_quarter_x_after": after_rear_x,
        "rear_quarter_x_error_m": rear_quarter_error,
        "tabletop_gap_error_m": gap_error,
        "pass": bool(
            long_axis_index == 0
            and abs(gap_error) <= 2e-4
            and abs(rear_quarter_error) <= 2e-3
            and after_size[0] > after_size[1]
            and after_size[0] > after_size[2]
        ),
    }
    placement["status"] = "PASS_TABLETOP_FLAT_REAR_QUARTER_PLACEMENT" if placement["pass"] else "FAIL_TABLETOP_FLAT_REAR_QUARTER_PLACEMENT"
    return position.tolist(), quat_wxyz.tolist(), placement


def _update_bottle_pose(translate_op, orient_op, Gf, left, local_offset: np.ndarray) -> tuple[list[float], list[float]]:
    pos, quat_xyzw = _body_pose_xyzw(left, LEFT_EE_BODY)
    rot = _quat_xyzw_to_matrix(quat_xyzw)
    bottle_pos = pos + rot @ local_offset
    translate_op.Set(Gf.Vec3d(*[float(v) for v in bottle_pos]))
    orient_op.Set(Gf.Quatf(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
    return bottle_pos.tolist(), quat_xyzw.tolist()


def _update_bottle_grasp_pose(translate_op, orient_op, Gf, left, t_object_gripper: np.ndarray) -> tuple[list[float], list[float]]:
    pos, quat_xyzw = _body_pose_xyzw(left, LEFT_EE_BODY)
    t_world_gripper = np.eye(4, dtype=np.float64)
    t_world_gripper[:3, :3] = _quat_xyzw_to_matrix(quat_xyzw)
    t_world_gripper[:3, 3] = pos
    t_world_object = t_world_gripper @ np.linalg.inv(np.asarray(t_object_gripper, dtype=np.float64))
    bottle_pos, bottle_quat_wxyz = _pose_from_transform(t_world_object)
    translate_op.Set(Gf.Vec3d(*[float(v) for v in bottle_pos]))
    orient_op.Set(
        Gf.Quatd(
            float(bottle_quat_wxyz[0]),
            float(bottle_quat_wxyz[1]),
            float(bottle_quat_wxyz[2]),
            float(bottle_quat_wxyz[3]),
        )
    )
    return bottle_pos.tolist(), bottle_quat_wxyz.tolist()


def _try_update_bottle_pose(
    translate_op,
    orient_op,
    Gf,
    left,
    local_offset: np.ndarray,
    fallback_pos: list[float],
    fallback_quat: list[float],
) -> tuple[list[float], list[float], str | None]:
    try:
        pos, quat = _update_bottle_pose(translate_op, orient_op, Gf, left, local_offset)
    except (AttributeError, RuntimeError, ValueError) as exc:
        return fallback_pos, fallback_quat, f"{type(exc).__name__}: {exc}"
    return pos, quat, None


def _try_update_bottle_grasp_pose(
    translate_op,
    orient_op,
    Gf,
    left,
    t_object_gripper: np.ndarray,
    fallback_pos: list[float],
    fallback_quat: list[float],
) -> tuple[list[float], list[float], str | None]:
    try:
        pos, quat = _update_bottle_grasp_pose(translate_op, orient_op, Gf, left, t_object_gripper)
    except (AttributeError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return fallback_pos, fallback_quat, f"{type(exc).__name__}: {exc}"
    return pos, quat, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Loop a full workcell Original ALOHA qpos replay. By default the replay uses the Bottle500 USD asset "
            "and places it flat on the tabletop with the rear quarter aligned to the left gripper opening."
        )
    )
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--stage-usd", default=DEFAULT_STAGE_USD)
    parser.add_argument("--left-root", default=LEFT_ROOT)
    parser.add_argument("--right-root", default=RIGHT_ROOT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--render-headless", action="store_true")
    parser.add_argument("--workspace", type=int, default=2)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--bottle-local-offset", nargs=3, type=float, default=(0.035, 0.0, 0.0))
    parser.add_argument("--bottle-usd", default=DEFAULT_BOTTLE_USD)
    parser.add_argument("--bottle-usd-prim-path", default=DEFAULT_BOTTLE_USD_PRIM_PATH)
    parser.add_argument("--grasp-yaml", default=DEFAULT_GRASP_YAML)
    parser.add_argument("--grasp-name", default=DEFAULT_GRASP_NAME)
    parser.add_argument(
        "--bottle-motion-mode",
        choices=("tabletop_fixed", "held_by_gripper"),
        default="tabletop_fixed",
        help=(
            "tabletop_fixed keeps Bottle500 lying flat on the table; held_by_gripper keeps the old diagnostic "
            "behavior that binds Bottle500 to the left gripper through the selected grasp transform."
        ),
    )
    parser.add_argument("--tabletop-placement-frame", type=int, default=DEFAULT_TABLETOP_PLACEMENT_FRAME)
    parser.add_argument("--tabletop-top-z", type=float, default=DEFAULT_TABLETOP_TOP_Z)
    parser.add_argument("--tabletop-clearance", type=float, default=DEFAULT_TABLETOP_CLEARANCE)
    parser.add_argument("--rear-quarter-fraction", type=float, default=DEFAULT_REAR_QUARTER_FRACTION)
    parser.add_argument("--legacy-bottle-proxy", action="store_true")
    parser.add_argument("--show-bottle-collisions", action="store_true")
    parser.add_argument("--show-office-shell", action="store_true")
    parser.add_argument("--output-json", default="reports/aloha_isaac_replay/visual_replay/latest_full_workcell_qpos_replay.json")
    args = parser.parse_args()

    episode = Path(args.episode).expanduser().resolve()
    stage_usd = Path(args.stage_usd).expanduser()
    if not stage_usd.is_absolute():
        stage_usd = (Path.cwd() / stage_usd).resolve()
    bottle_usd = Path(args.bottle_usd).expanduser()
    if not bottle_usd.is_absolute():
        bottle_usd = (Path.cwd() / bottle_usd).resolve()
    grasp_yaml = Path(args.grasp_yaml).expanduser()
    if not grasp_yaml.is_absolute():
        grasp_yaml = (Path.cwd() / grasp_yaml).resolve()
    qpos = _load_qpos(episode, args.max_frames)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    phase = "before_simulation_app"

    def write_status(payload: dict[str, object]) -> None:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    write_status(
        {
            "status": "STARTING",
            "phase": phase,
            "episode": str(episode),
            "stage_usd": str(stage_usd),
            "frames": int(len(qpos)),
            "bottle_usd": str(bottle_usd),
            "grasp_yaml": str(grasp_yaml),
            "grasp_name": args.grasp_name,
            "bottle_motion_mode": "legacy_bottle_proxy" if args.legacy_bottle_proxy else args.bottle_motion_mode,
        }
    )

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "create_new_stage": False,
            "disable_viewport_updates": bool(args.headless and not args.render_headless),
            "width": int(args.window_width),
            "height": int(args.window_height),
            "window_width": int(args.window_width),
            "window_height": int(args.window_height),
            "multi_gpu": False,
            "sync_loads": True,
            "limit_cpu_threads": 12,
        }
    )
    try:
        phase = "after_simulation_app"
        write_status(
            {
                "status": "STARTING",
                "phase": phase,
                "episode": str(episode),
                "stage_usd": str(stage_usd),
                "frames": int(len(qpos)),
                "bottle_usd": str(bottle_usd),
                "grasp_yaml": str(grasp_yaml),
                "grasp_name": args.grasp_name,
                "bottle_motion_mode": "legacy_bottle_proxy" if args.legacy_bottle_proxy else args.bottle_motion_mode,
            }
        )
        phase = "import_isaac_modules"
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

        phase = "open_stage"
        World.clear_instance()
        if not stage_utils.open_stage(str(stage_usd)):
            raise RuntimeError(f"Failed to open stage: {stage_usd}")
        for _ in range(30):
            app.update()

        phase = "create_world"
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=1.0 / float(args.fps), rendering_dt=1.0 / float(args.fps))
        stage = omni.usd.get_context().get_stage()
        hidden_prims = []
        if not args.show_office_shell:
            for prim_path in ("/World/OfficeEnvironment",):
                if _make_invisible_if_present(stage, UsdGeom, prim_path):
                    hidden_prims.append(prim_path)

        light = UsdLux.DistantLight.Define(stage, "/World/codex_full_replay_light")
        light.CreateIntensityAttr(250.0)

        camera_path = _set_camera_look_at(
            stage,
            UsdGeom,
            Gf,
            "/World/codex_full_replay_camera",
            eye=(0.52, -0.72, 0.72),
            target=(-0.05, 0.21, 0.23),
        )

        bottle_root = "/World/ReplayBottleProxy" if args.legacy_bottle_proxy else "/World/ReplayBottle500"
        grasp_info = None
        bottle_composition = None
        bottle_kinematic = None
        if args.legacy_bottle_proxy:
            phase = "create_legacy_bottle_proxy"
            bottle_translate_op, bottle_orient_op = _create_bottle_proxy(stage, UsdGeom, Gf, bottle_root)
        else:
            phase = "load_grasp_yaml"
            grasp_info = _load_grasp_transform(grasp_yaml, args.grasp_name)
            phase = "create_bottle_usd_reference"
            bottle_translate_op, bottle_orient_op = _create_bottle_usd_reference(
                stage,
                Usd,
                UsdGeom,
                Gf,
                bottle_root,
                bottle_usd,
                args.bottle_usd_prim_path,
            )
            phase = "update_after_bottle_reference"
            for _ in range(5):
                app.update()
            phase = "inspect_bottle_composition"
            if not args.show_bottle_collisions:
                _make_invisible_if_present(stage, UsdGeom, f"{bottle_root}/Collisions")
            bottle_composition = _inspect_bottle_composition(stage, Usd, bottle_root)
            if not bottle_composition["pass"]:
                raise RuntimeError(f"BottleUSD composition failed: {bottle_composition}")
            bottle_kinematic = _make_bottle_kinematic(stage, Usd, UsdPhysics, bottle_root)

        phase = "create_articulations"
        left = world.scene.add(SingleArticulation(prim_path=args.left_root, name="left_full_aloha"))
        right = world.scene.add(SingleArticulation(prim_path=args.right_root, name="right_full_aloha"))
        phase = "world_reset"
        world.reset()

        phase = "resolve_dof_indices"
        left_indices = _resolve_indices(list(left.dof_names), LEFT_DOF_ORDER, "left")
        right_indices = _resolve_indices(list(right.dof_names), RIGHT_DOF_ORDER, "right")
        local_offset = np.asarray(args.bottle_local_offset, dtype=np.float64)

        phase = "initial_pose"
        _set_articulation_qpos(left, right, left_indices, right_indices, qpos[0])
        app.update()
        bottle_tabletop_placement = None
        finger_gap_row = None
        if args.legacy_bottle_proxy:
            initial_bottle_pos, initial_bottle_quat, initial_bottle_error = _try_update_bottle_pose(
                bottle_translate_op,
                bottle_orient_op,
                Gf,
                left,
                local_offset,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            )
        elif args.bottle_motion_mode == "held_by_gripper":
            initial_bottle_pos, initial_bottle_quat, initial_bottle_error = _try_update_bottle_grasp_pose(
                bottle_translate_op,
                bottle_orient_op,
                Gf,
                left,
                np.asarray(grasp_info["t_object_gripper"], dtype=np.float64),
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            )
        else:
            placement_frame = int(np.clip(args.tabletop_placement_frame, 0, len(qpos) - 1))
            _set_articulation_qpos(left, right, left_indices, right_indices, qpos[placement_frame])
            app.update()
            finger_gap_center, finger_gap_row = _finger_gap_center(left)
            initial_bottle_pos, initial_bottle_quat, bottle_tabletop_placement = _place_bottle_tabletop_rear_quarter(
                stage=stage,
                UsdGeom=UsdGeom,
                Gf=Gf,
                app=app,
                translate_op=bottle_translate_op,
                orient_op=bottle_orient_op,
                bottle_root=bottle_root,
                finger_gap_center=finger_gap_center,
                table_top_z=float(args.tabletop_top_z),
                clearance=float(args.tabletop_clearance),
                rear_fraction=float(args.rear_quarter_fraction),
            )
            initial_bottle_error = None
            _set_articulation_qpos(left, right, left_indices, right_indices, qpos[0])
            app.update()
        active_camera = _set_active_viewport_camera(camera_path) if not args.headless else False
        window_move = _move_window_to_workspace(args.workspace) if not args.headless else {"attempted": False, "reason": "headless"}

        phase = "write_initial_status"
        status = {
            "status": "RUNNING",
            "mode": (
                "full_workcell_qpos_replay_with_legacy_left_ee_bottle_proxy"
                if args.legacy_bottle_proxy
                else f"full_workcell_qpos_replay_with_bottle_usd_{args.bottle_motion_mode}"
            ),
            "episode": str(episode),
            "stage_usd": str(stage_usd),
            "frames": int(len(qpos)),
            "fps": float(args.fps),
            "left_root": args.left_root,
            "right_root": args.right_root,
            "left_dof_names": list(left.dof_names),
            "right_dof_names": list(right.dof_names),
            "left_indices": left_indices.tolist(),
            "right_indices": right_indices.tolist(),
            "bottle_root": bottle_root,
            "bottle_motion_mode": "legacy_bottle_proxy" if args.legacy_bottle_proxy else args.bottle_motion_mode,
            "bottle_usd": None if args.legacy_bottle_proxy else str(bottle_usd),
            "bottle_usd_prim_path": None if args.legacy_bottle_proxy else args.bottle_usd_prim_path,
            "bottle_composition": bottle_composition,
            "bottle_kinematic": bottle_kinematic,
            "grasp_yaml": None if args.legacy_bottle_proxy else str(grasp_yaml),
            "grasp_name": None if args.legacy_bottle_proxy else args.grasp_name,
            "grasp_position_object": None if grasp_info is None else grasp_info["position"],
            "grasp_quat_wxyz": None if grasp_info is None else grasp_info["quat_wxyz"],
            "tabletop_placement_frame": None if args.legacy_bottle_proxy else int(np.clip(args.tabletop_placement_frame, 0, len(qpos) - 1)),
            "tabletop_placement": bottle_tabletop_placement,
            "tabletop_finger_gap": finger_gap_row,
            "bottle_local_offset": local_offset.tolist() if args.legacy_bottle_proxy else None,
            "initial_bottle_position": initial_bottle_pos,
            "initial_bottle_quat_xyzw": initial_bottle_quat if args.legacy_bottle_proxy else None,
            "initial_bottle_quat_wxyz": None if args.legacy_bottle_proxy else initial_bottle_quat,
            "initial_bottle_pose_error": initial_bottle_error,
            "window_move": window_move,
            "hidden_prims": hidden_prims,
            "active_viewport_camera": active_camera,
            "camera_path": camera_path,
        }
        output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(f"Full workcell qpos replay ready: {output}", flush=True)

        max_left_arm_error = 0.0
        max_right_arm_error = 0.0
        max_left_finger_error = 0.0
        max_right_finger_error = 0.0
        last_bottle_pos = initial_bottle_pos
        last_bottle_quat = initial_bottle_quat
        last_bottle_pose_error = initial_bottle_error
        frame_sleep = 1.0 / float(args.fps)
        try:
            while True:
                for frame_index, qpos_frame in enumerate(qpos):
                    expected_left, expected_right = _set_articulation_qpos(left, right, left_indices, right_indices, qpos_frame)
                    app.update()
                    if args.legacy_bottle_proxy:
                        last_bottle_pos, last_bottle_quat, last_bottle_pose_error = _try_update_bottle_pose(
                            bottle_translate_op,
                            bottle_orient_op,
                            Gf,
                            left,
                            local_offset,
                            last_bottle_pos,
                            last_bottle_quat,
                        )
                    elif args.bottle_motion_mode == "held_by_gripper":
                        last_bottle_pos, last_bottle_quat, last_bottle_pose_error = _try_update_bottle_grasp_pose(
                            bottle_translate_op,
                            bottle_orient_op,
                            Gf,
                            left,
                            np.asarray(grasp_info["t_object_gripper"], dtype=np.float64),
                            last_bottle_pos,
                            last_bottle_quat,
                        )
                    else:
                        last_bottle_pose_error = None
                    app.update()
                    actual_left = np.asarray(left.get_joint_positions(joint_indices=left_indices), dtype=np.float64)
                    actual_right = np.asarray(right.get_joint_positions(joint_indices=right_indices), dtype=np.float64)
                    max_left_arm_error = max(max_left_arm_error, float(np.max(np.abs(actual_left[:6] - expected_left[:6]))))
                    max_right_arm_error = max(max_right_arm_error, float(np.max(np.abs(actual_right[:6] - expected_right[:6]))))
                    max_left_finger_error = max(max_left_finger_error, float(np.max(np.abs(actual_left[6:] - expected_left[6:]))))
                    max_right_finger_error = max(max_right_finger_error, float(np.max(np.abs(actual_right[6:] - expected_right[6:]))))
                    if not args.headless:
                        time.sleep(frame_sleep)
                    if frame_index % 100 == 0:
                        status.update(
                            {
                                "status": "REPLAYING",
                                "current_frame": int(frame_index),
                                "max_left_arm_readback_error": max_left_arm_error,
                                "max_right_arm_readback_error": max_right_arm_error,
                                "max_left_finger_readback_error": max_left_finger_error,
                                "max_right_finger_readback_error": max_right_finger_error,
                                "last_bottle_position": last_bottle_pos,
                                "last_bottle_pose_error": last_bottle_pose_error,
                            }
                        )
                        output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                        print(f"replay frame {frame_index}/{len(qpos)}", flush=True)
                status.update(
                    {
                        "status": "LOOPING" if args.loop else "DONE",
                        "current_frame": int(len(qpos) - 1),
                        "max_left_arm_readback_error": max_left_arm_error,
                        "max_right_arm_readback_error": max_right_arm_error,
                        "max_left_finger_readback_error": max_left_finger_error,
                        "max_right_finger_readback_error": max_right_finger_error,
                        "last_bottle_position": last_bottle_pos,
                        "last_bottle_pose_error": last_bottle_pose_error,
                        "completed_frames": int(len(qpos)),
                    }
                )
                output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
                if not args.loop:
                    break
        except BaseException as exc:
            status.update(
                {
                    "status": "ERROR",
                    "exception_type": type(exc).__name__,
                    "exception": repr(exc),
                    "current_frame": int(status.get("current_frame", -1)),
                }
            )
            output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
            print(f"Replay loop failed: {type(exc).__name__}: {exc!r}", flush=True)
            raise
        if not args.headless:
            while True:
                app.update()
                time.sleep(0.05)
        return 0
    except BaseException as exc:
        write_status(
            {
                "status": "ERROR",
                "phase": phase,
                "episode": str(episode),
                "stage_usd": str(stage_usd),
                "frames": int(len(qpos)),
                "bottle_usd": str(bottle_usd),
                "grasp_yaml": str(grasp_yaml),
                "grasp_name": args.grasp_name,
                "bottle_motion_mode": "legacy_bottle_proxy" if args.legacy_bottle_proxy else args.bottle_motion_mode,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
            }
        )
        print(f"Replay setup failed at {phase}: {type(exc).__name__}: {exc!r}", flush=True)
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise SystemExit(1) from exc
        raise
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())

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
LEFT_ROOT = "/scene/left_base_link/left_base_link"
RIGHT_ROOT = "/scene/right_base_link/right_base_link"
LEFT_EE_BODY = "left_gripper_link"

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


def _update_bottle_pose(translate_op, orient_op, Gf, left, local_offset: np.ndarray) -> tuple[list[float], list[float]]:
    pos, quat_xyzw = _body_pose_xyzw(left, LEFT_EE_BODY)
    rot = _quat_xyzw_to_matrix(quat_xyzw)
    bottle_pos = pos + rot @ local_offset
    translate_op.Set(Gf.Vec3d(*[float(v) for v in bottle_pos]))
    orient_op.Set(Gf.Quatf(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
    return bottle_pos.tolist(), quat_xyzw.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop a full workcell Original ALOHA qpos replay with a bottle proxy on the left EE.")
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
    parser.add_argument("--show-office-shell", action="store_true")
    parser.add_argument("--output-json", default="reports/aloha_isaac_replay/visual_replay/latest_full_workcell_qpos_replay.json")
    args = parser.parse_args()

    episode = Path(args.episode).expanduser().resolve()
    stage_usd = Path(args.stage_usd).expanduser()
    if not stage_usd.is_absolute():
        stage_usd = (Path.cwd() / stage_usd).resolve()
    qpos = _load_qpos(episode, args.max_frames)

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
        import isaacsim.core.utils.stage as stage_utils
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from pxr import Gf, UsdGeom, UsdLux

        World.clear_instance()
        if not stage_utils.open_stage(str(stage_usd)):
            raise RuntimeError(f"Failed to open stage: {stage_usd}")
        for _ in range(30):
            app.update()

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

        bottle_translate_op, bottle_orient_op = _create_bottle_proxy(stage, UsdGeom, Gf, "/World/ReplayBottleProxy")

        left = world.scene.add(SingleArticulation(prim_path=args.left_root, name="left_full_aloha"))
        right = world.scene.add(SingleArticulation(prim_path=args.right_root, name="right_full_aloha"))
        world.reset()

        left_indices = _resolve_indices(list(left.dof_names), LEFT_DOF_ORDER, "left")
        right_indices = _resolve_indices(list(right.dof_names), RIGHT_DOF_ORDER, "right")
        local_offset = np.asarray(args.bottle_local_offset, dtype=np.float64)

        _set_articulation_qpos(left, right, left_indices, right_indices, qpos[0])
        app.update()
        initial_bottle_pos, initial_bottle_quat = _update_bottle_pose(bottle_translate_op, bottle_orient_op, Gf, left, local_offset)
        active_camera = _set_active_viewport_camera(camera_path) if not args.headless else False
        window_move = _move_window_to_workspace(args.workspace) if not args.headless else {"attempted": False, "reason": "headless"}

        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "RUNNING",
            "mode": "full_workcell_qpos_replay_with_left_ee_bottle_proxy",
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
            "bottle_proxy": "/World/ReplayBottleProxy",
            "bottle_local_offset": local_offset.tolist(),
            "initial_bottle_position": initial_bottle_pos,
            "initial_bottle_quat_xyzw": initial_bottle_quat,
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
        frame_sleep = 1.0 / float(args.fps)
        try:
            while True:
                for frame_index, qpos_frame in enumerate(qpos):
                    expected_left, expected_right = _set_articulation_qpos(left, right, left_indices, right_indices, qpos_frame)
                    app.update()
                    last_bottle_pos, _ = _update_bottle_pose(bottle_translate_op, bottle_orient_op, Gf, left, local_offset)
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
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())

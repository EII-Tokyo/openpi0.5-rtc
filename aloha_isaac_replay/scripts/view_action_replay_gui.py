from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.controller_system_id.continuous_joints import nearest_equivalent_targets
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos


BODY_MARKER_COLORS = {
    "base": (0.35, 0.35, 0.35),
    "shoulder": (0.9, 0.2, 0.2),
    "upper_arm": (0.95, 0.55, 0.1),
    "forearm": (0.1, 0.6, 0.95),
    "wrist": (0.2, 0.9, 0.45),
    "ee": (0.95, 0.95, 0.15),
}


def _load_episode(path: Path, max_steps: int | None) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        qpos = h5["observations/qpos"][:]
        action = h5["action"][:]
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected observations/qpos shape (T, 14), got {qpos.shape}")
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Expected action shape (T, 14), got {action.shape}")
    steps = min(len(action), len(qpos) - 1)
    if max_steps is not None:
        steps = min(steps, max_steps)
    return qpos[: steps + 1], action[:steps]


def _side_name(logical_name: str, side: str) -> str:
    prefix = f"{side}/"
    if not logical_name.startswith(prefix):
        raise ValueError(f"Expected {logical_name!r} to start with {prefix!r}")
    return logical_name[len(prefix) :]


def _resolve_indices(actual_dof_names: list[str], mapped_names: list[str], side: str) -> list[int]:
    stripped = [_side_name(name, side) for name in mapped_names]
    missing = [name for name in stripped if name not in actual_dof_names]
    if missing:
        raise ValueError(f"{side} missing DOFs {missing}; actual={actual_dof_names}")
    indices = [actual_dof_names.index(name) for name in stripped]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{side} duplicate DOF indices: {indices}")
    return indices


def _arm_values(frame_14d: np.ndarray, mapping: dict, side: str) -> tuple[np.ndarray, list[str]]:
    targets = arm_only_targets_from_standard_qpos(frame_14d, mapping)
    values = [target.value for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    names = [target.isaac_dof_name for target in targets if target.isaac_dof_name.startswith(f"{side}/")]
    return np.asarray(values, dtype=np.float64), names


def _apply_base_offsets(stage, usd_geom, axis: str, separation: float) -> dict[str, object]:
    axis = axis.upper()
    if axis not in {"X", "Y"}:
        raise ValueError(f"base separation axis must be X or Y, got {axis!r}")
    offsets: dict[str, tuple[float, float, float]] = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        prim = stage.GetPrimAtPath(f"/World/{side}")
        if not prim.IsValid():
            raise RuntimeError(f"Missing base prim /World/{side}")
        xyz = (sign * separation / 2.0, 0.0, 0.0) if axis == "X" else (0.0, sign * separation / 2.0, 0.0)
        usd_geom.Xformable(prim).AddTranslateOp().Set(xyz)
        offsets[side] = xyz
    return {"axis": axis, "separation": separation, "offsets": offsets}


def _classify_body(name: str) -> str:
    lower = name.lower()
    if "ee" in lower or "gripper" in lower:
        return "ee"
    if "wrist" in lower:
        return "wrist"
    if "forearm" in lower or "elbow" in lower:
        return "forearm"
    if "upper_arm" in lower or "upperarm" in lower:
        return "upper_arm"
    if "shoulder" in lower:
        return "shoulder"
    return "base"


def _add_link_markers(stage, usd_geom, side: str, body_names: list[str], radius: float) -> list[str]:
    marker_paths: list[str] = []
    for body_name in body_names:
        body_path = f"/World/{side}/root_joint/{body_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            continue
        marker_path = f"{body_path}/codex_visual_marker"
        sphere = usd_geom.Sphere.Define(stage, marker_path)
        sphere.CreateRadiusAttr(radius)
        color = BODY_MARKER_COLORS[_classify_body(body_name)]
        sphere.GetDisplayColorAttr().Set([color])
        marker_paths.append(marker_path)
    return marker_paths


def _set_pose_from_qpos(left, right, left_indices, right_indices, qpos_frame: np.ndarray, mapping: dict) -> None:
    left_values, _ = _arm_values(qpos_frame, mapping, "left")
    right_values, _ = _arm_values(qpos_frame, mapping, "right")
    left.set_joint_positions(left_values, joint_indices=np.asarray(left_indices, dtype=np.int64))
    right.set_joint_positions(right_values, joint_indices=np.asarray(right_indices, dtype=np.int64))
    left.set_joint_velocities(np.zeros_like(left_values), joint_indices=np.asarray(left_indices, dtype=np.int64))
    right.set_joint_velocities(np.zeros_like(right_values), joint_indices=np.asarray(right_indices, dtype=np.int64))


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a GUI Isaac Sim window and loop a validated Original ALOHA action replay.")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mapping", default="configs/aloha/original_stationary_aloha_mapping.yaml")
    parser.add_argument("--left-usd", default="assets/isaac/original_stationary_aloha_arm_only/generated/vx300s_left.usd")
    parser.add_argument("--right-usd", default="assets/isaac/original_stationary_aloha_arm_only/generated/vx300s_right.usd")
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--base-separation", type=float, default=0.5)
    parser.add_argument("--base-separation-axis", choices=("X", "Y"), default="Y")
    parser.add_argument("--marker-radius", type=float, default=0.025)
    parser.add_argument("--pause-at-end-sec", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true", help="Loop the replay until the window is closed.")
    parser.add_argument("--output-json", default="reports/aloha_isaac_replay/gui_replay/latest_gui_replay.json")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "create_new_stage": False,
            "disable_viewport_updates": False,
            "width": 1280,
            "height": 720,
            "window_width": 1280,
            "window_height": 720,
            "multi_gpu": False,
            "sync_loads": True,
            "limit_cpu_threads": 12,
        }
    )
    try:
        import isaacsim.core.utils.stage as stage_utils
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, UsdGeom, UsdLux

        qpos, action = _load_episode(Path(args.episode), args.max_steps)
        mapping = load_mapping(args.mapping)

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        world.set_simulation_dt(physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
        stage = omni.usd.get_context().get_stage()
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        base_offsets = _apply_base_offsets(stage, UsdGeom, args.base_separation_axis, args.base_separation)

        light = UsdLux.DistantLight.Define(stage, "/World/codex_replay_light")
        light.CreateIntensityAttr(450.0)

        camera = UsdGeom.Camera.Define(stage, "/World/codex_replay_camera")
        camera.AddTranslateOp().Set(Gf.Vec3d(1.2, -1.8, 1.0))
        camera.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 36.0))
        camera.CreateFocalLengthAttr(20.0)

        left = world.scene.add(SingleArticulation(prim_path="/World/left/root_joint/root_joint", name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path="/World/right/root_joint/root_joint", name="right_vx300s"))
        world.reset()

        left_init, left_names = _arm_values(qpos[0], mapping, "left")
        right_init, right_names = _arm_values(qpos[0], mapping, "right")
        left_indices = _resolve_indices(list(left.dof_names), left_names, "left")
        right_indices = _resolve_indices(list(right.dof_names), right_names, "right")
        _set_pose_from_qpos(left, right, left_indices, right_indices, qpos[0], mapping)

        left_markers = _add_link_markers(stage, UsdGeom, "left", list(left._articulation_view.body_names), args.marker_radius)
        right_markers = _add_link_markers(stage, UsdGeom, "right", list(right._articulation_view.body_names), args.marker_radius)
        app.update()

        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "episode": str(Path(args.episode).resolve()),
                    "left_usd": str(Path(args.left_usd).resolve()),
                    "right_usd": str(Path(args.right_usd).resolve()),
                    "steps": int(action.shape[0]),
                    "mode": "gui_action_replay_arm_only_with_link_markers",
                    "base_offsets": base_offsets,
                    "left_markers": left_markers,
                    "right_markers": right_markers,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(f"GUI replay ready: {output}")

        left_idx_array = np.asarray(left_indices, dtype=np.int64)
        right_idx_array = np.asarray(right_indices, dtype=np.int64)
        while True:
            _set_pose_from_qpos(left, right, left_indices, right_indices, qpos[0], mapping)
            world.step(render=True)
            for action_frame in action:
                left_target, _ = _arm_values(action_frame, mapping, "left")
                right_target, _ = _arm_values(action_frame, mapping, "right")
                sim_pre_arm = np.concatenate(
                    [
                        left.get_joint_positions(joint_indices=left_idx_array),
                        right.get_joint_positions(joint_indices=right_idx_array),
                    ]
                ).astype(np.float64)
                raw_target_arm = np.concatenate([left_target, right_target]).astype(np.float64)
                nearest_target_arm, _ = nearest_equivalent_targets(raw_target_arm, sim_pre_arm, ARM_ONLY_NAMES)
                left.apply_action(ArticulationAction(joint_positions=nearest_target_arm[:6], joint_indices=left_idx_array))
                right.apply_action(ArticulationAction(joint_positions=nearest_target_arm[6:], joint_indices=right_idx_array))
                for _ in range(args.steps_per_action):
                    world.step(render=True)
                time.sleep(args.physics_dt)
            if args.pause_at_end_sec > 0:
                deadline = time.monotonic() + args.pause_at_end_sec
                while time.monotonic() < deadline:
                    app.update()
                    time.sleep(0.02)
            if not args.loop:
                break
        while True:
            app.update()
            time.sleep(0.05)
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())

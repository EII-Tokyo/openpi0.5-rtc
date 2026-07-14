from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from examples.aloha_isaac.scripts.apply_aloha_initial_pose import (
    REAL_RUNTIME_RESET_POSE,
    REAL_RUNTIME_SLEEP_POSE,
    split_real_start_pose_for_isaac_articulations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_LEFT_ARTICULATION_ROOT = "/scene/left_base_link/left_base_link"
DEFAULT_RIGHT_ARTICULATION_ROOT = "/scene/right_base_link/right_base_link"
POSES = {
    "home": REAL_RUNTIME_RESET_POSE,
    "sleep": REAL_RUNTIME_SLEEP_POSE,
}


class AlohaPoseController:
    def __init__(self, left: Any, right: Any, initial_pose_name: str = "home") -> None:
        self.left = left
        self.right = right
        self.current_pose_name = initial_pose_name

    def apply(self, pose_name: str) -> bool:
        pose = POSES[pose_name]
        if not _set_pose_on_initialized_articulations(self.left, self.right, pose):
            return False
        self.current_pose_name = pose_name
        return True

    def apply_home(self) -> bool:
        return self.apply("home")

    def apply_sleep(self) -> bool:
        return self.apply("sleep")

    def toggle(self) -> bool:
        return self.apply_sleep() if self.current_pose_name == "home" else self.apply_home()

    def reapply_current_pose(self) -> bool:
        return self.apply(self.current_pose_name)


def _set_real_start_pose_on_initialized_articulations(left: Any, right: Any) -> bool:
    return _set_pose_on_initialized_articulations(left, right, REAL_RUNTIME_RESET_POSE)


def _set_pose_on_initialized_articulations(left: Any, right: Any, pose: tuple[float, ...]) -> bool:
    if not getattr(left, "handles_initialized", False) or not getattr(right, "handles_initialized", False):
        return False

    left_pose, right_pose = split_real_start_pose_for_isaac_articulations(pose)
    _apply_pose_to_articulation(left, left_pose)
    _apply_pose_to_articulation(right, right_pose)
    return True


def _apply_pose_to_articulation(articulation: Any, pose: tuple[float, ...]) -> None:
    """Apply a pose both as current state and as the physics drive target."""
    pose_array = np.asarray(pose, dtype=np.float32)
    zero_array = np.zeros_like(pose_array)

    if hasattr(articulation, "set_joints_default_state"):
        articulation.set_joints_default_state(positions=pose_array, velocities=zero_array)
    articulation.set_joint_positions(pose_array)
    articulation.set_joint_velocities(zero_array)

    if hasattr(articulation, "apply_action"):
        from isaacsim.core.utils.types import ArticulationAction

        articulation.apply_action(ArticulationAction(joint_positions=pose_array, joint_velocities=zero_array))


def _build_pose_control_window(controller: AlohaPoseController):
    import omni.ui as ui

    window = ui.Window("ALOHA Pose Controls", width=320, height=0, visible=True)
    with window.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Simulation-only pose controls", word_wrap=True)
            status = ui.Label(f"Current pose: {controller.current_pose_name}", height=24)

            def apply_and_update(pose_name: str) -> None:
                ok = controller.apply(pose_name)
                status.text = f"Current pose: {controller.current_pose_name}" if ok else "Articulation not ready"

            def toggle_and_update() -> None:
                ok = controller.toggle()
                status.text = f"Current pose: {controller.current_pose_name}" if ok else "Articulation not ready"

            with ui.HStack(spacing=6, height=32):
                ui.Button("Home", clicked_fn=lambda: apply_and_update("home"), tooltip="Apply the real runtime home pose.")
                ui.Button("Sleep", clicked_fn=lambda: apply_and_update("sleep"), tooltip="Apply the real runtime sleep pose.")
            ui.Button("Toggle Home / Sleep", clicked_fn=toggle_and_update, height=32)
    return window, status


def _assert_articulation_pose(articulation: Any, expected_pose: tuple[float, ...], label: str, tolerance: float = 1e-3) -> None:
    import numpy as np

    actual = np.asarray(articulation.get_joint_positions(), dtype=float)
    expected = np.asarray(expected_pose, dtype=float)
    if actual.shape != expected.shape:
        raise RuntimeError(f"{label}: joint shape mismatch actual={actual.shape} expected={expected.shape}")
    max_error = float(np.max(np.abs(actual - expected)))
    print(f"{label}: max_abs_error={max_error:.6g} actual={actual.tolist()}")
    if max_error > tolerance:
        raise RuntimeError(f"{label}: pose error {max_error:.6g} exceeds tolerance {tolerance}")


def _run_pose_self_test(world: Any, left: Any, right: Any) -> None:
    controller = AlohaPoseController(left, right)
    home_left, home_right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_RESET_POSE)
    sleep_left, sleep_right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_SLEEP_POSE)

    controller.apply_home()
    world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left")
    _assert_articulation_pose(right, home_right, "home_right")

    controller.apply_sleep()
    world.step(render=False)
    _assert_articulation_pose(left, sleep_left, "sleep_left")
    _assert_articulation_pose(right, sleep_right, "sleep_right")

    controller.apply_home()
    world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left_after_toggle")
    _assert_articulation_pose(right, home_right, "home_right_after_toggle")

    world.play()
    controller.apply_sleep()
    for _ in range(5):
        world.step(render=False)
    _assert_articulation_pose(left, sleep_left, "sleep_left_during_playback")
    _assert_articulation_pose(right, sleep_right, "sleep_right_during_playback")

    controller.apply_home()
    for _ in range(5):
        world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left_after_playback_toggle")
    _assert_articulation_pose(right, home_right, "home_right_after_playback_toggle")
    world.pause()
    print("ALOHA pose self-test passed.")


def _apply_real_start_pose_to_articulations() -> tuple[object, object, object]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    world = World(stage_units_in_meters=1.0)
    left = SingleArticulation(DEFAULT_LEFT_ARTICULATION_ROOT, name="aloha_left")
    right = SingleArticulation(DEFAULT_RIGHT_ARTICULATION_ROOT, name="aloha_right")
    world.scene.add(left)
    world.scene.add(right)

    left_pose, right_pose = split_real_start_pose_for_isaac_articulations()
    zeros = [0.0] * len(left_pose)
    left.set_joints_default_state(positions=left_pose, velocities=zeros)
    right.set_joints_default_state(positions=right_pose, velocities=zeros)

    world.reset()
    world.pause()
    if not _set_real_start_pose_on_initialized_articulations(left, right):
        raise RuntimeError("ALOHA articulations were not initialized after World.reset().")
    return world, left, right


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the generated ALOHA Isaac workcell USD in Isaac Sim GUI.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--headless", action="store_true", help="Run Isaac without opening the GUI window.")
    parser.add_argument(
        "--self-test-poses",
        action="store_true",
        help="Initialize ALOHA, apply home/sleep/home, read articulation joints, and exit.",
    )
    parser.add_argument(
        "--no-real-start-pose",
        action="store_true",
        help="Open the USD without forcing the imported ALOHA articulations to the real START_ARM_POSE.",
    )
    args = parser.parse_args()

    usd_path = args.usd.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage does not exist: {usd_path}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": bool(args.headless), "window_title": "Isaac Sim - ALOHA Workcell"})
    try:
        import omni.kit.app
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac failed to open stage: {usd_path}")
        for _ in range(5):
            app.update()

        articulations = None
        pose_controls = None
        if not args.no_real_start_pose:
            articulations = _apply_real_start_pose_to_articulations()
            _, left, right = articulations
            pose_controller = AlohaPoseController(left, right)
            if args.self_test_poses:
                world, _, _ = articulations
                _run_pose_self_test(world, left, right)
                return
            pose_controls = _build_pose_control_window(pose_controller)
            print("Applied real home pose to ALOHA articulations.")

        kit_app = omni.kit.app.get_app()
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        while kit_app.is_running():
            app.update()
            if articulations is not None and not timeline.is_playing():
                pose_controller.reapply_current_pose()
                if pose_controls is not None:
                    _, status = pose_controls
                    status.text = f"Current pose: {pose_controller.current_pose_name}"
    finally:
        app.close()


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()

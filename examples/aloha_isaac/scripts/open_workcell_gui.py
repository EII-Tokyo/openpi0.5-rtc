from __future__ import annotations

import argparse
import os
import time
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
POSE_CONTROL_HZ = 50.0
POSE_CONTROL_DT_S = 1.0 / POSE_CONTROL_HZ


class AlohaPoseController:
    def __init__(
        self,
        left: Any,
        right: Any,
        initial_pose_name: str = "home",
        world: Any | None = None,
        transition_duration_s: float = 1.8,
    ) -> None:
        self.left = left
        self.right = right
        self.current_pose_name = initial_pose_name
        self.world = world
        self.transition_duration_s = max(float(transition_duration_s), 0.05)
        self._viewport_refresh_pending = False
        self._transition_target_pose_name: str | None = None
        self._transition_elapsed_s = 0.0
        self._transition_start_left: np.ndarray | None = None
        self._transition_start_right: np.ndarray | None = None
        self._transition_target_left: np.ndarray | None = None
        self._transition_target_right: np.ndarray | None = None

    @property
    def is_transitioning(self) -> bool:
        return self._transition_target_pose_name is not None

    def apply(self, pose_name: str, refresh: bool = True, animate: bool = True) -> bool:
        pose = POSES[pose_name]
        if not _articulations_are_initialized(self.left, self.right):
            return False
        if animate:
            self._start_transition(pose_name, pose)
        else:
            self._clear_transition()
            _set_pose_on_initialized_articulations(self.left, self.right, pose)
            self.current_pose_name = pose_name
        if refresh:
            self.request_viewport_refresh()
        return True

    def apply_home(self, animate: bool = True) -> bool:
        return self.apply("home", animate=animate)

    def apply_sleep(self, animate: bool = True) -> bool:
        return self.apply("sleep", animate=animate)

    def toggle(self) -> bool:
        return self.apply_sleep() if self.current_pose_name == "home" else self.apply_home()

    def reapply_current_pose(self) -> bool:
        if self.is_transitioning:
            return True
        return self.apply(self.current_pose_name, refresh=False, animate=False)

    def _start_transition(self, pose_name: str, pose: tuple[float, ...]) -> None:
        left_target, right_target = split_real_start_pose_for_isaac_articulations(pose)
        self._transition_target_pose_name = pose_name
        self._transition_elapsed_s = 0.0
        self._transition_start_left = _current_or_named_articulation_pose(
            self.left, POSES[self.current_pose_name], side="left"
        )
        self._transition_start_right = _current_or_named_articulation_pose(
            self.right, POSES[self.current_pose_name], side="right"
        )
        self._transition_target_left = np.asarray(left_target, dtype=np.float32)
        self._transition_target_right = np.asarray(right_target, dtype=np.float32)

    def _clear_transition(self) -> None:
        self._transition_target_pose_name = None
        self._transition_elapsed_s = 0.0
        self._transition_start_left = None
        self._transition_start_right = None
        self._transition_target_left = None
        self._transition_target_right = None

    def update_transition(self, dt_s: float) -> bool:
        if not self.is_transitioning:
            return False
        if (
            self._transition_start_left is None
            or self._transition_start_right is None
            or self._transition_target_left is None
            or self._transition_target_right is None
            or self._transition_target_pose_name is None
        ):
            self._clear_transition()
            return False

        self._transition_elapsed_s = min(self.transition_duration_s, self._transition_elapsed_s + max(float(dt_s), 0.0))
        alpha = _smoothstep(self._transition_elapsed_s / self.transition_duration_s)
        left_pose = _interpolate_pose(self._transition_start_left, self._transition_target_left, alpha)
        right_pose = _interpolate_pose(self._transition_start_right, self._transition_target_right, alpha)
        _apply_pose_to_articulation(self.left, left_pose)
        _apply_pose_to_articulation(self.right, right_pose)

        if self._transition_elapsed_s >= self.transition_duration_s:
            self.current_pose_name = self._transition_target_pose_name
            self._clear_transition()
        return True

    def request_viewport_refresh(self) -> None:
        self._viewport_refresh_pending = True

    def consume_viewport_refresh_request(self) -> bool:
        if not self._viewport_refresh_pending:
            return False
        self._viewport_refresh_pending = False
        return True

    def refresh_viewport(self) -> None:
        if self.world is None:
            return
        # Updating articulation state while the timeline is paused changes the
        # physics view, but the visible link transforms may remain stale. Step a
        # few rendered frames while the timeline is playing, then pause again so
        # the control buttons still behave like discrete pose commands.
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        was_playing = timeline.is_playing()
        if not was_playing:
            self.world.play()
        for _ in range(3):
            self.world.step(render=True)
        if not was_playing:
            self.world.pause()


def _max_pose_error(articulation: Any, expected_pose: tuple[float, ...]) -> float | None:
    if not hasattr(articulation, "get_joint_positions"):
        return None
    actual = np.asarray(articulation.get_joint_positions(), dtype=float)
    expected = np.asarray(expected_pose, dtype=float)
    if actual.shape != expected.shape:
        return None
    return float(np.max(np.abs(actual - expected)))


def _pose_status_message(controller: AlohaPoseController, pose_name: str, ok: bool) -> str:
    if not ok:
        return "Articulation not ready"
    if controller.is_transitioning:
        return f"Moving to: {pose_name}"
    pose = POSES[pose_name]
    left_pose, right_pose = split_real_start_pose_for_isaac_articulations(pose)
    left_error = _max_pose_error(controller.left, left_pose)
    right_error = _max_pose_error(controller.right, right_pose)
    if left_error is None or right_error is None:
        return f"Current pose: {controller.current_pose_name}"
    return f"Current pose: {controller.current_pose_name} | max error L={left_error:.4g} R={right_error:.4g}"


def _set_real_start_pose_on_initialized_articulations(left: Any, right: Any) -> bool:
    return _set_pose_on_initialized_articulations(left, right, REAL_RUNTIME_RESET_POSE)


def _articulations_are_initialized(left: Any, right: Any) -> bool:
    return bool(getattr(left, "handles_initialized", False) and getattr(right, "handles_initialized", False))


def _set_pose_on_initialized_articulations(left: Any, right: Any, pose: tuple[float, ...]) -> bool:
    if not _articulations_are_initialized(left, right):
        return False

    left_pose, right_pose = split_real_start_pose_for_isaac_articulations(pose)
    _apply_pose_to_articulation(left, left_pose)
    _apply_pose_to_articulation(right, right_pose)
    return True


def _current_or_named_articulation_pose(articulation: Any, named_pose: tuple[float, ...], side: str) -> np.ndarray:
    if hasattr(articulation, "get_joint_positions"):
        try:
            current = np.asarray(articulation.get_joint_positions(), dtype=np.float32)
            if current.size:
                return current
        except Exception:
            pass
    left_pose, right_pose = split_real_start_pose_for_isaac_articulations(named_pose)
    return np.asarray(left_pose if side == "left" else right_pose, dtype=np.float32)


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _interpolate_pose(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return start + (target - start) * float(alpha)


def _apply_pose_to_articulation(articulation: Any, pose: tuple[float, ...] | np.ndarray) -> None:
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

    window = ui.Window("ALOHA Pose Controls", width=420, height=150, visible=True, auto_resize=True)
    with window.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Simulation-only pose controls", word_wrap=True)
            status = ui.Label(f"Current pose: {controller.current_pose_name}", height=40, word_wrap=True)

            def apply_and_update(pose_name: str) -> None:
                ok = controller.apply(pose_name)
                status.text = _pose_status_message(controller, pose_name, ok)
                print(f"[ALOHA Pose Controls] button={pose_name} ok={ok} {status.text}", flush=True)

            def toggle_and_update() -> None:
                previous_pose = controller.current_pose_name
                target_pose = "sleep" if controller.current_pose_name == "home" else "home"
                ok = controller.toggle()
                status.text = _pose_status_message(controller, target_pose, ok)
                print(
                    f"[ALOHA Pose Controls] button=toggle from={previous_pose} ok={ok} {status.text}",
                    flush=True,
                )

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
    controller = AlohaPoseController(left, right, world=world)
    home_left, home_right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_RESET_POSE)
    sleep_left, sleep_right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_SLEEP_POSE)

    controller.apply_home(animate=False)
    world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left")
    _assert_articulation_pose(right, home_right, "home_right")

    controller.apply_sleep(animate=False)
    world.step(render=False)
    _assert_articulation_pose(left, sleep_left, "sleep_left")
    _assert_articulation_pose(right, sleep_right, "sleep_right")

    controller.apply_home(animate=False)
    world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left_after_toggle")
    _assert_articulation_pose(right, home_right, "home_right_after_toggle")

    world.play()
    controller.apply_sleep(animate=False)
    for _ in range(5):
        world.step(render=False)
    _assert_articulation_pose(left, sleep_left, "sleep_left_during_playback")
    _assert_articulation_pose(right, sleep_right, "sleep_right_during_playback")

    controller.apply_home(animate=False)
    for _ in range(5):
        world.step(render=False)
    _assert_articulation_pose(left, home_left, "home_left_after_playback_toggle")
    _assert_articulation_pose(right, home_right, "home_right_after_playback_toggle")

    controller.apply_sleep()
    controller.update_transition(controller.transition_duration_s / 2.0)
    world.step(render=False)
    if not controller.is_transitioning:
        raise RuntimeError("animated sleep transition finished too early")
    controller.update_transition(controller.transition_duration_s / 2.0)
    world.step(render=False)
    _assert_articulation_pose(left, sleep_left, "sleep_left_after_animated_transition")
    _assert_articulation_pose(right, sleep_right, "sleep_right_after_animated_transition")
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
            world, left, right = articulations
            pose_controller = AlohaPoseController(left, right, world=world)
            if args.self_test_poses:
                _run_pose_self_test(world, left, right)
                return
            pose_controls = _build_pose_control_window(pose_controller)
            print("Applied real home pose to ALOHA articulations.")

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        previous_time = time.monotonic()
        transition_accumulator_s = 0.0
        while app.is_running():
            app.update()
            now = time.monotonic()
            dt_s = min(now - previous_time, 0.1)
            previous_time = now
            if articulations is not None:
                if pose_controller.is_transitioning:
                    if not timeline.is_playing():
                        world.play()
                    transition_accumulator_s += dt_s
                    stepped = False
                    while transition_accumulator_s >= POSE_CONTROL_DT_S and pose_controller.is_transitioning:
                        pose_controller.update_transition(POSE_CONTROL_DT_S)
                        transition_accumulator_s -= POSE_CONTROL_DT_S
                        world.step(render=True)
                        stepped = True
                    if not stepped:
                        world.step(render=True)
                    if pose_controls is not None:
                        _, status = pose_controls
                        if pose_controller.is_transitioning:
                            status.text = f"Moving to: {pose_controller._transition_target_pose_name}"
                        else:
                            status.text = f"Current pose: {pose_controller.current_pose_name}"
                    if not pose_controller.is_transitioning:
                        transition_accumulator_s = 0.0
                        world.pause()
                    continue
                if not timeline.is_playing():
                    pose_controller.reapply_current_pose()
                if not timeline.is_playing() and pose_controller.consume_viewport_refresh_request():
                    pose_controller.refresh_viewport()
                if pose_controls is not None:
                    _, status = pose_controls
                    status.text = f"Current pose: {pose_controller.current_pose_name}"
    finally:
        app.close()


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()

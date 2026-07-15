from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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
STARTUP_VIEW_CAMERA_PATH = "/scene/StartupViewCamera"
# This pose matches the manually adjusted startup view captured on 2026-07-15:
# close perspective, both arms visible, table filling the viewport.
STARTUP_VIEW_CAMERA_POSITION = (-1.15, 2.75, 1.35)
STARTUP_VIEW_CAMERA_TARGET = (0.0, 0.0, 0.20)
STARTUP_VIEW_CAMERA_FOCAL_LENGTH = 32.0
STARTUP_VIEW_CAMERA_CLIPPING_RANGE = (0.01, 100.0)
STARTUP_WINDOW_SIZE = (1980, 1120)
POSE_CONTROL_WINDOW_POSITION = (1135, 760)
POSE_CONTROL_WINDOW_SIZE = (154, 190)
DEFAULT_STARTUP_WORKSPACE_NUMBER = 2
POSES = {
    "home": REAL_RUNTIME_RESET_POSE,
    "sleep": REAL_RUNTIME_SLEEP_POSE,
}
POSE_CONTROL_HZ = 50.0
POSE_CONTROL_DT_S = 1.0 / POSE_CONTROL_HZ


def _resolve_startup_usd_path(usd_path: Path, allow_noncanonical: bool = False) -> Path:
    """Resolve the startup USD while preventing accidental launch of stale ALOHA variants."""
    resolved = usd_path.resolve()
    canonical = DEFAULT_USD.resolve()
    if resolved != canonical and not allow_noncanonical:
        raise ValueError(
            "Refusing to open noncanonical ALOHA Isaac USD. "
            f"The only user-confirmed ALOHA Isaac startup USD is {canonical}. "
            "Pass --allow-noncanonical-usd only for an explicit experiment."
        )
    return resolved


def _workspace_index_from_number(workspace_number: int) -> int:
    if workspace_number < 1:
        raise ValueError(f"workspace number must be >= 1, got {workspace_number}")
    return workspace_number - 1


def _parse_xdotool_window_ids(stdout: str) -> list[str]:
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _move_current_process_window_to_workspace(
    workspace_number: int = DEFAULT_STARTUP_WORKSPACE_NUMBER,
    pid: int | None = None,
    attempts: int = 40,
    sleep_s: float = 0.25,
    runner=subprocess.run,
    sleep_fn=time.sleep,
) -> bool:
    """Move this Isaac GUI process window to a non-current workspace when X11 supports it."""
    if "DISPLAY" not in os.environ:
        print("Skipping Isaac workspace move: DISPLAY is not set.", flush=True)
        return False
    if shutil.which("xdotool") is None:
        print("Skipping Isaac workspace move: xdotool is not installed.", flush=True)
        return False

    target_index = _workspace_index_from_number(workspace_number)
    process_id = str(pid if pid is not None else os.getpid())

    try:
        desktops = runner(["xdotool", "get_num_desktops"], check=True, capture_output=True, text=True)
        desktop_count = int(desktops.stdout.strip())
        if desktop_count <= target_index:
            runner(["xdotool", "set_num_desktops", str(workspace_number)], check=True)
    except Exception as exc:
        print(f"Skipping Isaac workspace move: failed to inspect desktops: {exc}", flush=True)
        return False

    window_id = None
    for _ in range(max(1, attempts)):
        result = runner(["xdotool", "search", "--pid", process_id], check=False, capture_output=True, text=True)
        window_ids = _parse_xdotool_window_ids(result.stdout)
        if window_ids:
            window_id = window_ids[-1]
            break
        sleep_fn(sleep_s)

    if window_id is None:
        print(f"Skipping Isaac workspace move: no X11 window found for pid={process_id}.", flush=True)
        return False

    runner(["xdotool", "set_desktop_for_window", window_id, str(target_index)], check=True)
    actual_index = "unknown"
    for _ in range(20):
        try:
            actual = runner(["xdotool", "get_desktop_for_window", window_id], check=False, capture_output=True, text=True)
            actual_index = actual.stdout.strip()
        except Exception:
            actual_index = "unknown"
        if actual_index == str(target_index):
            break
        sleep_fn(0.1)
    print(
        f"Moved Isaac Sim window {window_id} for pid={process_id} to workspace {workspace_number} "
        f"(X11 desktop index {actual_index}).",
        flush=True,
    )
    return actual_index == str(target_index)


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


def _pose_control_window_kwargs() -> dict[str, float | bool]:
    return {
        "width": POSE_CONTROL_WINDOW_SIZE[0],
        "height": POSE_CONTROL_WINDOW_SIZE[1],
        "position_x": POSE_CONTROL_WINDOW_POSITION[0],
        "position_y": POSE_CONTROL_WINDOW_POSITION[1],
        "visible": True,
        "auto_resize": False,
    }


def _build_pose_control_window(controller: AlohaPoseController):
    import omni.ui as ui

    window = ui.Window("ALOHA Pose Controls", **_pose_control_window_kwargs())
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


def _configure_startup_view_camera() -> str | None:
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    camera_path = Sdf.Path(STARTUP_VIEW_CAMERA_PATH)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(float(STARTUP_VIEW_CAMERA_FOCAL_LENGTH))
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*STARTUP_VIEW_CAMERA_CLIPPING_RANGE))

    eye = Gf.Vec3d(*STARTUP_VIEW_CAMERA_POSITION)
    target = Gf.Vec3d(*STARTUP_VIEW_CAMERA_TARGET)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    camera_to_world = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(camera_to_world)
    return str(camera_path)


def _set_active_viewport_camera(camera_path: str | None) -> bool:
    if not camera_path:
        return False
    try:
        from pxr import Sdf
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.camera_path = Sdf.Path(camera_path)
        return True
    except Exception as exc:
        print(f"Failed to set startup viewport camera {camera_path}: {exc}", flush=True)
        return False


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
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help=f"USD stage to open. Default: {DEFAULT_USD}")
    parser.add_argument(
        "--allow-noncanonical-usd",
        action="store_true",
        help="Allow loading a USD other than the user-confirmed ALOHA startup stage. Use only for explicit experiments.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac without opening the GUI window.")
    parser.add_argument(
        "--startup-workspace",
        type=int,
        default=DEFAULT_STARTUP_WORKSPACE_NUMBER,
        help=(
            "Move the Isaac GUI window to this 1-based desktop/workspace after launch. "
            f"Default: {DEFAULT_STARTUP_WORKSPACE_NUMBER}."
        ),
    )
    parser.add_argument(
        "--no-move-to-startup-workspace",
        action="store_true",
        help="Keep the Isaac GUI window on the current desktop/workspace.",
    )
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

    usd_path = _resolve_startup_usd_path(args.usd, allow_noncanonical=args.allow_noncanonical_usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD stage does not exist: {usd_path}")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "window_title": "Isaac Sim - ALOHA Workcell",
            "window_width": STARTUP_WINDOW_SIZE[0],
            "window_height": STARTUP_WINDOW_SIZE[1],
        }
    )
    if not args.headless and not args.no_move_to_startup_workspace:
        _move_current_process_window_to_workspace(args.startup_workspace)
    try:
        import omni.kit.app
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac failed to open stage: {usd_path}")
        for _ in range(5):
            app.update()
        startup_camera_path = _configure_startup_view_camera()

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
        if _set_active_viewport_camera(startup_camera_path):
            print(f"Configured startup viewport camera: {startup_camera_path}", flush=True)
        for _ in range(5):
            app.update()

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

#!/usr/bin/env python3
"""Open the frozen ALOHA1 Stage at runtime-measured Sleep for GUI review."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.validate_aloha1_home_sleep_digital import ARTICULATION_PATHS
from tools.validate_aloha1_home_sleep_digital import EXPECTED_DOF_ORDER
from tools.validate_aloha1_home_sleep_digital import EXPECTED_RUNTIME
from tools.validate_aloha1_home_sleep_digital import POSITION_GATE_RAD
from tools.aloha1_mapping.gui_sleep_home_sleep_controller import GuiSleepHomeSleepController
from tools.aloha1_mapping.gui_sleep_home_sleep_controller import compose_arm_target
from tools.aloha1_mapping.gui_sleep_home_sleep_controller import build_gui_button_samples
from tools.aloha1_mapping.real_sync_bridge import INITIAL_POSE_GATE_RAD
from tools.aloha1_mapping.real_sync_bridge import build_remote_publisher_command
from tools.aloha1_mapping.real_sync_bridge import format_initial_pose_check
from tools.aloha1_mapping.real_sync_bridge import initial_pose_error_rad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
)
DEFAULT_STAGE_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
DEFAULT_MANIFEST = ROOT / "reports/aloha1_mapping/aloha1_runtime_measured_sleep_command_manifest.json"
DEFAULT_MANIFEST_SHA256 = "d48047eadc6a02664efb01cba3e0345b523bf64052791491bb237639f24dad3c"
DEFAULT_FINGER_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/"
    "configuration/finger_source_limits.usda"
)
DEFAULT_FINGER_SHA256 = "2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f"
DEFAULT_REPORT = ROOT / "reports/aloha1_mapping/aloha1_runtime_measured_sleep_gui_session.json"
FULL_EXPERIENCE_RELATIVE = Path(".venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit")
GUI_BUTTON_PHYSICS_HZ = 50
# Approximately three seconds, with a small margin for zero-velocity
# acceleration/deceleration at each endpoint.  The command stream remains 50 Hz.
GUI_BUTTON_MOVE_SECONDS = 3.5
REMOTE_MANIFEST_PATH = "/app/.codex/runtime/sleep_home_sleep_50hz_smooth_manifest.json"
REMOTE_RESULT_PATH = "/app/.codex/runtime/integrated_sleep_home_sleep_result.json"
REAL_START_DELAY_S = 4.0
DIGITAL_START_GUARD_S = 4.05
RIGHT_RUNTIME_INITIAL_REFERENCE_RAD = np.asarray(
    [0.0, -1.8, 1.55, 0.0, -1.57, 0.0],
    dtype=np.float32,
)
RIGHT_RUNTIME_SLEEP_SOURCE = "configs/aloha1_follower_right_runtime_sleep.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_full_experience() -> Path:
    """Resolve the installed Isaac Sim 5.1 full GUI experience."""

    candidates = []
    exp_path = os.environ.get("EXP_PATH")
    if exp_path:
        candidates.append(Path(exp_path) / "isaacsim.exp.full.kit")
    candidates.append(ROOT / FULL_EXPERIENCE_RELATIVE)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "isaacsim.exp.full.kit":
            return resolved
    raise FileNotFoundError("Isaac Sim 5.1 isaacsim.exp.full.kit was not found")


def _verify_hash(path: Path, expected: str, label: str) -> tuple[Path, str]:
    resolved = path.resolve(strict=True)
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return resolved, actual


def _read_remote_initial_arm_pose() -> list[float]:
    """Read follower_left arm state through ROS without creating a publisher."""

    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "192.168.1.103",
        "C=$(docker ps --format '{{.Names}}' | grep aloha_ros_nodes | head -1); "
        "docker exec \"$C\" bash -lc "
        "'source /opt/ros/noetic/setup.bash; timeout 5 rostopic echo -n 1 "
        "/puppet_left/joint_states'",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"read-only follower_left state failed: {completed.stderr[-500:]}")
    match = re.search(r"position:\s*\[([^\]]+)\]", completed.stdout)
    if match is None:
        raise RuntimeError("joint_states readback did not contain position")
    values = [float(value.strip()) for value in match.group(1).split(",")]
    if len(values) < 6:
        raise RuntimeError("joint_states readback contained fewer than six arm joints")
    return values[:6]


def load_verified_inputs(
    *,
    stage: Path,
    stage_sha256: str,
    manifest: Path,
    manifest_sha256: str,
    finger_limit_layer: Path,
    finger_limit_sha256: str,
) -> dict[str, Any]:
    """Verify all frozen files and the runtime-Sleep manifest contract."""

    stage_path, stage_hash = _verify_hash(stage, stage_sha256, "Stage")
    manifest_path, manifest_hash = _verify_hash(manifest, manifest_sha256, "Manifest")
    finger_path, finger_hash = _verify_hash(finger_limit_layer, finger_limit_sha256, "Finger layer")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("sequence_kind") != "SLEEP_HOME_SLEEP":
        raise ValueError("manifest is not the approved SLEEP_HOME_SLEEP sequence")
    if payload.get("initial_pose_label") != "runtime_measured_sleep":
        raise ValueError("manifest does not start at runtime_measured_sleep")
    if payload.get("terminal_pose_label") != "runtime_measured_sleep":
        raise ValueError("manifest does not end at runtime_measured_sleep")
    initial = payload.get("initial_arm_rad")
    if not isinstance(initial, list) or len(initial) != len(ARM_JOINT_ORDER):
        raise ValueError("manifest runtime Sleep must contain exactly six arm joints")
    if not np.isfinite(np.asarray(initial, dtype=np.float64)).all():
        raise ValueError("manifest runtime Sleep contains a non-finite value")
    return {
        "paths": {
            "stage": str(stage_path),
            "manifest": str(manifest_path),
            "finger_limit_layer": str(finger_path),
        },
        "hashes": {
            "stage": stage_hash,
            "manifest": manifest_hash,
            "finger_limit_layer": finger_hash,
        },
        "manifest": payload,
    }


def build_ready_report(
    *,
    inputs: dict[str, Any],
    runtime: dict[str, str],
    runtime_pid: int,
    window_id: str | None,
    workspace_number: int | None,
    workspace_move_passed: bool,
    active_workspace_before: int | None,
    active_workspace_after: int | None,
    timeline_paused: bool,
    target_arm_rad: list[float],
    readback_arm_rad: list[float],
    dof_order: list[str],
    stage_hash_after: str,
    session_layers: list[str],
) -> dict[str, Any]:
    """Build a fail-closed machine report for the paused review session."""

    target = np.asarray(target_arm_rad, dtype=np.float64)
    readback = np.asarray(readback_arm_rad, dtype=np.float64)
    maximum_error = float(np.max(np.abs(readback - target)))
    manifest = inputs["manifest"]
    compact_inputs = {
        "paths": dict(inputs["paths"]),
        "hashes": dict(inputs["hashes"]),
        "manifest": {
            "sequence_kind": manifest["sequence_kind"],
            "initial_pose_label": manifest["initial_pose_label"],
            "terminal_pose_label": manifest["terminal_pose_label"],
            "sample_count": manifest.get("sample_count"),
            "command_signature": manifest.get("command_signature"),
        },
    }
    gates = {
        "runtime_exact": runtime == EXPECTED_RUNTIME,
        "stage_hash_immutable": stage_hash_after == inputs["hashes"]["stage"],
        "runtime_sleep_readback": maximum_error <= POSITION_GATE_RAD,
        "dof_order": dof_order == EXPECTED_DOF_ORDER,
        "timeline_paused": bool(timeline_paused),
        "workspace_move_passed": bool(workspace_move_passed),
        "window_on_workspace_2": workspace_number == 2,
        "active_workspace_unchanged": active_workspace_before == active_workspace_after,
        "window_identified": bool(window_id),
    }
    ready = all(gates.values())
    return {
        "schema_version": 1,
        "status": "READY_FOR_USER_REVIEW" if ready else "FAIL_NOT_READY",
        "classification": "RUNTIME_MEASURED_SLEEP_PAUSED_GUI",
        "runtime": runtime,
        "runtime_pid": int(runtime_pid),
        "window_id": window_id,
        "workspace_number": workspace_number,
        "active_workspace_before": active_workspace_before,
        "active_workspace_after": active_workspace_after,
        "timeline_paused": bool(timeline_paused),
        "inputs": compact_inputs,
        "joint_order": list(ARM_JOINT_ORDER),
        "dof_order": dof_order,
        "target_arm_rad": target.tolist(),
        "readback_arm_rad": readback.tolist(),
        "maximum_sleep_error_rad": maximum_error,
        "position_gate_rad": POSITION_GATE_RAD,
        "session_layers": session_layers,
        "gates": gates,
        "real_robot_transport_constructed": False,
        "real_motion_commands": 0,
        "source_or_final_asset_modified": False,
        "candidate_promoted": False,
        "task8": "COMPLETE_WITH_NO_PROMOTION",
    }


def _runtime_versions(app: Any) -> dict[str, str]:
    import carb

    manager = app.get_extension_manager()
    physx_id = manager.get_enabled_extension_id("omni.physx")
    physx_record = manager.get_extension_dict(physx_id) if physx_id else {}
    return {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0],
        "physx": str(physx_record.get("package", {}).get("version", "")).split("+", maxsplit=1)[0],
    }


def _desktop_number() -> int | None:
    result = subprocess.run(["xdotool", "get_desktop"], check=False, capture_output=True, text=True)
    value = result.stdout.strip()
    return int(value) + 1 if result.returncode == 0 and value.isdigit() else None


def _window_state(pid: int) -> tuple[str | None, int | None]:
    found = subprocess.run(
        ["xdotool", "search", "--pid", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    window_ids = [line.strip() for line in found.stdout.splitlines() if line.strip()]
    if found.returncode != 0 or not window_ids:
        return None, None
    window_id = window_ids[-1]
    desktop = subprocess.run(
        ["xdotool", "get_desktop_for_window", window_id],
        check=False,
        capture_output=True,
        text=True,
    )
    value = desktop.stdout.strip()
    return window_id, (int(value) + 1 if desktop.returncode == 0 and value.isdigit() else None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--stage-sha256", default=DEFAULT_STAGE_SHA256)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--finger-limit-layer", type=Path, default=DEFAULT_FINGER_LAYER)
    parser.add_argument("--finger-limit-sha256", default=DEFAULT_FINGER_SHA256)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--startup-workspace", type=int, default=2)
    parser.add_argument("--run-digital-only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace, app: Any) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    from pxr import Sdf
    from pxr import Usd

    from examples.aloha_isaac.scripts.open_workcell_gui import _configure_startup_view_camera
    from examples.aloha_isaac.scripts.open_workcell_gui import _move_current_process_window_to_workspace
    from examples.aloha_isaac.scripts.open_workcell_gui import _set_active_viewport_camera
    from tools.validate_aloha1_home_sleep_digital import _apply_targets
    from tools.validate_aloha1_home_sleep_digital import _install_session_layers

    inputs = load_verified_inputs(
        stage=args.stage,
        stage_sha256=args.stage_sha256,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        finger_limit_layer=args.finger_limit_layer,
        finger_limit_sha256=args.finger_limit_sha256,
    )
    active_before = _desktop_number()
    moved = _move_current_process_window_to_workspace(args.startup_workspace)
    window_id, window_workspace = _window_state(os.getpid())

    if not open_stage(inputs["paths"]["stage"]):
        raise RuntimeError(f"failed to open frozen Stage: {inputs['paths']['stage']}")
    kit_app = omni.kit.app.get_app()
    for _ in range(20):
        kit_app.update()
    stage = omni.usd.get_context().get_stage()
    manifest = inputs["manifest"]
    button_samples = build_gui_button_samples(
        sleep=manifest["sleep_rad"],
        home=manifest["home_rad"],
        command_hz=GUI_BUTTON_PHYSICS_HZ,
        move_seconds=GUI_BUTTON_MOVE_SECONDS,
    )
    right_button_samples = build_gui_button_samples(
        sleep=RIGHT_RUNTIME_INITIAL_REFERENCE_RAD.astype(np.float64).tolist(),
        home=manifest["home_rad"],
        command_hz=GUI_BUTTON_PHYSICS_HZ,
        move_seconds=GUI_BUTTON_MOVE_SECONDS,
    )
    session = _install_session_layers(stage, Path(inputs["paths"]["finger_limit_layer"]), manifest)

    view_layer = Sdf.Layer.CreateAnonymous("aloha1_runtime_sleep_gui_view.usda")
    stage.GetSessionLayer().subLayerPaths.insert(0, view_layer.identifier)
    old_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(view_layer))
    camera_path = _configure_startup_view_camera()
    stage.SetEditTarget(old_target)
    if not _set_active_viewport_camera(camera_path):
        raise RuntimeError("failed to activate runtime Sleep review camera")

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / GUI_BUTTON_PHYSICS_HZ,
        rendering_dt=1.0 / GUI_BUTTON_PHYSICS_HZ,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, path in ARTICULATION_PATHS.items():
        articulation = SingleArticulation(
            prim_path=path,
            name=f"runtime_sleep_gui_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()

    left = articulations["follower_left"]
    right = articulations["follower_right"]
    if list(left.dof_names) != EXPECTED_DOF_ORDER:
        raise RuntimeError(f"unexpected follower_left DOF order: {left.dof_names}")
    sleep = np.asarray(manifest["initial_arm_rad"], dtype=np.float32)
    left_initial = np.asarray(left.get_joint_positions(), dtype=np.float32)
    right_initial = np.asarray(right.get_joint_positions(), dtype=np.float32)
    right_runtime_reference = right_initial.copy()
    right_runtime_reference[:6] = RIGHT_RUNTIME_INITIAL_REFERENCE_RAD
    left_target = left_initial.copy()
    left_target[:6] = sleep
    left.set_joint_positions(left_target)
    left.set_joint_velocities(np.zeros_like(left_target))
    _apply_targets(left, left_target[:8], range(8))
    right_initial_target = right_runtime_reference.copy()
    _apply_targets(right, right_initial_target[:8], range(8))
    world.play()
    for _ in range(30):
        _apply_targets(left, left_target[:8], range(8))
        _apply_targets(right, right_initial_target[:8], range(8))
        world.step(render=True)
    world.pause()
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    for _ in range(5):
        kit_app.update()

    active_after = _desktop_number()
    window_id, window_workspace = _window_state(os.getpid())
    readback = np.asarray(left.get_joint_positions(), dtype=np.float64)[:6].tolist()
    session_layers = list(stage.GetSessionLayer().subLayerPaths)
    session_layers.extend(value for key, value in session.items() if key.endswith("_layer") and isinstance(value, str))
    report = build_ready_report(
        inputs=inputs,
        runtime=_runtime_versions(kit_app),
        runtime_pid=os.getpid(),
        window_id=window_id,
        workspace_number=window_workspace,
        workspace_move_passed=moved,
        active_workspace_before=active_before,
        active_workspace_after=active_after,
        timeline_paused=not timeline.is_playing(),
        target_arm_rad=sleep.astype(np.float64).tolist(),
        readback_arm_rad=readback,
        dof_order=list(left.dof_names),
        stage_hash_after=_sha256(Path(inputs["paths"]["stage"])),
        session_layers=session_layers,
    )
    report["camera_path"] = camera_path
    report["stage_default_prim"] = str(stage.GetDefaultPrim().GetPath())
    report["stage_root_layer"] = str(Path(stage.GetRootLayer().realPath).resolve())
    report["stage_root_sublayers"] = list(stage.GetRootLayer().subLayerPaths)
    report["authored_references"] = [
        {"prim_path": str(prim.GetPath()), "references": str(prim.GetMetadata("references"))}
        for prim in stage.Traverse()
        if prim.HasAuthoredReferences()
    ]
    report["right_initial_reference"] = {
        "classification": "RIGHT_RUNTIME_LEGAL_SLEEP_CANDIDATE",
        "source_config": str((ROOT / RIGHT_RUNTIME_SLEEP_SOURCE).resolve()),
        "arm_q_rad": RIGHT_RUNTIME_INITIAL_REFERENCE_RAD.astype(np.float64).tolist(),
    }
    right_readback = np.asarray(right.get_joint_positions(), dtype=np.float64)[:6]
    right_error = float(np.max(np.abs(right_readback - RIGHT_RUNTIME_INITIAL_REFERENCE_RAD)))
    report["right_initial_reference"]["readback_arm_q_rad"] = right_readback.tolist()
    report["right_initial_reference"]["maximum_error_rad"] = right_error
    report["right_initial_reference"]["position_gate_rad"] = POSITION_GATE_RAD
    report["gates"]["right_runtime_sleep_readback"] = right_error <= POSITION_GATE_RAD
    if not report["gates"]["right_runtime_sleep_readback"]:
        report["status"] = "FAIL_NOT_READY"

    # The visible GUI is now the explicitly authorized single-source bridge.
    # It performs a read-only remote pose check before constructing the remote
    # publisher; no command is sent if the check fails or the dialog is closed.
    control_state: dict[str, Any] = {
        "requested": False,
        "index": 0,
        "status": "READY_REAL_POSE_CHECK_REQUIRED",
        "telemetry": [],
        "next_deadline": None,
        "started_monotonic": None,
        "real_launch_requested": False,
        "real_initial_pose": None,
        "real_initial_error_rad": None,
        "pending_start_deadline": None,
        "bridge_launch_monotonic": None,
        "digital_start_monotonic": None,
    }
    controller = GuiSleepHomeSleepController(real_armed=True)
    control_window = ui.Window("ALOHA1 Sleep/Home/Sleep Control", width=500, height=210, visible=True)

    def show_pose_dialog(message: str, *, can_confirm: bool, on_confirm: Any = None) -> None:
        dialog = ui.Window(
            "Initial Pose Check — follower_left",
            width=520,
            height=230,
            flags=ui.WINDOW_FLAGS_MODAL | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS,
            visible=True,
        )

        def close_dialog(*_args: Any) -> None:
            dialog.visible = False

        def confirm_dialog(*_args: Any) -> None:
            dialog.visible = False
            if on_confirm is not None:
                on_confirm()

        with dialog.frame:
            with ui.VStack(spacing=8, padding=12):
                ui.Label(message, word_wrap=True, height=0)
                with ui.HStack(height=32):
                    ui.Spacer()
                    if can_confirm:
                        ui.Button("Confirm synchronized real run", clicked_fn=confirm_dialog, width=220)
                    ui.Button("Cancel", clicked_fn=close_dialog, width=100)
                    ui.Spacer()

    def start_integrated_run(real_pose: list[float], error_rad: float) -> None:
        try:
            decision = controller.request_run(
                digital_at_sleep=float(np.max(np.abs(np.asarray(left.get_joint_positions(), dtype=np.float64)[:6] - sleep))) <= POSITION_GATE_RAD,
                real_ready=error_rad <= INITIAL_POSE_GATE_RAD,
            )
        except ValueError as exc:
            control_state["status"] = f"BLOCKED: {exc}"
            return
        if not decision.real_commands_allowed:
            control_state["status"] = f"BLOCKED: {decision.status}"
            return
        try:
            subprocess.Popen(
                build_remote_publisher_command(
                    manifest_path=REMOTE_MANIFEST_PATH,
                    output_path=REMOTE_RESULT_PATH,
                    start_delay_s=REAL_START_DELAY_S,
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            control_state["status"] = f"BLOCKED: remote publisher launch failed: {exc}"
            return
        control_state["real_launch_requested"] = True
        control_state["real_initial_pose"] = list(real_pose)
        control_state["real_initial_error_rad"] = float(error_rad)
        control_state["requested"] = False
        control_state["index"] = 0
        control_state["status"] = "WAITING_FOR_SHARED_START_BARRIER"
        control_state["bridge_launch_monotonic"] = time.monotonic()
        control_state["pending_start_deadline"] = (
            control_state["bridge_launch_monotonic"] + DIGITAL_START_GUARD_S
        )
        control_state["started_monotonic"] = None
        control_state["next_deadline"] = None
        # The GUI runner needs the Kit timeline playing for articulation
        # targets to advance.  While a run is active, ``world.step(render=True)``
        # is the sole update/physics call; the loop below deliberately skips a
        # second ``kit_app.update()`` so no extra unsynchronised step is added.
        timeline.pause()

    def request_integrated_run() -> None:
        try:
            digital_at_sleep = float(
                np.max(
                    np.abs(
                        np.asarray(left.get_joint_positions(), dtype=np.float64)[:6]
                        - sleep
                    )
                )
            ) <= POSITION_GATE_RAD
            digital_at_sleep = digital_at_sleep and (
                float(
                    np.max(
                        np.abs(
                            np.asarray(right.get_joint_positions(), dtype=np.float64)[:6]
                            - RIGHT_RUNTIME_INITIAL_REFERENCE_RAD
                        )
                    )
                )
                <= POSITION_GATE_RAD
            )
            if not digital_at_sleep:
                raise ValueError("digital articulation is not at Sleep")
            real_pose = _read_remote_initial_arm_pose()
            error_rad = initial_pose_error_rad(sleep, real_pose)
            message = format_initial_pose_check(
                max_error_rad=error_rad,
                gate_rad=INITIAL_POSE_GATE_RAD,
                real_position=real_pose,
            )
            show_pose_dialog(
                message,
                can_confirm=error_rad <= INITIAL_POSE_GATE_RAD,
                on_confirm=lambda: start_integrated_run(real_pose, error_rad),
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            control_state["status"] = f"BLOCKED_INITIAL_POSE_CHECK: {exc}"
            status_label.text = f"Status: {control_state['status']}"
            show_pose_dialog(
                "Initial pose check failed. No real publisher was created.\n\n"
                f"{exc}",
                can_confirm=False,
            )

    def start_digital_only_run() -> None:
        """Run one synchronized two-arm cycle without constructing ROS transport."""

        left_error = float(
            np.max(np.abs(np.asarray(left.get_joint_positions(), dtype=np.float64)[:6] - sleep))
        )
        right_error = float(
            np.max(
                np.abs(
                    np.asarray(right.get_joint_positions(), dtype=np.float64)[:6]
                    - RIGHT_RUNTIME_INITIAL_REFERENCE_RAD
                )
            )
        )
        if left_error > POSITION_GATE_RAD or right_error > POSITION_GATE_RAD:
            control_state["status"] = (
                f"BLOCKED_DIGITAL_INITIAL_POSE left={left_error:.6f} "
                f"right={right_error:.6f}"
            )
            status_label.text = f"Status: {control_state['status']}"
            return
        control_state["requested"] = True
        control_state["index"] = 0
        control_state["status"] = "DIGITAL_ONLY_RUNNING"
        control_state["real_launch_requested"] = False
        control_state["started_monotonic"] = time.monotonic()
        control_state["digital_start_monotonic"] = control_state["started_monotonic"]
        control_state["next_deadline"] = control_state["started_monotonic"]
        timeline.play()

    with control_window.frame:
        with ui.VStack(spacing=8, padding=10):
            ui.Label("Frozen runtime-measured Sleep manifest")
            ui.Button("Run DIGITAL ONLY — both followers Sleep → Home → Sleep", clicked_fn=start_digital_only_run, height=32)
            ui.Button("Check Initial Pose + Run Digital/Real Sleep -> Home -> Sleep", clicked_fn=request_integrated_run, height=32)
            ui.Label("ARM REAL ROBOT: AUTHORIZED FOR ONE follower_left CYCLE")
            status_label = ui.Label("Status: READY_REAL_POSE_CHECK_REQUIRED")
    if args.run_digital_only:
        # The explicit CLI mode is equivalent to pressing the Digital-only
        # button and cannot construct or launch the real ROS bridge.
        start_digital_only_run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "pid": report["runtime_pid"],
                "window_id": report["window_id"],
                "workspace": report["workspace_number"],
                "timeline_paused": report["timeline_paused"],
                "maximum_sleep_error_rad": report["maximum_sleep_error_rad"],
                "report": str(args.report.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if report["status"] != "READY_FOR_USER_REVIEW":
        return 2
    while app.is_running():
        if (
            control_state["pending_start_deadline"] is not None
            and time.monotonic() >= float(control_state["pending_start_deadline"])
        ):
            control_state["pending_start_deadline"] = None
            control_state["requested"] = True
            control_state["status"] = "DIGITAL_AND_REAL_RUNNING"
            control_state["started_monotonic"] = time.monotonic()
            control_state["digital_start_monotonic"] = control_state["started_monotonic"]
            control_state["next_deadline"] = control_state["started_monotonic"]
            timeline.play()
        if (
            control_state["requested"]
            and control_state["index"] < len(button_samples)
            and time.monotonic() >= float(control_state["next_deadline"])
        ):
            sample = button_samples[control_state["index"]]
            right_sample = right_button_samples[control_state["index"]]
            target = np.asarray(sample["q_rad"], dtype=np.float32)
            right_target = np.asarray(right_sample["q_rad"], dtype=np.float32)
            left_target_full = np.asarray(
                compose_arm_target(left.get_joint_positions(), target[:6]),
                dtype=np.float32,
            )
            _apply_targets(left, left_target_full, range(8))
            right_target_full = np.asarray(
                compose_arm_target(right.get_joint_positions(), right_target[:6]),
                dtype=np.float32,
            )
            _apply_targets(right, right_target_full, range(8))
            world.step(render=True)
            control_state["telemetry"].append(
                {
                    "sample_index": int(sample["index"]),
                    "cycle": int(sample["cycle"]),
                    "segment": str(sample["segment"]),
                    "target_arm_rad": target[:6].astype(np.float64).tolist(),
                    "readback_arm_rad": np.asarray(left.get_joint_positions(), dtype=np.float64)[:6].tolist(),
                    "right_target_arm_rad": right_target[:6].astype(np.float64).tolist(),
                    "right_readback_arm_rad": np.asarray(right.get_joint_positions(), dtype=np.float64)[:6].tolist(),
                }
            )
            control_state["index"] += 1
            control_state["next_deadline"] = float(control_state["next_deadline"]) + (1.0 / GUI_BUTTON_PHYSICS_HZ)
            status_label.text = f"Status: {control_state['status']} — {control_state['index']}/{len(button_samples)} @ {GUI_BUTTON_PHYSICS_HZ} Hz"
            if control_state["index"] >= len(button_samples):
                control_state["requested"] = False
                timeline.pause()
                control_state["status"] = "DIGITAL_AND_REAL_RUN_COMPLETE"
                status_label.text = "Status: DIGITAL_AND_REAL_RUN_COMPLETE"
                button_report = {
                    "schema_version": 1,
                    "status": control_state["status"],
                    "mode": (
                        "DIGITAL_ONLY_VISIBLE_GUI_BRIDGE"
                        if not control_state["real_launch_requested"]
                        else "DIGITAL_AND_REAL_VISIBLE_GUI_BRIDGE"
                    ),
                    "real_armed": True,
                    "real_commands_published": (
                        0 if not control_state["real_launch_requested"] else "REMOTE_RESULT_REQUIRED"
                    ),
                    "real_launch_requested": bool(control_state["real_launch_requested"]),
                    "real_initial_pose": control_state["real_initial_pose"],
                    "real_initial_error_rad": control_state["real_initial_error_rad"],
                    "real_start_delay_s": REAL_START_DELAY_S,
                    "digital_start_guard_s": DIGITAL_START_GUARD_S,
                    "bridge_launch_monotonic": control_state["bridge_launch_monotonic"],
                    "digital_start_monotonic": control_state["digital_start_monotonic"],
                    "manifest_sha256": inputs["hashes"]["manifest"],
                    "stage_sha256": inputs["hashes"]["stage"],
                    "command_signature": manifest.get("command_signature"),
                    "physics_rate_hz": GUI_BUTTON_PHYSICS_HZ,
                    "command_rate_hz": GUI_BUTTON_PHYSICS_HZ,
                    "move_seconds_each_way": GUI_BUTTON_MOVE_SECONDS,
                    "sample_count": len(button_samples),
                    "simulated_duration_s": 2.0 * GUI_BUTTON_MOVE_SECONDS,
                    "wall_duration_s": time.monotonic() - float(control_state["started_monotonic"]),
                    "telemetry": control_state["telemetry"],
                }
                button_report_path = args.report.with_name("aloha1_runtime_sleep_gui_button_run.json")
                button_report_path.write_text(json.dumps(button_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not control_state["requested"]:
            kit_app.update()
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    args = _parse_args()
    app = SimulationApp(
        {
            "headless": False,
            "create_new_stage": False,
            "window_title": "Isaac Sim - ALOHA1 Runtime Sleep Review",
            "window_width": 1980,
            "window_height": 1120,
        },
        experience=str(resolve_full_experience()),
    )
    try:
        return main(args, app)
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(run())

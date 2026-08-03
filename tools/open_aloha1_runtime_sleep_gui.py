#!/usr/bin/env python3
"""Open the frozen ALOHA1 Stage at runtime-measured Sleep for GUI review."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.validate_aloha1_home_sleep_digital import ARTICULATION_PATHS
from tools.validate_aloha1_home_sleep_digital import EXPECTED_DOF_ORDER
from tools.validate_aloha1_home_sleep_digital import EXPECTED_RUNTIME
from tools.validate_aloha1_home_sleep_digital import POSITION_GATE_RAD

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> tuple[Path, str]:
    resolved = path.resolve(strict=True)
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return resolved, actual


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
    return parser.parse_args()


def main(args: argparse.Namespace, app: Any) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    import omni.timeline
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
        physics_dt=1.0 / int(manifest["physics_rate_hz"]),
        rendering_dt=1.0 / int(manifest["physics_rate_hz"]),
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
    left_target = left_initial.copy()
    left_target[:6] = sleep
    left.set_joint_positions(left_target)
    left.set_joint_velocities(np.zeros_like(left_target))
    _apply_targets(left, left_target[:8], range(8))
    _apply_targets(right, right_initial[:8], range(8))
    world.play()
    for _ in range(30):
        _apply_targets(left, left_target[:8], range(8))
        _apply_targets(right, right_initial[:8], range(8))
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
        }
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

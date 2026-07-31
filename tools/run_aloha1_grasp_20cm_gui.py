#!/usr/bin/env python3
"""Open the Isaac Sim 5.1 ALOHA grasp-and-lift diagnostic window."""

# Isaac Sim 5.1 PhysX callback registration uses positional booleans.
# ruff: noqa: FBT003

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_grasp_20cm_gui.yaml"
DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260731-aloha1-grasp-20cm-button/runtime"
)
CLASSIFICATION = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
ABORTABLE_PHASES = (
    "RELEASE_DYNAMIC",
    "SETTLE",
    "OPEN_PREGRASP",
    "VERTICAL_DESCENT",
    "BILATERAL_CONTACT",
    "CLOSE_PRELOAD",
    "VERTICAL_LIFT",
    "HEIGHT_REACHED",
    "HOLD",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--startup-workspace", type=int, default=2)
    parser.add_argument(
        "--no-move-to-startup-workspace",
        action="store_true",
    )
    parser.add_argument(
        "--/app/useFabricSceneDelegate",
        dest="use_fabric_scene_delegate",
        choices=("true", "false"),
        default="false",
        help=(
            "Process-local delegate selection. This diagnostic defaults to "
            "the already validated OmniHydra workaround."
        ),
    )
    parser.add_argument(
        "--autorun",
        action="store_true",
        help="Invoke the same Run callback once after GUI initialization.",
    )
    parser.add_argument(
        "--close-after-terminal",
        action="store_true",
        help="Close after an autorun reaches PASS, FAIL, or ABORTED.",
    )
    parser.add_argument(
        "--autorun-abort-at-phase",
        choices=ABORTABLE_PHASES,
        default=None,
        help=(
            "Invoke the real Abort callback when autorun first reaches "
            "this phase."
        ),
    )
    parser.add_argument(
        "--reset-after-abort",
        action="store_true",
        help=(
            "After an automated Abort, invoke the real Reset callback, "
            "write its audit report, and close."
        ),
    )
    parser.add_argument(
        "--bottle-offset-x-m",
        type=float,
        default=0.0,
        help="Session-only Bottle500 world-X translation.",
    )
    parser.add_argument(
        "--bottle-offset-y-m",
        type=float,
        default=0.0,
        help="Session-only Bottle500 world-Y translation.",
    )
    parser.add_argument(
        "--bottle-world-from-object-json",
        type=Path,
        default=None,
        help=(
            "JSON file containing the frozen finite 4x4 Bottle500 "
            "world-from-object transform."
        ),
    )
    parser.add_argument(
        "--initial-arm-q-rad",
        type=float,
        nargs=6,
        default=None,
        help="Frozen six-DOF follower-left initial arm state in radians.",
    )
    parser.add_argument(
        "--initial-pose-hold-frames",
        type=int,
        default=60,
        help=(
            "Setup-only physics frames that hold and record the frozen "
            "initial arm pose before dynamic bottle settle."
        ),
    )
    parser.add_argument(
        "--additional-lift-margin-m",
        type=float,
        default=0.0,
        help=(
            "Diagnostic-only extra vertical lift distance; does not alter "
            "the 0.200 m measured bottle-clearance gate."
        ),
    )
    parser.add_argument(
        "--skip-collider-evidence",
        action="store_true",
        help=(
            "Keep the primary video clean; capture collider overlays in "
            "a separate deterministic repeat."
        ),
    )
    return parser.parse_args()


def _bounded_traceback() -> str:
    return "".join(traceback.format_exc(limit=20))[-12000:]


def _load_frozen_bottle_transform(
    path: Path,
) -> list[list[float]]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("world_from_object")
    from tools.aloha1_mapping.grasp_20cm_five_pose_ik import require_rigid_transform

    return require_rigid_transform(
        payload,
        name="bottle world_from_object JSON",
    ).tolist()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_abort_reset_evidence(
    *,
    requested_abort_phase: str,
    before_abort: dict[str, Any],
    after_abort: dict[str, Any],
    after_reset: dict[str, Any],
    machine_report: dict[str, Any],
    stage_sha256_before: str,
    stage_sha256_after: str,
) -> dict[str, Any]:
    """Evaluate Abort/Reset without allowing Reset writes to mask Abort."""

    gates = {
        "requested_phase_reached": (
            before_abort.get("phase") == requested_abort_phase
        ),
        "machine_report_is_aborted": (
            machine_report.get("status") == "ABORTED"
            and machine_report.get("reason") == "user_abort"
        ),
        "no_target_write_after_abort": (
            after_abort.get("target_write_count")
            == before_abort.get("target_write_count")
        ),
        "no_physics_sample_after_abort": (
            after_abort.get("telemetry_count")
            == before_abort.get("telemetry_count")
        ),
        "bottle_remained_dynamic_after_abort": (
            before_abort.get("bottle_kinematic_enabled") is False
            and after_abort.get("bottle_kinematic_enabled") is False
        ),
        "reset_returned_idle": after_reset.get("phase") == "IDLE",
        "reset_restored_setup_kinematic": (
            after_reset.get("bottle_kinematic_enabled") is True
        ),
        "stage_hash_unchanged": (
            stage_sha256_before == stage_sha256_after
            and before_abort.get("stage_sha256") == stage_sha256_before
            and after_abort.get("stage_sha256") == stage_sha256_before
            and after_reset.get("stage_sha256") == stage_sha256_before
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "classification": CLASSIFICATION,
        "requested_abort_phase": requested_abort_phase,
        "before_abort": before_abort,
        "after_abort": after_abort,
        "after_reset": after_reset,
        "machine_report": machine_report,
        "stage": {
            "sha256_before": stage_sha256_before,
            "sha256_after": stage_sha256_after,
        },
        "gates": gates,
        "boundaries": {
            "real_robot": False,
            "remote_103": False,
            "source_stage_modified": False,
            "final_collider_modified": False,
        },
        "task8": "NOT_RUN",
    }


class DiagnosticWindowController:
    """Bind a nonblocking runtime adapter to an Isaac-native window."""

    def __init__(
        self,
        *,
        adapter: Any,
        bindings: Any,
        timeline: Any,
        report_path: Path,
        ui: Any,
    ) -> None:
        self.adapter = adapter
        self.bindings = bindings
        self.timeline = timeline
        self.report_path = report_path
        self.ui = ui
        self.invalidated = False
        self.last_error = ""
        self.models = {
            name: ui.SimpleStringModel(value)
            for name, value in {
                "phase": "IDLE",
                "status": "READY",
                "clearance": "0.000 / 0.200 m",
                "maximum_clearance": "0.000 m",
                "contacts": "left=False right=False",
                "ee": "not sampled",
                "ik": "not solved",
                "fingers": "target/readback unavailable",
                "velocity": "linear/angular unavailable",
                "hold_drop": "0.000 m",
                "report": str(report_path),
                "classification": CLASSIFICATION,
            }.items()
        }
        self.run_button: Any | None = None
        self.abort_button: Any | None = None
        self.reset_button: Any | None = None
        self.window = self._build_window()
        self.refresh_models()

    def _build_window(self) -> Any:
        ui = self.ui
        window = ui.Window(
            "ALOHA Bottle500: Physical Grasp + 20 cm Clearance",
            width=560,
            height=530,
        )
        with window.frame, ui.VStack(spacing=6):
                ui.Label(
                    "Isaac Sim 5.1.0.0 / Kit 107.3.3",
                    height=22,
                )
                ui.Label(
                    CLASSIFICATION,
                    height=22,
                )
                for title, key in (
                    ("Phase", "phase"),
                    ("Status", "status"),
                    ("Bottle clearance", "clearance"),
                    ("Maximum clearance", "maximum_clearance"),
                    ("Finger contacts", "contacts"),
                    ("EE world position", "ee"),
                    ("IK", "ik"),
                    ("Finger target/readback", "fingers"),
                    ("Bottle velocity", "velocity"),
                    ("Hold drop", "hold_drop"),
                    ("Report", "report"),
                ):
                    with ui.HStack(height=24):
                        ui.Label(title, width=170)
                        ui.StringField(
                            model=self.models[key],
                            read_only=True,
                        )
                ui.Spacer(height=8)
                with ui.HStack(height=36, spacing=8):
                    self.run_button = ui.Button("Run: Grasp + Lift 20 cm",
                        clicked_fn=self.on_run_clicked,
                    )
                    self.abort_button = ui.Button("Abort",
                        clicked_fn=self.on_abort_clicked,
                    )
                    self.reset_button = ui.Button("Reset",
                        clicked_fn=self.on_reset_clicked,
                    )
                ui.Label(
                    (
                        "PASS requires Bottle500 collision minimum Z - table "
                        "top Z >= 0.200 m, then a 2 s dynamic hold."
                    ),
                    word_wrap=True,
                    height=48,
                )
                ui.Label(
                    (
                        "No SurfaceGripper, fixed joint, parent attachment, "
                        "real robot, remote 103, or Task 8."
                    ),
                    word_wrap=True,
                    height=48,
                )
        return window

    def _set_status(self, value: str) -> None:
        self.models["status"].set_value(value)

    def refresh_models(self) -> None:
        snapshot = self.bindings.ui_snapshot()
        self.models["phase"].set_value(self.adapter.phase.value)
        self.models["clearance"].set_value(
            f"{snapshot['clearance_m']:.3f} / 0.200 m"
        )
        self.models["maximum_clearance"].set_value(
            f"{snapshot['maximum_clearance_m']:.3f} m"
        )
        self.models["contacts"].set_value(
            "left={left_contact} right={right_contact}".format(**snapshot)
        )
        self.models["ee"].set_value(str(snapshot["ee_position_world_m"]))
        self.models["ik"].set_value(str(snapshot["ik"]))
        self.models["fingers"].set_value(str(snapshot["fingers"]))
        self.models["velocity"].set_value(str(snapshot["bottle_velocity"]))
        self.models["hold_drop"].set_value(
            f"{snapshot['hold_drop_m']:.4f} m"
        )
        self.models["report"].set_value(str(self.report_path))
        idle = self.adapter.phase.value == "IDLE"
        terminal = self.adapter.phase.value in {
            "IDLE",
            "PASS",
            "FAIL",
            "ABORTED",
        }
        self.run_button.enabled = bool(idle and not self.invalidated)
        self.abort_button.enabled = bool(self.adapter.is_running)
        self.reset_button.enabled = bool(
            terminal and not self.adapter.is_running
        )

    def on_run_clicked(self) -> None:
        if self.adapter.phase.value != "IDLE" or self.invalidated:
            return
        try:
            self.adapter.start()
            self.timeline.play()
            self._set_status("RUNNING")
        except Exception:
            self.last_error = _bounded_traceback()
            self._set_status("FAIL: startup exception")
            self.bindings.save_exception(self.last_error)
        self.refresh_models()

    def on_abort_clicked(self) -> None:
        if not self.adapter.is_running:
            return
        try:
            self.adapter.abort()
            self.timeline.pause()
            self._set_status("ABORTED")
        except Exception:
            self.last_error = _bounded_traceback()
            self._set_status("FAIL: abort exception")
            self.bindings.save_exception(self.last_error)
        self.refresh_models()

    def on_reset_clicked(self) -> None:
        if self.adapter.is_running:
            return
        try:
            self.adapter.reset()
            self.invalidated = False
            self._set_status("READY")
        except Exception:
            self.last_error = _bounded_traceback()
            self._set_status("FAIL: reset exception")
            self.bindings.save_exception(self.last_error)
        self.refresh_models()

    def on_physics_step(
        self,
        step_s: float,
        _context: Any | None = None,
    ) -> None:
        if not self.adapter.is_running:
            return
        try:
            transition = self.adapter.on_physics_step(step_s)
            self.refresh_models()
            if transition is not None and not self.adapter.is_running:
                self.timeline.pause()
                self._set_status(transition.current.value)
        except Exception:
            self.timeline.pause()
            self.last_error = _bounded_traceback()
            if self.adapter.is_running:
                self.adapter.fail_due_to_exception("runtime_exception")
            self._set_status("FAIL: runtime exception")
            self.bindings.save_exception(self.last_error)
        self.refresh_models()

    def invalidate(self, reason: str) -> None:
        self.invalidated = True
        if self.adapter.is_running:
            self.adapter.abort()
        self._set_status(f"INVALIDATED: {reason}; Reset required")
        self.refresh_models()


def main() -> int:
    args = _parse_args()
    sys.argv = [sys.argv[0]]
    if args.width <= 0 or args.height <= 0:
        raise ValueError("window dimensions must be positive")
    if args.startup_workspace < 1:
        raise ValueError("startup workspace must be >= 1")
    if not (
        math.isfinite(args.bottle_offset_x_m)
        and math.isfinite(args.bottle_offset_y_m)
    ):
        raise ValueError("Bottle500 XY offsets must be finite")
    if (
        not math.isfinite(args.additional_lift_margin_m)
        or args.additional_lift_margin_m < 0.0
    ):
        raise ValueError(
            "additional lift margin must be finite and non-negative"
        )
    if args.initial_pose_hold_frames < 1:
        raise ValueError("--initial-pose-hold-frames must be positive")
    if args.initial_arm_q_rad is not None and not all(
        math.isfinite(value) for value in args.initial_arm_q_rad
    ):
        raise ValueError("--initial-arm-q-rad values must be finite")
    if args.bottle_world_from_object_json is not None and (
        args.bottle_offset_x_m != 0.0 or args.bottle_offset_y_m != 0.0
    ):
        raise ValueError(
            "frozen bottle transform cannot be combined with legacy "
            "Bottle500 XY offsets"
        )
    if args.reset_after_abort and args.autorun_abort_at_phase is None:
        raise ValueError(
            "--reset-after-abort requires --autorun-abort-at-phase"
        )
    if args.autorun_abort_at_phase is not None and not args.autorun:
        raise ValueError("--autorun-abort-at-phase requires --autorun")

    sys.path.insert(0, str(ROOT))
    from tools.aloha1_mapping.grasp_20cm_runtime import Grasp20cmRuntimeAdapter
    from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
    from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file

    profile = load_and_verify_config(
        args.config.resolve(strict=True),
        project_root=ROOT,
    )
    bottle_world_from_object = (
        None
        if args.bottle_world_from_object_json is None
        else _load_frozen_bottle_transform(
            args.bottle_world_from_object_json
        )
    )
    config = profile["config"]
    stage_path = Path(
        profile["frozen_inputs"]["stage"]["absolute_path"]
    )
    stage_hash_before = sha256_file(stage_path)
    if stage_hash_before != config["stage"]["sha256"]:
        raise RuntimeError("approved Stage hash changed before SimulationApp")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": int(args.width),
            "height": int(args.height),
            "create_new_stage": False,
        }
    )
    exit_code = 1
    subscriptions: list[Any] = []
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / "aloha1_grasp_20cm_runtime.json"
    try:
        import carb.settings
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        import omni.kit.app
        from omni.physx import get_physx_interface
        import omni.timeline
        import omni.ui as ui
        import omni.usd

        from examples.aloha_isaac.scripts.open_workcell_gui import _move_current_process_window_to_workspace
        from tools.aloha1_mapping.grasp_20cm_isaac_bindings import IsaacGrasp20cmBindings
        from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage

        settings = carb.settings.get_settings()
        delegate_path = "/app/useFabricSceneDelegate"
        delegate_before = bool(settings.get(delegate_path))
        requested_fabric = args.use_fabric_scene_delegate == "true"
        settings.set_bool(delegate_path, requested_fabric)
        delegate_effective = bool(settings.get(delegate_path))
        if delegate_effective != requested_fabric:
            raise RuntimeError("delegate setting readback mismatch")

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open approved Stage: {stage_path}")
        stage = get_current_stage()
        validate_composed_stage(
            stage=stage,
            expected_root_prim=str(config["stage"]["root_prim"]),
            required_prims=[
                str(config["stage"]["articulation_prim"]),
                str(config["stage"]["table_prim"]),
            ],
        )
        bindings = IsaacGrasp20cmBindings(
            app=app,
            profile=profile,
            artifact_root=artifact_root,
            delegate_readback={
                "path": delegate_path,
                "before": delegate_before,
                "requested": requested_fabric,
                "effective": delegate_effective,
            },
            bottle_xy_offset_m=(
                float(args.bottle_offset_x_m),
                float(args.bottle_offset_y_m),
            ),
            bottle_world_from_object=bottle_world_from_object,
            initial_arm_q_rad=args.initial_arm_q_rad,
            initial_pose_hold_frames=int(
                args.initial_pose_hold_frames
            ),
            additional_lift_margin_m=float(
                args.additional_lift_margin_m
            ),
            capture_collider_evidence=not args.skip_collider_evidence,
        )
        adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        controller = DiagnosticWindowController(
            adapter=adapter,
            bindings=bindings,
            timeline=timeline,
            report_path=report_path,
            ui=ui,
        )

        subscriptions.append(
            get_physx_interface()
            .subscribe_physics_on_step_events(
                controller.on_physics_step,
                False,
                0,
            )
        )

        def on_stage_event(event: Any) -> None:
            if int(event.type) == int(omni.usd.StageEventType.OPENED):
                current = get_current_stage()
                if current is not stage:
                    controller.invalidate("Stage opened")

        subscriptions.append(
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop(
                on_stage_event,
                name="aloha1_grasp_20cm_stage_guard",
            )
        )

        def on_timeline_event(event: Any) -> None:
            if (
                int(event.type)
                == int(omni.timeline.TimelineEventType.STOP)
                and adapter.is_running
            ):
                controller.invalidate("timeline stopped")

        subscriptions.append(
            timeline.get_timeline_event_stream()
            .create_subscription_to_pop(
                on_timeline_event,
                name="aloha1_grasp_20cm_timeline_guard",
            )
        )
        if not args.no_move_to_startup_workspace:
            if args.startup_workspace == 2:
                _move_current_process_window_to_workspace(2)
            else:
                _move_current_process_window_to_workspace(
                    args.startup_workspace
                )

        autorun_state = {"updates": 0, "issued": False}

        def on_app_update(_event: Any) -> None:
            autorun_state["updates"] += 1
            if (
                args.autorun
                and not autorun_state["issued"]
                and autorun_state["updates"] >= 30
            ):
                autorun_state["issued"] = True
                controller.on_run_clicked()

        subscriptions.append(
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                on_app_update,
                name="aloha1_grasp_20cm_autorun",
            )
        )

        video_state = {
            "finalized": False,
            "error": None,
        }
        abort_reset_state = {
            "completed": False,
            "report_path": (
                artifact_root
                / "aloha1_grasp_20cm_abort_reset.json"
            ),
        }
        while app.is_running():
            app.update()
            try:
                if (
                    args.autorun_abort_at_phase is not None
                    and not abort_reset_state["completed"]
                    and adapter.is_running
                    and adapter.phase.value
                    == args.autorun_abort_at_phase
                ):
                    before_abort = bindings.abort_reset_snapshot(
                        phase=adapter.phase.value
                    )
                    controller.on_abort_clicked()
                    after_abort = bindings.abort_reset_snapshot(
                        phase=adapter.phase.value
                    )
                    machine_report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    if args.reset_after_abort:
                        controller.on_reset_clicked()
                        after_reset = bindings.abort_reset_snapshot(
                            phase=adapter.phase.value
                        )
                        evidence = evaluate_abort_reset_evidence(
                            requested_abort_phase=(
                                args.autorun_abort_at_phase
                            ),
                            before_abort=before_abort,
                            after_abort=after_abort,
                            after_reset=after_reset,
                            machine_report=machine_report,
                            stage_sha256_before=stage_hash_before,
                            stage_sha256_after=sha256_file(stage_path),
                        )
                        evidence["stage"]["absolute_path"] = str(
                            stage_path
                        )
                        _atomic_json(
                            abort_reset_state["report_path"],
                            evidence,
                        )
                        abort_reset_state["completed"] = True
                        controller._set_status(  # noqa: SLF001
                            "ABORT -> RESET "
                            f"{evidence['status']}"
                        )
                        controller.refresh_models()
                        omni.kit.app.get_app().post_quit()
                        continue
                captured = False
                if bindings.has_pending_video_frame:
                    timeline.pause()
                    captured = bindings.capture_pending_render_frame()
                terminal = (
                    not adapter.is_running
                    and adapter.phase.value
                    in {"PASS", "FAIL", "ABORTED"}
                )
                if captured:
                    bindings.capture_required_collider_evidence(
                        terminal=terminal,
                    )
                    if adapter.is_running:
                        timeline.play()
                if (
                    terminal
                    and args.autorun_abort_at_phase is None
                    and not bindings.has_pending_video_frame
                    and not video_state["finalized"]
                    and video_state["error"] is None
                ):
                    candidate = bindings.finalize_video_capture()
                    video_state["finalized"] = True
                    controller._set_status(  # noqa: SLF001
                        f"{adapter.phase.value}; VIDEO "
                        f"{candidate['promotion_status']}"
                    )
                    controller.refresh_models()
                if (
                    args.close_after_terminal
                    and autorun_state["issued"]
                    and terminal
                    and (
                        bool(video_state["finalized"])
                        or video_state["error"] is not None
                    )
                ):
                    omni.kit.app.get_app().post_quit()
            except Exception:
                video_state["error"] = _bounded_traceback()
                bindings.save_video_capture_exception(
                    str(video_state["error"])
                )
                controller._set_status(  # noqa: SLF001
                    "VIDEO FAIL; machine report preserved"
                )
                controller.refresh_models()
                if args.close_after_terminal:
                    omni.kit.app.get_app().post_quit()

        if sha256_file(stage_path) != stage_hash_before:
            raise RuntimeError("approved Stage hash changed during GUI session")
        exit_code = 0
    except Exception:
        _atomic_json(
            report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "classification": CLASSIFICATION,
                "stage": {
                    "absolute_path": str(stage_path),
                    "sha256_before": stage_hash_before,
                    "sha256_after": (
                        sha256_file(stage_path)
                        if stage_path.is_file()
                        else None
                    ),
                },
                "exception": _bounded_traceback(),
                "boundaries": {
                    "real_robot": False,
                    "remote_103": False,
                    "surface_gripper": False,
                    "fixed_joint": False,
                    "parent_attachment": False,
                    "source_stage_modified": False,
                    "final_collider_modified": False,
                    "task8": "NOT_RUN",
                },
            },
        )
        traceback.print_exc()
    finally:
        subscriptions.clear()
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

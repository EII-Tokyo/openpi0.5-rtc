from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

APPROVED_ACTIONS = frozenset(
    {
        "capture_state",
        "open_grasp_editor",
        "prepare_approved_session",
        "configure_approved_variant_b",
        "simulate_approved_variant_b",
        "export_approved_raw_grasp",
        "capture_evidence",
        "cleanup_approved_session",
    }
)


class BridgeActionError(RuntimeError):
    """Raised when a caller requests an action outside the fixed bridge."""


class _IsaacCaptureRuntime:
    """Read the live Kit state without changing Stage or timeline state."""

    @staticmethod
    def _layer_sha256(layer: Any) -> str:
        serialized = layer.ExportToString()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _layer_state(layer: Any) -> dict[str, Any]:
        return {
            "identifier": str(layer.identifier),
            "real_path": str(layer.realPath),
            "resolved_path": str(layer.resolvedPath),
            "dirty": bool(layer.dirty),
            "sublayers": list(layer.subLayerPaths),
            "serialized_sha256": _IsaacCaptureRuntime._layer_sha256(layer),
        }

    @staticmethod
    def _require_stage() -> tuple[Any, Any]:
        import omni.usd

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            raise BridgeActionError("No USD Stage is open")
        return context, stage

    def fingerprint(self) -> dict[str, Any]:
        context, stage = self._require_stage()
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()
        edit_target = stage.GetEditTarget().GetLayer()
        return {
            "stage_identifier": str(context.get_stage_url()),
            "root_layer": self._layer_state(root_layer),
            "session_layer": self._layer_state(session_layer),
            "edit_target_identifier": str(edit_target.identifier),
        }

    def read_state(self) -> dict[str, Any]:
        import omni.kit.app
        import omni.timeline

        context, stage = self._require_stage()
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()
        edit_target = stage.GetEditTarget().GetLayer()
        timeline = omni.timeline.get_timeline_interface()
        extension_manager = (
            omni.kit.app.get_app().get_extension_manager()
        )
        default_prim = stage.GetDefaultPrim()
        serialized_root = root_layer.ExportToString()
        return {
            "stage_identifier": str(context.get_stage_url()),
            "root_layer_identifier": str(root_layer.identifier),
            "session_layer_identifier": str(session_layer.identifier),
            "edit_target_identifier": str(edit_target.identifier),
            "default_prim_path": (
                str(default_prim.GetPath()) if default_prim.IsValid() else None
            ),
            "root_sublayers": list(root_layer.subLayerPaths),
            "session_sublayers": list(session_layer.subLayerPaths),
            "root_authored_reference_lines": sorted(
                line.strip()
                for line in serialized_root.splitlines()
                if "references" in line
            ),
            "selection": list(
                context.get_selection().get_selected_prim_paths()
            ),
            "timeline_playing": bool(timeline.is_playing()),
            "timeline_stopped": bool(timeline.is_stopped()),
            "grasp_editor_enabled": bool(
                extension_manager.is_extension_enabled(
                    "isaacsim.robot_setup.grasp_editor"
                )
            ),
        }


class ApprovedGraspEditorBridge:
    """Fixed, application-native bridge for the approved ALOHA experiment."""

    def __init__(self, runtime: Any | None = None) -> None:
        self._runtime = runtime if runtime is not None else _IsaacCaptureRuntime()
        self._heartbeat_monotonic = 0.0
        self._heartbeat_update_number = 0
        self._heartbeat_thread_ident: int | None = None

    def note_app_update(self) -> None:
        self._heartbeat_update_number += 1
        self._heartbeat_monotonic = time.monotonic()
        self._heartbeat_thread_ident = threading.get_ident()

    def execute(self, action: str) -> dict[str, Any]:
        if action not in APPROVED_ACTIONS:
            raise BridgeActionError(f"Action is not approved: {action!r}")
        handler = getattr(self, action)
        return handler()

    def capture_state(self) -> dict[str, Any]:
        fingerprint_before = self._runtime.fingerprint()
        readback = self._runtime.read_state()
        fingerprint_after = self._runtime.fingerprint()
        fingerprints_unchanged = fingerprint_before == fingerprint_after
        capture_thread_ident = threading.get_ident()
        main_thread_ident = threading.main_thread().ident
        callback_on_main_thread = (
            self._heartbeat_thread_ident == main_thread_ident
        )
        capture_on_main_thread = capture_thread_ident == main_thread_ident
        if not fingerprints_unchanged:
            status = "FAIL_READ_ONLY_FINGERPRINT_CHANGED"
        elif self._heartbeat_update_number < 1:
            status = "FAIL_NO_KIT_UPDATE_HEARTBEAT"
        elif not callback_on_main_thread or not capture_on_main_thread:
            status = "FAIL_NOT_KIT_MAIN_THREAD"
        else:
            status = "PASS"
        return {
            "action": "capture_state",
            "status": status,
            "heartbeat_monotonic": self._heartbeat_monotonic,
            "heartbeat_update_number": self._heartbeat_update_number,
            "heartbeat_thread_ident": self._heartbeat_thread_ident,
            "capture_thread_ident": capture_thread_ident,
            "main_thread_ident": main_thread_ident,
            "heartbeat_on_main_thread": callback_on_main_thread,
            "capture_on_main_thread": capture_on_main_thread,
            "fingerprints_unchanged": fingerprints_unchanged,
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "visual_tutor_extension": "my.isaac.visual_tutor",
            **readback,
        }

    def open_grasp_editor(self) -> dict[str, Any]:
        raise BridgeActionError("open_grasp_editor is reserved for Task 3")

    def prepare_approved_session(self) -> dict[str, Any]:
        raise BridgeActionError(
            "prepare_approved_session is reserved for Task 3"
        )

    def configure_approved_variant_b(self) -> dict[str, Any]:
        raise BridgeActionError(
            "configure_approved_variant_b is reserved for Task 3"
        )

    def simulate_approved_variant_b(self) -> dict[str, Any]:
        raise BridgeActionError(
            "simulate_approved_variant_b is reserved for Task 3"
        )

    def export_approved_raw_grasp(self) -> dict[str, Any]:
        raise BridgeActionError(
            "export_approved_raw_grasp is reserved for Task 3"
        )

    def capture_evidence(self) -> dict[str, Any]:
        raise BridgeActionError("capture_evidence is reserved for Task 3")

    def cleanup_approved_session(self) -> dict[str, Any]:
        raise BridgeActionError(
            "cleanup_approved_session is reserved for Task 3"
        )

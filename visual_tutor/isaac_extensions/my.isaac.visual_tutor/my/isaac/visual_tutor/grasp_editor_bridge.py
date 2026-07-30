from __future__ import annotations

from collections import deque
import hashlib
import re
import threading
import time
from typing import Any

APPROVED_MANIFEST_SHA256 = (
    "fecacb461c43e299e0ec1209ffde5bd8e9826ac93fd3defcdc80bfb4405e93ba"
)
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
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_RETAINED_ACKS = 16


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
            "selection": list(
                context.get_selection().get_selected_prim_paths()
            ),
            "timeline_playing": bool(timeline.is_playing()),
            "timeline_stopped": bool(timeline.is_stopped()),
            "timeline_current_time": float(timeline.get_current_time()),
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
        self._request_sequence = 0
        self._pending_requests: deque[dict[str, Any]] = deque()
        self._acks: dict[tuple[str, int], dict[str, Any]] = {}
        self._ack_order: deque[tuple[str, int]] = deque()
        self._shutdown = False

    def note_app_update(self) -> None:
        if self._shutdown:
            return
        self._heartbeat_update_number += 1
        self._heartbeat_monotonic = time.monotonic()
        self._heartbeat_thread_ident = threading.get_ident()

    def request_capture_state(
        self,
        run_id: str,
        expected_manifest_sha: str,
    ) -> dict[str, Any]:
        if self._shutdown:
            raise BridgeActionError("Bridge is already shut down")
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise BridgeActionError(f"Invalid run_id: {run_id!r}")
        if expected_manifest_sha != APPROVED_MANIFEST_SHA256:
            raise BridgeActionError(
                "Expected manifest SHA does not match the approved manifest"
            )
        self._request_sequence += 1
        request = {
            "run_id": run_id,
            "request_sequence": self._request_sequence,
            "action": "capture_state",
            "expected_manifest_sha": expected_manifest_sha,
            "requested_monotonic": time.monotonic(),
        }
        self._pending_requests.append(request)
        return {**request, "status": "ENQUEUED"}

    def process_next_request(self) -> dict[str, Any] | None:
        if self._shutdown or not self._pending_requests:
            return None
        request = self._pending_requests.popleft()
        try:
            ack = self._capture_state_for_request(request)
        except Exception as error:
            ack = self._base_ack(request)
            ack.update(
                {
                    "status": "FAIL_CAPTURE_EXCEPTION",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "completed_monotonic": time.monotonic(),
                }
            )
        self._store_ack(ack)
        return ack

    def get_ack(
        self,
        run_id: str,
        request_sequence: int,
    ) -> dict[str, Any] | None:
        ack = self._acks.get((run_id, request_sequence))
        return dict(ack) if ack is not None else None

    def _base_ack(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            **request,
            "heartbeat_monotonic": self._heartbeat_monotonic,
            "update_number": self._heartbeat_update_number,
            "heartbeat_thread_ident": self._heartbeat_thread_ident,
            "callback_thread_ident": threading.get_ident(),
            "main_thread_ident": threading.main_thread().ident,
            "visual_tutor_extension": "my.isaac.visual_tutor",
        }

    def _capture_state_for_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        fingerprint_before = self._runtime.fingerprint()
        readback = self._runtime.read_state()
        fingerprint_after = self._runtime.fingerprint()
        fingerprints_unchanged = fingerprint_before == fingerprint_after
        main_thread_ident = threading.main_thread().ident
        callback_thread_ident = threading.get_ident()
        callback_on_main_thread = callback_thread_ident == main_thread_ident
        heartbeat_on_main_thread = (
            self._heartbeat_thread_ident == main_thread_ident
        )
        if not fingerprints_unchanged:
            status = "FAIL_READ_ONLY_FINGERPRINT_CHANGED"
        elif self._heartbeat_update_number < 1:
            status = "FAIL_NO_KIT_UPDATE_HEARTBEAT"
        elif not callback_on_main_thread or not heartbeat_on_main_thread:
            status = "FAIL_NOT_KIT_MAIN_THREAD"
        else:
            status = "PASS"
        return {
            **self._base_ack(request),
            "status": status,
            "heartbeat_on_main_thread": heartbeat_on_main_thread,
            "callback_on_main_thread": callback_on_main_thread,
            "fingerprints_unchanged": fingerprints_unchanged,
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "completed_monotonic": time.monotonic(),
            **readback,
        }

    def _store_ack(self, ack: dict[str, Any]) -> None:
        key = (str(ack["run_id"]), int(ack["request_sequence"]))
        self._acks[key] = dict(ack)
        self._ack_order.append(key)
        while len(self._ack_order) > _MAX_RETAINED_ACKS:
            expired = self._ack_order.popleft()
            self._acks.pop(expired, None)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        while self._pending_requests:
            request = self._pending_requests.popleft()
            ack = self._base_ack(request)
            ack.update(
                {
                    "status": "FAIL_EXTENSION_SHUTDOWN_PENDING_REQUEST",
                    "completed_monotonic": time.monotonic(),
                }
            )
            self._store_ack(ack)

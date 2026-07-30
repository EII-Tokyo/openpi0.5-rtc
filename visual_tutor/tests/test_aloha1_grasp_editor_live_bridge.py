from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / (
    "visual_tutor/isaac_extensions/my.isaac.visual_tutor/"
    "my/isaac/visual_tutor/grasp_editor_bridge.py"
)
EXTENSION = ROOT / (
    "visual_tutor/isaac_extensions/my.isaac.visual_tutor/"
    "my/isaac/visual_tutor/extension.py"
)
PROBE = ROOT / "tools/probe_aloha1_visual_tutor_live_bridge.py"

EXPECTED_ACTIONS = frozenset(
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
APPROVED_MANIFEST_SHA256 = (
    "fecacb461c43e299e0ec1209ffde5bd8e9826ac93fd3defcdc80bfb4405e93ba"
)


class FakeCaptureRuntime:
    def __init__(self, *, mutate: bool = False) -> None:
        self.mutate = mutate
        self.fingerprint_calls = 0

    def fingerprint(self) -> dict[str, object]:
        self.fingerprint_calls += 1
        version = self.fingerprint_calls if self.mutate else 1
        return {
            "root_layer_sha256": f"root-{version}",
            "session_layer_sha256": "session-1",
        }

    def read_state(self) -> dict[str, object]:
        return {
            "stage_identifier": "/approved/stage.usda",
            "root_layer_identifier": "/approved/stage.usda",
            "session_layer_identifier": "anon:session",
            "edit_target_identifier": "/approved/stage.usda",
            "default_prim_path": "/World",
            "root_sublayers": ["configuration.usda"],
            "session_sublayers": [],
            "selection": [],
            "timeline_playing": False,
            "timeline_stopped": True,
            "timeline_current_time": 0.0,
            "grasp_editor_enabled": False,
        }


def _load_bridge_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aloha1_grasp_editor_bridge_under_test",
        BRIDGE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aloha1_visual_tutor_live_probe_under_test",
        PROBE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_has_exact_approved_action_surface() -> None:
    module = _load_bridge_module()
    assert module.APPROVED_ACTIONS == EXPECTED_ACTIONS
    assert module.APPROVED_MANIFEST_SHA256 == APPROVED_MANIFEST_SHA256
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    with pytest.raises(module.BridgeActionError, match="run_id"):
        bridge.request_capture_state("", APPROVED_MANIFEST_SHA256)
    with pytest.raises(module.BridgeActionError, match="manifest"):
        bridge.request_capture_state("run-1", "0" * 64)


def test_capture_request_is_executed_only_on_app_update() -> None:
    module = _load_bridge_module()
    runtime = FakeCaptureRuntime()
    bridge = module.ApprovedGraspEditorBridge(runtime=runtime)
    request = bridge.request_capture_state(
        "run-1",
        APPROVED_MANIFEST_SHA256,
    )

    assert request["status"] == "ENQUEUED"
    assert bridge.get_ack("run-1", request["request_sequence"]) is None

    bridge.note_app_update()
    payload = bridge.process_next_request()

    assert payload is not None
    assert payload["action"] == "capture_state"
    assert payload["run_id"] == "run-1"
    assert payload["request_sequence"] == request["request_sequence"]
    assert payload["expected_manifest_sha"] == APPROVED_MANIFEST_SHA256
    assert payload["status"] == "PASS"
    assert payload["visual_tutor_extension"] == "my.isaac.visual_tutor"
    assert payload["update_number"] == 1
    assert payload["heartbeat_monotonic"] > 0.0
    assert payload["callback_thread_ident"] == payload["main_thread_ident"]
    assert payload["requested_monotonic"] < payload["completed_monotonic"]
    assert payload["stage_identifier"] == "/approved/stage.usda"
    assert payload["root_layer_identifier"] == "/approved/stage.usda"
    assert payload["session_layer_identifier"] == "anon:session"
    assert payload["edit_target_identifier"] == "/approved/stage.usda"
    assert payload["timeline_playing"] is False
    assert payload["timeline_stopped"] is True
    assert payload["fingerprints_unchanged"] is True
    assert payload["fingerprint_before"] == payload["fingerprint_after"]
    assert runtime.fingerprint_calls == 2
    assert bridge.get_ack("run-1", request["request_sequence"]) == payload


def test_two_requests_require_fresh_exact_ack_and_advance_sequence() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    first_request = bridge.request_capture_state(
        "run-2",
        APPROVED_MANIFEST_SHA256,
    )
    bridge.note_app_update()
    first_ack = bridge.process_next_request()
    second_request = bridge.request_capture_state(
        "run-2",
        APPROVED_MANIFEST_SHA256,
    )
    assert bridge.get_ack(
        "run-2",
        second_request["request_sequence"],
    ) is None
    bridge.note_app_update()
    second_ack = bridge.process_next_request()

    assert first_ack is not None
    assert second_ack is not None
    assert second_request["request_sequence"] > first_request["request_sequence"]
    assert second_ack["update_number"] > first_ack["update_number"]
    assert second_ack["heartbeat_monotonic"] > first_ack["heartbeat_monotonic"]
    assert bridge.get_ack(
        "stale-run-id",
        first_request["request_sequence"],
    ) is None


def test_queued_capture_fails_closed_if_stage_fingerprint_changes() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(
        runtime=FakeCaptureRuntime(mutate=True)
    )
    bridge.request_capture_state("run-3", APPROVED_MANIFEST_SHA256)
    bridge.note_app_update()

    payload = bridge.process_next_request()

    assert payload is not None
    assert payload["status"] == "FAIL_READ_ONLY_FINGERPRINT_CHANGED"
    assert payload["fingerprints_unchanged"] is False


def test_queued_capture_rejects_processing_outside_update_callback_thread() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    bridge.request_capture_state("run-thread", APPROVED_MANIFEST_SHA256)
    bridge.note_app_update()
    results: list[dict[str, object] | None] = []
    worker = threading.Thread(
        target=lambda: results.append(bridge.process_next_request())
    )

    worker.start()
    worker.join()

    assert results[0] is not None
    assert results[0]["status"] == "FAIL_NOT_KIT_MAIN_THREAD"
    assert results[0]["callback_thread_ident"] != results[0]["main_thread_ident"]
    assert (
        results[0]["heartbeat_thread_ident"]
        == results[0]["main_thread_ident"]
    )


def test_direct_capture_is_not_exposed() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    assert not hasattr(bridge, "capture_state")
    assert not hasattr(bridge, "execute")


def test_shutdown_completes_pending_request_with_stable_failure() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    request = bridge.request_capture_state(
        "run-shutdown",
        APPROVED_MANIFEST_SHA256,
    )

    bridge.shutdown()
    bridge.shutdown()
    payload = bridge.get_ack(
        "run-shutdown",
        request["request_sequence"],
    )

    assert payload is not None
    assert payload["status"] == "FAIL_EXTENSION_SHUTDOWN_PENDING_REQUEST"
    with pytest.raises(module.BridgeActionError, match="shut down"):
        bridge.request_capture_state("run-after", APPROVED_MANIFEST_SHA256)


def test_extension_uses_global_update_event_and_module_singleton_accessor() -> None:
    text = EXTENSION.read_text(encoding="utf-8")
    assert "ApprovedGraspEditorBridge" in text
    assert "GLOBAL_EVENT_UPDATE" in text
    assert "get_extension_instance" in text
    assert "get_live_bridge" in text
    assert "note_app_update" in text
    assert "process_next_request" in text
    callback = text[text.index("def _on_app_update") :]
    assert callback.index("note_app_update") < callback.index(
        "process_next_request"
    )
    assert "shutdown()" in text
    assert "_update_subscription = None" in text


def test_capture_bridge_contains_no_forbidden_control_fallbacks() -> None:
    text = BRIDGE.read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "xdotool",
        "pyautogui",
        "192.168.1.103",
        "SetEditTarget",
        "timeline.pause",
        "timeline.play",
        "timeline.stop",
        "kit.commands",
        "World(",
        "motion_generation",
    ):
        assert forbidden not in text


def test_probe_registers_parent_extension_path_and_writes_before_close() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert "extension_manager.add_path(str(extension_parent))" in text
    assert "get_live_bridge" in text
    assert "request_capture_state" in text
    assert "get_ack" in text
    assert ".capture_state()" not in text
    assert "threading.main_thread().ident" in text
    assert "request_sequence" in text
    assert "update_number" in text
    assert "fingerprints_unchanged" in text
    assert text.index("_write_report_before_close(") < text.index(
        "simulation_app.close()"
    )
    for forbidden in (
        "inverse_kinematics",
        "motion_generation",
        "GraspTester",
        "SurfaceGripper",
        "192.168.1.103",
    ):
        assert forbidden not in text


def test_probe_verifies_enabled_extension_identity_with_local_api() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert "get_extension_id_by_module" not in text
    assert "get_enabled_extension_id" in text
    assert "get_extension_path" in text
    assert "enable_result is not True" in text
    assert "is_extension_enabled" in text
    assert "approved_extension_dir" in text
    assert "grasp_editor_live_version" not in text


def test_probe_freezes_baseline_before_enable_and_compares_complete_state() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert text.index("runtime_baseline_before_enable =") < text.index(
        "enable_result ="
    )
    for field in (
        "stage_identifier",
        "root_layer",
        "session_layer",
        "edit_target_identifier",
        "timeline_playing",
        "timeline_stopped",
        "timeline_current_time",
        "default_prim_path",
        "required_prims",
    ):
        assert field in text
    assert "root_authored_reference_lines" not in text
    assert "runtime_baseline_after_ack" in text


def test_probe_preflights_report_before_app_and_disables_extension() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert text.index("_preflight_report_path(report_path)") < text.index(
        'SimulationApp({"headless": False})'
    )
    assert "set_extension_enabled_immediate" in text
    assert "cleanup" in text
    assert "extension_disabled" in text
    assert '_assert_equal("timeline stopped"' not in text


def test_probe_normalizes_plain_paths_and_file_urls() -> None:
    module = _load_probe_module()
    expected = Path("/tmp/approved.usda").resolve()
    assert module._normalize_stage_path(  # noqa: SLF001 - focused pure helper.
        "/tmp/approved.usda"
    ) == expected
    assert module._normalize_stage_path(  # noqa: SLF001 - focused pure helper.
        "file:///tmp/approved.usda"
    ) == expected

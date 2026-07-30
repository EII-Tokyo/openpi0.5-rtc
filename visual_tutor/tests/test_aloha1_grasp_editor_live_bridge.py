from __future__ import annotations

import importlib.util
from pathlib import Path
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
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())
    with pytest.raises(module.BridgeActionError, match="not approved"):
        bridge.execute("arbitrary_python")


def test_capture_state_is_read_only_and_reports_live_thread_evidence() -> None:
    module = _load_bridge_module()
    runtime = FakeCaptureRuntime()
    bridge = module.ApprovedGraspEditorBridge(runtime=runtime)
    bridge.note_app_update()
    bridge.note_app_update()

    payload = bridge.execute("capture_state")

    assert payload["action"] == "capture_state"
    assert payload["status"] == "PASS"
    assert payload["visual_tutor_extension"] == "my.isaac.visual_tutor"
    assert payload["heartbeat_update_number"] == 2
    assert payload["heartbeat_monotonic"] > 0.0
    assert payload["heartbeat_thread_ident"] == payload["main_thread_ident"]
    assert payload["capture_thread_ident"] == payload["main_thread_ident"]
    assert payload["stage_identifier"] == "/approved/stage.usda"
    assert payload["root_layer_identifier"] == "/approved/stage.usda"
    assert payload["session_layer_identifier"] == "anon:session"
    assert payload["edit_target_identifier"] == "/approved/stage.usda"
    assert payload["timeline_playing"] is False
    assert payload["timeline_stopped"] is True
    assert payload["fingerprints_unchanged"] is True
    assert payload["fingerprint_before"] == payload["fingerprint_after"]
    assert runtime.fingerprint_calls == 2


def test_capture_state_fails_closed_if_readback_changes_stage_fingerprint() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(
        runtime=FakeCaptureRuntime(mutate=True)
    )
    bridge.note_app_update()

    payload = bridge.capture_state()

    assert payload["status"] == "FAIL_READ_ONLY_FINGERPRINT_CHANGED"
    assert payload["fingerprints_unchanged"] is False


def test_capture_state_reports_missing_update_heartbeat_before_thread_gate() -> None:
    module = _load_bridge_module()
    bridge = module.ApprovedGraspEditorBridge(runtime=FakeCaptureRuntime())

    payload = bridge.capture_state()

    assert payload["status"] == "FAIL_NO_KIT_UPDATE_HEARTBEAT"


def test_extension_uses_global_update_event_and_module_singleton_accessor() -> None:
    text = EXTENSION.read_text(encoding="utf-8")
    assert "ApprovedGraspEditorBridge" in text
    assert "GLOBAL_EVENT_UPDATE" in text
    assert "get_extension_instance" in text
    assert "get_live_bridge" in text
    assert "note_app_update" in text
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
    assert "threading.main_thread().ident" in text
    assert "heartbeat_update_number" in text
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


def test_probe_normalizes_plain_paths_and_file_urls() -> None:
    module = _load_probe_module()
    expected = Path("/tmp/approved.usda").resolve()
    assert module._normalize_stage_path(  # noqa: SLF001 - focused pure helper.
        "/tmp/approved.usda"
    ) == expected
    assert module._normalize_stage_path(  # noqa: SLF001 - focused pure helper.
        "file:///tmp/approved.usda"
    ) == expected

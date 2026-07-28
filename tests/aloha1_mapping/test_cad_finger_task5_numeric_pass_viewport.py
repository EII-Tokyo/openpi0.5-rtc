from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tools/capture_aloha_viper_cad_finger_task5_numeric_pass_viewport.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "task5_numeric_pass_viewport",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_trace_frames_returns_distinct_open_partial_closed() -> None:
    module = _load_module()
    trace = [
        {"frame": index, "readback_left_m": 0.057 - index * 0.003}
        for index in range(13)
    ]

    selected = module.select_trace_frames(trace)

    assert [item["phase"] for item in selected] == [
        "open_maximum_legal_aperture",
        "partially_closed",
        "closed",
    ]
    assert [item["record"]["frame"] for item in selected] == [0, 6, 12]


def test_capture_uses_local_viewport_api_and_preserves_scope() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "capture_viewport_to_file" in source
    assert "get_active_viewport" in source
    assert "RUNTIME_READBACK_REPLAY_AUXILIARY" in source
    assert '"bottle_contact_grasp": "NOT_RUN"' in source
    assert '"task7": "NOT_RUN"' in source
    assert '"task8": "NOT_RUN"' in source
    assert "runtime_finger_center_world_m" in source
    assert "fixed_camera_for_all_phases" in source
    assert "source_stage_immutable" in source

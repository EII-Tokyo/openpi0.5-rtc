from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.validate_aloha_viper_cad_finger_task5_structure import _read_nonblank_rgba

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = (
    ROOT
    / "tools/capture_aloha_viper_cad_finger_task5_drive_replay.py"
)
ANNOTATION_SCRIPT = (
    ROOT
    / "tools/annotate_aloha_viper_cad_finger_task5_drive_replay.py"
)


def test_drive_replay_capture_is_explicitly_auxiliary_and_no_bottle() -> None:
    source = CAPTURE_SCRIPT.read_text(encoding="utf-8")

    assert '{"headless": True, "width": 1280, "height": 900}' in source
    assert "RUNTIME_READBACK_REPLAY_AUXILIARY" in source
    assert "symmetric_close" in source
    assert '"bottle_contact_grasp": "NOT_RUN"' in source
    assert '"task8": "NOT_RUN"' in source
    assert "stage_immutable" in source
    assert "numeric_report_immutable" in source
    assert "world.reset()" in source
    assert "articulation.set_joint_positions" in source
    assert "world.step(" not in source


def test_drive_replay_annotation_preserves_dynamic_failure() -> None:
    source = ANNOTATION_SCRIPT.read_text(encoding="utf-8")

    assert "DYNAMIC DRIVE GATE = FAIL" in source
    assert "RUNTIME READBACK REPLAY — AUXILIARY" in source
    assert "NO BOTTLE / CONTACT / GRASP CLAIM" in source
    assert "PENDING_VISUAL_MODEL_REVIEW" in source
    assert "Blue = left_finger / CAD +X" in source
    assert "Orange = right_finger / CAD -X" in source
    assert "PASS_NUMERIC_ONLY" in source
    assert "DYNAMIC NUMERIC GATE = PASS" in source


def test_camera_poll_skips_transient_empty_rgba() -> None:
    class World:
        def render(self) -> None:
            return None

    class Camera:
        def __init__(self) -> None:
            self._frames = [
                np.empty((0,), dtype=np.float32),
                np.asarray(
                    [
                        [
                            [0.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0],
                        ]
                    ],
                    dtype=np.float32,
                ),
            ]

        def get_rgba(self) -> np.ndarray:
            return self._frames.pop(0)

    result = _read_nonblank_rgba(
        World(),
        Camera(),
        maximum_render_updates=2,
    )

    assert result.shape == (1, 2, 4)

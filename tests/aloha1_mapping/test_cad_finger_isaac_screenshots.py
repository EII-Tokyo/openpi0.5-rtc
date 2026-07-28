from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = (
    ROOT / "tools/capture_aloha_viper_cad_finger_diagnostic.py"
)


def test_capture_script_freezes_stage_and_uses_four_paired_views() -> None:
    source = CAPTURE_SCRIPT.read_text(encoding="utf-8")

    assert "EXPECTED_DIAGNOSTIC_STAGE_SHA256" in source
    assert "EXPECTED_DIAGNOSTIC_LAYER_SHA256" in source
    assert "stage.GetSessionLayer()" in source
    assert "stage.GetRootLayer().Save()" not in source
    assert "VISUAL_SESSION_QPOS_PROJECTION" in source
    assert "_set_visual_q_projection" in source
    assert "world.reset()" in source
    assert "world.step(render=True)" not in source
    assert "_paired_pose_signature" in source
    assert '("closed", (0.021, -0.021))' in source
    assert '("open", (0.057, -0.057))' in source
    for view in (
        "true_top",
        "true_bottom",
        "tip_end",
        "base_oblique",
    ):
        assert f'"{view}"' in source
    assert "required_capture_count" in source
    assert "8" in source

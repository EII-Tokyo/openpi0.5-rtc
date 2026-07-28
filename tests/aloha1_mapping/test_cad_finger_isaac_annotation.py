from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/annotate_aloha_viper_cad_finger_isaac.py"


def test_annotation_script_preserves_visual_gate_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Blue = left_finger / CAD +X" in source
    assert "Orange = right_finger / CAD -X" in source
    assert "CAD-derived annotation sample" in source
    assert "not a physical contact point" in source
    assert "CAD INSTALLATION VISUAL GATE ONLY" in source
    assert "NO COLLISION / CONTACT / GRASP CLAIM" in source
    assert "annotated_absolute_path" in source
    assert "camera" in source

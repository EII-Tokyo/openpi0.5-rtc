from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tools/annotate_aloha_viper_cad_finger_task5_structure.py"
)


def test_structure_annotation_keeps_physics_failure_explicit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "STRUCTURE VISUAL GATE PASS ONLY" in source
    assert "DYNAMIC DRIVE / MIMIC = FAIL" in source
    assert "NO BOTTLE / CONTACT / GRASP CLAIM" in source
    assert "Blue = left_finger / CAD +X" in source
    assert "Orange = right_finger / CAD -X" in source
    assert "CAD-derived inward-surface sample" in source
    assert "not a physical contact point" in source
    assert "PENDING_VISUAL_MODEL_REVIEW" in source

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "tools/compare_aloha_viper_cad_finger_task5_drive_probes.py"
)


def test_comparison_keeps_dynamic_and_bottle_gates_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"status": "FAIL"' in source
    assert '"bottle_test_allowed": False' in source
    assert '"bottle_contact_grasp": "NOT_RUN"' in source
    assert '"task8": "NOT_RUN"' in source
    assert "disjoint root/assembly frames" in source

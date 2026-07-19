from __future__ import annotations

from pathlib import Path


def test_rl_drive_target_smoke_entrypoint_exists() -> None:
    wrapper = Path("scripts/run_aloha_isaac_rl_drive_target_smoke.py")
    implementation = Path("aloha_isaac_replay/scripts/run_rl_drive_target_smoke.py")
    assert wrapper.exists()
    assert implementation.exists()
    assert "run_rl_drive_target_smoke" in wrapper.read_text()
    text = implementation.read_text()
    assert "--episode" in text
    assert "--max-controlled-error" in text
    assert "--causality-probe" in text
    assert "--gui" in text

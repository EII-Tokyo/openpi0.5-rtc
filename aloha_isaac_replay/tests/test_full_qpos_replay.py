from __future__ import annotations

from pathlib import Path


def test_full_qpos_replay_entrypoint_exists_but_is_runtime_gated() -> None:
    path = Path("scripts/replay_aloha_qpos_full.py")
    assert path.exists()
    assert "replay_aloha_qpos_full" in path.read_text()


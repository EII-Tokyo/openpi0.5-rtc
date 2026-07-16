from __future__ import annotations

from aloha_isaac_replay.controller_system_id.timebase import target_hold_seconds


def test_target_hold_50hz() -> None:
    assert target_hold_seconds(fps=50.0, steps_per_target=1) == 0.02
    assert target_hold_seconds(fps=50.0, steps_per_target=5) == 0.10


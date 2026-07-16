from __future__ import annotations

from aloha_isaac_replay.controller_system_id.offline_models import split_episode_ids


def test_episode_split_no_leakage() -> None:
    splits = split_episode_ids([f"ep{i}" for i in range(10)])
    assert set(splits["identification"]).isdisjoint(splits["validation"])
    assert set(splits["identification"]).isdisjoint(splits["heldout"])
    assert set(splits["validation"]).isdisjoint(splits["heldout"])
    assert len(splits["identification"]) == 6
    assert len(splits["validation"]) == 2
    assert len(splits["heldout"]) == 2


import json
import random

import pytest

from openpi.shared.long_memory import (
    LongMemoryConfig, MemoryEntry, MemoryTimeline, condition_at,
    decode_memory, drop_memory, encode_memory, video_indices,
)


def label(state, action):
    return json.dumps(dict(bottle_description=["Geyser blue cap"], bottle_state=state, subtask=action))


def timeline():
    return MemoryTimeline([0, 1, 2, 3, 4, 5, 6],
                          [label("on table", "pick up")] * 3 + [label("held and capped", "unscrew")] * 4)


def test_no_future_or_invented_completion():
    t = timeline()
    assert len(t.at(2.99)) == 1
    assert t.at(2.99)[-1].intent == "pick up"
    assert t.at(3)[-1].observation == "held and capped"
    assert len(t.at(-1)) == 0


def test_partial_labels_do_not_invent_scene_facts():
    t = MemoryTimeline([0, 1], ['{"bottle_state":null,"subtask":null}', '{"subtask":"return to home position"}'])
    assert not t.at(0)
    assert t.at(1)[0].observation == "Unknown (not annotated)"


def test_every_dropout_count_and_nonprefix_subsets_are_possible():
    entries = tuple(MemoryEntry(i, "b", "s", str(i)) for i in range(6))
    rng = random.Random(42)
    samples = [drop_memory(entries, rng) for _ in range(2000)]
    assert {len(s) for s in samples} == set(range(7))
    assert any(s and s[0].time_s > 0 for s in samples)
    assert all(list(s) == sorted(s, key=lambda e: e.time_s) for s in samples)


def test_late_in_phase_stale_or_empty_memory_is_trained():
    t, rng = timeline(), random.Random(5)
    samples = [condition_at(t, 6, LongMemoryConfig(), training=True, rng=rng) for _ in range(300)]
    assert any(not s for s in samples)
    assert any(s and s[-1].intent == "pick up" for s in samples)
    assert any(s and s[-1].intent == "unscrew" for s in samples)
    assert all(e.time_s <= 6 for s in samples for e in s)


def test_video_uses_only_past_and_never_other_episode():
    times = [0, .2, .6, 1.1, 2.2, 3.1, 3.8]
    idx = video_indices(times, 3.5)
    assert all(times[i] <= 3.5 for i in idx)
    assert video_indices([0, .02], 0) == (0,)
    with pytest.raises(ValueError):
        video_indices([1, 0], 1)


def test_future_and_malformed_high_level_output_rejected():
    e = MemoryEntry(4, "b", "held", "unscrew")
    with pytest.raises(ValueError):
        decode_memory(encode_memory([e]), now_s=3)
    with pytest.raises(ValueError):
        decode_memory('{"events":[],"execute":"move"}', now_s=3)
    assert decode_memory(encode_memory([e]), now_s=4) == (e,)


def test_dropout_mixture_is_not_uniform_count_overdropping():
    entries = tuple(MemoryEntry(i, "b", "s", str(i)) for i in range(6))
    rng = random.Random(19)
    samples = [drop_memory(entries, rng) for _ in range(10000)]
    assert .18 < sum(not s for s in samples) / len(samples) < .22
    assert .48 < sum(s == entries for s in samples) / len(samples) < .52
    assert .28 < sum(0 < len(s) < 6 for s in samples) / len(samples) < .32
    settings = LongMemoryConfig()
    assert settings.stale_probability == .2
    with pytest.raises(ValueError):
        LongMemoryConfig(empty_probability=.8, partial_probability=.3)


def test_serialized_memory_time_never_rounds_into_the_future():
    event = MemoryEntry(1.2346, "b", "held", "unscrew")
    assert decode_memory(encode_memory([event]), now_s=1.2346)[0].time_s <= 1.2346


def test_home_intent_is_not_completion_and_next_bottle_starts_clean():
    t = MemoryTimeline([0,1,2,3,4], [label('held bottle A','unscrew'),label('empty hands','Return to initial pose'),
        label('empty hands','Return to initial pose'),label('bottle B on table','pick up'),label('bottle B held','unscrew')],reset_on_home=True)
    assert t.at(1)[-1].intent == 'Return to initial pose'
    assert len(t.at(2)) == 2
    assert len(t.at(3)) == 1 and t.at(3)[0].time_s == 3
    assert all(e.time_s >= 3 for e in t.at(4))


def test_terminal_home_and_new_episode_have_no_old_memory():
    t = MemoryTimeline([0,1,2],[label('held','unscrew'),label('empty','return home'),label('empty','return home')],reset_on_home=True)
    assert t.at(1)
    assert t.at(2) == ()
    assert t.at(10) == ()
    new = MemoryTimeline([0],[label('new bottle','pick up')],reset_on_home=True)
    assert len(new.at(0)) == 1 and new.at(0)[0].observation == 'new bottle'

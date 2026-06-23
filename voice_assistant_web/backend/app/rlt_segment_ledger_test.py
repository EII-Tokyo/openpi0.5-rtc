
from voice_assistant_web.backend.app.rlt_segment_ledger import RLTSegmentLedger


def test_segment_ledger_tracks_commit_void_and_stats(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")

    ledger.record_started("seg-success", phase="warmup")
    ledger.record_accepted("seg-success", reward=1, phase="warmup")
    ledger.record_committed("seg-success", reward=1, phase="warmup", shard_path="/tmp/success.npz", num_replay_transitions=3)
    ledger.record_started("seg-failure", phase="warmup")
    ledger.record_accepted("seg-failure", reward=0, phase="warmup")
    ledger.record_committed("seg-failure", reward=0, phase="warmup", shard_path="/tmp/failure.npz", num_replay_transitions=2)

    stats = ledger.stats()
    assert stats["warmup_count"] == 2
    assert stats["warmup_success"] == 1
    assert stats["warmup_failure"] == 1
    assert stats["warmup_invalid"] == 0

    ledger.void_segment("seg-success", reason="wrong_bounds")
    stats = ledger.stats()
    assert stats["warmup_count"] == 1
    assert stats["warmup_success"] == 0
    assert stats["warmup_failure"] == 1
    assert stats["warmup_invalid"] == 1
    assert ledger.get_segment("seg-success")["status"] == "voided"


def test_segment_ledger_is_idempotent_for_duplicate_commit(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_started("seg", phase="warmup")
    ledger.record_accepted("seg", reward=1, phase="warmup")
    ledger.record_committed("seg", reward=1, phase="warmup", shard_path="/tmp/a.npz", num_replay_transitions=1)
    ledger.record_committed("seg", reward=1, phase="warmup", shard_path="/tmp/a.npz", num_replay_transitions=1)

    assert ledger.stats()["warmup_count"] == 1


def test_segment_ledger_crop_updates_committed_shard_path_and_transition_count(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_committed("seg", reward=1, phase="warmup", shard_path="/tmp/raw.npz", num_replay_transitions=8)

    ledger.record_cropped(
        "seg",
        reward=1,
        phase="warmup",
        shard_path="/tmp/clean.npz",
        num_replay_transitions=4,
        reason="operator_crop",
    )

    segment = ledger.get_segment("seg")
    assert segment["status"] == "committed"
    assert segment["shard_path"] == "/tmp/clean.npz"
    assert segment["num_replay_transitions"] == 4
    assert ledger.stats()["warmup_count"] == 1


def test_segment_ledger_lists_and_batch_restores_segments(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_committed("seg-a", reward=1, phase="warmup", shard_path="/tmp/a.npz", num_replay_transitions=3)
    ledger.record_committed("seg-b", reward=0, phase="warmup", shard_path="/tmp/b.npz", num_replay_transitions=2)
    ledger.void_segment("seg-b", reason="wrong_bounds")

    listed = ledger.list_segments()
    assert [item["key_region_id"] for item in listed] == ["seg-b", "seg-a"]
    assert listed[0]["status"] == "voided"

    changed = ledger.restore_segments(["seg-b"], reason="reviewed_ok")

    assert changed == ["seg-b"]
    assert ledger.get_segment("seg-b")["status"] == "committed"
    assert ledger.stats()["warmup_count"] == 2


def test_segment_ledger_refuses_restore_without_shard_path(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_discarded("seg", phase="warmup", reason="bad_start")

    assert ledger.restore_segments(["seg"], reason="reviewed_ok") == []
    assert ledger.get_segment("seg")["status"] == "discarded"


def test_segment_ledger_batch_void_only_changes_committed_existing_segments(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_committed("committed", reward=1, phase="warmup", shard_path="/tmp/ok.npz", num_replay_transitions=3)
    ledger.record_rejected("rejected", phase="warmup", reason="too_short")

    changed = ledger.void_segments(["committed", "rejected", "missing"], reason="batch_review")

    assert changed == ["committed"]
    assert ledger.get_segment("committed")["status"] == "voided"
    assert ledger.get_segment("rejected")["status"] == "rejected"
    assert ledger.get_segment("missing") is None


def test_segment_ledger_rescore_updates_reward_and_stats(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_committed("seg", reward=1, phase="warmup", shard_path="/tmp/a.npz", num_replay_transitions=3)

    ledger.record_rescored(
        "seg",
        reward=0,
        phase="warmup",
        shard_path="/tmp/a.npz",
        num_replay_transitions=3,
        reason="operator_rescore",
    )

    segment = ledger.get_segment("seg")
    assert segment["status"] == "committed"
    assert segment["reward"] == 0
    assert segment["shard_path"] == "/tmp/a.npz"
    assert ledger.stats()["warmup_success"] == 0
    assert ledger.stats()["warmup_failure"] == 1


def test_segment_ledger_records_quality_review_without_changing_binary_reward(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_committed("seg", reward=0, phase="rl", shard_path="/tmp/a.npz", num_replay_transitions=5)

    ledger.record_quality_review(
        "seg",
        quality_score=2,
        jitter_level="smooth",
        actor_train_mode="low_weight",
        quality_final=0.5,
        source="ui",
        notes="near miss",
    )

    segment = ledger.get_segment("seg")
    assert segment["reward"] == 0
    assert segment["quality_score"] == 2
    assert segment["jitter_level"] == "smooth"
    assert segment["actor_train_mode"] == "low_weight"
    assert segment["quality_final"] == 0.5
    assert segment["quality_source"] == "human"
    assert segment["quality_notes"] == "near miss"


def test_segment_ledger_delete_blocks_late_commit_ack(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")
    ledger.record_started("seg", phase="rl")

    changed = ledger.delete_segments(["seg"])
    ledger.record_committed("seg", reward=0, phase="rl", shard_path="/tmp/late.npz", num_replay_transitions=7)

    assert changed == ["seg"]
    segment = ledger.get_segment("seg")
    assert segment["status"] == "deleted"
    assert segment["shard_path"] is None
    assert ledger.stats()["auto_rollout_count"] == 0


def test_segment_ledger_delete_tombstones_missing_segment(tmp_path):
    ledger = RLTSegmentLedger(tmp_path / "segments.sqlite3")

    changed = ledger.delete_segments(["seg"])
    ledger.record_committed("seg", reward=1, phase="warmup", shard_path="/tmp/late.npz", num_replay_transitions=3)

    assert changed == ["seg"]
    assert ledger.get_segment("seg")["status"] == "deleted"
    assert ledger.stats()["warmup_count"] == 0

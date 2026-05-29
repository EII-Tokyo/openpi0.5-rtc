
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

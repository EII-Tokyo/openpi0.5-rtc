
from voice_assistant_web.backend.app.rlt_control import RLTControlStore
from voice_assistant_web.backend.app.rlt_segment_ledger import RLTSegmentLedger
from voice_assistant_web.backend.app.schemas import RLTControlRequest
from voice_assistant_web.backend.app.schemas import RLTControlState


class _FakeRedis:
    def __init__(self):
        self.messages = []

    def publish(self, channel, payload):
        self.messages.append((channel, payload))


class _Store(RLTControlStore):
    def _load(self) -> None:
        return

    def _persist_locked(self) -> None:
        return


def _store(*, warmup_target=2, tmp_path=None):
    ledger = RLTSegmentLedger(":memory:" if tmp_path is None else tmp_path / "segments.sqlite3")
    store = _Store(_FakeRedis(), segment_ledger=ledger)
    store._state = RLTControlState(warmup_target=warmup_target)
    store._refresh_derived_locked()
    return store


def test_score_records_attempt_but_does_not_increment_valid_warmup_count():
    store = _store(warmup_target=1)

    store.start_key_region(RLTControlRequest(source="test"))
    store.end_key_region(RLTControlRequest(source="test"))
    state = store.score_key_region(1, source="test")

    assert state.phase == "pending_replay"
    assert state.warmup_attempts == 1
    assert state.warmup_count == 0
    assert state.warmup_success == 0
    assert state.actor_effective is False
    assert state.actor_locked_reason == "warmup"


def test_valid_replay_ack_increments_warmup_and_unlocks_only_when_actor_ready_and_balanced():
    store = _store(warmup_target=2)
    store.update_runtime_metrics(
        {
            "type": "rlt_replay_segment_written",
            "phase": "warmup",
            "reward": 1,
            "replay_ready": True,
            "replay_status": "written",
            "num_replay_transitions": 3,
        }
    )
    store.update_runtime_metrics(
        {
            "type": "rlt_replay_segment_written",
            "phase": "warmup",
            "reward": 0,
            "replay_ready": True,
            "replay_status": "written",
            "num_replay_transitions": 3,
        }
    )

    state = store.snapshot()
    assert state.warmup_count == 2
    assert state.warmup_success == 1
    assert state.warmup_failure == 1
    assert state.actor_effective is False
    assert state.actor_locked_reason == "actor_not_ready"

    state = store.update_config(type("Req", (), {"warmup_target": None, "beta": None, "intervention_scale": None, "max_delta": None, "wandb_url": None, "actor_enabled": True})())
    assert state.actor_effective is False
    assert state.actor_locked_reason == "actor_not_ready"

    store.update_runtime_metrics(
        {
            "type": "rlt_trainer_metrics",
            "latest_actor_path": "/tmp/actor",
            "latest_actor_step": 500,
        }
    )
    state = store.snapshot()
    assert state.actor_ready is True
    assert state.actor_effective is True
    assert state.actor_locked_reason is None


def test_invalid_replay_ack_does_not_increment_warmup_or_unlock_actor():
    store = _store(warmup_target=1)
    store.update_runtime_metrics(
        {
            "type": "rlt_replay_segment_written",
            "phase": "warmup",
            "reward": 1,
            "replay_ready": False,
            "replay_status": "too_short",
            "num_replay_transitions": 0,
        }
    )
    store.update_runtime_metrics(
        {
            "type": "rlt_trainer_metrics",
            "latest_actor_path": "/tmp/actor",
            "latest_actor_step": 500,
        }
    )
    state = store.update_config(type("Req", (), {"warmup_target": None, "beta": None, "intervention_scale": None, "max_delta": None, "wandb_url": None, "actor_enabled": True})())

    assert state.warmup_invalid == 1
    assert state.warmup_count == 0
    assert state.actor_ready is True
    assert state.actor_effective is False
    assert state.actor_locked_reason == "warmup"



def test_confirm_publishes_score_after_review(tmp_path):
    store = _store(warmup_target=1, tmp_path=tmp_path)
    store.start_key_region(RLTControlRequest(source="test"))
    store.end_key_region(RLTControlRequest(source="test"))
    reviewed = store.score_key_region(1, source="test")
    key_region_id = reviewed.active_key_region_id

    assert reviewed.phase == "pending_replay"
    assert not any('"type": "score"' in payload for _, payload in store._redis.messages)

    confirmed = store.confirm_key_region(source="test")

    assert confirmed.phase == "idle"
    assert any('"type": "score"' in payload for _, payload in store._redis.messages)
    assert store._segment_ledger.get_segment(key_region_id)["status"] == "accepted"


def test_discard_does_not_publish_score_or_increment_attempts(tmp_path):
    store = _store(warmup_target=1, tmp_path=tmp_path)
    store.start_key_region(RLTControlRequest(source="test"))
    state = store.discard_key_region(source="test", reason="bad_start")

    assert state.phase == "idle"
    assert state.warmup_attempts == 0
    assert not any('"type": "score"' in payload for _, payload in store._redis.messages)


def test_void_segment_removes_committed_segment_from_counts(tmp_path):
    store = _store(warmup_target=2, tmp_path=tmp_path)
    store.update_runtime_metrics(
        {
            "type": "rlt_replay_segment_committed",
            "key_region_id": "seg-1",
            "phase": "warmup",
            "reward": 1,
            "replay_ready": True,
            "num_replay_transitions": 3,
            "shard_path": "/tmp/seg-1.npz",
        }
    )
    assert store.snapshot().warmup_count == 1

    state = store.void_segment("seg-1", source="test", reason="wrong_bounds")

    assert state.warmup_count == 0
    assert state.warmup_invalid == 1
    assert store._segment_ledger.get_segment("seg-1")["status"] == "voided"

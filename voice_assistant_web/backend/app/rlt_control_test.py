import numpy as np
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




def _write_trainable_shard(path, *, reward: int, num_transitions: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_seq = np.zeros((num_transitions, 10), dtype=np.float32)
    done = np.zeros((num_transitions,), dtype=np.bool_)
    done[-1] = True
    reward_seq[-1, 9] = float(reward)
    np.savez(
        path,
        z_rl=np.zeros((num_transitions, 8), dtype=np.float32),
        proprio=np.zeros((num_transitions, 4), dtype=np.float32),
        action=np.zeros((num_transitions, 10, 14), dtype=np.float32),
        reference_action=np.zeros((num_transitions, 10, 14), dtype=np.float32),
        reward_seq=reward_seq,
        next_z_rl=np.zeros((num_transitions, 8), dtype=np.float32),
        next_proprio=np.zeros((num_transitions, 4), dtype=np.float32),
        next_reference_action=np.zeros((num_transitions, 10, 14), dtype=np.float32),
        done=done,
    )

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


def test_valid_replay_ack_increments_warmup_and_unlocks_only_when_actor_ready_and_balanced(tmp_path):
    store = _store(warmup_target=2, tmp_path=tmp_path)
    success_shard = tmp_path / "success.npz"
    failure_shard = tmp_path / "failure.npz"
    _write_trainable_shard(success_shard, reward=1)
    _write_trainable_shard(failure_shard, reward=0)
    store.update_runtime_metrics(
        {
            "type": "rlt_replay_segment_written",
            "phase": "warmup",
            "reward": 1,
            "replay_ready": True,
            "replay_status": "written",
            "num_replay_transitions": 3,
            "shard_path": str(success_shard),
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
            "shard_path": str(failure_shard),
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


def test_batch_void_and_restore_segments_update_counts(tmp_path):
    store = _store(warmup_target=2, tmp_path=tmp_path)
    for key_region_id, reward in (("seg-1", 1), ("seg-2", 0)):
        store.update_runtime_metrics(
            {
                "type": "rlt_replay_segment_committed",
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": reward,
                "replay_ready": True,
                "num_replay_transitions": 3,
                "shard_path": f"/tmp/{key_region_id}.npz",
            }
        )
    assert store.snapshot().warmup_count == 2

    state = store.void_segments(["seg-1", "seg-2"], source="test", reason="batch_review")
    assert state.warmup_count == 0
    assert state.warmup_invalid == 2

    state = store.restore_segments(["seg-1"], source="test", reason="reviewed_ok")
    assert state.warmup_count == 1
    assert state.warmup_success == 1
    assert state.warmup_invalid == 1


def test_trainer_metrics_update_actor_critic_diagnostics():
    store = _store(warmup_target=1)

    store.update_runtime_metrics(
        {
            "type": "rlt_trainer_metrics",
            "trainer_step": 120,
            "critic_loss": 1.25,
            "critic_q1_loss": 0.75,
            "critic_q2_loss": 0.5,
            "actor_loss": -0.4,
            "actor_q_value": 1.7,
            "actor_delta_norm": 0.03,
            "reference_q_value": 1.2,
            "q_advantage": 0.5,
            "q1_mean": 1.1,
            "q2_mean": 0.9,
            "target_q_mean": 1.0,
            "q_gap": 0.2,
            "actor_updated": True,
            "publish_actor": False,
            "beta": 8.0,
            "steps_per_sec": 2.5,
            "replay_size": 256,
            "replay_shards": 6,
            "bad_shards": 1,
            "success_episodes": 4,
            "failure_episodes": 3,
            "replay_action_horizon": 50,
            "train_action_horizon": 10,
            "timestamp": 1234.5,
        }
    )

    state = store.snapshot()

    assert state.trainer_step == 120
    assert state.critic_q1_loss == 0.75
    assert state.critic_q2_loss == 0.5
    assert state.actor_q_value == 1.7
    assert state.actor_delta_norm == 0.03
    assert state.reference_q_value == 1.2
    assert state.q_advantage == 0.5
    assert state.q1_mean == 1.1
    assert state.q2_mean == 0.9
    assert state.target_q_mean == 1.0
    assert state.q_gap == 0.2
    assert state.actor_updated is True
    assert state.publish_actor is False
    assert state.beta == 10.0
    assert state.steps_per_sec == 2.5
    assert state.success_episodes == 4
    assert state.failure_episodes == 3
    assert state.replay_action_horizon == 50
    assert state.train_action_horizon == 10
    assert state.rlt_metrics_timestamp == 1234.5


def test_config_update_publishes_critic_gate_settings():
    store = _store(warmup_target=1)

    request = type(
        "Req",
        (),
        {
            "warmup_target": None,
            "beta": None,
            "intervention_scale": None,
            "max_delta": None,
            "wandb_url": None,
            "actor_enabled": None,
            "critic_gate_enabled": True,
            "critic_gate_margin": 0.15,
            "critic_gate_temperature": 0.2,
        },
    )()
    state = store.update_config(request)

    assert state.critic_gate_enabled is True
    assert state.critic_gate_margin == 0.15
    assert state.critic_gate_temperature == 0.2
    payload = store._redis.messages[-1][1]
    assert '"critic_gate_enabled": true' in payload


def test_runtime_metrics_update_inference_gate_diagnostics():
    store = _store(warmup_target=1)

    store.update_runtime_metrics(
        {
            "type": "runtime_state",
            "inference_actor_active": True,
            "inference_delta_norm": 0.03,
            "inference_gate_reason": "critic_gate_actor_active",
            "key_region_probability": 0.9,
            "loaded_actor_step": 32,
            "inference_reference_q_value": 0.2,
            "inference_actor_q_value": 0.7,
            "inference_q_advantage": 0.5,
            "critic_ready": True,
        }
    )

    state = store.snapshot()
    assert state.inference_actor_active is True
    assert state.inference_delta_norm == 0.03
    assert state.inference_gate_reason == "critic_gate_actor_active"
    assert state.key_region_probability == 0.9
    assert state.loaded_actor_step == 32
    assert state.inference_reference_q_value == 0.2
    assert state.inference_actor_q_value == 0.7
    assert state.inference_q_advantage == 0.5
    assert state.critic_ready is True

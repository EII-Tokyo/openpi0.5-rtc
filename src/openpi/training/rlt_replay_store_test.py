import json
import sqlite3

import pytest

import numpy as np

from openpi.training import rlt_replay_store


def _arrays(num_transitions: int, *, reward: float = 1.0) -> dict[str, np.ndarray]:
    action = np.ones((num_transitions, 3, 2), dtype=np.float32)
    reward_seq = np.zeros((num_transitions, 3), dtype=np.float32)
    done = np.zeros((num_transitions,), dtype=np.bool_)
    done[-1] = True
    reward_seq[-1, -1] = reward
    return {
        "z_rl": np.ones((num_transitions, 4), dtype=np.float32),
        "proprio": np.ones((num_transitions, 5), dtype=np.float32),
        "action": action,
        "reference_action": action * 0.5,
        "reward_seq": reward_seq,
        "next_z_rl": np.ones((num_transitions, 4), dtype=np.float32) * 2,
        "next_proprio": np.ones((num_transitions, 5), dtype=np.float32) * 3,
        "next_reference_action": action * 0.25,
        "done": done,
    }


def _write_segment_db(db_path, rows):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE segments (
                key_region_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                reward INTEGER,
                shard_path TEXT,
                num_replay_transitions INTEGER NOT NULL DEFAULT 0,
                invalid_reason TEXT,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO segments (key_region_id, status, phase, reward, shard_path, num_replay_transitions) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_rlt_replay_store_loads_committed_shards_and_samples(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    np.savez(shards_dir / "shard_000000.npz", **_arrays(5, reward=1.0))
    np.savez(shards_dir / "shard_000001.npz", **_arrays(7, reward=0.0))
    (shards_dir / "shard_000002.npz.tmp").write_bytes(b"not committed")

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    added = store.scan()

    assert len(added) == 2
    assert store.stats.replay_size == 12
    assert store.stats.num_shards == 2
    assert store.stats.success_episodes == 1
    assert store.stats.failure_episodes == 1
    assert store.shape == rlt_replay_store.ReplayShape(
        z_dim=4,
        proprio_dim=5,
        action_horizon=3,
        action_dim=2,
    )

    batch = store.sample_batch(np.random.default_rng(0), batch_size=4)
    assert batch.x.shape == (4, 9)
    assert batch.action.shape == (4, 3, 2)
    assert batch.reference_action.shape == (4, 3, 2)
    assert batch.reward_seq.shape == (4, 3)
    assert batch.done.shape == (4,)
    assert batch.episode_success.shape == (4,)
    assert bool(np.any(np.asarray(batch.episode_success)))
    assert bool(np.any(~np.asarray(batch.episode_success)))


def test_rlt_replay_store_rejects_invalid_shards_and_loads_new_shards(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    bad_arrays = _arrays(3)
    bad_arrays.pop("next_z_rl")
    np.savez(shards_dir / "bad.npz", **bad_arrays)

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    assert store.scan() == []
    assert store.stats.bad_shards == 1
    assert store.stats.replay_size == 0

    np.savez(shards_dir / "good.npz", **_arrays(4))
    added = store.scan()

    assert len(added) == 1
    assert added[0].num_transitions == 4
    assert store.stats.replay_size == 4


def test_rlt_replay_store_samples_train_action_horizon_from_c10_replay(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    arrays = _arrays(5)
    arrays["action"] = np.ones((5, 10, 2), dtype=np.float32)
    arrays["reference_action"] = arrays["action"] * 0.5
    arrays["next_reference_action"] = arrays["action"] * 0.25
    arrays["reward_seq"] = np.zeros((5, 10), dtype=np.float32)
    arrays["reward_seq"][-1, 9] = 1.0
    np.savez(shards_dir / "c10_horizon.npz", **arrays)

    store = rlt_replay_store.RLTReplayStore(tmp_path, sample_action_horizon=10)
    store.scan()

    assert store.shape == rlt_replay_store.ReplayShape(z_dim=4, proprio_dim=5, action_horizon=10, action_dim=2)
    assert store.sample_shape == rlt_replay_store.ReplayShape(z_dim=4, proprio_dim=5, action_horizon=10, action_dim=2)
    batch = store.sample_batch(np.random.default_rng(0), batch_size=3)
    assert batch.action.shape == (3, 10, 2)
    assert batch.reference_action.shape == (3, 10, 2)
    assert batch.next_reference_action.shape == (3, 10, 2)
    assert batch.reward_seq.shape == (3, 10)


def test_rlt_replay_store_rejects_sample_horizon_longer_than_replay(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    np.savez(shards_dir / "short.npz", **_arrays(5))

    store = rlt_replay_store.RLTReplayStore(tmp_path, sample_action_horizon=4)

    assert store.scan() == []
    assert store.shape is None
    assert any("sample_action_horizon" in reason for reason in store.bad_shards().values())



def test_rlt_replay_store_rejects_incompatible_schema_manifest(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "train_chunk_horizon": 3,
        "policy_horizon": 50,
        "action_space": "pi_internal",
        "action_dim": 2,
        "reward_placement": "terminal_last_train_step",
    }
    np.savez(shards_dir / "bad_schema.npz", **_arrays(5), manifest=json.dumps(manifest))

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    assert store.scan() == []
    assert store.stats.bad_shards == 1
    assert any("action_space" in reason for reason in store.bad_shards().values())


def test_rlt_replay_store_filters_by_segment_ledger(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    accepted = shards_dir / "accepted.npz"
    voided = shards_dir / "voided.npz"
    np.savez(accepted, **_arrays(5, reward=1.0), manifest=json.dumps({"schema_version": 1, "train_eligible": True, "voided": False}))
    np.savez(voided, **_arrays(7, reward=0.0), manifest=json.dumps({"schema_version": 1, "train_eligible": True, "voided": False}))
    db_path = tmp_path / "segments.sqlite3"
    _write_segment_db(
        db_path,
        [
            ("accepted", "committed", "warmup", 1, str(accepted), 5),
            ("voided", "voided", "warmup", 0, str(voided), 7),
        ],
    )

    store = rlt_replay_store.RLTReplayStore(tmp_path, segment_db_path=db_path)
    added = store.scan()

    assert [info.path for info in added] == [accepted.resolve()]
    assert store.stats.replay_size == 5
    assert store.stats.success_episodes == 1
    assert store.stats.failure_episodes == 0


def test_rlt_replay_store_evicts_loaded_shard_when_ledger_voids_it(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard = shards_dir / "accepted.npz"
    np.savez(shard, **_arrays(5, reward=1.0), manifest=json.dumps({"schema_version": 1, "train_eligible": True, "voided": False}))
    db_path = tmp_path / "segments.sqlite3"
    _write_segment_db(db_path, [("accepted", "committed", "warmup", 1, str(shard), 5)])

    store = rlt_replay_store.RLTReplayStore(tmp_path, segment_db_path=db_path)
    store.scan()
    assert store.stats.replay_size == 5

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE segments SET status='voided' WHERE key_region_id='accepted'")

    store.scan()

    assert store.stats.replay_size == 0
    assert store.loaded_paths == ()


def test_rlt_replay_store_rejects_manifest_marked_not_train_eligible(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    manifest = {"schema_version": 1, "train_eligible": False, "voided": False}
    np.savez(shards_dir / "bad_manifest.npz", **_arrays(5), manifest=json.dumps(manifest))

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    assert store.scan() == []
    assert any("train_eligible" in reason for reason in store.bad_shards().values())


def test_rlt_replay_store_rejects_manifest_marked_voided(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    manifest = {"schema_version": 1, "train_eligible": True, "voided": True}
    np.savez(shards_dir / "voided_manifest.npz", **_arrays(5), manifest=json.dumps(manifest))

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    assert store.scan() == []
    assert any("voided" in reason for reason in store.bad_shards().values())

def test_rlt_replay_store_ready_can_require_committed_shard_count(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    np.savez(shards_dir / "success.npz", **_arrays(5, reward=1.0))
    np.savez(shards_dir / "failure.npz", **_arrays(7, reward=0.0))

    store = rlt_replay_store.RLTReplayStore(tmp_path)
    store.scan()

    assert store.ready(
        min_replay_samples=12,
        min_replay_shards=2,
        min_success_episodes=1,
        min_failure_episodes=1,
    )
    assert not store.ready(
        min_replay_samples=12,
        min_replay_shards=3,
        min_success_episodes=1,
        min_failure_episodes=1,
    )

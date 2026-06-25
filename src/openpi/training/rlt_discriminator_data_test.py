import json

import numpy as np
import pytest

from openpi.training import rlt_discriminator_data as disc_data


def _write_shard(path, *, reward: int, n: int = 5, z_dim: int = 4, proprio_dim: int = 3, horizon: int = 2, action_dim: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_seq = np.zeros((n, horizon), dtype=np.float32)
    reward_seq[-1, -1] = float(reward)
    done = np.zeros((n,), dtype=np.bool_)
    done[-1] = True
    manifest = {
        "key_region_id": path.stem,
        "reward": reward,
        "num_replay_transitions": n,
    }
    np.savez(
        path,
        z_rl=np.arange(n * z_dim, dtype=np.float32).reshape(n, z_dim),
        proprio=np.ones((n, proprio_dim), dtype=np.float32),
        action=np.ones((n, horizon, action_dim), dtype=np.float32) * (2.0 if reward else -2.0),
        reference_action=np.zeros((n, horizon, action_dim), dtype=np.float32),
        reward_seq=reward_seq,
        next_z_rl=np.zeros((n, z_dim), dtype=np.float32),
        next_proprio=np.zeros((n, proprio_dim), dtype=np.float32),
        next_reference_action=np.zeros((n, horizon, action_dim), dtype=np.float32),
        done=done,
        manifest=json.dumps(manifest),
    )


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_transition_dataset_and_feature_variants(tmp_path):
    shard_a = tmp_path / "2026-06-17" / "shards" / "key_region_a.npz"
    shard_b = tmp_path / "2026-06-19" / "shards" / "key_region_b.npz"
    _write_shard(shard_a, reward=1)
    _write_shard(shard_b, reward=0)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"shard_path": str(shard_a), "batch": "2026-06-17", "reward": 1},
            {"shard_path": str(shard_b), "batch": "2026-06-19", "reward": 0},
        ],
    )

    dataset = disc_data.load_transition_dataset(manifest)

    assert len(dataset.rows) == 10
    assert dataset.z_rl.shape == (10, 4)
    assert dataset.proprio.shape == (10, 3)
    assert dataset.action.shape == (10, 2, 2)
    assert dataset.next_z_rl.shape == (10, 4)
    assert dataset.next_proprio.shape == (10, 3)
    assert dataset.labels.tolist() == [1] * 5 + [0] * 5
    assert set(dataset.sources.tolist()) == {"2026-06-17", "2026-06-19"}
    assert disc_data.build_features(dataset, "state_action").shape == (10, 11)
    assert disc_data.build_features(dataset, "state_only").shape == (10, 7)
    assert disc_data.build_features(dataset, "state_action_next").shape == (10, 18)
    assert disc_data.build_features(dataset, "state_action_next_delta").shape == (10, 25)
    assert disc_data.build_features(dataset, "state_next_only").shape == (10, 21)
    assert disc_data.build_features(dataset, "shuffled_action_next_delta").shape == (10, 25)
    shuffled = disc_data.build_features(dataset, "shuffled_action", rng=np.random.default_rng(0))
    assert shuffled.shape == (10, 11)


def test_episode_and_leave_one_source_splits_do_not_leak(tmp_path):
    rows = []
    for index, source in enumerate(["2026-06-17", "2026-06-17", "2026-06-19", "2026-06-22"]):
        shard = tmp_path / source / "shards" / f"key_region_{index}.npz"
        _write_shard(shard, reward=index % 2)
        rows.append({"shard_path": str(shard), "batch": source, "reward": index % 2})
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    dataset = disc_data.load_transition_dataset(manifest)

    split = disc_data.episode_random_split(dataset, holdout_ratio=0.25, seed=0)
    train_eps = set(dataset.episode_ids[split.train_indices])
    holdout_eps = set(dataset.episode_ids[split.holdout_indices])
    assert train_eps.isdisjoint(holdout_eps)

    source_splits = disc_data.leave_one_source_out_splits(dataset)
    assert {split.name for split in source_splits} == {"holdout_2026-06-17", "holdout_2026-06-19", "holdout_2026-06-22"}
    for split in source_splits:
        holdout_sources = set(dataset.sources[split.holdout_indices])
        assert len(holdout_sources) == 1
        assert not holdout_sources.intersection(set(dataset.sources[split.train_indices]))


def test_metrics_include_auc_gap_and_balanced_accuracy():
    labels = np.asarray([1, 1, 0, 0])
    probs = np.asarray([0.9, 0.8, 0.4, 0.2])

    metrics = disc_data.binary_classification_metrics(labels, probs)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mean_D_success"] == pytest.approx(0.85)
    assert metrics["mean_D_failure"] == pytest.approx(0.3)
    assert metrics["D_gap"] == pytest.approx(0.55)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)

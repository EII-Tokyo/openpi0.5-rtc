import json

import numpy as np
import pytest

from openpi.training import rlt_terminal_return as terminal_return


def _write_shard(path, *, reward: int, n: int = 5, z_dim: int = 4, proprio_dim: int = 3, horizon: int = 2, action_dim: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_seq = np.zeros((n, horizon), dtype=np.float32)
    reward_seq[-1, -1] = float(reward)
    done = np.zeros((n,), dtype=np.bool_)
    done[-1] = True
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
        manifest=json.dumps({"key_region_id": path.stem, "reward": reward}),
    )


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_dataset_builds_discounted_terminal_targets(tmp_path):
    success = tmp_path / "2026-06-17" / "shards" / "success.npz"
    failure = tmp_path / "2026-06-19" / "shards" / "failure.npz"
    _write_shard(success, reward=1, n=4)
    _write_shard(failure, reward=0, n=4)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"shard_path": str(success), "batch": "2026-06-17"},
            {"shard_path": str(failure), "batch": "2026-06-19"},
        ],
    )

    dataset = terminal_return.load_terminal_return_dataset(manifest, gamma=0.9)

    assert dataset.targets[:4].tolist() == pytest.approx([0.9**3, 0.9**2, 0.9, 1.0])
    assert dataset.targets[4:].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert dataset.labels.tolist() == [1] * 4 + [0] * 4
    assert dataset.transition_indices.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert dataset.progress.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0, 0.0, 1 / 3, 2 / 3, 1.0])


def test_critical_ratio_keeps_last_part_of_each_episode(tmp_path):
    shard = tmp_path / "2026-06-17" / "shards" / "success.npz"
    _write_shard(shard, reward=1, n=10)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [{"shard_path": str(shard), "batch": "2026-06-17"}])

    dataset = terminal_return.load_terminal_return_dataset(manifest, gamma=0.99, critical_ratio=0.4)

    assert dataset.transition_indices.tolist() == [6, 7, 8, 9]
    assert dataset.targets.tolist() == pytest.approx([0.99**3, 0.99**2, 0.99, 1.0])


def test_split_metrics_rank_success_above_failure():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.8, 0.7, 0.2, 0.1])
    targets = np.asarray([1.0, 0.9, 0.0, 0.0])

    metrics = terminal_return.score_metrics(labels, scores, targets)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["q_gap"] == pytest.approx(0.6)
    assert metrics["mse"] == pytest.approx(np.mean((scores - targets) ** 2))


def test_intra_source_episode_splits_stay_within_each_source(tmp_path):
    rows = []
    for source in ["2026-06-17", "2026-06-19"]:
        for index in range(6):
            shard = tmp_path / source / "shards" / f"{source}_{index}.npz"
            _write_shard(shard, reward=index % 2)
            rows.append({"shard_path": str(shard), "batch": source})
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    dataset = terminal_return.load_terminal_return_dataset(manifest, gamma=0.99)

    splits = terminal_return.intra_source_episode_splits(dataset, holdout_ratio=0.33, seed=0)

    assert {split.holdout_source for split in splits} == {"2026-06-17", "2026-06-19"}
    for split in splits:
        train_eps = set(dataset.episode_ids[split.train_indices])
        holdout_eps = set(dataset.episode_ids[split.holdout_indices])
        assert train_eps.isdisjoint(holdout_eps)
        assert set(dataset.sources[split.train_indices]) == {split.holdout_source}
        assert set(dataset.sources[split.holdout_indices]) == {split.holdout_source}
        assert set(dataset.labels[split.train_indices]) == {0, 1}
        assert set(dataset.labels[split.holdout_indices]) == {0, 1}

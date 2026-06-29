from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpi.training.rlt_replay_store import RLTReplayStore
from scripts.augment_no_actor_q_replay import AugmentArgs
from scripts.augment_no_actor_q_replay import augment_no_actor_q_replay


def _write_shard(path: Path, *, reward: int, action_offset: float, rows: int = 6) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    z_rl = np.arange(rows * 8, dtype=np.float32).reshape(rows, 8) + action_offset
    proprio = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    action = np.full((rows, 3, 2), action_offset, dtype=np.float32)
    for row in range(rows):
        action[row] += row * 0.1
    reward_seq = np.zeros((rows, 3), dtype=np.float32)
    reward_seq[-1, -1] = float(reward)
    done = np.zeros((rows,), dtype=np.bool_)
    done[-1] = True
    manifest = {
        "schema_version": 1,
        "shard_path": str(path),
        "reward": reward,
        "train_eligible": True,
        "voided": False,
        "policy_horizon": 3,
        "train_horizon": 3,
        "action_dim": 2,
        "action_space": "aloha_exec",
        "reward_placement": "terminal_last_train_step",
        "source_type": "test",
        "label": "source",
        "rl_token_checkpoint_path": "checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/run/9999",
        "rl_token_config_name": "eii_rinse_11repo_cam4_fullft_rl_token_small_query",
        "z_cache_path": str(path.with_suffix(".z_cache.npz")),
        "z_rl_dim": 8,
    }
    np.savez_compressed(
        path,
        z_rl=z_rl,
        proprio=proprio,
        action=action,
        reference_action=action.copy(),
        reward_seq=reward_seq,
        next_z_rl=z_rl.copy(),
        next_proprio=proprio.copy(),
        next_reference_action=action.copy(),
        done=done,
        manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return manifest


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_augment_success_shard_writes_dense_reward(tmp_path):
    source = tmp_path / "source" / "success.npz"
    source_manifest = _write_shard(source, reward=1, action_offset=1.0)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, [source_manifest])

    output_manifest = tmp_path / "augmented.jsonl"
    summary = augment_no_actor_q_replay(
        AugmentArgs(
            manifest_path=manifest,
            output_root=tmp_path / "augmented",
            output_manifest_path=output_manifest,
            dense_start_progress=0.5,
            dense_min_reward=0.2,
            create_hard_negatives=False,
            overwrite=True,
        )
    )

    assert summary.copied_shards == 1
    assert summary.negative_shards == 0
    row = json.loads(output_manifest.read_text(encoding="utf-8").splitlines()[0])
    with np.load(row["shard_path"], allow_pickle=False) as data:
        reward_seq = data["reward_seq"]
        assert reward_seq.shape == (6, 3)
        assert float(reward_seq[:3].sum()) == 0.0
        assert float(reward_seq[3].sum()) == pytest.approx(0.2)
        assert float(reward_seq[4].sum()) == pytest.approx(0.6)
        assert float(reward_seq[5].sum()) == pytest.approx(1.0)
        assert data["done"].tolist() == [False, False, False, False, False, True]
        manifest_data = json.loads(str(data["manifest"].item()))
    assert manifest_data["reward_placement"] == "terminal_last_train_step"
    assert manifest_data["augmentation"]["dense_reward"] is True
    assert manifest_data["augmentation"]["reward_mode"] == "dense_progress_terminal"
    assert manifest_data["rl_token_checkpoint_path"] == source_manifest["rl_token_checkpoint_path"]
    assert manifest_data["rl_token_config_name"] == source_manifest["rl_token_config_name"]
    assert manifest_data["z_cache_path"] == source_manifest["z_cache_path"]
    assert manifest_data["z_rl_dim"] == source_manifest["z_rl_dim"]


def test_augment_success_shard_adds_hard_negative_with_mismatched_actions(tmp_path):
    success_a = tmp_path / "source" / "success_a.npz"
    success_b = tmp_path / "source" / "success_b.npz"
    rows = [
        _write_shard(success_a, reward=1, action_offset=1.0),
        _write_shard(success_b, reward=1, action_offset=9.0),
    ]
    source_manifest = tmp_path / "source.jsonl"
    _write_manifest(source_manifest, rows)

    output_manifest = tmp_path / "augmented.jsonl"
    summary = augment_no_actor_q_replay(
        AugmentArgs(
            manifest_path=source_manifest,
            output_root=tmp_path / "augmented",
            output_manifest_path=output_manifest,
            create_hard_negatives=True,
            hard_negative_ratio=1.0,
            seed=0,
            overwrite=True,
        )
    )

    assert summary.copied_shards == 2
    assert summary.negative_shards == 2
    manifest_rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]
    negative_rows = [row for row in manifest_rows if row["label"] == "hard_negative_action_mismatch"]
    assert len(negative_rows) == 2
    with np.load(negative_rows[0]["shard_path"], allow_pickle=False) as negative:
        assert np.all(negative["reward_seq"] == 0.0)
        assert bool(negative["done"][-1])
        assert np.max(np.abs(negative["action"] - negative["reference_action"])) > 1.0
        manifest_data = json.loads(str(negative["manifest"].item()))
    assert manifest_data["reward"] == 0
    assert manifest_data["source_type"] == "hard_negative"
    assert manifest_data["actor_enabled"] is False
    assert manifest_data["rl_token_checkpoint_path"] == rows[0]["rl_token_checkpoint_path"]
    assert manifest_data["rl_token_config_name"] == rows[0]["rl_token_config_name"]
    assert manifest_data["z_cache_path"] == rows[0]["z_cache_path"]
    assert manifest_data["z_rl_dim"] == rows[0]["z_rl_dim"]

    store = RLTReplayStore(tmp_path / "augmented", manifest_path=output_manifest)
    store.scan()
    assert store.stats.success_episodes == 2
    assert store.stats.failure_episodes == 2
    assert store.stats.bad_shards == 0

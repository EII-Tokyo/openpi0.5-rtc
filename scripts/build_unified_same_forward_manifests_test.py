from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import build_unified_same_forward_manifests as build


def _write_shard(path: Path, *, key_region_id: str, reward: int, transitions: int = 3) -> None:
    manifest = {
        "key_region_id": key_region_id,
        "replay_state_grain": "paper_subsampled_anchor",
        "z_rl_dim": 2048,
        "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
        "num_replay_transitions": transitions,
        "reward": reward,
    }
    np.savez(
        path,
        z_rl=np.zeros((transitions, 2048), dtype=np.float32),
        next_z_rl=np.ones((transitions, 2048), dtype=np.float32),
        action=np.zeros((transitions, 10, 14), dtype=np.float32),
        reward_seq=np.zeros((transitions, 10), dtype=np.float32),
        manifest=json.dumps(manifest),
    )


def _row(shard_path: Path, key_region_id: str, reward: int) -> dict:
    return {
        "key_region_id": key_region_id,
        "reward": reward,
        "shard_path": str(shard_path),
    }


def test_normalize_manifest_entry_reads_formal_metadata_from_npz(tmp_path: Path) -> None:
    shard = tmp_path / "key_region_a.npz"
    _write_shard(shard, key_region_id="a", reward=1, transitions=5)

    entry = build.normalize_manifest_entry(_row(shard, "a", 1), source_group="original_train")

    assert entry["key_region_id"] == "a"
    assert entry["source_group"] == "original_train"
    assert entry["replay_state_grain"] == "paper_subsampled_anchor"
    assert entry["z_dim"] == 2048
    assert entry["num_transitions"] == 5
    assert entry["z_rl_source"] == "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder"


def test_stratified_holdout_split_preserves_reward_classes() -> None:
    entries = [
        {"key_region_id": f"s{i}", "reward": 1, "source_group": "base142"} for i in range(5)
    ] + [
        {"key_region_id": f"f{i}", "reward": 0, "source_group": "base142"} for i in range(5)
    ]

    train, holdout = build.stratified_holdout_split(entries, holdout_ratio=0.2, seed=7)

    assert len(holdout) == 2
    assert {row["reward"] for row in holdout} == {0, 1}
    assert not {row["key_region_id"] for row in train} & {row["key_region_id"] for row in holdout}


def test_build_outputs_all_train_and_eval_train_with_reserved_holdouts(tmp_path: Path) -> None:
    manifest_paths = {}
    for label, rewards in {
        "original_train": [1, 0],
        "original_holdout": [1],
        "base142": [1, 1, 0, 0],
        "actor93": [1, 0, 0, 1],
    }.items():
        rows = []
        for index, reward in enumerate(rewards):
            key_region_id = f"{label}_{index}"
            shard = tmp_path / f"{key_region_id}.npz"
            _write_shard(shard, key_region_id=key_region_id, reward=reward)
            rows.append(_row(shard, key_region_id, reward))
        manifest = tmp_path / f"{label}.jsonl"
        build.write_jsonl(manifest, rows)
        manifest_paths[label] = manifest

    result = build.build_unified_manifests(
        original_train_manifest=manifest_paths["original_train"],
        original_holdout_manifest=manifest_paths["original_holdout"],
        base142_manifest=manifest_paths["base142"],
        actor93_manifest=manifest_paths["actor93"],
        output_dir=tmp_path / "out",
        holdout_ratio=0.5,
        seed=11,
    )

    all_train = build.read_jsonl(result["train_all"])
    eval_train = build.read_jsonl(result["train_eval"])
    original_holdout = build.read_jsonl(result["holdout_original"])
    combined_holdout = build.read_jsonl(result["holdout_combined"])

    assert len(all_train) == 11
    assert len(eval_train) == 6
    assert {row["key_region_id"] for row in original_holdout} == {"original_holdout_0"}
    assert "original_holdout_0" not in {row["key_region_id"] for row in eval_train}
    assert {row["key_region_id"] for row in eval_train}.isdisjoint(
        {row["key_region_id"] for row in combined_holdout}
    )

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_canonical_rlt_2048_dataset import (
    BuildArgs,
    SourceSpec,
    build_canonical_dataset,
    parse_source_spec,
)


def _write_shard(
    path: Path,
    *,
    key_region_id: str,
    z_dim: int = 2048,
    rows: int = 3,
    reward: float = 1.0,
    z_rl_source: str = "rl_token_reencoded",
    replay_state_grain: str = "paper_subsampled_anchor",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "key_region_id": key_region_id,
        "task": "twist_off_the_bottle_cap",
        "date": "2026-07-02",
        "reward": reward,
        "z_rl_source": z_rl_source,
        "replay_state_grain": replay_state_grain,
        "rl_token_config_name": "eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer",
        "rl_token_checkpoint_path": "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint",
    }
    rewards = np.zeros((rows,), dtype=np.float32)
    rewards[-1] = reward
    np.savez_compressed(
        path,
        z_rl=np.zeros((rows, z_dim), dtype=np.float32),
        next_z_rl=np.ones((rows, z_dim), dtype=np.float32),
        action=np.zeros((rows, 10, 14), dtype=np.float32),
        reference_action=np.zeros((rows, 10, 14), dtype=np.float32),
        reward=rewards,
        manifest=np.asarray(json.dumps(manifest)),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_parse_source_spec_requires_five_fields(tmp_path: Path) -> None:
    spec = parse_source_spec(f"rlt_raw|train|103|batch_a|{tmp_path}")

    assert spec == SourceSpec(
        kind="rlt_raw",
        split="train",
        machine="103",
        batch="batch_a",
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="kind\\|split\\|machine\\|batch\\|root"):
        parse_source_spec("rlt_raw|train|103")


def test_build_canonical_dataset_writes_split_manifests(tmp_path: Path) -> None:
    train_root = tmp_path / "src_train"
    holdout_root = tmp_path / "src_holdout"
    _write_shard(train_root / "shards" / "key_region_train.npz", key_region_id="train")
    _write_shard(
        holdout_root / "shards" / "key_region_holdout.npz",
        key_region_id="holdout",
        reward=0.0,
    )

    summary = build_canonical_dataset(
        BuildArgs(
            canonical_root=tmp_path / "canonical",
            manifest_root=tmp_path / "manifests",
            sources=[
                SourceSpec("bootstrap", "train", "103", "batch_a", train_root),
                SourceSpec("bootstrap", "holdout", "103", "batch_a", holdout_root),
            ],
        )
    )

    assert summary["total_rows"] == 2
    assert summary["by_split"] == {"holdout": 1, "train": 1}
    train_rows = _read_jsonl(tmp_path / "manifests" / "canonical_2048_train.jsonl")
    holdout_rows = _read_jsonl(tmp_path / "manifests" / "canonical_2048_holdout.jsonl")
    assert [row["key_region_id"] for row in train_rows] == ["train"]
    assert [row["key_region_id"] for row in holdout_rows] == ["holdout"]
    assert Path(train_rows[0]["canonical_path"]).exists()
    assert train_rows[0]["z_dim"] == 2048
    assert train_rows[0]["rl_token_config"] == "eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer"
    assert holdout_rows[0]["reward"] == 0.0


def test_build_canonical_dataset_rejects_legacy_512_shards(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    _write_shard(source_root / "shards" / "key_region_legacy.npz", key_region_id="legacy", z_dim=512)

    summary = build_canonical_dataset(
        BuildArgs(
            canonical_root=tmp_path / "canonical",
            manifest_root=tmp_path / "manifests",
            sources=[SourceSpec("rlt_clean", "unsplit", "local", "legacy", source_root)],
        )
    )

    assert summary["total_rows"] == 0
    assert summary["skipped"]["invalid_z_dim"] == 1
    assert _read_jsonl(tmp_path / "manifests" / "canonical_2048_all.jsonl") == []


def test_build_canonical_dataset_rejects_fixed_segments_shards(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    _write_shard(
        source_root / "shards" / "key_region_fixed.npz",
        key_region_id="fixed",
        z_rl_source="rl_token_reencoded_aligned_to_proprio_segments",
        replay_state_grain="proprio_segment_aligned",
    )

    summary = build_canonical_dataset(
        BuildArgs(
            canonical_root=tmp_path / "canonical",
            manifest_root=tmp_path / "manifests",
            sources=[SourceSpec("bootstrap", "train", "103", "fixed_segments", source_root)],
        )
    )

    assert summary["total_rows"] == 0
    assert summary["skipped"]["fixed_segments_not_paper_subsampled"] == 1
    assert _read_jsonl(tmp_path / "manifests" / "canonical_2048_all.jsonl") == []


def test_build_canonical_dataset_records_duplicate_key_region_ids(tmp_path: Path) -> None:
    source_a = tmp_path / "src_a"
    source_b = tmp_path / "src_b"
    _write_shard(source_a / "shards" / "key_region_same_a.npz", key_region_id="same")
    _write_shard(source_b / "shards" / "key_region_same_b.npz", key_region_id="same")

    summary = build_canonical_dataset(
        BuildArgs(
            canonical_root=tmp_path / "canonical",
            manifest_root=tmp_path / "manifests",
            sources=[
                SourceSpec("rlt_clean", "unsplit", "local", "a", source_a),
                SourceSpec("expert", "unsplit", "local", "b", source_b),
            ],
        )
    )

    rows = _read_jsonl(tmp_path / "manifests" / "canonical_2048_all.jsonl")
    assert summary["duplicate_key_region_ids"] == {"same": 2}
    assert len(rows) == 2
    assert {row["kind"] for row in rows} == {"rlt_clean", "expert"}

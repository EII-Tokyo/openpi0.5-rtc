from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.rlt_replay_store import RLTReplayStore
from scripts.convert_expert_crops_to_q_replay import ConversionArgs
from scripts.convert_expert_crops_to_q_replay import convert_expert_crops


def _write_lerobot_dataset(root: Path, dataset_id: str, *, frames: int = 30) -> Path:
    dataset_dir = root / dataset_id
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(json.dumps({"fps": 10}), encoding="utf-8")

    rows = list(range(frames))
    states = [[float(i + j * 0.01) for j in range(14)] for i in rows]
    actions = [[float(i + j * 0.1) for j in range(14)] for i in rows]
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), list_size=14)),
            "action": pa.array(actions, type=pa.list_(pa.float32(), list_size=14)),
            "timestamp": pa.array([i / 10.0 for i in rows], type=pa.float64()),
            "frame_index": pa.array(rows, type=pa.int64()),
            "episode_index": pa.array([0 for _ in rows], type=pa.int64()),
            "index": pa.array(rows, type=pa.int64()),
        }
    )
    pq.write_table(table, dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    return dataset_dir


def _write_crop(crop_root: Path, dataset_id: str, *, end_sec: float = 3.0, reward: int = 1) -> Path:
    output_dir = crop_root / dataset_id
    output_dir.mkdir(parents=True)
    crop_path = output_dir / "episode_000000_crop_000000.json"
    crop_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "start_sec": 0.0,
                "end_sec": end_sec,
                "reward": reward,
                "label": "expert",
            }
        ),
        encoding="utf-8",
    )
    return crop_path


def test_convert_expert_crop_writes_no_actor_q_replay_shard(tmp_path):
    dataset_id = "demo-rinse"
    dataset_root = tmp_path / "hf"
    crop_root = tmp_path / "crops"
    output_root = tmp_path / "q_replay"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_lerobot_dataset(dataset_root, dataset_id, frames=30)
    _write_crop(crop_root, dataset_id, end_sec=3.0, reward=1)

    summary = convert_expert_crops(
        ConversionArgs(
            dataset_root=dataset_root,
            crop_root=crop_root,
            output_root=output_root,
            manifest_path=manifest_path,
            allow_dummy_z=True,
            train_horizon=10,
            chunk_stride=2,
            proprio_dim=32,
            z_dim=512,
        )
    )

    assert summary.converted == 1
    assert summary.skipped == {}
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    shard_path = Path(rows[0]["shard_path"])
    assert shard_path.exists()

    with np.load(shard_path, allow_pickle=True) as data:
        assert data["z_rl"].shape == (6, 512)
        assert data["proprio"].shape == (6, 32)
        assert data["action"].shape == (6, 10, 14)
        assert data["reference_action"].shape == (6, 10, 14)
        np.testing.assert_allclose(data["action"], data["reference_action"])
        np.testing.assert_allclose(data["next_reference_action"][0], data["reference_action"][5])
        assert data["reward_seq"].shape == (6, 10)
        assert float(data["reward_seq"][-1, -1]) == 1.0
        assert float(data["reward_seq"][:-1].sum()) == 0.0
        assert data["done"].tolist() == [False, False, False, False, False, True]
        manifest = json.loads(str(data["manifest"].item()))
    assert manifest["source_type"] == "human_expert"
    assert manifest["actor_enabled"] is False
    assert manifest["rlt_actor_applied_ratio"] == 0.0
    assert manifest["action_reference_delta"]["all_max_abs"] == 0.0
    assert manifest["z_rl_source"] == "dummy_deterministic_not_for_training"

    store = RLTReplayStore(output_root, manifest_path=manifest_path)
    store.scan()
    assert store.stats.replay_size == 6
    assert store.stats.success_episodes == 1
    assert store.stats.failure_episodes == 0


def test_convert_expert_crop_skips_too_short_segments(tmp_path):
    dataset_id = "demo-rinse"
    dataset_root = tmp_path / "hf"
    crop_root = tmp_path / "crops"
    output_root = tmp_path / "q_replay"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_lerobot_dataset(dataset_root, dataset_id, frames=12)
    _write_crop(crop_root, dataset_id, end_sec=1.2, reward=1)

    summary = convert_expert_crops(
        ConversionArgs(
            dataset_root=dataset_root,
            crop_root=crop_root,
            output_root=output_root,
            manifest_path=manifest_path,
            allow_dummy_z=True,
            train_horizon=10,
            chunk_stride=2,
        )
    )

    assert summary.converted == 0
    assert summary.skipped == {"too_short": 1}
    assert not manifest_path.exists()


def test_convert_expert_crop_uses_precomputed_z_cache(tmp_path):
    dataset_id = "demo-rinse"
    dataset_root = tmp_path / "hf"
    crop_root = tmp_path / "crops"
    output_root = tmp_path / "q_replay"
    z_cache_root = tmp_path / "z_cache"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_lerobot_dataset(dataset_root, dataset_id, frames=30)
    _write_crop(crop_root, dataset_id, end_sec=3.0, reward=1)
    cache_dir = z_cache_root / dataset_id
    cache_dir.mkdir(parents=True)
    z_rl = np.arange(30 * 8, dtype=np.float32).reshape(30, 8)
    np.savez(cache_dir / "episode_000000_z_rl.npz", frame_index=np.arange(30, dtype=np.int64), z_rl=z_rl)

    summary = convert_expert_crops(
        ConversionArgs(
            dataset_root=dataset_root,
            crop_root=crop_root,
            output_root=output_root,
            manifest_path=manifest_path,
            z_cache_root=z_cache_root,
            train_horizon=10,
            chunk_stride=2,
            proprio_dim=32,
            z_dim=8,
        )
    )

    assert summary.converted == 1
    shard_path = Path(json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])["shard_path"])
    with np.load(shard_path, allow_pickle=True) as data:
        np.testing.assert_allclose(data["z_rl"][0], z_rl[0])
        np.testing.assert_allclose(data["next_z_rl"][0], z_rl[10])
        manifest = json.loads(str(data["manifest"].item()))
    assert manifest["z_rl_source"] == "precomputed_frame_cache"

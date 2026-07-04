from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_expert_crops import AuditArgs, audit_expert_crops


def _write_dataset(root: Path, dataset_id: str, *, frames: int = 32) -> None:
    dataset_dir = root / dataset_id
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    rows = list(range(frames))
    table = pa.table(
        {
            "episode_index": pa.array([0] * frames, type=pa.int64()),
            "frame_index": pa.array(rows, type=pa.int64()),
            "index": pa.array(rows, type=pa.int64()),
            "timestamp": pa.array([i / 10.0 for i in rows], type=pa.float64()),
            "observation.state": pa.array([[float(i)] * 14 for i in rows], type=pa.list_(pa.float32(), list_size=14)),
            "action": pa.array([[float(i)] * 14 for i in rows], type=pa.list_(pa.float32(), list_size=14)),
        }
    )
    pq.write_table(table, dataset_dir / "data" / "chunk-000" / "file-000.parquet")


def _write_crop(root: Path, dataset_id: str) -> None:
    crop_dir = root / dataset_id
    crop_dir.mkdir(parents=True)
    (crop_dir / "episode_000000_crop_000000.json").write_text(
        json.dumps({"dataset_id": dataset_id, "episode_index": 0, "start_sec": 0.0, "end_sec": 3.0, "reward": 1}),
        encoding="utf-8",
    )


def _write_cache(root: Path, dataset_id: str, *, z_dim: int, config: str) -> None:
    cache_dir = root / dataset_id
    cache_dir.mkdir(parents=True)
    np.savez(
        cache_dir / "episode_000000_z_rl.npz",
        frame_index=np.arange(32, dtype=np.int64),
        z_rl=np.zeros((32, z_dim), dtype=np.float32),
        metadata=np.asarray(
            json.dumps(
                {
                    "rl_token_config_name": config,
                    "rl_token_checkpoint_path": "checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint",
                }
            )
        ),
    )


def test_audit_expert_crops_accepts_matching_lower_right_cache(tmp_path: Path) -> None:
    dataset_id = "demo"
    _write_dataset(tmp_path / "hf", dataset_id)
    _write_crop(tmp_path / "crops", dataset_id)
    _write_cache(
        tmp_path / "cache",
        dataset_id,
        z_dim=2048,
        config="eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer",
    )

    summary = audit_expert_crops(
        AuditArgs(
            dataset_root=tmp_path / "hf",
            crop_root=tmp_path / "crops",
            z_cache_root=tmp_path / "cache",
            require_camera=(),
        )
    )

    assert summary["crop_count"] == 1
    assert summary["z_cache"]["files"] == 1
    assert summary["issues"] == {}
    assert summary["is_usable"] is True


def test_audit_expert_crops_flags_wrong_cache_dim_and_config(tmp_path: Path) -> None:
    dataset_id = "demo"
    _write_dataset(tmp_path / "hf", dataset_id)
    _write_crop(tmp_path / "crops", dataset_id)
    _write_cache(tmp_path / "cache", dataset_id, z_dim=512, config="old_config")

    summary = audit_expert_crops(
        AuditArgs(
            dataset_root=tmp_path / "hf",
            crop_root=tmp_path / "crops",
            z_cache_root=tmp_path / "cache",
            require_camera=(),
        )
    )

    assert summary["issues"]["bad_z_cache_dim"] == 1
    assert summary["issues"]["bad_z_cache_metadata"] >= 1
    assert summary["is_usable"] is False

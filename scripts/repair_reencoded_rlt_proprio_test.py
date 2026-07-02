from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.repair_reencoded_rlt_proprio import RepairArgs
from scripts.repair_reencoded_rlt_proprio import discover_shards
from scripts.repair_reencoded_rlt_proprio import repair_reencoded_proprio


def _write_shard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "key_region_id": "abc",
        "task": "twist_off_the_bottle_cap",
        "date": "2026-07-01",
        "phase": "rl",
        "train_horizon": 10,
        "chunk_stride": 2,
    }
    np.savez_compressed(
        path,
        z_rl=np.zeros((2, 2048), dtype=np.float32),
        proprio=np.zeros((2, 32), dtype=np.float32),
        action=np.zeros((2, 10, 14), dtype=np.float32),
        reference_action=np.zeros((2, 10, 14), dtype=np.float32),
        reward_seq=np.zeros((2, 10), dtype=np.float32),
        next_z_rl=np.zeros((2, 2048), dtype=np.float32),
        next_proprio=np.zeros((2, 32), dtype=np.float32),
        next_reference_action=np.zeros((2, 10, 14), dtype=np.float32),
        done=np.array([False, True]),
        manifest=np.asarray(json.dumps(manifest)),
    )


def test_discover_shards_returns_npz_files(tmp_path: Path) -> None:
    shard = tmp_path / "input" / "shards" / "a.npz"
    _write_shard(shard)
    (tmp_path / "input" / "ignore.txt").write_text("not a shard", encoding="utf-8")

    assert discover_shards(tmp_path / "input") == [shard]


def test_dry_run_reports_shards_without_writing(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _write_shard(input_root / "shards" / "a.npz")

    summary = repair_reencoded_proprio(
        RepairArgs(
            input_root=input_root,
            rollout_root=tmp_path / "rollouts",
            output_root=tmp_path / "output",
            execute=False,
        )
    )

    assert summary.planned == 1
    assert summary.converted == 0
    assert not (tmp_path / "output").exists()

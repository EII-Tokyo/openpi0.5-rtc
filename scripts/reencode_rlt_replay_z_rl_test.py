from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reencode_rlt_replay_z_rl import (
    ReencodeReplayArgs,
    discover_replay_shards,
    reencode_rlt_replay,
)


def _write_shard(path: Path, *, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "key_region_id": "abc",
        "task": "twist_off_the_bottle_cap",
        "date": "2026-07-01",
        "phase": "rl",
    }
    np.savez_compressed(
        path,
        z_rl=np.zeros((rows, 512), dtype=np.float32),
        next_z_rl=np.zeros((rows, 512), dtype=np.float32),
        action=np.ones((rows, 10, 14), dtype=np.float32),
        reference_action=np.zeros((rows, 10, 14), dtype=np.float32),
        manifest=np.asarray(json.dumps(manifest)),
    )


def test_discover_replay_shards_keeps_actor_affected_data(tmp_path: Path) -> None:
    shard = tmp_path / "replay" / "task" / "2026-07-01" / "shards" / "key_region_abc.npz"
    _write_shard(shard)

    discovered = discover_replay_shards(tmp_path / "replay")

    assert discovered == [shard]


def test_dry_run_reports_all_replay_shards_without_writing(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    _write_shard(replay_root / "shards" / "key_region_abc.npz")

    summary = reencode_rlt_replay(
        ReencodeReplayArgs(
            replay_root=replay_root,
            rollout_root=tmp_path / "rollouts",
            output_root=tmp_path / "out",
            execute=False,
        )
    )

    assert summary.planned == 1
    assert summary.converted == 0
    assert not (tmp_path / "out").exists()

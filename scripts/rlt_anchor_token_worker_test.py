from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.training import rlt_anchor_token_cache as cache
from scripts import rlt_anchor_token_worker as worker


def _source_arrays(rows: int = 2) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.zeros((rows, 4), dtype=np.float32),
        "proprio": np.zeros((rows, 2), dtype=np.float32),
        "action": np.ones((rows, 3, 2), dtype=np.float32),
        "reference_action": np.ones((rows, 3, 2), dtype=np.float32) * 2,
        "reward_seq": np.zeros((rows, 3), dtype=np.float32),
        "next_z_rl": np.zeros((rows, 4), dtype=np.float32),
        "next_proprio": np.zeros((rows, 2), dtype=np.float32),
        "next_reference_action": np.ones((rows, 3, 2), dtype=np.float32) * 3,
        "done": np.asarray([False, True], dtype=np.bool_)[:rows],
    }


def test_candidate_from_job_uses_paper_anchor_fields(tmp_path: Path) -> None:
    source = tmp_path / "key_region_a.npz"
    rollout = tmp_path / "rollout" / "key_region_a"
    source.write_bytes(b"placeholder")
    rollout.mkdir(parents=True)
    job = cache.write_pending_job(
        job_root=tmp_path / "jobs",
        manifest={
            "key_region_id": "a",
            "reward": 0,
            "num_frames": 29,
            "num_replay_transitions": 5,
            "train_horizon": 10,
            "chunk_stride": 2,
        },
        rollout_dir=rollout,
        source_shard_path=source,
    )

    candidate = worker.candidate_from_job(job.payload)

    assert candidate.key_region_id == "a"
    assert candidate.source_shard_path == source.resolve()
    assert candidate.rollout_dir == rollout.resolve()
    assert candidate.reward == 0
    assert candidate.num_frames == 29
    assert candidate.num_replay_transitions == 5
    assert candidate.train_horizon == 10
    assert candidate.chunk_stride == 2
    assert candidate.collection_group == "async_anchor_token_job"


def test_assemble_ready_caches_moves_jobs_to_ready_and_appends_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source" / "key_region_a.npz"
    source.parent.mkdir()
    np.savez_compressed(
        source,
        **_source_arrays(rows=2),
        manifest=json.dumps(
            {
                "key_region_id": "a",
                "reward": 1,
                "replay_state_grain": "runtime_action_cache_block",
                "requires_offline_reencode": True,
                "train_horizon": 3,
                "chunk_stride": 1,
            }
        ),
    )
    job = cache.write_pending_job(
        job_root=tmp_path / "jobs",
        manifest={
            "key_region_id": "a",
            "reward": 1,
            "num_frames": 5,
            "num_replay_transitions": 2,
            "train_horizon": 3,
            "chunk_stride": 1,
        },
        rollout_dir=tmp_path / "rollout" / "key_region_a",
        source_shard_path=source,
    )
    encoded_cache = tmp_path / "encoded_cache" / "key_region_a.npz"
    encoded_cache.parent.mkdir()
    np.savez_compressed(
        encoded_cache,
        z_rl=np.ones((2, 4), dtype=np.float32),
        next_z_rl=np.ones((2, 4), dtype=np.float32) * 2,
        proprio=np.ones((2, 2), dtype=np.float32) * 3,
        next_proprio=np.ones((2, 2), dtype=np.float32) * 4,
        current_frames=np.asarray([0, 1], dtype=np.int64),
        next_frames=np.asarray([3, 4], dtype=np.int64),
    )

    summary = worker.assemble_ready_caches(
        job_root=tmp_path / "jobs",
        encoded_cache_root=tmp_path / "encoded_cache",
        output_root=tmp_path / "formal",
        manifest_path=tmp_path / "formal_manifest.jsonl",
        limit=None,
        overwrite=False,
    )

    assert summary == {"assembled": 1, "missing_cache": 0, "failed": 0}
    assert not job.path.exists()
    ready_job = tmp_path / "jobs" / "ready" / "key_region_a.json"
    assert ready_job.exists()
    ready_payload = json.loads(ready_job.read_text(encoding="utf-8"))
    assert ready_payload["formal_shard_path"].endswith("key_region_a.npz")
    rows = [json.loads(line) for line in (tmp_path / "formal_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["replay_state_grain"] == "paper_subsampled_anchor"

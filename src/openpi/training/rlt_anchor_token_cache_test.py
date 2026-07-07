from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.training import rlt_anchor_token_cache as cache


def _arrays(rows: int = 3) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.full((rows, 4), -1.0, dtype=np.float32),
        "proprio": np.full((rows, 2), -2.0, dtype=np.float32),
        "action": np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3),
        "reference_action": np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3) + 0.5,
        "reward_seq": np.zeros((rows, 2), dtype=np.float32),
        "next_z_rl": np.full((rows, 4), -3.0, dtype=np.float32),
        "next_proprio": np.full((rows, 2), -4.0, dtype=np.float32),
        "next_reference_action": np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3) + 1.5,
        "done": np.asarray([False, False, True], dtype=np.bool_)[:rows],
    }


def test_write_pending_job_records_anchor_semantics(tmp_path: Path) -> None:
    job = cache.write_pending_job(
        job_root=tmp_path / "jobs",
        manifest={
            "key_region_id": "abc",
            "task": "twist",
            "phase": "rl",
            "reward": 1,
            "num_frames": 42,
            "num_replay_transitions": 12,
            "train_horizon": 10,
            "chunk_stride": 2,
            "replay_state_grain": "runtime_action_cache_block",
        },
        rollout_dir=tmp_path / "rollouts" / "key_region_abc",
        source_shard_path=tmp_path / "replay" / "key_region_abc.npz",
    )

    assert job.path == tmp_path / "jobs" / "pending" / "key_region_abc.json"
    payload = json.loads(job.path.read_text(encoding="utf-8"))
    assert payload["status"] == "pending"
    assert payload["key_region_id"] == "abc"
    assert payload["rollout_dir"].endswith("key_region_abc")
    assert payload["source_runtime_cache_block_shard_path"].endswith("key_region_abc.npz")
    assert payload["formal_replay_state_grain"] == "paper_subsampled_anchor"
    assert payload["subsampled_transition_semantics"] == "x_i_action_i_to_i_plus_c_next_x_i_plus_c"
    assert payload["train_horizon"] == 10
    assert payload["chunk_stride"] == 2


def test_assemble_formal_replay_from_encoded_cache_replaces_runtime_state(tmp_path: Path) -> None:
    source = tmp_path / "source" / "key_region_abc.npz"
    source.parent.mkdir()
    source_manifest = {
        "key_region_id": "abc",
        "reward": 1,
        "replay_state_grain": "runtime_action_cache_block",
        "requires_offline_reencode": True,
        "train_horizon": 2,
        "chunk_stride": 1,
    }
    np.savez_compressed(source, **_arrays(rows=3), manifest=json.dumps(source_manifest))
    token_cache = tmp_path / "cache" / "key_region_abc.npz"
    token_cache.parent.mkdir()
    np.savez_compressed(
        token_cache,
        z_rl=np.full((3, 4), 10.0, dtype=np.float32),
        next_z_rl=np.full((3, 4), 20.0, dtype=np.float32),
        proprio=np.full((3, 2), 30.0, dtype=np.float32),
        next_proprio=np.full((3, 2), 40.0, dtype=np.float32),
        current_frames=np.asarray([0, 1, 2], dtype=np.int64),
        next_frames=np.asarray([2, 3, 4], dtype=np.int64),
    )
    job = cache.write_pending_job(
        job_root=tmp_path / "jobs",
        manifest={**source_manifest, "num_frames": 5, "num_replay_transitions": 3},
        rollout_dir=tmp_path / "rollouts" / "key_region_abc",
        source_shard_path=source,
    )

    result = cache.assemble_formal_replay_from_encoded_cache(
        job_path=job.path,
        encoded_cache_path=token_cache,
        output_root=tmp_path / "formal",
        overwrite=False,
    )

    with np.load(result.shard_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["z_rl"], np.full((3, 4), 10.0, dtype=np.float32))
        np.testing.assert_array_equal(data["next_z_rl"], np.full((3, 4), 20.0, dtype=np.float32))
        np.testing.assert_array_equal(data["proprio"], np.full((3, 2), 30.0, dtype=np.float32))
        np.testing.assert_array_equal(data["action"], _arrays(rows=3)["action"])
        manifest = json.loads(str(data["manifest"]))

    assert result.manifest["shard_path"] == str(result.shard_path.resolve())
    assert manifest["replay_state_grain"] == "paper_subsampled_anchor"
    assert manifest["requires_offline_reencode"] is False
    assert manifest["formal_replay_ready"] is True
    assert manifest["train_eligible"] is True
    assert manifest["z_rl_source"] == "async_anchor_token_cache_vla_same_forward"
    assert manifest["current_frames"] == [0, 1, 2]
    assert manifest["next_frames"] == [2, 3, 4]

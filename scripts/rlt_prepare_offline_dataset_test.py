import json

import numpy as np

from scripts import rlt_prepare_offline_dataset


def test_prepare_offline_dataset_crops_rescores_and_writes_manifest_jsonl(tmp_path):
    raw_root = tmp_path / "raw"
    edited_root = tmp_path / "edited"
    rollout_dir = raw_root / "rollouts" / "key_regions" / "rinse" / "2026-06-18" / "warmup" / "key_region_a"
    shard_path = raw_root / "replay" / "rlt_key_regions" / "rinse" / "2026-06-18" / "shards" / "key_region_a.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True)
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": "a",
                "task": "rinse",
                "phase": "warmup",
                "reward": 0,
                "duration_seconds": 3.0,
                "num_frames": 150,
                "num_replay_transitions": 66,
                "train_horizon": 10,
                "chunk_stride": 2,
                "train_eligible": True,
                "voided": False,
                "shard_path": str(shard_path),
            }
        )
    )
    reward_seq = np.zeros((66, 10), dtype=np.float32)
    np.savez(
        shard_path,
        z_rl=np.arange(66, dtype=np.float32).reshape(66, 1),
        proprio=np.arange(66, dtype=np.float32).reshape(66, 1),
        reward_seq=reward_seq,
        done=np.asarray([False] * 65 + [True]),
        manifest=json.dumps({"key_region_id": "a"}),
    )
    edits_path = tmp_path / "edits.jsonl"
    edits_path.write_text(
        json.dumps({"key_region_id": "a", "start_sec": 1.0, "end_sec": 2.0, "reward": 1}) + "\n"
    )

    summary = rlt_prepare_offline_dataset.prepare_offline_dataset(
        raw_root=raw_root,
        output_root=edited_root,
        edits_path=edits_path,
    )

    assert summary["written"] == 1
    output_shard = edited_root / "replay" / "rlt_key_regions" / "rinse" / "2026-06-18" / "shards" / "key_region_a.npz"
    assert output_shard.exists()
    with np.load(output_shard, allow_pickle=False) as cropped:
        assert cropped["z_rl"][:, 0].tolist() == list(range(25, 51))
        assert float(cropped["reward_seq"][-1, 9]) == 1.0
    manifest_jsonl = output_shard.parent.parent / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_jsonl.read_text().splitlines()]
    assert rows[0]["key_region_id"] == "a"
    assert rows[0]["reward"] == 1
    assert rows[0]["num_replay_transitions"] == 26

import json

import numpy as np

from voice_assistant_web.backend.app.rlt_key_region_crop import crop_key_region_files, rescore_key_region_files


def test_crop_key_region_replay_shard_keeps_selected_samples_and_terminal_reward(tmp_path):
    rollout_dir = tmp_path / "rollouts" / "key_regions/task/2026-06-01/warmup/key_region_crop"
    shard_path = tmp_path / "replay" / "rlt_key_regions/task/2026-06-01/shards/key_region_crop.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True)
    manifest_path = rollout_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "key_region_id": "crop",
                "phase": "warmup",
                "reward": 1,
                "duration_seconds": 4.0,
                "num_replay_transitions": 8,
                "segment_status": "committed",
                "train_eligible": True,
                "shard_path": str(shard_path),
                "train_horizon": 10,
            }
        )
    )
    np.savez(
        shard_path,
        z_rl=np.arange(8 * 2).reshape(8, 2),
        proprio=np.arange(8 * 3).reshape(8, 3),
        reward_seq=np.zeros((8, 50), dtype=np.float32),
        done=np.asarray([False, False, False, False, False, False, False, True]),
        manifest=json.dumps({"key_region_id": "crop"}),
    )

    clean_shard_path = shard_path.parent / "key_region_crop.crop_123.npz"
    result = crop_key_region_files(rollout_dir, shard_path, clean_shard_path, start_sec=1.0, end_sec=3.0)

    with np.load(clean_shard_path, allow_pickle=False) as cropped:
        assert cropped["z_rl"].shape[0] == 4
        assert cropped["z_rl"][0].tolist() == [4, 5]
        assert cropped["z_rl"][-1].tolist() == [10, 11]
        assert cropped["done"].tolist() == [False, False, False, True]
        assert float(cropped["reward_seq"][-1, 9]) == 1.0
    with np.load(shard_path, allow_pickle=False) as raw:
        assert raw["z_rl"].shape[0] == 8
    manifest = json.loads(manifest_path.read_text())
    assert result["num_replay_transitions"] == 4
    assert result["shard_path"] == str(clean_shard_path)
    assert result["source_shard_path"] == str(shard_path)
    assert manifest["crop_start_sec"] == 1.0
    assert manifest["crop_end_sec"] == 3.0
    assert manifest["duration_seconds"] == 4.0
    assert manifest["crop_duration_seconds"] == 2.0
    assert manifest["crop_original_num_replay_transitions"] == 8
    assert manifest["num_replay_transitions"] == 4
    assert manifest["shard_path"] == str(clean_shard_path)


def test_crop_key_region_replay_shard_rejects_invalid_range(tmp_path):
    rollout_dir = tmp_path / "key_region_crop"
    shard_path = tmp_path / "key_region_crop.npz"
    rollout_dir.mkdir()
    (rollout_dir / "manifest.json").write_text(json.dumps({"key_region_id": "crop", "duration_seconds": 4.0}))
    np.savez(shard_path, z_rl=np.arange(4), done=np.asarray([False, False, False, True]))

    try:
        crop_key_region_files(rollout_dir, shard_path, tmp_path / "clean.npz", start_sec=2.0, end_sec=2.0)
    except ValueError as exc:
        assert "end_sec must be greater" in str(exc)
    else:
        raise AssertionError("expected invalid crop range to fail")


def test_rescore_key_region_replay_shard_rewrites_terminal_reward(tmp_path):
    rollout_dir = tmp_path / "rollouts" / "key_region_rescore"
    shard_path = tmp_path / "replay" / "key_region_rescore.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True)
    manifest_path = rollout_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "key_region_id": "rescore",
                "phase": "warmup",
                "reward": 1,
                "duration_seconds": 2.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
                "shard_path": str(shard_path),
                "train_horizon": 10,
            }
        )
    )
    reward_seq = np.zeros((3, 10), dtype=np.float32)
    reward_seq[0, 9] = 1.0
    reward_seq[2, 9] = 1.0
    np.savez(
        shard_path,
        z_rl=np.zeros((3, 2), dtype=np.float32),
        reward_seq=reward_seq,
        done=np.asarray([False, False, True]),
        manifest=json.dumps({"key_region_id": "rescore", "reward": 1}),
    )

    result = rescore_key_region_files(rollout_dir, shard_path, reward=0)

    with np.load(shard_path, allow_pickle=False) as rescored:
        assert rescored["reward_seq"].sum() == 0.0
        npz_manifest = json.loads(str(rescored["manifest"]))
        assert npz_manifest["reward"] == 0
    manifest = json.loads(manifest_path.read_text())
    assert result["reward"] == 0
    assert manifest["reward"] == 0
    assert manifest["score_timeout"] is False

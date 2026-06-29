from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reencode_clean_no_actor_z_rl import compute_replay_frame_indices
from scripts.reencode_clean_no_actor_z_rl import dedupe_no_actor_shards
from scripts.reencode_clean_no_actor_z_rl import is_no_actor_shard
from scripts.reencode_clean_no_actor_z_rl import rewrite_shard_z_rl
from scripts.reencode_clean_no_actor_z_rl import validate_required_cameras


def _write_replay_shard(
    path: Path,
    *,
    actor_delta: float = 0.0,
    key_region_id: str = "abc",
    source_shard_path: str | None = None,
    crop_end_sample: int = 7,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 4
    horizon = 10
    reference = np.arange(n * horizon * 14, dtype=np.float32).reshape(n, horizon, 14)
    action = reference.copy()
    action[0, 0, 0] += actor_delta
    manifest = {
        "key_region_id": key_region_id,
        "task": "twist_off_the_bottle_cap",
        "date": "2026-06-17",
        "phase": "warmup",
        "crop_start_sample": 3,
        "crop_end_sample": crop_end_sample,
        "crop_original_num_replay_transitions": 10,
        "train_horizon": horizon,
        "chunk_stride": 2,
        "reward": 1,
    }
    if source_shard_path is not None:
        manifest["source_shard_path"] = source_shard_path
    np.savez_compressed(
        path,
        z_rl=np.ones((n, 512), dtype=np.float32),
        proprio=np.ones((n, 32), dtype=np.float32),
        action=action,
        reference_action=reference,
        reward_seq=np.zeros((n, horizon), dtype=np.float32),
        next_z_rl=np.ones((n, 512), dtype=np.float32) * 2,
        next_proprio=np.ones((n, 32), dtype=np.float32) * 3,
        next_reference_action=reference + 1,
        done=np.array([False, False, False, True]),
        manifest=np.asarray(json.dumps(manifest)),
    )


def test_is_no_actor_shard_uses_action_reference_delta(tmp_path):
    no_actor = tmp_path / "no_actor.npz"
    actor = tmp_path / "actor.npz"
    _write_replay_shard(no_actor, actor_delta=0.0)
    _write_replay_shard(actor, actor_delta=0.25)

    assert is_no_actor_shard(no_actor, threshold=1e-6)
    assert not is_no_actor_shard(actor, threshold=1e-6)


def test_compute_replay_frame_indices_maps_clean_rows_to_original_episode_frames():
    manifest = {
        "crop_start_sample": 3,
        "crop_end_sample": 7,
        "crop_original_num_replay_transitions": 10,
        "train_horizon": 10,
        "chunk_stride": 2,
    }

    current, nxt = compute_replay_frame_indices(manifest, clean_rows=4, episode_frames=38)

    np.testing.assert_array_equal(current, np.array([6, 8, 10, 12]))
    np.testing.assert_array_equal(nxt, np.array([16, 18, 20, 22]))


def test_rewrite_shard_z_rl_replaces_only_token_arrays_and_updates_manifest(tmp_path):
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "out" / "input.npz"
    _write_replay_shard(input_path)
    new_z = np.zeros((4, 2048), dtype=np.float32)
    new_next_z = np.ones((4, 2048), dtype=np.float32)

    rewrite_shard_z_rl(
        input_path,
        output_path,
        z_rl=new_z,
        next_z_rl=new_next_z,
        checkpoint_path=Path("/checkpoint/12000"),
        config_name="demo_config",
    )

    with np.load(input_path, allow_pickle=False) as before, np.load(output_path, allow_pickle=False) as after:
        np.testing.assert_allclose(after["z_rl"], new_z)
        np.testing.assert_allclose(after["next_z_rl"], new_next_z)
        for key in (
            "proprio",
            "action",
            "reference_action",
            "reward_seq",
            "next_proprio",
            "next_reference_action",
            "done",
        ):
            np.testing.assert_allclose(after[key], before[key])
        manifest = json.loads(str(after["manifest"].item()))

    assert manifest["z_rl_source"] == "rl_token_reencoded"
    assert manifest["z_rl_dim"] == 2048
    assert manifest["previous_z_rl_shape"] == [4, 512]
    assert manifest["rl_token_checkpoint_path"] == "/checkpoint/12000"
    assert manifest["rl_token_config_name"] == "demo_config"


def test_dedupe_no_actor_shards_keeps_latest_clean_version_per_source(tmp_path):
    source = "/raw/replay/key_region_abc.npz"
    old_manual = tmp_path / "manual" / "key_region_abc.crop_old.npz"
    latest_clean = tmp_path / "twist_off_the_bottle_cap" / "2026-06-17" / "shards" / "key_region_abc.crop_new.npz"
    other = tmp_path / "twist_off_the_bottle_cap" / "2026-06-17" / "shards" / "key_region_xyz.crop.npz"
    _write_replay_shard(old_manual, key_region_id="abc", source_shard_path=source, crop_end_sample=5)
    _write_replay_shard(latest_clean, key_region_id="abc", source_shard_path=source, crop_end_sample=8)
    _write_replay_shard(other, key_region_id="xyz", source_shard_path="/raw/replay/key_region_xyz.npz")

    selected = dedupe_no_actor_shards([old_manual, latest_clean, other])

    assert set(selected) == {other, latest_clean}


def test_validate_required_cameras_rejects_missing_cam_low():
    try:
        validate_required_cameras(("cam_high", "cam_left_wrist", "cam_right_wrist"), ("cam_low",))
    except ValueError as exc:
        assert "cam_low" in str(exc)
    else:
        raise AssertionError("expected missing cam_low to fail")

    validate_required_cameras(("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"), ("cam_low",))

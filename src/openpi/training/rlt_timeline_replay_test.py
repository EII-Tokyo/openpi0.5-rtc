import json

import h5py
import numpy as np
import pytest

from openpi.training import rlt_timeline_replay


def _write_timeline_hdf5(path, *, frames: int = 8) -> None:
    with h5py.File(path, "w") as root:
        root.attrs["key_region_id"] = "demo"
        root.attrs["reward"] = 1
        root.attrs["replay_state_grain"] = "raw_frame_timeline"
        root.attrs["behavior_policy"] = "rlt_actor"
        root.attrs["action_source"] = "rlt_actor_adjusted_action"
        root.attrs["reference_action_source"] = "vla_same_forward_reference_action"
        root.attrs["actor_checkpoint_path"] = "/app/local_rlt_runs/demo_actor/00004500"
        root.attrs["actor_checkpoint_step"] = 4500
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=np.arange(frames * 4, dtype=np.float32).reshape(frames, 4))
        obs.create_dataset("qvel", data=np.zeros((frames, 4), dtype=np.float32))
        obs.create_dataset("effort", data=np.zeros((frames, 4), dtype=np.float32))
        root.create_dataset("action", data=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        root.create_dataset("reference_action", data=100 + np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        timeline = root.create_group("rlt_timeline")
        timeline.attrs["state_grain"] = "raw_frame_timeline"
        timeline.attrs["z_rl_source"] = "vla_same_forward"
        timeline.attrs["rl_token_checkpoint_path"] = "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        timeline.attrs["behavior_policy"] = "rlt_actor"
        timeline.attrs["action_source"] = "rlt_actor_adjusted_action"
        timeline.attrs["reference_action_source"] = "vla_same_forward_reference_action"
        timeline.attrs["actor_checkpoint_path"] = "/app/local_rlt_runs/demo_actor/00004500"
        timeline.attrs["actor_checkpoint_step"] = 4500
        timeline.create_dataset("z_rl", data=1000 + np.arange(frames * 3, dtype=np.float32).reshape(frames, 3))
        timeline.create_dataset("proprio", data=2000 + np.arange(frames * 5, dtype=np.float32).reshape(frames, 5))
        timeline.create_dataset("step_index", data=np.arange(frames, dtype=np.int64))
        timeline.create_dataset("valid", data=np.ones((frames,), dtype=np.bool_))
        root.create_dataset("timestamps", data=np.arange(frames, dtype=np.float64))


def _write_policy_forward_event_hdf5(
    path,
    *,
    frames: int = 8,
    event_steps: tuple[int, ...] = (0, 2, 4, 6),
    include_policy_proprio: bool = False,
    include_qpos: bool = False,
) -> None:
    with h5py.File(path, "w") as root:
        root.attrs["key_region_id"] = "event-demo"
        root.attrs["reward"] = 1
        root.attrs["replay_state_grain"] = "raw_frame_timeline"
        root.attrs["behavior_policy"] = "rlt_actor"
        root.attrs["action_source"] = "rlt_actor_adjusted_action"
        root.attrs["reference_action_source"] = "vla_same_forward_reference_action"
        root.attrs["actor_checkpoint_path"] = "/app/local_rlt_runs/demo_actor/00004500"
        root.attrs["actor_checkpoint_step"] = 4500
        root.attrs["rl_token_checkpoint_path"] = "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        if include_qpos:
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=np.arange(frames * 14, dtype=np.float32).reshape(frames, 14))
        root.create_dataset("action", data=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        root.create_dataset("reference_action", data=100 + np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        timeline = root.create_group("rlt_timeline")
        timeline.attrs["state_grain"] = "raw_frame_timeline"
        timeline.attrs["z_rl_source"] = "policy_forward_events"
        timeline.attrs["behavior_policy"] = "rlt_actor"
        timeline.attrs["action_source"] = "rlt_actor_adjusted_action"
        timeline.attrs["reference_action_source"] = "vla_same_forward_reference_action"
        timeline.attrs["actor_checkpoint_path"] = "/app/local_rlt_runs/demo_actor/00004500"
        timeline.attrs["actor_checkpoint_step"] = 4500
        timeline.attrs["rl_token_checkpoint_path"] = "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        timeline.create_dataset("step_index", data=np.arange(frames, dtype=np.int64))
        timeline.create_dataset("valid", data=np.ones((frames,), dtype=np.bool_))
        if include_policy_proprio:
            timeline.create_dataset(
                "policy_proprio",
                data=3000 + np.arange(frames * 5, dtype=np.float32).reshape(frames, 5),
            )
        events = root.create_group("rlt_policy_forward_events")
        events.attrs["state_grain"] = "vla_same_forward_policy_forward"
        events.attrs["z_rl_source"] = "vla_same_forward_runtime_output"
        events.attrs["rl_token_checkpoint_path"] = "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        events.create_dataset("step_index", data=np.asarray(event_steps, dtype=np.int64))
        events.create_dataset("policy_forward_id", data=np.arange(len(event_steps), dtype=np.int64))
        events.create_dataset("action_start_index", data=np.zeros((len(event_steps),), dtype=np.int64))
        events.create_dataset(
            "z_rl",
            data=1000 + np.arange(len(event_steps) * 3, dtype=np.float32).reshape(len(event_steps), 3),
        )
        events.create_dataset(
            "proprio",
            data=2000 + np.arange(len(event_steps) * 5, dtype=np.float32).reshape(len(event_steps), 5),
        )
        events.create_dataset(
            "z_rl_source",
            data=np.asarray(["vla_same_forward_runtime_output"] * len(event_steps), dtype="S"),
        )


def _write_legacy_emission_event_hdf5(path, *, frames: int = 90) -> None:
    emission_steps = (10, 35, 60)
    action_start = (10, 10, 10)
    with h5py.File(path, "w") as root:
        root.attrs["key_region_id"] = "legacy-event-demo"
        root.attrs["reward"] = 1
        root.create_dataset("action", data=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        root.create_dataset("reference_action", data=100 + np.arange(frames * 2, dtype=np.float32).reshape(frames, 2))
        timeline = root.create_group("rlt_timeline")
        timeline.attrs["state_grain"] = "raw_frame_timeline"
        timeline.attrs["z_rl_source"] = "policy_forward_events"
        timeline.create_dataset("step_index", data=np.arange(frames, dtype=np.int64))
        timeline.create_dataset("valid", data=np.ones((frames,), dtype=np.bool_))
        events = root.create_group("rlt_policy_forward_events")
        events.attrs["state_grain"] = "vla_same_forward_policy_forward"
        events.attrs["z_rl_source"] = "vla_same_forward_runtime_output"
        events.create_dataset("step_index", data=np.asarray(emission_steps, dtype=np.int64))
        events.create_dataset("action_start_index", data=np.asarray(action_start, dtype=np.int64))
        events.create_dataset("z_rl", data=1000 + np.arange(9, dtype=np.float32).reshape(3, 3))
        events.create_dataset("proprio", data=2000 + np.arange(15, dtype=np.float32).reshape(3, 5))


def test_build_paper_replay_arrays_from_frame_timeline(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_timeline_hdf5(h5_path, frames=8)

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=2,
        chunk_stride=2,
        policy_event_alignment="exact_event_pairs",
    )

    assert arrays["action"].shape == (3, 2, 2)
    assert arrays["reference_action"].shape == (3, 2, 2)
    assert arrays["next_reference_action"].shape == (3, 2, 2)
    np.testing.assert_allclose(arrays["z_rl"][:, 0], [1000, 1006, 1012])
    np.testing.assert_allclose(arrays["next_z_rl"][:, 0], [1006, 1012, 1018])
    np.testing.assert_allclose(arrays["action"][1, :, 0], [4, 6])
    np.testing.assert_allclose(arrays["next_reference_action"][1, :, 0], [108, 110])
    assert arrays["done"].tolist() == [False, False, True]
    assert arrays["reward_seq"].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    assert manifest["replay_state_grain"] == "paper_subsampled_anchor"
    assert manifest["source_format"] == "rlt_timeline_hdf5"
    assert manifest["behavior_policy"] == "rlt_actor"
    assert manifest["action_source"] == "rlt_actor_adjusted_action"
    assert manifest["reference_action_source"] == "vla_same_forward_reference_action"
    assert manifest["actor_checkpoint_path"] == "/app/local_rlt_runs/demo_actor/00004500"
    assert manifest["actor_checkpoint_step"] == 4500
    assert manifest["rl_token_checkpoint_path"] == "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
    assert manifest["current_frames"] == [0, 2, 4]
    assert manifest["next_frames"] == [2, 4, 6]


def test_build_paper_replay_arrays_from_exact_policy_forward_events(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(h5_path, frames=8, event_steps=(0, 2, 4, 6))

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=2,
        chunk_stride=2,
        policy_event_alignment="exact_event_pairs",
    )

    assert arrays["action"].shape == (3, 2, 2)
    np.testing.assert_allclose(arrays["z_rl"][:, 0], [1000, 1003, 1006])
    np.testing.assert_allclose(arrays["next_z_rl"][:, 0], [1003, 1006, 1009])
    np.testing.assert_allclose(arrays["action"][1, :, 0], [4, 6])
    assert manifest["z_alignment"] == "policy_forward_event_exact_step_pairs"
    assert manifest["current_frames"] == [0, 2, 4]
    assert manifest["next_frames"] == [2, 4, 6]


def test_policy_forward_events_require_exact_next_event(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(h5_path, frames=8, event_steps=(0, 3, 6))

    with pytest.raises(ValueError, match="no exact policy-forward event pairs"):
        rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
            h5_path,
            train_horizon=2,
            chunk_stride=1,
            policy_event_alignment="exact_event_pairs",
        )


def test_policy_forward_events_can_share_trunk_tokens_for_short_stride(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(
        h5_path,
        frames=70,
        event_steps=(0, 25, 50),
        include_policy_proprio=True,
    )

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=10,
        chunk_stride=2,
        policy_event_alignment="trunk_shared",
    )

    assert arrays["action"].shape == (26, 10, 2)
    assert manifest["current_frames"][:4] == [0, 2, 4, 6]
    assert manifest["next_frames"][:4] == [10, 12, 14, 16]
    assert manifest["z_alignment"] == "policy_forward_event_trunk_shared"
    assert manifest["proprio_alignment"] == "rlt_timeline_policy_proprio"
    assert manifest["replay_state_grain"] == "trunk_shared_z_subsampled_anchor"
    np.testing.assert_allclose(arrays["z_rl"][:8, 0], np.full((8,), 1000, dtype=np.float32))
    np.testing.assert_allclose(arrays["next_z_rl"][:8, 0], np.full((8,), 1000, dtype=np.float32))
    np.testing.assert_allclose(arrays["next_z_rl"][8:13, 0], np.full((5,), 1003, dtype=np.float32))
    np.testing.assert_allclose(arrays["proprio"][:4, 0], [3000, 3010, 3020, 3030])
    np.testing.assert_allclose(arrays["next_proprio"][:4, 0], [3050, 3060, 3070, 3080])
    np.testing.assert_allclose(arrays["action"][1, :, 0], np.arange(4, 24, 2, dtype=np.float32))


def test_policy_forward_events_fill_conservative_provenance_defaults(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(
        h5_path,
        frames=70,
        event_steps=(0, 25, 50),
        include_policy_proprio=True,
    )
    with h5py.File(h5_path, "a") as root:
        for key in (
            "behavior_policy",
            "action_source",
            "reference_action_source",
            "rl_token_checkpoint_path",
            "actor_checkpoint_path",
            "actor_checkpoint_step",
        ):
            root.attrs.pop(key, None)
            root["rlt_timeline"].attrs.pop(key, None)
            root["rlt_policy_forward_events"].attrs.pop(key, None)

    _, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=10,
        chunk_stride=2,
        policy_event_alignment="trunk_shared",
    )

    assert manifest["behavior_policy"] == "runtime_unknown"
    assert manifest["action_source"] == "runtime_executed_action"
    assert manifest["reference_action_source"] == "vla_same_forward_reference_action"
    assert (
        manifest["rl_token_checkpoint_path"]
        == "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
    )
    assert "actor_checkpoint_path" not in manifest
    assert "actor_checkpoint_step" not in manifest


def test_trunk_shared_policy_forward_events_require_frame_policy_proprio(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(h5_path, frames=70, event_steps=(0, 25, 50))

    with pytest.raises(ValueError, match="policy-space proprio"):
        rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
            h5_path,
            train_horizon=10,
            chunk_stride=2,
            policy_event_alignment="trunk_shared",
        )


def test_trunk_shared_policy_forward_events_can_derive_policy_proprio_from_qpos(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_policy_forward_event_hdf5(h5_path, frames=70, event_steps=(0, 25, 50), include_qpos=True)

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=10,
        chunk_stride=2,
        policy_event_alignment="trunk_shared",
    )

    signs = rlt_timeline_replay.ALOHA_JOINT_FLIP_MASK
    expected_start_2 = np.arange(2 * 14, 3 * 14, dtype=np.float32) * signs
    expected_next_12 = np.arange(12 * 14, 13 * 14, dtype=np.float32) * signs
    assert manifest["proprio_alignment"] == "derived_from_observations_qpos_sign_flip_pad32"
    assert arrays["proprio"].shape == (26, 32)
    np.testing.assert_allclose(arrays["proprio"][1, :14], expected_start_2)
    np.testing.assert_allclose(arrays["proprio"][1, 14:], np.zeros((18,), dtype=np.float32))
    np.testing.assert_allclose(arrays["next_proprio"][1, :14], expected_next_12)
    np.testing.assert_allclose(arrays["next_proprio"][1, 14:], np.zeros((18,), dtype=np.float32))


def test_legacy_policy_forward_events_subtract_emission_lag(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_legacy_emission_event_hdf5(h5_path)

    arrays, manifest = rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
        h5_path,
        train_horizon=25,
        chunk_stride=25,
        policy_event_alignment="exact_event_pairs",
    )

    assert manifest["current_frames"] == [0, 25]
    assert manifest["next_frames"] == [25, 50]
    np.testing.assert_allclose(arrays["z_rl"][0], [1000, 1001, 1002])
    np.testing.assert_allclose(arrays["next_z_rl"][0], [1003, 1004, 1005])
    np.testing.assert_allclose(arrays["action"][0, :, 0], np.arange(0, 50, 2, dtype=np.float32))


def test_build_paper_replay_rejects_runtime_cache_block_z(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    _write_timeline_hdf5(h5_path, frames=8)
    with h5py.File(h5_path, "a") as root:
        root["rlt_timeline"].attrs["z_rl_source"] = "runtime_action_cache_block"

    with pytest.raises(ValueError, match="z_rl_source"):
        rlt_timeline_replay.build_paper_replay_from_timeline_hdf5(
            h5_path,
            train_horizon=2,
            chunk_stride=2,
        )


def test_write_paper_replay_shard_from_timeline_hdf5(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    out = tmp_path / "shards" / "key_region_demo.npz"
    _write_timeline_hdf5(h5_path, frames=8)

    rlt_timeline_replay.write_paper_replay_shard_from_timeline_hdf5(
        h5_path,
        out,
        train_horizon=2,
        chunk_stride=2,
        overwrite=False,
    )

    with np.load(out, allow_pickle=False) as data:
        assert data["z_rl"].shape == (3, 3)
        assert data["action"].shape == (3, 2, 2)
        manifest = json.loads(str(data["manifest"].item()))
    assert manifest["shard_path"] == str(out.resolve())
    assert manifest["formal_replay_ready"] is True

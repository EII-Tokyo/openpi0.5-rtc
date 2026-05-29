import sys
import types

import numpy as np

sys.modules.setdefault("h5py", types.SimpleNamespace(File=None))

from examples.aloha_real import rlt_key_region_recorder as recorder


def _record(step: int, *, include_full: bool = True) -> recorder.StepRecord:
    action = np.full((14,), step, dtype=np.float32)
    reference_action = np.full((14,), step + 0.5, dtype=np.float32)
    return recorder.StepRecord(
        step_index=step,
        timestamp=float(step),
        qpos=np.zeros((14,), dtype=np.float32),
        qvel=np.zeros((14,), dtype=np.float32),
        effort=np.zeros((14,), dtype=np.float32),
        action=action,
        reference_action=reference_action,
        action_full=np.full((50, 14), step, dtype=np.float32) if include_full else None,
        reference_action_full=np.full((50, 14), step + 0.5, dtype=np.float32) if include_full else None,
        z_rl=np.full((8,), step, dtype=np.float32),
        proprio=np.full((4,), step, dtype=np.float32),
        images={},
    )


def test_key_region_replay_saves_train_horizon_and_full_horizon(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
    )
    try:
        records = [_record(step) for step in range(70)]
        arrays, missing = store._build_replay_arrays(records, {"reward": 1})
    finally:
        store.close()
        store.close()

    assert missing == []
    assert arrays is not None
    assert arrays["action"].shape == (6, 50, 14)
    assert arrays["reference_action"].shape == (6, 50, 14)
    assert arrays["reward_seq"].shape == (6, 50)
    assert arrays["next_reference_action"].shape == (6, 50, 14)
    assert arrays["next_z_rl"][-1, 0] == 60
    assert arrays["done"].tolist() == [False, False, False, False, False, True]
    assert arrays["reward_seq"][-1, 9] == 1
    assert arrays["reward_seq"][-1, 49] == 0


def test_key_region_replay_requires_full_horizon_metadata(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
    )
    try:
        arrays, missing = store._build_replay_arrays([_record(step, include_full=False) for step in range(25)], {"reward": 0})
    finally:
        store.close()

    assert arrays is None
    assert "action_full" in missing
    assert "reference_action_full" in missing



def test_key_region_replay_publishes_valid_and_invalid_ack(tmp_path):
    messages = []
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
        ack_publisher=messages.append,
    )
    try:
        valid_arrays, valid_missing = store._build_replay_arrays([_record(step) for step in range(25)], {"reward": 1})
        valid_manifest = {
            "key_region_id": "valid",
            "task": "task",
            "phase": "warmup",
            "reward": 1,
            "score_timeout": False,
            "num_replay_transitions": len(valid_arrays["z_rl"]),
            "missing_rlt_metadata": valid_missing,
            "replay_status": recorder._replay_status(valid_missing, valid_arrays),
            "replay_ready": valid_arrays is not None,
        }
        store._publish_replay_ack(valid_manifest, shard_path=tmp_path / "valid.npz")

        invalid_arrays, invalid_missing = store._build_replay_arrays([_record(step) for step in range(5)], {"reward": 0})
        invalid_manifest = {
            "key_region_id": "invalid",
            "task": "task",
            "phase": "warmup",
            "reward": 0,
            "score_timeout": False,
            "num_replay_transitions": 0,
            "missing_rlt_metadata": invalid_missing,
            "replay_status": recorder._replay_status(invalid_missing, invalid_arrays),
            "replay_ready": invalid_arrays is not None,
        }
        store._publish_replay_ack(invalid_manifest, shard_path=None)
    finally:
        store.close()

    assert messages[0]["type"] == "rlt_replay_segment_written"
    assert messages[0]["key_region_id"] == "valid"
    assert messages[0]["phase"] == "warmup"
    assert messages[0]["reward"] == 1
    assert messages[0]["replay_ready"] is True
    assert messages[0]["replay_status"] == "written"
    assert messages[0]["num_replay_transitions"] == 1
    assert messages[0]["shard_path"] == str(tmp_path / "valid.npz")
    assert messages[1]["key_region_id"] == "invalid"
    assert messages[1]["replay_ready"] is False
    assert messages[1]["replay_status"] == "too_short"
    assert messages[1]["shard_path"] is None



def test_key_region_manifest_includes_replay_schema_metadata(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
        ack_publisher=lambda payload: None,
    )
    try:
        arrays, missing = store._build_replay_arrays([_record(step) for step in range(25)], {"reward": 1})
        segment = recorder.KeyRegionSegment("kid", "task", "warmup", {"timestamp": 1.0}, {"timestamp": 2.0}, {"timestamp": 3.0, "reward": 1}, [])
        manifest = store._write_manifest(tmp_path / "manifest.json", segment, missing, arrays)
    finally:
        store.close()

    assert manifest["schema_version"] == 1
    assert manifest["train_chunk_horizon"] == 10
    assert manifest["policy_horizon"] == 50
    assert manifest["action_space"] == "aloha_exec"
    assert manifest["action_dim"] == 14
    assert manifest["reward_placement"] == "terminal_last_train_step"

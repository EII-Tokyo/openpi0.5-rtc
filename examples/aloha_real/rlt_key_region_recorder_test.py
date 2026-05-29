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
    assert arrays["action"].shape == (6, 10, 14)
    assert arrays["reference_action"].shape == (6, 10, 14)
    assert arrays["reward_seq"].shape == (6, 10)
    assert arrays["next_reference_action"].shape == (6, 10, 14)
    assert arrays["action_full"].shape == (6, 50, 14)
    assert arrays["reference_action_full"].shape == (6, 50, 14)
    assert arrays["reward_seq_full"].shape == (6, 50)
    assert arrays["next_z_rl"][-1, 0] == 60
    assert arrays["done"].tolist() == [False, False, False, False, False, True]
    assert arrays["reward_seq"][-1, -1] == 1
    assert arrays["reward_seq_full"][-1, 49] == 1


def test_key_region_replay_writes_train_samples_when_full_horizon_is_unavailable(tmp_path):
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

    assert missing == []
    assert arrays is not None
    assert arrays["action"].shape == (1, 10, 14)
    assert "action_full" not in arrays

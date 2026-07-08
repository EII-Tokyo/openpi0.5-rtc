import dataclasses
import sys
import types
import importlib

import numpy as np

if getattr(sys.modules.get("h5py"), "File", object()) is None:
    sys.modules.pop("h5py", None)
try:
    import h5py  # noqa: F401
except ImportError:
    sys.modules.setdefault("h5py", types.SimpleNamespace(File=None))

from examples.aloha_real import rlt_key_region_recorder as recorder


def _record(step: int, *, include_full: bool = True, include_step_actions: bool = True) -> recorder.StepRecord:
    action = np.full((14,), step, dtype=np.float32) if include_step_actions else None
    reference_action = np.full((14,), step + 0.5, dtype=np.float32) if include_step_actions else None
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
        runtime_z_rl=np.full((8,), step + 100, dtype=np.float32),
        runtime_proprio=np.full((4,), step + 200, dtype=np.float32),
        z_rl_source="vla_same_forward_runtime_output",
        policy_forward_id=step,
        policy_forward_action_start_index=0,
        policy_forward_z_rl=np.full((8,), step, dtype=np.float32),
        policy_forward_proprio=np.full((4,), step, dtype=np.float32),
        policy_forward_z_rl_source="vla_same_forward_runtime_output",
        behavior_policy="rlt_actor",
        action_source="rlt_actor_adjusted_action",
        reference_action_source="vla_same_forward_reference_action",
        actor_checkpoint_path="/app/local_rlt_runs/demo_actor/00004500",
        actor_checkpoint_step=4500,
        rl_token_checkpoint_path="/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint",
    )


def _delayed_forward_record(step: int, *, action_start_index: int) -> recorder.StepRecord:
    record = _record(step)
    return dataclasses.replace(record, policy_forward_action_start_index=action_start_index)


def _push_runtime_step(store: recorder.KeyRegionReplayRecorder) -> None:
    store.on_step(
        {
            "qpos": np.zeros((14,), dtype=np.float32),
            "qvel": np.zeros((14,), dtype=np.float32),
            "effort": np.zeros((14,), dtype=np.float32),
            "images": {},
        },
        {
            "actions": np.zeros((14,), dtype=np.float32),
            "reference_actions": np.zeros((14,), dtype=np.float32),
            "action_full": np.zeros((50, 14), dtype=np.float32),
            "reference_action_full": np.zeros((50, 14), dtype=np.float32),
            "z_rl": np.zeros((8,), dtype=np.float32),
            "z_rl_source": "vla_same_forward",
            "proprio": np.zeros((4,), dtype=np.float32),
            "rlt_policy_forward_event": True,
            "rlt_policy_forward_id": 0,
            "rlt_policy_forward_action_start_index": 0,
            "rlt_policy_forward_z_rl": np.zeros((8,), dtype=np.float32),
            "rlt_policy_forward_proprio": np.zeros((4,), dtype=np.float32),
            "rlt_policy_forward_z_rl_source": "vla_same_forward_runtime_output",
            "behavior_policy": "rlt_actor",
            "action_source": "rlt_actor_adjusted_action",
            "reference_action_source": "vla_same_forward_reference_action",
            "rlt_actor_checkpoint_path": "/app/local_rlt_runs/demo_actor/00004500",
            "rlt_actor_checkpoint_step": 4500,
            "rlt_rl_token_checkpoint_path": "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint",
        },
    )


def test_key_region_replay_second_stride_action_uses_step_window(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=2,
    )
    try:
        records = [_record(step) for step in range(22)]
        arrays, missing = store._build_replay_arrays(records, {"reward": 1})
    finally:
        store.close()

    assert missing == []
    assert arrays is not None
    assert arrays["action"].shape == (2, 10, 14)
    assert arrays["action"][1, :, 0].tolist() == list(range(2, 12))


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
    assert arrays["next_z_rl"][-1, 0] == 60
    assert arrays["done"].tolist() == [False, False, False, False, False, True]
    assert arrays["reward_seq"][-1, 9] == 1



def test_key_region_replay_always_marks_terminal_when_stride_misses_last_start(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=2,
    )
    try:
        records = [_record(step) for step in range(117)]
        arrays, missing = store._build_replay_arrays(records, {"reward": 0})
    finally:
        store.close()

    assert missing == []
    assert arrays is not None
    assert arrays["done"].sum() == 1
    assert arrays["done"][-1]
    assert arrays["reward_seq"][-1, 9] == 0


def test_key_region_replay_requires_step_action_metadata(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
    )
    try:
        arrays, missing = store._build_replay_arrays(
            [_record(step, include_step_actions=False) for step in range(25)],
            {"reward": 0},
        )
    finally:
        store.close()

    assert arrays is None
    assert "action" in missing
    assert "reference_action" in missing



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
            "train_eligible": False,
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

    assert messages[0]["type"] == "rlt_replay_segment_committed"
    assert messages[0]["key_region_id"] == "valid"
    assert messages[0]["phase"] == "warmup"
    assert messages[0]["reward"] == 1
    assert messages[0]["replay_ready"] is True
    assert messages[0]["train_eligible"] is False
    assert messages[0]["segment_status"] == "raw_timeline_committed"
    assert messages[0]["replay_status"] == "runtime_cache_block_requires_offline_reencode"
    assert messages[0]["num_replay_transitions"] == 2
    assert messages[0]["shard_path"] == str(tmp_path / "valid.npz")
    assert messages[1]["type"] == "rlt_replay_segment_committed"
    assert messages[1]["key_region_id"] == "invalid"
    assert messages[1]["replay_ready"] is False
    assert messages[1]["train_eligible"] is False
    assert messages[1]["segment_status"] == "raw_timeline_committed"
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
        segment = recorder.KeyRegionSegment(
            "kid",
            "task",
            "warmup",
            {"timestamp": 1.0},
            {"timestamp": 2.0},
            {"timestamp": 3.0, "reward": 1},
            [_record(step) for step in range(25)],
            active_start_step=0,
            active_end_step=50,
        )
        manifest = store._write_manifest(tmp_path / "manifest.json", segment, missing, arrays)
    finally:
        store.close()

    assert manifest["schema_version"] == 1
    assert manifest["train_chunk_horizon"] == 10
    assert manifest["policy_horizon"] == 10
    assert manifest["vla_policy_horizon"] == 50
    assert manifest["full_horizon"] == 50
    assert manifest["action_space"] == "aloha_exec"
    assert manifest["action_dim"] == 14
    assert manifest["reward_placement"] == "terminal_last_train_step"
    assert manifest["replay_state_grain"] == "runtime_action_cache_block"
    assert manifest["requires_offline_reencode"] is True
    assert manifest["formal_replay_state_grain"] == "paper_subsampled_anchor"
    assert manifest["formal_replay_ready"] is False
    assert manifest["behavior_policy"] == "rlt_actor"
    assert manifest["action_source"] == "rlt_actor_adjusted_action"
    assert manifest["reference_action_source"] == "vla_same_forward_reference_action"
    assert manifest["actor_checkpoint_path"] == "/app/local_rlt_runs/demo_actor/00004500"
    assert manifest["actor_checkpoint_step"] == 4500
    assert manifest["rl_token_checkpoint_path"] == "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
    assert manifest["pre_roll_seconds"] == 0.0
    assert manifest["post_roll_seconds"] == 0.0
    assert manifest["key_region_start_sec"] == 0.0
    assert manifest["key_region_end_sec"] == manifest["duration_seconds"]


def test_write_hdf5_includes_raw_frame_timeline_and_marks_runtime_cache_audit(tmp_path):
    pytest = __import__("pytest")
    h5py = pytest.importorskip("h5py")
    recorder.h5py = importlib.import_module("h5py")
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=2,
        full_horizon=4,
        chunk_stride=1,
        ack_publisher=lambda payload: None,
    )
    path = tmp_path / "episode.hdf5"
    segment = recorder.KeyRegionSegment(
        "kid",
        "task",
        "warmup",
        {"timestamp": 1.0},
        {"timestamp": 2.0},
        {"timestamp": 3.0, "reward": 1},
        [_record(step + 10) for step in range(5)],
        active_start_step=10,
        active_end_step=15,
    )
    try:
        store._write_hdf5(path, segment, missing_metadata=[])
    finally:
        store.close()

    with h5py.File(path, "r") as root:
        assert root.attrs["replay_state_grain"] == "raw_frame_timeline"
        assert root["rlt"].attrs["state_grain"] == "runtime_action_cache_block_audit"
        assert root["rlt_timeline"].attrs["state_grain"] == "raw_frame_timeline"
        assert root["rlt_timeline"].attrs["z_rl_source"] == "policy_forward_events"
        assert root["rlt_timeline"].attrs["behavior_policy"] == "rlt_actor"
        assert root["rlt_timeline"].attrs["action_source"] == "rlt_actor_adjusted_action"
        assert root["rlt_timeline"].attrs["reference_action_source"] == "vla_same_forward_reference_action"
        assert root["rlt_timeline"].attrs["actor_checkpoint_path"] == "/app/local_rlt_runs/demo_actor/00004500"
        assert root["rlt_timeline"].attrs["actor_checkpoint_step"] == 4500
        assert root["rlt_timeline"].attrs["rl_token_checkpoint_path"] == "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        assert root["rlt_policy_forward_events"].attrs["rl_token_checkpoint_path"] == "/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint"
        assert "z_rl" not in root["rlt_timeline"]
        assert "proprio" not in root["rlt_timeline"]
        assert root["rlt_policy_forward_events"].attrs["z_rl_source"] == "vla_same_forward_runtime_output"
        np.testing.assert_allclose(root["rlt_timeline/step_index"][:], np.arange(5, dtype=np.int64))
        np.testing.assert_allclose(root["rlt_timeline/global_step_index"][:], np.arange(10, 15, dtype=np.int64))
        np.testing.assert_allclose(root["rlt_policy_forward_events/z_rl"][:, 0], np.arange(10, 15, dtype=np.float32))
        np.testing.assert_allclose(root["rlt_policy_forward_events/proprio"][:, 0], np.arange(10, 15, dtype=np.float32))
        np.testing.assert_allclose(root["rlt_policy_forward_events/step_index"][:], np.arange(5, dtype=np.int64))
        np.testing.assert_allclose(root["rlt_policy_forward_events/global_step_index"][:], np.arange(10, 15, dtype=np.int64))
        np.testing.assert_allclose(root["rlt/cached_z_rl"][:, 0], np.arange(10, 15, dtype=np.float32) + 100)
        np.testing.assert_allclose(root["rlt/cached_proprio"][:, 0], np.arange(10, 15, dtype=np.float32) + 200)
        np.testing.assert_allclose(root["action"][:, 0], np.arange(10, 15, dtype=np.float32))
        np.testing.assert_allclose(root["reference_action"][:, 0], np.arange(10, 15, dtype=np.float32) + 0.5)


def test_write_hdf5_policy_forward_events_store_anchor_and_emission_steps(tmp_path):
    pytest = __import__("pytest")
    h5py = pytest.importorskip("h5py")
    recorder.h5py = importlib.import_module("h5py")
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=2,
        full_horizon=4,
        chunk_stride=1,
        ack_publisher=lambda payload: None,
    )
    path = tmp_path / "episode.hdf5"
    records = [
        dataclasses.replace(_record(20), policy_forward_id=None, policy_forward_z_rl=None, policy_forward_proprio=None),
        dataclasses.replace(_record(21), policy_forward_id=None, policy_forward_z_rl=None, policy_forward_proprio=None),
        _delayed_forward_record(22, action_start_index=2),
    ]
    segment = recorder.KeyRegionSegment(
        "kid",
        "task",
        "warmup",
        {"timestamp": 1.0},
        {"timestamp": 2.0},
        {"timestamp": 3.0, "reward": 1},
        records,
        active_start_step=20,
        active_end_step=23,
    )
    try:
        store._write_hdf5(path, segment, missing_metadata=[])
    finally:
        store.close()

    with h5py.File(path, "r") as root:
        events = root["rlt_policy_forward_events"]
        assert events.attrs["step_index_semantics"] == "anchor_observation_step_index"
        np.testing.assert_allclose(events["step_index"][:], [0])
        np.testing.assert_allclose(events["emission_step_index"][:], [2])
        np.testing.assert_allclose(events["global_step_index"][:], [20])
        np.testing.assert_allclose(events["emission_global_step_index"][:], [22])


def test_key_region_discard_clears_pending_region(tmp_path):
    messages = []
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        ack_publisher=messages.append,
    )
    try:
        start = {"type": "key_region_start", "key_region_id": "discard-me", "timestamp": 1.0}
        end = {"type": "key_region_end", "key_region_id": "discard-me", "timestamp": 2.0}
        score = {"type": "score", "key_region_id": "discard-me", "timestamp": 3.0, "reward": 1}
        store.on_key_region_start(start)
        store.on_key_region_end(end)
        store.on_key_region_discard({"type": "key_region_discard", "key_region_id": "discard-me", "timestamp": 2.5})
        store.on_key_region_score(score)
    finally:
        store.close()

    assert messages == []
    assert store._active_start_event is None
    assert store._pending_end_event is None


def test_key_region_records_exactly_between_start_and_end(tmp_path):
    saved_segments = []
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        fps=10.0,
        pre_roll_seconds=2.0,
        post_roll_seconds=2.0,
        ack_publisher=lambda payload: None,
    )
    store._write_segment = saved_segments.append
    try:
        for _ in range(3):
            _push_runtime_step(store)
        store.on_key_region_start({"type": "key_region_start", "key_region_id": "post", "timestamp": 100.0})
        for _ in range(4):
            _push_runtime_step(store)
        store.on_key_region_end({"type": "key_region_end", "key_region_id": "post", "timestamp": 104.0})
        for _ in range(5):
            _push_runtime_step(store)
        store.on_key_region_score({"type": "score", "key_region_id": "post", "timestamp": 106.0, "reward": 1})
        store._write_queue.join()
    finally:
        store.close()

    assert len(saved_segments) == 1
    segment = saved_segments[0]
    assert segment.active_start_step == 3
    assert segment.active_end_step == 7
    assert [record.step_index for record in segment.records] == list(range(3, 7))
    assert segment.score_event["timestamp"] == 106.0


def test_key_region_manifest_marks_train_eligibility(tmp_path):
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
        segment = recorder.KeyRegionSegment(
            "kid",
            "task",
            "warmup",
            {"timestamp": 1.0},
            {"timestamp": 2.0},
            {"timestamp": 3.0, "reward": 1},
            [],
            active_start_step=0,
            active_end_step=50,
        )
        manifest = store._write_manifest(tmp_path / "manifest.json", segment, missing, arrays)
    finally:
        store.close()

    assert manifest["train_eligible"] is False
    assert manifest["segment_status"] == "raw_timeline_committed"
    assert manifest["voided"] is False


def test_key_region_write_segment_enqueues_anchor_token_job(tmp_path):
    store = recorder.KeyRegionReplayRecorder(
        replay_root=str(tmp_path / "replay"),
        rollouts_root=str(tmp_path / "rollouts"),
        train_horizon=10,
        full_horizon=50,
        chunk_stride=10,
        ack_publisher=lambda payload: None,
        anchor_job_root=str(tmp_path / "anchor_jobs"),
    )
    store._write_videos = lambda rollout_dir, records: None
    store._write_hdf5 = lambda path, segment, missing_metadata: path.write_bytes(b"hdf5")
    segment = recorder.KeyRegionSegment(
        "kid",
        "task",
        "warmup",
        {"timestamp": 1.0},
        {"timestamp": 2.0},
        {"timestamp": 3.0, "reward": 1},
        [_record(step) for step in range(25)],
        active_start_step=0,
        active_end_step=25,
    )
    try:
        store._write_segment(segment)
    finally:
        store.close()

    job_path = tmp_path / "anchor_jobs" / "pending" / "key_region_kid.json"
    assert job_path.exists()
    payload = recorder.json.loads(job_path.read_text(encoding="utf-8"))
    assert payload["key_region_id"] == "kid"
    assert payload["status"] == "pending"
    assert payload["formal_replay_state_grain"] == "paper_subsampled_anchor"
    assert payload["source_runtime_cache_block_shard_path"].endswith("key_region_kid.npz")


class _FakeStdin:
    def __init__(self):
        self.closed = False
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, return_code=0):
        self.stdin = _FakeStdin()
        self._return_code = return_code

    def wait(self):
        return self._return_code


def test_ffmpeg_writer_uses_gpu_encoder_when_available_and_atomic_output(monkeypatch, tmp_path):
    popen_calls = []
    process = _FakeProcess(return_code=0)

    def fake_popen(args, stdin):
        popen_calls.append(args)
        return process

    monkeypatch.setattr(recorder.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(recorder, "_ffmpeg_supports_encoder", lambda encoder: encoder == "h264_nvenc")
    monkeypatch.setattr(recorder, "_ffmpeg_nvenc_smoke_test", lambda preset: True)
    output = tmp_path / "cam_high.mp4"

    writer = recorder._FfmpegMp4Writer(output, fps=50.0, width=2, height=2, prefer_gpu=True)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    writer._tmp_path.write_bytes(b"mp4")
    writer.close()

    args = popen_calls[0]
    assert args[args.index("-c:v") + 1] == "h264_nvenc"
    assert args[args.index("-preset") + 1] == "fast"
    assert args[args.index("-rc") + 1] == "vbr"
    assert args[args.index("-cq") + 1] == "23"
    assert args[args.index("-f", args.index("-movflags")) + 1] == "mp4"
    assert str(output) not in args
    assert str(output.with_suffix(".mp4.tmp")) in args
    assert output.read_bytes() == b"mp4"
    assert not output.with_suffix(".mp4.tmp").exists()


def test_ffmpeg_writer_falls_back_to_cpu_when_gpu_smoke_test_fails(monkeypatch, tmp_path):
    popen_calls = []
    process = _FakeProcess(return_code=0)

    def fake_popen(args, stdin):
        popen_calls.append(args)
        return process

    monkeypatch.setattr(recorder.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(recorder, "_ffmpeg_supports_encoder", lambda encoder: encoder == "h264_nvenc")
    monkeypatch.setattr(recorder, "_ffmpeg_nvenc_smoke_test", lambda preset: False)
    output = tmp_path / "cam_high.mp4"

    writer = recorder._FfmpegMp4Writer(output, fps=50.0, width=2, height=2, prefer_gpu=True)
    writer._tmp_path.write_bytes(b"mp4")
    writer.close()

    args = popen_calls[0]
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-crf") + 1] == "23"


def test_ffmpeg_writer_failure_removes_tmp_and_leaves_no_final_file(monkeypatch, tmp_path):
    process = _FakeProcess(return_code=1)
    monkeypatch.setattr(recorder.subprocess, "Popen", lambda *args, **kwargs: process)
    output = tmp_path / "cam_high.mp4"

    writer = recorder._FfmpegMp4Writer(output, fps=50.0, width=2, height=2, prefer_gpu=False)
    writer._tmp_path.write_bytes(b"partial")

    try:
        writer.close()
    except RuntimeError as exc:
        assert "ffmpeg failed" in str(exc)
    else:
        raise AssertionError("Expected ffmpeg failure")

    assert not output.exists()
    assert not output.with_suffix(".mp4.tmp").exists()

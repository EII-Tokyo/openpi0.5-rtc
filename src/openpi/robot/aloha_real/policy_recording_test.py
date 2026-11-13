"""No ROS: verify HDF5 payloads, preservation, and actuator pause invariants."""

from collections import deque
import json
import threading
from unittest.mock import Mock

import cv2
import h5py
import numpy as np
import pytest

from openpi.robot.aloha_real.policy_recording import CAMERAS, PolicyRecording
from openpi.robot.aloha_real.runtime import Runtime


def observation():
    """A small RGB test scene makes JPEG color/order and 14-DOF alignment inspectable."""
    return {"qpos": np.arange(14, dtype=np.float32), "qvel": np.zeros(14), "effort": np.ones(14),
            "images": {key: np.full((480, 640, 3), [220, 40, 10], np.uint8) for key in CAMERAS}}


def test_complete_self_contained_episode(tmp_path):
    """Match the current collector: four H.264 videos referenced by action/state HDF5."""
    recorder = PolicyRecording(tmp_path)
    recorder.start({"annotations": {"rating": 5}, "task": {"rating": 5}}, "batch_a")
    obs = observation()
    for i in range(3):
        recorder.append(obs, {"actions": np.full(14, i)}, {"rating": 5})
    recorder.finish({"rating": None, "is_error": True, "note": "failed grasp"})
    recorder.close()
    state = recorder.snapshot()
    assert state["status"] == "saved" and state["frames"] == 3, state
    with h5py.File(state["path"], "r") as root:
        assert not root.attrs["compress"] and not root.attrs["sim"]
        assert root.attrs["image_storage"] == "mp4"
        assert "observations/images" not in root and "compress_len" not in root
        assert root["action"].shape == (3, 14)
        np.testing.assert_array_equal(root["action"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(root["observations/qpos"][0], obs["qpos"])
        assert root["phase"].shape == (3,) and not root["done"][:].any()
        for camera in CAMERAS:
            reference = root[f"observations/videos/{camera}"]
            assert reference.asstr()[()] == f"{camera}.mp4"
            capture = cv2.VideoCapture(str(__import__('pathlib').Path(state["path"]).parent / reference.asstr()[()]))
            ok, bgr = capture.read()
            assert ok and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            capture.release()
            assert bgr.shape == (480, 640, 3) and bgr[0, 0, 2] > 200 and bgr[0, 0, 0] < 30
        labels = json.loads(root["collection_metadata"][()])
        assert labels["annotations"]["rating"] is None and labels["annotations"]["is_error"]
        assert labels["task"]["rating"] == 5  # Inference request is not outcome quality.
    recorder.start({"annotations": {"rating": 0}}, "batch_a")
    recorder.append(obs, {"actions": np.zeros(14)}, {})
    recorder.close()
    assert len(list(tmp_path.glob("batch_a/episode_*/episode.hdf5"))) == 2
    assert (tmp_path / "batch_a/episode_0/episode.hdf5").exists()
    assert (tmp_path / "batch_a/episode_1/episode.hdf5").exists()


def test_video_memory_records_latest_frame_without_changing_policy_input(tmp_path):
    """Six history images still produce one video frame per robot action."""
    recorder = PolicyRecording(tmp_path)
    obs = observation()
    for key in CAMERAS:
        latest = obs["images"][key]
        obs["images"][key] = np.stack([np.zeros_like(latest)] * 5 + [latest])
    before = {key: image.copy() for key, image in obs["images"].items()}
    recorder.start({}, "memory_test")
    try:
        for _ in range(3):
            recorder.append(obs, {"actions": np.zeros(14)}, {})
    finally:
        recorder.close()
    state = recorder.snapshot()
    assert state["status"] == "saved" and state["frames"] == 3, state
    with h5py.File(state["path"], "r") as root:
        assert root["action"].shape == (3, 14)
        for camera in CAMERAS:
            path = __import__("pathlib").Path(state["path"]).parent / f"{camera}.mp4"
            capture = cv2.VideoCapture(str(path))
            try:
                assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
                ok, bgr = capture.read()
                assert ok and bgr.shape == (480, 640, 3)
                assert bgr[0, 0, 2] > 200 and bgr[0, 0, 0] < 30
            finally:
                capture.release()
            np.testing.assert_array_equal(obs["images"][camera], before[camera])


def test_invalid_frame_retains_partial(tmp_path):
    """A failed recording must never appear as a successfully finalized episode."""
    recorder = PolicyRecording(tmp_path)
    recorder.start({}, "batch_a")
    obs = observation()
    obs["qpos"] = np.zeros(13)
    recorder.append(obs, {"actions": np.zeros(14)}, {})
    recorder.close()
    assert recorder.snapshot()["status"] == "error"
    assert not __import__('pathlib').Path(recorder.snapshot()["path"]).exists()
    assert len(list(tmp_path.glob("batch_a/.episode_*.staging-*/episode.hdf5"))) == 1


def test_collector_claim_interoperability_and_output_permissions(tmp_path):
    """A canonical collector claim must block our writer, and output must be user-owned."""
    from openpi.robot.aloha_real.collection_storage import EpisodeClaim
    from openpi.robot.aloha_real.collection_permissions import output_owner
    batch = tmp_path / "shared_batch"
    batch.mkdir()
    claim = EpisodeClaim.acquire(batch, 0)
    recorder = PolicyRecording(tmp_path)
    try:
        recorder.start({}, "shared_batch")
        assert recorder.snapshot()["path"] == str(batch / "episode_1/episode.hdf5")
        recorder.append(observation(), {"actions": np.zeros(14)}, {})
        recorder.close()
        assert recorder.snapshot()["status"] == "saved", recorder.snapshot()
        assert claim.claim_path.exists()  # Never remove another collector's reservation.
        assert not (batch / ".episode_1.claim").exists()
        path = batch / "episode_1/episode.hdf5"
        assert (path.stat().st_uid, path.stat().st_gid) == output_owner()
        assert path.stat().st_mode & 0o777 == 0o664
        assert path.parent.stat().st_mode & 0o777 == 0o775
    finally:
        recorder.close()
        claim.release()


def test_sequential_directories_preserve_old_files_and_partial_episodes(tmp_path):
    """Restarted or concurrent writers skip reserved directories and collector files."""
    batch_dir = tmp_path / "batch_a"
    batch_dir.mkdir()
    (batch_dir / "episode_0.partial.hdf5").write_bytes(b"keep partial")
    (batch_dir / "episode_1.hdf5").write_bytes(b"keep legacy collector file")
    (batch_dir / "episode_1792040355736025974.hdf5").write_bytes(b"keep timestamp file")
    first, second = PolicyRecording(tmp_path), PolicyRecording(tmp_path)
    first.start({}, "batch_a")
    second.start({}, "batch_a")
    try:
        for recorder in (first, second):
            recorder.append(observation(), {"actions": np.zeros(14)}, {})
            recorder.close()
            assert recorder.snapshot()["status"] == "saved", recorder.snapshot()
    finally:
        first.close()
        second.close()
    assert first.snapshot()["path"] == str(batch_dir / "episode_2/episode.hdf5")
    assert second.snapshot()["path"] == str(batch_dir / "episode_3/episode.hdf5")
    assert (batch_dir / "episode_0.partial.hdf5").read_bytes() == b"keep partial"
    assert (batch_dir / "episode_1.hdf5").read_bytes() == b"keep legacy collector file"
    assert (batch_dir / "episode_1792040355736025974.hdf5").read_bytes() == b"keep timestamp file"
    first.start({}, "batch_b")
    first.append(observation(), {"actions": np.zeros(14)}, {})
    first.close()
    assert first.snapshot()["path"] == str(tmp_path / "batch_b/episode_0/episode.hdf5")


@pytest.mark.parametrize("name", [None, "", "../outside", "/tmp/outside", "a/b", "a\\b", "..", "a" * 97])
def test_batch_name_validation(tmp_path, name):
    """Web-provided batch names cannot escape the recording root."""
    with pytest.raises(ValueError):
        PolicyRecording(tmp_path).start({}, name)


@pytest.mark.parametrize("help_text,preset", [("preset options: default medium slow", "medium"),
                                             ("preset options: p1 p2 p3 p4", "p4")])
def test_encoder_preset_matches_installed_ffmpeg(monkeypatch, help_text, preset):
    """ROS1's old FFmpeg and the current collection image accept different preset names."""
    from types import SimpleNamespace
    from openpi.robot.aloha_real import collection_video
    collection_video.nvenc_options.cache_clear()
    try:
        monkeypatch.setattr(collection_video.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=help_text)))
        assert collection_video.nvenc_options() == ["-preset", preset, "-cq", "23", "-b:v", "0"]
    finally:
        collection_video.nvenc_options.cache_clear()


def runtime_for_test():
    """Construct the control logic without hardware, Redis, or initialization motion."""
    runtime = Runtime.__new__(Runtime)
    runtime._recording = Mock()
    runtime._recording.snapshot.return_value = {"status": "recording", "frames": 3}
    runtime._recording_commands = deque()
    runtime._recording_hold = threading.Event()
    runtime._recording_paused = False
    runtime._is_waiting_for_task = False
    runtime._current_task = {"task_num": "1", "task_name": "rinse bottle"}
    runtime._task_lock = threading.Lock()
    runtime._policy = Mock()
    runtime._environment = Mock()
    runtime._video_memory_preview = Mock()
    runtime._publish_runtime_state = Mock()
    runtime._subscribers = []
    runtime._model_task_nums = {"1", "2"}
    runtime._pending_inference_settings = None
    return runtime


def test_finish_pauses_without_home_and_resume_discards_chunks():
    """Only explicit resume permits policy execution after the file is saved."""
    runtime = runtime_for_test()
    runtime._recording_commands.append({"action": "stop", "annotations": {"rating": 0}})
    runtime._process_recording_commands()
    assert runtime._recording_paused and runtime._is_waiting_for_task
    runtime._step()
    runtime._environment.apply_action.assert_not_called()
    runtime._environment.stop.assert_not_called()
    runtime._environment.reset.assert_not_called()
    runtime._handle_task({"task_num": "2"})
    assert runtime._current_task["task_num"] == "1"
    runtime._recording.snapshot.return_value = {"status": "saving"}
    runtime._recording_commands.append({"action": "resume"})
    runtime._process_recording_commands()
    assert runtime._recording_paused
    runtime._recording.snapshot.return_value = {"status": "saved"}
    runtime._recording_commands.append({"action": "resume"})
    runtime._process_recording_commands()
    assert not runtime._recording_paused and not runtime._recording_hold.is_set()
    assert not runtime._is_waiting_for_task
    assert runtime._policy.reset.call_count == 2


def test_stop_during_inference_never_applies_returned_action():
    """Stop latches independently of the blocked foreground inference call."""
    runtime = runtime_for_test()
    obs = observation()
    runtime._environment.get_observation.return_value = {"origin_observation": obs, "images": obs["images"]}
    runtime._policy.infer.side_effect = lambda obs: (runtime._recording_hold.set() or {"actions": np.zeros(14)})
    runtime._step()
    runtime._environment.apply_action.assert_not_called()
    runtime._recording.append.assert_not_called()


def test_start_while_waiting_does_not_move_robot():
    """Recording is not an implicit robot start command."""
    runtime = runtime_for_test()
    runtime._is_waiting_for_task = True
    runtime._recording_commands.append({"action": "start"})
    runtime._process_recording_commands()
    runtime._recording.start.assert_not_called()
    runtime._environment.apply_action.assert_not_called()


def test_recording_api_commands_are_acknowledged_not_task_starts(monkeypatch):
    """The API uses only recording_control and validates offline/paused transitions."""
    from fastapi.testclient import TestClient
    from voice_assistant_web.backend.app import main
    state = {"timestamp": __import__('time').time(), "mode": "policy", "status": "idle", "paused": False}
    redis = Mock()
    redis.get.side_effect = lambda key: json.dumps(state)
    redis.publish.return_value = 1
    monkeypatch.setattr(main, "redis_client", redis)
    http = TestClient(main.app)
    assert http.post("/api/recording", json={"action": "start"}).status_code == 422
    response = http.post("/api/recording", json={"action": "start", "batch_name": "batch_a", "annotations": {"rating": 0, "is_error": True}})
    assert response.status_code == 200
    payload = json.loads(redis.publish.call_args.args[1])
    assert payload["type"] == "recording_control" and "task" not in payload
    assert payload["annotations"]["rating"] == 0
    assert payload["batch_name"] == "batch_a"
    assert http.post("/api/recording", json={"action": "resume"}).status_code == 409
    state.update(status="recording")
    assert http.post("/api/recording", json={"action": "stop"}).status_code == 200
    state.update(status="saved", paused=True, mode="paused")
    assert http.post("/api/recording", json={"action": "resume"}).status_code == 200
    state["timestamp"] = 0
    assert http.post("/api/recording", json={"action": "stop"}).status_code == 503
    for rating in [-1, 6, True, "1"]:
        assert http.post("/api/recording", json={"action": "start", "annotations": {"rating": rating}}).status_code == 422

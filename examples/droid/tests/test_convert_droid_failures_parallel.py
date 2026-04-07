import json
import os
import sys
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "h5py" not in sys.modules:
    # Avoid hard dependency during import; tests stub h5py as needed.
    sys.modules["h5py"] = SimpleNamespace(File=lambda *_args, **_kwargs: None)

import convert_droid_failures_to_lerobot_parallel as mod


class DummyH5File:
    def __init__(self, joint_velocity):
        self._joint_velocity = joint_velocity

    def __enter__(self):
        return {"action": {"joint_velocity": self._joint_velocity}}

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyDataset:
    def __init__(self):
        self.frames = []
        self.episodes = 0
        self.pushed = False
        self.meta = SimpleNamespace(total_episodes=0, total_frames=0)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes += 1

    def finalize(self):
        pass

    def clear_episode_buffer(self, delete_images=True):
        pass

    def push_to_hub(self, **kwargs):
        self.pushed = True


class DummyLeRobotDataset:
    last_dataset = None

    @classmethod
    def create(cls, **kwargs):
        cls.last_dataset = DummyDataset()
        return cls.last_dataset


def install_dummy_lerobot(monkeypatch, tmp_path):
    # Build a fake module chain: lerobot.datasets.lerobot_dataset
    lerobot_dataset_mod = SimpleNamespace(
        HF_LEROBOT_HOME=tmp_path,
        LeRobotDataset=DummyLeRobotDataset,
    )
    datasets_mod = SimpleNamespace(lerobot_dataset=lerobot_dataset_mod)
    lerobot_mod = SimpleNamespace(datasets=datasets_mod)

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_mod)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", lerobot_dataset_mod)


def test_process_episode_missing_metadata(monkeypatch):
    monkeypatch.setattr(mod, "gsutil_ls", lambda _: [])
    result = mod.process_episode("gs://bucket/episode/", {})
    assert "error" in result and "no metadata" in result["error"]


def test_process_episode_too_short(monkeypatch, tmp_path):
    def fake_ls(path):
        return ["gs://bucket/episode/metadata_0.json"]

    def fake_cp(_src, dst):
        with open(dst, "w") as f:
            json.dump({"uuid": "lab+user+2023-01-01-00h-00m-00s", "trajectory_length": 10}, f)

    monkeypatch.setattr(mod, "gsutil_ls", fake_ls)
    monkeypatch.setattr(mod, "gsutil_cp", fake_cp)

    result = mod.process_episode("gs://bucket/episode/", {})
    assert result == {"skip": "too_short"}


def test_process_episode_static(monkeypatch):
    monkeypatch.setattr(mod, "gsutil_ls", lambda _: ["gs://bucket/episode/metadata_0.json"])

    def fake_cp(_src, dst):
        with open(dst, "w") as f:
            json.dump({"uuid": "lab+user+2023-01-01-00h-00m-00s", "trajectory_length": 100}, f)

    monkeypatch.setattr(mod, "gsutil_cp", fake_cp)
    monkeypatch.setattr(mod, "h5py", SimpleNamespace(File=lambda *_args, **_kwargs: DummyH5File(np.zeros((10, 7)))))

    result = mod.process_episode("gs://bucket/episode/", {})
    assert result == {"skip": "static"}


def test_process_episode_empty_trajectory(monkeypatch):
    monkeypatch.setattr(mod, "gsutil_ls", lambda _: ["gs://bucket/episode/metadata_0.json"])

    def fake_cp(_src, dst):
        with open(dst, "w") as f:
            json.dump({"uuid": "lab+user+2023-01-01-00h-00m-00s", "trajectory_length": 100}, f)

    monkeypatch.setattr(mod, "gsutil_cp", fake_cp)
    monkeypatch.setattr(mod, "gsutil_cp_dir", lambda *_args, **_kwargs: None)
    joint_vel = np.tile(np.arange(10)[:, None], (1, 7)).astype(float)
    monkeypatch.setattr(mod, "h5py", SimpleNamespace(File=lambda *_args, **_kwargs: DummyH5File(joint_vel)))
    monkeypatch.setattr(mod, "find_next_annotation", lambda *_args, **_kwargs: "task")
    monkeypatch.setattr(mod, "load_trajectory", lambda *_args, **_kwargs: [])

    result = mod.process_episode("gs://bucket/episode/", {})
    assert "error" in result and "empty trajectory" in result["error"]


def test_process_episode_degenerate_cameras(monkeypatch):
    monkeypatch.setattr(mod, "gsutil_ls", lambda _: ["gs://bucket/episode/metadata_0.json"])

    def fake_cp(_src, dst):
        with open(dst, "w") as f:
            json.dump({"uuid": "lab+user+2023-01-01-00h-00m-00s", "trajectory_length": 100}, f)

    monkeypatch.setattr(mod, "gsutil_cp", fake_cp)
    monkeypatch.setattr(mod, "gsutil_cp_dir", lambda *_args, **_kwargs: None)
    joint_vel = np.tile(np.arange(10)[:, None], (1, 7)).astype(float)
    monkeypatch.setattr(mod, "h5py", SimpleNamespace(File=lambda *_args, **_kwargs: DummyH5File(joint_vel)))
    monkeypatch.setattr(mod, "find_next_annotation", lambda *_args, **_kwargs: "task")

    def fake_load(_h5, _mp4):
        return [
            {
                "observation": {
                    "camera_type": {"cam0": 0},
                    "image": {"cam0": np.zeros((10, 10, 3), dtype=np.uint8)},
                    "robot_state": {
                        "joint_positions": np.zeros(7),
                        "gripper_position": 0.0,
                    },
                },
                "action": {
                    "joint_velocity": np.ones(7),
                    "gripper_position": 0.0,
                },
            }
        ]

    monkeypatch.setattr(mod, "load_trajectory", fake_load)

    result = mod.process_episode("gs://bucket/episode/", {})
    assert "error" in result and "degenerate cameras" in result["error"]


def test_main_stress_many_episodes(monkeypatch, tmp_path):
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_args, **_kwargs: {})

    # Provide a deterministic episode list.
    episode_list = [f"gs://bucket/episode_{i}/" for i in range(20)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    # Alternate results: converted, error, skip
    def fake_process(path, _index):
        idx = int(path.rstrip("/").split("_")[-1])
        if idx % 3 == 0:
            return {"frames": [{"foo": idx}], "task": "t", "uuid": f"u{idx}"}
        if idx % 3 == 1:
            return {"error": f"boom {idx}"}
        return {"skip": "too_short"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    mod.main(
        repo_id="test/repo",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=4,
    )

    dataset = DummyLeRobotDataset.last_dataset
    assert dataset is not None
    # Converted count should be indices divisible by 3: 0,3,6,9,12,15,18 -> 7
    assert dataset.episodes == 7
    assert len(dataset.frames) == 7


def test_main_respects_max_episodes(monkeypatch, tmp_path):
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_args, **_kwargs: {})

    episode_list = [f"gs://bucket/episode_{i}/" for i in range(10)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    def fake_process(_path, _index):
        return {"frames": [{"foo": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    mod.main(
        repo_id="test/repo2",
        annotations_path="ignored.json",
        max_episodes=3,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=2,
    )

    dataset = DummyLeRobotDataset.last_dataset
    assert dataset.episodes == 3
    assert len(dataset.frames) == 3


def test_main_oom_in_worker(monkeypatch, tmp_path):
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_args, **_kwargs: {})

    episode_list = [f"gs://bucket/episode_{i}/" for i in range(5)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    def fake_process(_path, _index):
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(mod, "process_episode", fake_process)

    # Should abort gracefully without raising.
    mod.main(
        repo_id="test/oom",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=2,
    )

    dataset = DummyLeRobotDataset.last_dataset
    assert dataset.episodes == 0


def test_main_oom_in_writer(monkeypatch, tmp_path):
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_args, **_kwargs: {})

    episode_list = [f"gs://bucket/episode_{i}/" for i in range(3)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    def fake_process(_path, _index):
        return {"frames": [{"foo": 1}, {"foo": 2}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    dataset = DummyDataset()

    def create_with_oom(cls, **_kwargs):
        cls.last_dataset = dataset
        return dataset

    def oom_add_frame(_frame):
        raise MemoryError("simulated OOM in writer")

    dataset.add_frame = oom_add_frame
    monkeypatch.setattr(DummyLeRobotDataset, "create", classmethod(create_with_oom))

    mod.main(
        repo_id="test/oom2",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=2,
    )

    # Should abort without saving episodes.
    assert dataset.episodes == 0


def test_main_rlimit_oom_worker(monkeypatch, tmp_path):
    resource = pytest.importorskip("resource")
    if not hasattr(resource, "RLIMIT_AS"):
        pytest.skip("RLIMIT_AS not available on this platform")

    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_args, **_kwargs: {})

    episode_list = [f"gs://bucket/episode_{i}/" for i in range(2)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    limit_bytes = 256 * 1024 * 1024

    # Run in a subprocess so rlimit and potential hangs are contained.
    code = f"""
import os
import sys
import resource
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, {repr(str(ROOT))})
# Stub h5py to avoid heavy import under tight limits.
sys.modules.setdefault("h5py", SimpleNamespace(File=lambda *_a, **_k: None))
import convert_droid_failures_to_lerobot_parallel as mod

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
try:
    resource.setrlimit(resource.RLIMIT_AS, ({limit_bytes}, hard))
except (ValueError, PermissionError):
    sys.exit(200)

try:
    try:
        buf = bytearray({limit_bytes} * 2)
        buf[0] = 1
        buf[-1] = 1
    except MemoryError:
        pass
    else:
        sys.exit(201)

    def fake_process(_path, _index):
        buf = bytearray({limit_bytes} * 2)
        for i in range(0, len(buf), 4096):
            buf[i] = 1
        return {{"frames": [{{"foo": 1}}], "task": "t", "uuid": "u"}}

    mod.process_episode = fake_process
    mod.main(
        repo_id="test/rlimit",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path={repr(str(list_path))},
        num_workers=1,
    )
except MemoryError:
    # If MemoryError escapes, consider it a failure.
    sys.exit(1)
"""

    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(ROOT),
        timeout=30,
        capture_output=True,
        text=True,
    )
    if result.returncode == 200:
        pytest.skip("RLIMIT_AS cannot be set in this environment")
    if result.returncode == 201:
        pytest.skip("RLIMIT_AS not enforced; skipping stress test")
    assert result.returncode == 0, f"subprocess failed: rc={result.returncode}\\nstdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"


def test_main_peak_inflight_frames(monkeypatch, tmp_path):
    """
    Instruments process_episode and DummyDataset.add_frame to track the
    maximum number of frames held in memory simultaneously across all in-flight
    workers.  The throttle (max_inflight = num_workers * 2) must bound this to
    prevent OOM under real workloads.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    num_workers = 3
    frames_per_ep = 10
    num_episodes = 24
    max_inflight = num_workers * 2  # mirrors main()'s max_inflight

    episode_list = [f"gs://bucket/ep_{i}/" for i in range(num_episodes)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def fake_process(_path, _index):
        frames = [{"data": i} for i in range(frames_per_ep)]
        with lock:
            state["current"] += frames_per_ep
            if state["current"] > state["peak"]:
                state["peak"] = state["current"]
        return {"frames": frames, "task": "t", "uuid": "u"}

    original_add_frame = DummyDataset.add_frame

    def tracking_add_frame(self, frame):
        # Slow the writer so multiple worker results can accumulate, exercising
        # the throttle.  Each frame drains one unit from the in-flight count.
        time.sleep(0.001)
        with lock:
            state["current"] -= 1
        original_add_frame(self, frame)

    monkeypatch.setattr(mod, "process_episode", fake_process)
    monkeypatch.setattr(DummyDataset, "add_frame", tracking_add_frame)

    mod.main(
        repo_id="test/peak_frames",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=num_workers,
    )

    assert state["peak"] > 0, "No frames were tracked — instrumentation broken"
    upper_bound = max_inflight * frames_per_ep
    assert state["peak"] <= upper_bound, (
        f"Peak in-flight frames {state['peak']} exceeded theoretical max "
        f"{upper_bound} ({max_inflight} futures × {frames_per_ep} frames each); "
        "throttle is not bounding memory as expected"
    )


def test_main_inflight_worker_throttle(monkeypatch, tmp_path):
    """
    Verifies that the number of concurrently executing workers never exceeds
    num_workers (the ThreadPoolExecutor bound).  Workers sleep briefly so
    overlapping invocations are observable.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    num_workers = 3
    num_episodes = 20

    episode_list = [f"gs://bucket/ep_{i}/" for i in range(num_episodes)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def fake_process(_path, _index):
        with lock:
            state["active"] += 1
            if state["active"] > state["peak"]:
                state["peak"] = state["active"]
        time.sleep(0.02)  # hold slot so concurrent calls overlap
        with lock:
            state["active"] -= 1
        return {"frames": [{"x": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    mod.main(
        repo_id="test/throttle",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=num_workers,
    )

    assert state["peak"] >= 1, "No concurrent workers observed"
    assert state["peak"] <= num_workers, (
        f"Peak concurrent workers {state['peak']} exceeded num_workers={num_workers}; "
        "ThreadPoolExecutor is not limiting parallelism correctly"
    )


def test_main_write_error_continues(monkeypatch, tmp_path):
    """
    A non-OOM RuntimeError raised by dataset.add_frame is caught by the
    except-Exception branch and counted as an error; the main loop must
    continue processing all remaining episodes rather than aborting.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    num_episodes = 6
    episode_list = [f"gs://bucket/ep_{i}/" for i in range(num_episodes)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    attempted = {"n": 0}

    def fake_process(_path, _index):
        attempted["n"] += 1
        return {"frames": [{"x": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    # Inject a dataset whose add_frame raises on every even call.
    # add_frame is called from the main thread so the counter is sequential.
    dataset = DummyDataset()
    add_calls = {"n": 0}
    original_add = dataset.add_frame

    def flaky_add(frame):
        add_calls["n"] += 1
        if add_calls["n"] % 2 == 0:
            raise RuntimeError("disk full")
        original_add(frame)

    dataset.add_frame = flaky_add

    def make_dataset(cls, **_kwargs):
        cls.last_dataset = dataset
        return dataset

    monkeypatch.setattr(DummyLeRobotDataset, "create", classmethod(make_dataset))

    mod.main(
        repo_id="test/write_err",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=2,
    )

    # Every episode must be attempted even when some writes fail.
    assert attempted["n"] == num_episodes, (
        f"Only {attempted['n']}/{num_episodes} episodes attempted; "
        "write error may have aborted early"
    )
    # Episodes with successful writes (odd add_calls: 1, 3, 5) are saved.
    assert dataset.episodes == 3


def test_main_episodes_processed_exactly_once(monkeypatch, tmp_path):
    """
    Under concurrent workers no episode is skipped or processed twice;
    the union of all processed paths equals the full input list.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    num_episodes = 15
    episode_list = [f"gs://bucket/ep_{i}/" for i in range(num_episodes)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    lock = threading.Lock()
    processed = []

    def fake_process(path, _index):
        with lock:
            processed.append(path)
        return {"frames": [{"x": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    mod.main(
        repo_id="test/once",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=4,
    )

    assert len(processed) == num_episodes, (
        f"Expected {num_episodes} episodes processed, got {len(processed)}"
    )
    assert len(set(processed)) == num_episodes, (
        "Some episodes were processed more than once"
    )
    assert set(processed) == set(episode_list), (
        "Not all episodes from the input list were processed"
    )


def test_main_resume_skips_done_episodes(monkeypatch, tmp_path):
    """
    With a pre-existing progress file, already-processed episodes are skipped
    and only the remaining ones are passed to process_episode.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    num_total = 10
    num_done = 4
    episode_list = [f"gs://bucket/ep_{i}/" for i in range(num_total)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    # Pre-populate progress file for the first 4 episodes
    progress_path = tmp_path / "episodes.progress.jsonl"
    with open(progress_path, "w") as f:
        for i in range(num_done):
            f.write(json.dumps({"path": episode_list[i], "outcome": "converted"}) + "\n")

    processed = []

    def fake_process(path, _index):
        processed.append(path)
        return {"frames": [{"x": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)
    # Simulate existing dataset dir so _open_dataset_for_resume is used
    (tmp_path / "test" / "resume_skip").mkdir(parents=True)
    monkeypatch.setattr(mod, "_open_dataset_for_resume", lambda **_kw: DummyLeRobotDataset.create())

    mod.main(
        repo_id="test/resume_skip",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=True,
        episode_list_path=str(list_path),
        num_workers=2,
    )

    assert len(processed) == num_total - num_done, (
        f"Expected {num_total - num_done} episodes, got {len(processed)}"
    )
    # None of the already-done episodes should be re-processed
    done_set = set(ep.rstrip("/") + "/" for ep in episode_list[:num_done])
    for path in processed:
        assert path not in done_set, f"Re-processed already-done episode: {path}"


def test_main_resume_progress_file_written(monkeypatch, tmp_path):
    """
    Each episode outcome (converted, skip, error) is written to the progress
    file so a subsequent resume knows what to skip.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    episode_list = [
        "gs://bucket/ep_0/",   # will return skip:too_short
        "gs://bucket/ep_1/",   # will return error
        "gs://bucket/ep_2/",   # will be converted
    ]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    def fake_process(path, _index):
        if "ep_0" in path:
            return {"skip": "too_short"}
        if "ep_1" in path:
            return {"error": "something broke"}
        return {"frames": [{"x": 1}], "task": "t", "uuid": "u"}

    monkeypatch.setattr(mod, "process_episode", fake_process)

    mod.main(
        repo_id="test/progress_write",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=1,
    )

    progress_path = tmp_path / "episodes.progress.jsonl"
    assert progress_path.exists(), "Progress file should be created"
    records = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    assert len(records) == 3, f"Expected 3 progress records, got {len(records)}"

    outcomes_by_path = {r["path"]: r["outcome"] for r in records}
    assert outcomes_by_path["gs://bucket/ep_0/"] == "skip:too_short"
    assert outcomes_by_path["gs://bucket/ep_1/"] == "error"
    assert outcomes_by_path["gs://bucket/ep_2/"] == "converted"


def test_main_fresh_run_clears_progress_file(monkeypatch, tmp_path):
    """
    A non-resume run (resume=False) deletes any stale progress file so the
    new run starts clean.
    """
    install_dummy_lerobot(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "build_annotation_index", lambda *_a, **_k: {})

    list_path = tmp_path / "episodes.txt"
    list_path.write_text("gs://bucket/ep_0/\n")
    # Place a stale progress file from a previous run
    progress_path = tmp_path / "episodes.progress.jsonl"
    progress_path.write_text(json.dumps({"path": "gs://bucket/stale/", "outcome": "converted"}) + "\n")

    monkeypatch.setattr(
        mod, "process_episode",
        lambda _path, _idx: {"frames": [{"x": 1}], "task": "t", "uuid": "u"},
    )

    mod.main(
        repo_id="test/fresh_clears",
        annotations_path="ignored.json",
        max_episodes=-1,
        push_to_hub=False,
        resume=False,
        episode_list_path=str(list_path),
        num_workers=1,
    )

    records = [json.loads(l) for l in progress_path.read_text().splitlines() if l.strip()]
    # Only ep_0 should be in the fresh progress file, not the stale entry
    assert len(records) == 1
    assert records[0]["path"] == "gs://bucket/ep_0/"


@pytest.mark.slow
def test_main_real_stress(monkeypatch, tmp_path):
    if os.environ.get("REAL_STRESS") != "1":
        pytest.skip("Set REAL_STRESS=1 to enable real memory stress test")

    stress_mb = int(os.environ.get("STRESS_MB", "512"))
    workers = int(os.environ.get("STRESS_WORKERS", "2"))
    episodes = int(os.environ.get("STRESS_EPISODES", "4"))

    episode_list = [f"gs://bucket/episode_{i}/" for i in range(episodes)]
    list_path = tmp_path / "episodes.txt"
    list_path.write_text("\n".join(episode_list))

    peak_file = tmp_path / "peak_rss_kb.txt"
    peak_inflight_path = tmp_path / "peak_inflight_frames.txt"
    frames_per_ep = 5
    max_inflight = workers * 2  # mirrors main()'s max_inflight = num_workers * 2

    code = textwrap.dedent(
        f"""
        import os
        import sys
        import threading
        import time
        from types import SimpleNamespace
        from pathlib import Path

        sys.path.insert(0, {repr(str(ROOT))})
        sys.modules.setdefault("h5py", SimpleNamespace(File=lambda *_a, **_k: None))

        # --- In-flight frame counter (shared across worker threads + main thread) ---
        _inflight_lock = threading.Lock()
        _inflight_state = {{"current": 0, "peak": 0}}

        def _update_inflight(delta):
            with _inflight_lock:
                _inflight_state["current"] += delta
                if _inflight_state["current"] > _inflight_state["peak"]:
                    _inflight_state["peak"] = _inflight_state["current"]

        class DummyDataset:
            def add_frame(self, _frame):
                _update_inflight(-1)  # frame consumed by writer
            def save_episode(self): pass
            def finalize(self): pass
            def push_to_hub(self, **_kwargs): pass

        class DummyLeRobotDataset:
            @classmethod
            def create(cls, **_kwargs):
                return DummyDataset()

        lerobot_dataset_mod = SimpleNamespace(
            HF_LEROBOT_HOME=Path({repr(str(tmp_path))}),
            LeRobotDataset=DummyLeRobotDataset,
        )
        datasets_mod = SimpleNamespace(lerobot_dataset=lerobot_dataset_mod)
        lerobot_mod = SimpleNamespace(datasets=datasets_mod)
        sys.modules["lerobot"] = lerobot_mod
        sys.modules["lerobot.datasets"] = datasets_mod
        sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset_mod

        import convert_droid_failures_to_lerobot_parallel as mod
        mod.build_annotation_index = lambda *_a, **_k: {{}}

        peak_path = Path({repr(str(peak_file))})
        peak_inflight_path = Path({repr(str(peak_inflight_path))})
        alloc_bytes = {stress_mb} * 1024 * 1024 // max(1, {workers})

        # Background monitor samples RSS every 5 ms.  This is necessary because
        # the GIL serialises the Python page-touching loops, so workers never
        # truly run simultaneously — but they DO overlap while sleeping (GIL
        # released).  The monitor captures RSS during that sleep window, when
        # all workers hold their allocations concurrently.
        # Peak is kept in memory to avoid file-write latency masking brief
        # high-RSS windows; the final value is written once after shutdown.
        _monitor_stop = threading.Event()
        _peak_rss_kb = [0]
        _peak_lock = threading.Lock()

        def _rss_monitor():
            while not _monitor_stop.is_set():
                try:
                    with open("/proc/self/status") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                rss_kb = int(line.split()[1])
                                break
                        else:
                            _monitor_stop.wait(0.005)
                            continue
                    with _peak_lock:
                        if rss_kb > _peak_rss_kb[0]:
                            _peak_rss_kb[0] = rss_kb
                except Exception:
                    pass
                _monitor_stop.wait(0.005)

        _monitor_t = threading.Thread(target=_rss_monitor, daemon=True)
        _monitor_t.start()

        def fake_process(_path, _index):
            buf = bytearray(alloc_bytes)
            for i in range(0, len(buf), 4096):
                buf[i] = 1
            # Sleep with GIL released so other workers can also finish
            # allocating; during this overlap the monitor records the combined
            # RSS of all concurrently-live buffers.
            time.sleep(0.3)
            frames = [{{"foo": j}} for j in range({frames_per_ep})]
            _update_inflight(len(frames))  # frames now held in this result
            return {{"frames": frames, "task": "t", "uuid": "u"}}

        mod.process_episode = fake_process
        mod.main(
            repo_id="test/stress",
            annotations_path="ignored.json",
            max_episodes=-1,
            push_to_hub=False,
            resume=False,
            episode_list_path={repr(str(list_path))},
            num_workers={workers},
        )
        _monitor_stop.set()
        _monitor_t.join()
        peak_path.write_text(str(_peak_rss_kb[0]))
        peak_inflight_path.write_text(str(_inflight_state["peak"]))
        """
    )

    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(ROOT),
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed: rc={result.returncode}\\nstdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"

    if not peak_file.exists():
        pytest.fail("peak RSS file not written")
    peak_kb = int(peak_file.read_text().strip())
    peak_mb = peak_kb / 1024
    assert peak_mb >= stress_mb * 0.7, (
        f"peak RSS {peak_mb:.1f}MB < expected ~{stress_mb}MB; "
        "workers may not be allocating as expected"
    )
    assert peak_mb <= stress_mb * 2.5, (
        f"peak RSS {peak_mb:.1f}MB far exceeded expected ~{stress_mb}MB; "
        "possible memory leak or throttle failure"
    )

    if not peak_inflight_path.exists():
        pytest.fail("peak in-flight frames file not written")
    peak_inflight = int(peak_inflight_path.read_text().strip())
    assert peak_inflight > 0, "No in-flight frames were tracked"
    assert peak_inflight <= max_inflight * frames_per_ep, (
        f"Peak in-flight frames {peak_inflight} exceeded bound "
        f"{max_inflight * frames_per_ep} ({max_inflight} futures × {frames_per_ep} frames); "
        "throttle is not bounding memory as expected"
    )

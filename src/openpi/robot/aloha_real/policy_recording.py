"""User-delimited, self-contained ALOHA episodes; no ROS or actuator calls."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import re
import threading
import time

import h5py
import numpy as np

from openpi.robot.aloha_real.collection_video import CameraVideo, select_codec
from openpi.robot.aloha_real.collection_storage import EpisodeClaimCollision, StagedEpisode
from openpi.robot.aloha_real.collection_permissions import normalize_path

CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
ALIASES = {"camera_top": "cam_high", "camera_high": "cam_high", "camera_low": "cam_low",
           "camera_wrist_left": "cam_left_wrist", "camera_wrist_right": "cam_right_wrist"}


class PolicyRecording:
    """Stream four collector-compatible MP4 sidecars and a small action/state HDF5."""

    def __init__(self, directory, fps=50.0):
        self.directory = Path(directory)
        self.fps = fps
        self._codec = select_codec()  # Probe before any task/recording starts, not while frames queue.
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=32)
        self._finish = threading.Event()
        self._thread = None
        self._annotations = {}
        self._state = {"status": "idle", "frames": 0, "path": None, "error": None, "batch_name": None}

    def snapshot(self):
        """Expose acknowledgements from the actual writer, not optimistic web state."""
        with self._lock:
            return dict(self._state)

    def _update(self, **values):
        """Synchronize writer and runtime state without holding a lock during I/O."""
        with self._lock:
            self._state.update(values)

    def start(self, metadata, batch_name):
        """Create an exclusively owned output; never overwrite an earlier episode."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Previous recording is still recording or saving")
        if not isinstance(batch_name, str) or not re.fullmatch(r"\w[\w.-]{0,95}", batch_name):
            raise ValueError("Batch name must be 1–96 letters, numbers, underscores, dots or hyphens; no paths")
        self.directory.mkdir(parents=True, exist_ok=True)
        batch_dir = self.directory / batch_name
        if batch_dir.resolve().parent != self.directory.resolve():
            raise ValueError("Batch directory must remain inside the collection data root")
        batch_dir.mkdir(exist_ok=True)
        normalize_path(batch_dir)
        # Share the canonical collector's .claim and flock protocol, not just its filenames.
        episode_index = 0
        while True:
            episode_dir = batch_dir / f"episode_{episode_index}"
            if episode_dir.exists() or episode_dir.is_symlink() or (batch_dir / f"episode_{episode_index}.hdf5").exists() or (batch_dir / f"episode_{episode_index}.partial.hdf5").exists():
                episode_index += 1
                continue
            try:
                staged = StagedEpisode.create(batch_dir, episode_index)
                break
            except EpisodeClaimCollision:
                episode_index += 1
        path = episode_dir / "episode.hdf5"
        self._queue = queue.Queue(maxsize=32)
        self._finish.clear()
        self._annotations = dict(metadata.get("annotations", {}))
        self._update(status="recording", frames=0, path=str(path), error=None, batch_name=batch_name, staging_path=str(staged.staging_path))
        self._thread = threading.Thread(target=self._write, args=(staged, path, {**metadata, "batch_name": batch_name}), daemon=False)
        self._thread.start()

    def append(self, observation, action, metadata):
        """Copy the observation BEFORE action and the exact command sent to the robot."""
        state = self.snapshot()
        if state["status"] == "error":
            raise RuntimeError(state["error"])
        if state["status"] != "recording":
            return
        images = {ALIASES.get(k, k): v for k, v in observation["images"].items()}
        frame = {key: np.asarray(observation[key], np.float32).copy() for key in ("qpos", "qvel", "effort")}
        frame["images"] = {}
        for key in CAMERAS:
            image = np.asarray(images[key])
            # History is oldest -> newest; save one current image per action.
            # Leave the full history in the policy observation untouched.
            if image.ndim == 4 and image.shape[0] > 0 and image.shape[-1] == 3:
                image = image[-1]
            frame["images"][key] = CameraVideo.prepare(image).copy()
        frame["action"] = np.asarray(action["actions"], np.float32).copy()
        frame["timestamp"] = time.time()
        frame["metadata"] = json.dumps(metadata)
        try:
            self._queue.put_nowait(frame)
        except queue.Full as exc:
            self._update(status="error", error="Recording writer cannot keep up; robot paused; partial file retained")
            self._finish.set()
            raise RuntimeError(self.snapshot()["error"]) from exc

    def finish(self, annotations=None):
        """Return immediately: the robot can remain paused while disk I/O drains."""
        if annotations is not None:
            with self._lock:
                self._annotations = dict(annotations)
        if self.snapshot()["status"] == "recording":
            self._update(status="saving")
        self._finish.set()

    def close(self):
        """Flush queued data on normal runtime shutdown."""
        self.finish()
        if self._thread is not None:
            self._thread.join()

    def _write(self, staged, path, metadata):
        """Match the current collect.sh sidecar layout; publish only a validated bundle."""
        staging = staged.staging_path
        partial = staging / "episode.hdf5"
        writers = {}
        try:
            codec = self._codec
            self._update(encoder=codec)
            with h5py.File(partial, "x") as root:
                root.attrs.update(sim=False, compress=False, image_storage="mp4", video_fps=self.fps,
                                  source="policy_rollout", eii_storage_format="aloha_video_sidecar_v1",
                                  home_required=False, home_recording_mode="continuous", home_verified=False,
                                  home_start_frame=-1, home_verify_frame_count=0, video_encoder=codec)
                root.create_dataset("collection_metadata", data=json.dumps(metadata), dtype=h5py.string_dtype())
                obs = root.create_group("observations")
                for name in ("qpos", "qvel", "effort"):
                    obs.create_dataset(name, (0, 14), maxshape=(None, 14), dtype="float32", chunks=True)
                root.create_dataset("action", (0, 14), maxshape=(None, 14), dtype="float32", chunks=True)
                root.create_dataset("timestamps", (0,), maxshape=(None,), dtype="float64", chunks=True)
                root.create_dataset("inference_metadata", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
                for name, dtype in (("phase", "uint8"), ("done", "bool")):
                    root.create_dataset(name, (0,), maxshape=(None,), dtype=dtype, chunks=True)
                videos = obs.create_group("videos")
                videos.attrs["fps"] = self.fps
                count = 0
                try:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        while not self._finish.is_set() or not self._queue.empty():
                            try:
                                frame = self._queue.get(timeout=0.05)
                            except queue.Empty:
                                continue
                            for name in ("qpos", "qvel", "effort", "action"):
                                if frame[name].shape != (14,) or not np.isfinite(frame[name]).all():
                                    raise ValueError(f"Invalid {name}: expected 14 finite values")
                            for camera in CAMERAS:
                                if camera not in writers:
                                    writer = CameraVideo(staging / f"{camera}.mp4", frame["images"][camera], self.fps, codec)
                                    writers[camera] = writer
                                    reference = videos.create_dataset(camera, data=f"{camera}.mp4", dtype=h5py.string_dtype())
                                    reference.attrs.update(fps=self.fps, width=writer.width, height=writer.height, codec="h264")
                            futures = [pool.submit(writers[camera].write, frame["images"][camera]) for camera in CAMERAS]
                            for future in futures:
                                future.result()
                            for name in ("qpos", "qvel", "effort"):
                                obs[name].resize((count + 1, 14))
                                obs[name][count] = frame[name]
                            root["action"].resize((count + 1, 14))
                            root["action"][count] = frame["action"]
                            values = (("timestamps", frame["timestamp"]), ("inference_metadata", frame["metadata"]),
                                      ("phase", 0), ("done", False))
                            for name, value in values:
                                root[name].resize((count + 1,))
                                root[name][count] = value
                            count += 1
                            self._update(frames=count)
                            if count % 50 == 0:
                                root.flush()
                finally:
                    errors = []
                    for writer in writers.values():
                        try:
                            writer.close()
                        except Exception as exc:
                            errors.append(str(exc))
                    if errors:
                        raise RuntimeError("; ".join(errors))
                root.attrs["video_frame_count"] = count
                with self._lock:
                    metadata = {**metadata, "annotations": dict(self._annotations)}
                root["collection_metadata"][()] = json.dumps(metadata)
                if not count:
                    raise ValueError("No frames recorded; partial file retained")
                if self.snapshot()["status"] == "error":
                    return
            with h5py.File(partial, "r") as check:
                for key in ("action", "observations/qpos", "observations/qvel", "observations/effort"):
                    if check[key].shape != (count, 14):
                        raise ValueError(f"Invalid saved shape: {key}")
                if len(writers) != 4 or any(writer.frames != count for writer in writers.values()):
                    raise ValueError("Camera/action frame count mismatch")
            for entry in (staging, *staging.rglob("*")):
                normalize_path(entry)  # Same eii ownership and 775/664 modes as the collector.
            staged.publish(allow_existing_destination=False)
            self._update(status="saved", staging_path=None)
        except Exception as exc:
            self._update(status="error", error=f"{exc}; partial file: {partial}")
            self._finish.set()
        finally:
            # Preserve failed staging for diagnosis, but release only this writer's owned claim.
            staged.claim.release()

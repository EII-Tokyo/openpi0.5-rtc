import logging
import pathlib
import shutil
import subprocess
import time
from typing import Literal

import cv2
import h5py
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


VIDEO_FOURCCS = {
    "mp4v": "mp4v",
    "avc1": "avc1",
}


class FfmpegH264Writer:
    def __init__(self, path: pathlib.Path, fps: float, width: int, height: int) -> None:
        self._path = path
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image_rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is closed for {self._path}")
        self._process.stdin.write(np.ascontiguousarray(image_rgb).tobytes())

    def release(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code} for {self._path}")


class VideoHdf5Saver(_subscriber.Subscriber):
    """Save rollout metadata to a small HDF5 file and camera streams to MP4 files."""

    def __init__(
        self,
        dataset_dir: str,
        *,
        fps: float = 50.0,
        is_mobile: bool = False,
        split_on_reset: bool = False,
        reset_position: list[list[float]] | None = None,
        home_threshold: float = 0.15,
        leave_threshold: float = 0.30,
        stable_home_steps: int = 25,
        min_episode_steps: int = 50,
        video_codec: Literal["h264", "mp4v", "avc1"] = "h264",
    ) -> None:
        if split_on_reset and reset_position is None:
            raise ValueError("reset_position is required when split_on_reset=True")
        if split_on_reset and len(reset_position) != 2:
            raise ValueError("reset_position must contain left and right arm poses")
        if fps <= 0:
            raise ValueError("fps must be > 0 for video saving")
        if video_codec != "h264" and video_codec not in VIDEO_FOURCCS:
            raise ValueError(f"Unsupported video codec: {video_codec}")

        self._dataset_dir = pathlib.Path(dataset_dir)
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._is_mobile = is_mobile
        self._split_on_reset = split_on_reset
        self._reset_position = (
            np.asarray(reset_position, dtype=np.float32)
            if reset_position is not None
            else None
        )
        self._home_threshold = home_threshold
        self._leave_threshold = leave_threshold
        self._stable_home_steps = stable_home_steps
        self._min_episode_steps = min_episode_steps
        self._video_codec = video_codec
        self._fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCCS[video_codec]) if video_codec != "h264" else None

        self._recording_segment = not split_on_reset
        self._stable_home_count = 0
        self._episode_dir: pathlib.Path | None = None
        self._writers: dict[str, cv2.VideoWriter | FfmpegH264Writer] = {}
        self._camera_names: list[str] = []
        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._effort: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._base_actions: list[np.ndarray] = []
        self._timestamps: list[float] = []

    @override
    def on_episode_start(self) -> None:
        self._close_writers()
        self._clear_buffers()
        self._recording_segment = not self._split_on_reset
        self._stable_home_count = 0
        if not self._split_on_reset:
            self._episode_dir = self._next_episode_dir()

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        if self._split_on_reset:
            self._on_step_split_on_reset(observation, action)
            return

        if self._episode_dir is None:
            self._episode_dir = self._next_episode_dir()
        self._append_step(observation, action)

    def _on_step_split_on_reset(self, observation: dict, action: dict) -> None:
        home_error = self._reset_pose_error(observation)
        is_home = home_error <= self._home_threshold
        has_left_home = home_error >= self._leave_threshold

        if not self._recording_segment:
            if not has_left_home:
                return
            self._episode_dir = self._next_episode_dir()
            self._recording_segment = True
            self._stable_home_count = 0
            logging.info(
                "Started video rollout segment after leaving reset pose "
                "(reset error %.3f >= %.3f).",
                home_error,
                self._leave_threshold,
            )

        self._append_step(observation, action)

        if is_home:
            self._stable_home_count += 1
        else:
            self._stable_home_count = 0

        if self._stable_home_count >= self._stable_home_steps:
            if len(self._actions) >= self._min_episode_steps:
                self._finalize_current_episode()
            else:
                logging.info("Dropping short video rollout segment with %d steps.", len(self._actions))
                self._drop_current_episode()
            self._recording_segment = False
            self._stable_home_count = 0

    def _append_step(self, observation: dict, action: dict) -> None:
        if self._episode_dir is None:
            raise RuntimeError("episode directory is not initialized")
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_writers(observation)

        self._qpos.append(np.asarray(observation["qpos"], dtype=np.float32))
        self._qvel.append(np.asarray(observation["qvel"], dtype=np.float32))
        self._effort.append(np.asarray(observation["effort"], dtype=np.float32))
        self._actions.append(np.asarray(action["actions"], dtype=np.float32))
        if self._is_mobile:
            self._base_actions.append(np.asarray(observation["base_vel"], dtype=np.float32))
        self._timestamps.append(time.time())

        for cam_name in self._camera_names:
            image = observation["images"][cam_name]
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            image = np.asarray(image)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if self._video_codec == "h264":
                self._writers[cam_name].write(image)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self._writers[cam_name].write(image_bgr)

    def _ensure_writers(self, observation: dict) -> None:
        if self._writers:
            return

        self._camera_names = [
            name
            for name in observation.get("images", {}).keys()
            if "_depth" not in name
        ]
        if not self._camera_names:
            raise ValueError("No RGB camera images found in observation")

        for cam_name in self._camera_names:
            image = observation["images"][cam_name]
            if image.ndim == 3 and image.shape[0] == 3:
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[:2]

            path = self._episode_dir / f"{cam_name}.mp4"
            if self._video_codec == "h264":
                writer = FfmpegH264Writer(path, self._fps, width, height)
            else:
                writer = cv2.VideoWriter(str(path), self._fourcc, self._fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {path}")
            self._writers[cam_name] = writer

    def _reset_pose_error(self, observation: dict) -> float:
        if self._reset_position is None:
            raise RuntimeError("reset_position is required for split-on-reset mode")

        qpos = np.asarray(observation["qpos"], dtype=np.float32)
        if qpos.shape[0] < 13:
            raise ValueError(f"Expected qpos with at least 13 values, got shape {qpos.shape}")

        left_error = np.max(np.abs(qpos[:6] - self._reset_position[0]))
        right_error = np.max(np.abs(qpos[7:13] - self._reset_position[1]))
        return float(max(left_error, right_error))

    @override
    def on_episode_end(self, episode_subdir: str | None = None) -> None:
        if episode_subdir and self._episode_dir is not None:
            self._close_writers()
            target_dir = self._dataset_dir / episode_subdir / self._episode_dir.name
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            self._episode_dir.rename(target_dir)
            self._episode_dir = target_dir

        if self._split_on_reset and not self._actions:
            self._recording_segment = False
            self._stable_home_count = 0
            return

        if self._split_on_reset and len(self._actions) < self._min_episode_steps:
            logging.info("Dropping short trailing video rollout segment with %d steps.", len(self._actions))
            self._drop_current_episode()
            self._recording_segment = False
            self._stable_home_count = 0
            return

        self._finalize_current_episode()
        self._recording_segment = not self._split_on_reset
        self._stable_home_count = 0

    def _finalize_current_episode(self) -> None:
        if not self._actions:
            logging.warning("No data to save, skipping video episode.")
            self._drop_current_episode()
            return

        self._close_writers()
        assert self._episode_dir is not None
        hdf5_path = self._episode_dir / "episode.hdf5"
        with h5py.File(hdf5_path, "w") as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["images_external"] = True
            root.attrs["image_format"] = "mp4"
            root.attrs["camera_names"] = np.asarray(self._camera_names, dtype="S")
            root.attrs["fps"] = self._fps

            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=np.asarray(self._qpos, dtype=np.float32))
            obs.create_dataset("qvel", data=np.asarray(self._qvel, dtype=np.float32))
            obs.create_dataset("effort", data=np.asarray(self._effort, dtype=np.float32))
            root.create_dataset("action", data=np.asarray(self._actions, dtype=np.float32))
            root.create_dataset("timestamps", data=np.asarray(self._timestamps, dtype=np.float64))
            if self._is_mobile:
                root.create_dataset("base_action", data=np.asarray(self._base_actions, dtype=np.float32))

        logging.info(
            "Saved video episode to %s (%d frames, %d cameras).",
            self._episode_dir,
            len(self._actions),
            len(self._camera_names),
        )
        self._clear_buffers()
        self._episode_dir = None

    def _drop_current_episode(self) -> None:
        self._close_writers()
        if self._episode_dir is not None and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir)
        self._clear_buffers()
        self._episode_dir = None

    def _next_episode_dir(self) -> pathlib.Path:
        existing_dirs = [
            path
            for path in self._dataset_dir.glob("episode_[0-9]*")
            if path.is_dir() and path.name.split("_", 1)[1].isdigit()
        ]
        next_idx = max((int(path.name.split("_", 1)[1]) for path in existing_dirs), default=-1) + 1
        return self._dataset_dir / f"episode_{next_idx}"

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def _clear_buffers(self) -> None:
        self._camera_names = []
        self._qpos.clear()
        self._qvel.clear()
        self._effort.clear()
        self._actions.clear()
        self._base_actions.clear()
        self._timestamps.clear()

import logging
import pathlib
import time
from collections import deque

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from examples.aloha_real import hdf5_utils as _hdf5_utils
from typing_extensions import override


class H5dfSaver(_subscriber.Subscriber):
    """保存 episode 数据到 h5df 文件。"""

    def __init__(
        self,
        dataset_dir: str,
        compress_images: bool = True,
        is_mobile: bool = False,
        fps: float | None = None,
        max_buffer_seconds: float | None = 60.0,
        split_on_reset: bool = False,
        reset_position: list[list[float]] | None = None,
        home_threshold: float = 0.15,
        leave_threshold: float = 0.30,
        stable_home_steps: int = 25,
        min_episode_steps: int = 50,
    ) -> None:
        """
        初始化 H5dfSaver。

        :param dataset_dir: 数据集保存目录
        :param compress_images: 是否压缩图像
        :param is_mobile: 是否是移动机器人（需要保存 base_action）
        """
        if split_on_reset and reset_position is None:
            raise ValueError("reset_position is required when split_on_reset=True")
        if split_on_reset and len(reset_position) != 2:
            raise ValueError("reset_position must contain left and right arm poses")

        self._dataset_dir = pathlib.Path(dataset_dir)
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self._compress_images = compress_images
        self._is_mobile = is_mobile
        self._fps = fps
        self._max_buffer_seconds = max_buffer_seconds
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
        self._recording_segment = not split_on_reset
        self._stable_home_count = 0

        # 临时存储原始数据（不在on_step中填充data_dict）
        self._observations = deque()
        self._actions = deque()
        self._timestamps = deque()

    @override
    def on_episode_start(self) -> None:
        """Episode 开始时初始化数据存储。"""
        # 重置临时存储
        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._recording_segment = not self._split_on_reset
        self._stable_home_count = 0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """保存每个 step 的 observation 和 action（只收集原始数据，不填充data_dict）。"""
        if self._split_on_reset:
            self._on_step_split_on_reset(observation, action)
            return

        self._append_step(observation, action)

    def _on_step_split_on_reset(self, observation: dict, action: dict) -> None:
        home_error = self._reset_pose_error(observation)
        is_home = home_error <= self._home_threshold
        has_left_home = home_error >= self._leave_threshold

        if not self._recording_segment:
            if not has_left_home:
                return
            logging.info(
                "Started HDF5 rollout segment after leaving reset pose "
                "(reset error %.3f >= %.3f).",
                home_error,
                self._leave_threshold,
            )
            self._recording_segment = True
            self._stable_home_count = 0

        self._append_step(observation, action)

        if is_home:
            self._stable_home_count += 1
        else:
            self._stable_home_count = 0

        if self._stable_home_count >= self._stable_home_steps:
            if len(self._observations) >= self._min_episode_steps:
                logging.info(
                    "Saving HDF5 rollout segment after %d stable reset-pose steps.",
                    self._stable_home_count,
                )
                self._save_current_episode()
            else:
                logging.info(
                    "Dropping short HDF5 rollout segment with %d steps.",
                    len(self._observations),
                )
                self._clear_buffers()
            self._recording_segment = False
            self._stable_home_count = 0

    def _append_step(self, observation: dict, action: dict) -> None:
        now = time.time()
        self._observations.append(observation)
        self._actions.append(action)
        self._timestamps.append(now)

        if self._max_buffer_seconds is not None and self._max_buffer_seconds > 0:
            while self._timestamps and now - self._timestamps[0] > self._max_buffer_seconds:
                self._observations.popleft()
                self._actions.popleft()
                self._timestamps.popleft()

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
        """Episode 结束时保存数据到 h5df 文件。"""
        if self._split_on_reset and not self._observations:
            self._recording_segment = False
            self._stable_home_count = 0
            return

        if self._split_on_reset and self._observations and len(self._observations) < self._min_episode_steps:
            logging.info(
                "Dropping short trailing HDF5 rollout segment with %d steps.",
                len(self._observations),
            )
            self._clear_buffers()
            self._recording_segment = False
            self._stable_home_count = 0
            return

        self._save_current_episode(episode_subdir=episode_subdir)

    def _save_current_episode(self, episode_subdir: str | None = None) -> None:
        if not self._observations:
            logging.warning("没有数据可保存，跳过保存。")
            return

        observations = list(self._observations)
        actions = [action["actions"] for action in self._actions]
        timestamps = list(self._timestamps)
        dataset_dir = self._dataset_dir
        if episode_subdir:
            dataset_dir = self._dataset_dir / episode_subdir
            dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path, compressed_len = _hdf5_utils.save_hdf5_episode(
            observations,
            actions,
            dataset_dir,
            compress_images=self._compress_images,
            is_mobile=self._is_mobile,
            fps=self._fps,
            timestamps=timestamps,
        )

        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()

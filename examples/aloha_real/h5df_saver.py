import logging
import pathlib
import time
from collections import deque

from openpi_client.runtime import subscriber as _subscriber
from examples.aloha_real import hdf5_utils as _hdf5_utils
from typing_extensions import override

class H5dfSaver(_subscriber.Subscriber):
    """保存 episode 数据到 h5df 文件。"""

    _MAX_BUFFER_SECONDS = 5.0

    def __init__(
        self,
        dataset_dir: str,
        compress_images: bool = True,
        is_mobile: bool = False,
        fps: float | None = None,
    ) -> None:
        """
        初始化 H5dfSaver。

        :param dataset_dir: 数据集保存目录
        :param compress_images: 是否压缩图像
        :param is_mobile: 是否是移动机器人（需要保存 base_action）
        """
        self._dataset_dir = pathlib.Path(dataset_dir)
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self._compress_images = compress_images
        self._is_mobile = is_mobile
        self._fps = fps

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

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """保存每个 step 的 observation 和 action（只收集原始数据，不填充data_dict）。"""
        # 只保存原始数据
        now = time.time()
        self._observations.append(observation)
        self._actions.append(action)
        self._timestamps.append(now)

        while self._timestamps and now - self._timestamps[0] > self._MAX_BUFFER_SECONDS:
            self._observations.popleft()
            self._actions.popleft()
            self._timestamps.popleft()

    @override
    def on_episode_end(self, episode_subdir: str | None = None) -> None:
        """Episode 结束时保存数据到 h5df 文件。"""
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

        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()

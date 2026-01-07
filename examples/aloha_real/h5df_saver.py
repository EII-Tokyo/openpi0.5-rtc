import logging
import os
import pathlib
import time
from typing import Dict, List

import cv2
import h5py
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class H5dfSaver(_subscriber.Subscriber):
    """保存 episode 数据到 h5df 文件。"""

    def __init__(
        self,
        dataset_dir: str,
        compress_images: bool = True,
        is_mobile: bool = False,
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

        # 数据存储
        self._data_dict: Dict[str, List] = {}
        self._camera_names: List[str] = []
        self._episode_idx = 0

    @override
    def on_episode_start(self) -> None:
        """Episode 开始时初始化数据存储。"""
        self._data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/observations/effort": [],
            "/action": [],
        }
        if self._is_mobile:
            self._data_dict["/base_action"] = []

        # 重置相机名称列表，将在第一次 on_step 时确定
        self._camera_names = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """保存每个 step 的 observation 和 action。"""
        # 第一次 step 时确定相机名称
        if not self._camera_names and "images" in observation:
            self._camera_names = list(observation["images"].keys())
            for cam_name in self._camera_names:
                self._data_dict[f"/observations/images/{cam_name}"] = []

        # 保存 qpos, qvel, effort
        if "qpos" in observation:
            self._data_dict["/observations/qpos"].append(
                np.array(observation["qpos"], dtype=np.float32)
            )
        elif "state" in observation:
            # 兼容性：如果没有 qpos，使用 state
            self._data_dict["/observations/qpos"].append(
                np.array(observation["state"], dtype=np.float32)
            )

        if "qvel" in observation:
            self._data_dict["/observations/qvel"].append(
                np.array(observation["qvel"], dtype=np.float32)
            )

        if "effort" in observation:
            self._data_dict["/observations/effort"].append(
                np.array(observation["effort"], dtype=np.float32)
            )

        # 保存 action
        if "actions" in action:
            self._data_dict["/action"].append(
                np.array(action["actions"], dtype=np.float32)
            )

        # 保存 base_action（如果是移动机器人）
        if self._is_mobile and "base_vel" in observation:
            self._data_dict["/base_action"].append(
                np.array(observation["base_vel"], dtype=np.float32)
            )

        # 保存图像
        if "images" in observation:
            for cam_name in self._camera_names:
                if cam_name in observation["images"]:
                    img = observation["images"][cam_name]
                    # 图像格式可能是 [C, H, W]，需要转换为 [H, W, C] 用于保存
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                    self._data_dict[f"/observations/images/{cam_name}"].append(img)

    @override
    def on_episode_end(self) -> None:
        """Episode 结束时保存数据到 h5df 文件。"""
        if not self._data_dict["/observations/qpos"]:
            logging.warning("没有数据可保存，跳过保存。")
            return

        # 获取 episode 索引
        existing_files = list(self._dataset_dir.glob("episode_[0-9]*.hdf5"))
        if existing_files:
            max_idx = max(
                [
                    int(f.stem.split("_")[1])
                    for f in existing_files
                    if f.stem.split("_")[1].isdigit()
                ],
                default=-1,
            )
            self._episode_idx = max_idx + 1
        else:
            self._episode_idx = 0

        dataset_path = self._dataset_dir / f"episode_{self._episode_idx}.hdf5"
        logging.info(f"保存数据到: {dataset_path}")

        max_timesteps = len(self._data_dict["/observations/qpos"])

        # 处理图像压缩
        compressed_len = None
        if self._compress_images and self._camera_names:
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            compressed_len = []
            for cam_name in self._camera_names:
                image_list = self._data_dict[f"/observations/images/{cam_name}"]
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode(
                        ".jpg", image, encode_param
                    )
                    if not result:
                        logging.error(f"图像编码失败: {cam_name}")
                        encoded_image = np.array([], dtype=np.uint8)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                self._data_dict[f"/observations/images/{cam_name}"] = compressed_list
            logging.info(f"图像压缩耗时: {time.time() - t0:.2f}s")

            # 填充压缩图像以确保一致性
            if compressed_len:
                t0 = time.time()
                compressed_len = np.array(compressed_len)
                padded_size = compressed_len.max()
                for cam_name in self._camera_names:
                    padded_images = []
                    for compressed_image in self._data_dict[
                        f"/observations/images/{cam_name}"
                    ]:
                        padded_img = np.zeros(padded_size, dtype="uint8")
                        padded_img[: len(compressed_image)] = compressed_image
                        padded_images.append(padded_img)
                    self._data_dict[
                        f"/observations/images/{cam_name}"
                    ] = padded_images
                logging.info(f"图像填充耗时: {time.time() - t0:.2f}s")

        # 转换为 numpy 数组
        for key in self._data_dict:
            if self._data_dict[key]:
                self._data_dict[key] = np.array(self._data_dict[key])

        # 计算 total_size（用于 qpos, qvel, effort, action）
        # 假设是双机械臂，每个 7 个自由度（6 个关节 + 1 个 gripper）
        total_size = len(self._data_dict["/action"][0]) if self._data_dict["/action"] else 14

        # 写入 h5df 文件
        t0 = time.time()
        with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = self._compress_images

            obs = root.create_group("observations")

            # 创建图像数据集
            if self._camera_names:
                image_group = obs.create_group("images")
                for cam_name in self._camera_names:
                    if self._compress_images:
                        padded_size = compressed_len.max() if compressed_len is not None else 0
                        shape = (max_timesteps, padded_size)
                    else:
                        # 假设图像尺寸为 480x640x3
                        shape = (max_timesteps, 480, 640, 3)
                    _ = image_group.create_dataset(
                        cam_name, shape, dtype="uint8", chunks=(1, shape[1] if len(shape) > 1 else 1)
                    )

            # 创建 qpos, qvel, effort 数据集
            _ = obs.create_dataset("qpos", (max_timesteps, total_size), dtype=np.float32)
            _ = obs.create_dataset("qvel", (max_timesteps, total_size), dtype=np.float32)
            _ = obs.create_dataset("effort", (max_timesteps, total_size), dtype=np.float32)
            _ = root.create_dataset("action", (max_timesteps, total_size), dtype=np.float32)

            if self._is_mobile:
                _ = root.create_dataset("base_action", (max_timesteps, 2), dtype=np.float32)

            # 写入数据
            for name, array in self._data_dict.items():
                if name.startswith("/"):
                    # 移除开头的 "/"
                    name = name[1:]
                root[name][...] = array

            # 保存压缩长度信息
            if self._compress_images and compressed_len is not None:
                _ = root.create_dataset(
                    "compress_len", (len(self._camera_names), max_timesteps), dtype=np.int32
                )
                root["/compress_len"][...] = compressed_len

        logging.info(f"保存完成，耗时: {time.time() - t0:.1f}s")


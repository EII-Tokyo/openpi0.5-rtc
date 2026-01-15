import logging
import pathlib
import time
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np


def save_hdf5_episode(
    observations: Sequence[dict],
    actions: Sequence,
    dataset_dir: str | pathlib.Path,
    *,
    compress_images: bool = True,
    is_mobile: bool = False,
    episode_idx: Optional[int] = None,
    dataset_prefix: str = "episode_",
    fps: Optional[float] = None,
    timestamps: Optional[Sequence[float]] = None,
) -> Tuple[pathlib.Path, Optional[np.ndarray]]:
    """Save an episode to an hdf5 file and return (path, compress_len)."""
    if not observations:
        logging.warning("没有数据可保存，跳过保存。")
        return pathlib.Path(dataset_dir), None

    start_time = time.time()
    num_frames = len(observations)

    dataset_dir = pathlib.Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    camera_names = [
        name
        for name in observations[0].get("images", {}).keys()
        if "_depth" not in name
    ]
    data_dict: dict[str, list] = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
    }
    if is_mobile:
        data_dict["/base_action"] = []
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    for obs, action in zip(observations, actions):
        data_dict["/observations/qpos"].append(np.array(obs["qpos"], dtype=np.float32))
        data_dict["/observations/qvel"].append(np.array(obs["qvel"], dtype=np.float32))
        data_dict["/observations/effort"].append(np.array(obs["effort"], dtype=np.float32))
        data_dict["/action"].append(np.array(action, dtype=np.float32))
        if is_mobile:
            data_dict["/base_action"].append(np.array(obs["base_vel"], dtype=np.float32))
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(obs["images"][cam_name])

    logging.info("处理图像数据...")
    for cam_name in camera_names:
        processed_images = []
        for img in data_dict[f"/observations/images/{cam_name}"]:
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            processed_images.append(img)
        data_dict[f"/observations/images/{cam_name}"] = processed_images

    if episode_idx is None:
        existing_files = list(dataset_dir.glob(f"{dataset_prefix}[0-9]*.hdf5"))
        if existing_files:
            max_idx = max(
                [
                    int(f.stem.split("_")[1])
                    for f in existing_files
                    if f.stem.split("_")[1].isdigit()
                ],
                default=-1,
            )
            episode_idx = max_idx + 1
        else:
            episode_idx = 0

    dataset_path = dataset_dir / f"{dataset_prefix}{episode_idx}.hdf5"
    logging.info(f"保存数据到: {dataset_path}")

    max_timesteps = len(data_dict["/observations/qpos"])

    compressed_len = None
    if compress_images and camera_names:
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode(".jpg", image, encode_param)
                if not result:
                    logging.error(f"图像编码失败: {cam_name}")
                    encoded_image = np.array([], dtype=np.uint8)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        logging.info(f"图像压缩耗时: {time.time() - t0:.2f}s")

        if compressed_len:
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in camera_names:
                padded_images = []
                for compressed_image in data_dict[
                    f"/observations/images/{cam_name}"
                ]:
                    padded_img = np.zeros(padded_size, dtype="uint8")
                    padded_img[: len(compressed_image)] = compressed_image
                    padded_images.append(padded_img)
                data_dict[f"/observations/images/{cam_name}"] = padded_images
            logging.info(f"图像填充耗时: {time.time() - t0:.2f}s")

    for key in data_dict:
        if len(data_dict[key]) > 0:
            data_dict[key] = np.array(data_dict[key])

    total_size = len(data_dict["/action"][0]) if len(data_dict["/action"]) > 0 else 14

    t0 = time.time()
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = compress_images

        obs = root.create_group("observations")
        if camera_names:
            image_group = obs.create_group("images")
            for cam_name in camera_names:
                if compress_images:
                    padded_size = compressed_len.max() if compressed_len is not None else 0
                    shape = (max_timesteps, padded_size)
                else:
                    shape = (max_timesteps, 480, 640, 3)
                _ = image_group.create_dataset(
                    cam_name, shape, dtype="uint8", chunks=(1, shape[1] if len(shape) > 1 else 1)
                )

        _ = obs.create_dataset("qpos", (max_timesteps, total_size), dtype=np.float32)
        _ = obs.create_dataset("qvel", (max_timesteps, total_size), dtype=np.float32)
        _ = obs.create_dataset("effort", (max_timesteps, total_size), dtype=np.float32)
        _ = root.create_dataset("action", (max_timesteps, total_size), dtype=np.float32)
        if is_mobile:
            _ = root.create_dataset("base_action", (max_timesteps, 2), dtype=np.float32)

        for name, array in data_dict.items():
            if name.startswith("/"):
                name = name[1:]
            root[name][...] = array

        if compress_images and compressed_len is not None:
            _ = root.create_dataset(
                "compress_len", (len(camera_names), max_timesteps), dtype=np.int32
            )
            root["/compress_len"][...] = compressed_len

    total_time = time.time() - start_time
    if fps and fps > 0:
        duration = num_frames / fps
        logging.info(f"hdf5 预计时长: {duration:.2f}s (fps={fps})")
    if timestamps and len(timestamps) >= 2:
        duration = timestamps[-1] - timestamps[0]
        logging.info(f"hdf5 采集时长: {duration:.2f}s (last-first)")
    logging.info(f"保存完成，耗时: {time.time() - t0:.1f}s")
    logging.info(f"hdf5 保存总耗时: {total_time:.1f}s, 帧数: {num_frames}")
    return dataset_path, compressed_len

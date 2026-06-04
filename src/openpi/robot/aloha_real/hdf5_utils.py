import logging
import pathlib
import shutil
import subprocess
import time
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np


_CAMERA_NAME_MAP = {
    "camera_top": "cam_high",
    "camera_low": "cam_low",
    "camera_wrist_right": "cam_right_wrist",
    "camera_wrist_left": "cam_left_wrist",
}


def _save_camera_name(raw_name: str) -> str:
    return _CAMERA_NAME_MAP.get(raw_name, raw_name)


class _FfmpegMp4Writer:
    def __init__(self, video_path: str | pathlib.Path, fps: float, frame_size: tuple[int, int]) -> None:
        width, height = frame_size
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
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(float(max(1, int(round(fps))))),
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
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image_bgr: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self._process.stdin.write(np.ascontiguousarray(image_bgr).tobytes())

    def release(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        returncode = self._process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode}")


def _open_mp4_writer(video_path: str | pathlib.Path, fps: float, frame_size: tuple[int, int]):
    """Create a compact MP4 writer matching the collection script's H.264 intent."""
    if shutil.which("ffmpeg") is not None:
        return _FfmpegMp4Writer(video_path, fps, frame_size)

    logging.warning("ffmpeg not found; falling back to OpenCV mp4v, which creates larger videos.")
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(round(fps)))),
        frame_size,
    )
    if writer.isOpened():
        return writer
    writer.release()
    raise RuntimeError(f"无法创建 MP4 视频写入器: {video_path}")


def _next_episode_index(dataset_dir: pathlib.Path, dataset_prefix: str) -> int:
    max_idx = -1
    for path in dataset_dir.iterdir() if dataset_dir.exists() else []:
        stem = path.stem if path.is_file() else path.name
        if not stem.startswith(dataset_prefix):
            continue
        suffix = stem[len(dataset_prefix) :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def _as_hwc_uint8(image: np.ndarray, cam_name: str, frame_idx: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise RuntimeError(f"相机 {cam_name} 第 {frame_idx} 帧图像无效: shape={image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


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
) -> Tuple[pathlib.Path, None]:
    """Save an episode using the current Aloha layout.

    Each episode is saved as ``episode_N/episode.hdf5``. Images are written as
    sidecar MP4 files in the same episode directory, and the HDF5 file stores
    video filenames under ``/observations/videos``. ``compress_images`` is kept
    for old callers but no longer writes JPEG-compressed images into HDF5.
    """
    del compress_images
    if not observations:
        logging.warning("没有数据可保存，跳过保存。")
        return pathlib.Path(dataset_dir), None
    if len(observations) != len(actions):
        raise ValueError(f"observations/actions 长度不一致: {len(observations)} vs {len(actions)}")

    start_time = time.time()
    dataset_dir = pathlib.Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if episode_idx is None:
        episode_idx = _next_episode_index(dataset_dir, dataset_prefix)
    episode_dir = dataset_dir / f"{dataset_prefix}{episode_idx}"
    dataset_path = episode_dir / "episode.hdf5"
    if episode_dir.exists() or dataset_path.exists():
        raise FileExistsError(f"Dataset already exists at {episode_dir}")
    episode_dir.mkdir(parents=True, exist_ok=False)

    first_images = observations[0].get("images", {})
    camera_map = {
        raw_name: _save_camera_name(raw_name)
        for raw_name in first_images.keys()
        if "_depth" not in raw_name
    }

    data_dict: dict[str, list] = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
    }
    if is_mobile:
        data_dict["/base_action"] = []

    video_fps = float(fps or 50.0)
    mp4_paths: dict[str, pathlib.Path] = {}
    mp4_writers: dict[str, cv2.VideoWriter] = {}
    mp4_shapes: dict[str, tuple[int, int]] = {}

    try:
        for frame_idx, (obs, action) in enumerate(zip(observations, actions)):
            frame_bundle = {}
            images = obs.get("images", {})
            for raw_name, save_name in camera_map.items():
                if raw_name not in images:
                    raise RuntimeError(f"相机 {raw_name} 第 {frame_idx} 帧缺失")
                frame_bundle[raw_name] = _as_hwc_uint8(images[raw_name], raw_name, frame_idx)

            data_dict["/observations/qpos"].append(np.asarray(obs["qpos"], dtype=np.float32))
            data_dict["/observations/qvel"].append(np.asarray(obs["qvel"], dtype=np.float32))
            data_dict["/observations/effort"].append(np.asarray(obs["effort"], dtype=np.float32))
            data_dict["/action"].append(np.asarray(action, dtype=np.float32))
            if is_mobile:
                data_dict["/base_action"].append(np.asarray(obs.get("base_vel", [0.0, 0.0]), dtype=np.float32))

            for raw_name, save_name in camera_map.items():
                image_bgr = cv2.cvtColor(frame_bundle[raw_name], cv2.COLOR_RGB2BGR)
                height, width = image_bgr.shape[:2]
                shape_hw = (height, width)
                if save_name not in mp4_writers:
                    mp4_path = episode_dir / f"{save_name}.mp4"
                    mp4_paths[save_name] = mp4_path
                    mp4_shapes[save_name] = shape_hw
                    mp4_writers[save_name] = _open_mp4_writer(mp4_path, video_fps, (width, height))
                    logging.info("视频写入 %s: %s (%dx%d @ %.0ffps)", save_name, mp4_path, width, height, video_fps)
                elif mp4_shapes[save_name] != shape_hw:
                    raise RuntimeError(
                        f"相机 {raw_name} 分辨率变化，原始 {mp4_shapes[save_name]}，当前 {shape_hw}"
                    )
                mp4_writers[save_name].write(image_bgr)
    finally:
        for writer in mp4_writers.values():
            writer.release()

    arrays = {name: np.asarray(values) for name, values in data_dict.items()}
    timesteps = len(arrays["/action"])

    logging.info("保存 episode 到: %s", episode_dir)
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        root.attrs["image_storage"] = "mp4"
        root.attrs["video_fps"] = video_fps
        root.attrs["video_frame_count"] = timesteps

        obs_group = root.create_group("observations")
        if mp4_paths:
            video_group = obs_group.create_group("videos")
            str_dtype = h5py.string_dtype(encoding="utf-8")
            for cam_name, mp4_path in mp4_paths.items():
                video_group.create_dataset(cam_name, data=mp4_path.name, dtype=str_dtype)

        for name, array in arrays.items():
            root.create_dataset(name.lstrip("/"), data=array, dtype=array.dtype)

    if fps and fps > 0:
        logging.info("hdf5 预计时长: %.2fs (fps=%.1f)", len(observations) / fps, fps)
    if timestamps and len(timestamps) >= 2:
        logging.info("hdf5 采集时长: %.2fs (last-first)", timestamps[-1] - timestamps[0])
    logging.info("保存完成: %s, 帧数=%d, 总耗时=%.1fs", dataset_path, timesteps, time.time() - start_time)
    return dataset_path, None

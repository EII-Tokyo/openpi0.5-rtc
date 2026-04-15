import base64
import io
import inspect
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def create_aloha_subtask_dataset(
    repo_id: str,
    *,
    image_size: tuple[int, int] | None = (640, 480),
    schema_image_size: tuple[int, int] | None = None,
    overwrite: bool = True,
    use_videos: bool = False,
    data_files_size_in_mb: int = 300,
    image_writer_processes: int = 0,
    image_writer_threads: int = 0,
    batch_encoding_size: int = 1,
    image_feature_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
) -> LeRobotDataset:
    if image_size is None and schema_image_size is None:
        raise ValueError("Either image_size or schema_image_size must be provided")
    width, height = schema_image_size or image_size
    all_camera_keys = (
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action"],
        },
        "train_action": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["train_action_dim0", "train_action_dim1"],
        },
        "subtask": {
            "dtype": "string",
            "shape": (1,),
            "names": ["subtask"],
        },
        "cam_high_mask": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["cam_high_mask_dim0", "cam_high_mask_dim1"],
        },
        "cam_low_mask": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["cam_low_mask_dim0", "cam_low_mask_dim1"],
        },
        "cam_left_wrist_mask": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["cam_left_wrist_mask_dim0", "cam_left_wrist_mask_dim1"],
        },
        "cam_right_wrist_mask": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["cam_right_wrist_mask_dim0", "cam_right_wrist_mask_dim1"],
        },
    }
    for camera_key in all_camera_keys:
        if camera_key in image_feature_keys:
            features[camera_key] = {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
        else:
            features[camera_key] = {
                "dtype": "uint8",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }

    dataset_path = Path(LEROBOT_HOME) / repo_id
    if overwrite and dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
    )
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "update_chunk_settings"):
        dataset.meta.update_chunk_settings(data_files_size_in_mb=data_files_size_in_mb)
    return dataset


def normalize_pil_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(size, resample=Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def load_pil_image(image: Image.Image | dict | torch.Tensor | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (torch.Tensor, np.ndarray)):
        return Image.fromarray(chw_float_to_hwc_uint8(image)).convert("RGB")
    if image.get("bytes") is not None:
        loaded = Image.open(io.BytesIO(image["bytes"]))
    else:
        loaded = Image.open(image["path"])
    return loaded.convert("RGB")


def resize_hwc_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return normalize_pil_image(Image.fromarray(np.asarray(image, dtype=np.uint8)), size)


def decode_data_url_image(data_url: str, size: tuple[int, int]) -> np.ndarray:
    _, encoded = data_url.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    return normalize_pil_image(image, size)


def normalize_hf_image(example_image: dict | Image.Image, size: tuple[int, int]) -> np.ndarray:
    return normalize_pil_image(load_pil_image(example_image), size)


def chw_float_to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return arr


def random_image_like(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def save_episode_if_needed(dataset: LeRobotDataset, frames_in_episode: int, frames_per_episode: int) -> int:
    if frames_in_episode >= frames_per_episode:
        save_episode_compat(dataset)
        return 0
    return frames_in_episode


def save_episode_compat(dataset: LeRobotDataset) -> None:
    params = inspect.signature(dataset.save_episode).parameters
    if "parallel_encoding" in params:
        dataset.save_episode(parallel_encoding=False)
        return
    dataset.save_episode()


def add_frame_compat(dataset: LeRobotDataset, frame: dict) -> None:
    params = inspect.signature(dataset.add_frame).parameters
    if "task" in params:
        payload = dict(frame)
        task = payload.pop("task", "")
        dataset.add_frame(payload, task=task)
        return
    dataset.add_frame(frame)


def push_dataset_to_hub_robust(dataset: LeRobotDataset, *, prefer_large_folder: bool = True) -> None:
    try:
        dataset.push_to_hub(upload_large_folder=prefer_large_folder)
        return
    except Exception as exc:
        message = str(exc)
        if "Failed to preupload LFS" not in message and "failed to fill whole buffer" not in message:
            raise
        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        try:
            dataset.push_to_hub(upload_large_folder=False)
        finally:
            if old_hf_transfer is None:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            else:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

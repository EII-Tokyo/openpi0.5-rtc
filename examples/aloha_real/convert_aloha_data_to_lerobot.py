"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
import gc
import sys
import tracemalloc
from pathlib import Path
import psutil
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import cv2
import torch
import tqdm
import tyro
from PIL import Image
import io

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_size_mb(obj, name="object"):
    """Get the size of an object in MB"""
    try:
        size_bytes = sys.getsizeof(obj)
        if hasattr(obj, '__dict__'):
            size_bytes += sum(sys.getsizeof(v) for v in obj.__dict__.values())
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size_bytes += sum(sys.getsizeof(item) for item in obj)
            except:
                pass
        return size_bytes / (1024 * 1024)  # Convert to MB
    except:
        return 0.0


def analyze_memory_with_tracemalloc(stage_name=""):
    """Use tracemalloc to analyze memory allocations"""
    print(f"\n📊 Tracemalloc Analysis - {stage_name}")
    
    if not tracemalloc.is_tracing():
        print("Tracemalloc not started")
        return
    
    # Get current memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory allocations:")
    for index, stat in enumerate(top_stats[:10], 1):
        size_mb = stat.size / (1024 * 1024)
        print(f"{index:2d}. {stat.traceback.format()[-1]} - {size_mb:.2f} MB")
    
    # Get total traced memory
    total_traced = sum(stat.size for stat in top_stats)
    total_traced_mb = total_traced / (1024 * 1024)
    print(f"Total traced memory: {total_traced_mb:.2f} MB")
    
    print("=" * 50)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def get_camera_image_at_frame(ep: h5py.File, camera: str, frame_idx: int) -> np.ndarray:
    """Get a single image for a specific camera at a specific frame"""
    true_cameras = [
        "camera_high",
        "camera_low", 
        "camera_wrist_left",
        "camera_wrist_right",
    ]
    
    camera_mapping = {
        "cam_high": "camera_high",
        "cam_low": "camera_low",
        "cam_left_wrist": "camera_wrist_left",
        "cam_right_wrist": "camera_wrist_right",
    }
    
    true_camera_name = camera_mapping[camera]
    camera_path = f"/observations/images/{true_camera_name}"
    
    uncompressed = ep[camera_path].ndim == 4
    
    if uncompressed:
        # Direct access to uncompressed image
        return ep[camera_path][frame_idx]
    else:
        # Decode compressed image with memory management
        
        compressed_data = ep[camera_path][frame_idx]
        
        decoded_image = Image.open(io.BytesIO(compressed_data))

        decoded_image = np.array(decoded_image)

        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
        
        # decoded_image = cv2.cvtColor(np.array(Image.open(io.BytesIO(compressed_data))), cv2.COLOR_RGB2BGR)
        # cv2.imwrite("decoded_image.jpg", decoded_image)
        # decoded_image = cv2.imdecode(compressed_data, 1)

        # del compressed_data, decoded_image
        # gc.collect()
        return decoded_image


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[h5py.File, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    ep = h5py.File(ep_path, "r")
    
    state = torch.from_numpy(np.concatenate([ep["/observations/qpos"][:, 7:], ep["/observations/qpos"][:, :7]], axis=1))
    action = torch.from_numpy(np.concatenate([ep["/action"][:, 7:], ep["/action"][:, :7]], axis=1))

    velocity = None
    if "/observations/qvel" in ep:
        velocity = torch.from_numpy(np.concatenate([ep["/observations/qvel"][:, 7:], ep["/observations/qvel"][:, :7]], axis=1))

    effort = None
    if "/observations/effort" in ep:
        effort = torch.from_numpy(np.concatenate([ep["/observations/effort"][:, 7:], ep["/observations/effort"][:, :7]], axis=1))

    return ep, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    # Start memory tracing
    tracemalloc.start()
    print("🔍 Started memory tracing with tracemalloc")
    
    process = psutil.Process()

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        print("start processing path: ", ep_path)
        
        # Analyze memory before loading episode
        analyze_memory_with_tracemalloc(f"Before Episode {ep_idx}")
        
        ep, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        cameras = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }

            # Load images one by one for this frame
            for camera in cameras:
                img = get_camera_image_at_frame(ep, camera, i)
                frame[f"observation.images.{camera}"] = img
                # Immediately delete the image to free memory
                # del img

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)
                
            # Force garbage collection every 50 frames
            if i % 50 == 0:
                gc.collect()

        dataset.save_episode()
        
        # break
        
        # Analyze memory after cleanup
        analyze_memory_with_tracemalloc(f"After Cleanup Episode {ep_idx}")
        
        # Log memory usage after each episode
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        print(f"Episode {ep_idx} done. Memory usage: {mem_gb:.2f} GB")

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # if not raw_dir.exists():
    #     if raw_repo_id is None:
    #         raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
    #     download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    # for hdf5_file in hdf5_files:
    #     print(hdf5_file)
    # exit()
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # tyro.cli(port_aloha)
    port_aloha(
        raw_dir=Path(f"../aloha-2.0/aloha_data/aloha_stationary/6.medium_full/"),
        repo_id=f"lyl472324464/remove-label-20251021",
        task="Remove the label from the bottle with the knife in the right hand.",
        push_to_hub=False,
    )
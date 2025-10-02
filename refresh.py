import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import dataclasses
from typing import Literal
from pathlib import Path
import shutil
import cv2
import numpy as np

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

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


    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    if Path('/home/eii/.cache/huggingface/lerobot/' + repo_id).exists():
        shutil.rmtree('/home/eii/.cache/huggingface/lerobot/' + repo_id)
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

datasets = [
    "twist",
    "aloha_static_battery",
    "aloha_static_candy",
    "aloha_static_coffee",
    "aloha_static_coffee_new",
    "aloha_static_cups_open",
    "aloha_static_fork_pick_up",
    "aloha_static_pingpong_test",
    "aloha_static_screw_driver",
    "aloha_static_tape",
    "aloha_static_thread_velcro",
    "aloha_static_towel",
    "aloha_static_vinh_cup",
    "aloha_static_vinh_cup_left",
    "aloha_static_ziploc_slide",
]

for dataset_name in datasets:
    repo_id = f"lyl472324464/{dataset_name}"

    new_repo_id = f"lyl472324464/{dataset_name}_new"

    print(f"Processing {dataset_name}...")   

    # 1) Load from the Hub (cached locally)
    dataset = LeRobotDataset(repo_id)

    has_velocity = "observation.velocity" in dataset[0]
    has_effort = "observation.effort" in dataset[0]

    new_dataset = create_empty_dataset(new_repo_id, "aloha", "image", has_velocity=has_velocity, has_effort=has_effort)

    # 2) Random access by index
    last_eposide_index = 0
    for i in range(len(dataset)):
        data = dataset[i]
        new_data = {
            "observation.state": data["observation.state"],
            "action": data["action"],
            "task": data["task"],
        }
        if has_velocity:
            new_data["observation.velocity"] = data["observation.velocity"]
        if has_effort:
            new_data["observation.effort"] = data["observation.effort"]
        new_data["observation.images.cam_high"] = cv2.resize(cv2.cvtColor(np.transpose(np.array(data["observation.images.cam_high"]), (1, 2, 0)), cv2.COLOR_RGB2BGR), (640, 480))        
        new_data["observation.images.cam_low"] = cv2.resize(cv2.cvtColor(np.transpose(np.array(data["observation.images.cam_low"]), (1, 2, 0)), cv2.COLOR_RGB2BGR), (640, 480))
        new_data["observation.images.cam_left_wrist"] = cv2.resize(cv2.cvtColor(np.transpose(np.array(data["observation.images.cam_left_wrist"]), (1, 2, 0)), cv2.COLOR_RGB2BGR), (640, 480))
        new_data["observation.images.cam_right_wrist"] = cv2.resize(cv2.cvtColor(np.transpose(np.array(data["observation.images.cam_right_wrist"]), (1, 2, 0)), cv2.COLOR_RGB2BGR), (640, 480))
        if data["episode_index"] != last_eposide_index :
            new_dataset.save_episode()
            last_eposide_index = data["episode_index"]
        if i == len(dataset) - 1:
            new_dataset.add_frame(new_data)
            new_dataset.save_episode()
        else:
            new_dataset.add_frame(new_data)
    # new_dataset.push_to_hub()
    # shutil.rmtree(new_repo_id)
    # shutil.rmtree(repo_id)
    # shutil.rmtree(f"~/.cache/huggingface/lerobot/{new_repo_id}")
    shutil.rmtree(f"/home/eii/.cache/huggingface/lerobot/{repo_id}")
    shutil.move(f"/home/eii/.cache/huggingface/lerobot/{new_repo_id}", f"/home/eii/.cache/huggingface/lerobot/{repo_id}")
    shutil.rmtree(f"/home/eii/.cache/huggingface/datasets")
    # shutil.rmtree(f"~/.cache/huggingface/lerobot/{new_repo_id}_new")
    # shutil.rmtree(f"~/.cache/huggingface/lerobot/{repo_id}_new")
    # shutil.rmtree(f"~/.cache/huggingface/lerobot/{new_repo_id}_new")
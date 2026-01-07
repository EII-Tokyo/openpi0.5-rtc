"""
Convert Aloha hdf5 data to LeRobot v2.0 format (compatible with lerobot==0.3.2)

特性:
1. 自动检测 image_writer 并发参数（尽量快 & 不爆内存）
2. 断点续跑：使用 _progress.json 记录已完成的 episode
3. 单个 episode 失败不会中断整个任务
"""

import dataclasses
import gc
import json
from pathlib import Path
from typing import Literal, Optional, List

import psutil
import shutil
import io

import h5py
import numpy as np
import cv2
import torch
import tqdm
from PIL import Image

from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME as LEROBOT_HOME,
    LeRobotDataset,
)


# ========================
# 并发参数自动检测
# ========================

def auto_detect_parallelism() -> tuple[int, int]:
    """
    根据当前机器的内存 & CPU 自动选择:
    - image_writer_processes
    - image_writer_threads
    """
    print("\n🔍 Auto-parallel tuning")
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024 ** 3
    avail_gb = mem.available / 1024 ** 3
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

    print(f"🧠 Total RAM: {total_gb:.1f} GB")
    print(f"💾 Available: {avail_gb:.1f} GB")
    print(f"⚙️ CPU cores: {cpu_cores}")

    # 粗估：每帧 480x640x3，4 个 camera，排队长度 8
    frame_bytes = 480 * 640 * 3
    per_frame_mb = frame_bytes * 4 / (1024 * 1024)
    queue_len = 8
    per_proc_mb = per_frame_mb * queue_len

    if per_proc_mb <= 0:
        max_proc_by_mem = 1
    else:
        usable_mb = avail_gb * 1024 * 0.5
        max_proc_by_mem = max(1, int(usable_mb / per_proc_mb))

    procs = min(cpu_cores, max_proc_by_mem) - 2
    procs = max(1, procs)
    threads = 2

    print(f"→ image_writer_processes = {procs}")
    print(f"→ image_writer_threads   = {threads}\n")

    return procs, threads


# ========================
# Dataset 配置
# ========================

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: Optional[int] = None
    image_writer_threads: Optional[int] = None
    video_backend: Optional[str] = None
    batch_encoding_size: int = 1


_auto_procs, _auto_threads = auto_detect_parallelism()
DEFAULT_DATASET_CONFIG = DatasetConfig(
    image_writer_processes=_auto_procs,
    image_writer_threads=_auto_threads,
)


# ========================
# 构造 features（schema）
# ========================

MOTORS = [
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

CAMERAS = [
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
]


def has_velocity(hdf5_files: List[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "qvel" in ep["observations"]


def has_effort(hdf5_files: List[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "effort" in ep["observations"]


def build_features(
    hdf5_files: List[Path],
    mode: Literal["video", "image"],
) -> dict:
    """
    根据 Aloha 数据结构构造 LeRobot 的 features dict。
    """
    feats: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }

    if has_velocity(hdf5_files):
        feats["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        }

    if has_effort(hdf5_files):
        feats["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        }

    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": mode,  # "image" or "video"
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return feats


# ========================
# HDF5 读取工具
# ========================

CAMERA_MAPPING = {
    "cam_high": "camera_high",
    "cam_low": "camera_low",
    "cam_left_wrist": "camera_wrist_left",
    "cam_right_wrist": "camera_wrist_right",
}


def get_camera_image_at_frame(ep: h5py.File, camera: str, frame_idx: int) -> np.ndarray:
    """
    从 episode 文件中读取指定 camera 的第 frame_idx 帧图像。
    返回形状 (H, W, 3)，uint8，BGR（OpenCV 风格）。
    """
    true_name = CAMERA_MAPPING[camera]
    camera_path = f"observations/images/{true_name}"
    ds = ep[camera_path]

    # 未压缩: (T, H, W, C)
    if ds.ndim == 4:
        img = ds[frame_idx].astype(np.uint8)
        return img

    # 压缩: 每帧是一串 bytes（存成 1D 数组）
    compressed = ds[frame_idx]
    if isinstance(compressed, np.ndarray):
        compressed = compressed.tobytes()

    with io.BytesIO(compressed) as buff:
        pil_img = Image.open(buff)
        pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return bgr


def load_episode(
    ep_path: Path,
):
    """
    打开一个 episode，返回:
    - ep: h5py.File
    - state: (T, 14) torch.float32
    - action: (T, 14) torch.float32
    - velocity: (T, 14) or None
    - effort: (T, 14) or None
    """
    ep = h5py.File(ep_path, "r")

    qpos = ep["observations"]["qpos"][()]  # (T, 14)
    act = ep["action"][()]                 # (T, 14)

    def reorder(x: np.ndarray) -> np.ndarray:
        # 把右臂 7 维放前面，左臂 7 维放后面
        return np.concatenate([x[:, 7:], x[:, :7]], axis=1)

    state_np = reorder(qpos)
    action_np = reorder(act)

    state = torch.from_numpy(state_np.astype(np.float32))
    action = torch.from_numpy(action_np.astype(np.float32))

    velocity = None
    if "qvel" in ep["observations"]:
        qvel = ep["observations"]["qvel"][()]
        vel_np = reorder(qvel)
        velocity = torch.from_numpy(vel_np.astype(np.float32))

    effort = None
    if "effort" in ep["observations"]:
        qeff = ep["observations"]["effort"][()]
        eff_np = reorder(qeff)
        effort = torch.from_numpy(eff_np.astype(np.float32))

    return ep, state, action, velocity, effort


# ========================
# 创建 / 恢复 Dataset（不使用 local_only / overwrite）
# ========================

def safe_open_dataset(repo_id: str,
                      features: dict,
                      fps: int,
                      robot_type: str,
                      dataset_config: DatasetConfig):

    repo_dir = LEROBOT_HOME / repo_id
    progress_file = repo_dir / "_progress.json"

    # --- Case 1: brand new dataset ---
    if not repo_dir.exists():
        print("🆕 Creating new dataset directory")
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=robot_type,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
        )

    # --- Case 2: folder exists but no progress file → refuse ---
    if not progress_file.exists():
        raise RuntimeError("Dataset exists but _progress.json missing → cannot resume safely")

    # --- Case 3: RESUME ---
    print("🔁 RESUME mode → loading dataset instead of creating")
    return LeRobotDataset(repo_id)


# ========================
# 主转换逻辑 + 断点续跑
# ========================

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: List[Path],
    task: str,
    repo_id: str,
    episodes: Optional[List[int]] = None,
) -> LeRobotDataset:
    """
    遍历 HDF5 文件，填充 LeRobotDataset。
    使用 _progress.json 记录已完成的 episode，支持断点续跑。
    """
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    dataset_root = LEROBOT_HOME / repo_id
    progress_path = dataset_root / "_progress.json"

    finished: set[int] = set()
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text())
            finished = set(data.get("finished", []))
            print(f"🔁 Resume: already finished episodes: {sorted(finished)}")
        except Exception as e:
            print(f"⚠️ Failed to read progress file: {e}")

    process = psutil.Process()
    prev_mem_gb = process.memory_info().rss / (1024 ** 3)

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes"):
        if ep_idx in finished:
            print(f"⏭  Skip episode {ep_idx} (already done)")
            continue

        ep_path = hdf5_files[ep_idx]
        print(f"\n▶️  Start processing episode {ep_idx}: {ep_path}")

        try:
            ep, state, action, velocity, effort = load_episode(ep_path)
        except Exception as e:
            print(f"❌ Failed to load episode {ep_idx}: {e}")
            continue

        num_frames = state.shape[0]
        cams = CAMERAS

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            # 读取 4 个相机图像
            ok = True
            for cam in cams:
                try:
                    img = get_camera_image_at_frame(ep, cam, i)
                except Exception as e:
                    print(f"⚠️  Episode {ep_idx} frame {i} camera {cam} failed: {e}")
                    ok = False
                    break
                frame[f"observation.images.{cam}"] = img

            if not ok:
                # 这一帧有问题，直接跳过（不写入 dataset）
                continue

            dataset.add_frame(frame, task)
            del frame

            if i % 50 == 0:
                gc.collect()

        # 完成一个 episode
        try:
            dataset.save_episode()
        except Exception as e:
            print(f"❌ save_episode failed for episode {ep_idx}: {e}")
        finally:
            ep.close()
            del ep, state, action
            if velocity is not None:
                del velocity
            if effort is not None:
                del effort
            gc.collect()

        # 更新进度
        finished.add(ep_idx)
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({"finished": sorted(finished)}))
        except Exception as e:
            print(f"⚠️ Failed to write progress file: {e}")

        # 打印当前内存情况
        mem_gb = process.memory_info().rss / (1024 ** 3)
        delta = mem_gb - prev_mem_gb
        print(f"✅ Episode {ep_idx} done. Memory usage: {mem_gb:.2f} GB (Δ {delta:+.2f} GB)")
        prev_mem_gb = mem_gb

    return dataset


# ========================
# 顶层封装
# ========================

def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: Optional[str] = None,
    task: str = "DEBUG",
    *,
    episodes: Optional[List[int]] = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = Path(raw_dir)
    print(f"\n📂 Raw dir: {raw_dir}")
    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {raw_dir}")
    print(f"🧾 Episodes found: {len(hdf5_files)}")

    features = build_features(hdf5_files, mode)
    dataset = safe_open_dataset(
        repo_id=repo_id,
        features=features,
        fps=50,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        dataset_config=dataset_config,
    )

    dataset = populate_dataset(
        dataset=dataset,
        hdf5_files=hdf5_files,
        task=task,
        repo_id=repo_id,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    port_aloha(
        raw_dir=Path("/home/eii/aloha-2.0/aloha_data/cut_data/merged_twist_two"),
        repo_id="lyl472324464/twist_two_202511",
        task="Twist off the bottle cap.",
        push_to_hub=False,
        mode="image",
    )

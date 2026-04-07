"""
Converts raw DROID failure episodes from GCS to LeRobot format.

Pipeline per episode:
  1. Download metadata JSON (tiny) to get trajectory_length + uuid
  2. Filter: skip if trajectory_length < MIN_FRAMES (< 5s at 15fps)
  3. Download trajectory.h5 to check joint movement
  4. Filter: skip if joints barely move (all joint velocity stds < STATIC_THRESHOLD)
  5. Find language annotation from local droid_language_annotations.json by looking
     up the chronologically next success episode by the same lab+user_id
  6. Download 3 MP4s to a temp dir, write frames to LeRobot dataset, clean up

Usage:
  uv run examples/droid/convert_droid_failures_to_lerobot.py \\
    --repo_id jnogga/droid_failure_v2 \\
    --annotations_path /path/to/droid_language_annotations.json \\
    --max_episodes 10   # for testing; omit for full run

Requirements:
  - gcloud authenticated (for gsutil)
  - HF_TOKEN set in environment (for push_to_hub)
"""

import copy
import glob
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

GCS_BASE = "gs://gresearch/robotics/droid_raw/1.0.1"
FPS = 15
MIN_FRAMES = 75  # 5 seconds at 15fps
STATIC_THRESHOLD = 1e-4  # if ALL joint velocity stds are below this, episode is static


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def parse_timestamp(ts: str) -> datetime:
    """Parse '2023-07-07-09h-45m-39s' into a datetime."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d{2})h-(\d{2})m-(\d{2})s", ts)
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts}")
    date_str, h, mi, s = m.groups()
    return datetime.strptime(f"{date_str} {h}:{mi}:{s}", "%Y-%m-%d %H:%M:%S")


def build_annotation_index(annotations_path: str) -> dict:
    """
    Returns a dict: (lab, user_id) -> sorted list of (datetime, language_instruction1).
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    index = defaultdict(list)
    for key, value in annotations.items():
        parts = key.split("+")
        if len(parts) != 3:
            continue
        lab, user_id, ts = parts
        try:
            dt = parse_timestamp(ts)
        except ValueError:
            continue
        instruction = value.get("language_instruction1", "")
        index[(lab, user_id)].append((dt, instruction))

    # Sort each list by datetime
    for k in index:
        index[k].sort(key=lambda x: x[0])

    return dict(index)


def find_next_annotation(failure_uuid: str, annotation_index: dict) -> str:
    """
    Given a failure episode UUID like 'AUTOLab+5d05c5aa+2023-07-07-09h-45m-39s',
    find the language instruction of the chronologically next success episode
    by the same lab+user_id. Returns "" if not found.
    """
    parts = failure_uuid.split("+")
    if len(parts) != 3:
        return ""
    lab, user_id, ts = parts
    try:
        failure_dt = parse_timestamp(ts)
    except ValueError:
        return ""

    candidates = annotation_index.get((lab, user_id), [])
    for dt, instruction in candidates:  # already sorted ascending
        if dt > failure_dt and (dt - failure_dt).total_seconds() <= 300:
            return instruction
    return ""


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def gsutil_ls(gcs_path: str) -> list[str]:
    result = subprocess.run(
        ["gsutil", "ls", gcs_path],
        capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.strip().endswith(":")]


def gsutil_cp(gcs_path: str, local_path: str) -> None:
    subprocess.run(
        ["gsutil", "-q", "cp", gcs_path, local_path],
        check=True,
    )


def gsutil_cp_dir(gcs_dir: str, local_dir: str) -> None:
    """Copy all files in a GCS directory to local_dir (non-recursive)."""
    subprocess.run(
        ["gsutil", "-q", "-m", "cp", f"{gcs_dir}*", local_dir],
        check=True,
    )


# ---------------------------------------------------------------------------
# Idle detection (per-frame)
# ---------------------------------------------------------------------------

def compute_idle_flags(joint_velocities: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array of shape (T,) where True means the frame is idle.
    Logic: frame t is idle if |v[t] - v[t-1]| < 1e-3 for ALL joints.
    First frame is never idle.
    """
    idle = np.zeros(len(joint_velocities), dtype=bool)
    if len(joint_velocities) < 2:
        return idle
    diffs = np.abs(joint_velocities[1:] - joint_velocities[:-1])
    idle[1:] = np.all(diffs < 1e-3, axis=1)
    return idle


# ---------------------------------------------------------------------------
# DROID data loading (copied from convert_droid_data_to_lerobot.py)
# ---------------------------------------------------------------------------

camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    return camera_type_to_string_dict[camera_type_dict[cam_id]]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        self.serial_number = serial_number
        self._index = 0
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError(f"Corrupted MP4 File: {filepath}")

    def set_reading_parameters(self, image=True, concatenate_images=False, resolution=(0, 0), resize_func=None):
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image

    def read_camera(self, ignore_data=False, correct_timestamp=None):
        if self.skip_reading:
            return {}
        success, frame = self._mp4_reader.read()
        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None
        data_dict = {}
        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: frame}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": frame[:, :single_width, :],
                self.serial_number + "_right": frame[:, single_width:, :],
            }
        return data_dict

    def set_frame_index(self, index):
        if self.skip_reading:
            return
        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index
        while self._index < index:
            self.read_camera(ignore_data=True)

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):
        self.camera_kwargs = camera_kwargs
        mp4_filepaths = glob.glob(str(recording_folderpath) + "/*.mp4")
        self.camera_dict = {}
        for f in mp4_filepaths:
            serial_number = Path(f).stem
            if f.endswith(".mp4"):
                self.camera_dict[serial_number] = MP4Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):
        full_obs_dict = defaultdict(dict)
        for cam_id in list(self.camera_dict.keys()):
            if "stereo" in cam_id:
                continue
            cam_type = camera_type_dict.get(cam_id)
            if cam_type is None:
                continue
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)
            data_dict = self.camera_dict[cam_id].read_camera()
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=[]):
    length = None
    for key in hdf5_file:
        if key in keys_to_ignore:
            continue
        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError
        if length is None:
            length = curr_length
        assert curr_length == length, f"Length mismatch: {curr_length} vs {length}"
    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):
    data_dict = {}
    for key in hdf5_file:
        if key in keys_to_ignore:
            continue
        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError
    return data_dict


def load_trajectory(filepath, recording_folderpath):
    """Load all timesteps from an HDF5 trajectory + MP4 cameras."""
    hdf5_file = h5py.File(filepath, "r")
    length = get_hdf5_length(hdf5_file)
    camera_reader = RecordedMultiCameraWrapper(recording_folderpath)

    timestep_list = []
    for i in range(length):
        keys_to_ignore = ["videos"]
        timestep = load_hdf5_to_dict(hdf5_file, i, keys_to_ignore=keys_to_ignore)
        camera_type_dict_i = {
            k: camera_type_to_string_dict[v]
            for k, v in timestep["observation"]["camera_type"].items()
        }
        camera_obs = camera_reader.read_cameras(
            index=i,
            camera_type_dict=camera_type_dict_i,
            timestamp_dict=timestep["observation"]["timestamp"]["cameras"],
        )
        if camera_obs is None:
            break
        timestep["observation"].update(camera_obs)
        timestep_list.append(timestep)

    hdf5_file.close()
    return timestep_list


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    repo_id: str,
    annotations_path: str = "droid_language_annotations.json",
    max_episodes: int = -1,  # -1 = no limit; set small for testing
    push_to_hub: bool = False,
    resume: bool = False,  # if True, skip episodes already in the dataset
    episode_list_path: str = "failure_episode_paths.txt",  # cache enumerated GCS paths
):
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists() and not resume:
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Build annotation index
    print("Building annotation index...")
    annotation_index = build_annotation_index(annotations_path)
    print(f"  {sum(len(v) for v in annotation_index.values())} annotated episodes indexed "
          f"across {len(annotation_index)} (lab, user_id) pairs")

    # Create or resume LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
            "is_idle": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["is_idle"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    # Enumerate all failure episode GCS paths.
    # gsutil ls .../*/failure/ lists the *contents* of each failure/ dir (date folders),
    # so one more level gets us the actual episode dirs.
    if Path(episode_list_path).exists():
        print(f"Loading episode list from {episode_list_path}...")
        all_episode_gcs_paths = Path(episode_list_path).read_text().splitlines()
    else:
        print("Enumerating failure episodes from GCS (this may take ~30s)...")
        date_dirs = gsutil_ls(f"{GCS_BASE}/*/failure/")  # ~485 date dirs across all labs
        all_episode_gcs_paths = []
        for date_dir in tqdm(date_dirs, desc="Enumerating dates"):
            episode_dirs = [p for p in gsutil_ls(date_dir) if p.endswith("/")]
            all_episode_gcs_paths.extend(episode_dirs)
        Path(episode_list_path).write_text("\n".join(all_episode_gcs_paths))
        print(f"Saved episode list to {episode_list_path}")

    print(f"Found {len(all_episode_gcs_paths)} total failure episodes")

    if max_episodes > 0:
        all_episode_gcs_paths = all_episode_gcs_paths[:max_episodes]
        print(f"Limiting to {max_episodes} episodes for testing")

    # Stats
    stats = {"total": len(all_episode_gcs_paths), "too_short": 0, "static": 0, "converted": 0, "errors": 0}

    for episode_gcs_path in tqdm(all_episode_gcs_paths, desc="Processing episodes"):
        episode_gcs_path = episode_gcs_path.rstrip("/") + "/"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            try:
                # --- Step 1: Download metadata JSON ---
                meta_files = gsutil_ls(f"{episode_gcs_path}metadata_*.json")
                if not meta_files:
                    print(f"  [WARN] No metadata JSON in {episode_gcs_path}, skipping")
                    stats["errors"] += 1
                    continue
                meta_gcs = meta_files[0]
                meta_local = tmpdir / "metadata.json"
                gsutil_cp(meta_gcs, str(meta_local))

                with open(meta_local) as f:
                    meta = json.load(f)

                uuid = meta["uuid"]
                traj_length = meta.get("trajectory_length", 0)

                # --- Step 2: Filter too-short ---
                if traj_length < MIN_FRAMES:
                    stats["too_short"] += 1
                    continue

                # --- Step 3: Download trajectory.h5 ---
                h5_local = tmpdir / "trajectory.h5"
                gsutil_cp(f"{episode_gcs_path}trajectory.h5", str(h5_local))

                # --- Step 4: Filter static episodes ---
                with h5py.File(h5_local, "r") as f:
                    joint_velocities = f["action"]["joint_velocity"][:]  # shape (T, 7)

                joint_vel_stds = np.std(joint_velocities, axis=0)
                if np.all(joint_vel_stds < STATIC_THRESHOLD):
                    stats["static"] += 1
                    continue

                # --- Step 5: Find annotation ---
                task = find_next_annotation(uuid, annotation_index)
                print(f"  [TASK] {uuid} -> {task!r}")

                # --- Step 6: Download MP4s ---
                mp4_local_dir = tmpdir / "MP4"
                mp4_local_dir.mkdir()
                gsutil_cp_dir(f"{episode_gcs_path}recordings/MP4/", str(mp4_local_dir) + "/")

                # --- Step 7: Load trajectory + write to dataset ---
                idle_flags = compute_idle_flags(joint_velocities)
                trajectory = load_trajectory(str(h5_local), str(mp4_local_dir))

                if len(trajectory) == 0:
                    print(f"  [WARN] Empty trajectory for {uuid}, skipping")
                    stats["errors"] += 1
                    continue

                for i, step in enumerate(trajectory):
                    cam_type_dict = step["observation"]["camera_type"]
                    wrist_ids = [k for k, v in cam_type_dict.items() if v == 0]
                    exterior_ids = [k for k, v in cam_type_dict.items() if v != 0]

                    if len(wrist_ids) == 0 or len(exterior_ids) < 2:
                        # Degenerate episode — skip it
                        break

                    dataset.add_frame({
                        "exterior_image_1_left": resize_image(
                            step["observation"]["image"][exterior_ids[0]][..., ::-1], (320, 180)
                        ),
                        "exterior_image_2_left": resize_image(
                            step["observation"]["image"][exterior_ids[1]][..., ::-1], (320, 180)
                        ),
                        "wrist_image_left": resize_image(
                            step["observation"]["image"][wrist_ids[0]][..., ::-1], (320, 180)
                        ),
                        "joint_position": np.asarray(
                            step["observation"]["robot_state"]["joint_positions"], dtype=np.float32
                        ),
                        "gripper_position": np.asarray(
                            step["observation"]["robot_state"]["gripper_position"][None], dtype=np.float32
                        ),
                        "actions": np.concatenate([
                            step["action"]["joint_velocity"],
                            step["action"]["gripper_position"][None],
                        ], dtype=np.float32),
                        "is_idle": np.array([float(idle_flags[i])], dtype=np.float32),
                        "task": task,
                    })
                else:
                    dataset.save_episode()
                    stats["converted"] += 1
                    continue

                # If we broke out of the loop (degenerate episode)
                stats["errors"] += 1

            except Exception as e:
                print(f"  [ERROR] {episode_gcs_path}: {e}")
                stats["errors"] += 1

    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print(f"  Total episodes:   {stats['total']}")
    print(f"  Too short (<5s):  {stats['too_short']}")
    print(f"  Static joints:    {stats['static']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Converted:        {stats['converted']}")
    print("=" * 60)

    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {repo_id}")
        dataset.push_to_hub(
            tags=["droid", "panda", "failure"],
            private=False,
            push_videos=True,
            license="cc-by-4.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

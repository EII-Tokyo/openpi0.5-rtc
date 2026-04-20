import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_dataset_build_utils import add_frame_compat
from lerobot_dataset_build_utils import create_aloha_subtask_dataset
from lerobot_dataset_build_utils import normalize_hf_image
from lerobot_dataset_build_utils import push_dataset_to_hub_robust
from lerobot_dataset_build_utils import save_episode_compat


def _log(message: str) -> None:
    print(message, flush=True)


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _as_array(value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _load_task_map(source: LeRobotDataset) -> dict[int, str]:
    tasks_path = Path(source.root) / "meta" / "tasks.parquet"
    table = pq.read_table(tasks_path)
    rows = table.to_pylist()
    mapping: dict[int, str] = {}
    for row in rows:
        task_index = int(row["task_index"])
        task_text = str(row["__index_level_0__"]).strip()
        mapping[task_index] = task_text
    return mapping


@dataclasses.dataclass(frozen=True)
class Args:
    source_repo_id: str = "lyl472324464/twist_subset_balanced_100k_448_multi_repo"
    repo_id: str = "lyl472324464/twist_subset_balanced_100k_448_multi_repo_300mb"
    image_size: tuple[int, int] = (448, 448)
    schema_image_size: tuple[int, int] | None = None
    overwrite: bool = True
    push_to_hub: bool = True
    frames_per_episode: int = 375
    data_files_size_in_mb: int = 300
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    batch_encoding_size: int = 1
    log_every: int = 1000


def main(args: Args) -> None:
    _log(
        f"[main] source_repo_id={args.source_repo_id} repo_id={args.repo_id} "
        f"frames_per_episode={args.frames_per_episode} data_files_size_in_mb={args.data_files_size_in_mb}"
    )
    _log("[main] loading source dataset")
    source = LeRobotDataset(args.source_repo_id, force_cache_sync=False, download_videos=False)
    _log(f"[main] loaded source dataset rows={len(source.hf_dataset)}")
    task_map = _load_task_map(source)
    _log(f"[main] loaded task map entries={len(task_map)}")

    dataset = create_aloha_subtask_dataset(
        args.repo_id,
        image_size=args.image_size,
        schema_image_size=args.schema_image_size or args.image_size,
        overwrite=args.overwrite,
        use_videos=False,
        data_files_size_in_mb=args.data_files_size_in_mb,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    current_episode = None
    frames_in_episode = 0
    total_rows = 0

    for row in source.hf_dataset:
        episode_index = int(np.asarray(row["episode_index"]).item())
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            if frames_in_episode > 0:
                save_episode_compat(dataset)
                frames_in_episode = 0
            current_episode = episode_index

        if total_rows < 5 or total_rows % args.log_every == 0:
            _log(
                f"[row] idx={total_rows} episode_index={episode_index} "
                f"source_index={int(np.asarray(row['index']).item())}"
            )

        frame = {
            "observation.state": _as_array(row["observation.state"], np.float32),
            "action": _as_array(row["action"], np.float32),
            "train_action": _as_array(row.get("train_action", [[0]]), np.int64, (1, 1)),
            "task": task_map[int(np.asarray(row["task_index"]).item())],
            "subtask": row["subtask"],
            "cam_high_mask": _as_array(row.get("cam_high_mask", [[1]]), np.int64, (1, 1)),
            "cam_low_mask": _as_array(row.get("cam_low_mask", [[1]]), np.int64, (1, 1)),
            "cam_left_wrist_mask": _as_array(row.get("cam_left_wrist_mask", [[1]]), np.int64, (1, 1)),
            "cam_right_wrist_mask": _as_array(row.get("cam_right_wrist_mask", [[1]]), np.int64, (1, 1)),
        }
        for camera_key in CAMERA_KEYS:
            frame[camera_key] = normalize_hf_image(row[camera_key], args.image_size)

        add_frame_compat(dataset, frame)
        frames_in_episode += 1
        total_rows += 1
        if frames_in_episode >= args.frames_per_episode:
            save_episode_compat(dataset)
            frames_in_episode = 0

    if frames_in_episode > 0:
        save_episode_compat(dataset)

    _log("[main] finalizing dataset")
    dataset.finalize()

    _log("[main] removing local images cache")
    shutil.rmtree(Path(dataset.root) / "images", ignore_errors=True)

    if args.push_to_hub:
        _log("[main] pushing to hub")
        push_dataset_to_hub_robust(dataset, prefer_large_folder=True)
    _log("[main] done")


if __name__ == "__main__":
    main(tyro.cli(Args))

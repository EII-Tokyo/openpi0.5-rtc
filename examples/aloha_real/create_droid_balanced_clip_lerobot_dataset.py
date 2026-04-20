from __future__ import annotations

import dataclasses
import gc
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import datasets
import pyarrow.parquet as pq
import tyro
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
import lerobot.datasets.utils as lerobot_dataset_utils
from PIL import Image

from lerobot_dataset_build_utils import add_frame_compat
from lerobot_dataset_build_utils import load_pil_image
from lerobot_dataset_build_utils import push_dataset_to_hub_robust
from lerobot_dataset_build_utils import resize_hwc_uint8
from lerobot_dataset_build_utils import save_episode_compat


IMAGE_KEY_EXTERIOR = "observation.images.exterior_1_left"
IMAGE_KEY_WRIST = "observation.images.wrist_left"
JOINT_KEY = "observation.state.joint_position"
GRIPPER_KEY = "observation.state.gripper_position"
ACTION_KEY = "action"
LANGUAGE_KEYS = ("task", "language_instruction", "language_instruction_2", "language_instruction_3")


@dataclasses.dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    start_index: int
    num_frames: int
    task: str


@dataclasses.dataclass(frozen=True)
class ClipSelection:
    episode: EpisodeMeta
    clip_length: int


@dataclasses.dataclass(frozen=True)
class ClipPlan:
    selection: ClipSelection
    start_offset: int


@dataclasses.dataclass(frozen=True)
class Args:
    source_repo_id: str = "lerobot/droid_1.0.1"
    repo_id: str = "lyl472324464/droid_balanced_clip_100k_448"
    num_frames: int = 100000
    clip_length: int = 96
    action_horizon: int = 50
    image_size: tuple[int, int] = (448, 448)
    overwrite: bool = True
    push_to_hub: bool = True
    seed: int = 0
    data_files_size_in_mb: int = 300
    stats_json: Path | None = Path("logs/droid_balanced_clip_100k_448_stats.json")
    read_episode_batch_size: int = 64


def _normalize_task(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _choose_task_from_values(values: list[object]) -> str:
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if not isinstance(value, str):
            continue
        cleaned = _normalize_task(value)
        if cleaned:
            return cleaned
    return ""


def _scan_episode_metadata(dataset: LeRobotDataset) -> list[EpisodeMeta]:
    per_episode: dict[int, dict[str, int | str]] = {}
    parquet_paths = sorted((Path(dataset.root) / "data").glob("**/*.parquet"))
    print(f"[scan] scanning {len(parquet_paths)} parquet files from {dataset.root}", flush=True)
    wanted_columns = {"episode_index", "index", "frame_index", *LANGUAGE_KEYS}
    for parquet_idx, parquet_path in enumerate(parquet_paths, start=1):
        pf = pq.ParquetFile(parquet_path)
        present = [name for name in wanted_columns if name in pf.schema.names]
        if not {"episode_index", "index"} <= set(present):
            continue
        table = pf.read(columns=present)
        columns = {name: table.column(name).to_pylist() for name in present}
        num_rows = len(columns["index"])
        for row_i in range(num_rows):
            episode_index = int(columns["episode_index"][row_i])
            global_index = int(columns["index"][row_i])
            task = _choose_task_from_values([columns[key][row_i] for key in LANGUAGE_KEYS if key in columns])
            meta = per_episode.get(episode_index)
            if meta is None:
                per_episode[episode_index] = {
                    "start_index": global_index,
                    "num_frames": 1,
                    "task": task,
                }
                continue
            meta["start_index"] = min(int(meta["start_index"]), global_index)
            meta["num_frames"] = int(meta["num_frames"]) + 1
            if task and not meta["task"]:
                meta["task"] = task
        if parquet_idx == 1 or parquet_idx % 10 == 0 or parquet_idx == len(parquet_paths):
            print(
                f"[scan] parquet {parquet_idx}/{len(parquet_paths)} -> {len(per_episode)} episodes seen",
                flush=True,
            )
    episodes = [
        EpisodeMeta(
            episode_index=episode_index,
            start_index=int(meta["start_index"]),
            num_frames=int(meta["num_frames"]),
            task=str(meta["task"]),
        )
        for episode_index, meta in per_episode.items()
        if str(meta["task"])
    ]
    episodes.sort(key=lambda item: item.episode_index)
    return episodes


def _allocate_clip_lengths(num_frames: int, clip_length: int, min_clip_length: int) -> list[int]:
    if clip_length < min_clip_length:
        raise ValueError(f"clip_length={clip_length} must be >= min_clip_length={min_clip_length}")
    lengths: list[int] = []
    remaining = num_frames
    while remaining > 0:
        take = min(clip_length, remaining)
        lengths.append(take)
        remaining -= take
    if not lengths:
        return []
    if lengths[-1] >= min_clip_length:
        return lengths
    deficit = min_clip_length - lengths[-1]
    for idx in range(len(lengths) - 2, -1, -1):
        borrowable = lengths[idx] - min_clip_length
        if borrowable <= 0:
            continue
        moved = min(borrowable, deficit)
        lengths[idx] -= moved
        lengths[-1] += moved
        deficit -= moved
        if deficit == 0:
            break
    if deficit > 0:
        raise ValueError(
            f"Unable to allocate exact num_frames={num_frames} into clips with min_clip_length={min_clip_length}. "
            f"Try increasing num_frames or clip_length."
        )
    return lengths


def _select_balanced_clips(episodes: list[EpisodeMeta], clip_lengths: list[int], rng: random.Random) -> list[ClipSelection]:
    eligible = [episode for episode in episodes if episode.num_frames >= min(clip_lengths)]
    task_to_episodes: dict[str, list[EpisodeMeta]] = defaultdict(list)
    for episode in eligible:
        task_to_episodes[episode.task].append(episode)
    for values in task_to_episodes.values():
        rng.shuffle(values)

    tasks = list(task_to_episodes)
    rng.shuffle(tasks)
    selections: list[ClipSelection] = []
    task_counts = Counter()

    for clip_length in clip_lengths:
        candidates = [
            task
            for task in tasks
            if task_to_episodes[task]
            and task_to_episodes[task][-1].num_frames >= clip_length
        ]
        if not candidates:
            raise RuntimeError(f"No episodes left that can satisfy clip_length={clip_length}.")
        min_count = min(task_counts[task] for task in candidates) if candidates else 0
        balanced_tasks = [task for task in candidates if task_counts[task] == min_count]
        task = rng.choice(balanced_tasks)
        episode = task_to_episodes[task].pop()
        selections.append(ClipSelection(episode=episode, clip_length=clip_length))
        task_counts[task] += 1
    return selections


def _load_rgb_image(value: object, image_size: tuple[int, int]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(load_pil_image(value))
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return resize_hwc_uint8(arr, image_size)


def _extract_vector(
    row: dict,
    key: str,
    *,
    fallback_keys: tuple[str, ...] = (),
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    for candidate in (key, *fallback_keys):
        if candidate in row:
            value = np.asarray(row[candidate], dtype=np.float32)
            if expected_shape is not None:
                value = value.reshape(expected_shape)
            return value
    raise KeyError(f"Missing required key {key!r} and fallbacks {fallback_keys!r}")


def _create_dataset(args: Args) -> LeRobotDataset:
    width, height = args.image_size
    features = {
        "observation.exterior_image_1_left": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.wrist_image_left": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint"],
        },
        "observation.gripper_position": {
            "dtype": "float32",
            "shape": (),
            "names": [],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["action"],
        },
        "train_action": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": ["train_action_dim0", "train_action_dim1"],
        },
        "source_task": {
            "dtype": "string",
            "shape": (),
            "names": [],
        },
        "source_episode_index": {
            "dtype": "int64",
            "shape": (),
            "names": [],
        },
        "source_frame_index": {
            "dtype": "int64",
            "shape": (),
            "names": [],
        },
    }
    dataset_path = Path(LEROBOT_HOME) / args.repo_id
    if args.overwrite and dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=15,
        robot_type="droid",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "update_chunk_settings"):
        dataset.meta.update_chunk_settings(data_files_size_in_mb=args.data_files_size_in_mb)
    return dataset


def _patch_scalar_feature_support() -> None:
    def _get_hf_features_from_features(features: dict) -> datasets.Features:
        hf_features = {}
        for key, ft in features.items():
            if ft["dtype"] == "video":
                continue
            if ft["dtype"] == "image":
                hf_features[key] = datasets.Image()
            elif ft["shape"] == ():
                hf_features[key] = datasets.Value(dtype=ft["dtype"])
            elif ft["shape"] == (1,):
                hf_features[key] = datasets.Value(dtype=ft["dtype"])
            elif len(ft["shape"]) == 1:
                hf_features[key] = datasets.Sequence(
                    length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
                )
            elif len(ft["shape"]) == 2:
                hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 3:
                hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 4:
                hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 5:
                hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
            else:
                raise ValueError(f"Corresponding feature is not valid: {ft}")
        return datasets.Features(hf_features)

    lerobot_dataset_utils.get_hf_features_from_features = _get_hf_features_from_features
    lerobot_dataset_module.get_hf_features_from_features = _get_hf_features_from_features


def _build_episode_batches(
    selected_episode_indices: list[int],
    batch_size: int,
) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch_episode_indices: list[list[int]] = []

    for start in range(0, len(selected_episode_indices), batch_size):
        episode_batch = selected_episode_indices[start : start + batch_size]
        batch_episode_indices.append(episode_batch)

    return batch_episode_indices


def _build_source_batch_root(source_repo_id: str, batch_idx: int) -> Path:
    safe_repo_id = source_repo_id.replace("/", "__")
    return Path(LEROBOT_HOME) / "_droid_batch_cache" / safe_repo_id / f"batch-{batch_idx:04d}"


def _load_episode_batch(
    source_repo_id: str,
    episode_batch: list[int],
    root: Path,
) -> tuple[LeRobotDataset, dict[int, int], dict[int, int]]:
    dataset = LeRobotDataset(
        source_repo_id,
        root=root,
        revision="main",
        force_cache_sync=True,
        download_videos=True,
        episodes=episode_batch,
    )

    ep_col = dataset.hf_dataset.data.column("episode_index").combine_chunks()
    ep_arr = ep_col.to_numpy(zero_copy_only=False)
    base_relative_index: dict[int, int] = {}
    episode_lengths: dict[int, int] = {}
    if ep_arr.size:
        unique_eps, starts, counts = np.unique(ep_arr, return_index=True, return_counts=True)
        for ep, start, count in zip(unique_eps.tolist(), starts.tolist(), counts.tolist(), strict=True):
            base_relative_index[int(ep)] = int(start)
            episode_lengths[int(ep)] = int(count)

    missing = [ep for ep in episode_batch if ep not in base_relative_index]
    if missing:
        raise RuntimeError(f"Loaded batch is missing requested episodes: {missing[:10]}")

    return dataset, base_relative_index, episode_lengths


def main(args: Args) -> None:
    rng = random.Random(args.seed)
    _patch_scalar_feature_support()
    print(f"[start] source={args.source_repo_id} target={args.repo_id}", flush=True)
    print(
        f"[start] num_frames={args.num_frames} clip_length={args.clip_length} action_horizon={args.action_horizon} "
        f"image_size={args.image_size} push_to_hub={args.push_to_hub}",
        flush=True,
    )
    source_dataset = LeRobotDataset(args.source_repo_id, revision="main", force_cache_sync=False, download_videos=False)
    print(f"[start] source root={source_dataset.root}", flush=True)
    episodes = _scan_episode_metadata(source_dataset)
    print(f"[scan] collected {len(episodes)} episodes with non-empty tasks", flush=True)
    del source_dataset
    gc.collect()
    clip_lengths = _allocate_clip_lengths(args.num_frames, args.clip_length, args.action_horizon)
    print(
        f"[sample] allocated {len(clip_lengths)} clips, min={min(clip_lengths)}, max={max(clip_lengths)}, "
        f"total_frames={sum(clip_lengths)}",
        flush=True,
    )
    selections = _select_balanced_clips(episodes, clip_lengths, rng)
    print(f"[sample] selected {len(selections)} balanced clips", flush=True)
    selected_episode_indices = sorted({selection.episode.episode_index for selection in selections})
    print(f"[sample] selected {len(selected_episode_indices)} unique source episodes", flush=True)
    batch_episode_indices = _build_episode_batches(selected_episode_indices, args.read_episode_batch_size)
    episode_to_batch = {
        episode_index: batch_idx
        for batch_idx, episode_batch in enumerate(batch_episode_indices)
        for episode_index in episode_batch
    }
    batch_plans: dict[int, list[ClipPlan]] = defaultdict(list)
    for selection in selections:
        start_offset = rng.randint(0, selection.episode.num_frames - selection.clip_length)
        batch_idx = episode_to_batch[int(selection.episode.episode_index)]
        batch_plans[batch_idx].append(ClipPlan(selection=selection, start_offset=start_offset))
    dataset = _create_dataset(args)
    print(f"[write] target root={dataset.root}", flush=True)

    true_scalar = np.asarray([[1]], dtype=np.int64)
    false_scalar = np.asarray([[0]], dtype=np.int64)
    task_counts = Counter()
    source_episode_counts = Counter()
    clips_written = 0

    for batch_idx, episode_batch in enumerate(batch_episode_indices):
        plans = batch_plans.get(batch_idx, [])
        if not plans:
            continue
        batch_root = _build_source_batch_root(args.source_repo_id, batch_idx)
        shutil.rmtree(batch_root, ignore_errors=True)
        source_dataset, base_relative_index, episode_lengths = _load_episode_batch(
            args.source_repo_id,
            episode_batch,
            batch_root,
        )
        print(
            f"[read] loaded episode batch {batch_idx + 1}/{len(batch_episode_indices)} "
            f"with {len(episode_batch)} episodes from {batch_root}",
            flush=True,
        )
        try:
            for plan in sorted(plans, key=lambda item: (item.selection.episode.episode_index, item.start_offset)):
                selection = plan.selection
                start_offset = plan.start_offset
                loaded_length = episode_lengths[int(selection.episode.episode_index)]
                # Parquet scan lengths can disagree with LeRobotDataset(episodes=[...]) row counts; trust loaded_length.
                effective_clip = min(selection.clip_length, loaded_length)
                if effective_clip < 1:
                    print(
                        f"[warn] skipping episode {selection.episode.episode_index}: loaded_length={loaded_length}",
                        flush=True,
                    )
                    continue
                if loaded_length < selection.clip_length:
                    print(
                        f"[warn] episode {selection.episode.episode_index}: scanned num_frames={selection.episode.num_frames} "
                        f"but loaded_length={loaded_length} < clip_length={selection.clip_length}; writing {effective_clip} frames",
                        flush=True,
                    )
                start_offset = min(int(start_offset), max(0, loaded_length - effective_clip))
                task_counts[selection.episode.task] += 1
                source_episode_counts[selection.episode.episode_index] += 1
                relative_start = base_relative_index[int(selection.episode.episode_index)] + start_offset
                for frame_offset in range(effective_clip):
                    row = source_dataset[int(relative_start + frame_offset)]
                    valid_action = frame_offset < (effective_clip - args.action_horizon + 1)
                    frame = {
                        "observation.exterior_image_1_left": _load_rgb_image(row[IMAGE_KEY_EXTERIOR], args.image_size),
                        "observation.wrist_image_left": _load_rgb_image(row[IMAGE_KEY_WRIST], args.image_size),
                        "observation.joint_position": _extract_vector(
                            row,
                            JOINT_KEY,
                            fallback_keys=("observation/joint_position",),
                            expected_shape=(7,),
                        ),
                        "observation.gripper_position": np.asarray(
                            _extract_vector(
                                row,
                                GRIPPER_KEY,
                                fallback_keys=("observation/gripper_position",),
                                expected_shape=(1,),
                            ).reshape(-1)[0],
                            dtype=np.float32,
                        ),
                        "action": _extract_vector(row, ACTION_KEY, expected_shape=(8,)),
                        "train_action": true_scalar.copy() if valid_action else false_scalar.copy(),
                        "source_task": selection.episode.task,
                        "source_episode_index": np.asarray(selection.episode.episode_index, dtype=np.int64),
                        "source_frame_index": np.asarray(start_offset + frame_offset, dtype=np.int64),
                        "task": selection.episode.task,
                    }
                    add_frame_compat(dataset, frame)
                save_episode_compat(dataset)
                clips_written += 1
                if clips_written % 100 == 0:
                    print(f"[clip] wrote {clips_written}/{len(selections)} clips", flush=True)
        finally:
            del source_dataset
            gc.collect()
            shutil.rmtree(batch_root, ignore_errors=True)
            print(f"[read] cleaned batch cache {batch_root}", flush=True)

    print("[finalize] finalizing dataset", flush=True)
    dataset.finalize()
    print("[finalize] removing local images cache before optional push", flush=True)
    shutil.rmtree(Path(dataset.root) / "images", ignore_errors=True)

    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(
                {
                    "source_repo_id": args.source_repo_id,
                    "repo_id": args.repo_id,
                    "num_frames": args.num_frames,
                    "clip_length": args.clip_length,
                    "action_horizon": args.action_horizon,
                    "num_clips": len(selections),
                    "num_unique_tasks": len(task_counts),
                    "task_clip_counts": dict(task_counts),
                    "source_episode_counts": {str(k): v for k, v in source_episode_counts.items()},
                },
                indent=2,
                ensure_ascii=True,
            )
        )

    if args.push_to_hub:
        print("[push] pushing to hub", flush=True)
        push_dataset_to_hub_robust(dataset, prefer_large_folder=True)


if __name__ == "__main__":
    main(tyro.cli(Args))

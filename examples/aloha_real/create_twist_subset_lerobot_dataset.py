import dataclasses
print("[top] module import start", flush=True)
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import shutil
import sys
import time

from PIL import Image
print("[top] importing LeRobotDataset", flush=True)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("[top] imported LeRobotDataset", flush=True)
print("[top] importing numpy", flush=True)
import numpy as np
print("[top] imported numpy", flush=True)
print("[top] importing tyro", flush=True)
import tyro
print("[top] imported tyro", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("[top] importing local helpers", flush=True)
from lerobot_dataset_build_utils import add_frame_compat
from lerobot_dataset_build_utils import create_aloha_subtask_dataset
from lerobot_dataset_build_utils import push_dataset_to_hub_robust
from lerobot_dataset_build_utils import resize_hwc_uint8
from lerobot_dataset_build_utils import save_episode_compat
from lerobot_dataset_build_utils import save_episode_if_needed
from language_templates import choose_answer
from language_templates import load_templates
from language_templates import shorten_target
from openpi_client.runtime.low_level_subtask_defaults import DEFAULT_STATE_SUBTASK_PAIRS
print("[top] imported all helpers", flush=True)


DEFAULT_SOURCE_REPOS = (
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
)
PAIR_KEYS = [
    f"{state}|||{subtask}"
    for state, subtask in DEFAULT_STATE_SUBTASK_PAIRS
]
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _log(message: str) -> None:
    print(message, flush=True)


@dataclasses.dataclass(frozen=True)
class Candidate:
    repo_id: str
    index: int
    episode_index: int
    descriptions: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Args:
    source_repo_ids: tuple[str, ...] = DEFAULT_SOURCE_REPOS
    repo_id: str = "lyl472324464/twist_subset_balanced_100k_224_multi_repo"
    num_samples: int = 100000
    seed: int = 0
    overwrite: bool = True
    use_videos: bool = False
    image_size: tuple[int, int] | None = (224, 224)
    schema_image_size: tuple[int, int] | None = None
    push_to_hub: bool = True
    frames_per_episode: int = 1000
    data_files_size_in_mb: int = 300
    require_exact_num_samples: bool = True
    preserve_original_images: bool = False
    stats_json: Path | None = Path("logs/twist_subset_balanced_100k_224_multi_repo_stats.json")
    video_backend: str | None = None
    read_order: str = "repo_sorted"
    log_every: int = 1000
    source_decode_batch_size: int = 32
    image_writer_processes: int = 0
    image_writer_threads: int = 8


def _normalize_descriptions(raw: object) -> tuple[str, ...]:
    vals: list[str] = []
    if isinstance(raw, str):
        cleaned = shorten_target(raw)
        if cleaned:
            vals.append(cleaned)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                cleaned = shorten_target(item)
                if cleaned:
                    vals.append(cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for val in vals:
        if val not in seen:
            seen.add(val)
            out.append(val)
    return tuple(out)


def _sample_target_prompt(templates: dict, rng: random.Random, target: str) -> str:
    variants = templates.get("target_prompt_variants") or ["What should the robot do for {target}?"]
    return rng.choice(variants).format(target=target)


def _parse_subtask(raw: object) -> tuple[str, str, tuple[str, ...]] | None:
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    state = payload.get("bottle_state")
    subtask = payload.get("subtask")
    descriptions = _normalize_descriptions(payload.get("bottle_description"))
    if not isinstance(state, str) or not isinstance(subtask, str):
        return None
    return state, subtask, descriptions


def _sample_indices(candidates: list[Candidate], n: int, rng: random.Random) -> list[Candidate]:
    if not candidates:
        return []
    if len(candidates) >= n:
        return rng.sample(candidates, n)
    return [rng.choice(candidates) for _ in range(n)]


def _tensor_to_numpy_rgb(image: object, image_size: tuple[int, int] | None, preserve_original_images: bool) -> Image.Image | np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if preserve_original_images:
        return Image.fromarray(arr, mode="RGB")
    if image_size is None:
        raise ValueError("image_size must be provided when preserve_original_images is False")
    return resize_hwc_uint8(arr, image_size)


def _build_buckets(source_repo_ids: tuple[str, ...], video_backend: str | None) -> tuple[dict[str, list[Candidate]], dict[str, LeRobotDataset]]:
    buckets = defaultdict(list)
    datasets: dict[str, LeRobotDataset] = {}
    for repo_id in source_repo_ids:
        _log(f"[build_buckets] loading {repo_id}")
        ds = LeRobotDataset(repo_id, force_cache_sync=False, download_videos=True, video_backend=video_backend)
        _log(f"[build_buckets] loaded {repo_id}: hf_rows={len(ds.hf_dataset)}")
        datasets[repo_id] = ds
        added = 0
        for row in ds.hf_dataset:
            parsed = _parse_subtask(row.get("subtask"))
            if parsed is None:
                continue
            state, subtask, descriptions = parsed
            key = f"{state}|||{subtask}"
            if key not in PAIR_KEYS:
                continue
            buckets[key].append(
                Candidate(
                    repo_id=repo_id,
                    index=int(row["index"]),
                    episode_index=int(row["episode_index"]),
                    descriptions=descriptions,
                )
            )
            added += 1
        _log(f"[build_buckets] bucketed {repo_id}: matched_rows={added}")
    return buckets, datasets


def _order_samples(sampled: list[tuple[str, Candidate]], read_order: str, rng: random.Random) -> list[tuple[str, Candidate]]:
    if read_order == "random":
        ordered = list(sampled)
        rng.shuffle(ordered)
        return ordered
    if read_order == "repo_sorted":
        ordered = list(sampled)
        ordered.sort(key=lambda item: (item[1].repo_id, item[1].episode_index, item[1].index))
        return ordered
    raise ValueError(f"Unsupported read_order={read_order!r}")


def _column_length(batch: object) -> int:
    if isinstance(batch, dict):
        first_value = next(iter(batch.values()))
        return len(first_value)
    return len(batch)


def _value_at(column: object, idx: int) -> object:
    if isinstance(column, np.ndarray):
        return column[idx]
    return column[idx]


def _ensure_frame_batch(frames: object) -> object:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 3:
            return frames[None, ...]
        return frames
    if hasattr(frames, "ndim") and getattr(frames, "ndim") == 3:
        return frames.unsqueeze(0)
    return frames


def _fetch_rows_batched(
    ds: LeRobotDataset,
    candidates: list[Candidate],
) -> list[dict[str, object]]:
    ds._ensure_hf_dataset_loaded()
    indices = [candidate.index for candidate in candidates]
    batch = ds.hf_dataset[indices]
    batch_size = _column_length(batch)
    if batch_size != len(candidates):
        raise RuntimeError(f"Expected {len(candidates)} rows, got {batch_size}")

    episode_indices = [_value_at(batch["episode_index"], i) for i in range(batch_size)]
    first_episode = int(np.asarray(episode_indices[0]).item())
    if any(int(np.asarray(ep).item()) != first_episode for ep in episode_indices):
        raise RuntimeError("Batched fetch requires all rows to come from the same source episode")

    timestamps = [float(np.asarray(_value_at(batch["timestamp"], i)).item()) for i in range(batch_size)]
    query_timestamps = {camera_key: timestamps for camera_key in CAMERA_KEYS}
    video_frames = {
        camera_key: _ensure_frame_batch(frames)
        for camera_key, frames in ds._query_videos(query_timestamps, first_episode).items()
    }

    rows: list[dict[str, object]] = []
    for i in range(batch_size):
        row = {
            "observation.state": _value_at(batch["observation.state"], i),
            "action": _value_at(batch["action"], i),
        }
        for camera_key in CAMERA_KEYS:
            row[camera_key] = _value_at(video_frames[camera_key], i)
        rows.append(row)
    return rows


def _iter_batched_samples(
    ordered_samples: list[tuple[str, Candidate]],
    source_decode_batch_size: int,
) -> list[list[tuple[str, Candidate]]]:
    batches: list[list[tuple[str, Candidate]]] = []
    for _, group in itertools.groupby(
        ordered_samples,
        key=lambda item: (item[1].repo_id, item[1].episode_index),
    ):
        episode_samples = list(group)
        for start in range(0, len(episode_samples), source_decode_batch_size):
            batches.append(episode_samples[start : start + source_decode_batch_size])
    return batches


def main(args: Args) -> None:
    wall_start = time.perf_counter()
    _log(f"[main] start repo_id={args.repo_id} num_samples={args.num_samples} video_backend={args.video_backend}")
    rng = random.Random(args.seed)
    templates = load_templates(Path("assets/short_language_templates.json"))
    _log("[main] creating output dataset")
    schema_image_size = args.schema_image_size
    if args.preserve_original_images and schema_image_size is None:
        schema_image_size = (640, 480)
    dataset = create_aloha_subtask_dataset(
        args.repo_id,
        image_size=args.image_size,
        schema_image_size=schema_image_size,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
        data_files_size_in_mb=args.data_files_size_in_mb,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    _log("[main] building buckets from source repos")
    bucket_start = time.perf_counter()
    buckets, source_datasets = _build_buckets(args.source_repo_ids, args.video_backend)
    bucket_seconds = time.perf_counter() - bucket_start
    _log("[main] finished building buckets")
    per_class = args.num_samples // len(PAIR_KEYS)
    remainder = args.num_samples % len(PAIR_KEYS)

    sampled: list[tuple[str, Candidate]] = []
    class_counts = Counter()
    source_counts = Counter()
    for i, key in enumerate(PAIR_KEYS):
        _log(f"[sampling] class={key} available={len(buckets[key])}")
        target = per_class + (1 if i < remainder else 0)
        picks = _sample_indices(buckets[key], target, rng)
        sampled.extend((key, pick) for pick in picks)
        class_counts[key] += len(picks)
        for pick in picks:
            source_counts[pick.repo_id] += 1
    sample_plan_seconds = time.perf_counter() - wall_start - bucket_seconds
    order_start = time.perf_counter()
    ordered_samples = _order_samples(sampled, args.read_order, rng)
    batched_samples = _iter_batched_samples(ordered_samples, args.source_decode_batch_size)
    order_seconds = time.perf_counter() - order_start
    _log(f"[sampling] total_sampled={len(sampled)}")
    if args.require_exact_num_samples and len(sampled) != args.num_samples:
        missing = args.num_samples - len(sampled)
        raise RuntimeError(
            f"Requested {args.num_samples} samples for {args.repo_id}, but only sampled {len(sampled)} "
            f"(missing {missing}). At least one state/subtask bucket is empty or underfilled."
        )

    false_scalar = np.asarray([[0]], dtype=np.int64)
    true_scalar = np.asarray([[1]], dtype=np.int64)
    frames_in_episode = 0
    timing = {
        "bucket_build_seconds": bucket_seconds,
        "sample_plan_seconds": sample_plan_seconds,
        "order_seconds": order_seconds,
        "fetch_row_seconds": 0.0,
        "build_frame_seconds": 0.0,
        "image_convert_seconds": 0.0,
        "add_frame_seconds": 0.0,
        "save_episode_seconds": 0.0,
    }

    sample_idx = 0
    for batch in batched_samples:
        if not batch:
            continue
        fetch_start = time.perf_counter()
        rows = _fetch_rows_batched(
            source_datasets[batch[0][1].repo_id],
            [candidate for _, candidate in batch],
        )
        timing["fetch_row_seconds"] += time.perf_counter() - fetch_start
        for (key, candidate), row in zip(batch, rows, strict=True):
            if sample_idx < 5 or sample_idx % args.log_every == 0:
                _log(
                    f"[sample] idx={sample_idx} key={key} repo={candidate.repo_id} "
                    f"source_index={candidate.index}"
                )
            frame_start = time.perf_counter()
            state, subtask = key.split("|||", 1)
            target = rng.choice(candidate.descriptions) if candidate.descriptions else shorten_target(state)
            question = _sample_target_prompt(templates, rng, target)
            answer = choose_answer(templates, rng, state, subtask)

            state_vec = np.asarray(row["observation.state"], dtype=np.float32)
            action_vec = np.asarray(row["action"], dtype=np.float32)
            frame = {
                "observation.state": state_vec,
                "action": action_vec,
                # This dataset is built from randomly sampled frames, so the action chunk is not
                # temporally valid for action-expert training.
                "train_action": false_scalar.copy(),
                "task": question,
                "subtask": json.dumps(
                    {
                        "bottle_state": state,
                        "subtask": subtask,
                        "answer": answer,
                        "bottle_description": list(candidate.descriptions),
                    },
                    ensure_ascii=True,
                ),
                "cam_high_mask": true_scalar.copy(),
                "cam_low_mask": true_scalar.copy(),
                "cam_left_wrist_mask": true_scalar.copy(),
                "cam_right_wrist_mask": true_scalar.copy(),
            }
            image_convert_start = time.perf_counter()
            for camera_key in CAMERA_KEYS:
                frame[camera_key] = _tensor_to_numpy_rgb(
                    row[camera_key],
                    args.image_size,
                    args.preserve_original_images,
                )
            timing["image_convert_seconds"] += time.perf_counter() - image_convert_start
            timing["build_frame_seconds"] += time.perf_counter() - frame_start
            add_start = time.perf_counter()
            add_frame_compat(dataset, frame)
            timing["add_frame_seconds"] += time.perf_counter() - add_start
            if sample_idx < 5:
                _log(f"[sample] wrote frame idx={sample_idx}")
            frames_in_episode += 1
            if frames_in_episode >= args.frames_per_episode:
                save_start = time.perf_counter()
                frames_in_episode = save_episode_if_needed(dataset, frames_in_episode, args.frames_per_episode)
                timing["save_episode_seconds"] += time.perf_counter() - save_start
            sample_idx += 1

    _log(f"[main] finished frame loop frames_in_episode={frames_in_episode}")
    if frames_in_episode > 0:
        save_start = time.perf_counter()
        save_episode_compat(dataset)
        timing["save_episode_seconds"] += time.perf_counter() - save_start
        _log("[main] saved final episode")

    if args.stats_json is not None:
        _log(f"[main] writing stats to {args.stats_json}")
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "num_samples": len(ordered_samples),
                    "read_order": args.read_order,
                    "source_decode_batch_size": args.source_decode_batch_size,
                    "frames_per_episode": args.frames_per_episode,
                    "image_writer_threads": args.image_writer_threads,
                    "per_class_counts": dict(class_counts),
                    "per_source_repo_counts": dict(source_counts),
                    "timing_seconds": timing,
                    "total_wall_seconds": time.perf_counter() - wall_start,
                },
                indent=2,
                ensure_ascii=True,
            )
        )

    dataset.finalize()
    shutil.rmtree(Path(dataset.root) / "images", ignore_errors=True)

    if args.push_to_hub:
        _log("[main] pushing to hub")
        push_dataset_to_hub_robust(dataset, prefer_large_folder=True)
    _log("[main] done")


if __name__ == "__main__":
    main(tyro.cli(Args))

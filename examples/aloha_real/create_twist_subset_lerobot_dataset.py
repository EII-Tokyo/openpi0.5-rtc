import dataclasses
print("[top] module import start", flush=True)
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import shutil
import sys

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
    descriptions: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Args:
    source_repo_ids: tuple[str, ...] = DEFAULT_SOURCE_REPOS
    repo_id: str = "lyl472324464/twist_subset_balanced_100k_224_multi_repo"
    num_samples: int = 100000
    seed: int = 0
    overwrite: bool = True
    use_videos: bool = False
    image_size: tuple[int, int] = (224, 224)
    push_to_hub: bool = True
    frames_per_episode: int = 100
    stats_json: Path | None = Path("logs/twist_subset_balanced_100k_224_multi_repo_stats.json")
    video_backend: str | None = None


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


def _tensor_to_numpy_rgb(image: object, image_size: tuple[int, int]) -> np.ndarray:
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
            buckets[key].append(Candidate(repo_id=repo_id, index=int(row["index"]), descriptions=descriptions))
            added += 1
        _log(f"[build_buckets] bucketed {repo_id}: matched_rows={added}")
    return buckets, datasets


def main(args: Args) -> None:
    _log(f"[main] start repo_id={args.repo_id} num_samples={args.num_samples} video_backend={args.video_backend}")
    rng = random.Random(args.seed)
    templates = load_templates(Path("assets/short_language_templates.json"))
    _log("[main] creating output dataset")
    dataset = create_aloha_subtask_dataset(
        args.repo_id,
        image_size=args.image_size,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
    )
    _log("[main] building buckets from source repos")
    buckets, source_datasets = _build_buckets(args.source_repo_ids, args.video_backend)
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
    rng.shuffle(sampled)
    _log(f"[sampling] total_sampled={len(sampled)}")

    false_scalar = np.asarray([[0]], dtype=np.int64)
    true_scalar = np.asarray([[1]], dtype=np.int64)
    frames_in_episode = 0

    for sample_idx, (key, candidate) in enumerate(sampled):
        if sample_idx < 5 or sample_idx % 100 == 0:
            _log(f"[sample] idx={sample_idx} key={key} repo={candidate.repo_id} source_index={candidate.index}")
        state, subtask = key.split("|||", 1)
        target = rng.choice(candidate.descriptions) if candidate.descriptions else shorten_target(state)
        question = _sample_target_prompt(templates, rng, target)
        answer = choose_answer(templates, rng, state, subtask)

        row = source_datasets[candidate.repo_id][candidate.index]
        if sample_idx < 5:
            _log(f"[sample] fetched row idx={sample_idx}")
        state_vec = np.asarray(row["observation.state"], dtype=np.float32)
        action_vec = np.asarray(row["action"], dtype=np.float32)
        frame = {
            "observation.state": state_vec,
            "action": action_vec,
            "train_action": true_scalar.copy(),
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
        for camera_key in CAMERA_KEYS:
            frame[camera_key] = _tensor_to_numpy_rgb(row[camera_key], args.image_size)
        add_frame_compat(dataset, frame)
        if sample_idx < 5:
            _log(f"[sample] wrote frame idx={sample_idx}")
        frames_in_episode += 1
        frames_in_episode = save_episode_if_needed(dataset, frames_in_episode, args.frames_per_episode)

    _log(f"[main] finished frame loop frames_in_episode={frames_in_episode}")
    if frames_in_episode > 0:
        save_episode_compat(dataset)
        _log("[main] saved final episode")

    if args.stats_json is not None:
        _log(f"[main] writing stats to {args.stats_json}")
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "num_samples": len(sampled),
                    "per_class_counts": dict(class_counts),
                    "per_source_repo_counts": dict(source_counts),
                },
                indent=2,
                ensure_ascii=True,
            )
        )

    dataset.finalize()
    shutil.rmtree(Path(dataset.root) / "images", ignore_errors=True)

    if args.push_to_hub:
        _log("[main] pushing to hub")
        dataset.push_to_hub()
    _log("[main] done")


if __name__ == "__main__":
    main(tyro.cli(Args))

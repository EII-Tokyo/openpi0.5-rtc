#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from qwen_high_level_utils import (
    DEFAULT_CLASS_SPEC_PATH,
    describe_task_mode,
    infer_task_mode_from_repo_id,
    load_class_map,
    row_class_id,
    system_prompt,
)


DEFAULT_CAMERA_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _repo_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")


def _camera_name(column: str) -> str:
    return column.rsplit(".", 1)[-1]


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _user_prompt(camera_columns: tuple[str, ...]) -> str:
    camera_names = ", ".join(_camera_name(column) for column in camera_columns)
    return f"Use all {len(camera_columns)} images in this order: {camera_names}. Classify the current scene."


def _read_repo_ids(repo_ids: list[str], repo_id_file: Path | None) -> list[str]:
    merged = list(repo_ids)
    if repo_id_file is not None:
        for line in repo_id_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                merged.append(cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for repo_id in merged:
        cleaned = repo_id.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _repo_cache_root(base_root: Path, repo_id: str) -> Path:
    return base_root / repo_id.replace("/", "__").replace(".", "_")


def _choose_target_per_class(train_frame_counts: Counter[int], class_ids: list[int], mode: str, explicit_target: int | None) -> int:
    if explicit_target is not None and explicit_target > 0:
        return explicit_target
    nonzero = sorted(train_frame_counts[class_id] for class_id in class_ids if train_frame_counts[class_id] > 0)
    if not nonzero:
        raise RuntimeError("No train frames found for any class.")
    if mode == "min":
        return nonzero[0]
    if mode == "max":
        return nonzero[-1]
    if mode == "median":
        return int(nonzero[len(nonzero) // 2])
    raise ValueError(f"Unsupported target-per-class mode: {mode}")


def _build_val_episode_set(
    episode_class_counts: dict[tuple[str, int], Counter[int]],
    total_frames_per_class: Counter[int],
    class_ids: list[int],
    val_ratio: float,
    seed: int,
) -> set[tuple[str, int]]:
    rng = random.Random(seed)
    target = {
        class_id: max(1, int(round(total_frames_per_class[class_id] * val_ratio))) if total_frames_per_class[class_id] > 0 else 0
        for class_id in class_ids
    }
    remaining = dict(target)
    chosen: set[tuple[str, int]] = set()
    episode_items = list(episode_class_counts.items())
    rng.shuffle(episode_items)
    episode_items.sort(key=lambda item: sum(item[1].values()), reverse=True)

    while True:
        best_episode = None
        best_gain = 0
        for episode_key, counts in episode_items:
            if episode_key in chosen:
                continue
            gain = 0
            for class_id, count in counts.items():
                deficit = remaining.get(class_id, 0)
                if deficit > 0:
                    gain += min(deficit, count)
            if gain > best_gain:
                best_episode = episode_key
                best_gain = gain
        if best_episode is None or best_gain <= 0:
            break
        chosen.add(best_episode)
        for class_id, count in episode_class_counts[best_episode].items():
            remaining[class_id] = max(0, remaining[class_id] - count)
    return chosen


def _balanced_sample_train_rows(
    train_rows_by_class_repo_episode: dict[int, dict[str, dict[int, list[int]]]],
    train_frame_counts: Counter[int],
    class_ids: list[int],
    target_per_class: int,
    seed: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    selected_by_repo: dict[str, list[int]] = defaultdict(list)

    for class_id in class_ids:
        class_total = train_frame_counts[class_id]
        if class_total <= 0:
            continue
        repo_map = train_rows_by_class_repo_episode[class_id]
        repo_ids = sorted(repo_map)
        per_repo_target: dict[str, int] = {}
        remaining = target_per_class
        repo_order = repo_ids[:]
        rng.shuffle(repo_order)
        for i, repo_id in enumerate(repo_order):
            repos_left = len(repo_order) - i
            share = int(math.ceil(remaining / repos_left))
            max_available = sum(len(rows) for rows in repo_map[repo_id].values())
            allocated = min(share, max_available if class_total >= target_per_class else share)
            per_repo_target[repo_id] = allocated
            remaining -= allocated
        if remaining > 0:
            for repo_id in repo_order:
                per_repo_target[repo_id] += 1
                remaining -= 1
                if remaining == 0:
                    break

        for repo_id in repo_ids:
            repo_target = per_repo_target.get(repo_id, 0)
            if repo_target <= 0:
                continue
            episode_map = repo_map[repo_id]
            episode_ids = sorted(episode_map)
            for _ in range(repo_target):
                episode_id = rng.choice(episode_ids)
                row_idx = rng.choice(episode_map[episode_id])
                selected_by_repo[repo_id].append(row_idx)

    return selected_by_repo


def _write_selected_rows(
    *,
    repo_rows: dict[str, Any],
    image_repo_rows: dict[str, Any],
    selected_indices_by_repo: dict[str, list[int] | set[int]],
    output_jsonl: Path,
    image_dir: Path,
    camera_columns: tuple[str, ...],
    class_map: dict[int, tuple[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for repo_id, selected_indices in selected_indices_by_repo.items():
            ds = image_repo_rows[repo_id]
            repo_slug = _repo_name(repo_id)
            task_mode = infer_task_mode_from_repo_id(repo_id)
            prompt = system_prompt(class_map, task_mode)
            user_prompt = f"{_user_prompt(camera_columns)} Known task mode: {describe_task_mode(task_mode)}."
            written = 0
            ordered_indices = sorted(selected_indices) if isinstance(selected_indices, set) else list(selected_indices)
            for export_idx, row_idx in enumerate(ordered_indices):
                row = dict(ds[row_idx])
                class_id = row_class_id(row, class_map)
                if class_id is None:
                    continue
                image_paths: list[str] = []
                missing_camera = False
                for column in camera_columns:
                    if column not in row:
                        missing_camera = True
                        break
                    camera = _camera_name(column)
                    path = image_dir / repo_slug / f"{export_idx:08d}_src{row_idx:08d}_{camera}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _to_pil(row[column]).save(path, quality=95)
                    image_paths.append(str(path.resolve()))
                if missing_camera:
                    continue
                example = {
                    "task_mode": task_mode,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"{'<image>' * len(image_paths)}\n{user_prompt}"},
                        {"role": "assistant", "content": str(class_id)},
                    ],
                    "images": image_paths,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
            counts[repo_id] = written
    return counts


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", action="append", default=[])
    parser.add_argument("--repo-id-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-spec", type=Path, default=DEFAULT_CLASS_SPEC_PATH)
    parser.add_argument("--camera-columns", nargs="+", default=list(DEFAULT_CAMERA_COLUMNS))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-per-class", type=int, default=0)
    parser.add_argument("--target-total-samples", type=int, default=0)
    parser.add_argument("--target-per-class-mode", choices=("median", "min", "max"), default="median")
    parser.add_argument("--lerobot-root", type=Path, default=Path(".cache/lerobot"))
    parser.add_argument("--skip-train-export-if-exists", action="store_true")
    parser.add_argument("--skip-val-export-if-exists", action="store_true")
    args = parser.parse_args()

    repo_ids = _read_repo_ids(args.repo_id, args.repo_id_file)
    if not repo_ids:
        raise RuntimeError("No repo ids provided.")

    class_map = load_class_map(args.class_spec)
    class_ids = sorted(class_map)
    camera_columns = tuple(args.camera_columns)
    repo_rows: dict[str, Any] = {}
    image_repo_rows: dict[str, Any] = {}
    skipped_repos: dict[str, str] = {}

    total_frame_counts = Counter()
    repo_frame_counts: dict[str, Counter[int]] = {}
    repo_episode_counts: dict[str, Counter[int]] = {}
    episode_class_counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)

    for repo_id in repo_ids:
        try:
            base_ds = LeRobotDataset(
                repo_id,
                root=_repo_cache_root(args.lerobot_root, repo_id),
                force_cache_sync=True,
                download_videos=False,
                delta_timestamps=None,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_repos[repo_id] = str(exc)
            continue
        repo_rows[repo_id] = base_ds.hf_dataset
        repo_counter = Counter()
        repo_episode_counter = Counter()
        seen_episode_class: set[tuple[int, int]] = set()
        for row_idx, row in enumerate(base_ds.hf_dataset):
            row_dict = dict(row)
            class_id = row_class_id(row_dict, class_map)
            if class_id is None:
                continue
            episode_index = _scalar_int(row_dict.get("episode_index", -1))
            total_frame_counts[class_id] += 1
            repo_counter[class_id] += 1
            episode_class_counts[(repo_id, episode_index)][class_id] += 1
            episode_class_counts[(repo_id, episode_index)]["_rows"] = episode_class_counts[(repo_id, episode_index)].get("_rows", 0) + 1  # type: ignore[index]
            key = (episode_index, class_id)
            if key not in seen_episode_class:
                seen_episode_class.add(key)
                repo_episode_counter[class_id] += 1
        repo_frame_counts[repo_id] = repo_counter
        repo_episode_counts[repo_id] = repo_episode_counter

    for repo_id in repo_rows:
        try:
            image_repo_rows[repo_id] = LeRobotDataset(
                repo_id,
                root=_repo_cache_root(args.lerobot_root, repo_id),
                force_cache_sync=False,
                download_videos=True,
                delta_timestamps=None,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_repos[repo_id] = f"video load failed: {exc}"
    repo_rows = {repo_id: ds for repo_id, ds in repo_rows.items() if repo_id in image_repo_rows}
    repo_frame_counts = {repo_id: counter for repo_id, counter in repo_frame_counts.items() if repo_id in image_repo_rows}
    repo_episode_counts = {repo_id: counter for repo_id, counter in repo_episode_counts.items() if repo_id in image_repo_rows}

    clean_episode_class_counts: dict[tuple[str, int], Counter[int]] = {}
    for key, counts in episode_class_counts.items():
        clean_counter = Counter()
        for class_id in class_ids:
            if counts[class_id] > 0:
                clean_counter[class_id] = counts[class_id]
        clean_episode_class_counts[key] = clean_counter

    val_episode_set = _build_val_episode_set(
        clean_episode_class_counts,
        total_frame_counts,
        class_ids,
        args.val_ratio,
        args.seed,
    )
    val_episodes_by_repo: dict[str, list[int]] = defaultdict(list)
    for repo_id, episode_index in sorted(val_episode_set):
        val_episodes_by_repo[repo_id].append(episode_index)

    train_rows_by_class_repo_episode: dict[int, dict[str, dict[int, list[int]]]] = {
        class_id: defaultdict(lambda: defaultdict(list)) for class_id in class_ids
    }
    val_row_indices_by_repo: dict[str, set[int]] = defaultdict(set)
    train_frame_counts = Counter()
    val_frame_counts = Counter()

    for repo_id, ds in repo_rows.items():
        for row_idx, row in enumerate(ds):
            row_dict = dict(row)
            class_id = row_class_id(row_dict, class_map)
            if class_id is None:
                continue
            episode_index = _scalar_int(row_dict.get("episode_index", -1))
            if (repo_id, episode_index) in val_episode_set:
                val_row_indices_by_repo[repo_id].add(row_idx)
                val_frame_counts[class_id] += 1
            else:
                train_rows_by_class_repo_episode[class_id][repo_id][episode_index].append(row_idx)
                train_frame_counts[class_id] += 1

    if args.target_total_samples > 0:
        active_class_ids = [class_id for class_id in class_ids if train_frame_counts[class_id] > 0]
        if not active_class_ids:
            raise RuntimeError("No train frames found for any class.")
        target_per_class = int(math.ceil(args.target_total_samples / len(active_class_ids)))
    else:
        target_per_class = _choose_target_per_class(
            train_frame_counts,
            class_ids,
            args.target_per_class_mode,
            args.target_per_class if args.target_per_class > 0 else None,
        )
    balanced_train_indices_by_repo = _balanced_sample_train_rows(
        train_rows_by_class_repo_episode,
        train_frame_counts,
        class_ids,
        target_per_class,
        args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = args.output_dir / "train.jsonl"
    val_jsonl = args.output_dir / "val.jsonl"
    if args.skip_train_export_if_exists and train_jsonl.exists():
        train_counts = {"__existing_total__": _count_jsonl_lines(train_jsonl)}
        print(f"Skipping train export because {train_jsonl} already exists.")
    else:
        train_counts = _write_selected_rows(
            repo_rows=repo_rows,
            image_repo_rows=image_repo_rows,
            selected_indices_by_repo=balanced_train_indices_by_repo,
            output_jsonl=train_jsonl,
            image_dir=args.output_dir / "images" / "train",
            camera_columns=camera_columns,
            class_map=class_map,
        )
    if args.skip_val_export_if_exists and val_jsonl.exists():
        val_counts = {"__existing_total__": _count_jsonl_lines(val_jsonl)}
        print(f"Skipping val export because {val_jsonl} already exists.")
    else:
        val_counts = _write_selected_rows(
            repo_rows=repo_rows,
            image_repo_rows=image_repo_rows,
            selected_indices_by_repo=val_row_indices_by_repo,
            output_jsonl=val_jsonl,
            image_dir=args.output_dir / "images" / "val",
            camera_columns=camera_columns,
            class_map=class_map,
        )

    metadata = {
        "class_spec": str(args.class_spec),
        "camera_columns": list(camera_columns),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "target_per_class": target_per_class,
        "target_total_samples": args.target_total_samples,
        "target_per_class_mode": args.target_per_class_mode,
        "num_accessible_repos": len(repo_rows),
        "num_skipped_repos": len(skipped_repos),
        "skipped_repos": skipped_repos,
        "class_map": {
            str(class_id): {"bottle_state": class_map[class_id][0], "subtask": class_map[class_id][1]}
            for class_id in class_ids
        },
        "total_frame_counts": {str(class_id): total_frame_counts[class_id] for class_id in class_ids},
        "train_frame_counts_before_balancing": {str(class_id): train_frame_counts[class_id] for class_id in class_ids},
        "val_frame_counts": {str(class_id): val_frame_counts[class_id] for class_id in class_ids},
        "train_export_counts_by_repo": train_counts,
        "val_export_counts_by_repo": val_counts,
        "val_episodes_by_repo": {repo_id: sorted(episodes) for repo_id, episodes in sorted(val_episodes_by_repo.items())},
        "repo_frame_counts": {
            repo_id: {str(class_id): counter[class_id] for class_id in class_ids if counter[class_id] > 0}
            for repo_id, counter in sorted(repo_frame_counts.items())
        },
        "repo_episode_counts": {
            repo_id: {str(class_id): counter[class_id] for class_id in class_ids if counter[class_id] > 0}
            for repo_id, counter in sorted(repo_episode_counts.items())
        },
    }
    (args.output_dir / "stats.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "val_episodes_by_repo.json").write_text(
        json.dumps(metadata["val_episodes_by_repo"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from qwen_high_level_utils import DEFAULT_CLASS_SPEC_PATH, load_class_map, row_class_id


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


def _scalar_int(x: Any) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def _build_val_split(
    episode_class_counts: dict[tuple[str, int], Counter[int]],
    total_frames_per_class: Counter[int],
    val_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    target = {
        class_id: max(1, int(round(total * val_ratio))) if total > 0 else 0
        for class_id, total in total_frames_per_class.items()
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
            remaining[class_id] = max(0, remaining.get(class_id, 0) - count)

    by_repo: dict[str, list[int]] = defaultdict(list)
    for repo_id, episode_index in sorted(chosen):
        by_repo[repo_id].append(int(episode_index))
    for repo_id in by_repo:
        by_repo[repo_id].sort()
    return dict(by_repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", action="append", default=[])
    parser.add_argument("--repo-id-file", type=Path, default=None)
    parser.add_argument("--class-spec", type=Path, default=DEFAULT_CLASS_SPEC_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, default=Path(".cache/lerobot"))
    args = parser.parse_args()

    repo_ids = _read_repo_ids(args.repo_id, args.repo_id_file)
    if not repo_ids:
        raise RuntimeError("No repo ids provided.")
    class_map = load_class_map(args.class_spec)
    frame_counts = Counter()
    repo_frame_counts: dict[str, Counter[int]] = {}
    repo_episode_counts: dict[str, Counter[int]] = {}
    episode_class_counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    skipped_repos: dict[str, str] = {}
    skipped_rows = 0

    for repo_id in repo_ids:
        try:
            ds = LeRobotDataset(
                repo_id,
                root=_repo_cache_root(args.lerobot_root, repo_id),
                force_cache_sync=True,
                download_videos=False,
                delta_timestamps=None,
            ).hf_dataset
        except Exception as exc:  # noqa: BLE001
            skipped_repos[repo_id] = str(exc)
            continue
        repo_counter = Counter()
        repo_episode_counter = Counter()
        repo_seen_episode_class: set[tuple[int, int]] = set()
        for row in ds:
            row_dict = dict(row)
            class_id = row_class_id(row_dict, class_map)
            if class_id is None:
                skipped_rows += 1
                continue
            episode_index = _scalar_int(row_dict.get("episode_index", -1))
            frame_counts[class_id] += 1
            repo_counter[class_id] += 1
            episode_class_counts[(repo_id, episode_index)][class_id] += 1
            key = (episode_index, class_id)
            if key not in repo_seen_episode_class:
                repo_seen_episode_class.add(key)
                repo_episode_counter[class_id] += 1
        repo_frame_counts[repo_id] = repo_counter
        repo_episode_counts[repo_id] = repo_episode_counter

    val_episodes_by_repo = _build_val_split(episode_class_counts, frame_counts, args.val_ratio, args.seed)
    val_episode_set = {(repo_id, ep) for repo_id, eps in val_episodes_by_repo.items() for ep in eps}
    train_frames = Counter()
    val_frames = Counter()
    for episode_key, counts in episode_class_counts.items():
        target = val_frames if episode_key in val_episode_set else train_frames
        for class_id, count in counts.items():
            target[class_id] += count

    result = {
        "class_spec": str(args.class_spec),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_accessible_repos": len(repo_frame_counts),
        "num_skipped_repos": len(skipped_repos),
        "skipped_repos": skipped_repos,
        "skipped_rows": skipped_rows,
        "classes": {
            str(class_id): {
                "bottle_state": class_map[class_id][0],
                "subtask": class_map[class_id][1],
                "total_frames": frame_counts[class_id],
                "train_frames": train_frames[class_id],
                "val_frames": val_frames[class_id],
            }
            for class_id in sorted(class_map)
        },
        "repo_frame_counts": {
            repo_id: {str(class_id): counter[class_id] for class_id in sorted(class_map) if counter[class_id] > 0}
            for repo_id, counter in sorted(repo_frame_counts.items())
        },
        "repo_episode_counts": {
            repo_id: {str(class_id): counter[class_id] for class_id in sorted(class_map) if counter[class_id] > 0}
            for repo_id, counter in sorted(repo_episode_counts.items())
        },
        "val_episodes_by_repo": val_episodes_by_repo,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

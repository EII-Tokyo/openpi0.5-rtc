#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi_client import subtask_parsing as _subtask_parsing


@dataclasses.dataclass(frozen=True)
class Args:
    output_path: Path = Path("assets/gpt_high_level_subtask_manifest.json")
    namespace: str = "lyl472324464"
    exclude_repo_ids: tuple[str, ...] = ("lyl472324464/vqa_sample_100",)


def _discover_repo_ids(namespace: str, exclude_repo_ids: set[str]) -> list[str]:
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / namespace
    if not base.exists():
        raise FileNotFoundError(f"LeRobot cache directory not found: {base}")

    repo_ids: list[str] = []
    for repo_dir in sorted(base.iterdir()):
        info_path = repo_dir / "meta" / "info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        features = info.get("features", {})
        repo_id = f"{namespace}/{repo_dir.name}"
        if repo_id in exclude_repo_ids:
            continue
        if "subtask" not in features:
            continue
        repo_ids.append(repo_id)
    return repo_ids


def main(args: Args) -> None:
    repo_ids = _discover_repo_ids(args.namespace, set(args.exclude_repo_ids))
    if not repo_ids:
        raise SystemExit("No source datasets with `subtask` field were found.")

    subtask_counter: Counter[str] = Counter()
    bottle_state_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    repo_frame_counts: dict[str, int] = {}
    repo_episode_counts: dict[str, int] = {}
    total_frames = 0
    total_episodes = 0

    for repo_id in repo_ids:
        ds = LeRobotDataset(
            repo_id,
            revision="main",
            force_cache_sync=False,
            download_videos=False,
            delta_timestamps=None,
        )
        repo_frame_counts[repo_id] = len(ds)
        repo_episode_counts[repo_id] = int(ds.num_episodes)
        total_frames += len(ds)
        total_episodes += int(ds.num_episodes)

        for raw in ds.hf_dataset["subtask"]:
            parsed = _subtask_parsing.parse_structured_fields(raw)
            subtask = str(parsed.get("subtask") or "").strip() or "UNKNOWN"
            bottle_state = str(parsed.get("bottle_state") or "").strip() or "UNKNOWN"
            subtask_counter[subtask] += 1
            bottle_state_counter[bottle_state] += 1
            pair_counter[f"{bottle_state} -> {subtask}"] += 1

    def _to_fraction_dict(counter: Counter[str]) -> dict[str, float]:
        return {key: value / total_frames for key, value in counter.most_common()} if total_frames else {}

    manifest = {
        "description": "Zero-copy manifest for GPT high-level generated subtask datasets.",
        "task": "Process all bottles",
        "repo_ids": repo_ids,
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "repo_frame_counts": repo_frame_counts,
        "repo_episode_counts": repo_episode_counts,
        "subtask_counts": dict(subtask_counter.most_common()),
        "subtask_fractions": _to_fraction_dict(subtask_counter),
        "bottle_state_counts": dict(bottle_state_counter.most_common()),
        "bottle_state_fractions": _to_fraction_dict(bottle_state_counter),
        "pair_counts": dict(pair_counter.most_common()),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved manifest to {args.output_path}")
    print(f"Source repos: {len(repo_ids)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print("Top subtasks:")
    for subtask, count in subtask_counter.most_common(10):
        print(f"  {subtask}: {count} ({count / total_frames:.2%})")


if __name__ == "__main__":
    main(tyro.cli(Args))

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from pymongo import MongoClient

from openpi.models.tokenizer import get_good_bad_action_label

DEFAULT_REPOS = [
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
]

TARGET_LABELS = ("good action", "bad action")
TRUNCATE_LABEL = "need to be truncated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HF dataset good/bad frame labels vs Mongo slices by episode. "
            "By default, applies truncate logic: remove frames in 'need to be truncated', "
            "and drop episodes where one truncate slice length equals episode length."
        )
    )
    parser.add_argument("--mongo-uri", default="mongodb://192.168.1.40:27017/")
    parser.add_argument("--mongo-db", default="eii_data_system")
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="HF dataset repo id. Repeat to pass multiple repos.",
    )
    parser.add_argument(
        "--no-apply-truncate",
        action="store_true",
        help="Disable truncate logic and compare raw Mongo slices only.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/eii/openpi0.5-rtc/tmp/hf_mongo_good_bad_compare.json",
        help="Path to write detailed comparison JSON.",
    )
    return parser.parse_args()


def frames_to_ranges(frames: set[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    xs = sorted(frames)
    out: list[tuple[int, int]] = []
    start = end = xs[0]
    for x in xs[1:]:
        if x == end + 1:
            end = x
        else:
            out.append((start, end))
            start = end = x
    out.append((start, end))
    return out


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    xs = sorted((int(a), int(b)) for a, b in ranges if int(b) >= int(a))
    merged: list[tuple[int, int]] = []
    s, e = xs[0]
    for a, b in xs[1:]:
        if a <= e + 1:
            e = max(e, b)
        else:
            merged.append((s, e))
            s, e = a, b
    merged.append((s, e))
    return merged


def subtract_ranges(
    source_ranges: list[tuple[int, int]], cut_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    source = merge_ranges(source_ranges)
    cuts = merge_ranges(cut_ranges)
    if not source or not cuts:
        return source

    out: list[tuple[int, int]] = []
    cut_ptr = 0
    for start, end in source:
        parts = [(start, end)]

        while cut_ptr < len(cuts) and cuts[cut_ptr][1] < start:
            cut_ptr += 1

        ptr = cut_ptr
        while ptr < len(cuts) and cuts[ptr][0] <= end and parts:
            cut_s, cut_e = cuts[ptr]
            new_parts: list[tuple[int, int]] = []
            for part_s, part_e in parts:
                if cut_e < part_s or cut_s > part_e:
                    new_parts.append((part_s, part_e))
                    continue
                if cut_s > part_s:
                    new_parts.append((part_s, cut_s - 1))
                if cut_e < part_e:
                    new_parts.append((cut_e + 1, part_e))
            parts = new_parts
            ptr += 1

        out.extend(parts)

    return merge_ranges(out)


def ranges_to_set(ranges: list[tuple[int, int]]) -> set[int]:
    frames: set[int] = set()
    for a, b in ranges:
        frames.update(range(a, b + 1))
    return frames


def build_hf_map(repo: str) -> tuple[dict[int, dict[str, set[int]]], set[int]]:
    ep_map: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: {"good action": set(), "bad action": set()}
    )

    local_dir = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["data/**/*.parquet"],
    )

    all_hf_eps: set[int] = set()

    for f in sorted(Path(local_dir).glob("data/**/*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "frame_index", "subtask"])
        eps = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        subtasks = table["subtask"].to_pylist()

        for ep, frame, subtask in zip(eps, frames, subtasks):
            all_hf_eps.add(int(ep))
            label = get_good_bad_action_label(subtask)
            if label in TARGET_LABELS:
                ep_map[int(ep)][label].add(int(frame))

    return ep_map, all_hf_eps


def build_mongo_map(
    db, project_id: str, apply_truncate: bool
) -> tuple[dict[int, str], dict[int, dict[str, set[int]]], list[int]]:
    episodes = list(
        db.episodes.find(
            {"project_id": project_id}, {"_id": 1, "episode_index": 1, "data_count": 1}
        )
    )
    episodes = sorted(episodes, key=lambda x: int(x["episode_index"]))

    ep_idx_to_id = {int(e["episode_index"]): str(e["_id"]) for e in episodes if "episode_index" in e}
    ep_id_to_idx = {v: k for k, v in ep_idx_to_id.items()}
    ep_idx_to_count = {
        int(e["episode_index"]): (int(e["data_count"]) if e.get("data_count") is not None else None)
        for e in episodes
        if "episode_index" in e
    }

    raw_ranges: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"good action": [], "bad action": [], TRUNCATE_LABEL: []}
    )

    if ep_idx_to_id:
        cursor = db.slices.find(
            {
                "episode_id": {"$in": list(ep_idx_to_id.values())},
                "label": {"$in": [*TARGET_LABELS, TRUNCATE_LABEL]},
            },
            {"episode_id": 1, "label": 1, "start_index": 1, "end_index": 1},
        )
        for s in cursor:
            ep_id = s.get("episode_id")
            if ep_id not in ep_id_to_idx:
                continue
            label = s.get("label")
            if label not in (TARGET_LABELS + (TRUNCATE_LABEL,)):
                continue
            start_idx = int(s.get("start_index", 0))
            end_idx = int(s.get("end_index", -1))
            if end_idx < start_idx:
                continue
            raw_ranges[ep_id_to_idx[ep_id]][label].append((start_idx, end_idx))

    if not apply_truncate:
        ep_map: dict[int, dict[str, set[int]]] = defaultdict(
            lambda: {"good action": set(), "bad action": set()}
        )
        for ep in sorted(ep_idx_to_id):
            for label in TARGET_LABELS:
                ep_map[ep][label] = ranges_to_set(merge_ranges(raw_ranges[ep][label]))
        return ep_idx_to_id, ep_map, []

    # Apply truncate policy and compact episode indices.
    deleted_episodes: list[int] = []
    remap_old_to_new: dict[int, int] = {}
    new_ep_idx = 0

    ep_map: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: {"good action": set(), "bad action": set()}
    )

    for old_ep in sorted(ep_idx_to_id):
        truncate_ranges = merge_ranges(raw_ranges[old_ep][TRUNCATE_LABEL])
        data_count = ep_idx_to_count.get(old_ep)

        full_delete = False
        if data_count is not None:
            for start_idx, end_idx in truncate_ranges:
                if end_idx - start_idx + 1 == data_count:
                    full_delete = True
                    break

        if full_delete:
            deleted_episodes.append(old_ep)
            continue

        remap_old_to_new[old_ep] = new_ep_idx
        for label in TARGET_LABELS:
            kept_ranges = subtract_ranges(raw_ranges[old_ep][label], truncate_ranges)
            ep_map[new_ep_idx][label] = ranges_to_set(kept_ranges)

        new_ep_idx += 1

    remapped_idx_to_id = {new_idx: ep_idx_to_id[old_idx] for old_idx, new_idx in remap_old_to_new.items()}
    return remapped_idx_to_id, ep_map, deleted_episodes


def compare_repo(db, repo: str, apply_truncate: bool) -> dict[str, Any]:
    projects = list(db.projects.find({"dataset_repo": repo}, {"_id": 1, "name": 1, "dataset_repo": 1}))
    if len(projects) != 1:
        return {
            "repo": repo,
            "error": f"expected 1 project for dataset_repo, got {len(projects)}",
        }

    project = projects[0]
    project_id = str(project["_id"])

    hf_map, hf_all_eps = build_hf_map(repo)
    ep_idx_to_id, mongo_map, deleted_episodes = build_mongo_map(
        db, project_id, apply_truncate=apply_truncate
    )

    all_eps = sorted(set(hf_map.keys()) | set(mongo_map.keys()) | set(ep_idx_to_id.keys()))
    missing_in_hf = sorted(set(ep_idx_to_id.keys()) - set(hf_all_eps))
    extra_in_hf = sorted(set(hf_all_eps) - set(ep_idx_to_id.keys()))

    mismatches: list[dict[str, Any]] = []
    for ep in all_eps:
        for label in TARGET_LABELS:
            hf_frames = hf_map.get(ep, {}).get(label, set())
            mg_frames = mongo_map.get(ep, {}).get(label, set())
            if hf_frames != mg_frames:
                mismatches.append(
                    {
                        "episode_index": ep,
                        "label": label,
                        "hf_frame_count": len(hf_frames),
                        "mongo_frame_count": len(mg_frames),
                        "hf_ranges": frames_to_ranges(hf_frames),
                        "mongo_ranges": frames_to_ranges(mg_frames),
                    }
                )

    return {
        "repo": repo,
        "project_id": project_id,
        "project_name": project.get("name"),
        "truncate_policy_applied": apply_truncate,
        "deleted_episodes_by_truncate": deleted_episodes,
        "mongo_episode_count": len(ep_idx_to_id),
        "hf_episode_count": len(hf_all_eps),
        "hf_episode_with_good_bad_count": len(set(hf_map.keys())),
        "missing_episode_in_hf": missing_in_hf,
        "extra_episode_in_hf": extra_in_hf,
        "mismatch_pair_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    repos = args.repos or DEFAULT_REPOS

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.mongo_db]

    apply_truncate = not args.no_apply_truncate
    results = [compare_repo(db, repo, apply_truncate=apply_truncate) for repo in repos]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved detailed result to: {out_path}")
    for r in results:
        if "error" in r:
            print(f"[ERROR] {r['repo']}: {r['error']}")
            continue
        print(f"\n{r['repo']}")
        print(f"  project: {r['project_name']} ({r['project_id']})")
        print(f"  truncate_policy_applied: {r['truncate_policy_applied']}")
        print(f"  deleted_episodes_by_truncate: {r['deleted_episodes_by_truncate']}")
        print(f"  mongo_episode_count: {r['mongo_episode_count']}")
        print(f"  hf_episode_count: {r['hf_episode_count']}")
        print(f"  missing_episode_in_hf: {r['missing_episode_in_hf']}")
        print(f"  extra_episode_in_hf: {r['extra_episode_in_hf']}")
        print(f"  mismatch_pair_count(ep,label): {r['mismatch_pair_count']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Populate LeRobot dataset with per-frame `subtask` from Mongo slices and rewrite prompt tasks."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import urllib.request

import pandas as pd
from pymongo import MongoClient


SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1QZcUjSNiFQZOsQs2Y2VaSksd_hsWwWIJqPh75sw8Rh4/export"
    "?format=csv&gid=280976830"
)


def _build_subtask_from_sheet_row(row: dict[str, str]) -> str:
    action = (row.get("action") or "").strip().replace("_", " ")
    target = (row.get("target") or "").strip().replace("_", " ")
    context = (row.get("context_or_result") or "").strip().replace("_", " ")
    if not action:
        return ""
    parts = [action]
    if target and target != "none":
        parts.append(target)
    if context:
        parts.append(context)
    return " ".join(parts).strip()


def load_label_to_subtask() -> dict[str, str]:
    text = urllib.request.urlopen(SHEET_CSV_URL, timeout=20).read().decode("utf-8", "ignore")
    rows = csv.DictReader(io.StringIO(text))
    mapping: dict[str, str] = {}
    for row in rows:
        file_name = (row.get("file_name") or "").strip()
        if not file_name:
            continue
        phrase = _build_subtask_from_sheet_row(row)
        if phrase:
            mapping[file_name] = phrase
    mapping["bad action"] = "perform a bad action"
    mapping["need to be truncated"] = "truncate the trajectory"
    return mapping


def load_episode_slices(
    mongo_uri: str,
    db_name: str,
    project_id: str,
) -> dict[int, list[tuple[int, int, str]]]:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]

    episodes = list(db["episodes"].find({"project_id": project_id}, {"_id": 1, "episode_index": 1}))
    ep_id_to_index = {str(e["_id"]): int(e["episode_index"]) for e in episodes}
    episode_ids = list(ep_id_to_index.keys())
    if not episode_ids:
        raise RuntimeError(f"No episodes found for project_id={project_id} in {db_name}.")

    by_episode: dict[int, list[tuple[int, int, str]]] = {}
    cursor = db["slices"].find(
        {"episode_id": {"$in": episode_ids}},
        {"episode_id": 1, "start_index": 1, "end_index": 1, "label": 1},
    )
    for doc in cursor:
        ep_idx = ep_id_to_index.get(doc.get("episode_id"))
        if ep_idx is None:
            continue
        start = int(doc.get("start_index", 0))
        end = int(doc.get("end_index", -1))
        label = str(doc.get("label", "")).strip()
        by_episode.setdefault(ep_idx, []).append((start, end, label))

    # Stable ordering: earliest slice first, then shorter interval first.
    for ep_idx in by_episode:
        by_episode[ep_idx].sort(key=lambda x: (x[0], x[1] - x[0]))
    return by_episode


def pick_subtask(frame_index: int, slices: list[tuple[int, int, str]], label_to_subtask: dict[str, str]) -> str:
    for start, end, label in slices:
        if start <= frame_index <= end:
            return label_to_subtask.get(label, label.replace("__", " ").replace("_", " ").strip())
    return "return to home position"


def rewrite_dataset(
    repo_root: Path,
    episode_slices: dict[int, list[tuple[int, int, str]]],
    label_to_subtask: dict[str, str],
    prompts: list[str],
    *,
    dry_run: bool,
) -> tuple[int, int]:
    tasks_path = repo_root / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)

    old_bad_idx = None
    for desc, row in tasks_df.iterrows():
        if "[bad action]" in str(desc):
            old_bad_idx = int(row["task_index"])
            break
    if old_bad_idx is None:
        old_bad_idx = 1

    n = len(prompts)
    new_good_prompts = prompts
    new_bad_prompts = [f"[bad action] {p}" for p in prompts]

    # Rewrite tasks mapping.
    new_task_rows = [
        {"task": p, "task_index": i} for i, p in enumerate(new_good_prompts + new_bad_prompts)
    ]
    new_tasks_df = pd.DataFrame(new_task_rows).set_index("task")

    updated_rows = 0
    data_files = sorted((repo_root / "data").rglob("*.parquet"))
    for file in data_files:
        df = pd.read_parquet(file)
        # keep old bad-action signal, but remap to prompt variants.
        is_bad = (df["task_index"].astype(int) == old_bad_idx).astype(int)
        variant = (df["episode_index"].astype(int) % n).astype(int)
        df["task_index"] = variant + is_bad * n

        subtasks = []
        for ep, fr in zip(df["episode_index"].astype(int), df["frame_index"].astype(int), strict=True):
            slices = episode_slices.get(ep, [])
            subtasks.append(pick_subtask(fr, slices, label_to_subtask))
        df["subtask"] = subtasks
        updated_rows += len(df)
        if not dry_run:
            df.to_parquet(file, index=False)

    if not dry_run:
        new_tasks_df.to_parquet(tasks_path, index=True)

        info_path = repo_root / "meta" / "info.json"
        if info_path.exists():
            import json

            info = json.loads(info_path.read_text())
            features = info.get("features", {})
            if "subtask" not in features:
                features["subtask"] = {"dtype": "string", "shape": [1], "names": None}
            info["features"] = features
            info["total_tasks"] = len(new_good_prompts) + len(new_bad_prompts)
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return len(data_files), updated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lyl472324464/2025-12-23-twist-one-bottle")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="eii_data_system")
    parser.add_argument("--project-id", default="6970f2f66cf3c292e7924317")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "twist off the bottle cap",
            "unscrew the bottle cap",
            "remove the bottle cap",
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.repo_root is None:
        repo_root = Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
    else:
        repo_root = args.repo_root

    if not repo_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {repo_root}")

    label_to_subtask = load_label_to_subtask()
    episode_slices = load_episode_slices(args.mongo_uri, args.mongo_db, args.project_id)
    files, rows = rewrite_dataset(
        repo_root,
        episode_slices,
        label_to_subtask,
        prompts=args.prompts,
        dry_run=args.dry_run,
    )
    print(f"updated files={files}, rows={rows}, dry_run={args.dry_run}")

    if args.push and not args.dry_run:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        dataset = lerobot_dataset.LeRobotDataset(args.repo_id)
        dataset.push_to_hub()
        print(f"pushed to hub: {args.repo_id}")


if __name__ == "__main__":
    main()

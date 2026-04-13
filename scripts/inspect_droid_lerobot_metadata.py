#!/usr/bin/env python3
"""Lightweight metadata inspection for LeRobot DROID without full download."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--sample-file", default="data/chunk-000/file-000.parquet")
    parser.add_argument("--sample-rows", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_info(repo_id: str, repo_type: str) -> dict:
    info_path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename="meta/info.json")
    with open(info_path) as f:
        return json.load(f)


def load_tasks(repo_id: str, repo_type: str) -> tuple[int, int]:
    tasks_path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename="meta/tasks.parquet")
    table = pq.read_table(tasks_path, columns=["task_index", "__index_level_0__"])
    task_strings = table.column("__index_level_0__").to_pylist()
    unique_all = len(set(task_strings))
    unique_non_empty = len({t for t in task_strings if t})
    return unique_all, unique_non_empty


def load_episode_task_stats(repo_id: str, repo_type: str, top_k: int) -> dict:
    episode_files = [f"meta/episodes/chunk-000/file-{i:03d}.parquet" for i in range(7)]
    episode_task_counter = Counter()
    frame_task_counter = Counter()
    total_episodes = 0

    for ep_file in episode_files:
        path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=ep_file)
        table = pq.read_table(path, columns=["tasks", "length"])
        task_lists = table.column("tasks").to_pylist()
        lengths = table.column("length").to_pylist()
        total_episodes += len(task_lists)

        for tasks, length in zip(task_lists, lengths, strict=False):
            if tasks is None:
                continue
            if isinstance(tasks, str):
                unique_tasks = [tasks]
            else:
                unique_tasks = list(dict.fromkeys(tasks))
            for task in unique_tasks:
                episode_task_counter[task] += 1
                frame_task_counter[task] += int(length)

    top_episode = episode_task_counter.most_common(top_k)
    top_frame = frame_task_counter.most_common(top_k)
    return {
        "episodes_total": total_episodes,
        "unique_tasks_in_episode_metadata": len(episode_task_counter),
        "empty_task_episodes": episode_task_counter.get("", 0),
        "top_tasks_by_episode_mentions": top_episode,
        "top_tasks_by_frame_weighted_mentions": top_frame,
    }


def inspect_action_fields(repo_id: str, repo_type: str, sample_file: str, sample_rows: int) -> dict:
    path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=sample_file)
    cols = [
        "action",
        "action.joint_position",
        "action.joint_velocity",
        "action.gripper_position",
        "action.gripper_velocity",
        "action.original",
    ]
    table = pq.read_table(path, columns=cols)
    n = min(sample_rows, table.num_rows)
    action = np.asarray(table.column("action").slice(0, n).to_pylist(), dtype=np.float32)
    joint_position = np.asarray(table.column("action.joint_position").slice(0, n).to_pylist(), dtype=np.float32)
    gripper_position = (
        np.asarray(table.column("action.gripper_position").slice(0, n).to_pylist(), dtype=np.float32).reshape(-1, 1)
    )
    joint_velocity = np.asarray(table.column("action.joint_velocity").slice(0, n).to_pylist(), dtype=np.float32)
    gripper_velocity = (
        np.asarray(table.column("action.gripper_velocity").slice(0, n).to_pylist(), dtype=np.float32).reshape(-1, 1)
    )

    reconstructed_position = np.concatenate([joint_position, gripper_position], axis=1)
    reconstructed_velocity = np.concatenate([joint_velocity, gripper_velocity], axis=1)
    max_diff_position = float(np.max(np.abs(action - reconstructed_position)))
    max_diff_velocity = float(np.max(np.abs(action - reconstructed_velocity)))
    action_original = np.asarray(table.column("action.original").slice(0, min(5, n)).to_pylist(), dtype=np.float32)

    return {
        "sample_rows_checked": n,
        "action_shape": list(action.shape),
        "action_original_shape_first_rows": list(action_original.shape),
        "max_abs_diff_action_vs_joint_position_plus_gripper_position": max_diff_position,
        "max_abs_diff_action_vs_joint_velocity_plus_gripper_velocity": max_diff_velocity,
    }


def main() -> None:
    args = parse_args()
    info = load_info(args.repo_id, args.repo_type)
    unique_all, unique_non_empty = load_tasks(args.repo_id, args.repo_type)
    episode_stats = load_episode_task_stats(args.repo_id, args.repo_type, args.top_k)
    action_stats = inspect_action_fields(args.repo_id, args.repo_type, args.sample_file, args.sample_rows)

    report = {
        "repo_id": args.repo_id,
        "info": {
            "total_tasks": info["total_tasks"],
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "fps": info["fps"],
        },
        "task_string_cardinality_from_meta_tasks_parquet": {
            "unique_tasks_including_empty": unique_all,
            "unique_tasks_excluding_empty": unique_non_empty,
        },
        "episode_task_stats": episode_stats,
        "action_field_analysis": action_stats,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

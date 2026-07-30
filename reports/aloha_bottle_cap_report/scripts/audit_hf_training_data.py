#!/usr/bin/env python3
"""Audit the exact Hugging Face repositories in the deployed training recipe."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any

import pyarrow.parquet as pq
import requests


API_BASE = "https://huggingface.co/api/datasets/"
RAW_BASE = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def get_bytes(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    ordered = sorted(values)

    def linear(q: float) -> float:
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        if lo == hi:
            return float(ordered[lo])
        return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)

    return {
        "min": ordered[0],
        "q25": linear(0.25),
        "median": statistics.median(ordered),
        "q75": linear(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def feature_summary(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features") or {}
    cameras = []
    for key, value in features.items():
        if key.startswith("observation.images."):
            cameras.append(
                {
                    "key": key,
                    "shape": value.get("shape"),
                    "video_info": value.get("info"),
                }
            )
    return {
        "state_shape": (features.get("observation.state") or {}).get("shape"),
        "action_shape": (features.get("action") or {}).get("shape"),
        "camera_features": cameras,
        "has_language_task_index": "task_index" in features,
        "has_training_mask": "is_for_training" in features,
        "has_reward": any("reward" in key.lower() for key in features),
        "has_success": any("success" in key.lower() for key in features),
        "feature_keys": sorted(features),
    }


def audit_repo(session: requests.Session, repo_id: str, weight: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hub = get_json(session, API_BASE + repo_id)
    revision = hub["sha"]
    info = get_json(session, RAW_BASE.format(repo_id=repo_id, revision=revision, path="meta/info.json"))
    episode_files = sorted(
        sibling["rfilename"]
        for sibling in hub.get("siblings", [])
        if sibling.get("rfilename", "").startswith("meta/episodes/")
        and sibling["rfilename"].endswith(".parquet")
    )
    episode_rows: list[dict[str, Any]] = []
    desired_columns = [
        "episode_index",
        "tasks",
        "length",
        "stats/is_for_training/count",
        "stats/is_for_training/mean",
    ]
    for path in episode_files:
        payload = get_bytes(session, RAW_BASE.format(repo_id=repo_id, revision=revision, path=path))
        parquet = pq.ParquetFile(io.BytesIO(payload))
        available = set(parquet.schema_arrow.names)
        columns = [column for column in desired_columns if column in available]
        for row in parquet.read(columns=columns).to_pylist():
            count_value = row.get("stats/is_for_training/count")
            mean_value = row.get("stats/is_for_training/mean")
            mask_count = int(count_value[0]) if isinstance(count_value, list) and count_value else None
            mask_mean = float(mean_value[0]) if isinstance(mean_value, list) and mean_value else None
            trainable = round(mask_count * mask_mean) if mask_count is not None and mask_mean is not None else None
            episode_rows.append(
                {
                    "repo_id": repo_id,
                    "episode_index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "tasks": row.get("tasks") or [],
                    "training_mask_count": mask_count,
                    "training_mask_mean": mask_mean,
                    "trainable_frames": trainable,
                }
            )

    lengths = [row["length"] for row in episode_rows]
    trainable_values = [row["trainable_frames"] for row in episode_rows if row["trainable_frames"] is not None]
    tasks = sorted({task for row in episode_rows for task in row["tasks"]})
    summary = {
        "repo_id": repo_id,
        "sampling_weight": weight,
        "hub_revision": revision,
        "hub_private": hub.get("private"),
        "hub_created_at": hub.get("createdAt"),
        "hub_last_modified": hub.get("lastModified"),
        "hub_used_storage_bytes": hub.get("usedStorage"),
        "info_codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "declared_total_episodes": info.get("total_episodes"),
        "declared_total_frames": info.get("total_frames"),
        "declared_total_tasks": info.get("total_tasks"),
        "splits": info.get("splits"),
        "episode_metadata_rows": len(episode_rows),
        "episode_length_frames": quantiles(lengths),
        "episode_length_seconds": quantiles([round(value / info["fps"]) for value in lengths]) if info.get("fps") else {},
        "trainable_frames_from_episode_metadata": sum(trainable_values) if len(trainable_values) == len(episode_rows) else None,
        "excluded_frames_from_episode_metadata": (
            sum(lengths) - sum(trainable_values) if len(trainable_values) == len(episode_rows) else None
        ),
        "unique_tasks": tasks,
        "features": feature_summary(info),
        "cross_checks": {
            "episode_rows_match_declared": len(episode_rows) == info.get("total_episodes"),
            "length_sum_matches_declared_frames": sum(lengths) == info.get("total_frames"),
            "training_mask_count_matches_length": all(
                row["training_mask_count"] in (None, row["length"]) for row in episode_rows
            ),
        },
    }
    return summary, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-audit", type=Path, required=True)
    parser.add_argument("--wandb-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    args = parser.parse_args()

    recipe_audit = json.loads(args.recipe_audit.read_text())
    recipe_source = "current code"
    recipe_config = recipe_audit["training_data_recipe"]["config"]["name"]
    weighted = recipe_audit["training_data_recipe"]["repository_weights"]
    if args.wandb_audit is not None:
        wandb_audit = json.loads(args.wandb_audit.read_text())
        repo_ids = wandb_audit["run"]["config"]["data"]["repo_ids"]
        weighted = [
            {"repo_id": repo_id, "weight": count}
            for repo_id, count in Counter(repo_ids).most_common()
        ]
        recipe_source = "W&B immutable run configuration for the deployed checkpoint's training run"
        recipe_config = wandb_audit["run"]["config"]["name"]
    session = requests.Session()
    session.headers["User-Agent"] = "aloha-technical-report-readonly-audit/1.0"

    repositories = []
    all_episode_rows = []
    failures = []
    for entry in weighted:
        try:
            summary, rows = audit_repo(session, entry["repo_id"], int(entry["weight"]))
            repositories.append(summary)
            all_episode_rows.extend(rows)
        except Exception as exc:
            failures.append({"repo_id": entry["repo_id"], "error": f"{type(exc).__name__}: {exc}"})

    total_episodes = sum(int(repo["declared_total_episodes"] or 0) for repo in repositories)
    total_frames = sum(int(repo["declared_total_frames"] or 0) for repo in repositories)
    trainable = [repo["trainable_frames_from_episode_metadata"] for repo in repositories]
    totals = {
        "unique_repositories_audited": len(repositories),
        "repositories_failed": len(failures),
        "unique_episodes": total_episodes,
        "unique_frames": total_frames,
        "unique_duration_sec_at_declared_fps": sum(
            repo["declared_total_frames"] / repo["fps"]
            for repo in repositories
            if repo.get("declared_total_frames") is not None and repo.get("fps")
        ),
        "trainable_frames": sum(trainable) if all(value is not None for value in trainable) else None,
        "sampling_entries_after_weights": sum(repo["sampling_weight"] for repo in repositories),
        "weighted_episode_exposure": sum(repo["declared_total_episodes"] * repo["sampling_weight"] for repo in repositories),
        "weighted_frame_exposure": sum(repo["declared_total_frames"] * repo["sampling_weight"] for repo in repositories),
    }
    camera_set_counts = Counter(
        tuple(camera["key"] for camera in repo["features"]["camera_features"]) for repo in repositories
    )
    result = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Public Hugging Face Hub dataset API and immutable revision URLs",
        "recipe_source": recipe_source,
        "recipe_config": recipe_config,
        "totals": totals,
        "camera_set_counts": [
            {"camera_keys": list(keys), "repository_count": count}
            for keys, count in camera_set_counts.items()
        ],
        "repositories": repositories,
        "failures": failures,
        "interpretation_limits": [
            "Unique counts count each repository once; weighted exposure is not additional collected data.",
            "The Hub metadata proves dataset contents, not that every frame was sampled equally during training.",
            "Repository names are not used as success labels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    args.episodes_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_csv.open("w", encoding="utf-8", newline="") as stream:
        stream.write("repo_id,episode_index,length,trainable_frames,tasks\n")
        for row in all_episode_rows:
            tasks = " | ".join(row["tasks"]).replace('"', '""')
            stream.write(
                f'"{row["repo_id"]}",{row["episode_index"]},{row["length"]},'
                f'{"" if row["trainable_frames"] is None else row["trainable_frames"]},"{tasks}"\n'
            )


if __name__ == "__main__":
    main()

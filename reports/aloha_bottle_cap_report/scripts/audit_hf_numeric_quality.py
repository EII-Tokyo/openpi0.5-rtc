#!/usr/bin/env python3
"""Numerically inspect state/action/timing columns for the exact training data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import requests


RAW_BASE = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def data_files(repo_id: str, revision: str, session: requests.Session) -> list[str]:
    response = session.get(f"https://huggingface.co/api/datasets/{repo_id}", timeout=45)
    response.raise_for_status()
    return sorted(
        row["rfilename"]
        for row in response.json().get("siblings", [])
        if row.get("rfilename", "").startswith("data/")
        and row["rfilename"].endswith(".parquet")
    )


def load_table(repo_id: str, revision: str, path: str, session: requests.Session) -> Any:
    response = session.get(
        RAW_BASE.format(repo_id=repo_id, revision=revision, path=path),
        timeout=120,
    )
    response.raise_for_status()
    parquet = pq.ParquetFile(io.BytesIO(response.content))
    wanted = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "is_for_training",
    ]
    missing = [key for key in wanted if key not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{repo_id}/{path} missing columns: {missing}")
    return parquet.read(columns=wanted), len(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_audit = json.loads(args.dataset_audit.read_text())
    session = requests.Session()
    session.headers["User-Agent"] = "aloha-technical-report-readonly-quality-audit/1.0"

    totals = Counter()
    repository_rows = []
    trajectory_hashes: dict[str, list[str]] = defaultdict(list)
    failures = []

    for repo in dataset_audit["repositories"]:
        repo_id = repo["repo_id"]
        revision = repo["hub_revision"]
        repo_counts = Counter()
        repo_bytes = 0
        timing_bad_episodes: set[int] = set()
        frame_index_bad_episodes: set[int] = set()
        try:
            for path in data_files(repo_id, revision, session):
                table, payload_bytes = load_table(repo_id, revision, path, session)
                repo_bytes += payload_bytes
                state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
                action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
                frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
                episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
                training = np.asarray(table["is_for_training"].to_pylist(), dtype=bool)

                rows = len(table)
                repo_counts["rows"] += rows
                repo_counts["trainable_rows"] += int(training.sum())
                repo_counts["nonfinite_state_values"] += int((~np.isfinite(state)).sum())
                repo_counts["nonfinite_action_values"] += int((~np.isfinite(action)).sum())
                repo_counts["all_zero_action_rows"] += int(np.all(action == 0.0, axis=1).sum())
                repo_counts["all_zero_state_rows"] += int(np.all(state == 0.0, axis=1).sum())

                for episode in np.unique(episode_index):
                    mask = episode_index == episode
                    order = np.argsort(frame_index[mask], kind="stable")
                    episode_frames = frame_index[mask][order]
                    episode_times = timestamp[mask][order]
                    episode_state = np.ascontiguousarray(state[mask][order])
                    episode_action = np.ascontiguousarray(action[mask][order])
                    episode_training = np.ascontiguousarray(training[mask][order])
                    if not np.array_equal(episode_frames, np.arange(len(episode_frames))):
                        frame_index_bad_episodes.add(int(episode))
                    expected = episode_frames / float(repo["fps"])
                    if not np.allclose(episode_times, expected, atol=2e-5, rtol=0):
                        timing_bad_episodes.add(int(episode))
                    digest = hashlib.sha256()
                    digest.update(episode_state.tobytes())
                    digest.update(episode_action.tobytes())
                    digest.update(episode_training.tobytes())
                    trajectory_hashes[digest.hexdigest()].append(f"{repo_id}#{int(episode)}")
                    repo_counts["episodes_seen"] += 1

            repo_counts["timing_bad_episodes"] = len(timing_bad_episodes)
            repo_counts["frame_index_bad_episodes"] = len(frame_index_bad_episodes)
            repo_counts["downloaded_parquet_bytes"] = repo_bytes
            repository_rows.append({"repo_id": repo_id, **dict(repo_counts)})
            totals.update(repo_counts)
        except Exception as exc:
            failures.append({"repo_id": repo_id, "error": f"{type(exc).__name__}: {exc}"})

    duplicate_groups = [
        {"sha256": digest, "members": members, "count": len(members)}
        for digest, members in trajectory_hashes.items()
        if len(members) > 1
    ]
    duplicate_groups.sort(key=lambda row: (-row["count"], row["sha256"]))
    duplicate_members = sum(row["count"] for row in duplicate_groups)

    result = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Numeric Parquet columns only: state, action, timestamp, frame index, episode index, training mask.",
        "totals": dict(totals),
        "repositories": repository_rows,
        "failures": failures,
        "exact_duplicate_trajectory_groups": duplicate_groups,
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_member_count": duplicate_members,
        "checks_not_performed": [
            "Exhaustive video decode and visual corruption scan",
            "Semantic duplicate detection under time shift or small numeric perturbation",
            "Robot task success-label validation (no success field in audited features)",
            "Train/validation/test leakage audit (all repositories declare train-only splits)",
        ],
        "interpretation_limits": [
            "All-zero action rows are reported as a data characteristic; they are not automatically errors.",
            "Exact hashes include state, action, and training mask and only detect byte-identical trajectories.",
            "Timestamp checks use the declared per-repository frame rate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

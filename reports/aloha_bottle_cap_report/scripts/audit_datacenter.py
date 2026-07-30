#!/usr/bin/env python3
"""Collect read-only aggregate evidence from the production data center."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_PROGRAM = r'''
import json
from app.mongo_client import get_collection

projects = get_collection("projects")
episodes = get_collection("episodes")

project_docs = list(projects.find(
    {"type": "aloha", "is_deleted": {"$ne": True}},
    {
        "_id": 1,
        "name": 1,
        "fps": 1,
        "status": 1,
        "created_at": 1,
        "episode_count": 1,
        "total_data_count": 1,
        "total_duration": 1,
    },
).sort("created_at", 1))

active_ids = [str(row["_id"]) for row in project_docs]
active_episode_query = {
    "type": "aloha_episode",
    "project_id": {"$in": active_ids},
    "is_deleted": {"$ne": True},
}

episode_totals = list(episodes.aggregate([
    {"$match": active_episode_query},
    {"$group": {
        "_id": None,
        "episode_documents": {"$sum": 1},
        "frames": {"$sum": {"$ifNull": ["$data_count", 0]}},
        "duration_sec": {"$sum": {"$ifNull": ["$duration", 0]}},
    }},
]))
episode_totals = episode_totals[0] if episode_totals else {
    "episode_documents": 0,
    "frames": 0,
    "duration_sec": 0,
}
episode_totals.pop("_id", None)

by_project = {
    row["_id"]: row
    for row in episodes.aggregate([
        {"$match": active_episode_query},
        {"$group": {
            "_id": "$project_id",
            "episode_documents": {"$sum": 1},
            "frames": {"$sum": {"$ifNull": ["$data_count", 0]}},
            "duration_sec": {"$sum": {"$ifNull": ["$duration", 0]}},
        }},
    ])
}

rows = []
for project in project_docs:
    project_id = str(project["_id"])
    observed = by_project.get(project_id, {})
    declared_episodes = int(project.get("episode_count") or 0)
    observed_episodes = int(observed.get("episode_documents") or 0)
    declared_frames = int(project.get("total_data_count") or 0)
    observed_frames = int(observed.get("frames") or 0)
    rows.append({
        "project_id": project_id,
        "name": project.get("name"),
        "fps": project.get("fps"),
        "status": project.get("status"),
        "created_at": project.get("created_at"),
        "declared_episode_count": declared_episodes,
        "observed_episode_documents": observed_episodes,
        "episode_count_difference": observed_episodes - declared_episodes,
        "declared_frame_count": declared_frames,
        "observed_frame_count": observed_frames,
        "frame_count_difference": observed_frames - declared_frames,
        "declared_duration_sec": float(project.get("total_duration") or 0),
        "observed_duration_sec": float(observed.get("duration_sec") or 0),
    })

project_totals = {
    "projects": len(project_docs),
    "declared_episodes": sum(row["declared_episode_count"] for row in rows),
    "declared_frames": sum(row["declared_frame_count"] for row in rows),
    "declared_duration_sec": sum(row["declared_duration_sec"] for row in rows),
}

unmatched_episode_totals = list(episodes.aggregate([
    {"$match": {
        "type": "aloha_episode",
        "project_id": {"$nin": active_ids},
        "is_deleted": {"$ne": True},
    }},
    {"$group": {
        "_id": None,
        "episode_documents": {"$sum": 1},
        "frames": {"$sum": {"$ifNull": ["$data_count", 0]}},
        "duration_sec": {"$sum": {"$ifNull": ["$duration", 0]}},
    }},
]))
unmatched_episode_totals = unmatched_episode_totals[0] if unmatched_episode_totals else {
    "episode_documents": 0,
    "frames": 0,
    "duration_sec": 0,
}
unmatched_episode_totals.pop("_id", None)

print(json.dumps({
    "project_totals": project_totals,
    "active_project_episode_totals": episode_totals,
    "episode_documents_outside_active_projects": unmatched_episode_totals,
    "projects": rows,
}, ensure_ascii=False, default=str))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="datacenter")
    parser.add_argument("--container", default="eii-data-system-prod-backend-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    remote_command = " ".join(
        [
            "docker",
            "exec",
            shlex.quote(args.container),
            "python3",
            "-c",
            shlex.quote(f"exec({REMOTE_PROGRAM!r})"),
        ]
    )
    command = ["ssh", args.host, remote_command]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    payload["audit_generated_utc"] = datetime.now(timezone.utc).isoformat()
    payload["scope"] = (
        "Active, non-deleted ALOHA projects and non-deleted ALOHA episode documents "
        "whose project_id belongs to those projects."
    )
    payload["read_only"] = True
    payload["limitations"] = [
        "Project summary counters and episode-document aggregates are reported separately.",
        "No S3 objects or image/video payloads were opened by this audit.",
        "Project membership in a specific training recipe requires a separate mapping audit.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

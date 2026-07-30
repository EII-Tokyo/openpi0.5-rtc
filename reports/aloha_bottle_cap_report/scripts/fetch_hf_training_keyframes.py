#!/usr/bin/env python3
"""Extract representative frames from immutable Hugging Face training videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import pyarrow.parquet as pq
import requests


SELECTIONS = [
    {
        "label": "ordinary_full_task",
        "repo_id": "lyl472324464/2025-12-10-twist-one-bottle-merged-adjust-pickup",
    },
    {
        "label": "no_cap",
        "repo_id": "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse-merged-adjust-pickup",
    },
    {
        "label": "direction",
        "repo_id": "lyl472324464/2026-04-21_direction-lerobot-without-rinse-merged-adjust-pickup",
    },
    {
        "label": "turn_over",
        "repo_id": "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    },
    {
        "label": "free_spinning",
        "repo_id": "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    },
    {
        "label": "return_home",
        "repo_id": "lyl472324464/2026-05-12_twist2-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    },
]
FRACTIONS = [0.2, 0.5, 0.8]
CAMERA = "observation.images.cam_high"


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def ffmpeg_frame(url: str, timestamp: float) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            url,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "aloha-technical-report-readonly-keyframes/1.0"
    records = []

    for selection in SELECTIONS:
        repo_id = selection["repo_id"]
        hub = get_json(session, f"https://huggingface.co/api/datasets/{repo_id}")
        revision = hub["sha"]
        info_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/meta/info.json"
        info = get_json(session, info_url)
        episode_path = next(
            sibling["rfilename"]
            for sibling in hub["siblings"]
            if sibling["rfilename"].startswith("meta/episodes/")
            and sibling["rfilename"].endswith(".parquet")
        )
        episode_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{episode_path}"
        episode_response = session.get(episode_url, timeout=60)
        episode_response.raise_for_status()
        table = pq.read_table(
            io.BytesIO(episode_response.content),
            columns=[
                "episode_index",
                "length",
                f"videos/{CAMERA}/chunk_index",
                f"videos/{CAMERA}/file_index",
                f"videos/{CAMERA}/from_timestamp",
                f"videos/{CAMERA}/to_timestamp",
            ],
        )
        rows = table.to_pylist()
        row = rows[len(rows) // 2]
        video_path = info["video_path"].format(
            video_key=CAMERA,
            chunk_index=row[f"videos/{CAMERA}/chunk_index"],
            file_index=row[f"videos/{CAMERA}/file_index"],
        )
        video_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{video_path}"
        start = float(row[f"videos/{CAMERA}/from_timestamp"])
        end = float(row[f"videos/{CAMERA}/to_timestamp"])
        for fraction in FRACTIONS:
            timestamp = start + (end - start) * fraction
            filename = (
                f"{selection['label']}__episode_{int(row['episode_index']):04d}"
                f"__p{int(fraction * 100):02d}.png"
            )
            output = args.output_dir / filename
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(ffmpeg_frame(video_url, timestamp))
            records.append(
                {
                    "label": selection["label"],
                    "repo_id": repo_id,
                    "hub_revision": revision,
                    "episode_index": int(row["episode_index"]),
                    "episode_length_frames": int(row["length"]),
                    "camera": CAMERA,
                    "fraction": fraction,
                    "video_timestamp_sec": timestamp,
                    "source_video_path": video_path,
                    "output_file": str(output),
                    "evidence_role": "training demonstration example, not autonomous evaluation",
                }
            )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "For six question-driven condition categories, choose the median episode "
            "and frames at 20%, 50%, and 80% of its top-camera interval."
        ),
        "records": records,
        "limitations": [
            "Uniform temporal positions are review candidates, not automatically key task events.",
            "Frames are from training demonstrations and cannot prove autonomous policy success.",
            "Final report inclusion requires visual review and a condition-relevant explanation.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

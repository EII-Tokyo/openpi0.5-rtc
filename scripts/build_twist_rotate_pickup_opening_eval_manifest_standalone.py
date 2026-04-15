#!/usr/bin/env python3
"""Build a rotate/pick-up opening-stage eval manifest without project config deps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


REPO_IDS = [
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
]

TARGET_PAIRS = {
    ("Bottle on table, opening faces left", "Rotate so opening faces right"): 0,
    ("Bottle on table, opening faces right", "Pick up with left hand"): 1,
}


def parse_pair(cell: object):
    if cell is None:
        return None
    text = cell if isinstance(cell, str) else str(cell)
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    bs = str(obj.get("bottle_state") or "").strip()
    st = str(obj.get("subtask") or "").strip()
    if not bs or not st:
        return None
    return bs, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames-per-episode", type=int, default=50)
    args = ap.parse_args()

    rows = []
    per_repo = Counter()
    per_class = Counter()
    selected_eps = 0

    for repo_id in REPO_IDS:
        print(f"Scanning {repo_id}", flush=True)
        ds = load_dataset(
            "parquet",
            data_files=f"hf://datasets/{repo_id}/data/*/*.parquet",
            split="train",
        )
        subtask_col = ds.select_columns(["subtask", "episode_index"])
        seen_per_episode = Counter()
        added_eps = set()
        for idx in range(len(subtask_col)):
            row = subtask_col[int(idx)]
            ep = int(row["episode_index"])
            if seen_per_episode[ep] >= args.frames_per_episode:
                continue
            seen_per_episode[ep] += 1
            pair = parse_pair(row["subtask"])
            if pair is None or pair not in TARGET_PAIRS:
                continue
            cid = TARGET_PAIRS[pair]
            rows.append(
                {
                    "repo_id": repo_id,
                    "frame_index": int(idx),
                    "episode_index": ep,
                    "class_id": cid,
                    "bottle_state": pair[0],
                    "subtask": pair[1],
                }
            )
            per_repo[repo_id] += 1
            per_class[cid] += 1
            added_eps.add(ep)
        selected_eps += len(added_eps)

    rows.sort(key=lambda r: (r["repo_id"], r["episode_index"], r["frame_index"]))
    payload = {
        "summary": {
            "num_repos": len(REPO_IDS),
            "selected_rows": len(rows),
            "selected_episodes": selected_eps,
            "frames_per_episode": args.frames_per_episode,
            "per_repo_counts": dict(per_repo),
            "per_class_counts": dict(per_class),
        },
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved manifest to: {out}")


if __name__ == "__main__":
    main()

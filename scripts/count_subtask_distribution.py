#!/usr/bin/env python3
"""Count subtask / bottle_state distribution across LeRobot repos (matches a TrainConfig).

Examples:
  uv run scripts/count_subtask_distribution.py --config-name twist_and_static_mixture_full_finetune
  uv run scripts/count_subtask_distribution.py --repo-id lyl472324464/2026-03-04-one-direction

Requires HuggingFace dataset cache or network to download parquet (videos are skipped).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi.training import config as openpi_config


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, str):
        return x
    if hasattr(x, "item"):
        try:
            return _as_str(x.item())
        except Exception:
            pass
    return str(x)


def _parse_json_subtask(raw: str) -> tuple[str, str, str]:
    """Returns (subtask_field, bottle_state_field, raw_or_fallback)."""
    s = raw.strip()
    if not s:
        return "", "", "<empty>"
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return "", "", s[:200] + ("…" if len(s) > 200 else "")
    if not isinstance(d, dict):
        return "", "", s[:200] + ("…" if len(s) > 200 else "")
    st = d.get("subtask")
    bs = d.get("bottle_state")
    st_s = st.strip() if isinstance(st, str) else ""
    bs_s = bs.strip() if isinstance(bs, str) else ""
    return st_s, bs_s, "<json>"


@dataclass
class Args:
    config_name: str | None = None
    """TrainConfig name (uses its data.repo_ids)."""

    repo_id: str | None = None
    """If set, only scan this single repo (ignores config_name)."""


def _count_repo(repo_id: str) -> dict[str, Any]:
    logging.info("Loading %s (parquet only, no videos)...", repo_id)
    ds = LeRobotDataset(
        repo_id,
        revision="main",
        force_cache_sync=True,
        download_videos=False,
        delta_timestamps=None,
    )
    hf = ds.hf_dataset
    n = len(hf)
    if "subtask" not in hf.column_names:
        return {
            "repo_id": repo_id,
            "num_frames": n,
            "error": "no 'subtask' column",
        }

    subtask_counts: Counter[str] = Counter()
    bottle_state_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    empty_frames = 0
    non_json_plain = 0

    # Columnar scan (loads subtask only per batch)
    batch_size = 8192
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        col = hf[start:end]["subtask"]
        for raw in col:
            s = _as_str(raw).strip()
            if not s:
                empty_frames += 1
                subtask_counts["<empty>"] += 1
                continue
            st, bs, fallback = _parse_json_subtask(s)
            if st:
                subtask_counts[st] += 1
            elif fallback == "<json>":
                subtask_counts["<json_without_subtask_field>"] += 1
            else:
                non_json_plain += 1
                subtask_counts[f"<plain:{fallback[:80]}>"] += 1
            if bs:
                bottle_state_counts[bs] += 1
            if st and bs:
                pair_counts[f"{bs} || {st}"] += 1

    return {
        "repo_id": repo_id,
        "num_frames": n,
        "empty_subtask_frames": empty_frames,
        "non_json_subtask_strings": non_json_plain,
        "subtask_counts": subtask_counts,
        "bottle_state_counts": bottle_state_counts,
        "pair_counts": pair_counts,
    }


def _print_counter(title: str, c: Counter[str], *, limit: int = 30) -> None:
    print(f"\n=== {title} ===")
    if not c:
        print("(none)")
        return
    total = sum(c.values())
    for k, v in c.most_common(limit):
        pct = 100.0 * v / total if total else 0.0
        print(f"  {v:8d}  ({pct:5.1f}%)  {k}")
    if len(c) > limit:
        print(f"  ... {len(c) - limit} more keys")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.repo_id:
        repo_ids = [args.repo_id]
    elif args.config_name:
        cfg = openpi_config.get_config(args.config_name)
        data = cfg.data
        repo_ids = getattr(data, "repo_ids", None)
        if not repo_ids:
            raise SystemExit(f"Config {args.config_name!r} has no data.repo_ids")
    else:
        raise SystemExit("Pass --config-name or --repo-id")

    global_sub: Counter[str] = Counter()
    global_bs: Counter[str] = Counter()
    global_pair: Counter[str] = Counter()
    total_frames_labeled = 0
    total_frames_no_column = 0

    per_repo: list[dict[str, Any]] = []
    for rid in repo_ids:
        r = _count_repo(rid)
        per_repo.append(r)
        if "error" in r:
            print(f"\n[{rid}] ERROR: {r['error']} (frames={r.get('num_frames', 0)})")
            if r.get("error") == "no 'subtask' column":
                total_frames_no_column += int(r.get("num_frames", 0))
            continue
        total_frames_labeled += int(r["num_frames"])
        global_sub.update(r["subtask_counts"])
        global_bs.update(r["bottle_state_counts"])
        global_pair.update(r["pair_counts"])

    total_all = total_frames_labeled + total_frames_no_column
    print(f"\nConfigs/repos scanned: {len(repo_ids)}")
    print(
        f"Frames in repos WITH subtask column (used in stats below): {total_frames_labeled}\n"
        f"Frames in repos WITHOUT subtask column (training uses empty default): {total_frames_no_column}\n"
        f"Approx. total frames if all repos mixed in dataloader: {total_all}"
    )
    _print_counter("GLOBAL: subtask field (parsed JSON .subtask, else plain)", global_sub, limit=40)
    _print_counter("GLOBAL: bottle_state field", global_bs, limit=25)
    _print_counter("GLOBAL: (bottle_state || subtask) pairs", global_pair, limit=25)

    print("\n=== Per-repo: top subtasks (up to 8 each) ===")
    for r in per_repo:
        rid = r.get("repo_id", "?")
        if "error" in r:
            continue
        sc: Counter[str] = r["subtask_counts"]
        print(f"\n-- {rid}  (n={r['num_frames']}, empty={r['empty_subtask_frames']})")
        for k, v in sc.most_common(8):
            pct = 100.0 * v / r["num_frames"] if r["num_frames"] else 0
            print(f"     {v:7d} ({pct:4.1f}%)  {k}")


if __name__ == "__main__":
    main(tyro.cli(Args))

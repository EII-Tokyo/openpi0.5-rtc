#!/usr/bin/env python3
"""Reorganize clean RLT shards from manual/ into date-based task folders.

The script copies files instead of moving them, verifies hashes, and writes a
new manifest with updated shard paths. It never overwrites different content.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np


DEFAULT_REPLAY_ROOT = pathlib.Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean")
DEFAULT_MANIFEST = pathlib.Path("local_rlt_manifests/trainable_clean_committed_20260623.jsonl")
DEFAULT_TASK = "twist_off_the_bottle_cap"
DEFAULT_TIMEZONE = "Asia/Tokyo"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_npz_manifest(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        if "manifest" not in data.files:
            return {}
        value = data["manifest"]
        if value.shape == ():
            raw = value.item()
        else:
            raw = value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def _transition_count(path: pathlib.Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        for key in ("z_rl", "action", "reward_seq", "proprio"):
            if key in data.files:
                return int(data[key].shape[0])
    return 0


def _reward_from(path: pathlib.Path, manifest: dict, manifest_row: dict) -> int:
    if "reward" in manifest:
        return int(float(manifest["reward"]) > 0.0)
    if "reward" in manifest_row:
        return int(float(manifest_row["reward"]) > 0.0)
    with np.load(path, allow_pickle=True) as data:
        if "reward_seq" in data.files:
            return int(float(np.asarray(data["reward_seq"]).sum()) > 0.0)
    return 0


def _timestamp_from(path: pathlib.Path, manifest: dict, timezone: ZoneInfo) -> tuple[float, str]:
    for key in ("start_time", "score_time"):
        value = manifest.get(key)
        if value is not None:
            try:
                return float(value), f"manifest.{key}"
            except (TypeError, ValueError):
                pass
    matches = re.findall(r"crop_(\d{13})", path.name)
    if matches:
        return float(matches[-1]) / 1000.0, "filename.crop_timestamp_ms"
    for part in path.parts:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            local = dt.datetime.strptime(part, "%Y-%m-%d").replace(tzinfo=timezone)
            return local.timestamp(), "path.date"
    raise ValueError(f"Cannot infer generation time for {path}")


def _date_from_timestamp(timestamp: float, timezone: ZoneInfo) -> str:
    return dt.datetime.fromtimestamp(timestamp, timezone).date().isoformat()


def _manual_target_path(path: pathlib.Path, replay_root: pathlib.Path, task: str, date: str) -> pathlib.Path:
    return replay_root / task / date / "shards" / path.name


def _is_manual_path(path: pathlib.Path, replay_root: pathlib.Path) -> bool:
    try:
        rel = path.relative_to(replay_root)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[0] == "manual"


def build_plan(
    *,
    manifest_path: pathlib.Path,
    replay_root: pathlib.Path,
    task: str,
    timezone_name: str,
) -> tuple[list[dict], list[dict], dict]:
    timezone = ZoneInfo(timezone_name)
    rows = _read_jsonl(manifest_path)
    move_plan: list[dict] = []
    new_rows: list[dict] = []
    seen_new_paths: set[str] = set()

    for row in rows:
        old_path = pathlib.Path(row["shard_path"]).expanduser().resolve()
        new_row = dict(row)
        if _is_manual_path(old_path, replay_root):
            npz_manifest = _load_npz_manifest(old_path)
            timestamp, time_source = _timestamp_from(old_path, npz_manifest, timezone)
            date = _date_from_timestamp(timestamp, timezone)
            new_path = _manual_target_path(old_path, replay_root, task, date)
            reward = _reward_from(old_path, npz_manifest, row)
            transitions = _transition_count(old_path)
            plan_row = {
                "old_path": str(old_path),
                "new_path": str(new_path),
                "date": date,
                "timestamp": timestamp,
                "time_source": time_source,
                "reward": reward,
                "num_replay_transitions": transitions,
                "key_region_id": npz_manifest.get("key_region_id") or row.get("key_region_id"),
                "source_shard_path": npz_manifest.get("source_shard_path") or row.get("source_shard_path"),
            }
            move_plan.append(plan_row)
            new_row["shard_path"] = str(new_path)
            new_row["source_shard_path"] = row.get("source_shard_path", str(old_path))
            new_row["batch"] = date
            new_row["time_sorted_from"] = str(old_path)
            new_row["time_sort_source"] = time_source
            new_row["reward"] = reward
            new_row["num_replay_transitions"] = transitions
            new_row["success_episodes"] = int(reward > 0)
            new_row["failure_episodes"] = int(reward == 0)

        path_key = str(pathlib.Path(new_row["shard_path"]).expanduser().resolve())
        if path_key in seen_new_paths:
            raise ValueError(f"Duplicate target manifest path: {path_key}")
        seen_new_paths.add(path_key)
        new_rows.append(new_row)

    summary = summarize(new_rows, move_plan)
    return move_plan, new_rows, summary


def summarize(manifest_rows: list[dict], move_plan: list[dict]) -> dict:
    by_date: dict[str, dict[str, int]] = {}
    total_success = 0
    total_failure = 0
    total_transitions = 0
    for row in manifest_rows:
        date = row.get("batch")
        if not date:
            path = str(row.get("shard_path", ""))
            match = re.search(r"/(\d{4}-\d{2}-\d{2})/", path)
            date = match.group(1) if match else "unknown"
        reward = int(float(row.get("reward", 0)) > 0.0)
        transitions = int(row.get("num_replay_transitions") or 0)
        bucket = by_date.setdefault(date, {"success": 0, "failure": 0, "total": 0, "transitions": 0})
        if reward:
            bucket["success"] += 1
            total_success += 1
        else:
            bucket["failure"] += 1
            total_failure += 1
        bucket["total"] += 1
        bucket["transitions"] += transitions
        total_transitions += transitions
    return {
        "num_manifest_rows": len(manifest_rows),
        "num_reorganized_manual_rows": len(move_plan),
        "success_episodes": total_success,
        "failure_episodes": total_failure,
        "total_transitions": total_transitions,
        "date_range": [min(by_date), max(by_date)] if by_date else [None, None],
        "by_date": dict(sorted(by_date.items())),
    }


def validate_targets(move_plan: list[dict], *, execute: bool) -> list[dict]:
    results = []
    for row in move_plan:
        old_path = pathlib.Path(row["old_path"])
        new_path = pathlib.Path(row["new_path"])
        old_hash = _sha256(old_path)
        status = "planned"
        new_hash = None
        if new_path.exists():
            new_hash = _sha256(new_path)
            if new_hash != old_hash:
                raise FileExistsError(f"Target exists with different content: {new_path}")
            status = "exists_same_hash"
        elif execute:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            new_hash = _sha256(new_path)
            if new_hash != old_hash:
                raise IOError(f"Copied hash mismatch: {old_path} -> {new_path}")
            status = "copied"
        results.append({**row, "old_sha256": old_hash, "new_sha256": new_hash, "status": status})
    return results


def write_summary_markdown(path: pathlib.Path, summary: dict, *, execute: bool, output_manifest: pathlib.Path) -> None:
    lines = [
        "# Manual Clean Shard Time Reorganization",
        "",
        f"- Mode: {'execute' if execute else 'dry-run'}",
        f"- Output manifest: `{output_manifest}`",
        f"- Manifest rows: `{summary['num_manifest_rows']}`",
        f"- Reorganized manual rows: `{summary['num_reorganized_manual_rows']}`",
        f"- Success / failure episodes: `{summary['success_episodes']} / {summary['failure_episodes']}`",
        f"- Total transitions: `{summary['total_transitions']}`",
        f"- Date range: `{summary['date_range'][0]} -> {summary['date_range'][1]}`",
        "",
        "| date | success | failure | total | transitions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for date, bucket in summary["by_date"].items():
        lines.append(
            f"| {date} | {bucket['success']} | {bucket['failure']} | {bucket['total']} | {bucket['transitions']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay-root", type=pathlib.Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    replay_root = args.replay_root.resolve()
    output_dir = args.output_dir
    output_manifest = args.output_manifest

    move_plan, new_rows, summary = build_plan(
        manifest_path=manifest_path,
        replay_root=replay_root,
        task=args.task,
        timezone_name=args.timezone,
    )
    copy_results = validate_targets(move_plan, execute=args.execute)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "move_plan.jsonl", move_plan)
    _write_jsonl(output_dir / "copy_results.jsonl", copy_results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    write_summary_markdown(output_dir / "summary.md", summary, execute=args.execute, output_manifest=output_manifest)

    if args.execute:
        _write_jsonl(output_manifest, new_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if not args.execute:
        print("Dry-run only. Re-run with --execute to copy files and write the output manifest.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

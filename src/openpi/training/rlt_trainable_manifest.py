from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import sqlite3
from collections import Counter
from typing import Any

import numpy as np


REQUIRED_REPLAY_KEYS: tuple[str, ...] = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


@dataclasses.dataclass(frozen=True)
class TrainableManifestSummary:
    num_shards: int
    num_transitions: int
    success_episodes: int
    failure_episodes: int
    by_batch: dict[str, dict[str, int]]


@dataclasses.dataclass(frozen=True)
class BuildManifestResult:
    output_path: pathlib.Path
    summary: TrainableManifestSummary
    skipped_by_reason: dict[str, int]


def resolve_replay_path(path: str | pathlib.Path, *, clean_root: pathlib.Path | str | None = None) -> pathlib.Path:
    """Resolve a replay shard path that may have been recorded inside the container."""

    candidate = pathlib.Path(path)
    if clean_root is not None and candidate.is_absolute():
        clean_root_path = pathlib.Path(clean_root)
        container_prefix = pathlib.PurePosixPath("/app/replay/rlt_key_regions_clean")
        try:
            rel = pathlib.PurePosixPath(candidate.as_posix()).relative_to(container_prefix)
        except ValueError:
            pass
        else:
            return (clean_root_path / pathlib.Path(*rel.parts)).resolve()
    return candidate.expanduser().resolve()


def read_manifest_paths(manifest_path: pathlib.Path | str) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    with pathlib.Path(manifest_path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            shard_path = row.get("shard_path")
            if shard_path:
                paths.append(pathlib.Path(shard_path).expanduser().resolve())
    return paths


def summarize_manifest(manifest_path: pathlib.Path | str) -> TrainableManifestSummary:
    by_batch: dict[str, Counter[str]] = {}
    total = Counter()
    with pathlib.Path(manifest_path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            batch = str(row.get("batch") or "unknown")
            bucket = by_batch.setdefault(batch, Counter())
            _accumulate(bucket, row)
            _accumulate(total, row)
    return _summary_from_counters(total, by_batch)


def build_manifest_from_segment_db(
    segment_db_path: pathlib.Path | str,
    *,
    output_path: pathlib.Path | str,
    clean_root: pathlib.Path | str | None = None,
) -> BuildManifestResult:
    rows = _committed_segment_rows(pathlib.Path(segment_db_path))
    manifest_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    by_batch: dict[str, Counter[str]] = {}
    total = Counter()
    seen_paths: set[pathlib.Path] = set()

    for row in rows:
        source_shard_path = str(row["shard_path"] or "")
        shard_path = resolve_replay_path(source_shard_path, clean_root=clean_root)
        if shard_path in seen_paths:
            skipped["duplicate_path"] += 1
            continue
        if not shard_path.exists():
            skipped["missing_file"] += 1
            continue
        try:
            shard_info = _inspect_shard(shard_path)
        except (OSError, KeyError, ValueError) as exc:
            skipped["invalid_shard"] += 1
            continue

        seen_paths.add(shard_path)
        batch = _batch_from_path(source_shard_path, shard_path)
        manifest_row = {
            "key_region_id": row["key_region_id"],
            "batch": batch,
            "phase": row["phase"],
            "reward": row["reward"],
            "shard_path": str(shard_path),
            "source_shard_path": source_shard_path,
            "num_replay_transitions": shard_info["num_transitions"],
            "success_episodes": shard_info["success_episodes"],
            "failure_episodes": shard_info["failure_episodes"],
            "updated_at": row["updated_at"],
        }
        manifest_rows.append(manifest_row)
        bucket = by_batch.setdefault(batch, Counter())
        _accumulate(bucket, manifest_row)
        _accumulate(total, manifest_row)

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return BuildManifestResult(
        output_path=output,
        summary=_summary_from_counters(total, by_batch),
        skipped_by_reason=dict(sorted(skipped.items())),
    )


def _committed_segment_rows(segment_db_path: pathlib.Path) -> list[sqlite3.Row]:
    with sqlite3.connect(segment_db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT key_region_id, status, phase, reward, shard_path, num_replay_transitions, updated_at
            FROM segments
            WHERE status = 'committed' AND shard_path IS NOT NULL
            ORDER BY updated_at, key_region_id
            """
        ).fetchall()


def _inspect_shard(path: pathlib.Path) -> dict[str, int]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_REPLAY_KEYS if key not in data]
        if missing:
            raise ValueError(f"missing required replay keys: {missing}")
        num_transitions = int(len(data["action"]))
        done = np.asarray(data["done"]).astype(np.bool_)
        reward_seq = np.asarray(data["reward_seq"], dtype=np.float32)
        terminal_rewards = reward_seq[done].sum(axis=-1) if np.any(done) else np.asarray([], dtype=np.float32)
        success = int(np.sum(terminal_rewards > 0.0))
        failure = int(np.sum(done) - success)
    return {
        "num_transitions": num_transitions,
        "success_episodes": success,
        "failure_episodes": failure,
    }


def _batch_from_path(source_path: str, resolved_path: pathlib.Path) -> str:
    for path_text in (source_path, str(resolved_path)):
        if "/manual/" in path_text:
            return "manual"
        match = re.search(r"2026-06-\d+", path_text)
        if match:
            return match.group(0)
    return "unknown"


def _accumulate(counter: Counter[str], row: dict[str, Any]) -> None:
    counter["num_shards"] += 1
    counter["num_transitions"] += int(row.get("num_replay_transitions") or 0)
    counter["success_episodes"] += int(row.get("success_episodes") or 0)
    counter["failure_episodes"] += int(row.get("failure_episodes") or 0)


def _summary_from_counters(
    total: Counter[str],
    by_batch: dict[str, Counter[str]],
) -> TrainableManifestSummary:
    return TrainableManifestSummary(
        num_shards=int(total["num_shards"]),
        num_transitions=int(total["num_transitions"]),
        success_episodes=int(total["success_episodes"]),
        failure_episodes=int(total["failure_episodes"]),
        by_batch={batch: {key: int(value) for key, value in sorted(counter.items())} for batch, counter in sorted(by_batch.items())},
    )

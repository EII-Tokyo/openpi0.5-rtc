from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sqlite3
from zoneinfo import ZoneInfo


DEFAULT_DATA_ROOT = Path("/data/openpi0.5-rtc-reward-learning")
DEFAULT_TASK = "twist_off_the_bottle_cap"
DEFAULT_BATCHES = ("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07")
KEY_REGION_ID_RE = re.compile(r"key_region_([A-Za-z0-9_.-]+)")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass
class ArchiveResult:
    archive_root: str
    batches: list[str]
    task: str
    executed: bool
    moved_paths: list[str]
    missing_paths: list[str]
    archived_key_region_ids: list[str]
    deleted_segments: int
    deleted_events: int


def _batch_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), tz=ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def _batch_from_path(path: str | None) -> str | None:
    if not path:
        return None
    for part in Path(path).parts:
        if DATE_RE.fullmatch(part):
            return part
    return None


def _key_region_id_from_path(path: Path) -> str | None:
    match = KEY_REGION_ID_RE.search(path.name)
    if match is None:
        return None
    key_region_id = match.group(1)
    for suffix in (".crop", ".npz", ".json"):
        if suffix in key_region_id:
            key_region_id = key_region_id.split(suffix, 1)[0]
    return key_region_id


def _selected_segment_rows(db_path: Path, batches: set[str]) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute("SELECT * FROM segments"))
    finally:
        conn.close()
    selected = []
    for row in rows:
        row_batch = (
            _batch_from_path(row["shard_path"])
            or _batch_from_timestamp(row["created_at"])
            or _batch_from_timestamp(row["updated_at"])
        )
        if row_batch in batches:
            selected.append(row)
    return selected


def _collect_replay_key_region_ids(paths: list[Path]) -> set[str]:
    key_region_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for shard_path in path.rglob("key_region_*.npz"):
            if key_region_id := _key_region_id_from_path(shard_path):
                key_region_ids.add(key_region_id)
    return key_region_ids


def _move_path(source: Path, destination: Path, *, execute: bool) -> bool:
    if not source.exists():
        return False
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        shutil.move(str(source), str(destination))
    return True


def archive_key_region_batches(
    *,
    data_root: Path,
    archive_root: Path,
    task: str,
    batches: list[str],
    execute: bool,
) -> ArchiveResult:
    data_root = data_root.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    selected_batches = set(batches)
    db_path = data_root / "segment_db/segments.sqlite3"
    state_path = data_root / "segment_db/rlt_control_state.json"

    replay_paths = [data_root / "replay/rlt_key_regions" / task / batch for batch in batches]
    clean_replay_paths = [data_root / "replay/rlt_key_regions_clean" / task / batch for batch in batches]
    rollout_paths = [data_root / "rollouts/key_regions" / task / batch for batch in batches]
    segment_rows = _selected_segment_rows(db_path, selected_batches)
    key_region_ids = {str(row["key_region_id"]) for row in segment_rows}
    key_region_ids.update(_collect_replay_key_region_ids(replay_paths + clean_replay_paths))

    pending_paths = [
        data_root / "replay/rlt_anchor_token_jobs/pending" / f"key_region_{key_region_id}.json"
        for key_region_id in sorted(key_region_ids)
    ]
    candidate_paths = replay_paths + clean_replay_paths + rollout_paths + pending_paths

    moved_paths: list[str] = []
    missing_paths: list[str] = []
    if execute:
        archive_root.mkdir(parents=True, exist_ok=False)
        if db_path.exists():
            (archive_root / "segment_db").mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_path, archive_root / "segment_db/segments.sqlite3.before_archive")
        if state_path.exists():
            (archive_root / "segment_db").mkdir(parents=True, exist_ok=True)
            shutil.copy2(state_path, archive_root / "segment_db/rlt_control_state.json.before_archive")

    for source in candidate_paths:
        destination = archive_root / source.relative_to(data_root)
        if _move_path(source, destination, execute=execute):
            moved_paths.append(str(source))
        else:
            missing_paths.append(str(source))

    deleted_segments = 0
    deleted_events = 0
    if execute and db_path.exists() and key_region_ids:
        placeholders = ",".join("?" for _ in key_region_ids)
        ordered_ids = sorted(key_region_ids)
        conn = sqlite3.connect(db_path)
        try:
            deleted_events = conn.execute(
                f"DELETE FROM segment_events WHERE key_region_id IN ({placeholders})", ordered_ids
            ).rowcount
            deleted_segments = conn.execute(
                f"DELETE FROM segments WHERE key_region_id IN ({placeholders})", ordered_ids
            ).rowcount
            conn.commit()
        finally:
            conn.close()

    result = ArchiveResult(
        archive_root=str(archive_root),
        batches=batches,
        task=task,
        executed=execute,
        moved_paths=moved_paths,
        missing_paths=missing_paths,
        archived_key_region_ids=sorted(key_region_ids),
        deleted_segments=deleted_segments,
        deleted_events=deleted_events,
    )
    if execute:
        (archive_root / "archive_manifest.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return result


def _default_archive_root(data_root: Path) -> Path:
    timestamp = datetime.now(tz=ZoneInfo("Asia/Tokyo")).strftime("%Y%m%dT%H%M%S%z")
    return data_root / "archive" / f"key_regions_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive active 103 key-region batches out of review roots.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--batch", action="append", dest="batches", default=[])
    parser.add_argument("--execute", action="store_true", help="Move files and delete archived rows from active segment DB.")
    args = parser.parse_args()

    batches = args.batches or list(DEFAULT_BATCHES)
    archive_root = args.archive_root or _default_archive_root(args.data_root)
    result = archive_key_region_batches(
        data_root=args.data_root,
        archive_root=archive_root,
        task=args.task,
        batches=batches,
        execute=args.execute,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

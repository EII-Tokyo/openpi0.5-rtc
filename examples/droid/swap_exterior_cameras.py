#!/usr/bin/env python3
"""Swap exterior cameras 1 and 2 in canonical LeRobot v3.0 DROID datasets.

For each target repository this script:
  1. Swaps the video folder contents for exterior_1_left <-> exterior_2_left
  2. Swaps camera_extrinsics.exterior_{1,2}_left columns in all data parquet files
  3. Swaps the relevant entries in meta/stats.json
  4. Swaps ALL per-episode video metadata and stats in meta/episodes parquet files:
       - videos/{camera}/chunk_index, file_index  (which video file holds the episode)
       - videos/{camera}/from_timestamp, to_timestamp  (timing offsets within the file)
       - stats/{camera}/...  (pixel and extrinsics statistics per episode)
  5. Pushes the modified dataset back to HuggingFace Hub

Modes:
  (default)       Full swap of everything above.
  --fix-remainder Swap only the episode columns that are still in their original
                  (unswapped) state: chunk_index, file_index, and stat columns.
                  Use this when from/to_timestamp was already swapped in a prior run
                  but chunk_index/file_index/stats were not (or were double-swapped).

Usage:
    uv run python -m examples.droid.swap_exterior_cameras
    uv run python -m examples.droid.swap_exterior_cameras --dry-run
    uv run python -m examples.droid.swap_exterior_cameras --no-push
    uv run python -m examples.droid.swap_exterior_cameras --repos michios/droid_xxjd_7_canonical
    uv run python -m examples.droid.swap_exterior_cameras --fix-remainder \\
        --repos michios/droid_xxjd_7_canonical michios/droid_xxjd_8_2_canonical michios/droid_xxjd_20260202
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

log = logging.getLogger(__name__)

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ImportError:
    HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

DEFAULT_REPOS = [
    "michios/droid_xxjd_7_canonical",
    "michios/droid_xxjd_8_2_canonical",
    "michios/droid_xxjd_20260202",
    "michios/droid_xxjd_20260421",
]

CAM1 = "exterior_1_left"
CAM2 = "exterior_2_left"
VIDEO_KEY_1 = f"observation.images.{CAM1}"
VIDEO_KEY_2 = f"observation.images.{CAM2}"
EXTRINSICS_KEY_1 = f"camera_extrinsics.{CAM1}"
EXTRINSICS_KEY_2 = f"camera_extrinsics.{CAM2}"

# Episode parquet: per-episode stat column prefixes
EPISODE_STAT_PREFIXES = (
    f"stats/{VIDEO_KEY_1}/",
    f"stats/{VIDEO_KEY_2}/",
    f"stats/{EXTRINSICS_KEY_1}/",
    f"stats/{EXTRINSICS_KEY_2}/",
)

# Episode parquet: video file pointer columns (which file, which time range)
# chunk_index and file_index say which video file the episode lives in;
# from/to_timestamp say where in that file the episode starts and ends.
# All four must travel together with the swapped video files.
_VIDEO_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
EPISODE_VIDEO_COLUMN_PAIRS = tuple(
    (f"videos/{VIDEO_KEY_1}/{f}", f"videos/{VIDEO_KEY_2}/{f}")
    for f in _VIDEO_FIELDS
)
# Subset used in --fix-remainder (timestamps were already swapped; index columns were not)
EPISODE_INDEX_COLUMN_PAIRS = tuple(
    (f"videos/{VIDEO_KEY_1}/{f}", f"videos/{VIDEO_KEY_2}/{f}")
    for f in ("chunk_index", "file_index")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_column_swaps(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    cols = set(df.columns)
    for col1, col2 in pairs:
        if col1 not in cols or col2 not in cols:
            log.warning("    column pair not found, skipping: %s / %s", col1, col2)
            continue
        tmp = df[col1].copy()
        df[col1] = df[col2]
        df[col2] = tmp
    return df


def _stat_swap_pairs(cols: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for col in cols:
        if col.startswith(EPISODE_STAT_PREFIXES[0]):
            cp = col.replace(EPISODE_STAT_PREFIXES[0], EPISODE_STAT_PREFIXES[1], 1)
            if cp in cols:
                pairs.append((col, cp))
        elif col.startswith(EPISODE_STAT_PREFIXES[2]):
            cp = col.replace(EPISODE_STAT_PREFIXES[2], EPISODE_STAT_PREFIXES[3], 1)
            if cp in cols:
                pairs.append((col, cp))
    return pairs


# ---------------------------------------------------------------------------
# Video folder swap
# ---------------------------------------------------------------------------

def swap_video_folders(root: Path, dry_run: bool) -> None:
    cam1_dir = root / "videos" / VIDEO_KEY_1
    cam2_dir = root / "videos" / VIDEO_KEY_2
    tmp_dir  = root / "videos" / "_swap_tmp"
    if not cam1_dir.exists() or not cam2_dir.exists():
        raise FileNotFoundError(f"Expected both camera dirs:\n  {cam1_dir}\n  {cam2_dir}")
    log.info("  [video] Swapping %s <-> %s", VIDEO_KEY_1, VIDEO_KEY_2)
    if not dry_run:
        cam1_dir.rename(tmp_dir)
        cam2_dir.rename(cam1_dir)
        tmp_dir.rename(cam2_dir)


# ---------------------------------------------------------------------------
# Data parquet: swap camera_extrinsics columns
# ---------------------------------------------------------------------------

def swap_data_parquets(root: Path, dry_run: bool) -> None:
    data_dir = root / "data"
    if not data_dir.exists():
        log.warning("  [data] No data/ directory found, skipping")
        return
    for pq in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(pq)
        if EXTRINSICS_KEY_1 not in df.columns or EXTRINSICS_KEY_2 not in df.columns:
            log.warning("  [data] Skipping %s (extrinsics columns missing)", pq.name)
            continue
        log.info("  [data] Swapping extrinsics in %s", pq.relative_to(root))
        if not dry_run:
            tmp = df[EXTRINSICS_KEY_1].copy()
            df[EXTRINSICS_KEY_1] = df[EXTRINSICS_KEY_2]
            df[EXTRINSICS_KEY_2] = tmp
            df.to_parquet(pq, index=False)


# ---------------------------------------------------------------------------
# meta/stats.json
# ---------------------------------------------------------------------------

def swap_stats_json(root: Path, dry_run: bool) -> None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        log.warning("  [stats] meta/stats.json not found, skipping")
        return
    stats = json.loads(stats_path.read_text())
    for k1, k2 in ((VIDEO_KEY_1, VIDEO_KEY_2), (EXTRINSICS_KEY_1, EXTRINSICS_KEY_2)):
        if k1 in stats and k2 in stats:
            log.info("  [stats] Swapping %s <-> %s", k1, k2)
            if not dry_run:
                stats[k1], stats[k2] = stats[k2], stats[k1]
        else:
            log.warning("  [stats] Keys not found: %s / %s", k1, k2)
    if not dry_run:
        stats_path.write_text(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# meta/episodes parquet
# ---------------------------------------------------------------------------

def _process_episodes_parquets(root: Path, dry_run: bool, extra_pairs: list[tuple[str, str]]) -> None:
    """Swap stat columns plus whatever extra_pairs are provided."""
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        log.warning("  [episodes] meta/episodes/ not found, skipping")
        return
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        log.warning("  [episodes] No parquet files under meta/episodes/")
        return
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        pairs = _stat_swap_pairs(df.columns.tolist()) + extra_pairs
        # Deduplicate
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str]] = []
        for p in pairs:
            key = (min(p), max(p))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        log.info("  [episodes] Swapping %d column pair(s) in %s", len(unique), pq.relative_to(root))
        if not dry_run:
            _apply_column_swaps(df, unique)
            df.to_parquet(pq, index=False)


def swap_episodes_parquets(root: Path, dry_run: bool) -> None:
    """Full swap: stats + chunk_index + file_index + from/to_timestamp."""
    _process_episodes_parquets(root, dry_run, list(EPISODE_VIDEO_COLUMN_PAIRS))


def fix_remainder_episodes_parquets(root: Path, dry_run: bool) -> None:
    """Swap only stats + chunk_index + file_index (timestamps already done)."""
    _process_episodes_parquets(root, dry_run, list(EPISODE_INDEX_COLUMN_PAIRS))


# ---------------------------------------------------------------------------
# Hub push
# ---------------------------------------------------------------------------

def push_to_hub(root: Path, repo_id: str, dry_run: bool) -> None:
    log.info("  [hub] Pushing -> %s ...", repo_id)
    if not dry_run:
        HfApi().upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="fix: swap exterior_1_left and exterior_2_left camera footage and extrinsics",
        )
    log.info("  [hub] https://huggingface.co/datasets/%s", repo_id)


# ---------------------------------------------------------------------------
# Per-repo orchestration
# ---------------------------------------------------------------------------

def process_repo(repo_id: str, *, push: bool, dry_run: bool, fix_remainder: bool) -> None:
    root = Path(HF_LEROBOT_HOME) / repo_id
    log.info("")
    log.info("=== %s ===", repo_id)
    log.info("    root: %s", root)
    if not root.exists():
        log.error("  Dataset not found locally at %s -- skipping", root)
        return

    if fix_remainder:
        log.info("  [mode] fix-remainder: swapping chunk_index, file_index, stats only")
        fix_remainder_episodes_parquets(root, dry_run)
    else:
        swap_video_folders(root, dry_run)
        swap_data_parquets(root, dry_run)
        swap_stats_json(root, dry_run)
        swap_episodes_parquets(root, dry_run)

    if push:
        push_to_hub(root, repo_id, dry_run)
    else:
        log.info("  [hub] Skipping push (--no-push)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS, metavar="REPO_ID")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fix-remainder",
        "--timestamps-only",  # back-compat alias
        action="store_true",
        dest="fix_remainder",
        help=(
            "Swap only chunk_index, file_index, and stat columns in episodes parquet. "
            "Use when from/to_timestamp was already swapped but the index columns were not."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.dry_run:
        log.info("DRY RUN -- no files will be modified")
    if args.fix_remainder:
        log.info("FIX-REMAINDER mode")

    for repo_id in args.repos:
        process_repo(repo_id, push=not args.no_push, dry_run=args.dry_run, fix_remainder=args.fix_remainder)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

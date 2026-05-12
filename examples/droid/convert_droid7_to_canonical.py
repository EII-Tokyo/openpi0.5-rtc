#!/usr/bin/env python3
"""Convert michios/droid_xxjd_7 (old LeRobot format, PNG bytes in parquet)
to michios/droid_xxjd_7_2 in canonical LeRobot v3.0 format.

Field mapping:
  joint_position (7,)   -> observation.state.joint_position + part of observation.state
  gripper_position      -> observation.state.gripper_position + part of observation.state
  actions[:7]           -> action.joint_position
  actions[7]            -> action.gripper_position
  actions (8,)          -> action / action.source_joint_velocity_gripper
  task_index            -> language_instruction (via tasks.parquet lookup)
  Cartesian/velocity fields -> zeros (not in source)
  datetime              -> None
  building              -> EII
  collector_id          -> ed198318
  conveyor_speed        -> 0
  camera_extrinsics     -> zeros

Usage:
    uv run python -m examples.droid.convert_droid7_to_canonical
    uv run python -m examples.droid.convert_droid7_to_canonical --push-to-hub
    uv run python -m examples.droid.convert_droid7_to_canonical --overwrite --push-to-hub
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

from examples.droid.canonical_lerobot import CanonicalDatasetWriter, ensure_scalar_array

SOURCE_REPO_ID = "michios/droid_xxjd_7"
DEST_REPO_ID = "michios/droid_xxjd_7_2"
BUILDING = "EII"
COLLECTOR_ID = "ed198318"

log = logging.getLogger(__name__)


def decode_image(img_dict: dict) -> np.ndarray:
    """Decode HuggingFace image dict {'bytes': ..., 'path': ...} to numpy (H, W, 3) uint8."""
    return np.array(Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB"))


def load_task_map(source_root: Path) -> dict[int, str]:
    """Return {task_index_int: task_description_str}."""
    df = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    # Old LeRobot format: task description is the DataFrame row-index,
    # 'task_index' column holds the integer id.
    if "task_index" in df.columns:
        return {int(task_idx): str(task_text) for task_text, task_idx in zip(df.index, df["task_index"])}
    # Fallback: two columns, first is description, second is index
    df = df.reset_index()
    return {int(row.iloc[1]): str(row.iloc[0]) for _, row in df.iterrows()}


def build_frame(
    row: pd.Series,
    task_map: dict[int, str],
    *,
    is_first: bool,
    is_last: bool,
) -> dict:
    joint_pos = np.array(row["joint_position"], dtype=np.float32)   # (7,)
    gripper_pos = np.float32(row["gripper_position"])
    actions = np.array(row["actions"], dtype=np.float32)             # (8,)

    task_idx = int(row["task_index"])
    language_instruction = task_map.get(task_idx, "")

    zeros6 = np.zeros(6, dtype=np.float32)
    zeros7 = np.zeros(7, dtype=np.float32)

    def fa(v) -> np.ndarray:  # scalar float32 array
        return ensure_scalar_array(v, dtype=np.float32)

    def ia(v) -> np.ndarray:  # scalar int64 array
        return ensure_scalar_array(v, dtype=np.int64)

    def ba(v) -> np.ndarray:  # scalar bool array
        return ensure_scalar_array(bool(v), dtype=np.bool_)

    return {
        # Images (numpy HWC uint8 — writer encodes to mp4)
        # Swap ext 1 <-> ext 2: source labels are inverted vs physical cameras
        "observation.images.exterior_1_left": decode_image(row["exterior_image_2_left"]),
        "observation.images.exterior_2_left": decode_image(row["exterior_image_1_left"]),
        "observation.images.wrist_left":      decode_image(row["wrist_image_left"]),
        # Episode flags
        "is_first":    ba(is_first),
        "is_last":     ba(is_last),
        "is_terminal": ba(is_last),
        # Language (writer manages task_index internally via "task")
        "task":                   language_instruction,
        "language_instruction":   language_instruction,
        "language_instruction_2": "",
        "language_instruction_3": "",
        # Observation state
        "observation.state.gripper_position":   fa(gripper_pos),
        "observation.state.cartesian_position": zeros6,
        "observation.state.joint_position":     joint_pos,
        "observation.state": np.concatenate([joint_pos, [gripper_pos]]),  # (8,)
        # Action  (actions[:7] = joint targets, actions[7] = gripper)
        "action.joint_position":                actions[:7],
        "action.joint_velocity":                zeros7,
        "action.cartesian_position":            zeros6,
        "action.cartesian_velocity":            zeros6,
        "action.gripper_position":              fa(actions[7]),
        "action.gripper_velocity":              fa(0.0),
        "action.original":                      zeros7,
        "action.source_joint_velocity_gripper": actions,
        "action":                               actions,
        # Metadata
        "discount":                      fa(1.0),
        "reward":                        fa(1.0),
        "task_category":                 "",
        "building":                      BUILDING,
        "collector_id":                  COLLECTOR_ID,
        "datetime":                      "",
        "camera_extrinsics.wrist_left":      zeros6,
        "camera_extrinsics.exterior_1_left": zeros6,
        "camera_extrinsics.exterior_2_left": zeros6,
        "is_episode_successful":         ba(True),
        "environment.conveyor_speed":    fa(0.0),
        "subtask_index":                 ia(0),
        # NOTE: episode_index, frame_index, index, timestamp, task_index
        # are managed internally by the LeRobot dataset writer — do not include.
    }


def flush_episode(
    writer: CanonicalDatasetWriter,
    task_map: dict[int, str],
    source_repo_id: str,
    ep_idx: int,
    rows: list[pd.Series],
) -> None:
    n = len(rows)
    frames = [
        build_frame(row, task_map, is_first=(i == 0), is_last=(i == n - 1))
        for i, row in enumerate(rows)
    ]
    dest_idx = writer.add_episode(
        frames,
        source_repo_id=source_repo_id,
        source_episode_index=ep_idx,
    )
    log.info("  episode %d -> dest %d  (%d frames)", ep_idx, dest_idx, n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-repo-id", default=SOURCE_REPO_ID)
    parser.add_argument("--dest-repo-id", default=DEST_REPO_ID)
    parser.add_argument("--dest-root", type=Path, default=Path("/tmp/droid_xxjd_7_2_canonical"))
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Download source (skip video files — they don't exist in old format)
    log.info("Downloading %s ...", args.source_repo_id)
    source_root = Path(
        snapshot_download(
            args.source_repo_id,
            repo_type="dataset",
            ignore_patterns=["videos/**", "*.gitattributes"],
        )
    )
    log.info("Source root: %s", source_root)

    task_map = load_task_map(source_root)
    log.info("Loaded %d tasks", len(task_map))

    parquet_files = sorted((source_root / "data" / "chunk-000").glob("*.parquet"))
    log.info("Found %d data parquet files", len(parquet_files))

    writer = CanonicalDatasetWriter(
        repo_id=args.dest_repo_id,
        root=args.dest_root,
        overwrite=args.overwrite,
    )

    # Stream parquet files one at a time to keep memory low.
    # Episodes are stored contiguously so we flush as soon as we see
    # a new episode_index.
    current_ep_idx: int | None = None
    current_rows: list[pd.Series] = []

    for pq in parquet_files:
        df = pd.read_parquet(pq).sort_values(["episode_index", "frame_index"])
        for _, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            if current_ep_idx is not None and ep_idx != current_ep_idx:
                flush_episode(writer, task_map, args.source_repo_id, current_ep_idx, current_rows)
                current_rows = []
            current_ep_idx = ep_idx
            current_rows.append(row)

    # Flush the final episode
    if current_rows and current_ep_idx is not None:
        flush_episode(writer, task_map, args.source_repo_id, current_ep_idx, current_rows)

    writer.finalize()
    log.info("Finalized dataset at %s", writer.root)

    if args.push_to_hub:
        log.info("Pushing to Hub: %s ...", args.dest_repo_id)
        HfApi().upload_folder(
            folder_path=str(writer.root),
            repo_id=args.dest_repo_id,
            repo_type="dataset",
            commit_message="feat: convert droid_xxjd_7 to canonical v3.0",
        )
        log.info("Done: https://huggingface.co/datasets/%s", args.dest_repo_id)
    else:
        log.info("Skipping Hub push — pass --push-to-hub to upload")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Dry-run port_droid for N episodes — prints TFDS episode metadata and compares
against the cadene/droid_1.0.1_v30 LeRobot dataset to confirm ordering matches.

Usage:
    uv run examples/port_droid_lite.py --raw-dir /path/to/droid/1.0.1 --n-episodes 20

The raw-dir should point to the TFDS data directory, e.g.:
    ~/tensorflow_datasets/droid/1.0.1
"""

import argparse
import pathlib

import pandas as pd
import tensorflow_datasets as tfds


def is_episode_successful(tf_episode_metadata) -> bool:
    return "/success/" in tf_episode_metadata["file_path"].numpy().decode()


def extract_episode_info(tf_episode, episode_index: int) -> dict:
    m = tf_episode["episode_metadata"]
    file_path = m["file_path"].numpy().decode()
    collector_id = m["collector_id"].numpy().decode()
    building = m["building"].numpy().decode()
    date = m["date"].numpy().decode()
    successful = is_episode_successful(m)

    # Count frames and grab first language instruction
    n_frames = 0
    lang = ""
    for step in tf_episode["steps"]:
        if n_frames == 0:
            lang = step["language_instruction"].numpy().decode()
        n_frames += 1

    return {
        "tfds_index":    episode_index,
        "file_path":     file_path,
        "collector_id":  collector_id,
        "building":      building,
        "date":          date,
        "successful":    successful,
        "n_frames":      n_frames,
        "language_instruction": lang,
    }


def load_lerobot_episodes(n: int) -> pd.DataFrame:
    """Load first N episodes from the cached cadene/droid_1.0.1_v30 metadata."""
    cache = pathlib.Path.home() / ".cache/huggingface/lerobot/cadene/droid_1.0.1_v30"
    ep_chunk0 = cache / "meta/episodes/chunk-000"
    if not ep_chunk0.exists():
        print("  (LeRobot cache not found, skipping comparison)")
        return pd.DataFrame()

    parquet_file = next(ep_chunk0.iterdir(), None)
    if parquet_file is None:
        return pd.DataFrame()

    df = pd.read_parquet(parquet_file)
    cols = ["episode_index", "tasks", "length"]
    return df[cols].head(n).reset_index(drop=True)


def run(raw_dir: pathlib.Path, n_episodes: int, shard_index: int):
    dataset_name = raw_dir.parent.name
    version = raw_dir.name
    data_dir = raw_dir.parent.parent

    print(f"Building TFDS dataset from {raw_dir} ...")
    builder = tfds.builder(f"{dataset_name}/{version}", data_dir=data_dir, version="")

    split = f"train[{shard_index}shard]" if shard_index is not None else "train"
    raw_dataset = builder.as_dataset(split=split)

    print(f"Iterating first {n_episodes} episodes from split='{split}' ...\n")

    tfds_rows = []
    for i, episode in enumerate(raw_dataset):
        if i >= n_episodes:
            break
        info = extract_episode_info(episode, i)
        tfds_rows.append(info)

        print(
            f"[{i:3d}] {'SUCC' if info['successful'] else 'FAIL'} | "
            f"frames={info['n_frames']:4d} | "
            f"collector={info['collector_id']} | "
            f"task='{info['language_instruction'][:60]}'"
        )
        print(f"       file_path={info['file_path']}")

    tfds_df = pd.DataFrame(tfds_rows)

    print("\n" + "="*70)
    print("LeRobot cadene/droid_1.0.1_v30 — first N episodes from cache:")
    print("="*70)
    lr_df = load_lerobot_episodes(n_episodes)
    if not lr_df.empty:
        for _, row in lr_df.iterrows():
            task = row["tasks"][0] if row["tasks"] else ""
            print(f"[{int(row['episode_index']):3d}] frames={int(row['length']):4d} | task='{task[:60]}'")
    else:
        print("  (no cached LeRobot metadata found)")

    print("\n" + "="*70)
    print("Comparison: TFDS vs LeRobot episode order")
    print("="*70)
    if not lr_df.empty and len(tfds_df) == len(lr_df):
        matches = 0
        for i, (tfds_row, (_, lr_row)) in enumerate(zip(tfds_rows, lr_df.iterrows())):
            tfds_task = tfds_row["language_instruction"]
            lr_task   = lr_row["tasks"][0] if lr_row["tasks"] else ""
            tfds_frames = tfds_row["n_frames"]
            lr_frames   = int(lr_row["length"])
            task_match   = tfds_task == lr_task
            frames_match = tfds_frames == lr_frames
            match = task_match and frames_match
            if match:
                matches += 1
            status = "OK" if match else "MISMATCH"
            print(
                f"[{i:3d}] {status} | frames tfds={tfds_frames} lr={lr_frames} | "
                f"task_match={task_match}"
            )
            if not match:
                print(f"       tfds: '{tfds_task[:80]}'")
                print(f"       lr:   '{lr_task[:80]}'")
        print(f"\n{matches}/{n_episodes} episodes matched exactly.")
    else:
        print("  (cannot compare — different lengths or no LeRobot cache)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True,
                        help="Path to TFDS data dir, e.g. ~/tensorflow_datasets/droid/1.0.1")
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of episodes to inspect (default: 20)")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="TFDS shard index to read from (default: full dataset)")
    args = parser.parse_args()
    run(args.raw_dir, args.n_episodes, args.shard_index)


if __name__ == "__main__":
    main()

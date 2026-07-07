#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.training import rlt_timeline_replay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw RLT timeline HDF5 into trainable paper-anchor replay.")
    parser.add_argument("--hdf5", type=Path, required=True, help="Input episode.hdf5 containing /rlt_timeline.")
    parser.add_argument("--output", type=Path, required=True, help="Output trainable replay .npz shard.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest to append.")
    parser.add_argument("--train-horizon", type=int, required=True)
    parser.add_argument("--chunk-stride", type=int, required=True)
    parser.add_argument(
        "--policy-event-alignment",
        choices=(
            rlt_timeline_replay.POLICY_EVENT_ALIGNMENT_EXACT,
            rlt_timeline_replay.POLICY_EVENT_ALIGNMENT_TRUNK_SHARED,
        ),
        default=rlt_timeline_replay.DEFAULT_POLICY_EVENT_ALIGNMENT,
        help="How to align sparse same-forward policy events to sampled training anchors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def append_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = rlt_timeline_replay.write_paper_replay_shard_from_timeline_hdf5(
        args.hdf5,
        args.output,
        train_horizon=args.train_horizon,
        chunk_stride=args.chunk_stride,
        policy_event_alignment=args.policy_event_alignment,
        overwrite=args.overwrite,
    )
    if args.manifest is not None:
        append_manifest_row(
            args.manifest,
            {
                "shard_path": str(args.output.resolve()),
                "key_region_id": manifest["key_region_id"],
                "source_format": manifest["source_format"],
                "replay_state_grain": manifest["replay_state_grain"],
                "num_transitions": int(np.prod(manifest["replay_array_shapes"]["done"])),
                "z_dim": int(manifest["z_rl_dim"]),
            },
        )


if __name__ == "__main__":
    main()

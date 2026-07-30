#!/usr/bin/env python3
"""Fetch an explicit, bounded set of attention review images over read-only SSH."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


SELECTIONS = {
    "20260724-043023": [0, 72, 144, 216, 288],
    "20260724-050216": [0, 280, 560, 840, 1120, 1400, 1694],
    "20260729-043606": [
        0, 730, 1460,
        2180, 2185, 2190, 2195, 2200, 2205, 2210,
        2920, 3650, 4381,
    ],
}
FILES = ["overview.jpg", "cam_high.jpg", "cam_left_wrist.jpg", "cam_right_wrist.jpg", "metadata.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="aloha")
    parser.add_argument("--remote-root", default="/home/eii/openpi0.5-rtc/attention_debug")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for run_id, indices in SELECTIONS.items():
        for index in indices:
            sample = f"sample_{index:06d}"
            destination = args.output_dir / run_id / sample
            destination.mkdir(parents=True, exist_ok=True)
            for filename in FILES:
                remote_path = f"{args.remote_root}/{run_id}/{sample}/{filename}"
                result = subprocess.run(
                    ["ssh", args.host, f"cat {shlex.quote(remote_path)}"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                (destination / filename).write_bytes(result.stdout)


if __name__ == "__main__":
    main()

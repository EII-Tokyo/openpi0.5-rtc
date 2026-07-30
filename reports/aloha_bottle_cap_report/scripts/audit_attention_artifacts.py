#!/usr/bin/env python3
"""Audit real attention capture artifacts without interpreting them as success."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess


def ssh_text(host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", host, command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="aloha")
    parser.add_argument("--remote-root", default="/home/eii/openpi0.5-rtc/attention_debug")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    paths = ssh_text(
        args.host,
        f"find {shlex.quote(args.remote_root)} -mindepth 2 -maxdepth 2 "
        "-name manifest.jsonl -type f -print | sort",
    ).splitlines()
    runs = []
    all_rows = []
    failures = []
    shape_counter: dict[str, int] = {}
    probe_counter: dict[str, int] = {}

    for path in paths:
        run_id = Path(path).parent.name
        try:
            raw = ssh_text(args.host, f"cat {shlex.quote(path)}")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not rows:
                continue
            capture_ms = [float(row["capture_ms"]) for row in rows]
            warmup_filtered = capture_ms[1:] if len(capture_ms) > 1 else capture_ms
            run_start = min(float(row["unix_time"]) for row in rows)
            run_end = max(float(row["unix_time"]) for row in rows)
            camera_shares = {key: [] for key in rows[0]["camera_order"]}
            modes: dict[str, int] = {}
            for row in rows:
                total_mass = sum(float(value) for value in row["mean_attention_mass"].values())
                for camera in camera_shares:
                    camera_shares[camera].append(
                        float(row["mean_attention_mass"][camera]) / total_mass if total_mass else 0.0
                    )
                mode = row.get("chunking_mode", "unknown")
                modes[mode] = modes.get(mode, 0) + 1
                shape_key = json.dumps(row.get("attention_shape"), sort_keys=True)
                shape_counter[shape_key] = shape_counter.get(shape_key, 0) + 1
                probe_key = json.dumps(row.get("probe"), sort_keys=True)
                probe_counter[probe_key] = probe_counter.get(probe_key, 0) + 1
                all_rows.append(
                    {
                        "run_id": run_id,
                        "sample": row["sample"],
                        "unix_time": row["unix_time"],
                        "chunking_mode": mode,
                        "capture_ms": row["capture_ms"],
                        **{
                            f"{camera}_share": (
                                float(row["mean_attention_mass"][camera]) / total_mass if total_mass else 0.0
                            )
                            for camera in camera_shares
                        },
                    }
                )
            runs.append(
                {
                    "run_id": run_id,
                    "samples": len(rows),
                    "start_utc": datetime.fromtimestamp(run_start, timezone.utc).isoformat(),
                    "end_utc": datetime.fromtimestamp(run_end, timezone.utc).isoformat(),
                    "span_sec": run_end - run_start,
                    "chunking_modes": modes,
                    "capture_ms_after_first": {
                        "median": statistics.median(warmup_filtered),
                        "p95": percentile(warmup_filtered, 0.95),
                        "max": max(warmup_filtered),
                    },
                    "mean_camera_share": {
                        camera: statistics.fmean(values) for camera, values in camera_shares.items()
                    },
                }
            )
        except Exception as exc:
            failures.append({"manifest": path, "error": f"{type(exc).__name__}: {exc}"})

    runs.sort(key=lambda row: row["run_id"])
    result = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": args.remote_root,
        "manifest_count": len(paths),
        "runs_with_samples": len(runs),
        "total_samples": len(all_rows),
        "total_manifest_span_sec": sum(row["span_sec"] for row in runs),
        "runs": runs,
        "attention_shape_variants": [
            {"shape": json.loads(key), "samples": count} for key, count in shape_counter.items()
        ],
        "probe_variants": [
            {"probe": json.loads(key), "samples": count} for key, count in probe_counter.items()
        ],
        "failures": failures,
        "supports": [
            "Real model inference emitted three-camera attention tensors.",
            "Camera attention allocation and capture overhead can be summarized.",
            "Attention dimensions cover 18 layers, 50 action queries and a 16x16 image-token grid.",
        ],
        "does_not_support": [
            "Task success or failure: manifests contain no task outcome label.",
            "Causal importance: no occlusion or intervention comparison is recorded.",
            "The field-observed one-hour production claim: a manifest span may include idle time.",
            "Specific diagnosis of no-cap air-unscrewing or upside-down grasping without aligned labels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    fields = [
        "run_id",
        "sample",
        "unix_time",
        "chunking_mode",
        "capture_ms",
        "cam_high_share",
        "cam_left_wrist_share",
        "cam_right_wrist_share",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()

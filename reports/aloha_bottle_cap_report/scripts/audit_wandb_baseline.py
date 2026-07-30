#!/usr/bin/env python3
"""Export the exact baseline training run from W&B through the ALOHA host."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import statistics
import subprocess


RUN_NAME = "no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520"

REMOTE_PROGRAM = rf'''
import json
import wandb

api = wandb.Api(timeout=90)
entity = api.default_entity
runs = api.runs(
    f"{{entity}}/openpi",
    filters={{"display_name": {{"$eq": {RUN_NAME!r}}}}},
)
if len(runs) != 1:
    raise RuntimeError(f"expected one run, found {{len(runs)}}")
run = runs[0]
config_keys = [
    "name", "exp_name", "batch_size", "num_workers", "fsdp_devices",
    "gradient_accumulation_steps", "num_train_steps", "seed",
    "save_interval", "keep_period", "ema_decay", "lr_schedule",
    "optimizer", "freeze_filter", "model", "data", "resume", "overwrite",
]
history_keys = [
    "_step", "_timestamp", "_runtime", "loss", "grad_norm", "param_norm",
    "effective_batch_size", "gradient_accumulation_steps", "data_wait_time",
    "train_block_time", "train_dispatch_time", "train_step_time",
]
history = []
for row in run.scan_history(keys=history_keys, page_size=1000):
    history.append({{key: row.get(key) for key in history_keys if row.get(key) is not None}})
print(json.dumps({{
    "run": {{
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "last_history_step": run.lastHistoryStep,
        "history_rows": len(history),
        "config": {{key: run.config.get(key) for key in config_keys}},
        "summary": {{key: value for key, value in run.summary.items() if not key.startswith("_")}},
    }},
    "history": history,
}}, ensure_ascii=False, default=str))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="aloha")
    parser.add_argument("--repo", default="/home/eii/openpi0.5-rlt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-csv", type=Path, required=True)
    args = parser.parse_args()

    remote = (
        f"cd {shlex.quote(args.repo)} && "
        "WANDB_SILENT=true .venv/bin/python -c "
        + shlex.quote(f"exec({REMOTE_PROGRAM!r})")
    )
    result = subprocess.run(
        ["ssh", args.host, remote],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(result.stdout)
    history = payload.pop("history")
    timestamps = [float(row["_timestamp"]) for row in history if "_timestamp" in row]
    steps = [int(row["_step"]) for row in history if "_step" in row]
    losses = [float(row["loss"]) for row in history if "loss" in row]
    gaps = []
    for previous, current in zip(history, history[1:]):
        if "_timestamp" not in previous or "_timestamp" not in current:
            continue
        gap = float(current["_timestamp"]) - float(previous["_timestamp"])
        step_gap = int(current.get("_step", 0)) - int(previous.get("_step", 0))
        if gap > 3600:
            gaps.append(
                {
                    "from_step": previous.get("_step"),
                    "to_step": current.get("_step"),
                    "gap_sec": gap,
                    "step_gap": step_gap,
                }
            )
    payload["audit_generated_utc"] = datetime.now(timezone.utc).isoformat()
    payload["history_analysis"] = {
        "first_logged_step": min(steps) if steps else None,
        "last_logged_step": max(steps) if steps else None,
        "first_timestamp_utc": datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat() if timestamps else None,
        "last_timestamp_utc": datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat() if timestamps else None,
        "wall_span_sec": max(timestamps) - min(timestamps) if timestamps else None,
        "pause_gaps_over_one_hour": gaps,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_median": statistics.median(losses) if losses else None,
    }
    payload["evidence_limits"] = [
        "The W&B run proves optimization history, not robot task success.",
        "The run continued to step 59990, while the inspected/deployed checkpoint directory contains step 19000 only.",
        "Wall span includes pauses and resume gaps and is not equal to active accelerator time.",
        "No validation or robot evaluation metric is present in this run history.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    columns = [
        "_step", "_timestamp", "_runtime", "loss", "grad_norm", "param_norm",
        "effective_batch_size", "gradient_accumulation_steps", "data_wait_time",
        "train_block_time", "train_dispatch_time", "train_step_time",
    ]
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.history_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()

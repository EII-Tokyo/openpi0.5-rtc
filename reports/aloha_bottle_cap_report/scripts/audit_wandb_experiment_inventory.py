#!/usr/bin/env python3
"""Inventory purpose-relevant W&B experiments without ranking unlike losses."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_PROGRAM = r'''
import json
import wandb

api = wandb.Api(timeout=120)
entity = api.default_entity
runs = api.runs(
    f"{entity}/openpi",
    filters={
        "created_at": {
            "$gte": "2026-04-01T00:00:00Z",
            "$lte": "2026-05-31T23:59:59Z",
        }
    },
    order="+created_at",
    per_page=200,
)
rows = []
for run in runs:
    name = run.name or ""
    config_name = str(run.config.get("name") or "")
    text = (name + " " + config_name).lower()
    baseline = any(token in text for token in [
        "no_rinse", "no-rinse", "twist_direction", "twist_only",
        "pi05-twist", "twist-one", "return_home_29repo",
    ])
    rinse = "rinse" in text or "water" in text
    if not baseline and not rinse:
        continue
    family = "baseline_bottle_sorting" if baseline else "rinse_or_insertion_exploration"
    rows.append({
        "id": run.id,
        "name": name,
        "config_name": config_name,
        "family": family,
        "created_at": run.created_at,
        "state": run.state,
        "last_step": run.lastHistoryStep,
        "batch_size": run.config.get("batch_size"),
        "seed": run.config.get("seed"),
        "summary_loss": run.summary.get("loss"),
    })
print(json.dumps(rows, ensure_ascii=False, default=str))
'''


def family_summary(rows: list[dict]) -> dict:
    state_counts = Counter(row["state"] for row in rows)
    return {
        "run_attempts": len(rows),
        "unique_run_names": len({row["name"] for row in rows}),
        "unique_config_names": len({row["config_name"] for row in rows}),
        "states": dict(state_counts),
        "runs_with_any_history": sum(int(row["last_step"] or -1) >= 0 for row in rows),
        "runs_reaching_1000_steps": sum(int(row["last_step"] or -1) >= 1000 for row in rows),
        "runs_reaching_10000_steps": sum(int(row["last_step"] or -1) >= 10000 for row in rows),
        "runs_reaching_25000_steps": sum(int(row["last_step"] or -1) >= 25000 for row in rows),
        "batch_sizes": sorted({row["batch_size"] for row in rows if row["batch_size"] is not None}),
        "seeds": sorted({row["seed"] for row in rows if row["seed"] is not None}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="aloha")
    parser.add_argument("--repo", default="/home/eii/openpi0.5-rlt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
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
    rows = json.loads(result.stdout)
    families = sorted({row["family"] for row in rows})
    payload = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_window_utc": ["2026-04-01T00:00:00Z", "2026-05-31T23:59:59Z"],
        "selection_rule": (
            "W&B project openpi; keep runs whose run/config name explicitly contains "
            "bottle baseline (no-rinse/twist) or rinse/water task tokens."
        ),
        "summary": {"all_selected": family_summary(rows)},
        "family_summary": {
            family: family_summary([row for row in rows if row["family"] == family])
            for family in families
        },
        "runs": rows,
        "interpretation_limits": [
            "A W&B run is an experiment attempt, not necessarily a completed training.",
            "Repeated names may be retries and are retained to measure engineering effort.",
            "Losses from different objectives, model heads or datasets are not ranked.",
            "Run state crashed/failed does not by itself identify the root cause.",
            "All selected runs use one recorded seed unless a row says otherwise.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    fields = [
        "id", "name", "config_name", "family", "created_at", "state",
        "last_step", "batch_size", "seed", "summary_loss",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

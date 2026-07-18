from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import numpy as np

from aloha_isaac_replay.scripts.run_phase117_diagnostic_held_bottle_replay import _phase117_args
from aloha_isaac_replay.scripts.run_phase117_diagnostic_held_bottle_replay import DEFAULT_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_DB = Path("/home/eii/data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3")
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/twist_off_the_bottle_cap")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase120_success_hdf5_empirical_pipe_cluster_20260719"


def _replace_arg(command: list[str], flag: str, values: list[str]) -> None:
    index = command.index(flag)
    stop = index + 1
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    command[index + 1 : stop] = values


def _select_hdf5_by_reward(segment_db: Path, rollout_root: Path, reward: int, limit: int, date: str | None) -> list[dict[str, str]]:
    con = sqlite3.connect(segment_db)
    rows = con.execute(
        """
        select key_region_id, updated_at
        from segments
        where reward=? and status!='deleted'
        order by updated_at desc
        """,
        (reward,),
    ).fetchall()
    con.close()
    selected: list[dict[str, str]] = []
    for key_region_id, updated_at in rows:
        matches = sorted(rollout_root.glob(f"*/warmup/key_region_{key_region_id}/episode.hdf5"))
        if date:
            matches = [path for path in matches if path.parts[-4] == date]
        if not matches:
            continue
        path = matches[0]
        selected.append(
            {
                "key_region_id": key_region_id,
                "updated_at": str(updated_at),
                "reward": str(reward),
                "date": path.parts[-4],
                "hdf5": str(path),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        return subprocess.call(command, cwd=cwd, stdout=stdout, stderr=stderr)


def _run_one(
    item: dict[str, str],
    output_root: Path,
    isaac_python: str,
    workcell_contact_policy: Path,
) -> dict[str, Any]:
    short_id = item["key_region_id"][:8]
    item_dir = output_root / f"{item['date']}_{short_id}"
    replay_dir = item_dir / "replay"
    fit_dir = item_dir / "fit"

    command = [isaac_python, *_phase117_args(replay_dir, workcell_contact_policy, start_frame=0)]
    _replace_arg(command, "--hdf5-gripper-episode", [item["hdf5"]])
    replay_rc = _run_command(command, REPO_ROOT, item_dir / "isaac_replay_stdout.txt", item_dir / "isaac_replay_stderr.txt")

    fit_rc: int | None = None
    replay_csv = replay_dir / "gripper_passive_contact_timeseries.csv"
    fit_summary_path = fit_dir / "empirical_pipe_candidate_summary.json"
    if replay_csv.exists():
        fit_command = [
            sys.executable,
            str(REPO_ROOT / "aloha_isaac_replay/scripts/fit_phase119_empirical_pipe_candidate.py"),
            "--csv",
            str(replay_csv),
            "--output-dir",
            str(fit_dir),
        ]
        fit_rc = _run_command(fit_command, REPO_ROOT, item_dir / "fit_stdout.txt", item_dir / "fit_stderr.txt")

    replay_metrics_path = replay_dir / "gripper_passive_contact_metrics.json"
    replay_status: dict[str, Any] = {}
    if replay_metrics_path.exists():
        metrics = json.loads(replay_metrics_path.read_text())
        replay_status = {
            "status": metrics.get("status"),
            "overall_pass": metrics.get("overall_pass"),
            "failure_reasons": metrics.get("failure_reasons", []),
            "controller_tracking_gate": metrics.get("controller_tracking_gate"),
        }

    fit_summary = None
    if fit_summary_path.exists():
        fit_summary = json.loads(fit_summary_path.read_text())

    return {
        **item,
        "item_dir": str(item_dir),
        "replay_dir": str(replay_dir),
        "fit_dir": str(fit_dir),
        "replay_returncode": replay_rc,
        "replay_csv_exists": replay_csv.exists(),
        "replay_status": replay_status,
        "fit_returncode": fit_rc,
        "fit_summary": fit_summary,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in results if row.get("fit_returncode") == 0 and row.get("fit_summary")]
    entries = np.asarray([row["fit_summary"]["empirical_candidate"]["entry"] for row in usable], dtype=np.float64)
    axes = np.asarray([row["fit_summary"]["empirical_candidate"]["axis_unit_base_to_entry"] for row in usable], dtype=np.float64)
    if len(usable) == 0:
        return {"usable_count": 0, "status": "NO_USABLE_FITS"}
    mean_axis = np.mean(axes, axis=0)
    mean_axis = mean_axis / max(float(np.linalg.norm(mean_axis)), 1e-12)
    return {
        "status": "PASS",
        "usable_count": len(usable),
        "replay_gate_pass_count": sum(1 for row in usable if row.get("replay_returncode") == 0),
        "replay_gate_failed_but_fit_count": sum(1 for row in usable if row.get("replay_returncode") != 0),
        "entry_mean": np.mean(entries, axis=0).tolist(),
        "entry_std": np.std(entries, axis=0).tolist(),
        "entry_rms_spread_m": float(np.sqrt(np.mean(np.sum((entries - np.mean(entries, axis=0)) ** 2, axis=1)))),
        "axis_mean_unit": mean_axis.tolist(),
        "result_ids": [row["key_region_id"] for row in usable],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded Phase120 cluster probe over annotated HDF5 key regions.")
    parser.add_argument("--segment-db", type=Path, default=DEFAULT_SEGMENT_DB)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workcell-contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--isaac-python", default=str(REPO_ROOT / ".venv_issac/bin/python"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--date", default="2026-07-08")
    parser.add_argument("--reward", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = _select_hdf5_by_reward(args.segment_db, args.rollout_root, args.reward, args.limit, args.date)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_hdf5.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"selected": selected}, indent=2))
        return 0

    results = []
    for item in selected:
        result = _run_one(item, args.output_dir, args.isaac_python, args.workcell_contact_policy)
        results.append(result)
        print(
            json.dumps(
                {
                    "key_region_id": item["key_region_id"],
                    "replay_returncode": result["replay_returncode"],
                    "fit_returncode": result["fit_returncode"],
                    "item_dir": result["item_dir"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {"selected_count": len(selected), "aggregate": _aggregate(results), "results": results}
    (args.output_dir / "phase120_cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if results and all(row["fit_returncode"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

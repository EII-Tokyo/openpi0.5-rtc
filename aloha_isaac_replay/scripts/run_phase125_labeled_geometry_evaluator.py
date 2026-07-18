from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase125_labeled_geometry_evaluator_20260719"
ALL_LIMIT = 100000


def _normalized_limit(limit: int) -> int:
    return ALL_LIMIT if limit <= 0 else limit


def _build_phase120_command(*, reward: int, limit: int, date: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "aloha_isaac_replay/scripts/run_phase120_success_hdf5_empirical_pipe_cluster.py"),
        "--reward",
        str(reward),
        "--limit",
        str(_normalized_limit(limit)),
        "--date",
        date,
        "--output-dir",
        str(output_dir),
    ]


def _build_compare_command(*, success_root: Path, failure_root: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "aloha_isaac_replay/scripts/compare_phase122_success_failure_geometry_metrics.py"),
        "--success-root",
        str(success_root),
        "--failure-root",
        str(failure_root),
        "--output-dir",
        str(output_dir),
    ]


def _build_plan(*, date: str, success_limit: int, failure_limit: int, output_dir: Path) -> dict[str, Any]:
    success_root = output_dir / "success_cluster"
    failure_root = output_dir / "failure_cluster"
    comparison_root = output_dir / "geometry_comparison"
    steps = [
        {
            "name": "success_cluster",
            "summary_path": str(success_root / "phase120_cluster_summary.json"),
            "command": " ".join(shlex.quote(part) for part in _build_phase120_command(reward=1, limit=success_limit, date=date, output_dir=success_root)),
        },
        {
            "name": "failure_cluster",
            "summary_path": str(failure_root / "phase120_cluster_summary.json"),
            "command": " ".join(shlex.quote(part) for part in _build_phase120_command(reward=0, limit=failure_limit, date=date, output_dir=failure_root)),
        },
        {
            "name": "geometry_comparison",
            "summary_path": str(comparison_root / "success_failure_geometry_metrics.json"),
            "command": " ".join(
                shlex.quote(part)
                for part in _build_compare_command(success_root=success_root, failure_root=failure_root, output_dir=comparison_root)
            ),
        },
    ]
    return {
        "date": date,
        "output_dir": str(output_dir),
        "success_limit": _normalized_limit(success_limit),
        "failure_limit": _normalized_limit(failure_limit),
        "steps": steps,
    }


def _run_step(step: dict[str, str], log_dir: Path, skip_existing: bool) -> dict[str, Any]:
    summary_path = Path(step["summary_path"])
    if skip_existing and summary_path.exists():
        return {"name": step["name"], "returncode": 0, "skipped": True, "summary_path": str(summary_path)}

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{step['name']}_stdout.log"
    stderr_path = log_dir / f"{step['name']}_stderr.log"
    command = shlex.split(step["command"])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        returncode = subprocess.call(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)
    return {
        "name": step["name"],
        "returncode": returncode,
        "skipped": False,
        "summary_path": str(summary_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run labeled HDF5 geometry replay and success/failure separation evaluation.")
    parser.add_argument("--date", default="2026-07-08")
    parser.add_argument("--success-limit", type=int, default=0, help="0 means all available selected success HDF5s.")
    parser.add_argument("--failure-limit", type=int, default=0, help="0 means all available selected failure HDF5s.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = _build_plan(date=args.date, success_limit=args.success_limit, failure_limit=args.failure_limit, output_dir=args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase125_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    results = []
    for step in plan["steps"]:
        result = _run_step(step, args.output_dir / "logs", args.skip_existing)
        results.append(result)
        print(json.dumps(result), flush=True)
        if result["returncode"] != 0:
            break

    summary = {"plan": plan, "results": results}
    (args.output_dir / "phase125_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if results and all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

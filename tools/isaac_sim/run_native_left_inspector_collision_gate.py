"""Run the ALOHA left-arm collision gate in three fresh Full Kit processes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


PROJECT_ROOT = Path("/home/eii/project/openpi0.5-rtc-reward-learning")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PYTHON = PROJECT_ROOT / ".venv_issac/bin/python"
ISAACSIM = PROJECT_ROOT / ".venv_issac/bin/isaacsim"
FULL_KIT = (
    PROJECT_ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit"
)
TRIAL_SCRIPT = SOURCE_ROOT / "tools/isaac_sim/native_left_inspector_collision_trial.py"
EXPECTED_STAGE_SHA256 = (
    "165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2"
)
TRIAL_COUNT = 3
TRIAL_TIMEOUT_SECONDS = 240
REPORT_EXIT_GRACE_SECONDS = 5
TERMINATE_GRACE_SECONDS = 15


@dataclass(frozen=True)
class TrialLaunch:
    command: list[str]
    environment: dict[str, str]
    output_dir: Path


def build_trial_launch(output_root: Path, trial_index: int) -> TrialLaunch:
    """Build one isolated Full Kit launch without creating its trial directory."""
    if trial_index not in range(1, TRIAL_COUNT + 1):
        raise ValueError(f"trial_index must be 1..{TRIAL_COUNT}")
    trial_dir = Path(output_root) / f"trial_{trial_index:02d}"
    environment = dict(os.environ)
    source_pythonpath = str(SOURCE_ROOT)
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_pythonpath}{os.pathsep}{previous_pythonpath}"
        if previous_pythonpath
        else source_pythonpath
    )
    environment["CODEX_NATIVE_TRIAL_INDEX"] = str(trial_index)
    environment["CODEX_NATIVE_TRIAL_OUTPUT_DIR"] = str(trial_dir)
    return TrialLaunch(
        command=[
            str(ISAAC_PYTHON),
            str(ISAACSIM),
            str(FULL_KIT),
            "--exec",
            str(TRIAL_SCRIPT),
        ],
        environment=environment,
        output_dir=trial_dir,
    )


def aggregate_trial_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exactly three numbered, disposable, passing native trials."""
    if len(reports) != TRIAL_COUNT:
        raise ValueError("collision gate requires exactly three trial reports")
    indices = [int(report.get("trial_index", -1)) for report in reports]
    if indices != [1, 2, 3]:
        raise ValueError("collision gate requires ordered independent trials 1, 2, 3")

    failure_reasons: list[str] = []
    for report in reports:
        index = int(report["trial_index"])
        if report.get("status") != "PASS":
            failure_reasons.append(f"trial_{index:02d}_failed")
        if report.get("stage_sha256_after") != EXPECTED_STAGE_SHA256:
            failure_reasons.append(f"trial_{index:02d}_stage_hash_mismatch")
        if report.get("stage_saved") is not False:
            failure_reasons.append(f"trial_{index:02d}_stage_save_not_disproved")
        if report.get("real_robot_touched") is not False:
            failure_reasons.append(f"trial_{index:02d}_real_robot_touch_not_disproved")

    return {
        "status": "PASS" if not failure_reasons else "FAIL",
        "trial_count": len(reports),
        "full_kit": str(FULL_KIT),
        "failure_reasons": failure_reasons,
        "trials": reports,
    }


def run_gate(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    process_results: list[dict[str, Any]] = []

    for trial_index in range(1, TRIAL_COUNT + 1):
        launch = build_trial_launch(output_root, trial_index)
        log_path = output_root / f"trial_{trial_index:02d}.log"
        print(
            f"Starting native Full Kit collision trial {trial_index}/{TRIAL_COUNT}",
            flush=True,
        )
        report_path = launch.output_dir / "trial.json"
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                launch.command,
                cwd=SOURCE_ROOT,
                env=launch.environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + TRIAL_TIMEOUT_SECONDS
            report_seen_at: float | None = None
            timed_out = False
            controlled_termination = False
            while process.poll() is None:
                now = time.monotonic()
                if report_path.is_file():
                    if report_seen_at is None:
                        report_seen_at = now
                    elif now - report_seen_at >= REPORT_EXIT_GRACE_SECONDS:
                        process.terminate()
                        controlled_termination = True
                        break
                if now >= deadline:
                    process.terminate()
                    timed_out = True
                    break
                time.sleep(0.25)
            try:
                returncode: int | None = process.wait(
                    timeout=TERMINATE_GRACE_SECONDS
                )
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=TERMINATE_GRACE_SECONDS)

        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "trial_index": trial_index,
                "status": "FAIL",
                "stage_saved": None,
                "real_robot_touched": None,
                "failure_reasons": ["missing_trial_report"],
            }
        if timed_out:
            report["status"] = "FAIL"
            report.setdefault("failure_reasons", []).append("trial_timed_out")
        elif returncode != 0 and not controlled_termination:
            report["status"] = "FAIL"
            report.setdefault("failure_reasons", []).append(
                f"full_kit_exit_code_{returncode}"
            )
        reports.append(report)
        process_results.append(
            {
                "trial_index": trial_index,
                "returncode": returncode,
                "timed_out": timed_out,
                "controlled_termination_after_report": controlled_termination,
                "log": str(log_path),
                "report": str(report_path),
            }
        )
        print(
            f"Finished trial {trial_index}: {report.get('status')} log={log_path}",
            flush=True,
        )

    aggregate = aggregate_trial_reports(reports)
    aggregate["processes"] = process_results
    aggregate_path = output_root / "native_collision_gate_report.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"NATIVE_INSPECTOR_COLLISION_GATE_{aggregate['status']} "
        f"report={aggregate_path}",
        flush=True,
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    return 0 if run_gate(args.output_root)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

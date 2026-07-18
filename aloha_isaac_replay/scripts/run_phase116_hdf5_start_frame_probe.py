from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from aloha_isaac_replay.scripts.run_phase115_strict_measured_workcell_no_support_plane_gate import _phase115_args
from aloha_isaac_replay.scripts.run_phase115_strict_measured_workcell_no_support_plane_gate import DEFAULT_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase116_hdf5_start_frame_probe_20260719"
DEFAULT_START_FRAMES = [80, 100, 120, 143]


def _replace_arg(command: list[str], flag: str, values: list[str]) -> None:
    index = command.index(flag)
    stop = index + 1
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    command[index + 1 : stop] = values


def _slug_for_start_frame(start_frame: int) -> str:
    return f"start_{start_frame:04d}"


def _summarize_run(start_frame: int, output_dir: Path, exit_code: int) -> dict[str, Any]:
    metrics_path = output_dir / "gripper_passive_contact_metrics.json"
    row: dict[str, Any] = {
        "start_frame": start_frame,
        "exit_code": exit_code,
        "output_dir": str(output_dir),
        "metrics_json": str(metrics_path),
    }
    if not metrics_path.exists():
        row["status"] = "MISSING_METRICS"
        return row
    data = json.loads(metrics_path.read_text())
    row.update(
        {
            "status": data.get("status"),
            "contact_trace_status": data.get("contact_trace_status"),
            "failure_reasons": data.get("failure_reasons"),
            "controller_tracking_status": (data.get("controller_tracking_gate") or {}).get("status"),
            "max_controlled_error": (data.get("controller_tracking_gate") or {}).get("max_controlled_error"),
            "target_limit_status": (data.get("target_runtime_limit_summary") or {}).get("status"),
            "workcell_policy_status": (data.get("workcell_contact_policy_gate") or {}).get("status"),
            "object_contact_categories": sorted((data.get("object_contact_categories") or {}).keys()),
            "total_object_displacement": data.get("total_object_displacement"),
            "object_final_center": data.get("object_final_center"),
        }
    )
    return row


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Phase116 HDF5 Start-Frame Probe",
        "",
        "| start frame | status | contact | tracking | workcell policy | categories | displacement |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.get('start_frame')}` | "
            f"`{row.get('status')}` | "
            f"`{row.get('contact_trace_status')}` | "
            f"`{row.get('controller_tracking_status')}` | "
            f"`{row.get('workcell_policy_status')}` | "
            f"`{row.get('object_contact_categories')}` | "
            f"`{row.get('total_object_displacement')}` |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe how far the strict Phase115 measured-workcell replay can move the HDF5 start frame earlier. "
            "Only the start frame changes; all other contact and controller settings are inherited from Phase115."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workcell-contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--start-frame", type=int, action="append", default=[])
    parser.add_argument("--require-all-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start_frames = args.start_frame or list(DEFAULT_START_FRAMES)
    commands: list[list[str]] = []
    for start_frame in start_frames:
        run_output_dir = args.output_dir / _slug_for_start_frame(start_frame)
        command = _phase115_args(run_output_dir, args.workcell_contact_policy)
        _replace_arg(command, "--hdf5-gripper-start-frame", [str(start_frame)])
        commands.append([args.python, *command])

    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for start_frame, command in zip(start_frames, commands, strict=True):
        run_output_dir = args.output_dir / _slug_for_start_frame(start_frame)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        with (run_output_dir / "stdout.log").open("w", encoding="utf-8") as stdout_file:
            with (run_output_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file:
                exit_code = subprocess.call(command, cwd=REPO_ROOT, stdout=stdout_file, stderr=stderr_file)
        rows.append(_summarize_run(start_frame, run_output_dir, exit_code))

    _write_summary(args.output_dir, rows)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "rows": rows}, ensure_ascii=False), flush=True)
    if args.require_all_pass and any(row.get("status") != "PASS" for row in rows):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

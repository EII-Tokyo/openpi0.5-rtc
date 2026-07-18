from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from aloha_isaac_replay.scripts.run_phase97_drive_target_reference import REPO_ROOT
from aloha_isaac_replay.scripts.run_phase97_drive_target_reference import _phase97_args


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase101_phase97_active_grasp_negative_control_20260719"
)
EXPECTED_FAILURE_REASON = "active_target_contact_gate_failed"
EXPECTED_CONTACT_TRACE_STATUS = "FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE"


def _negative_control_args(output_dir: Path) -> list[str]:
    command = _phase97_args(output_dir)
    try:
        command.remove("--already-in-contact-setup")
    except ValueError as exc:
        raise RuntimeError("Phase97 command no longer contains --already-in-contact-setup") from exc
    command.append("--require-active-target-contact")
    return command


def _load_metrics(output_dir: Path) -> dict:
    metrics_path = output_dir / "gripper_passive_contact_metrics.json"
    with metrics_path.open() as f:
        return json.load(f)


def _is_expected_negative(metrics: dict, child_returncode: int) -> bool:
    failure_reasons = set(metrics.get("failure_reasons") or [])
    gate = metrics.get("active_target_contact_gate") or {}
    return bool(
        child_returncode == 3
        and metrics.get("status") == "FAILED_GATE"
        and EXPECTED_FAILURE_REASON in failure_reasons
        and metrics.get("contact_trace_status") == EXPECTED_CONTACT_TRACE_STATUS
        and gate.get("status") == EXPECTED_CONTACT_TRACE_STATUS
        and gate.get("first_target_contact_found_phase") == "settle"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Phase97 negative control. The same already-contacting replay must fail if it is "
            "misclassified as an active grasp that first contacts the object during close."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [args.python, *_negative_control_args(args.output_dir)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0

    child = subprocess.run(command, cwd=REPO_ROOT, check=False)
    try:
        metrics = _load_metrics(args.output_dir)
    except FileNotFoundError:
        print(
            json.dumps(
                {
                    "status": "NEGATIVE_CONTROL_EXCEPTION",
                    "child_returncode": child.returncode,
                    "reason": "metrics file was not written",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    ok = _is_expected_negative(metrics, child.returncode)
    print(
        json.dumps(
            {
                "status": "PASS_EXPECTED_NEGATIVE_CONTROL" if ok else "FAIL_UNEXPECTED_NEGATIVE_CONTROL_RESULT",
                "child_returncode": child.returncode,
                "metrics_status": metrics.get("status"),
                "contact_trace_status": metrics.get("contact_trace_status"),
                "failure_reasons": metrics.get("failure_reasons"),
                "active_target_contact_gate": metrics.get("active_target_contact_gate"),
                "json": str(args.output_dir / "gripper_passive_contact_metrics.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

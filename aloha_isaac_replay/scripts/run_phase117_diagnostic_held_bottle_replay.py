from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from aloha_isaac_replay.scripts.run_phase115_strict_measured_workcell_no_support_plane_gate import _phase115_args
from aloha_isaac_replay.scripts.run_phase115_strict_measured_workcell_no_support_plane_gate import DEFAULT_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase117_diagnostic_held_bottle_replay_20260719"


def _replace_arg(command: list[str], flag: str, values: list[str]) -> None:
    index = command.index(flag)
    stop = index + 1
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    command[index + 1 : stop] = values


def _remove_flag(command: list[str], flag: str, value_count: int = 0) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index : index + 1 + value_count]


def _phase117_args(output_dir: Path, policy: Path, start_frame: int) -> list[str]:
    command = _phase115_args(output_dir, policy)
    _replace_arg(command, "--hdf5-gripper-start-frame", [str(start_frame)])
    _replace_arg(command, "--object-placement", ["grasp_yaml"])
    _replace_arg(command, "--max-object-displacement", ["2.0"])
    _remove_flag(command, "--trace-contact-pairs")
    _remove_flag(command, "--fail-on-non-target-object-contact")
    _remove_flag(command, "--allowed-non-target-object-contact-category", value_count=1)
    _remove_flag(command, "--workcell-contact-policy", value_count=1)
    _remove_flag(command, "--already-in-contact-setup")
    command.extend(["--diagnostic-held-object-mode", "follow_gripper"])
    command.extend(["--disable-object-rigid-body"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostic held-bottle replay. The bottle is placed from the grasp YAML and then kinematically "
            "updated from the left gripper frame at every replay step. This is a carried-object trajectory "
            "diagnostic, not a dynamic grasp/contact proof."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workcell-contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--start-frame", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [args.python, *_phase117_args(args.output_dir, args.workcell_contact_policy, args.start_frame)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

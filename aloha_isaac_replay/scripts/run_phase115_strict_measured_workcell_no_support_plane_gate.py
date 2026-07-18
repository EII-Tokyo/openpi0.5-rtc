from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from aloha_isaac_replay.scripts.run_phase107_bottleusd_hdf5_drive_target_gate import _phase107_args


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase115_strict_measured_workcell_no_support_plane_offset0_20260719"
)
DEFAULT_STAGE = (
    "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
    "aloha2_menagerie_scene_deep_black_real_start_pose_measured_workcell_no_gripper_base_bar_runtime.usda"
)
DEFAULT_POLICY = REPO_ROOT / "examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml"


def _replace_arg(command: list[str], flag: str, values: list[str]) -> None:
    index = command.index(flag)
    stop = index + 1
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    command[index + 1 : stop] = values


def _phase115_args(output_dir: Path, policy: Path) -> list[str]:
    command = _phase107_args(output_dir)
    _replace_arg(command, "--stage-usd", [DEFAULT_STAGE])
    _replace_arg(command, "--object-center-offset", ["0.0", "0.0", "0.0"])
    command.extend(["--workcell-contact-policy", str(policy)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strict measured-workcell BottleUSD HDF5 drive-target replay gate. "
            "This uses the measured /World/Table runtime stage, disables the stale legacy "
            "worldBody rail/table/gripper-base colliders identified in Phases 112-113, "
            "uses object offset 0.0 from Phase114, and does not add a diagnostic support plane."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workcell-contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [args.python, *_phase115_args(args.output_dir, args.workcell_contact_policy)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

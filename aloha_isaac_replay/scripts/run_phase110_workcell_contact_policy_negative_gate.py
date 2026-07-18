from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from aloha_isaac_replay.scripts.run_phase108_bottleusd_hdf5_diagnostic_table_gate import _phase108_args


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/phase110_workcell_contact_policy_negative_gate_20260719"
)
DEFAULT_POLICY = REPO_ROOT / "examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml"


def _phase110_args(output_dir: Path, policy: Path) -> list[str]:
    return [
        *_phase108_args(output_dir),
        "--workcell-contact-policy",
        str(policy),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase108 BottleUSD HDF5 replay with a conservative workcell contact policy. "
            "This negative control is expected to fail because Phase109 identified the current object/workcell "
            "contact as a frame or rail collision, not calibrated tabletop support."
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

    command = [args.python, *_phase110_args(args.output_dir, args.workcell_contact_policy)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and print one machine-readable container-stop observation."""

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aloha.container_stop_gate import validate_stop_observation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state_path", type=Path)
    parser.add_argument("recorder_pid", type=int)
    parser.add_argument("expected_recovery_id")
    args = parser.parse_args(argv)

    with args.state_path.open(encoding="utf-8") as state_file:
        payload = json.load(state_file)
    observation = validate_stop_observation(
        payload,
        recorder_pid=args.recorder_pid,
        expected_recovery_id=(
            None
            if args.expected_recovery_id == "-"
            else args.expected_recovery_id
        ),
    )
    fields = (
        observation.state,
        observation.recovery_id or "-",
        str(observation.owner_pid),
        observation.source,
        "true" if observation.safe_to_stop else "false",
    )
    print("|".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

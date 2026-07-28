#!/usr/bin/env python3
"""Validate the measured-preload ALOHA1 gripper hold gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--repeats", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.project_root.resolve(strict=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from tools.aloha1_mapping.gripper_force_runtime import validate_hold_v2
        from tools.aloha1_mapping.gripper_force_runtime import write_json

        report = validate_hold_v2(root, repeats=args.repeats)
        write_json(
            root / "reports/aloha1_mapping/gripper_force_diagnosis/hold_v2.json",
            report,
        )
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

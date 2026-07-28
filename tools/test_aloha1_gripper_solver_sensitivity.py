#!/usr/bin/env python3
"""Conditionally test ALOHA1 hold sensitivity to dt and solver iterations."""

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
        from tools.aloha1_mapping.gripper_force_runtime import build_root_cause_v2
        from tools.aloha1_mapping.gripper_force_runtime import test_solver_sensitivity
        from tools.aloha1_mapping.gripper_force_runtime import write_json
        from tools.aloha1_mapping.gripper_force_runtime import write_markdown

        solver = test_solver_sensitivity(root, repeats=args.repeats)
        write_json(
            root / "reports/aloha1_mapping/gripper_solver_sensitivity.json",
            solver,
        )
        root_cause, markdown = build_root_cause_v2(root)
        write_json(
            root / "reports/aloha1_mapping/gripper_hold_root_cause_v2.json",
            root_cause,
        )
        write_markdown(
            root / "reports/aloha1_mapping/gripper_hold_root_cause_v2.md",
            markdown,
        )
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

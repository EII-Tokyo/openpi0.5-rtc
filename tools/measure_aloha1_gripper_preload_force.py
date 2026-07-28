#!/usr/bin/env python3
"""Measure the ALOHA1 fixed-bottle gripper preload curve in Isaac Sim 5.1."""

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
        from tools.aloha1_mapping.gripper_force_runtime import measure_preload_force_curve
        from tools.aloha1_mapping.gripper_force_runtime import preload_markdown
        from tools.aloha1_mapping.gripper_force_runtime import write_json
        from tools.aloha1_mapping.gripper_force_runtime import write_markdown
        from tools.aloha1_mapping.gripper_force_runtime import write_preload_csv

        report = measure_preload_force_curve(root, repeats=args.repeats)
        base = root / "reports/aloha1_mapping/gripper_preload_force_curve"
        write_json(base.with_suffix(".json"), report)
        write_preload_csv(base.with_suffix(".csv"), report)
        write_markdown(base.with_suffix(".md"), preload_markdown(report))
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

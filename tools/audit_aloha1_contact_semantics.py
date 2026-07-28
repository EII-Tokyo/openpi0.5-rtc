#!/usr/bin/env python3
"""Audit ALOHA1 finger/bottle contact semantics in Isaac Sim 5.1."""

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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.project_root.resolve(strict=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from tools.aloha1_mapping.gripper_force_runtime import audit_contact_semantics
        from tools.aloha1_mapping.gripper_force_runtime import contact_semantics_markdown
        from tools.aloha1_mapping.gripper_force_runtime import write_json
        from tools.aloha1_mapping.gripper_force_runtime import write_markdown

        report = audit_contact_semantics(root)
        write_json(
            root / "reports/aloha1_mapping/gripper_contact_semantics.json",
            report,
        )
        write_markdown(
            root / "reports/aloha1_mapping/gripper_contact_semantics.md",
            contact_semantics_markdown(report),
        )
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

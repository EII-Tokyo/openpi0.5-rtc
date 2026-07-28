#!/usr/bin/env python3
"""Create the isolated supplier-CAD convex-hull Task 5 diagnostic asset."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-stage",
        type=Path,
        default=ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=(
            ROOT
            / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
            "viper_gripper/tessellation_angular_controlled/run_a"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "cad_finger_task5_convex_hull"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=(
            ROOT
            / "reports/aloha1_mapping/"
            "aloha_viper_cad_finger_task5_asset.json"
        ),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=(
            ROOT
            / "reports/aloha1_mapping/"
            "aloha_viper_cad_finger_task5_asset.md"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        from tools.aloha1_mapping.cad_finger_task5_asset import create_task5_diagnostic_asset
        from tools.aloha1_mapping.cad_finger_task5_asset import write_task5_asset_report

        report = create_task5_diagnostic_asset(
            source_stage_path=args.source_stage,
            left_obj_path=args.mesh_root / "left_finger.obj",
            right_obj_path=args.mesh_root / "right_finger.obj",
            output_root=args.output_root,
        )
        write_task5_asset_report(
            report,
            args.report_json,
            args.report_md,
        )
        print(f"status={report['status']}")
        print(
            "root_usd="
            f"{report['outputs']['root_usd']['absolute_path']}"
        )
        print(f"report_json={args.report_json.resolve()}")
        print(f"report_md={args.report_md.resolve()}")
        exit_code = 0 if report["status"] in {"PASS", "PARTIAL"} else 1
    except Exception:  # Kit's exception hook may otherwise exit zero.
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

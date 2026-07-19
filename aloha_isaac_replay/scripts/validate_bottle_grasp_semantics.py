from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_isaac_replay.validation.bottle_grasp_semantics import evaluate_grasp_file
from aloha_isaac_replay.validation.bottle_grasp_semantics import write_semantic_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRASP_YAML = REPO_ROOT / "assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/bottle_grasp_semantics_20260719"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Bottle500 grasp semantics before Isaac replay. This checks the bottle-local long axis, "
            "rear-quarter grasp location, side approach direction, and gripper closing-axis perpendicularity."
        )
    )
    parser.add_argument("--grasp-yaml", type=Path, default=DEFAULT_GRASP_YAML)
    parser.add_argument("--selected-grasp", default="grasp_rear_quarter")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = evaluate_grasp_file(args.grasp_yaml, selected_grasp=args.selected_grasp)
    output_dir = args.output_dir.resolve()
    output_json = output_dir / "bottle_grasp_semantics.json"
    output_md = output_dir / "bottle_grasp_semantics.md"
    write_semantic_report(result, output_json, output_md)
    print(
        json.dumps(
            {
                "status": "PASS" if result["pass"] else "FAILED_GATE",
                "selected_grasp": args.selected_grasp,
                "json": str(output_json.relative_to(REPO_ROOT)),
                "markdown": str(output_md.relative_to(REPO_ROOT)),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase103_two_finger_active_proxy_smoke_20260719"


def _phase103_args(output_dir: Path) -> list[str]:
    return [
        str(REPO_ROOT / "aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py"),
        "--stage-usd",
        "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda",
        "--stage-units-in-meters",
        "1.0",
        "--contact-proxy-profile",
        "scene_base_link",
        "--output-dir",
        str(output_dir),
        "--side",
        "left",
        "--settle-steps",
        "20",
        "--physics-dt",
        "0.02",
        "--gravity",
        "0.0",
        "--finger-kp",
        "200",
        "--finger-kd",
        "50",
        "--open-offset",
        "0.014",
        "--close-offset",
        "-0.014",
        "--right-finger-close-sign",
        "1.0",
        "--limit-margin",
        "0.001",
        "--object-fill-fraction",
        "0.90",
        "--object-placement",
        "gap_center",
        "--object-creation",
        "raw_usd",
        "--object-shape",
        "cube",
        "--object-contact-offset",
        "0.001",
        "--object-rest-offset",
        "0.0",
        "--proxy-contact-offset",
        "0.001",
        "--proxy-rest-offset",
        "0.0",
        "--support-plane-mode",
        "none",
        "--closure-profile",
        "linear",
        "--moving-fingers",
        "both",
        "--trace-contact-pairs",
        "--require-active-target-contact",
        "--min-contact-motion",
        "1e-05",
        "--max-object-displacement",
        "1.0",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-finger active-contact proxy smoke test. This validates bilateral close-phase "
            "target contact on scene_base_link ALOHA1 proxies, not a full Bottle500 grasp."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [args.python, *_phase103_args(args.output_dir)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

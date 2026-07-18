from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718"
)


def _phase97_args(output_dir: Path) -> list[str]:
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
        "-9.81",
        "--arm-kp",
        "1600",
        "--arm-kd",
        "100",
        "--finger-kp",
        "200",
        "--finger-kd",
        "50",
        "--object-fill-fraction",
        "0.6",
        "--object-placement",
        "gap_center",
        "--object-clearance",
        "0.001",
        "--object-creation",
        "raw_usd",
        "--object-shape",
        "bottle_usd",
        "--object-axis",
        "X",
        "--object-length-multiplier",
        "4.0",
        "--object-usd",
        "assets/bottle_500ml/isaac/bottle_500ml_sim.usd",
        "--object-usd-prim-path",
        "/Bottle500",
        "--object-contact-offset",
        "0.001",
        "--object-rest-offset",
        "0.0",
        "--support-plane-mode",
        "none",
        "--closure-profile",
        "abrupt",
        "--moving-fingers",
        "both",
        "--hdf5-gripper-episode",
        "local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_15c193959d7d449783517a9c9d257529/episode.hdf5",
        "--hdf5-replay-mode",
        "left_arm_and_gripper",
        "--hdf5-replay-actuation-mode",
        "drive_target",
        "--hdf5-replay-target-hold-steps",
        "1",
        "--mapping",
        "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml",
        "--hdf5-gripper-start-frame",
        "143",
        "--trace-contact-pairs",
        "--fail-on-non-target-object-contact",
        "--allowed-non-target-object-contact-category",
        "workcell_or_environment",
        "--already-in-contact-setup",
        "--min-contact-motion",
        "1e-05",
        "--max-object-displacement",
        "1.0",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current Phase97 ALOHA1 Isaac drive-target replay gate.")
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [args.python, *_phase97_args(args.output_dir)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

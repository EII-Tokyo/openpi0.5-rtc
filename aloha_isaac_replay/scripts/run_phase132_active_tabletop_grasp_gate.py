from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase132_active_tabletop_bottle_visual_cylinder_proxy_grasp_gate_20260719"
)


def _phase132_args(output_dir: Path, *, save_debug_stage: bool = False) -> list[str]:
    args = [
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
        "0.55",
        "--object-placement",
        "hdf5_open_finger_rear_quarter_tabletop",
        "--object-tabletop-top-z",
        "0.004086510930165169",
        "--object-tabletop-clearance",
        "0.001",
        "--object-rear-quarter-fraction",
        "0.25",
        "--object-rear-quarter-tolerance",
        "0.07",
        "--max-closing-long-axis-dot",
        "0.25",
        "--object-clearance",
        "0.001",
        "--object-creation",
        "raw_usd",
        "--object-shape",
        "bottle_usd_cylinder_proxy",
        "--object-axis",
        "X",
        "--object-length-multiplier",
        "4.0",
        "--object-contact-offset",
        "0.001",
        "--object-rest-offset",
        "0.0",
        "--support-plane-mode",
        "fixed_box",
        "--support-plane-center",
        "0.0",
        "0.0",
        "-0.015913489069834833",
        "--support-plane-size-x",
        "1.22",
        "--support-plane-size-y",
        "0.625",
        "--support-plane-thickness",
        "0.04",
        "--support-plane-clearance",
        "0.001",
        "--closure-profile",
        "abrupt",
        "--moving-fingers",
        "both",
        "--hdf5-gripper-episode",
        "local_rlt_data/raw_from_103/rollouts/key_regions/unknown_task/2026-06-17/rl/key_region_2b4324798b114b018aee8fc92580bccd/episode.hdf5",
        "--hdf5-replay-mode",
        "left_arm_and_gripper",
        "--hdf5-replay-actuation-mode",
        "drive_target",
        "--hdf5-replay-target-hold-steps",
        "1",
        "--mapping",
        "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml",
        "--hdf5-gripper-start-frame",
        "326",
        "--hdf5-gripper-end-frame",
        "360",
        "--trace-contact-pairs",
        "--fail-on-non-target-object-contact",
        "--allowed-non-target-object-contact-category",
        "workcell_or_environment",
        "--workcell-contact-policy",
        "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml",
        "--require-active-target-contact",
        "--min-contact-motion",
        "1e-05",
        "--max-object-displacement",
        "1.0",
    ]
    if save_debug_stage:
        args.append("--save-debug-stage")
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first active tabletop ALOHA1 left-gripper grasp gate that starts from an open HDF5 frame "
            "and requires first target contact during close. This uses BottleUSD for visual/semantic geometry "
            "and a cylinder body proxy for physics until the detailed BottleUSD collider is cleaned."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--save-debug-stage",
        action="store_true",
        help="Export the composed stage after object placement for visual inspection. This can create a large file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [args.python, *_phase132_args(args.output_dir, save_debug_stage=args.save_debug_stage)]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

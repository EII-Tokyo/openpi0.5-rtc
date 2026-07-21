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
MEASURED_TABLETOP_REFERENCE_PATH = "/World/Table"


def _phase132_args(
    output_dir: Path,
    *,
    save_debug_stage: bool = False,
    hdf5_start_frame: int = 208,
    hdf5_end_frame: int = 245,
    min_object_lift: float = 0.0,
    physics_dt: float = 0.02,
    target_hold_steps: int = 1,
    enable_contact_materials: bool = True,
    enforce_object_width_finger_stop: bool = True,
    object_effective_contact_width: float = 0.052,
    diagnostic_force_target_overlap: str = "none",
) -> list[str]:
    args = [
        str(REPO_ROOT / "aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py"),
        "--stage-usd",
        (
            "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/"
            "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe_inner_pad_proxy_runtime.usda"
        ),
        "--stage-units-in-meters",
        "1.0",
        "--contact-proxy-profile",
        "scene_base_link_inner_pad",
        "--output-dir",
        str(output_dir),
        "--side",
        "left",
        "--settle-steps",
        "20",
        "--physics-dt",
        str(physics_dt),
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
        "--object-tabletop-reference-path",
        MEASURED_TABLETOP_REFERENCE_PATH,
        "--object-tabletop-clearance",
        "0.001",
        "--derive-tabletop-top-z-from-open-finger-height",
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
        "--object-axis-source",
        "open_finger_horizontal_perpendicular",
        "--object-length-multiplier",
        "4.0",
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
        "abrupt",
        "--moving-fingers",
        "both",
        "--hdf5-gripper-episode",
        "/home/eii/project/bottles_data/episode_18.hdf5",
        "--hdf5-gripper-source",
        "qpos",
        "--hdf5-replay-mode",
        "hdf5_arm_start_then_gripper_only",
        "--hdf5-replay-actuation-mode",
        "drive_target",
        "--hdf5-replay-target-hold-steps",
        str(target_hold_steps),
        "--mapping",
        "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml",
        "--hdf5-gripper-start-frame",
        str(hdf5_start_frame),
        "--hdf5-gripper-end-frame",
        str(hdf5_end_frame),
        "--object-effective-contact-width",
        str(object_effective_contact_width),
        "--object-effective-contact-width-source",
        "soft_bottle_estimated_from_user_observation_20260721",
        "--finger-gap-projection-model",
        "oriented_box",
        "--trace-contact-pairs",
        "--fail-on-non-target-object-contact",
        "--allowed-non-target-object-contact-category",
        "workcell_or_environment",
        "--workcell-contact-policy",
        "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml",
        "--require-active-target-contact",
        "--min-contact-motion",
        "1e-05",
        "--min-object-lift",
        str(min_object_lift),
        "--max-object-displacement",
        "1.0",
    ]
    if enable_contact_materials:
        args.extend(
            [
                "--object-static-friction",
                "2.0",
                "--object-dynamic-friction",
                "1.5",
                "--object-restitution",
                "0.0",
                "--finger-static-friction",
                "2.0",
                "--finger-dynamic-friction",
                "1.5",
                "--finger-restitution",
                "0.0",
            ]
        )
    if enforce_object_width_finger_stop:
        args.append("--enforce-object-width-finger-stop")
    if diagnostic_force_target_overlap != "none":
        args.extend(["--diagnostic-force-target-overlap", diagnostic_force_target_overlap])
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
        "--hdf5-start-frame",
        type=int,
        default=208,
        help="First HDF5 frame to replay. Default preserves the episode_18 open-to-close tabletop grasp window.",
    )
    parser.add_argument(
        "--hdf5-end-frame",
        type=int,
        default=245,
        help=(
            "Exclusive HDF5 end frame. Default 245 covers episode_18 frame 208 through frame 244, the "
            "user-identified open-gripper to clamp/lift-onset window. Gripper-only mode must not be "
            "reported as a lift gate."
        ),
    )
    parser.add_argument(
        "--object-effective-contact-width",
        type=float,
        default=0.052,
        help=(
            "Soft Bottle500 contact width in meters. The visible bottle keeps the true 68mm external diameter; "
            "this narrower proxy models mineral-water bottle compression under ALOHA finger pressure."
        ),
    )
    parser.add_argument(
        "--min-object-lift",
        type=float,
        default=0.0,
        help=(
            "Minimum object center Z increase in meters. Keep 0 for contact-only validation; set a positive "
            "value only when the replay window includes a lift phase."
        ),
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=0.02,
        help=(
            "Isaac physics step. Default 0.02 keeps the historical 50 Hz contact gate; use 0.004 with "
            "--hdf5-replay-target-hold-steps 5 to run 250 Hz PhysX under each 50 Hz HDF5 target."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-target-hold-steps",
        type=int,
        default=1,
        help=(
            "Number of physics steps to hold each recorded HDF5 target. This must preserve zero-order hold; "
            "it does not interpolate or edit recorded targets."
        ),
    )
    parser.add_argument(
        "--disable-physical-contact-materials",
        action="store_true",
        help="Ablation only: do not bind explicit object/finger friction materials.",
    )
    parser.add_argument(
        "--disable-object-width-finger-stop",
        action="store_true",
        help="Ablation only: allow recorded finger targets to keep closing past the object width.",
    )
    parser.add_argument(
        "--save-debug-stage",
        action="store_true",
        help="Export the composed stage after object placement for visual inspection. This can create a large file.",
    )
    parser.add_argument(
        "--diagnostic-force-target-overlap",
        choices=("none", "nearest", "lower", "upper"),
        default="none",
        help=(
            "Diagnostic positive control only. It forces the target proxy into a finger to test contact "
            "reporting and prevents the run from being considered a formal Gate2 pass."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [
        args.python,
        *_phase132_args(
            args.output_dir,
            save_debug_stage=args.save_debug_stage,
            hdf5_start_frame=args.hdf5_start_frame,
            hdf5_end_frame=args.hdf5_end_frame,
            min_object_lift=args.min_object_lift,
            physics_dt=args.physics_dt,
            target_hold_steps=args.hdf5_replay_target_hold_steps,
            enable_contact_materials=not args.disable_physical_contact_materials,
            enforce_object_width_finger_stop=not args.disable_object_width_finger_stop,
            object_effective_contact_width=args.object_effective_contact_width,
            diagnostic_force_target_overlap=args.diagnostic_force_target_overlap,
        ),
    ]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

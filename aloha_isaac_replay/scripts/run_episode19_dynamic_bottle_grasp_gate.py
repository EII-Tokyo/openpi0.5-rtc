from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE19_HDF5 = Path("/home/eii/project/bottles_data/episode_19.hdf5")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/episode19_dynamic_bottle_grasp_gate_20260721"
)
ALOHA_REPLAY_RATE_HZ = 50.0
ALOHA_REPLAY_DT = 1.0 / ALOHA_REPLAY_RATE_HZ
DEFAULT_PHYSICS_RATE_HZ = 250.0
DEFAULT_PHYSICS_DT = 1.0 / DEFAULT_PHYSICS_RATE_HZ
DEFAULT_TARGET_HOLD_STEPS = int(DEFAULT_PHYSICS_RATE_HZ / ALOHA_REPLAY_RATE_HZ)
DEFAULT_MAX_COMMAND_TARGET_VELOCITY = 2.0
EPISODE19_LONG_VIDEO_APPROACH_FRAME = 2843
EPISODE19_LONG_VIDEO_LIFT_CONFIRM_FRAME = 3003
MEASURED_TABLETOP_REFERENCE_PATH = "/World/Table"


def _episode19_dynamic_args(
    output_dir: Path,
    *,
    hdf5_path: Path = EPISODE19_HDF5,
    save_debug_stage: bool = False,
    start_frame: int = EPISODE19_LONG_VIDEO_APPROACH_FRAME,
    end_frame: int = EPISODE19_LONG_VIDEO_LIFT_CONFIRM_FRAME + 1,
    settle_steps: int = 20,
    close_steps: int | None = None,
    object_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    already_in_contact_setup: bool = False,
    require_active_target_contact: bool = True,
    enforce_object_width_finger_stop: bool = True,
    physics_dt: float = DEFAULT_PHYSICS_DT,
    stage_time_codes_per_second: float = ALOHA_REPLAY_RATE_HZ,
    hdf5_replay_rate_hz: float = ALOHA_REPLAY_RATE_HZ,
    hdf5_replay_substep_mode: str = "zero_order_hold",
    arm_kp: float = 1600.0,
    arm_kd: float = 100.0,
    drive_profile_name: str = "episode19_long_video_pose_candidate_250hz",
    drive_profile_provenance: str = (
        "run_episode19_dynamic_bottle_grasp_gate.py:"
        "long_hdf5_video_open_close_lift_hypothesis_close2883"
    ),
    object_axis: str = "X",
    object_shape: str = "bottle_usd_cylinder_proxy",
    target_hold_steps: int = DEFAULT_TARGET_HOLD_STEPS,
    diagnostic_held_object_mode: str = "none",
    max_command_target_velocity: float | None = DEFAULT_MAX_COMMAND_TARGET_VELOCITY,
) -> list[str]:
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
        str(settle_steps),
        "--physics-dt",
        str(physics_dt),
        "--stage-time-codes-per-second",
        str(stage_time_codes_per_second),
        "--gravity",
        "-9.81",
        "--arm-kp",
        str(arm_kp),
        "--arm-kd",
        str(arm_kd),
        "--finger-kp",
        "200",
        "--finger-kd",
        "50",
        "--drive-profile-name",
        drive_profile_name,
        "--drive-profile-provenance",
        drive_profile_provenance,
        "--object-fill-fraction",
        "0.55",
        "--object-side-length",
        "0.068",
        "--object-placement",
        "hdf5_open_finger_rear_quarter_tabletop",
        "--object-tabletop-reference-path",
        MEASURED_TABLETOP_REFERENCE_PATH,
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
        "--object-center-offset",
        str(object_center_offset[0]),
        str(object_center_offset[1]),
        str(object_center_offset[2]),
        "--object-creation",
        "raw_usd",
        "--object-shape",
        object_shape,
        "--object-axis",
        object_axis,
        "--diagnostic-held-object-mode",
        diagnostic_held_object_mode,
        "--object-length-multiplier",
        "3.0294117647058822",
        "--object-contact-offset",
        "0.001",
        "--object-rest-offset",
        "0.0",
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
        "--support-plane-mode",
        "none",
        "--closure-profile",
        "abrupt",
        "--moving-fingers",
        "both",
        "--hdf5-gripper-episode",
        str(hdf5_path),
        "--hdf5-replay-mode",
        "left_arm_and_gripper",
        "--hdf5-replay-actuation-mode",
        "drive_target",
        "--hdf5-replay-substep-mode",
        hdf5_replay_substep_mode,
        "--hdf5-replay-target-hold-steps",
        str(target_hold_steps),
        "--hdf5-replay-rate-hz",
        str(hdf5_replay_rate_hz),
        "--mapping",
        "configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml",
        "--hdf5-gripper-start-frame",
        str(start_frame),
        "--hdf5-gripper-end-frame",
        str(end_frame),
        "--trace-contact-pairs",
        "--fail-on-non-target-object-contact",
        "--allowed-non-target-object-contact-category",
        "workcell_or_environment",
        "--workcell-contact-policy",
        "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml",
        "--min-contact-motion",
        "1e-05",
        "--min-object-lift",
        "0.005",
        "--max-object-displacement",
        "1.0",
    ]
    if max_command_target_velocity is not None:
        args.extend(["--max-command-target-velocity", str(max_command_target_velocity)])
    if already_in_contact_setup:
        args.append("--already-in-contact-setup")
    if close_steps is not None:
        args.extend(["--close-steps", str(close_steps)])
    if require_active_target_contact:
        args.append("--require-active-target-contact")
    if enforce_object_width_finger_stop:
        args.append("--enforce-object-width-finger-stop")
    if save_debug_stage:
        args.append("--save-debug-stage")
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the episode_19 dynamic Bottle500 visual + cylinder physics proxy tabletop grasp gate. "
            "This requires target finger contact during close and object lift, so it is a stronger check "
            "than the fixed visual replay."
        )
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv_issac/bin/python"),
        help="Python executable with Isaac Sim installed. Defaults to the project Isaac virtualenv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=EPISODE19_HDF5,
        help=(
            "Source HDF5 episode. Defaults to the historical episode_19 path for compatibility, "
            "but candidate replay validation should pass the selected HDF5 explicitly."
        ),
    )
    parser.add_argument("--save-debug-stage", action="store_true")
    parser.add_argument("--start-frame", type=int, default=EPISODE19_LONG_VIDEO_APPROACH_FRAME)
    parser.add_argument("--end-frame", type=int, default=EPISODE19_LONG_VIDEO_LIFT_CONFIRM_FRAME + 1)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument(
        "--close-steps",
        type=int,
        default=None,
        help=(
            "Optional validator close-step cap. If omitted, the validator default applies. "
            "For HDF5 replay this caps target frames before the per-target hold. Leave omitted for full windows."
        ),
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=DEFAULT_PHYSICS_DT,
        help=(
            "Isaac physics timestep. Default is 0.004s: 250 Hz physics with each 50 Hz HDF5 target "
            "held for 5 physics steps."
        ),
    )
    parser.add_argument(
        "--stage-time-codes-per-second",
        type=float,
        default=ALOHA_REPLAY_RATE_HZ,
        help="USD timeline metadata for the replay stage. Default is 50 to match ALOHA rollout frames.",
    )
    parser.add_argument(
        "--hdf5-replay-rate-hz",
        type=float,
        default=ALOHA_REPLAY_RATE_HZ,
        help="Nominal HDF5 qpos frame rate. ALOHA rollout data is expected to be 50 Hz.",
    )
    parser.add_argument(
        "--object-center-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "World-frame object center offset passed through to the validator. "
            "Use this for controlled placement sweeps; the offset is reported in the metrics."
        ),
    )
    parser.add_argument("--target-hold-steps", type=int, default=DEFAULT_TARGET_HOLD_STEPS)
    parser.add_argument(
        "--max-command-target-velocity",
        type=float,
        default=DEFAULT_MAX_COMMAND_TARGET_VELOCITY,
        help=(
            "Formal replay feasibility gate for absolute 50 Hz command target velocity in rad/s. "
            "The default exposes target jumps before tuning CCD, collider detail, or drive gains."
        ),
    )
    parser.add_argument(
        "--disable-command-smoothness-gate",
        action="store_true",
        help="Diagnostic only: report command smoothness without failing the replay feasibility gate.",
    )
    parser.add_argument(
        "--hdf5-replay-substep-mode",
        choices=("zero_order_hold", "linear_interpolation_diagnostic"),
        default="zero_order_hold",
        help=(
            "Substep target mode for each 50 Hz HDF5 frame. zero_order_hold is formal replay; "
            "linear_interpolation_diagnostic is an ablation only."
        ),
    )
    parser.add_argument("--arm-kp", type=float, default=1600.0)
    parser.add_argument("--arm-kd", type=float, default=100.0)
    parser.add_argument("--drive-profile-name", default="episode19_long_video_pose_candidate_250hz")
    parser.add_argument(
        "--drive-profile-provenance",
        default=(
            "run_episode19_dynamic_bottle_grasp_gate.py:"
            "long_hdf5_video_open_close_lift_hypothesis_close2883"
        ),
    )
    parser.add_argument("--object-axis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument(
        "--object-shape",
        choices=(
            "bottle_usd",
            "bottle_usd_cylinder_proxy",
            "bottle_usd_segmented_proxy",
            "bottle_usd_grasp_band_proxy",
            "bottle_usd_grasp_box_proxy",
        ),
        default="bottle_usd_cylinder_proxy",
        help=(
            "Runtime Bottle500 collision representation. bottle_usd uses the asset's own collision prims; "
            "bottle_usd_cylinder_proxy uses one explicit cylindrical physics proxy; "
            "bottle_usd_segmented_proxy uses separate body, neck, and mouth proxy colliders; "
            "bottle_usd_grasp_band_proxy uses a local bottle-body contact band for diagnostic grasp isolation; "
            "bottle_usd_grasp_box_proxy uses a local box coupon for soft-bottle grasp stability tests."
        ),
    )
    parser.add_argument(
        "--diagnostic-held-object-mode",
        choices=("none", "follow_gripper", "follow_after_bilateral_contact"),
        default="none",
        help=(
            "Diagnostic only. follow_after_bilateral_contact starts held-object replay only after both "
            "expected finger CONTACT_FOUND events happen during close. This is not dynamic grasp proof."
        ),
    )
    parser.add_argument("--already-in-contact-setup", action="store_true")
    parser.add_argument("--no-require-active-target-contact", action="store_true")
    parser.add_argument(
        "--no-enforce-object-width-finger-stop",
        action="store_true",
        help=(
            "Diagnostic only. Disables the target guard that prevents finger commands from closing past "
            "the bottle body width. Use to detect whether apparent lift depends on unphysical penetration."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running Isaac.")
    args = parser.parse_args()

    command = [
        args.python,
        *_episode19_dynamic_args(
            args.output_dir,
            hdf5_path=args.hdf5,
            save_debug_stage=args.save_debug_stage,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            settle_steps=args.settle_steps,
            close_steps=args.close_steps,
            object_center_offset=tuple(float(v) for v in args.object_center_offset),
            already_in_contact_setup=args.already_in_contact_setup,
            require_active_target_contact=not args.no_require_active_target_contact,
            enforce_object_width_finger_stop=not args.no_enforce_object_width_finger_stop,
            physics_dt=args.physics_dt,
            stage_time_codes_per_second=args.stage_time_codes_per_second,
            hdf5_replay_rate_hz=args.hdf5_replay_rate_hz,
            hdf5_replay_substep_mode=args.hdf5_replay_substep_mode,
            arm_kp=args.arm_kp,
            arm_kd=args.arm_kd,
            drive_profile_name=args.drive_profile_name,
            drive_profile_provenance=args.drive_profile_provenance,
            object_axis=args.object_axis,
            object_shape=args.object_shape,
            target_hold_steps=args.target_hold_steps,
            diagnostic_held_object_mode=args.diagnostic_held_object_mode,
            max_command_target_velocity=(
                None if args.disable_command_smoothness_gate else args.max_command_target_velocity
            ),
        ),
    ]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

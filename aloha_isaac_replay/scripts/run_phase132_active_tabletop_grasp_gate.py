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
    post_close_hold_steps: int = 0,
    post_close_lift_source: str = "none",
    post_close_lift_hdf5_start_frame: int | None = None,
    post_close_lift_hdf5_end_frame: int | None = None,
    post_close_lift_gripper_mode: str = "hold_final_close",
    post_close_lift_height: float = 0.01,
    post_close_lift_steps: int = 40,
    post_close_lift_hold_steps: int = 0,
    post_close_lift_jacobian_eps: float = 0.002,
    post_close_lift_jacobian_damping: float = 1e-6,
    post_close_lift_max_joint_delta: float = 0.08,
    diagnostic_loaded_clamp_squeeze_depth: float = 0.0,
    finger_kp: float = 200.0,
    finger_kd: float = 50.0,
    enable_contact_materials: bool = True,
    enforce_object_width_finger_stop: bool = True,
    object_effective_contact_width: float = 0.052,
    enable_episode18_loaded_qpos_calibration: bool = False,
    diagnostic_force_target_overlap: str = "none",
    diagnostic_force_target_overlap_m: float = 0.001,
    placement_basis: str = "open",
    tabletop_mode: str = "diagnostic_shift_to_open_finger",
    hdf5_replay_mode: str = "hdf5_arm_start_then_gripper_only",
    hdf5_arm_hold_frame_offset: int = 0,
    hdf5_replay_substep_mode: str = "zero_order_hold",
    object_shape: str = "bottle_usd_cylinder_proxy",
    object_clearance: float = 0.001,
    closing_axis_gap_solver_basis: str = "open",
    object_width_stop_predictive_margin: float = 0.0,
    object_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[str]:
    if placement_basis == "open":
        object_placement = "hdf5_open_finger_rear_quarter_tabletop"
    elif placement_basis == "close":
        object_placement = "hdf5_close_finger_rear_quarter_tabletop"
    else:
        raise ValueError(f"Unsupported placement_basis: {placement_basis!r}")
    if tabletop_mode not in {"diagnostic_shift_to_open_finger", "fixed_reference"}:
        raise ValueError(f"Unsupported tabletop_mode: {tabletop_mode!r}")
    if hdf5_replay_mode not in {"hdf5_arm_start_then_gripper_only", "left_arm_and_gripper"}:
        raise ValueError(f"Unsupported hdf5_replay_mode: {hdf5_replay_mode!r}")

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
        str(float(finger_kp)),
        "--finger-kd",
        str(float(finger_kd)),
        "--object-fill-fraction",
        "0.55",
        "--object-placement",
        object_placement,
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
        str(float(object_clearance)),
        "--closing-axis-gap-solver-basis",
        closing_axis_gap_solver_basis,
        "--object-width-stop-predictive-margin",
        str(float(object_width_stop_predictive_margin)),
        "--object-creation",
        "raw_usd",
        "--object-shape",
        object_shape,
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
        hdf5_replay_mode,
        "--hdf5-arm-hold-frame-offset",
        str(int(hdf5_arm_hold_frame_offset)),
        "--hdf5-replay-actuation-mode",
        "drive_target",
        "--hdf5-replay-substep-mode",
        hdf5_replay_substep_mode,
        "--hdf5-replay-target-hold-steps",
        str(target_hold_steps),
        "--post-close-hold-steps",
        str(int(post_close_hold_steps)),
        "--post-close-lift-source",
        post_close_lift_source,
        "--post-close-lift-gripper-mode",
        post_close_lift_gripper_mode,
        "--post-close-lift-height",
        str(float(post_close_lift_height)),
        "--post-close-lift-steps",
        str(int(post_close_lift_steps)),
        "--post-close-lift-hold-steps",
        str(int(post_close_lift_hold_steps)),
        "--post-close-lift-jacobian-eps",
        str(float(post_close_lift_jacobian_eps)),
        "--post-close-lift-jacobian-damping",
        str(float(post_close_lift_jacobian_damping)),
        "--post-close-lift-max-joint-delta",
        str(float(post_close_lift_max_joint_delta)),
        "--diagnostic-loaded-clamp-squeeze-depth",
        str(float(diagnostic_loaded_clamp_squeeze_depth)),
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
    if post_close_lift_hdf5_end_frame is not None:
        args.extend(["--post-close-lift-hdf5-end-frame", str(int(post_close_lift_hdf5_end_frame))])
    if post_close_lift_hdf5_start_frame is not None:
        args.extend(["--post-close-lift-hdf5-start-frame", str(int(post_close_lift_hdf5_start_frame))])
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
    if tabletop_mode == "diagnostic_shift_to_open_finger":
        args.append("--derive-tabletop-top-z-from-open-finger-height")
    if enable_episode18_loaded_qpos_calibration:
        args.extend(
            [
                "--loaded-gripper-qpos-calibration-mode",
                "affine_open_to_contact",
                "--loaded-gripper-open-qpos",
                "0.9473305344581604",
                "--loaded-gripper-contact-qpos",
                "0.5712134838104248",
                "--loaded-gripper-effective-contact-width",
                str(object_effective_contact_width),
                "--loaded-gripper-open-standard",
                "1.0",
                "--loaded-gripper-qpos-calibration-source",
                "episode18_frames_208_220_open_max_and_236_244_loaded_plateau_20260722",
            ]
        )
    if diagnostic_force_target_overlap != "none":
        args.extend(["--diagnostic-force-target-overlap", diagnostic_force_target_overlap])
        args.extend(["--diagnostic-force-target-overlap-m", str(float(diagnostic_force_target_overlap_m))])
    if any(abs(float(v)) > 0.0 for v in object_center_offset):
        args.extend(["--object-center-offset", *(str(float(v)) for v in object_center_offset)])
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
        "--hdf5-replay-substep-mode",
        choices=("zero_order_hold", "linear_interpolation_diagnostic"),
        default="zero_order_hold",
        help=(
            "How each recorded HDF5 target is applied across target-hold physics substeps. zero_order_hold "
            "is the formal replay path; linear_interpolation_diagnostic is a non-formal target-jump ablation."
        ),
    )
    parser.add_argument(
        "--post-close-hold-steps",
        type=int,
        default=0,
        help=(
            "Extra physics steps to hold the final close target after the recorded HDF5 window. "
            "Default 0 preserves the raw replay; positive values test clamp persistence without editing frames."
        ),
    )
    parser.add_argument(
        "--post-close-lift-source",
        choices=("none", "hdf5_continuation", "jacobian_vertical"),
        default="none",
        help=(
            "Diagnostic only. Append a post-hold lift phase after the raw close window. hdf5_continuation "
            "uses subsequent episode_18 arm qpos targets; jacobian_vertical generates a small local upward "
            "joint-space diagnostic. Neither is a formal close replay pass."
        ),
    )
    parser.add_argument(
        "--post-close-lift-hdf5-start-frame",
        type=int,
        default=None,
        help="Optional episode_18 anchor frame for hdf5_continuation lift diagnostics.",
    )
    parser.add_argument(
        "--post-close-lift-hdf5-end-frame",
        type=int,
        default=None,
        help="Exclusive episode_18 frame for hdf5_continuation lift diagnostics.",
    )
    parser.add_argument(
        "--post-close-lift-gripper-mode",
        choices=("hold_final_close", "hdf5_qpos"),
        default="hold_final_close",
        help="How to drive the gripper during post-close lift diagnostics.",
    )
    parser.add_argument("--post-close-lift-height", type=float, default=0.01)
    parser.add_argument("--post-close-lift-steps", type=int, default=40)
    parser.add_argument("--post-close-lift-hold-steps", type=int, default=0)
    parser.add_argument("--post-close-lift-jacobian-eps", type=float, default=0.002)
    parser.add_argument("--post-close-lift-jacobian-damping", type=float, default=1e-6)
    parser.add_argument("--post-close-lift-max-joint-delta", type=float, default=0.08)
    parser.add_argument(
        "--finger-kp",
        type=float,
        default=200.0,
        help=(
            "Runtime finger drive stiffness passed to the validator. Default preserves the phase132 "
            "reference setting; use explicit values only for drive-authority diagnostics."
        ),
    )
    parser.add_argument(
        "--finger-kd",
        type=float,
        default=50.0,
        help=(
            "Runtime finger drive damping passed to the validator. Default preserves the phase132 reference "
            "setting; use explicit values only for drive-authority diagnostics."
        ),
    )
    parser.add_argument(
        "--diagnostic-loaded-clamp-squeeze-depth",
        type=float,
        default=0.0,
        help=(
            "Diagnostic-only total extra closing distance in meters for post-close hold/lift. "
            "Each finger closes half this amount; original HDF5 close frames are unchanged."
        ),
    )
    parser.add_argument(
        "--hdf5-replay-mode",
        choices=("hdf5_arm_start_then_gripper_only", "left_arm_and_gripper"),
        default="hdf5_arm_start_then_gripper_only",
        help=(
            "HDF5 replay semantics. The default keeps the historical contact-isolation diagnostic: replay "
            "the start arm pose and only close the gripper. Use left_arm_and_gripper for the user-confirmed "
            "episode_18 frame 208-244 grasp window, where the left arm also moves during clamp/lift onset."
        ),
    )
    parser.add_argument(
        "--hdf5-arm-hold-frame-offset",
        type=int,
        default=0,
        help=(
            "For hdf5_arm_start_then_gripper_only: selected-window frame offset whose arm qpos is held. "
            "Default 0 holds frame 208; use 28 to hold episode_18 frame 236 while replaying the gripper sequence."
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
        "--enable-episode18-loaded-qpos-calibration",
        action="store_true",
        help=(
            "Diagnostic only: interpret episode_18 loaded qpos plateau as the soft-bottle contact-width "
            "anchor. This preserves the raw HDF5 frame window and does not make the run a formal lift gate."
        ),
    )
    parser.add_argument(
        "--placement-basis",
        choices=("open", "close"),
        default="open",
        help=(
            "How to infer tabletop bottle placement from episode_18. 'open' uses the open-frame approach "
            "pose; 'close' uses the loaded/clamped frame for a stricter diagnostic of the actual grasp band. "
            "Both modes keep the bottle on the tabletop and do not edit recorded HDF5 frames."
        ),
    )
    parser.add_argument(
        "--tabletop-mode",
        choices=("diagnostic_shift_to_open_finger", "fixed_reference"),
        default="diagnostic_shift_to_open_finger",
        help=(
            "Tabletop reset contract. diagnostic_shift_to_open_finger preserves the historical contact-proxy "
            "diagnostic by moving /World/Table to the open-finger height. fixed_reference keeps /World/Table "
            "fixed and places the bottle on the measured table collider; this is the RL-ready semantics."
        ),
    )
    parser.add_argument(
        "--object-center-offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Diagnostic world-frame offset in meters passed through to the validator. Use this only for "
            "pose-calibration sweeps; the default is zero and does not alter formal replay placement."
        ),
    )
    parser.add_argument(
        "--object-shape",
        choices=(
            "bottle_usd_cylinder_proxy",
            "bottle_usd_segmented_proxy",
            "bottle_usd_grasp_band_proxy",
            "bottle_usd_grasp_box_proxy",
        ),
        default="bottle_usd_cylinder_proxy",
        help=(
            "BottleUSD visual with a simplified physics proxy. Default keeps the current full cylinder proxy; "
            "grasp_band_proxy and grasp_box_proxy are controlled diagnostics for local soft-bottle contact validation."
        ),
    )
    parser.add_argument(
        "--object-clearance",
        type=float,
        default=0.001,
        help=(
            "Clearance margin in meters for the validator's object-width finger-stop guard. "
            "Default preserves the historical 1 mm guard."
        ),
    )
    parser.add_argument(
        "--closing-axis-gap-solver-basis",
        choices=("open", "placement"),
        default="open",
        help=(
            "Finger boxes used by the validator's closing-axis gap solver. "
            "open preserves historical behavior; placement lets close placement "
            "center the contact proxy in the close-frame gap."
        ),
    )
    parser.add_argument(
        "--object-width-stop-predictive-margin",
        type=float,
        default=0.0,
        help=(
            "Extra early-stop margin in meters for the validator's object-width finger-stop guard."
        ),
    )
    parser.add_argument(
        "--save-debug-stage",
        action="store_true",
        help="Export the composed stage after object placement for visual inspection. This can create a large file.",
    )
    parser.add_argument(
        "--diagnostic-force-target-overlap",
        choices=("none", "nearest", "lower", "upper", "left_finger", "right_finger"),
        default="none",
        help=(
            "Diagnostic positive control only. It forces the target proxy into a finger to test contact "
            "reporting and prevents the run from being considered a formal Gate2 pass."
        ),
    )
    parser.add_argument(
        "--diagnostic-force-target-overlap-m",
        type=float,
        default=0.001,
        help="Diagnostic overlap depth in meters for --diagnostic-force-target-overlap.",
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
            post_close_hold_steps=args.post_close_hold_steps,
            post_close_lift_source=args.post_close_lift_source,
            post_close_lift_hdf5_start_frame=args.post_close_lift_hdf5_start_frame,
            post_close_lift_hdf5_end_frame=args.post_close_lift_hdf5_end_frame,
            post_close_lift_gripper_mode=args.post_close_lift_gripper_mode,
        post_close_lift_height=args.post_close_lift_height,
        post_close_lift_steps=args.post_close_lift_steps,
        post_close_lift_hold_steps=args.post_close_lift_hold_steps,
        post_close_lift_jacobian_eps=args.post_close_lift_jacobian_eps,
            post_close_lift_jacobian_damping=args.post_close_lift_jacobian_damping,
            post_close_lift_max_joint_delta=args.post_close_lift_max_joint_delta,
            diagnostic_loaded_clamp_squeeze_depth=args.diagnostic_loaded_clamp_squeeze_depth,
            finger_kp=args.finger_kp,
            finger_kd=args.finger_kd,
            enable_contact_materials=not args.disable_physical_contact_materials,
            enforce_object_width_finger_stop=not args.disable_object_width_finger_stop,
            object_effective_contact_width=args.object_effective_contact_width,
            enable_episode18_loaded_qpos_calibration=args.enable_episode18_loaded_qpos_calibration,
            diagnostic_force_target_overlap=args.diagnostic_force_target_overlap,
            diagnostic_force_target_overlap_m=args.diagnostic_force_target_overlap_m,
            placement_basis=args.placement_basis,
            tabletop_mode=args.tabletop_mode,
            hdf5_replay_mode=args.hdf5_replay_mode,
            hdf5_arm_hold_frame_offset=args.hdf5_arm_hold_frame_offset,
            hdf5_replay_substep_mode=args.hdf5_replay_substep_mode,
            object_shape=args.object_shape,
            object_clearance=args.object_clearance,
            closing_axis_gap_solver_basis=args.closing_axis_gap_solver_basis,
            object_width_stop_predictive_margin=args.object_width_stop_predictive_margin,
            object_center_offset=tuple(args.object_center_offset),
        ),
    ]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

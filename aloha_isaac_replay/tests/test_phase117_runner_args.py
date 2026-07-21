from pathlib import Path

from aloha_isaac_replay.scripts.run_phase117_diagnostic_held_bottle_replay import _phase117_args
from aloha_isaac_replay.scripts.run_phase132_active_tabletop_grasp_gate import _phase132_args
from aloha_isaac_replay.scripts.run_episode19_dynamic_bottle_grasp_gate import _episode19_dynamic_args


def test_phase117_diagnostic_runner_uses_kinematic_held_object_boundary() -> None:
    args = _phase117_args(Path("out"), Path("policy.yaml"), start_frame=80)

    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "80"
    assert args[args.index("--object-placement") + 1] == "grasp_yaml"
    assert args[args.index("--object-grasp-name") + 1] == "grasp_rear_quarter"
    assert args[args.index("--diagnostic-held-object-mode") + 1] == "follow_gripper"
    assert args[args.index("--support-plane-mode") + 1] == "none"
    assert "--disable-object-rigid-body" in args

    assert "--trace-contact-pairs" not in args
    assert "--fail-on-non-target-object-contact" not in args
    assert "--workcell-contact-policy" not in args
    assert "--already-in-contact-setup" not in args


def test_phase132_active_tabletop_runner_uses_open_frame_proxy_body_grasp() -> None:
    args = _phase132_args(Path("out"))

    assert args[args.index("--object-placement") + 1] == "hdf5_open_finger_rear_quarter_tabletop"
    assert args[args.index("--object-shape") + 1] == "bottle_usd_cylinder_proxy"
    assert args[args.index("--object-axis-source") + 1] == "open_finger_horizontal_perpendicular"
    assert args[args.index("--object-effective-contact-width") + 1] == "0.052"
    assert args[args.index("--object-effective-contact-width-source") + 1] == (
        "soft_bottle_estimated_from_user_observation_20260721"
    )
    assert args[args.index("--finger-gap-projection-model") + 1] == "oriented_box"
    assert args[args.index("--contact-proxy-profile") + 1] == "scene_base_link_inner_pad"
    assert args[args.index("--hdf5-gripper-episode") + 1] == "/home/eii/project/bottles_data/episode_18.hdf5"
    assert args[args.index("--hdf5-gripper-source") + 1] == "qpos"
    assert args[args.index("--hdf5-replay-mode") + 1] == "hdf5_arm_start_then_gripper_only"
    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "208"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "245"
    assert args[args.index("--min-object-lift") + 1] == "0.0"
    assert args[args.index("--physics-dt") + 1] == "0.02"
    assert args[args.index("--hdf5-replay-target-hold-steps") + 1] == "1"
    assert args[args.index("--object-static-friction") + 1] == "2.0"
    assert args[args.index("--object-dynamic-friction") + 1] == "1.5"
    assert args[args.index("--finger-static-friction") + 1] == "2.0"
    assert args[args.index("--finger-dynamic-friction") + 1] == "1.5"
    assert args[args.index("--proxy-contact-offset") + 1] == "0.001"
    assert args[args.index("--proxy-rest-offset") + 1] == "0.0"
    assert "--enforce-object-width-finger-stop" in args
    assert args[args.index("--max-closing-long-axis-dot") + 1] == "0.25"
    assert "--object-tabletop-top-z" not in args
    assert args[args.index("--object-tabletop-reference-path") + 1] == "/World/Table"
    assert "--derive-tabletop-top-z-from-open-finger-height" in args
    assert args[args.index("--support-plane-mode") + 1] == "none"
    assert args[args.index("--workcell-contact-policy") + 1] == (
        "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml"
    )
    assert "--require-active-target-contact" in args
    assert "--already-in-contact-setup" not in args
    assert "--save-debug-stage" not in args


def test_phase132_can_opt_in_to_debug_stage_export() -> None:
    args = _phase132_args(Path("out"), save_debug_stage=True)

    assert "--save-debug-stage" in args


def test_phase132_can_extend_to_recorded_hdf5_lift_gate() -> None:
    args = _phase132_args(
        Path("out"),
        hdf5_start_frame=220,
        hdf5_end_frame=520,
        min_object_lift=0.02,
        physics_dt=0.004,
        target_hold_steps=5,
    )

    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "220"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "520"
    assert args[args.index("--min-object-lift") + 1] == "0.02"
    assert args[args.index("--physics-dt") + 1] == "0.004"
    assert args[args.index("--hdf5-replay-target-hold-steps") + 1] == "5"


def test_phase132_can_disable_physical_material_and_width_stop_for_ablation() -> None:
    args = _phase132_args(
        Path("out"),
        enable_contact_materials=False,
        enforce_object_width_finger_stop=False,
    )

    assert "--object-static-friction" not in args
    assert "--finger-static-friction" not in args
    assert "--enforce-object-width-finger-stop" not in args


def test_phase132_can_select_soft_bottle_effective_contact_width() -> None:
    args = _phase132_args(Path("out"), object_effective_contact_width=0.052)

    assert args[args.index("--object-effective-contact-width") + 1] == "0.052"


def test_phase132_can_opt_in_to_forced_overlap_diagnostic() -> None:
    default_args = _phase132_args(Path("out"))
    assert "--diagnostic-force-target-overlap" not in default_args

    diagnostic_args = _phase132_args(Path("out"), diagnostic_force_target_overlap="nearest")
    assert diagnostic_args[diagnostic_args.index("--diagnostic-force-target-overlap") + 1] == "nearest"


def test_episode19_dynamic_runner_uses_bottle_visual_proxy_and_lift_gate() -> None:
    args = _episode19_dynamic_args(Path("out"))

    assert args[args.index("--hdf5-gripper-episode") + 1] == "/home/eii/project/bottles_data/episode_19.hdf5"
    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "2843"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "3004"
    assert args[args.index("--physics-dt") + 1] == "0.004"
    assert args[args.index("--stage-time-codes-per-second") + 1] == "50.0"
    assert args[args.index("--hdf5-replay-target-hold-steps") + 1] == "5"
    assert args[args.index("--hdf5-replay-rate-hz") + 1] == "50.0"
    assert args[args.index("--hdf5-replay-substep-mode") + 1] == "zero_order_hold"
    assert args[args.index("--max-command-target-velocity") + 1] == "2.0"
    assert args[args.index("--drive-profile-name") + 1] == "episode19_long_video_pose_candidate_250hz"
    assert (
        args[args.index("--drive-profile-provenance") + 1]
        == "run_episode19_dynamic_bottle_grasp_gate.py:long_hdf5_video_open_close_lift_hypothesis_close2883"
    )
    assert args[args.index("--arm-kp") + 1] == "1600.0"
    assert args[args.index("--arm-kd") + 1] == "100.0"
    assert "--close-steps" not in args
    assert args[args.index("--object-placement") + 1] == "hdf5_open_finger_rear_quarter_tabletop"
    assert args[args.index("--object-shape") + 1] == "bottle_usd_cylinder_proxy"
    assert args[args.index("--object-side-length") + 1] == "0.068"
    assert args[args.index("--diagnostic-held-object-mode") + 1] == "none"
    assert args[args.index("--object-length-multiplier") + 1] == "3.0294117647058822"
    assert args[args.index("--object-static-friction") + 1] == "2.0"
    assert args[args.index("--object-dynamic-friction") + 1] == "1.5"
    assert args[args.index("--object-restitution") + 1] == "0.0"
    assert args[args.index("--finger-static-friction") + 1] == "2.0"
    assert args[args.index("--finger-dynamic-friction") + 1] == "1.5"
    assert args[args.index("--finger-restitution") + 1] == "0.0"
    assert "--object-tabletop-top-z" not in args
    assert args[args.index("--object-tabletop-reference-path") + 1] == "/World/Table"
    assert args[args.index("--support-plane-mode") + 1] == "none"
    assert args[args.index("--object-rear-quarter-fraction") + 1] == "0.25"
    assert args[args.index("--min-object-lift") + 1] == "0.005"
    assert "--enforce-object-width-finger-stop" in args
    assert "--require-active-target-contact" in args
    assert "--fail-on-non-target-object-contact" in args


def test_episode19_dynamic_runner_can_select_replay_window_and_contact_mode() -> None:
    args = _episode19_dynamic_args(
        Path("out"),
        hdf5_path=Path("/tmp/episode_10.hdf5"),
        start_frame=490,
        end_frame=900,
        settle_steps=7,
        close_steps=4096,
        object_center_offset=(0.0, 0.012, -0.003),
        already_in_contact_setup=True,
        require_active_target_contact=False,
        physics_dt=0.01,
        target_hold_steps=2,
    )

    assert args[args.index("--hdf5-gripper-episode") + 1] == "/tmp/episode_10.hdf5"
    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "490"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "900"
    assert args[args.index("--settle-steps") + 1] == "7"
    assert args[args.index("--close-steps") + 1] == "4096"
    assert args[args.index("--physics-dt") + 1] == "0.01"
    offset_index = args.index("--object-center-offset")
    assert args[offset_index + 1 : offset_index + 4] == ["0.0", "0.012", "-0.003"]
    assert args[args.index("--hdf5-replay-target-hold-steps") + 1] == "2"
    assert "--already-in-contact-setup" in args
    assert "--require-active-target-contact" not in args


def test_episode19_dynamic_runner_can_disable_width_stop_for_diagnostic_only() -> None:
    args = _episode19_dynamic_args(Path("out"), enforce_object_width_finger_stop=False)

    assert "--enforce-object-width-finger-stop" not in args


def test_episode19_dynamic_runner_can_select_object_long_axis() -> None:
    args = _episode19_dynamic_args(Path("out"), object_axis="Y")

    assert args[args.index("--object-axis") + 1] == "Y"


def test_episode19_dynamic_runner_can_select_contact_triggered_diagnostic_hold() -> None:
    args = _episode19_dynamic_args(Path("out"), diagnostic_held_object_mode="follow_after_bilateral_contact")

    assert args[args.index("--diagnostic-held-object-mode") + 1] == "follow_after_bilateral_contact"


def test_episode19_dynamic_runner_can_select_diagnostic_substep_interpolation() -> None:
    args = _episode19_dynamic_args(Path("out"), hdf5_replay_substep_mode="linear_interpolation_diagnostic")

    assert args[args.index("--hdf5-replay-substep-mode") + 1] == "linear_interpolation_diagnostic"


def test_episode19_dynamic_runner_can_disable_command_smoothness_gate_for_diagnostic_only() -> None:
    args = _episode19_dynamic_args(Path("out"), max_command_target_velocity=None)

    assert "--max-command-target-velocity" not in args


def test_episode19_dynamic_runner_can_select_named_drive_profile() -> None:
    args = _episode19_dynamic_args(
        Path("out"),
        arm_kp=2400.0,
        arm_kd=50.0,
        drive_profile_name="episode19_250hz_arm_fast_tracking_v1",
        drive_profile_provenance="expert_recommended_after_drive_authority_audit_20260721",
    )

    assert args[args.index("--arm-kp") + 1] == "2400.0"
    assert args[args.index("--arm-kd") + 1] == "50.0"
    assert args[args.index("--drive-profile-name") + 1] == "episode19_250hz_arm_fast_tracking_v1"
    assert args[args.index("--drive-profile-provenance") + 1] == (
        "expert_recommended_after_drive_authority_audit_20260721"
    )


def test_episode19_dynamic_runner_can_select_native_bottle_usd_collision() -> None:
    args = _episode19_dynamic_args(Path("out"), object_shape="bottle_usd")

    assert args[args.index("--object-shape") + 1] == "bottle_usd"


def test_episode19_dynamic_runner_can_select_segmented_bottle_usd_proxy() -> None:
    args = _episode19_dynamic_args(Path("out"), object_shape="bottle_usd_segmented_proxy")

    assert args[args.index("--object-shape") + 1] == "bottle_usd_segmented_proxy"


def test_episode19_dynamic_runner_can_select_grasp_band_bottle_usd_proxy() -> None:
    args = _episode19_dynamic_args(Path("out"), object_shape="bottle_usd_grasp_band_proxy")

    assert args[args.index("--object-shape") + 1] == "bottle_usd_grasp_band_proxy"

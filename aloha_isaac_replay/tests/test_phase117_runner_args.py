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
    assert args[args.index("--hdf5-arm-hold-frame-offset") + 1] == "0"
    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "208"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "245"
    assert args[args.index("--min-object-lift") + 1] == "0.0"
    assert args[args.index("--physics-dt") + 1] == "0.02"
    assert args[args.index("--hdf5-replay-substep-mode") + 1] == "zero_order_hold"
    assert args[args.index("--hdf5-replay-target-hold-steps") + 1] == "1"
    assert args[args.index("--post-close-hold-steps") + 1] == "0"
    assert args[args.index("--post-close-lift-source") + 1] == "none"
    assert args[args.index("--post-close-lift-gripper-mode") + 1] == "hold_final_close"
    assert args[args.index("--diagnostic-loaded-clamp-squeeze-depth") + 1] == "0.0"
    assert args[args.index("--finger-kp") + 1] == "200.0"
    assert args[args.index("--finger-kd") + 1] == "50.0"
    assert "--post-close-lift-hdf5-start-frame" not in args
    assert "--post-close-lift-hdf5-end-frame" not in args
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


def test_phase132_can_select_diagnostic_loaded_clamp_squeeze() -> None:
    args = _phase132_args(Path("out"), diagnostic_loaded_clamp_squeeze_depth=0.001)

    assert args[args.index("--diagnostic-loaded-clamp-squeeze-depth") + 1] == "0.001"


def test_phase132_can_select_finger_drive_for_authority_diagnostic() -> None:
    args = _phase132_args(Path("out"), finger_kp=400.0, finger_kd=75.0)

    assert args[args.index("--finger-kp") + 1] == "400.0"
    assert args[args.index("--finger-kd") + 1] == "75.0"


def test_phase132_can_select_grasp_band_proxy_without_changing_default() -> None:
    default_args = _phase132_args(Path("out"))
    grasp_band_args = _phase132_args(Path("out"), object_shape="bottle_usd_grasp_band_proxy")
    grasp_box_args = _phase132_args(Path("out"), object_shape="bottle_usd_grasp_box_proxy")

    assert default_args[default_args.index("--object-shape") + 1] == "bottle_usd_cylinder_proxy"
    assert grasp_band_args[grasp_band_args.index("--object-shape") + 1] == "bottle_usd_grasp_band_proxy"
    assert grasp_box_args[grasp_box_args.index("--object-shape") + 1] == "bottle_usd_grasp_box_proxy"


def test_phase132_can_select_object_clearance_without_changing_default() -> None:
    default_args = _phase132_args(Path("out"))
    clearance_args = _phase132_args(Path("out"), object_clearance=0.003)

    assert default_args[default_args.index("--object-clearance") + 1] == "0.001"
    assert clearance_args[clearance_args.index("--object-clearance") + 1] == "0.003"


def test_phase132_can_select_closing_axis_gap_solver_basis_without_changing_default() -> None:
    default_args = _phase132_args(Path("out"))
    placement_args = _phase132_args(Path("out"), closing_axis_gap_solver_basis="placement")

    assert default_args[default_args.index("--closing-axis-gap-solver-basis") + 1] == "open"
    assert placement_args[placement_args.index("--closing-axis-gap-solver-basis") + 1] == "placement"


def test_phase132_can_select_object_width_stop_predictive_margin_without_changing_default() -> None:
    default_args = _phase132_args(Path("out"))
    margin_args = _phase132_args(Path("out"), object_width_stop_predictive_margin=0.003)

    assert default_args[default_args.index("--object-width-stop-predictive-margin") + 1] == "0.0"
    assert margin_args[margin_args.index("--object-width-stop-predictive-margin") + 1] == "0.003"


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


def test_phase132_can_add_post_close_hold_without_extending_raw_hdf5_window() -> None:
    args = _phase132_args(Path("out"), post_close_hold_steps=12)

    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "208"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "245"
    assert args[args.index("--post-close-hold-steps") + 1] == "12"


def test_phase132_can_add_post_close_hdf5_lift_without_extending_raw_close_window() -> None:
    args = _phase132_args(
        Path("out"),
        post_close_hold_steps=20,
        post_close_lift_source="hdf5_continuation",
        post_close_lift_hdf5_start_frame=236,
        post_close_lift_hdf5_end_frame=280,
    )

    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "208"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "245"
    assert args[args.index("--post-close-hold-steps") + 1] == "20"
    assert args[args.index("--post-close-lift-source") + 1] == "hdf5_continuation"
    assert args[args.index("--post-close-lift-hdf5-start-frame") + 1] == "236"
    assert args[args.index("--post-close-lift-hdf5-end-frame") + 1] == "280"
    assert args[args.index("--post-close-lift-gripper-mode") + 1] == "hold_final_close"


def test_phase132_can_opt_in_to_nonformal_substep_interpolation_diagnostic() -> None:
    args = _phase132_args(Path("out"), hdf5_replay_substep_mode="linear_interpolation_diagnostic")

    assert args[args.index("--hdf5-replay-substep-mode") + 1] == "linear_interpolation_diagnostic"


def test_phase132_can_add_jacobian_vertical_lift_without_hdf5_continuation_range() -> None:
    args = _phase132_args(
        Path("out"),
        post_close_lift_source="jacobian_vertical",
        post_close_lift_height=0.012,
        post_close_lift_steps=30,
        post_close_lift_jacobian_eps=0.003,
        post_close_lift_jacobian_damping=2e-6,
        post_close_lift_max_joint_delta=0.05,
    )

    assert args[args.index("--post-close-lift-source") + 1] == "jacobian_vertical"
    assert "--post-close-lift-hdf5-end-frame" not in args
    assert args[args.index("--post-close-lift-height") + 1] == "0.012"
    assert args[args.index("--post-close-lift-steps") + 1] == "30"
    assert args[args.index("--post-close-lift-jacobian-eps") + 1] == "0.003"
    assert args[args.index("--post-close-lift-jacobian-damping") + 1] == "2e-06"
    assert args[args.index("--post-close-lift-max-joint-delta") + 1] == "0.05"


def test_phase132_can_replay_full_left_arm_and_gripper_hdf5_window() -> None:
    default_args = _phase132_args(Path("out"))
    assert default_args[default_args.index("--hdf5-replay-mode") + 1] == "hdf5_arm_start_then_gripper_only"

    full_args = _phase132_args(Path("out"), hdf5_replay_mode="left_arm_and_gripper")
    assert full_args[full_args.index("--hdf5-replay-mode") + 1] == "left_arm_and_gripper"


def test_phase132_can_hold_loaded_hdf5_arm_frame_for_static_gripper_diagnostic() -> None:
    args = _phase132_args(Path("out"), hdf5_arm_hold_frame_offset=28)

    assert args[args.index("--hdf5-replay-mode") + 1] == "hdf5_arm_start_then_gripper_only"
    assert args[args.index("--hdf5-arm-hold-frame-offset") + 1] == "28"


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


def test_phase132_can_select_episode18_close_frame_placement_basis() -> None:
    default_args = _phase132_args(Path("out"))
    assert default_args[default_args.index("--object-placement") + 1] == (
        "hdf5_open_finger_rear_quarter_tabletop"
    )

    close_args = _phase132_args(Path("out"), placement_basis="close")
    assert close_args[close_args.index("--object-placement") + 1] == (
        "hdf5_close_finger_rear_quarter_tabletop"
    )


def test_phase132_can_keep_tabletop_fixed_for_formal_replay_semantics() -> None:
    default_args = _phase132_args(Path("out"))
    assert "--derive-tabletop-top-z-from-open-finger-height" in default_args

    fixed_args = _phase132_args(Path("out"), tabletop_mode="fixed_reference")
    assert fixed_args[fixed_args.index("--object-tabletop-reference-path") + 1] == "/World/Table"
    assert "--derive-tabletop-top-z-from-open-finger-height" not in fixed_args


def test_phase132_can_pass_diagnostic_object_center_offset() -> None:
    default_args = _phase132_args(Path("out"))
    assert "--object-center-offset" not in default_args

    offset_args = _phase132_args(Path("out"), object_center_offset=(0.0, -0.02, 0.0))
    offset_index = offset_args.index("--object-center-offset")
    assert offset_args[offset_index + 1 : offset_index + 4] == ["0.0", "-0.02", "0.0"]


def test_phase132_loaded_qpos_calibration_is_explicit_opt_in() -> None:
    default_args = _phase132_args(Path("out"))
    assert "--loaded-gripper-qpos-calibration-mode" not in default_args

    calibrated_args = _phase132_args(Path("out"), enable_episode18_loaded_qpos_calibration=True)

    assert calibrated_args[calibrated_args.index("--loaded-gripper-qpos-calibration-mode") + 1] == (
        "affine_open_to_contact"
    )
    assert calibrated_args[calibrated_args.index("--loaded-gripper-open-qpos") + 1] == "0.9473305344581604"
    assert calibrated_args[calibrated_args.index("--loaded-gripper-contact-qpos") + 1] == "0.5712134838104248"
    assert calibrated_args[calibrated_args.index("--loaded-gripper-effective-contact-width") + 1] == "0.052"
    assert calibrated_args[calibrated_args.index("--loaded-gripper-qpos-calibration-source") + 1] == (
        "episode18_frames_208_220_open_max_and_236_244_loaded_plateau_20260722"
    )


def test_phase132_can_opt_in_to_forced_overlap_diagnostic() -> None:
    default_args = _phase132_args(Path("out"))
    assert "--diagnostic-force-target-overlap" not in default_args

    diagnostic_args = _phase132_args(Path("out"), diagnostic_force_target_overlap="nearest")
    assert diagnostic_args[diagnostic_args.index("--diagnostic-force-target-overlap") + 1] == "nearest"
    assert diagnostic_args[diagnostic_args.index("--diagnostic-force-target-overlap-m") + 1] == "0.001"

    right_finger_args = _phase132_args(
        Path("out"),
        diagnostic_force_target_overlap="right_finger",
        diagnostic_force_target_overlap_m=0.02,
    )
    assert right_finger_args[right_finger_args.index("--diagnostic-force-target-overlap") + 1] == "right_finger"
    assert right_finger_args[right_finger_args.index("--diagnostic-force-target-overlap-m") + 1] == "0.02"


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


def test_episode19_dynamic_runner_can_select_grasp_box_bottle_usd_proxy() -> None:
    args = _episode19_dynamic_args(Path("out"), object_shape="bottle_usd_grasp_box_proxy")

    assert args[args.index("--object-shape") + 1] == "bottle_usd_grasp_box_proxy"

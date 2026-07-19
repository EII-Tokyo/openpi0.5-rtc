from pathlib import Path

from aloha_isaac_replay.scripts.run_phase117_diagnostic_held_bottle_replay import _phase117_args
from aloha_isaac_replay.scripts.run_phase132_active_tabletop_grasp_gate import _phase132_args


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
    assert args[args.index("--object-fill-fraction") + 1] == "0.55"
    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "326"
    assert args[args.index("--hdf5-gripper-end-frame") + 1] == "360"
    assert args[args.index("--max-closing-long-axis-dot") + 1] == "0.25"
    assert args[args.index("--object-tabletop-top-z") + 1] == "0.004086510930165169"
    assert args[args.index("--workcell-contact-policy") + 1] == (
        "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml"
    )
    assert "--require-active-target-contact" in args
    assert "--already-in-contact-setup" not in args
    assert "--save-debug-stage" not in args


def test_phase132_can_opt_in_to_debug_stage_export() -> None:
    args = _phase132_args(Path("out"), save_debug_stage=True)

    assert "--save-debug-stage" in args

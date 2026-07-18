from pathlib import Path

from aloha_isaac_replay.scripts.run_phase117_diagnostic_held_bottle_replay import _phase117_args


def test_phase117_diagnostic_runner_uses_kinematic_held_object_boundary() -> None:
    args = _phase117_args(Path("out"), Path("policy.yaml"), start_frame=80)

    assert args[args.index("--hdf5-gripper-start-frame") + 1] == "80"
    assert args[args.index("--object-placement") + 1] == "grasp_yaml"
    assert args[args.index("--diagnostic-held-object-mode") + 1] == "follow_gripper"
    assert args[args.index("--support-plane-mode") + 1] == "none"
    assert "--disable-object-rigid-body" in args

    assert "--trace-contact-pairs" not in args
    assert "--fail-on-non-target-object-contact" not in args
    assert "--workcell-contact-policy" not in args
    assert "--already-in-contact-setup" not in args

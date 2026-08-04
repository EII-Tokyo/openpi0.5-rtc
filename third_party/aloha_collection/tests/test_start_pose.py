import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "scripts" / "record_episodes_copy.py"
LEFT_ACQUISITION_HOME = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]
RIGHT_ACQUISITION_HOME = [0.0, -0.96, 1.16, 0.0, 0.0, 0.0]


def load_recorder():
    spec = importlib.util.spec_from_file_location(
        "record_episodes_copy",
        RECORDER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_start_pose_uses_wenshun_side_specific_acquisition_home():
    recorder = load_recorder()

    assert recorder.DEFAULT_START_ARM_POSE == {
        "left_arm": LEFT_ACQUISITION_HOME,
        "right_arm": RIGHT_ACQUISITION_HOME,
    }

    selected = recorder._choose_episode_start_arm_pose()
    assert selected["left_arm"] == LEFT_ACQUISITION_HOME
    assert selected["right_arm"] == RIGHT_ACQUISITION_HOME

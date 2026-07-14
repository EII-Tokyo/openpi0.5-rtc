import math

import pytest

from examples.aloha_isaac.scripts.apply_aloha_initial_pose import (
    ALOHA_USD_JOINTS,
    FINGER_PRISMATIC_LOWER_LIMIT,
    FINGER_PRISMATIC_UPPER_LIMIT,
    REAL_START_ARM_POSE,
    REAL_RUNTIME_RESET_QPOS14,
    REAL_RUNTIME_RESET_POSE,
    REAL_RUNTIME_SLEEP_POSE,
    REAL_RUNTIME_SLEEP_QPOS14,
    build_pose_records,
    pose_to_usd_joint_positions,
    puppet_gripper_joint_to_isaac_finger_position,
    qpos14_to_isaac_joint_pose,
    root_joint_world_anchor_from_body_translation,
    split_real_start_pose_for_isaac_articulations,
)
from examples.aloha_real.constants import (
    PUPPET_GRIPPER_JOINT_CLOSE,
    PUPPET_GRIPPER_JOINT_OPEN,
    PUPPET_GRIPPER_POSITION_CLOSE,
    PUPPET_GRIPPER_POSITION_OPEN,
)


def test_real_start_pose_has_one_value_per_usd_joint() -> None:
    assert len(REAL_START_ARM_POSE) == len(ALOHA_USD_JOINTS)
    assert len(REAL_RUNTIME_RESET_QPOS14) == 14
    assert len(REAL_RUNTIME_SLEEP_QPOS14) == 14
    assert len(REAL_RUNTIME_RESET_POSE) == len(ALOHA_USD_JOINTS)
    assert len(REAL_RUNTIME_SLEEP_POSE) == len(ALOHA_USD_JOINTS)
    assert FINGER_PRISMATIC_LOWER_LIMIT == pytest.approx(0.01844)
    assert FINGER_PRISMATIC_UPPER_LIMIT == pytest.approx(0.058)


def test_pose_to_usd_joint_positions_converts_only_revolute_joints_to_degrees() -> None:
    converted = pose_to_usd_joint_positions(REAL_RUNTIME_RESET_POSE)

    assert converted[0] == pytest.approx(0.0)
    assert converted[1] == pytest.approx(math.degrees(-0.96))
    assert converted[2] == pytest.approx(math.degrees(1.16))
    assert converted[3] == pytest.approx(math.degrees(1.57))
    assert converted[4] == pytest.approx(0.0)
    assert converted[5] == pytest.approx(math.degrees(-1.57))
    assert converted[6] == pytest.approx(0.058)
    assert converted[7] == pytest.approx(0.058)


def test_puppet_gripper_joint_angle_maps_to_isaac_finger_position() -> None:
    assert puppet_gripper_joint_to_isaac_finger_position(PUPPET_GRIPPER_JOINT_CLOSE) == pytest.approx(
        PUPPET_GRIPPER_POSITION_CLOSE
    )
    assert puppet_gripper_joint_to_isaac_finger_position(PUPPET_GRIPPER_JOINT_OPEN) == pytest.approx(
        PUPPET_GRIPPER_POSITION_OPEN
    )
    assert puppet_gripper_joint_to_isaac_finger_position(
        (PUPPET_GRIPPER_JOINT_OPEN + PUPPET_GRIPPER_JOINT_CLOSE) / 2.0
    ) == pytest.approx((PUPPET_GRIPPER_POSITION_OPEN + PUPPET_GRIPPER_POSITION_CLOSE) / 2.0)


def test_runtime_qpos14_converts_to_imported_isaac_gripper_layout() -> None:
    qpos = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        PUPPET_GRIPPER_JOINT_OPEN,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        PUPPET_GRIPPER_JOINT_CLOSE,
    )

    assert qpos14_to_isaac_joint_pose(qpos) == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            PUPPET_GRIPPER_POSITION_OPEN,
            PUPPET_GRIPPER_POSITION_OPEN,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
        )
    )


def test_build_pose_records_preserves_joint_paths_and_drive_types() -> None:
    records = build_pose_records(REAL_RUNTIME_RESET_POSE)

    assert records[0].joint_path == "/scene/joints/left_waist"
    assert records[0].drive_type == "angular"
    assert records[0].position == pytest.approx(0.0)
    assert records[6].joint_path == "/scene/joints/left_left_finger"
    assert records[6].drive_type == "linear"
    assert records[6].position == pytest.approx(0.058)
    assert records[-1].joint_path == "/scene/joints/right_right_finger"
    assert records[-1].drive_type == "linear"
    assert records[-1].position == pytest.approx(0.058)


def test_runtime_reset_pose_for_isaac_articulations_matches_real_robot_reset() -> None:
    left, right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_RESET_POSE)

    assert left == pytest.approx((0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.058, 0.058))
    assert right == pytest.approx((0.0, -0.96, 1.16, 0.0, 0.0, 0.0, 0.058, 0.058))


def test_runtime_sleep_pose_for_isaac_articulations_matches_real_sleep_arms() -> None:
    left, right = split_real_start_pose_for_isaac_articulations(REAL_RUNTIME_SLEEP_POSE)

    assert left == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))
    assert right == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))


def test_root_joint_world_anchor_matches_body_translation_for_world_root_joint() -> None:
    local_pos0, local_pos1 = root_joint_world_anchor_from_body_translation((-0.469, -0.019, 0.02))

    assert local_pos0 == pytest.approx((-0.469, -0.019, 0.02))
    assert local_pos1 == pytest.approx((0.0, 0.0, 0.0))

class _FakeArticulation:
    def __init__(self, initialized: bool) -> None:
        self.handles_initialized = initialized
        self.positions = []
        self.velocities = []
        self.default_states = []

    def set_joint_positions(self, positions):
        self.positions.append(tuple(positions))

    def set_joint_velocities(self, velocities):
        self.velocities.append(tuple(velocities))

    def set_joints_default_state(self, positions, velocities):
        self.default_states.append((tuple(positions), tuple(velocities)))

    def get_joint_positions(self):
        return self.positions[-1]


def test_set_real_start_pose_requires_initialized_handles() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import _set_real_start_pose_on_initialized_articulations

    left = _FakeArticulation(initialized=False)
    right = _FakeArticulation(initialized=True)

    assert _set_real_start_pose_on_initialized_articulations(left, right) is False
    assert left.positions == []
    assert right.positions == []


def test_set_real_start_pose_applies_when_both_handles_initialized() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import _set_real_start_pose_on_initialized_articulations

    left = _FakeArticulation(initialized=True)
    right = _FakeArticulation(initialized=True)

    assert _set_real_start_pose_on_initialized_articulations(left, right) is True
    assert left.positions[0] == pytest.approx((0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.058, 0.058))
    assert right.positions[0] == pytest.approx((0.0, -0.96, 1.16, 0.0, 0.0, 0.0, 0.058, 0.058))
    assert left.velocities == [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    assert right.velocities == [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    assert left.default_states[0][0] == pytest.approx(left.positions[0])
    assert right.default_states[0][0] == pytest.approx(right.positions[0])


def test_pose_control_window_uses_pinned_startup_position() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import _pose_control_window_kwargs

    kwargs = _pose_control_window_kwargs()

    assert kwargs["position_x"] == pytest.approx(1135)
    assert kwargs["position_y"] == pytest.approx(760)
    assert kwargs["width"] == pytest.approx(154)
    assert kwargs["height"] == pytest.approx(190)


def test_pose_controller_applies_home_sleep_and_toggle() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import AlohaPoseController

    left = _FakeArticulation(initialized=True)
    right = _FakeArticulation(initialized=True)
    controller = AlohaPoseController(left, right)

    assert controller.apply_home(animate=False) is True
    assert controller.current_pose_name == "home"
    assert left.positions[-1] == pytest.approx((0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.058, 0.058))

    assert controller.apply_sleep(animate=False) is True
    assert controller.current_pose_name == "sleep"
    assert left.positions[-1] == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))
    assert right.positions[-1] == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))

    assert controller.apply_home(animate=False) is True
    assert controller.current_pose_name == "home"
    assert left.positions[-1] == pytest.approx((0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.058, 0.058))


def test_pose_controller_animates_home_to_sleep_before_finishing() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import AlohaPoseController

    left = _FakeArticulation(initialized=True)
    right = _FakeArticulation(initialized=True)
    controller = AlohaPoseController(left, right, transition_duration_s=2.0)

    assert controller.apply_home(animate=False) is True
    home_left = left.positions[-1]

    assert controller.apply_sleep() is True
    assert controller.is_transitioning is True
    assert left.positions[-1] == pytest.approx(home_left)

    assert controller.update_transition(1.0) is True
    assert controller.is_transitioning is True
    assert left.positions[-1][1] == pytest.approx((-0.96 + -1.84) / 2.0)
    assert left.positions[-1][2] == pytest.approx((1.16 + 1.60) / 2.0)
    assert left.positions[-1] != pytest.approx(home_left)

    assert controller.update_transition(1.0) is True
    assert controller.is_transitioning is False
    assert controller.current_pose_name == "sleep"
    assert left.positions[-1] == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))
    assert right.positions[-1] == pytest.approx((0.0, -1.84, 1.60, 0.0, -1.60, 0.0, 0.058, 0.058))


def test_pose_controller_restart_transition_from_current_intermediate_pose() -> None:
    from examples.aloha_isaac.scripts.open_workcell_gui import AlohaPoseController

    left = _FakeArticulation(initialized=True)
    right = _FakeArticulation(initialized=True)
    controller = AlohaPoseController(left, right, transition_duration_s=2.0)

    assert controller.apply_home(animate=False) is True
    assert controller.apply_sleep() is True
    assert controller.update_transition(1.0) is True
    midway_left = left.positions[-1]

    assert controller.apply_home() is True
    assert controller.is_transitioning is True
    assert left.positions[-1] == pytest.approx(midway_left)

    assert controller.update_transition(1.0) is True
    assert left.positions[-1][1] == pytest.approx((midway_left[1] + -0.96) / 2.0)
    assert controller.update_transition(1.0) is True
    assert controller.current_pose_name == "home"
    assert left.positions[-1] == pytest.approx((0.0, -0.96, 1.16, 1.57, 0.0, -1.57, 0.058, 0.058))

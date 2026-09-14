import math
from types import SimpleNamespace
import unittest

import numpy as np

from openpi.robot.aloha_real import robot_utils
from openpi.robot.aloha_real.real_env import get_action


class BimanualJointPositionUnwrapperTest(unittest.TestCase):
    def test_opposite_sides_of_pi_do_not_add_full_turn(self):
        unwrapper = robot_utils.BimanualJointPositionUnwrapper()
        left = np.array([0.0, 0.0, 0.0, 2.48, 0.0, -3.203])
        right = np.array([0.0, 0.0, 0.0, 0.038, 0.0, 0.035])

        left_out, right_out = unwrapper.unwrap(left, right)

        np.testing.assert_allclose(left_out, left)
        np.testing.assert_allclose(right_out, right)

    def test_get_action_keeps_each_leader_on_its_own_revolution(self):
        def make_bot(positions):
            joint_states = SimpleNamespace(position=[*positions, 0.0])
            return SimpleNamespace(dxl=SimpleNamespace(joint_states=joint_states))

        left = np.array([0.0, 0.0, 0.0, 2.48, 0.0, -3.203])
        right = np.array([0.0, 0.0, 0.0, 0.038, 0.0, 0.035])
        action = get_action(
            make_bot(left),
            make_bot(right),
            joint_unwrapper=robot_utils.BimanualJointPositionUnwrapper(),
        )

        np.testing.assert_allclose(action[:6], left)
        np.testing.assert_allclose(action[7:13], right)
        self.assertNotAlmostEqual(action[12], right[5] - 2 * math.pi)

    def test_each_arm_is_unwrapped_against_its_own_history(self):
        unwrapper = robot_utils.BimanualJointPositionUnwrapper()
        left = np.zeros(6)
        right = np.zeros(6)
        left[5] = -math.pi + 0.01
        right[5] = math.pi - 0.02
        unwrapper.unwrap(left, right)

        left[5] = math.pi - 0.01
        right[5] = -math.pi + 0.02
        left_out, right_out = unwrapper.unwrap(left, right)

        self.assertAlmostEqual(left_out[5], -math.pi - 0.01)
        self.assertAlmostEqual(right_out[5], math.pi + 0.02)


if __name__ == "__main__":
    unittest.main()

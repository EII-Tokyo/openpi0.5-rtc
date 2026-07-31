from __future__ import annotations

import inspect

from tools.probe_aloha1_task7b2_horizontal_kinematics import solve_adaptive_linear_ik
from tools.validate_aloha1_gripper_coupling_ab import author_coupling_variant
from tools.validate_aloha1_task7b2_horizontal_grasp import _author_session_finger_drive_type
from tools.validate_aloha1_task7b2_horizontal_grasp import _collect_rigid_local_collision_points
from tools.validate_aloha1_task7b2_horizontal_grasp import _solve_settled_bottle_runtime_ik
from tools.validate_aloha1_task7b2_horizontal_grasp import read_physx_bottle_state
from tools.validate_aloha1_task7b2_horizontal_grasp import transform_local_points_to_world_bounds


def test_grasp_runtime_helper_import_closure() -> None:
    helpers = (
        author_coupling_variant,
        _author_session_finger_drive_type,
        _collect_rigid_local_collision_points,
        _solve_settled_bottle_runtime_ik,
        read_physx_bottle_state,
        transform_local_points_to_world_bounds,
    )

    assert all(callable(helper) for helper in helpers)
    assert "start_orientation_wxyz" in inspect.signature(
        solve_adaptive_linear_ik
    ).parameters

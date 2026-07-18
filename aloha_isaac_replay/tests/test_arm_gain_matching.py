from __future__ import annotations

import numpy as np

from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_arm_gains
from aloha_isaac_replay.scripts.right_shoulder_runtime_audit import _apply_named_dof_gains


class _FakeArticulationView:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def set_gains(self, *, kps, kds, joint_indices, save_to_usd: bool) -> None:
        self.calls.append(
            {
                "kps": None if kps is None else np.asarray(kps, dtype=np.float64).tolist(),
                "kds": None if kds is None else np.asarray(kds, dtype=np.float64).tolist(),
                "joint_indices": np.asarray(joint_indices, dtype=np.int64).tolist(),
                "save_to_usd": save_to_usd,
            }
        )


class _FakeArticulation:
    def __init__(self, dof_names: list[str]) -> None:
        self.dof_names = dof_names
        self._articulation_view = _FakeArticulationView()


def test_apply_arm_gains_matches_scene_base_link_prefixed_arm_dofs() -> None:
    art = _FakeArticulation(
        [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_left_finger",
            "left_right_finger",
        ]
    )

    _apply_arm_gains(art, kp=200.0, kd=200.0)

    assert art._articulation_view.calls == [
        {
            "kps": [200.0] * 6,
            "kds": [200.0] * 6,
            "joint_indices": [0, 1, 2, 3, 4, 5],
            "save_to_usd": False,
        }
    ]


def test_apply_arm_gains_matches_unprefixed_arm_dofs() -> None:
    art = _FakeArticulation(["waist", "shoulder", "elbow", "left_finger"])

    _apply_arm_gains(art, kp=None, kd=100.0)

    assert art._articulation_view.calls == [
        {"kps": None, "kds": [100.0, 100.0, 100.0], "joint_indices": [0, 1, 2], "save_to_usd": False}
    ]


def test_apply_named_dof_gains_only_matches_requested_fingers() -> None:
    art = _FakeArticulation(
        [
            "left_waist",
            "left_shoulder",
            "left_left_finger",
            "left_right_finger",
            "left_gripper",
            "right_left_finger",
        ]
    )

    _apply_named_dof_gains(art, ["left_left_finger", "left_right_finger"], kp=80.0, kd=40.0)

    assert art._articulation_view.calls == [
        {
            "kps": [80.0, 80.0],
            "kds": [40.0, 40.0],
            "joint_indices": [2, 3],
            "save_to_usd": False,
        }
    ]


def test_apply_named_dof_gains_ignores_missing_names() -> None:
    art = _FakeArticulation(["left_left_finger"])

    _apply_named_dof_gains(art, ["missing", "left_left_finger"], kp=10.0, kd=None)

    assert art._articulation_view.calls == [
        {"kps": [10.0], "kds": None, "joint_indices": [0], "save_to_usd": False}
    ]

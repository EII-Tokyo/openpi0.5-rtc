from __future__ import annotations

import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos


def _left_only_mapping() -> dict:
    return {
        "dof_mapping": [
            {
                "canonical_name": name,
                "dataset_index": idx,
                "isaac_dof_name": f"left/{name}",
                "sign": 1.0,
                "offset": 0.0,
                "scale": 1.0,
            }
            for idx, name in enumerate(ARM_ONLY_NAMES[:6])
        ]
    }


def test_arm_only_mapping_uses_only_twelve_arm_joints_and_skips_grippers() -> None:
    mapping = load_mapping("configs/aloha/original_stationary_aloha_mapping.yaml")
    targets = arm_only_targets_from_standard_qpos(np.arange(14, dtype=np.float64), mapping)
    assert [target.canonical_name for target in targets] == list(ARM_ONLY_NAMES)
    assert {target.dataset_index for target in targets} == set(range(6)) | set(range(7, 13))
    assert 6 not in {target.dataset_index for target in targets}
    assert 13 not in {target.dataset_index for target in targets}


def test_arm_only_mapping_can_require_only_one_side_for_left_replay() -> None:
    targets = arm_only_targets_from_standard_qpos(
        np.arange(14, dtype=np.float64),
        _left_only_mapping(),
        side="left",
    )

    assert [target.canonical_name for target in targets] == list(ARM_ONLY_NAMES[:6])
    assert [target.dataset_index for target in targets] == list(range(6))


def test_arm_only_mapping_rejects_unknown_side_name() -> None:
    try:
        arm_only_targets_from_standard_qpos(np.arange(14, dtype=np.float64), _left_only_mapping(), side="center")
    except ValueError as exc:
        assert "side must be one of" in str(exc)
    else:
        raise AssertionError("expected unknown side to raise ValueError")


def test_trossen_scene_mapping_is_left_side_scoped() -> None:
    mapping = load_mapping("configs/aloha/trossen_scene_base_link_aloha1_left_mapping.yaml")
    targets = arm_only_targets_from_standard_qpos(np.arange(14, dtype=np.float64), mapping, side="left")

    assert [target.canonical_name for target in targets] == list(ARM_ONLY_NAMES[:6])
    assert [target.isaac_dof_name for target in targets] == [
        "left/left_waist",
        "left/left_shoulder",
        "left/left_elbow",
        "left/left_forearm_roll",
        "left/left_wrist_angle",
        "left/left_wrist_rotate",
    ]

from __future__ import annotations

import numpy as np

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.replay.arm_only_mapping import ARM_ONLY_NAMES
from aloha_isaac_replay.replay.arm_only_mapping import arm_only_targets_from_standard_qpos


def test_arm_only_mapping_uses_only_twelve_arm_joints_and_skips_grippers() -> None:
    mapping = load_mapping("configs/aloha/original_stationary_aloha_mapping.yaml")
    targets = arm_only_targets_from_standard_qpos(np.arange(14, dtype=np.float64), mapping)
    assert [target.canonical_name for target in targets] == list(ARM_ONLY_NAMES)
    assert {target.dataset_index for target in targets} == set(range(6)) | set(range(7, 13))
    assert 6 not in {target.dataset_index for target in targets}
    assert 13 not in {target.dataset_index for target in targets}


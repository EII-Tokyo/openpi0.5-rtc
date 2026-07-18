from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.adapters.standard_aloha import require_aloha_14d


ARM_ONLY_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
)


@dataclasses.dataclass(frozen=True)
class ArmOnlyTarget:
    canonical_name: str
    dataset_index: int
    isaac_dof_name: str
    value: float
    sign: float
    offset: float
    scale: float


def arm_only_targets_from_standard_qpos(
    qpos_14d: np.ndarray,
    mapping: dict[str, Any],
    *,
    side: str | None = None,
) -> list[ArmOnlyTarget]:
    qpos = require_aloha_14d(qpos_14d, name="qpos_14d")
    if side not in (None, "left", "right"):
        raise ValueError(f"side must be one of None, 'left', or 'right'; got {side!r}")
    names = ARM_ONLY_NAMES if side is None else tuple(name for name in ARM_ONLY_NAMES if name.startswith(f"{side}_"))
    entries = {entry["canonical_name"]: entry for entry in mapping.get("dof_mapping", [])}
    targets: list[ArmOnlyTarget] = []
    for name in names:
        if name not in entries:
            raise ValueError(f"missing mapping entry for {name}")
        entry = entries[name]
        dataset_index = int(entry["dataset_index"])
        if STANDARD_ALOHA_14D_NAMES[dataset_index] != name:
            raise ValueError(f"dataset index {dataset_index} does not match canonical name {name}")
        sign = float(entry["sign"])
        offset = float(entry["offset"])
        scale = float(entry["scale"])
        value = sign * float(qpos[dataset_index]) * scale + offset
        targets.append(
            ArmOnlyTarget(
                canonical_name=name,
                dataset_index=dataset_index,
                isaac_dof_name=str(entry["isaac_dof_name"]),
                value=value,
                sign=sign,
                offset=offset,
                scale=scale,
            )
        )
    dof_names = [target.isaac_dof_name for target in targets]
    if len(dof_names) != len(set(dof_names)):
        raise ValueError(f"duplicate Isaac DOF names in arm-only mapping: {dof_names}")
    return targets

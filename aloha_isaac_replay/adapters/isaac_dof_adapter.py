from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class DofMappingEntry:
    canonical_name: str
    dataset_index: int | None
    openpi_index: int | None
    isaac_dof_name: str
    isaac_dof_index: int | None
    sign: float
    offset: float
    scale: float
    unit: str
    source: str
    confidence: float


def load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def validate_mapping(mapping: dict[str, Any], dof_names: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    entries = mapping.get("dof_mapping", [])
    canonical = [entry["canonical_name"] for entry in entries]
    if len(canonical) != len(set(canonical)):
        errors.append("duplicate canonical_name entries")
    isaac_names = [entry["isaac_dof_name"] for entry in entries if entry.get("isaac_dof_name")]
    if len(isaac_names) != len(set(isaac_names)):
        errors.append("duplicate isaac_dof_name entries")
    arm_entries = [entry for entry in entries if entry.get("dataset_index") is not None and entry["dataset_index"] not in (6, 13)]
    if len(arm_entries) != 12:
        errors.append(f"expected 12 arm joint mappings, got {len(arm_entries)}")
    gripper_entries = [entry for entry in entries if entry.get("canonical_name", "").endswith("finger")]
    if len(gripper_entries) < 4:
        errors.append(f"expected explicit left/right finger gripper mappings for both arms, got {len(gripper_entries)}")
    if dof_names is not None:
        missing = sorted(set(isaac_names) - set(dof_names))
        if missing:
            errors.append(f"mapping references missing Isaac DOFs: {missing}")
    return errors


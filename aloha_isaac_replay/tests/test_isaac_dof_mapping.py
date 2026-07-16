from __future__ import annotations

from aloha_isaac_replay.adapters.isaac_dof_adapter import load_mapping
from aloha_isaac_replay.adapters.isaac_dof_adapter import validate_mapping


def test_mapping_config_has_complete_standard_aloha_coverage() -> None:
    mapping = load_mapping("configs/aloha/original_stationary_aloha_mapping.yaml")
    assert validate_mapping(mapping) == []


def test_mapping_validator_reports_missing_isaac_dofs_with_clear_names() -> None:
    mapping = load_mapping("configs/aloha/original_stationary_aloha_mapping.yaml")
    errors = validate_mapping(mapping, dof_names=["left/waist"])
    assert errors
    assert "right/wrist_rotate" in errors[0]


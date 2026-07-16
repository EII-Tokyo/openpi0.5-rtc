from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import OPENPI_JOINT_FLIP_MASK
from aloha_isaac_replay.adapters.standard_aloha import openpi_internal_to_standard
from aloha_isaac_replay.adapters.standard_aloha import standard_to_openpi_internal


ALOHA_POLICY_PATH = Path("src/openpi/policies/aloha_policy.py")


def _source_text() -> str:
    return ALOHA_POLICY_PATH.read_text()


def _joint_flip_mask_from_source() -> np.ndarray:
    module = ast.parse(_source_text(), filename=str(ALOHA_POLICY_PATH))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_joint_flip_mask":
            for statement in ast.walk(node):
                if isinstance(statement, ast.List):
                    values = [ast.literal_eval(element) for element in statement.elts]
                    return np.asarray(values, dtype=np.float64)
    raise AssertionError("_joint_flip_mask list literal not found in aloha_policy.py")


def test_joint_flip_mask_matches_current_openpi_source_literal() -> None:
    assert np.array_equal(OPENPI_JOINT_FLIP_MASK, _joint_flip_mask_from_source())


def test_gripper_angular_conversion_is_still_disabled_in_current_openpi_source() -> None:
    source = _source_text()
    assert "# state[[6, 13]] = _gripper_to_angular(state[[6, 13]])" in source
    assert "# actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])" in source
    assert "# actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])" in source


def test_standard_to_openpi_internal_matches_source_sign_flip_semantics() -> None:
    standard = np.linspace(-0.7, 0.7, 14)
    expected = _joint_flip_mask_from_source() * standard
    assert np.allclose(standard_to_openpi_internal(standard), expected)


def test_openpi_internal_to_standard_matches_source_sign_flip_semantics() -> None:
    internal_actions = np.linspace(-0.5, 0.5, 28).reshape(2, 14)
    expected = _joint_flip_mask_from_source() * internal_actions
    assert np.allclose(openpi_internal_to_standard(internal_actions), expected)

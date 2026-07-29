from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.isaac_grasp_spec import IsaacGraspFile

VALID_GRASP_YAML = """\
format: isaac_grasp
format_version: 1.0
object_frame: /World/Bottle500/grasp_reference
gripper_frame: /World/follower_left/gripper_link
grasps:
  horizontal_body_grasp:
    confidence: 1.0
    position: [0.0, 0.0, 0.1]
    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}
    cspace_position: {left_finger: 0.021, right_finger: -0.021}
    pregrasp_cspace_position: {left_finger: 0.057, right_finger: -0.057}
"""


def _load_text(tmp_path: Path, text: str) -> IsaacGraspFile:
    path = tmp_path / "grasp.yaml"
    path.write_text(text, encoding="utf-8")
    return IsaacGraspFile.load(path)


def test_loads_exact_isaac_grasp_1_format(tmp_path: Path) -> None:
    spec = _load_text(tmp_path, VALID_GRASP_YAML)
    grasp = spec.grasp("horizontal_body_grasp")

    assert spec.object_frame == "/World/Bottle500/grasp_reference"
    assert spec.gripper_frame == "/World/follower_left/gripper_link"
    assert grasp.confidence == 1.0
    assert grasp.object_from_gripper == pytest.approx(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert grasp.cspace_position == {"left_finger": 0.021, "right_finger": -0.021}
    assert grasp.pregrasp_cspace_position == {"left_finger": 0.057, "right_finger": -0.057}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("format: isaac_grasp", "format: another_format", "isaac_grasp"),
        ("format_version: 1.0", "format_version: 2.0", "format_version 1.0"),
        (
            "object_frame: /World/Bottle500/grasp_reference",
            "object_frame: ''",
            "object_frame",
        ),
        (
            "    confidence: 1.0",
            "    confidence: .nan",
            "confidence",
        ),
        (
            "    position: [0.0, 0.0, 0.1]",
            "    position: [0.0, 0.1]",
            "position",
        ),
        (
            "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}",
            "    orientation: {w: 2.0, xyz: [0.0, 0.0, 0.0]}",
            "unit quaternion",
        ),
        (
            "    cspace_position: {left_finger: 0.021, right_finger: -0.021}",
            "    cspace_position: {left_finger: 0.021}",
            "exact left_finger/right_finger",
        ),
        (
            "    pregrasp_cspace_position: {left_finger: 0.057, right_finger: -0.057}",
            "    pregrasp_cspace_position: {left_finger: true, right_finger: -0.057}",
            "left_finger",
        ),
    ],
)
def test_rejects_invalid_isaac_grasp_records(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _load_text(tmp_path, VALID_GRASP_YAML.replace(old, new))


def test_rejects_missing_horizontal_body_grasp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="horizontal_body_grasp"):
        _load_text(
            tmp_path,
            VALID_GRASP_YAML.replace("horizontal_body_grasp:", "another_grasp:"),
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("grasps:", "unexpected: true\ngrasps:", "top-level"),
        (
            "    confidence: 1.0",
            "    unexpected: true\n    confidence: 1.0",
            "horizontal_body_grasp fields",
        ),
        (
            "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}",
            "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0], xyzw: []}",
            "orientation fields",
        ),
    ],
)
def test_rejects_unknown_fields(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _load_text(tmp_path, VALID_GRASP_YAML.replace(old, new))


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    duplicate = VALID_GRASP_YAML.replace(
        "format_version: 1.0",
        "format_version: 1.0\nformat_version: 1.0",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        _load_text(tmp_path, duplicate)


def test_round_trip_writes_byte_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(VALID_GRASP_YAML, encoding="utf-8")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    IsaacGraspFile.load(source).write(first)
    IsaacGraspFile.load(first).write(second)

    assert first.read_bytes() == second.read_bytes()
    assert IsaacGraspFile.load(first).grasp("horizontal_body_grasp").object_from_gripper == pytest.approx(
        IsaacGraspFile.load(second).grasp("horizontal_body_grasp").object_from_gripper
    )

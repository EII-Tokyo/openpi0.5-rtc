from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"
TEMPLATE = REPO_ROOT / "examples/aloha_isaac/config/phase65_table_to_base_calibration_template.yaml"


def test_phase63_table_candidate_blocks_when_base_transforms_are_not_calibrated() -> None:
    payload = audit_table_frame(CONFIG)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert payload["frame_status"]["T_world_table"] == "diagnostic_candidate"
    assert payload["frame_status"]["T_table_left_base"] == "not_calibrated"
    assert payload["frame_status"]["T_table_right_base"] == "not_calibrated"


def test_phase63_table_candidate_reports_table_top_geometry() -> None:
    payload = audit_table_frame(CONFIG)
    geometry = payload["table_geometry"]

    assert geometry["top_center_world"] == pytest.approx([0.593227851197621, 0.7853100288947757, -0.2971450733686908])
    assert geometry["top_corners_world"]["xmin_ymin"] == pytest.approx(
        [
            -0.01677214880237899,
            0.4728100288947757,
            -0.2971450733686908,
        ]
    )
    assert geometry["top_corners_world"]["xmax_ymax"] == pytest.approx(
        [
            1.203227851197621,
            1.0978100288947758,
            -0.2971450733686908,
        ]
    )


def test_phase65_template_blocks_until_measured_transforms_exist() -> None:
    payload = audit_table_frame(TEMPLATE)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert payload["calibration_ready"] is False
    assert "T_table_left_base or T_table_right_base is missing/not calibrated" in payload["blocking_reasons"]


def test_measured_table_to_base_config_passes_and_composes_world_transforms(tmp_path: Path) -> None:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg["calibration_evidence"] = build_evidence_record(
        evidence_path,
        evidence_type="unit_test",
        real_robot_touched=False,
        remote_103_touched=False,
    )
    cfg["support_plane"]["center"] = [1.0, 2.0, 0.48]
    cfg["support_plane"]["size"] = [1.22, 0.625, 0.04]
    cfg["table_frame"]["T_world_table"].update(
        {
            "source": "user_measured",
            "translation": [1.0, 2.0, 0.5],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    cfg["table_frame"]["T_table_left_base"].update(
        {
            "source": "user_measured",
            "translation": [-0.25, 0.1, 0.0],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    cfg["table_frame"]["T_table_right_base"].update(
        {
            "source": "user_measured",
            "translation": [0.25, 0.1, 0.0],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    config_path = tmp_path / "measured_table_to_base.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    payload = audit_table_frame(config_path)

    assert payload["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
    assert payload["calibration_ready"] is True
    assert payload["blocking_reasons"] == []
    assert payload["world_base_transforms"]["T_world_left_base"]["translation"] == pytest.approx([0.75, 2.1, 0.5])
    assert payload["world_base_transforms"]["T_world_right_base"]["translation"] == pytest.approx([1.25, 2.1, 0.5])


def test_measured_config_blocks_robot_state_source_even_with_valid_evidence(tmp_path: Path) -> None:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg["calibration_evidence"] = build_evidence_record(
        evidence_path,
        evidence_type="unit_test",
        real_robot_touched=False,
        remote_103_touched=False,
    )
    cfg["support_plane"]["center"] = [1.0, 2.0, 0.48]
    cfg["support_plane"]["size"] = [1.22, 0.625, 0.04]
    cfg["table_frame"]["T_world_table"].update(
        {
            "source": "hdf5_qpos",
            "translation": [1.0, 2.0, 0.5],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    cfg["table_frame"]["T_table_left_base"].update(
        {
            "source": "joint_states",
            "translation": [-0.25, 0.1, 0.0],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    cfg["table_frame"]["T_table_right_base"].update(
        {
            "source": "dynamixel_registers",
            "translation": [0.25, 0.1, 0.0],
            "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "status": "measured",
        }
    )
    config_path = tmp_path / "robot_state_source_table_to_base.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    payload = audit_table_frame(config_path)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert any("hdf5_qpos cannot provide table-to-base geometry" in reason for reason in payload["blocking_reasons"])
    assert any("joint_states cannot provide table-to-base geometry" in reason for reason in payload["blocking_reasons"])
    assert any(
        "dynamixel_registers cannot provide table-to-base geometry" in reason for reason in payload["blocking_reasons"]
    )


def test_measured_config_blocks_when_table_top_center_mismatch(tmp_path: Path) -> None:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["support_plane"]["center"] = [0.0, 0.0, 0.0]
    cfg["support_plane"]["size"] = [1.22, 0.625, 0.04]
    for name in ("T_world_table", "T_table_left_base", "T_table_right_base"):
        cfg["table_frame"][name].update(
            {
                "source": "user_measured",
                "translation": [0.0, 0.0, 0.0],
                "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "status": "measured",
            }
        )
    config_path = tmp_path / "bad_table_center.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    payload = audit_table_frame(config_path)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert any("does not match support_plane top center" in reason for reason in payload["blocking_reasons"])


def test_measured_config_blocks_when_quaternion_is_not_normalized(tmp_path: Path) -> None:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["support_plane"]["center"] = [0.0, 0.0, -0.02]
    for name in ("T_world_table", "T_table_left_base", "T_table_right_base"):
        cfg["table_frame"][name].update(
            {
                "source": "user_measured",
                "translation": [0.0, 0.0, 0.0],
                "rotation_quat_wxyz": [2.0, 0.0, 0.0, 0.0],
                "status": "measured",
            }
        )
    config_path = tmp_path / "bad_quat.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    payload = audit_table_frame(config_path)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert any("rotation_quat_wxyz norm" in reason for reason in payload["blocking_reasons"])

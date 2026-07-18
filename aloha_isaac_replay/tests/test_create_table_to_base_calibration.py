from __future__ import annotations

import pytest
import yaml

from aloha_isaac_replay.scripts.audit_table_frame_candidate import audit_table_frame
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_calibration_config


def test_build_calibration_config_requires_evidence_when_all_transforms_are_measured(tmp_path) -> None:
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
    )
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    payload = audit_table_frame(path)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"
    assert "calibration_evidence missing" in payload["blocking_reasons"]


def test_build_calibration_config_passes_audit_when_measured_and_evidence_is_present(tmp_path) -> None:
    evidence_path = tmp_path / "worksheet.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    payload = audit_table_frame(path)

    assert payload["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
    assert payload["support_plane"]["center"] == pytest.approx([1.0, 2.0, 0.48])
    assert payload["table_geometry"]["top_center_world"] == pytest.approx([1.0, 2.0, 0.5])
    assert payload["world_base_transforms"]["T_world_left_base"]["translation"] == pytest.approx([0.7, 2.1, 0.5])
    assert payload["world_base_transforms"]["T_world_right_base"]["translation"] == pytest.approx([1.3, 2.1, 0.5])


def test_build_calibration_config_rotates_support_center_with_table_yaw(tmp_path) -> None:
    evidence_path = tmp_path / "worksheet_yaw.yaml"
    evidence_path.write_text("measurement: synthetic yaw\n")
    cfg = build_calibration_config(
        table_top_center=[0.0, 0.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=90.0,
        left_base_in_table=[0.1, 0.0, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.0, 0.1, 0.0],
        right_yaw_deg=0.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "calibration_yaw.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    payload = audit_table_frame(path)

    assert payload["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
    assert payload["support_plane"]["center"] == pytest.approx([0.0, 0.0, 0.48])
    assert payload["world_base_transforms"]["T_world_left_base"]["translation"] == pytest.approx([0.0, 0.1, 0.5])
    assert payload["world_base_transforms"]["T_world_right_base"]["translation"] == pytest.approx([-0.1, 0.0, 0.5])

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import build_from_worksheet


def test_incomplete_measurement_worksheet_reports_missing_fields(tmp_path: Path) -> None:
    worksheet = tmp_path / "worksheet.yaml"
    worksheet.write_text(
        yaml.safe_dump(
            {
                "measurement": {"source": "user_measured", "status": "measured"},
                "table": {"size_m": [1.22, 0.625, 0.04]},
                "output": {"calibration_path": str(tmp_path / "calibration.yaml")},
            },
            sort_keys=False,
        )
    )

    payload = build_from_worksheet(worksheet)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASUREMENT_FIELDS"
    assert "table.top_center_world_m" in payload["missing_fields"]
    assert "left_base.translation_table_m" in payload["missing_fields"]
    assert "right_base.translation_table_m" in payload["missing_fields"]
    assert payload["missing_field_details"]["table.top_center_world_m"]["unit"] == "m"
    assert payload["missing_field_details"]["left_base.translation_table_m"]["shape"] == "[x, y, z]"
    assert "right ALOHA base" in payload["missing_field_details"]["right_base.translation_table_m"]["how_to_measure"]


def test_complete_measurement_worksheet_generates_calibration_and_passes_audit(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.yaml"
    worksheet = tmp_path / "worksheet.yaml"
    worksheet.write_text(
        yaml.safe_dump(
            {
                "measurement": {
                    "source": "user_measured",
                    "status": "measured",
                    "measured_at": "2026-07-18T00:00:00+09:00",
                    "measured_by": "unit_test",
                    "units": "meters",
                    "coordinate_frame": "Isaac world +Z up",
                    "tool": "synthetic",
                    "uncertainty_m": 0.001,
                    "real_robot_touched": False,
                    "remote_103_touched": False,
                },
                "table": {
                    "top_center_world_m": [1.0, 2.0, 0.5],
                    "size_m": [1.22, 0.625, 0.04],
                    "yaw_deg": 0.0,
                },
                "left_base": {"translation_table_m": [-0.3, 0.1, 0.0], "yaw_deg": 0.0},
                "right_base": {"translation_table_m": [0.3, 0.1, 0.0], "yaw_deg": 180.0},
                "output": {"calibration_path": str(calibration)},
            },
            sort_keys=False,
        )
    )

    payload = build_from_worksheet(worksheet)

    assert payload["status"] == "PASS_MEASUREMENT_WORKSHEET_TO_CALIBRATION"
    assert calibration.exists()
    assert payload["calibration_audit"]["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
    assert payload["calibration_audit"]["calibration_evidence"]["path"] == str(worksheet)
    assert payload["calibration_audit"]["world_base_transforms"]["T_world_left_base"]["translation"] == pytest.approx(
        [0.7, 2.1, 0.5]
    )


def test_complete_measurement_worksheet_can_override_output_path(tmp_path: Path) -> None:
    worksheet_output = tmp_path / "unused.yaml"
    override_output = tmp_path / "override.yaml"
    worksheet = tmp_path / "worksheet.yaml"
    worksheet.write_text(
        yaml.safe_dump(
            {
                "measurement": {
                    "source": "read_from_103",
                    "status": "read_from_103",
                    "measured_at": "2026-07-18T00:00:00+09:00",
                    "measured_by": "unit_test",
                    "units": "meters",
                    "coordinate_frame": "Isaac world +Z up",
                    "tool": "synthetic read-only transcript",
                    "uncertainty_m": 0.001,
                    "real_robot_touched": False,
                    "remote_103_touched": "readonly",
                },
                "table": {
                    "top_center_world_m": [0.0, 0.0, 0.2],
                    "size_m": [1.22, 0.625, 0.04],
                    "yaw_deg": 0.0,
                },
                "left_base": {"translation_table_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
                "right_base": {"translation_table_m": [0.1, 0.0, 0.0], "yaw_deg": 180.0},
                "output": {"calibration_path": str(worksheet_output)},
            },
            sort_keys=False,
        )
    )

    payload = build_from_worksheet(worksheet, output_calibration=override_output)

    assert payload["status"] == "PASS_MEASUREMENT_WORKSHEET_TO_CALIBRATION"
    assert override_output.exists()
    assert not worksheet_output.exists()


def test_read_from_103_worksheet_requires_readonly_remote_marker(tmp_path: Path) -> None:
    worksheet = tmp_path / "worksheet.yaml"
    worksheet.write_text(
        yaml.safe_dump(
            {
                "measurement": {
                    "source": "read_from_103",
                    "status": "read_from_103",
                    "measured_at": "2026-07-18T00:00:00+09:00",
                    "measured_by": "unit_test",
                    "units": "meters",
                    "coordinate_frame": "Isaac world +Z up",
                    "tool": "synthetic",
                    "uncertainty_m": 0.001,
                    "real_robot_touched": False,
                    "remote_103_touched": False,
                },
                "table": {
                    "top_center_world_m": [0.0, 0.0, 0.2],
                    "size_m": [1.22, 0.625, 0.04],
                    "yaw_deg": 0.0,
                },
                "left_base": {"translation_table_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
                "right_base": {"translation_table_m": [0.1, 0.0, 0.0], "yaw_deg": 180.0},
                "output": {"calibration_path": str(tmp_path / "calibration.yaml")},
            },
            sort_keys=False,
        )
    )

    payload = build_from_worksheet(worksheet)

    assert payload["status"] == "BLOCKED_REQUIRES_MEASUREMENT_FIELDS"
    assert "measurement.remote_103_touched must be readonly when source is read_from_103" in payload["missing_fields"]

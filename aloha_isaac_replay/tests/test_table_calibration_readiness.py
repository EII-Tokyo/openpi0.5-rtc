from __future__ import annotations

from pathlib import Path

import yaml

from aloha_isaac_replay.scripts.summarize_table_calibration_readiness import summarize_readiness


def _complete_worksheet(path: Path, calibration: Path) -> None:
    path.write_text(
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
        ),
        encoding="utf-8",
    )


def test_readiness_reports_missing_fields_without_generating_calibration(tmp_path: Path) -> None:
    worksheet = tmp_path / "worksheet.yaml"
    calibration = tmp_path / "calibration.yaml"
    worksheet.write_text(
        yaml.safe_dump(
            {
                "measurement": {"source": "user_measured", "status": "measured"},
                "table": {"size_m": [1.22, 0.625, 0.04]},
                "output": {"calibration_path": str(calibration)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = summarize_readiness(worksheet=worksheet, calibration=calibration, output_dir=tmp_path / "out")

    assert payload["status"] == "BLOCKED_REQUIRES_TABLE_BASE_MEASUREMENT"
    assert payload["worksheet"]["status"] == "BLOCKED_REQUIRES_MEASUREMENT_FIELDS"
    assert "table.top_center_world_m" in payload["worksheet"]["missing_fields"]
    assert not calibration.exists()


def test_readiness_complete_worksheet_stays_read_only_without_try_generate(tmp_path: Path) -> None:
    worksheet = tmp_path / "worksheet.yaml"
    calibration = tmp_path / "calibration.yaml"
    _complete_worksheet(worksheet, calibration)

    payload = summarize_readiness(worksheet=worksheet, calibration=calibration, output_dir=tmp_path / "out")

    assert payload["status"] == "BLOCKED_REQUIRES_TABLE_BASE_MEASUREMENT"
    assert payload["worksheet"]["status"] == "READY_TO_GENERATE_CALIBRATION"
    assert not calibration.exists()


def test_readiness_can_generate_calibration_when_explicitly_requested(tmp_path: Path) -> None:
    worksheet = tmp_path / "worksheet.yaml"
    calibration = tmp_path / "calibration.yaml"
    base = tmp_path / "base.usd"
    base.write_text("#usda 1.0\n", encoding="utf-8")
    _complete_worksheet(worksheet, calibration)

    payload = summarize_readiness(
        worksheet=worksheet,
        calibration=calibration,
        base_usd=base,
        output_dir=tmp_path / "out",
        try_generate_calibration=True,
        try_generate_overlay=True,
    )

    assert payload["status"] == "READY_FOR_CALIBRATED_OVERLAY"
    assert calibration.exists()
    assert payload["calibration"]["audit_status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"
    assert payload["overlay"]["status"] == "PASS_CALIBRATED_OVERLAY_READY_FOR_REVIEW"

from __future__ import annotations

from pathlib import Path

import yaml

from aloha_isaac_replay.scripts.create_calibrated_table_base_overlay import build_overlay
from aloha_isaac_replay.scripts.create_table_to_base_calibration_from_worksheet import build_from_worksheet


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


def test_blocked_calibration_does_not_generate_overlay(tmp_path: Path) -> None:
    calibration = tmp_path / "blocked_calibration.yaml"
    calibration.write_text(
        yaml.safe_dump(
            {
                "support_plane": {"center": [0.0, 0.0, 0.0], "size": [1.22, 0.625, 0.04]},
                "table_frame": {
                    "T_world_table": {"status": "diagnostic_candidate"},
                    "T_table_left_base": {"status": "unknown"},
                    "T_table_right_base": {"status": "unknown"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.usd"
    base.write_text("#usda 1.0\n", encoding="utf-8")

    payload = build_overlay(calibration_path=calibration, base_usd=base, output_dir=tmp_path / "out")

    assert payload["status"] == "BLOCKED_CALIBRATION_AUDIT_NOT_READY"
    assert not (tmp_path / "out" / "aloha1_calibrated_table_base_overlay.usda").exists()
    assert "T_world_table is diagnostic, not calibrated" in payload["blocking_reasons"]


def test_calibrated_overlay_uses_audited_world_base_transforms(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.yaml"
    worksheet = tmp_path / "worksheet.yaml"
    _complete_worksheet(worksheet, calibration)
    generated = build_from_worksheet(worksheet)
    assert generated["status"] == "PASS_MEASUREMENT_WORKSHEET_TO_CALIBRATION"
    base = tmp_path / "base.usd"
    base.write_text("#usda 1.0\n", encoding="utf-8")

    payload = build_overlay(
        calibration_path=calibration,
        base_usd=base,
        output_dir=tmp_path / "out",
        left_target_prim="/scene/left_base_link",
        right_target_prim="/scene/right_base_link",
    )

    assert payload["status"] == "PASS_CALIBRATED_OVERLAY_READY_FOR_REVIEW"
    overlay = tmp_path / "out" / "aloha1_calibrated_table_base_overlay.usda"
    manifest = tmp_path / "out" / "replay_command_manifest.json"
    assert overlay.exists()
    assert manifest.exists()
    overlay_text = overlay.read_text(encoding="utf-8")
    assert "@/tmp/" in overlay_text
    assert 'over "left_base_link"' in overlay_text
    assert 'over "right_base_link"' in overlay_text
    assert "double3 xformOp:translate = (0.7, 2.1, 0.5)" in overlay_text
    assert "double3 xformOp:translate = (1.3, 2.1, 0.5)" in overlay_text
    manifest_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["safety"]["simulation_only"] is True
    assert manifest_payload["status"] == "READY_FOR_REVIEW_NOT_EXECUTED"
    assert manifest_payload["world_base_transforms"]["T_world_left_base"]["translation"] == [0.7, 2.1, 0.5]
    assert "--stage-units-in-meters 1.0" in manifest_payload["contact_validation_command"]
    assert "--require-calibrated-table-frame" in manifest_payload["contact_validation_command"]
    assert "--contact-proxy-profile scene_base_link" in manifest_payload["contact_validation_command"]

import json
from pathlib import Path

from tools.build_aloha_viper_supplier_cad_follower_right_robot_schema_asset import _normalized_usda_text

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_robot_schema_asset.json"
)
BUILDER = (
    ROOT
    / "tools/"
    "build_aloha_viper_supplier_cad_follower_right_robot_schema_asset.py"
)


def test_usda_normalization_preserves_one_terminal_newline() -> None:
    assert _normalized_usda_text("#usda 1.0\n\n") == "#usda 1.0\n"


def test_right_schema_asset_is_isolated_and_machine_readable() -> None:
    assert BUILDER.is_file()
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["scope"] == "ROBOTRULES_SCHEMA_ONLY_DIAGNOSTIC"
    assert report["readback"]["robot_api"] is True
    assert len(report["readback"]["robot_links"]) >= 10
    assert len(report["readback"]["robot_joints"]) >= 10
    assert report["thumbnail"]["resolution"] == [256, 256]
    assert report["source_right_asset"]["modified"] is False
    assert report["source_geometry_layer"]["modified"] is False
    assert report["physical_right_stage_included"] is False
    assert report["final_default_collider_modified"] is False
    assert report["task8"] == "NOT_RUN"

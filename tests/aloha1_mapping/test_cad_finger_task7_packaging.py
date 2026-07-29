from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_robot_asset_v1_6.json"
)
SCHEMA_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_robot_schema_asset_v1_2.json"
)
VALIDATION_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task7_validation.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_editable_finger_colliders_read_back_as_guide_in_v1_6() -> None:
    report = _load(PHYSICAL_REPORT)
    purpose = report["collision_purpose"]

    assert report["status"] == "PASS"
    assert purpose["policy"] == (
        "DIRECT_EDITABLE_INVISIBLE_COLLIDERS_USE_GUIDE_PURPOSE"
    )
    assert purpose["authored_count"] == 2
    assert purpose["readback_count"] == 2
    assert all(
        item["visibility"] == "invisible"
        and item["purpose"] == "guide"
        for item in purpose["readback"]
    )
    assert purpose["remaining_count"] == 9
    assert all(
        item["is_instance_proxy"] is True
        and item["classification"]
        == "SOURCE_INSTANCE_PROXY_NOT_AUTHORABLE_IN_DIAGNOSTIC_LAYER"
        for item in purpose["remaining"]
    )


def test_schema_v1_2_has_deterministic_256_thumbnail() -> None:
    report = _load(SCHEMA_REPORT)
    thumbnail = report["thumbnail"]
    path = Path(thumbnail["absolute_path"])

    assert report["status"] == "PASS"
    assert path.is_file()
    assert _sha256(path) == thumbnail["sha256"]
    with Image.open(path) as opened:
        assert opened.size == (256, 256)
        assert opened.mode == "RGB"


def test_task7_rules_use_v1_6_and_schema_v1_2() -> None:
    report = _load(VALIDATION_REPORT)
    categories = {
        item["category"]: item
        for item in report["official_rules"]["categories"]
    }

    assert "/supplier_cad_follower_left/1.6/" in (
        report["validation_targets"]["IsaacSim.PhysicsRules"]
    )
    assert "/supplier_cad_follower_left_robot_schema/1.2/" in (
        report["validation_targets"]["IsaacSim.RobotRules"]
    )
    assert categories["IsaacSim.PhysicsRules"]["warning_count"] == 9
    assert categories["IsaacSim.PhysicsRules"]["status"] == "PARTIAL"
    assert categories["IsaacSim.RobotRules"]["warning_count"] == 4
    assert all(
        item["rule"] != "ThumbnailExists"
        for item in categories["IsaacSim.RobotRules"]["issues"]
    )

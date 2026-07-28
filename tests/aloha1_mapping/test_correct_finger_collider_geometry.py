from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_correct_finger_geometry_entrypoint_cooks_existing_diagnostic_assets() -> None:
    source = (
        PROJECT_ROOT / "tools/audit_aloha1_correct_finger_colliders.py"
    ).read_text(encoding="utf-8")

    assert "_cook_finger_colliders(" in source
    assert "_local_api_probe(" in source
    assert "_render_cooked(" in source
    assert "gripper_correct_finger_collider_comparison.json" in source
    assert "gripper_correct_finger_all_screenshot_manifest.json" in source


def test_correct_finger_collider_report_has_cooked_geometry_and_source_hashes() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_collider_comparison.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert report["default_asset_collider_modified"] is False
    assert report["task8"] == "NOT_RUN"
    assert set(report["profiles"]) == {"convex_hull", "convex_decomposition"}
    for profile in report["profiles"].values():
        assert len(profile["assets"]) == 2
        for asset in profile["assets"]:
            assert len(asset["colliders"]) == 2
            for collider in asset["colliders"].values():
                assert collider["piece_count"] > 0
                assert len(collider["source_stl_sha256"]) == 64
                assert collider["approximation_readback"] in {
                    "convexHull",
                    "convexDecomposition",
                }
                assert collider["visualization"]["status"] == "PASS"


def test_all_correct_finger_screenshots_are_absolute_and_complete() -> None:
    manifest = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_all_screenshot_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "PASS"
    assert manifest["required_capture_count"] == 36
    assert manifest["observed_capture_count"] == 36
    assert set(manifest["required_captures"]) == {
        "asset_preflight",
        "collider_geometry",
        "runtime_open",
        "bilateral_contact",
        "release_hold",
    }
    for capture in manifest["captures"]:
        path = Path(capture["absolute_path"])
        assert path.is_absolute()
        assert path.is_file()
        assert capture["resolution"] == [1280, 900]
        assert len(capture["file_sha256"]) == 64
        assert len(capture["decoded_pixel_sha256"]) == 64

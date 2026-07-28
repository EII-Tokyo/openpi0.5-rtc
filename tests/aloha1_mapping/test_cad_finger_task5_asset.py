from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.cad_finger_task5_asset import TASK5_COLLISION_POLICY
from tools.aloha1_mapping.cad_finger_task5_asset import task5_finger_paths

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "tools/build_aloha_viper_cad_finger_task5_asset.py"


def test_task5_policy_is_isolated_supplier_cad_convex_hull_only() -> None:
    assert TASK5_COLLISION_POLICY == {
        "approximation": "convexHull",
        "source_generic_finger_colliders": "DEACTIVATED_IN_DIAGNOSTIC_ONLY",
        "supplier_cad_mesh_role": "VISUAL_AND_DIAGNOSTIC_COLLISION",
        "source_stage_modified": False,
        "default_configuration_modified": False,
        "final_default_collider_modified": False,
        "task8": "NOT_RUN",
    }


def test_task5_paths_cover_both_followers_and_handed_fingers() -> None:
    paths = task5_finger_paths()
    assert set(paths) == {"follower_left", "follower_right"}
    stage_names = {
        "follower_left": "vx300s_left",
        "follower_right": "vx300s_right",
    }
    for robot, sides in paths.items():
        assert set(sides) == {"left", "right"}
        assert sides["left"]["cad_product"] == "Part__Feature007"
        assert sides["right"]["cad_product"] == "Part__Feature008"
        stage_robot = stage_names[robot]
        assert sides["left"]["link"].endswith(
            f"{stage_robot}_left_finger_link"
        )
        assert sides["right"]["link"].endswith(
            f"{stage_robot}_right_finger_link"
        )


def test_task5_builder_returns_nonzero_when_kit_swallows_an_exception() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "except Exception" in source
    assert "traceback.print_exc()" in source
    assert "return exit_code" in source


def test_saved_task5_asset_replaces_only_existing_follower_fingers() -> None:
    report = json.loads(
        (
            ROOT
            / "reports/aloha1_mapping/"
            "aloha_viper_cad_finger_task5_asset.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "PARTIAL"
    assert all(report["gates"].values())
    assert report["collision_policy"] == TASK5_COLLISION_POLICY
    assert report["source_follower_presence"] == {
        "follower_left": True,
        "follower_right": False,
    }
    assert len(report["new_finger_colliders"]) == 2
    assert len(report["deactivated_generic_finger_colliders"]) == 2
    assert all(
        "/vx300s_left/" in item["path"]
        for item in report["new_finger_colliders"]
    )
    assert not any(
        "/vx300s_right/" in item["path"]
        for item in report["new_finger_colliders"]
    )
    assert report["hard_blockers"] == [
        {
            "code": "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT",
            "evidence": "/workcell/vx300s_right is absent",
            "scope": "follower_right runtime Task 5",
        }
    ]
    assert report["nonfinger_collision_inventory_unchanged"] is True
    assert report["source_stage"]["sha256_before"] == report[
        "source_stage"
    ]["sha256_after"]
    for collider in report["new_finger_colliders"]:
        assert collider["approximation"] == "convexHull"
        assert collider["collision_api"] is True
        assert collider["mesh_collision_api"] is True
        assert collider["point_count"] == 831
        assert collider["face_count"] == 1662

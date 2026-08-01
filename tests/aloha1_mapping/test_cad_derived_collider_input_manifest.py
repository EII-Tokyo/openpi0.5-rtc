from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "reports/aloha1_mapping/aloha1_cad_derived_collider_input_manifest.json"
)
EXPECTED_CAD_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
EXPECTED_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)


def test_cad_derived_collider_manifest_freezes_inputs_and_boundaries() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["status"] == "PASS"
    assert manifest["schema_version"] == 1
    assert manifest["source_cad"]["sha256"] == EXPECTED_CAD_SHA256
    assert manifest["source_cad"]["read_only"] is True
    assert Path(manifest["source_cad"]["absolute_path"]).is_file()

    stage = manifest["approved_stage"]
    assert stage["sha256"] == EXPECTED_STAGE_SHA256
    assert stage["hash_matches_approval"] is True
    assert stage["default_prim"] == "/World"
    assert stage["root_prim"] == "/World"
    assert stage["sublayers"]
    assert stage["references"]
    assert stage["required_prims"] == {
        "/World/follower_left": True,
        "/World/follower_right": True,
    }

    assert manifest["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "asset_validation_extension": "1.1.0",
    }
    assert manifest["freecad"]["version"] == "1.1.1"
    assert manifest["freecad"]["occt"] == "7.8.1"
    assert manifest["freecad"]["linear_deflection_mm"] == 0.20
    assert manifest["freecad"]["angular_deflection_deg"] == 20.0
    assert manifest["freecad"]["relative"] is False

    physics = manifest["collision_baseline"]
    assert physics["profile"] == "CAD_SUBPART_COMPOUND_CONVEX_HULL"
    assert physics["visual_mesh_is_accepted_collider"] is False
    assert physics["self_collision_final_policy_modified"] is False
    assert physics["friction_modified"] is False
    assert physics["drive_modified"] is False
    assert physics["mimic_modified"] is False
    assert physics["mass_modified"] is False
    assert physics["timestep_modified"] is False
    assert physics["solver_iterations_modified"] is False

    mcp = manifest["direct_nvidia_mcp"]
    assert mcp["status"] == "PASS"
    assert mcp["transport"] == "DIRECT_NOT_MCPJUNGLE"
    assert mcp["queries"]
    assert mcp["local_5_1_source_is_version_authority"] is True

    assert manifest["source_cad_read_only"] is True
    assert manifest["source_usd_read_only"] is True
    assert manifest["final_or_default_collider_modified"] is False
    assert manifest["real_robot_connected"] is False
    assert manifest["remote_192_168_1_103_accessed"] is False
    assert manifest["task8"] == "NOT_RUN"

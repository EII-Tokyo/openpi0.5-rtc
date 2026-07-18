from __future__ import annotations

from pathlib import Path

from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_workcell_contact_policy
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _workcell_contact_policy_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml"


def _contact_summary() -> dict:
    object_path = "/World/phase43_passive_contact_cube"
    return {
        "object_contact_categories": {
            "target_finger": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/Collisions/COL_Body_00/COL_Body_00Mesh",
                        "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy",
                    ]
                ]
            },
            "workcell_or_environment": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/Collisions/COL_Body_09/COL_Body_09Mesh",
                        "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
                    ]
                ]
            },
        }
    }


def test_workcell_contact_policy_denies_phase109_frame_rail_collision() -> None:
    policy = _load_workcell_contact_policy(POLICY)

    gate = _workcell_contact_policy_gate(
        contact_summary=_contact_summary(),
        object_path="/World/phase43_passive_contact_cube",
        policy=policy,
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_WORKCELL_CONTACT_POLICY"
    assert gate["denied_semantic_classes"] == ["denied_frame_rail_collision"]
    assert gate["denied_rows"][0]["matched_path_prefix"] == "/scene/worldBody/__22"


def test_workcell_contact_policy_skips_when_not_configured() -> None:
    gate = _workcell_contact_policy_gate(
        contact_summary=_contact_summary(),
        object_path="/World/phase43_passive_contact_cube",
        policy=None,
    )

    assert gate["pass"] is True
    assert gate["status"] == "SKIPPED_NO_WORKCELL_CONTACT_POLICY"

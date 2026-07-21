from __future__ import annotations

import pytest
from pathlib import Path

from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_workcell_contact_policy
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _classify_object_contact
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import (
    _derive_open_finger_horizontal_perpendicular_axis,
)
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _local_object_support_patch_size
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _unique_pair_summaries
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _workcell_contact_policy_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml"
ACTIVE_TABLETOP_POLICY = REPO_ROOT / "examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml"


def test_object_contact_classification_accepts_no_diagnostic_support_paths() -> None:
    category = _classify_object_contact(
        {
            "collider0": "/World/phase43_passive_contact_cube/physics_proxy",
            "collider1": "/World/Table",
        },
        object_path="/World/phase43_passive_contact_cube",
        expected_finger_paths=[],
        same_side_robot_root=None,
        other_side_robot_root=None,
        diagnostic_contact_paths=None,
    )

    assert category == "workcell_or_environment"


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


def test_workcell_contact_policy_allows_measured_pipe_placeholder_contact() -> None:
    policy = _load_workcell_contact_policy(POLICY)
    object_path = "/World/Bottle500"
    contact_summary = {
        "object_contact_categories": {
            "workcell_or_environment": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/Collisions/mouth",
                        "/World/PipePlaceholder/axis/collision",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_WORKCELL_CONTACT_POLICY"
    assert gate["rows"][0]["semantic_class"] == "candidate_measured_pipe_contact"
    assert gate["rows"][0]["decision"] == "allow"


def test_workcell_contact_policy_explicitly_denies_measured_table_contact() -> None:
    policy = _load_workcell_contact_policy(POLICY)
    object_path = "/World/Bottle500"
    contact_summary = {
        "object_contact_categories": {
            "workcell_or_environment": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/Collisions/body",
                        "/World/Table/collision",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_WORKCELL_CONTACT_POLICY"
    assert gate["denied_semantic_classes"] == ["measured_table_contact_not_yet_task_valid"]


def test_active_tabletop_policy_allows_scene_table_support() -> None:
    policy = _load_workcell_contact_policy(ACTIVE_TABLETOP_POLICY)
    object_path = "/World/phase43_passive_contact_cube"
    contact_summary = {
        "object_contact_categories": {
            "workcell_or_environment": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/physics_proxy",
                        "/scene/worldBody/table/collisions/table/table/table",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_WORKCELL_CONTACT_POLICY"
    assert gate["rows"][0]["semantic_class"] == "active_tabletop_support"
    assert gate["rows"][0]["decision"] == "allow"


def test_active_tabletop_policy_allows_user_measured_table_support() -> None:
    policy = _load_workcell_contact_policy(ACTIVE_TABLETOP_POLICY)
    object_path = "/World/phase43_passive_contact_cube"
    contact_summary = {
        "object_contact_categories": {
            "workcell_or_environment": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/physics_proxy",
                        "/World/Table",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_WORKCELL_CONTACT_POLICY"
    assert gate["rows"][0]["semantic_class"] == "active_tabletop_support"
    assert gate["rows"][0]["decision"] == "allow"


def test_local_object_support_patch_size_is_derived_from_object_footprint() -> None:
    size_x, size_y = _local_object_support_patch_size(
        {"size": [0.068, 0.206, 0.068]},
        margin=0.04,
    )

    assert size_x == pytest.approx(0.148)
    assert size_y == pytest.approx(0.286)


def test_open_finger_horizontal_perpendicular_axis_is_level_and_perpendicular() -> None:
    row = _derive_open_finger_horizontal_perpendicular_axis(
        left_box={"center": [-0.0243, -0.1590, 0.1422]},
        right_box={"center": [0.0653, -0.1235, 0.1428]},
        preferred_axis="Y",
    )

    axis = row["object_axis_unit_world"]
    assert axis[2] == pytest.approx(0.0)
    assert row["abs_dot_closing_axis"] == pytest.approx(0.0, abs=1e-12)
    assert axis[1] > 0.0
    assert row["provenance"] == "DIAGNOSTIC_OPEN_FRAME_FINGER_DERIVED_BOTTLE_YAW"


def test_active_tabletop_policy_allows_diagnostic_local_object_patch() -> None:
    policy = _load_workcell_contact_policy(ACTIVE_TABLETOP_POLICY)
    object_path = "/World/phase43_passive_contact_cube"
    contact_summary = {
        "object_contact_categories": {
            "diagnostic_support": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/physics_proxy",
                        "/World/phase58_local_object_support_patch",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_WORKCELL_CONTACT_POLICY"
    assert gate["rows"][0]["semantic_class"] == "diagnostic_local_object_support_patch"
    assert gate["rows"][0]["decision"] == "allow"


def test_active_tabletop_policy_allows_diagnostic_contact_geometry_patch() -> None:
    policy = _load_workcell_contact_policy(ACTIVE_TABLETOP_POLICY)
    object_path = "/World/phase43_passive_contact_cube"
    contact_summary = {
        "object_contact_categories": {
            "diagnostic_support": {
                "unique_contact_pairs": [
                    [
                        f"{object_path}/physics_proxy",
                        "/World/phase58_contact_geometry_support_patch",
                    ]
                ]
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_WORKCELL_CONTACT_POLICY"
    assert gate["rows"][0]["semantic_class"] == "diagnostic_contact_geometry_support_patch"
    assert gate["rows"][0]["decision"] == "allow"


def test_active_tabletop_policy_still_denies_frame_rail_contact() -> None:
    policy = _load_workcell_contact_policy(ACTIVE_TABLETOP_POLICY)
    object_path = "/World/phase43_passive_contact_cube"
    contact_summary = {
        "object_contact_categories": {
            "workcell_or_environment": {
                "contact_pair_count": 3,
                "phase_counts": {"settle": 1, "close": 2},
                "first_contact_pair": {
                    "phase": "settle",
                    "step": 0,
                    "collider0": f"{object_path}/physics_proxy",
                    "collider1": "/scene/worldBody/table/collisions/table/table/table",
                    "sorted_pair": [
                        "/scene/worldBody/table/collisions/table/table/table",
                        f"{object_path}/physics_proxy",
                    ],
                },
                "unique_contact_pairs": [
                    [
                        "/scene/worldBody/table/collisions/table/table/table",
                        f"{object_path}/physics_proxy",
                    ],
                    [
                        f"{object_path}/physics_proxy",
                        "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
                    ],
                ],
                "unique_contact_pair_summaries": [
                    {
                        "pair": [
                            "/scene/worldBody/table/collisions/table/table/table",
                            f"{object_path}/physics_proxy",
                        ],
                        "contact_pair_count": 1,
                        "phase_counts": {"settle": 1},
                        "first_contact_pair": {
                            "phase": "settle",
                            "step": 0,
                            "collider0": f"{object_path}/physics_proxy",
                            "collider1": "/scene/worldBody/table/collisions/table/table/table",
                            "sorted_pair": [
                                "/scene/worldBody/table/collisions/table/table/table",
                                f"{object_path}/physics_proxy",
                            ],
                        },
                    },
                    {
                        "pair": [
                            f"{object_path}/physics_proxy",
                            "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
                        ],
                        "contact_pair_count": 2,
                        "phase_counts": {"close": 2},
                        "first_contact_pair": {
                            "phase": "close",
                            "step": 12,
                            "collider0": f"{object_path}/physics_proxy",
                            "collider1": "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
                            "sorted_pair": [
                                f"{object_path}/physics_proxy",
                                "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
                            ],
                        },
                    },
                ],
            }
        }
    }

    gate = _workcell_contact_policy_gate(
        contact_summary=contact_summary,
        object_path=object_path,
        policy=policy,
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_WORKCELL_CONTACT_POLICY"
    assert gate["denied_semantic_classes"] == ["denied_frame_rail_collision"]
    assert gate["denied_rows"][0]["category_contact_pair_count"] == 3
    assert gate["denied_rows"][0]["category_phase_counts"] == {"settle": 1, "close": 2}
    assert (
        gate["denied_rows"][0]["category_first_contact_pair"]["collider1"]
        == "/scene/worldBody/table/collisions/table/table/table"
    )
    assert gate["denied_rows"][0]["pair_contact_pair_count"] == 2
    assert gate["denied_rows"][0]["pair_phase_counts"] == {"close": 2}
    assert gate["denied_rows"][0]["pair_first_contact_pair"]["step"] == 12
    assert (
        gate["denied_rows"][0]["pair_first_contact_pair"]["collider1"]
        == "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220"
    )


def test_pair_summaries_are_generated_from_contact_rows() -> None:
    object_path = "/World/phase43_passive_contact_cube"
    rows = [
        {
            "phase": "settle",
            "step": 0,
            "sorted_pair": [
                "/scene/worldBody/table/collisions/table/table/table",
                f"{object_path}/physics_proxy",
            ],
            "collider0": f"{object_path}/physics_proxy",
            "collider1": "/scene/worldBody/table/collisions/table/table/table",
        },
        {
            "phase": "close",
            "step": 12,
            "sorted_pair": [
                f"{object_path}/physics_proxy",
                "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
            ],
            "collider0": f"{object_path}/physics_proxy",
            "collider1": "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
        },
        {
            "phase": "close",
            "step": 13,
            "sorted_pair": [
                f"{object_path}/physics_proxy",
                "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
            ],
            "collider0": f"{object_path}/physics_proxy",
            "collider1": "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
        },
    ]

    summaries = _unique_pair_summaries(rows)
    rail_summary = next(
        summary
        for summary in summaries
        if summary["pair"]
        == [
            f"{object_path}/physics_proxy",
            "/scene/worldBody/__22/collisions/__22/__22/extrusion_1220",
        ]
    )

    assert rail_summary["contact_pair_count"] == 2
    assert rail_summary["phase_counts"] == {"close": 2}
    assert rail_summary["first_contact_pair"]["step"] == 12

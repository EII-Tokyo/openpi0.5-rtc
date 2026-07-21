from __future__ import annotations

from aloha_isaac_replay.scripts.audit_aloha1_replay_collider_baseline import (
    baseline_findings,
    baseline_status,
    classify_path,
    ccd_policy_from_evidence,
    collider_shape_family,
)


def test_classify_path_finger_and_bottle() -> None:
    assert classify_path("/scene/left_base_link/left_left_finger_link/collisions/foo") == "finger"
    assert classify_path("/World/Bottle500/physics_proxy/body") == "bottle"
    assert classify_path("/World/phase43_passive_contact_cube/physics_proxy/body") == "bottle"
    assert (
        classify_path("/World/OfficeEnvironment/industrial_lab_corner/official_props/warehouse_bottle_reference")
        == "workcell_or_environment"
    )
    assert classify_path("/World/PipePlaceholder/support_base_placeholder") == "workcell_or_environment"


def test_dynamic_robot_complex_mesh_without_approximation_is_hard_finding() -> None:
    row = {
        "category": "robot_link",
        "shape_family": "complex_mesh_unspecified",
        "type_name": "Mesh",
        "mesh_points": 512,
        "mesh_faces": 700,
        "approximation": None,
        "has_rigid_body_api": False,
        "rigid_body_ancestor": "/scene/left_base_link",
        "bbox_max_dim": 0.25,
        "collision_enabled": True,
    }

    findings = baseline_findings(row)

    assert "dynamic_robot_collision_uses_complex_mesh_without_explicit_approximation" in findings
    assert "dynamic_mesh_collision_requires_explicit_supported_approximation_review" in findings
    assert baseline_status([{**row, "findings": findings}]) == "NEEDS_COLLIDER_BASELINE_REPAIR"


def test_bottle_cylinder_baseline_passes() -> None:
    row = {
        "category": "bottle",
        "shape_family": "cylinder",
        "type_name": "Cylinder",
        "mesh_points": None,
        "mesh_faces": None,
        "approximation": None,
        "has_rigid_body_api": False,
        "rigid_body_ancestor": "/World/Bottle500",
        "bbox_max_dim": 0.07,
        "collision_enabled": True,
    }

    findings = baseline_findings(row)

    assert findings == []
    assert baseline_status([{**row, "findings": findings}]) == "PASS_BASELINE_GEOMETRY_REVIEW"


def test_shape_family_uses_explicit_convex_hull_for_mesh() -> None:
    assert collider_shape_family("Mesh", "convexHull", 400, 600) == "mesh_convex_hull"
    assert collider_shape_family("Mesh", None, 400, 600) == "complex_mesh_unspecified"


def test_disabled_visual_collision_is_not_a_hard_gate() -> None:
    row = {
        "category": "bottle",
        "shape_family": "complex_mesh_unspecified",
        "type_name": "Mesh",
        "mesh_points": 1000,
        "mesh_faces": 1000,
        "approximation": None,
        "has_rigid_body_api": False,
        "rigid_body_ancestor": "/World/Bottle500",
        "bbox_max_dim": 0.2,
        "collision_enabled": False,
    }

    assert baseline_findings(row) == []


def test_ccd_policy_defaults_off_when_baseline_is_clean() -> None:
    policy = ccd_policy_from_evidence(
        baseline_status_value="PASS_BASELINE_GEOMETRY_REVIEW",
        active_rows=[
            {
                "path": "/World/phase43_passive_contact_cube/physics_proxy/body",
                "category": "bottle",
                "findings": [],
                "ccd_attrs": {},
            }
        ],
        command_spike_report=None,
    )

    assert policy["default_ccd_policy"] == "off_by_default"
    assert policy["ccd_enabled_any"] is False
    assert policy["ccd_recommendation"] == "CCD_NOT_NEEDED"
    assert policy["ccd_required_for_pass"] is False


def test_ccd_policy_blocks_on_bad_collider_baseline() -> None:
    policy = ccd_policy_from_evidence(
        baseline_status_value="NEEDS_COLLIDER_BASELINE_REPAIR",
        active_rows=[],
        command_spike_report=None,
    )

    assert policy["ccd_recommendation"] == "CCD_NOT_ALLOWED_BAD_COLLIDER_BASELINE"
    assert "COLLIDER_SEMANTICS_NOT_VERIFIED" in policy["ccd_blockers_before_recommendation"]


def test_ccd_policy_command_spike_precedes_ccd() -> None:
    policy = ccd_policy_from_evidence(
        baseline_status_value="PASS_BASELINE_GEOMETRY_REVIEW",
        active_rows=[],
        command_spike_report={"failure_classification": "REPEATED_SPIKE_CLUSTER"},
    )

    assert policy["ccd_recommendation"] == "FIX_COMMAND_TARGET_CONTINUITY_BEFORE_CCD"
    assert "COMMAND_SMOOTHNESS_NOT_VERIFIED" in policy["ccd_blockers_before_recommendation"]


def test_ccd_policy_reports_enabled_bodies_without_recommending_global_ccd() -> None:
    policy = ccd_policy_from_evidence(
        baseline_status_value="PASS_BASELINE_GEOMETRY_REVIEW",
        active_rows=[
            {
                "path": "/World/Bottle500",
                "category": "bottle",
                "findings": [],
                "ccd_attrs": {"physxRigidBody:enableCCD": True},
            }
        ],
        command_spike_report=None,
    )

    assert policy["ccd_enabled_any"] is True
    assert policy["active_ccd_body_count"] == 1
    assert policy["ccd_scope"] == "specific_bodies_only"

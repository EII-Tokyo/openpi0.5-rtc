import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.task7_physicsrules_root_cause import ALLOWED_HELPER_CLASSES
from tools.aloha1_mapping.task7_physicsrules_root_cause import FROZEN_STAGE_SHA256
from tools.aloha1_mapping.task7_physicsrules_root_cause import REQUIRED_RULE_COUNTS
from tools.aloha1_mapping.task7_physicsrules_root_cause import build_finding_inventory
from tools.aloha1_mapping.task7_physicsrules_root_cause import build_hypothesis_signature
from tools.aloha1_mapping.task7_physicsrules_root_cause import classify_collider_finding
from tools.aloha1_mapping.task7_physicsrules_root_cause import mapped_mimic_interval
from tools.aloha1_mapping.task7_physicsrules_root_cause import mapped_physx_mimic_interval
from tools.aloha1_mapping.task7_physicsrules_root_cause import should_escalate_screenshot
from tools.aloha1_mapping.task7_physicsrules_root_cause import summarize_runtime_trace

ROOT = Path(__file__).resolve().parents[2]
JOINT_AUDIT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_joint_state_geometry_audit.json"
)
HELPER_AUDIT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_helper_link_collider_audit.json"
)
MIMIC_AUDIT = ROOT / "reports/aloha1_mapping/aloha1_task7_mimic_limit_audit.json"
CANDIDATE_BUILD = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_candidates.json"
)
MATRIX = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_matrix.json"
)
TOPOLOGY_CANDIDATE = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_virtual_helper_topology_candidate.json"
)
HELPER_MASS_AUDIT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_virtual_helper_mass_audit.json"
)
COMBINED_CANDIDATE = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_physicsrules_combined_candidate.json"
)
ROOT_CAUSE_CLOSURE = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_closure.json"
)


def _load_candidate_issues(side: str) -> list[dict[str, object]]:
    report = json.loads(
        (
            ROOT
            / "reports/aloha1_mapping"
            / f"aloha1_cad_derived_task7_candidate_{side}_physics.json"
        ).read_text(encoding="utf-8")
    )
    return [item for item in report["issues"] if item["severity"] == "ERROR"]


def test_exact_frozen_stage_and_physicsrule_inventory() -> None:
    issues = _load_candidate_issues("left") + _load_candidate_issues("right")

    inventory = build_finding_inventory(issues)

    assert FROZEN_STAGE_SHA256 == (
        "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
    )
    assert inventory["finding_count"] == 20
    assert inventory["rule_counts"] == REQUIRED_RULE_COUNTS == {
        "JointHasCorrectTransformAndState": 10,
        "MimicAPICheck": 2,
        "RigidBodyHasCollider": 8,
    }
    assert inventory["prim_paths"] == sorted(inventory["prim_paths"])
    assert len(set(inventory["prim_paths"])) == 18


def test_finding_inventory_rejects_unexpected_or_duplicate_findings() -> None:
    issues = _load_candidate_issues("left") + _load_candidate_issues("right")

    with pytest.raises(ValueError, match="duplicate finding"):
        build_finding_inventory([*issues, dict(issues[0])])

    mutated = [dict(item) for item in issues]
    mutated[0]["rule"] = "UnexpectedRule"
    with pytest.raises(ValueError, match="unexpected rule counts"):
        build_finding_inventory(mutated)


def test_helper_classifications_are_closed_and_non_promotional() -> None:
    assert {
        "PHYSICAL_LINK_REQUIRES_COLLIDER",
        "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY",
        "INCONCLUSIVE",
    } == ALLOWED_HELPER_CLASSES


def test_two_fresh_failures_trigger_screenshot_escalation() -> None:
    one_failure = [{"fresh_process": 1, "status": "FAIL", "signature": "a"}]
    repeated_failure = [
        {"fresh_process": 1, "status": "FAIL", "signature": "a"},
        {"fresh_process": 2, "status": "FAIL", "signature": "a"},
    ]

    assert should_escalate_screenshot(one_failure) is False
    assert should_escalate_screenshot(repeated_failure) is True
    assert should_escalate_screenshot(
        [repeated_failure[0], {**repeated_failure[1], "status": "PASS"}]
    ) is False


def test_hypothesis_signature_is_order_independent_but_variable_sensitive() -> None:
    left = {
        "hypothesis": "helper_body_semantics",
        "changed_variable": "remove_rigid_body_api",
        "target_prims": ["/right", "/left"],
        "source_hash": "abc",
    }
    reordered = {**left, "target_prims": ["/left", "/right"]}
    different = {**left, "changed_variable": "add_collider"}

    assert build_hypothesis_signature(left) == build_hypothesis_signature(reordered)
    assert build_hypothesis_signature(left) != build_hypothesis_signature(different)


def test_joint_state_geometry_audit_is_read_only_and_complete() -> None:
    report = json.loads(JOINT_AUDIT.read_text(encoding="utf-8"))

    assert report["status"] in {"PASS", "PARTIAL"}
    assert report["stage"]["sha256_before"] == FROZEN_STAGE_SHA256
    assert report["stage"]["sha256_after"] == FROZEN_STAGE_SHA256
    assert report["stage"]["modified"] is False
    assert report["finding_count"] == 10
    assert len(report["joints"]) == 10
    assert report["candidate_authoring_allowed"] is False
    for joint in report["joints"]:
        assert joint["rule"] == "JointHasCorrectTransformAndState"
        assert len(joint["body0_targets"]) == 1
        assert len(joint["body1_targets"]) == 1
        assert len(joint["expected_transform_from_body0"]) == 4
        assert len(joint["expected_transform_from_body1"]) == 4
        assert joint["axis"] in {"X", "Y", "Z"}
        assert joint["joint_type"] in {
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
        }
        assert "authored_state_position" in joint
        assert "geometry_derived_state_position" in joint
        assert "residual_before" in joint
        assert joint["source_layer"]
        assert joint["usd_modified"] is False


def test_collider_finding_classification_requires_source_geometry() -> None:
    assert (
        classify_collider_finding(
            visual_count=1,
            collision_count=1,
            incoming_joint_types=["fixed"],
        )
        == "PHYSICAL_LINK_REQUIRES_COLLIDER"
    )
    assert (
        classify_collider_finding(
            visual_count=0,
            collision_count=0,
            incoming_joint_types=["fixed"],
        )
        == "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY"
    )
    assert (
        classify_collider_finding(
            visual_count=0,
            collision_count=0,
            incoming_joint_types=["revolute"],
        )
        == "INCONCLUSIVE"
    )


def test_helper_link_collider_audit_covers_all_eight_findings() -> None:
    report = json.loads(HELPER_AUDIT.read_text(encoding="utf-8"))

    assert report["status"] in {"PASS", "PARTIAL"}
    assert report["finding_count"] == 8
    assert report["stage"]["sha256_before"] == FROZEN_STAGE_SHA256
    assert report["stage"]["sha256_after"] == FROZEN_STAGE_SHA256
    assert report["stage"]["modified"] is False
    assert report["classification_counts"] == {
        "PHYSICAL_LINK_REQUIRES_COLLIDER": 2,
        "VIRTUAL_HELPER_SHOULD_NOT_BE_RIGID_BODY": 6,
    }
    for item in report["findings"]:
        assert item["classification"] in ALLOWED_HELPER_CLASSES
        assert item["source_urdf"]["incoming_joints"]
        assert item["usd"]["has_rigid_body_api"] is True
        assert item["usd"]["descendant_collision_count"] == 0
        assert item["usd_modified"] is False


def test_negative_gearing_maps_and_sorts_interval_endpoints() -> None:
    assert mapped_mimic_interval(
        reference_lower=0.021,
        reference_upper=0.057,
        gearing=-1.0,
    ) == pytest.approx((-0.057, -0.021))
    assert mapped_physx_mimic_interval(
        reference_lower=0.021,
        reference_upper=0.057,
        gearing=1.0,
    ) == pytest.approx((-0.057, -0.021))


def test_mimic_audit_covers_both_followers_without_mutation() -> None:
    report = json.loads(MIMIC_AUDIT.read_text(encoding="utf-8"))

    assert report["status"] in {"PASS", "PARTIAL"}
    assert report["finding_count"] == 2
    assert report["stage"]["sha256_before"] == FROZEN_STAGE_SHA256
    assert report["stage"]["sha256_after"] == FROZEN_STAGE_SHA256
    assert report["stage"]["modified"] is False
    for item in report["findings"]:
        assert item["mimic_joint"].endswith("/joints/right_finger")
        assert item["reference_joint"].endswith("/joints/left_finger")
        assert item["gearing"] == pytest.approx(1.0)
        assert item["mapped_reference_interval"] == pytest.approx(
            [-0.057, -0.021]
        )
        assert item["physx_equation"] == (
            "jointPosition + gearing * referenceJointPosition + offset = 0"
        )
        assert item["effective_reference_multiplier"] == pytest.approx(-1.0)
        assert item["usd_modified"] is False


def test_root_cause_candidates_are_isolated_and_single_variable() -> None:
    report = json.loads(CANDIDATE_BUILD.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["frozen_stage"]["sha256_before"] == FROZEN_STAGE_SHA256
    assert report["frozen_stage"]["sha256_after"] == FROZEN_STAGE_SHA256
    assert report["frozen_stage"]["modified"] is False
    assert set(report["profiles"]) == {
        "joint_state_zero",
        "virtual_helpers_without_rigid_body",
        "baseline_gripper_fixed_group_split",
    }
    for profile, candidates in report["profiles"].items():
        assert len(candidates) == 2
        for item in candidates:
            assert item["profile"] == profile
            assert item["scope"] == "DIAGNOSTIC_ONLY_NOT_FINAL"
            assert item["source_candidate"]["modified"] is False
            assert item["wrapper"]["sha256"]
            assert item["override_layer"]["sha256"]
            assert item["changed_variable_count"] == 1
            if profile == "baseline_gripper_fixed_group_split":
                after = item["change"]["after"]
                assert after["cad_group"]["active"] is False
                assert after["cad_group"]["collision_descendants"] == []
                assert after["source_gripper"]["active"] is True
                assert after["source_gripper"]["collision_descendants"]
                assert after["source_bar"]["active"] is True
                assert after["source_bar"]["collision_descendants"]
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_root_cause_matrix_records_targeted_deltas_and_rejected_failure_evidence() -> None:
    report = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["validator_fresh_process_count"] == 20
    assert report["runtime_fresh_process_count"] == 20
    assert report["frozen_stage_sha256"] == FROZEN_STAGE_SHA256
    assert report["profile_decisions"] == {
        "baseline_gripper_fixed_group_split": (
            "TARGETED_FIX_VERIFIED_RUNTIME_STABLE_GRASP_REGRESSION_REQUIRED"
        ),
        "combined_topology_joint_state": (
            "VALIDATOR_REDUCED_TO_KNOWN_MIMIC_CONFLICT_"
            "PHYSICS_EQUIVALENCE_BLOCKED"
        ),
        "joint_state_zero": "TARGETED_FIX_VERIFIED_RUNTIME_EQUIVALENT",
        "virtual_helpers_without_rigid_body": "REJECTED_REPEATABLE_REGRESSION",
        "virtual_helper_topology_collapse": (
            "TARGETED_TOPOLOGY_FIX_VERIFIED_PHYSICS_EQUIVALENCE_BLOCKED"
        ),
    }
    for profile in report["profiles"].values():
        for follower in profile["followers"].values():
            assert follower["fresh_process_count"] == 2
            assert follower["repeat_signatures_identical"] is True
    rejected = report["profiles"]["virtual_helpers_without_rigid_body"]
    assert rejected["new_rule_counts"] == {
        "NonAdjacentCollisionMeshesDoNotClash": 114
    }
    assert rejected["screenshot_escalation"]["status"] == "PASS"
    assert len(rejected["screenshot_escalation"]["review_reports"]) == 2
    assert report["mimic_decision"] == (
        "KEEP_VALID_PHYSX_107_3_AUTHORING_VALIDATOR_1_1_0_FORMULA_MISMATCH"
    )
    combined = report["profiles"]["combined_topology_joint_state"]
    assert combined["blocking_rule_counts"] == {"MimicAPICheck": 2}
    assert combined["runtime"]["all_pass"] is True
    assert combined["runtime"]["all_repeat_signatures_identical"] is True
    assert report["helper_mass_semantics"]["removed_mass_per_follower_kg"] \
        == pytest.approx(0.003)
    assert report["helper_mass_semantics"]["physically_calibrated"] is False
    assert report["next_gate"] == "HELPER_MASS_INERTIA_AGGREGATION_OR_AUTHORIZED_REMOVAL"
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_runtime_trace_uses_existing_first_frame_gate_and_keeps_finger_diagnostic() -> None:
    names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    stable = summarize_runtime_trace(
        dof_names=names,
        expected_dof_names=names,
        samples=[
            {"frame": 0, "positions": [0.0] * 9},
            {"frame": 1, "positions": [0.001] * 6 + [0.0, 0.002, -0.002]},
            {"frame": 120, "positions": [0.001] * 6 + [0.0, 0.003, -0.003]},
        ],
        first_frame_arm_gate_rad=0.020,
    )
    assert stable["status"] == "PASS"
    assert stable["first_frame_arm_jump_max_abs_rad"] == pytest.approx(0.001)
    assert stable["first_frame_finger_jump_max_abs_m"] == pytest.approx(0.002)
    assert stable["finger_jump_gate"] == "RECORDED_NOT_GATED_NO_FROZEN_TOLERANCE"

    jumping = summarize_runtime_trace(
        dof_names=names,
        expected_dof_names=names,
        samples=[
            {"frame": 0, "positions": [0.0] * 9},
            {"frame": 1, "positions": [0.0, 0.03] + [0.0] * 7},
        ],
        first_frame_arm_gate_rad=0.020,
    )
    assert jumping["status"] == "FAIL"
    assert jumping["failure_reasons"] == ["FIRST_FRAME_ARM_JUMP_EXCEEDS_FROZEN_GATE"]


def test_virtual_helper_topology_candidate_preserves_joint_world_frames() -> None:
    report = json.loads(TOPOLOGY_CANDIDATE.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["geometry_topology_status"] == "PASS"
    assert report["physics_equivalence"] == "PARTIAL_HELPER_MASS_NOT_CONSERVED"
    assert report["mass_semantics_modified"] is True
    assert report["other_physics_parameters_modified"] is False
    assert report["scope"] == "DIAGNOSTIC_ONLY_NOT_FINAL"
    assert report["input_profile"] == "baseline_gripper_fixed_group_split"
    assert report["frozen_stage_sha256"] == FROZEN_STAGE_SHA256
    assert len(report["candidates"]) == 2
    for candidate in report["candidates"]:
        assert candidate["helper_body_count"] == 3
        assert candidate["disabled_fixed_joint_count"] == 3
        assert candidate["reparented_joint_count"] == 4
        assert candidate["maximum_joint_world_frame_residual"] < 1.0e-9
        assert candidate["source_modified"] is False
        assert candidate["wrapper"]["sha256"]
        assert candidate["override_layer"]["sha256"]
    assert report["joint_state_modified"] is False
    assert report["mimic_modified"] is False
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_virtual_helper_mass_audit_blocks_uncompensated_collapse() -> None:
    report = json.loads(HELPER_MASS_AUDIT.read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["frozen_stage_sha256"] == FROZEN_STAGE_SHA256
    assert len(report["followers"]) == 2
    for follower in report["followers"]:
        assert len(follower["helper_bodies"]) == 3
        assert follower["total_helper_mass_kg"] > 0.0
        for body in follower["helper_bodies"]:
            assert body["mass_kg"] > 0.0
            assert body["center_of_mass_authored"] is False
            assert body["center_of_mass_raw_readback"] == ["-inf", "-inf", "-inf"]
            assert body["center_of_mass_effective_local_m"] == [0.0, 0.0, 0.0]
            assert body["center_of_mass_effective_source"] == (
                "URDF_INERTIAL_ORIGIN_DEFAULT_IDENTITY"
            )
            assert len(body["diagonal_inertia_kg_m2"]) == 3
            assert len(body["principal_axes_wxyz"]) == 4
            assert len(body["world_matrix"]) == 4
    assert report["uncompensated_collapse_allowed"] is False
    assert report["physical_calibration_status"] == (
        "SOURCE_AUTHORED_PLACEHOLDER_NOT_PHYSICALLY_VERIFIED"
    )
    assert report["final_or_default_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"


def test_combined_candidate_is_isolated_and_carries_mass_blocker() -> None:
    report = json.loads(COMBINED_CANDIDATE.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["scope"] == "DIAGNOSTIC_ONLY_NOT_FINAL"
    assert report["input_profile"] == "virtual_helper_topology_collapse"
    assert len(report["candidates"]) == 2
    for candidate in report["candidates"]:
        assert candidate["joint_state_position_count"] == 5
        assert candidate["source_modified"] is False
        assert candidate["wrapper"]["sha256"]
    assert report["physics_equivalence"] == "PARTIAL_HELPER_MASS_NOT_CONSERVED"
    assert report["mimic_modified"] is False
    assert report["final_or_default_asset_modified"] is False
    assert report["promotion_status"] == "BLOCKED_USER_REVIEW_AND_MASS_SEMANTICS"
    assert report["task8"] == "NOT_RUN"


def test_root_cause_closure_keeps_task7_partial_and_evidence_paths_absolute() -> None:
    report = json.loads(ROOT_CAUSE_CLOSURE.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["task7"] == "PARTIAL"
    assert report["task8"] == "NOT_RUN"
    assert report["original_physicsrules_finding_count"] == 20
    assert report["combined_candidate_literal_blocking_count"] == 2
    assert report["combined_candidate_literal_rule_counts"] == {"MimicAPICheck": 2}
    assert report["failure_evidence"]["status"] == "PARTIAL"
    assert report["failure_evidence"]["visual_evidence_legibility"] == "PASS"
    assert (
        report["failure_evidence"]["finger_installation_and_collision_gate"]
        == "NOT_RUN"
    )
    assert len(report["failure_evidence"]["captures"]) == 4
    for capture in report["failure_evidence"]["captures"]:
        assert Path(capture["raw_absolute_path"]).is_absolute()
        assert Path(capture["annotated_absolute_path"]).is_absolute()
        assert capture["raw_sha256"]
        assert capture["annotated_sha256"]
        assert capture["visual_evidence_legibility"] == "PASS"
        assert capture["finger_installation_and_collision_gate"] == "NOT_RUN"
    assert report["remaining_real_blockers"] == [
        "HELPER_MASS_COM_INERTIA_SEMANTICS_NOT_PRESERVED_IN_TOPOLOGY_CANDIDATE",
        "COLLIDER_SPLIT_AND_TOPOLOGY_CANDIDATE_NOT_PROMOTED_OR_GRASP_REGRESSED",
    ]
    assert report["final_or_default_asset_modified"] is False

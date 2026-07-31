from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.bottle_collision_runtime_audit import canonical_probe_signature
from tools.aloha1_mapping.bottle_collision_runtime_audit import classify_collision_root_cause
from tools.aloha1_mapping.bottle_collision_runtime_audit import evaluate_collision_probe
from tools.aloha1_mapping.bottle_collision_runtime_audit import evaluate_follower_finger_collision_probe
from tools.aloha1_mapping.bottle_collision_runtime_audit import measure_aabb_registration

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_bottle_collision_runtime_audit.yaml"
RUNTIME_DRIVER = ROOT / "tools/audit_aloha1_bottle_collision_runtime.py"
FINGER_AUDIT_DRIVER = ROOT / "tools/audit_aloha1_follower_finger_collision_registration.py"
FINGER_RUNTIME_DRIVER = ROOT / "tools/validate_aloha1_follower_finger_collision_runtime.py"
FINGER_ANNOTATION_DRIVER = (
    ROOT / "tools/annotate_aloha1_follower_finger_collision_runtime.py"
)
COLLISION_DIAGNOSIS_FINALIZER = (
    ROOT / "tools/finalize_aloha1_bottle_graspable_object_collision_diagnosis.py"
)


def _passing_probe() -> dict:
    return {
        "probe_kind": "STANDARD_KINEMATIC_PUSHER",
        "frozen_inputs_verified": True,
        "explicit_product_prim": "/Bottle500",
        "rigid_body": {
            "enabled": True,
            "kinematic_during_push": False,
            "gravity_enabled": True,
            "mass_kg": 0.020,
        },
        "colliders": {
            "count": 41,
            "all_enabled": True,
            "approximation_tokens": ["convexHull"],
            "filtered_pair_with_probe": False,
        },
        "registration": {
            "bottle_max_transform_residual_m": 0.0,
            "bottle_max_aabb_surface_gap_m": 0.0008,
            "probe_max_transform_residual_m": 0.0,
            "probe_max_aabb_surface_gap_m": 0.0001,
        },
        "contacts": [
            {
                "physical": True,
                "actor0_path": "/World/BottleCollisionDiagnosticSession/Pusher",
                "actor1_path": "/World/BottleCollisionDiagnosticSession/Bottle500",
                "collider0_path": "/World/BottleCollisionDiagnosticSession/Pusher",
                "collider1_path": (
                    "/World/BottleCollisionDiagnosticSession/Bottle500/Collisions/COL_Body_00/COL_Body_00Mesh"
                ),
                "impulse_ns": 0.001,
                "separation_m": -0.0001,
            }
        ],
        "response": {
            "push_direction_world": [1.0, 0.0, 0.0],
            "bottle_displacement_world_m": [0.006, 0.0001, 0.0],
            "maximum_speed_m_s": 0.04,
            "trajectory_intersects_collision_envelope": True,
        },
        "captures": {
            "required_phases": [
                "pre_contact",
                "first_contact",
                "maximum_compression",
                "post_contact",
            ],
            "paired_records": [
                {
                    "phase": phase,
                    "physics_frame": frame,
                    "normal_path": f"/tmp/{phase}_normal.png",
                    "overlay_path": f"/tmp/{phase}_overlay.png",
                    "same_camera_pose": True,
                    "same_physics_frame": True,
                }
                for frame, phase in enumerate(
                    (
                        "pre_contact",
                        "first_contact",
                        "maximum_compression",
                        "post_contact",
                    ),
                    start=10,
                )
            ],
        },
        "forbidden": {
            "surface_gripper": False,
            "fixed_joint": False,
            "parent_attachment": False,
            "runtime_bottle_teleport": False,
            "source_asset_modified": False,
        },
        "limits": {
            "minimum_response_m": 0.001,
            "maximum_transform_residual_m": 1.0e-6,
            "maximum_aabb_surface_gap_m": 0.002,
        },
    }


def test_passing_probe_requires_contact_response_and_synchronized_overlay() -> None:
    result = evaluate_collision_probe(_passing_probe())

    assert result["status"] == "PASS"
    assert result["root_cause"] == "COLLISION_PIPELINE_VERIFIED"
    assert result["gates"]["physical_contact"]
    assert result["gates"]["expected_direction_response"]
    assert result["gates"]["capture_pair_synchronization"]


def test_missing_colliders_is_a_hard_failure() -> None:
    probe = _passing_probe()
    probe["colliders"]["count"] = 0
    probe["colliders"]["all_enabled"] = False
    probe["contacts"] = []

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "BOTTLE_COLLISION_MISSING_OR_DISABLED"


def test_filtered_pair_cannot_be_reported_as_collision_pass() -> None:
    probe = _passing_probe()
    probe["colliders"]["filtered_pair_with_probe"] = True
    probe["contacts"] = []

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "COLLISION_FILTERING_OR_MASK"


def test_visual_collider_misregistration_rejects_contact_only_pass() -> None:
    probe = _passing_probe()
    probe["registration"]["bottle_max_transform_residual_m"] = 0.012

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "BOTTLE_VISUAL_COLLIDER_MISREGISTRATION"


def test_finger_misregistration_is_classified_independently() -> None:
    probe = _passing_probe()
    probe["probe_kind"] = "SUPPLIER_CAD_FOLLOWER_FINGER"
    probe["registration"]["probe_max_transform_residual_m"] = 0.009

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "FINGER_VISUAL_COLLIDER_MISREGISTRATION"


def test_no_bottle_response_rejects_finite_contact_event() -> None:
    probe = _passing_probe()
    probe["response"]["bottle_displacement_world_m"] = [0.00001, 0.0, 0.0]

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "BOTTLE_RIGID_BODY_CONFIGURATION"


def test_probe_that_never_reaches_bottle_is_inconclusive_not_solver_failure() -> None:
    probe = _passing_probe()
    probe["response"]["trajectory_intersects_collision_envelope"] = False
    probe["contacts"] = []

    result = evaluate_collision_probe(probe)

    assert result["status"] == "FAIL"
    assert result["root_cause"] == "INCONCLUSIVE"
    assert result["gates"]["trajectory_intersects_collision_envelope"] is False


def test_unsynchronized_overlay_is_not_visual_evidence() -> None:
    probe = _passing_probe()
    probe["captures"]["paired_records"][1]["same_physics_frame"] = False

    result = evaluate_collision_probe(probe)

    assert result["status"] == "PARTIAL"
    assert result["root_cause"] == "VIDEO_PHYSICS_FRAME_MISMATCH"


def test_signature_is_order_independent_for_mapping_keys() -> None:
    probe = _passing_probe()
    reordered = copy.deepcopy(probe)
    reordered["rigid_body"] = dict(reversed(list(reordered["rigid_body"].items())))

    assert canonical_probe_signature(probe) == canonical_probe_signature(reordered)


def test_signature_ignores_artifact_capture_paths() -> None:
    probe = _passing_probe()
    relocated = copy.deepcopy(probe)
    for record in relocated["captures"]["paired_records"]:
        record["normal_path"] = "/different/root/normal.png"
        record["overlay_path"] = "/different/root/overlay.png"

    assert canonical_probe_signature(probe) == canonical_probe_signature(relocated)


def test_aabb_registration_reports_center_and_surface_residuals() -> None:
    result = measure_aabb_registration(
        visual_minimum=[-0.01, -0.02, -0.03],
        visual_maximum=[0.01, 0.02, 0.03],
        collider_minimum=[-0.009, -0.021, -0.03],
        collider_maximum=[0.011, 0.019, 0.031],
    )

    assert result["center_delta_collider_minus_visual_m"] == pytest.approx(
        [0.001, -0.001, 0.0005]
    )
    assert result["maximum_center_residual_m"] == pytest.approx(0.001)
    assert result["maximum_surface_gap_m"] == pytest.approx(0.001)
    assert result["visual_size_m"] == pytest.approx([0.02, 0.04, 0.06])
    assert result["collider_size_m"] == pytest.approx([0.02, 0.04, 0.061])


def test_root_cause_classifier_uses_fixed_vocabulary() -> None:
    result = evaluate_collision_probe(_passing_probe())

    assert classify_collision_root_cause(result["gates"], "STANDARD_KINEMATIC_PUSHER") in {
        "BOTTLE_COLLISION_MISSING_OR_DISABLED",
        "BOTTLE_RIGID_BODY_CONFIGURATION",
        "COLLISION_FILTERING_OR_MASK",
        "BOTTLE_VISUAL_COLLIDER_MISREGISTRATION",
        "FINGER_VISUAL_COLLIDER_MISREGISTRATION",
        "VIDEO_PHYSICS_FRAME_MISMATCH",
        "SOLVER_OR_TUNNELING_SUSPECTED",
        "COLLISION_PIPELINE_VERIFIED",
        "INCONCLUSIVE",
    }


def test_config_freezes_stage_bottle_and_overlay_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "physics_frequency_hz": 60,
    }
    assert config["stage"]["absolute_path"].endswith(
        "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda"
    )
    assert config["stage"]["sha256"] == "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
    assert config["bottle"]["reference_prim"] == "/Bottle500"
    assert config["bottle"]["sha256"] == "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
    assert config["bottle"]["mass_kg"] == 0.020
    assert config["capture"]["paired_modes"] == ["normal", "physics_collider_overlay"]
    assert config["capture"]["same_camera_and_frame_required"] is True
    assert config["forbidden"] == {
        "surface_gripper": True,
        "fixed_joint": True,
        "parent_attachment": True,
        "runtime_bottle_teleport_after_release": True,
        "source_or_final_asset_write": True,
    }


def test_runtime_driver_has_explicit_reference_and_overlay_pair_contract() -> None:
    source = RUNTIME_DRIVER.read_text(encoding="utf-8")

    assert 'Sdf.Path("/Bottle500")' in source
    assert '"/persistent/physics/visualizationDisplayColliders"' in source
    assert "GetKinematicEnabledAttr().Set(False)" in source
    assert "subscribe_contact_report_events" in source
    assert "FilteredPairsAPI" in source
    assert "set_kinematic_targets" in source
    assert "AUTHORED_COLLIDER_GEOMETRY_OVERLAY" in source
    assert "set_clipping_range" in source
    assert '"normal"' in source
    assert '"physics_collider_overlay"' in source
    assert "same_physics_frame" in source
    assert ".Save(" not in source
    assert "SurfaceGripper" not in source
    assert "FixedJoint.Define" not in source


def test_finger_registration_driver_audits_both_handed_supplier_cad_fingers() -> None:
    source = FINGER_AUDIT_DRIVER.read_text(encoding="utf-8")

    assert "diagnostic_supplier_cad_left_finger" in source
    assert "diagnostic_supplier_cad_right_finger" in source
    assert "_collision_inventory" in source
    assert '"all_collision_prims"' in source
    assert '"additional_enabled_collision_prims"' in source
    assert '"named_supplier_cad_is_only_enabled_collider"' in source
    assert "UsdPhysics.CollisionGroup" in source
    assert "UsdPhysics.FilteredPairsAPI" in source
    assert "visualizationDisplayColliders" in source
    assert "AUTHORED_COLLIDER_GEOMETRY_NOT_COOKED_HULL_READBACK" in source
    assert ".Save(" not in source


def test_follower_finger_runtime_gate_requires_bilateral_physical_contact() -> None:
    probe = {
        "frozen_inputs_verified": True,
        "finger_colliders": {
            "left": {
                "enabled": True,
                "approximation": "convexHull",
                "maximum_registration_gap_m": 0.0,
            },
            "right": {
                "enabled": True,
                "approximation": "convexHull",
                "maximum_registration_gap_m": 0.0,
            },
        },
        "filtered_pair_with_bottle": False,
        "contacts": {
            "left": [
                {
                    "physical": True,
                    "impulse_ns": 0.001,
                    "separation_m": -0.00001,
                    "collider0_path": "/World/left_finger/collider",
                    "collider1_path": "/World/Bottle500/collider",
                }
            ],
            "right": [
                {
                    "physical": True,
                    "impulse_ns": 0.001,
                    "separation_m": -0.00001,
                    "collider0_path": "/World/right_finger/collider",
                    "collider1_path": "/World/Bottle500/collider",
                }
            ],
        },
        "bottle_response": {
            "maximum_displacement_m": 0.002,
            "minimum_required_displacement_m": 0.0001,
        },
        "captures": {
            "required_phases": [
                "open_pregrasp",
                "bilateral_contact",
                "maximum_closure",
                "hold_end",
            ],
            "paired_records": [
                {
                    "phase": phase,
                    "normal_path": f"/tmp/{phase}_normal.png",
                    "overlay_path": f"/tmp/{phase}_overlay.png",
                    "same_camera_pose": True,
                    "same_physics_frame": True,
                }
                for phase in (
                    "open_pregrasp",
                    "bilateral_contact",
                    "maximum_closure",
                    "hold_end",
                )
            ],
        },
        "forbidden_helpers_absent": True,
        "maximum_registration_gap_m": 1.0e-9,
    }

    passing = evaluate_follower_finger_collision_probe(probe)
    missing_right = copy.deepcopy(probe)
    missing_right["contacts"]["right"] = []
    failing = evaluate_follower_finger_collision_probe(missing_right)

    assert passing["status"] == "PASS"
    assert passing["classification"] == "FINGER_COLLISION_PIPELINE_VERIFIED"
    assert failing["status"] == "FAIL"
    assert failing["classification"] == "BILATERAL_FINGER_CONTACT_NOT_ESTABLISHED"


def test_finger_runtime_driver_pairs_normal_and_collider_overlay_captures() -> None:
    source = FINGER_RUNTIME_DRIVER.read_text(encoding="utf-8")

    assert "set_solve_articulation_contact_last(True)" in source
    assert "visualizationDisplayColliders" in source
    assert '"normal"' in source
    assert '"physics_collider_overlay"' in source
    assert '"left_contact_oblique"' in source
    assert '"right_contact_oblique"' in source
    assert "camera.get_intrinsics_matrix()" in source
    assert '"contact_evidence"' in source
    assert "set_kinematic_targets" not in source
    assert "SurfaceGripper" not in source
    assert "FixedJoint.Define" not in source
    assert ".Save(" not in source


def test_finger_collision_annotations_require_overlay_and_explicit_vision_review() -> None:
    source = FINGER_ANNOTATION_DRIVER.read_text(encoding="utf-8")

    assert "physics_collider_overlay" in source
    assert "displayColliders = 2" in source
    assert "contact_evidence" in source
    assert "camera_intrinsics_pixels" in source
    assert "PENDING_VISION_MODEL_REVIEW" in source
    assert "--vision-reviewed" in source
    assert "collision pipeline only" in source


def test_collision_diagnosis_finalizer_preserves_grasp_acceptance_boundary() -> None:
    source = COLLISION_DIAGNOSIS_FINALIZER.read_text(encoding="utf-8")

    assert "MISSING_BOTTLE_COLLIDER_FALSIFIED" in source
    assert "BOTTLE_AND_FINGER_COLLISION_PIPELINES_VERIFIED" in source
    assert "STATIC_GRASP_NOT_YET_REVALIDATED" in source
    assert "FIVE_RANDOM_POSITION_TRIALS_NOT_RUN" in source
    assert "Task 8" in source

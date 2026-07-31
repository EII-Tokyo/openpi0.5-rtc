from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

TOOL = Path("tools/run_aloha1_grasp_editor_variant_b_gui.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_editor_variant_b_gui",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeArticulation:
    dof_names = (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    )


class _FakeJointState:
    def __init__(self) -> None:
        self.active: list[tuple[str, float, float, float, float]] = []
        self.fixed: list[tuple[str, float]] = []

    def set_active_dof(
        self,
        articulation: object,
        dof_name: str,
        *,
        open_position: float,
        close_position: float,
        max_speed: float,
        max_effort: float,
    ) -> None:
        assert articulation is not None
        self.active.append(
            (
                dof_name,
                open_position,
                close_position,
                max_speed,
                max_effort,
            )
        )

    def set_fixed_dof(
        self,
        articulation: object,
        dof_name: str,
        *,
        fixed_position: float,
    ) -> None:
        assert articulation is not None
        self.fixed.append((dof_name, fixed_position))


class _FakeDropDown:
    def __init__(self, items: list[str]) -> None:
        self._items = list(items)
        self._selection = self._items[0] if self._items else None

    def get_items(self) -> list[str]:
        return list(self._items)

    def set_populate_fn(
        self,
        populate_fn,
        *,
        repopulate: bool = True,
    ) -> None:
        if repopulate:
            self._items = list(populate_fn())
            self._selection = self._items[0] if self._items else None

    def set_selection(self, selection: str) -> None:
        if selection in self._items:
            self._selection = selection

    def get_selection(self) -> str | None:
        return self._selection


def test_variant_b_contract_uses_aligned_stage_and_native_single_active_joint() -> None:
    module = _load_module()

    assert module.STAGE_PATH.resolve() == Path(
        "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/"
        "aloha1_table_support_aligned_workcell.usda"
    ).resolve()
    assert (
        module.EXPECTED_STAGE_SHA256
        == "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
    )
    assert module.EXPECTED_GRASP_CANDIDATE_SHA256 == (
        "b3307c86a44101eadd6ed2151722e7668bb7d644422378765d98eac906835cca"
    )
    assert module.ARTICULATION_PATH.endswith(
        "/follower_left/vx300s_left/root_joint"
    )
    assert module.GRIPPER_LINK_FRAME_PATH.endswith(
        "/follower_left_gripper_link"
    )
    assert module.GRIPPER_FRAME_PATH.endswith(
        "/follower_left_ee_gripper_link"
    )
    assert module.GRIPPER_FRAME_PATH.endswith(
        "/follower_left_ee_gripper_link"
    )
    assert module.GRASP_FRAME_PATH.endswith(
        "/follower_left_ee_gripper_link/"
        "aloha1_supplier_cad_clearance_grasp_frame"
    )
    assert module.VARIANT_B["active_joint"] == "left_finger"
    assert module.VARIANT_B["observer_joint"] == "right_finger"
    assert module.VARIANT_B["open_position_m"] == pytest.approx(0.057)
    assert module.VARIANT_B["fully_closed_position_m"] == pytest.approx(
        0.021
    )
    assert module.VARIANT_B["clearance_contact_position_m"] == pytest.approx(
        0.048316874538855845
    )
    assert module.VARIANT_B["observer_open_position_m"] == pytest.approx(
        -0.057
    )
    assert module.VARIANT_B[
        "observer_clearance_contact_position_m"
    ] == pytest.approx(
        -0.048316874538855845
    )
    assert pytest.approx(0.001) == module.MIMIC_ERROR_TOLERANCE_M
    assert module.ISAAC_WORKSPACE_INDEX == 1


def test_grasp_editor_authors_at_validated_collision_free_home_pose() -> None:
    module = _load_module()

    result = module.select_grasp_editor_authoring_pose()

    assert result == {
        "arm_q_rad": [0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
        "classification": (
            "COLLISION_FREE_ROBOT_LOCAL_AUTHORING_POSE_NOT_TASK_IK_TARGET"
        ),
        "source": "TASK7A_VALIDATED_HOME_ARM",
    }
    assert result["arm_q_rad"] != pytest.approx(module.EPISODE18_LIFT_ONSET_ARM_Q_RAD)


def test_mimic_settle_trace_distinguishes_lag_from_persistent_error() -> None:
    module = _load_module()

    lag = module.classify_mimic_settle_trace(
        [
            {"frame": 0, "residual_abs_m": 0.0017},
            {"frame": 12, "residual_abs_m": 0.0008},
        ],
        tolerance_m=0.001,
    )
    persistent = module.classify_mimic_settle_trace(
        [
            {"frame": 0, "residual_abs_m": 0.0017},
            {"frame": 120, "residual_abs_m": 0.0016},
        ],
        tolerance_m=0.001,
    )

    assert lag["classification"] == "SETTLES_WITHIN_TOLERANCE"
    assert lag["first_passing_frame"] == 12
    assert persistent["classification"] == "PERSISTENT_STEADY_STATE_ERROR"
    assert persistent["first_passing_frame"] is None


def test_mimic_load_case_changes_only_bottle_world_translation() -> None:
    module = _load_module()

    contact = module.select_mimic_load_case("bottle_contact")
    no_contact = module.select_mimic_load_case("no_object_contact")

    assert contact == {
        "name": "bottle_contact",
        "bottle_translation_delta_world_m": [0.0, 0.0, 0.0],
        "expected_bottle_contact": True,
        "native_export_policy": "VALIDATE_NATIVE_EXPORT",
    }
    assert no_contact == {
        "name": "no_object_contact",
        "bottle_translation_delta_world_m": [0.0, 1.5, 0.0],
        "expected_bottle_contact": False,
        "native_export_policy": "SKIP_DIAGNOSTIC_RELATIVE_POSE_CHANGED",
    }
    with pytest.raises(ValueError, match="unsupported mimic load case"):
        module.select_mimic_load_case("unknown")


def test_mimic_load_comparison_classifies_contact_only_error() -> None:
    module = _load_module()

    result = module.classify_mimic_load_comparison(
        contact_residual_m=0.0017,
        no_contact_residual_m=0.0002,
        tolerance_m=0.001,
    )

    assert result == {
        "status": "PASS",
        "classification": "LOAD_INDUCED_COMPLIANT_MIMIC_ERROR",
        "contact_residual_m": 0.0017,
        "no_contact_residual_m": 0.0002,
        "tolerance_m": 0.001,
        "amplification_ratio": 8.5,
    }


def test_mimic_load_comparison_detects_contact_amplification() -> None:
    module = _load_module()

    result = module.classify_mimic_load_comparison(
        contact_residual_m=0.0208,
        no_contact_residual_m=0.00145,
        tolerance_m=0.001,
    )

    assert result["status"] == "FAIL"
    assert result["classification"] == (
        "OBJECT_CONTACT_AMPLIFIES_PERSISTENT_MIMIC_ERROR"
    )
    assert result["amplification_ratio"] == pytest.approx(
        0.0208 / 0.00145
    )


def test_no_contact_diagnostic_records_native_false_positive_without_hiding_data() -> None:
    module = _load_module()

    passed = module.classify_no_contact_diagnostic_result(
        native_success=False,
        mimic_error_abs_m=0.0002,
        physical_bottle_contact_count=0,
    )
    false_positive = module.classify_no_contact_diagnostic_result(
        native_success=True,
        mimic_error_abs_m=0.0014,
        physical_bottle_contact_count=0,
    )

    assert passed["status"] == "PASS"
    assert passed["native_simulate"] == "EXPECTED_FAIL_NO_OBJECT_CONTACT"
    assert false_positive["status"] == "PASS"
    assert false_positive["native_simulate"] == (
        "FALSE_POSITIVE_NO_OBJECT_CONTACT"
    )
    assert false_positive["mimic_accuracy"] == "FAIL"
    assert false_positive["warnings"] == [
        "NATIVE_FALSE_POSITIVE_WITHOUT_PHYSICAL_CONTACT"
    ]


def test_external_close_targets_are_monotonic_and_stop_at_cad_contact_candidate() -> None:
    module = _load_module()

    targets = module.build_external_close_targets(
        open_position_m=0.057,
        contact_target_m=0.048316874538855845,
        speed_m_s=0.02,
        physics_dt_s=1.0 / 60.0,
    )

    assert targets
    assert targets[-1] == pytest.approx(0.048316874538855845)
    assert all(left > right for left, right in pairwise(targets))
    assert max(
        abs(left - right) for left, right in pairwise([0.057, *targets])
    ) <= 0.02 / 60.0 + 1e-12


def test_skip_sim_derived_yaml_restores_open_pregrasp_only() -> None:
    module = _load_module()
    raw = {
        "format": "isaac_grasp",
        "format_version": 1.0,
        "object_frame": "/World/Bottle",
        "gripper_frame": "/World/EE",
        "grasps": {
            "grasp_0": {
                "confidence": 1.0,
                "position": [1.0, 2.0, 3.0],
                "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
                "cspace_position": {"left_finger": 0.05},
                "pregrasp_cspace_position": {"left_finger": 0.05},
            }
        },
    }

    derived = module.derive_skip_sim_yaml_document(
        raw,
        open_position_m=0.057,
    )

    assert raw["grasps"]["grasp_0"]["pregrasp_cspace_position"] == {
        "left_finger": 0.05
    }
    assert derived["grasps"]["grasp_0"]["cspace_position"] == {
        "left_finger": 0.05
    }
    assert derived["grasps"]["grasp_0"]["pregrasp_cspace_position"] == {
        "left_finger": 0.057
    }
    assert derived["grasps"]["grasp_0"]["position"] == [1.0, 2.0, 3.0]


def test_external_skip_sim_gate_requires_bilateral_contact_and_mimic_accuracy() -> None:
    module = _load_module()

    passed = module.classify_external_skip_sim_result(
        mimic_error_abs_m=0.0008,
        contact_summary_status="PASS",
        raw_export_status="PASS",
        derived_export_status="PASS",
    )
    failed = module.classify_external_skip_sim_result(
        mimic_error_abs_m=0.0014,
        contact_summary_status="FAIL",
        raw_export_status="PASS",
        derived_export_status="PASS",
    )

    assert passed == {
        "status": "PASS",
        "execution_mode": "EXTERNAL_CONTACT_SKIP_SIM",
        "bilateral_contact": "PASS",
        "mimic_accuracy": "PASS",
        "raw_export": "PASS",
        "derived_export": "PASS",
        "failure_reasons": [],
    }
    assert failed["status"] == "FAIL"
    assert failed["failure_reasons"] == [
        "BILATERAL_PHYSICAL_CONTACT_FAILED",
        "MIMIC_ACCURACY_FAILED",
    ]


def test_grasp_frame_uses_frozen_complete_gripper_clearance_frame() -> None:
    module = _load_module()

    result = module.compute_supplier_cad_grasp_frame_definition()

    assert result["reference_path"] == module.GRIPPER_LINK_FRAME_PATH
    assert result["parent_path"] == module.GRIPPER_FRAME_PATH
    assert result["prim_path"] == module.GRASP_FRAME_PATH
    assert result["translation_reference_m"] == pytest.approx(
        [0.13552080444282988, 0.0, 0.0],
        abs=5e-12,
    )
    assert result["translation_parent_m"] == pytest.approx(
        [0.02832080444282988, 0.0, 0.0],
        abs=5e-12,
    )
    assert result["rotation_parent_xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert result["approach_axis_grasp"] == [1.0, 0.0, 0.0]
    assert result["finger_line_axis_grasp"] == [0.0, 1.0, 0.0]
    assert result["ee_endpoint_is_grasp_center"] is False
    assert result["grasp_center_offset_from_reference_m"] == pytest.approx(
        0.13552080444282988,
        abs=5e-12,
    )
    assert result["source_classification"] == (
        "FROZEN_SUPPLIER_CAD_COMPLETE_GRIPPER_CLEARANCE_FRAME"
    )
    assert result["static_clearance_gate"]["status"] == "PASS"
    assert result["screenshot_gate"]["status"] == "PASS"
    assert result["screenshot_gate"]["user_confirmed"] is True
    assert result["whole_pad_face_centroid_use"] == "REJECTED"
    assert result["bottle_axis_center_from_grasp_m"] == pytest.approx(
        [-0.003365816517218456, 0.0, 0.0],
        abs=1e-12,
    )


def test_global_closest_collider_midpoint_is_rejected_as_grasp_origin() -> None:
    module = _load_module()

    result = module.compute_supplier_cad_grasp_frame_definition()
    rejected = result["rejected_global_closest_collider_midpoint"]

    assert rejected["translation_reference_m"] == pytest.approx(
        [0.051750762479802675, -0.0001118751981699773, 0.0050000596164240105]
    )
    assert rejected["status"] == "REJECTED_NOT_EFFECTIVE_FINGERTIP_CONTACT_REGION"
    assert rejected["x_before_fingers_link_origin_m"] == pytest.approx(
        0.01694923752019733
    )


def test_grasp_frame_runtime_readback_rejects_ee_identity_alias() -> None:
    module = _load_module()
    definition = module.compute_supplier_cad_grasp_frame_definition()

    with pytest.raises(RuntimeError, match="translation"):
        module.validate_grasp_frame_runtime_readback(
            definition=definition,
            local_translation_m=[0.0, 0.0, 0.0],
            local_rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            source_prim_path=module.GRASP_FRAME_PATH,
        )


def test_grasp_frame_runtime_accepts_session_supplier_cad_pad_center() -> None:
    module = _load_module()
    definition = module.compute_supplier_cad_grasp_frame_definition()

    result = module.validate_grasp_frame_runtime_readback(
        definition=definition,
        local_translation_m=definition["translation_reference_m"],
        local_rotation_xyzw=definition["rotation_parent_xyzw"],
        source_prim_path=module.GRASP_FRAME_PATH,
    )

    assert result["status"] == "PASS"
    assert result["classification"] == (
        "COMPOSED_SESSION_SUPPLIER_CAD_CLEARANCE_GRASP_FRAME"
    )
    assert result["source_stage_modified"] is False
    assert result["session_layer_authored"] is True
    assert result["translation_error_m"] < 1e-12
    assert result["rotation_error_abs"] < 1e-12


def test_grasp_frame_runtime_readback_accepts_isaac_float_noise_only() -> None:
    module = _load_module()
    definition = module.compute_supplier_cad_grasp_frame_definition()

    result = module.validate_grasp_frame_runtime_readback(
        definition=definition,
        local_translation_m=[0.13552083753532865, 0.0, 0.0],
        local_rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        source_prim_path=module.GRASP_FRAME_PATH,
    )

    assert result["status"] == "PASS"
    assert result["translation_tolerance_m"] == pytest.approx(1e-6)
    assert result["translation_error_m"] == pytest.approx(
        3.3092498769038414e-8
    )


def test_runtime_authors_pad_center_only_in_session_layer() -> None:
    source = TOOL.read_text(encoding="utf-8")

    pose_marker = "world_gripper_position, world_gripper_quaternion = get_world_pose("
    dropdown_marker = "desired_frame_path=GRIPPER_FRAME_PATH,"

    assert "GRIPPER_FRAME_PATH" in source
    assert "author_session_supplier_cad_grasp_frame(" in source
    assert "stage.GetSessionLayer()" in source
    assert "session_layer.subLayerPaths.append(diagnostic_layer_identifier)" in source
    assert "stage.SetEditTarget(diagnostic_layer)" in source
    assert "validate_existing_grasp_frame_runtime(" in source
    assert "GRIPPER_FRAME_PATH" in source[source.index(pose_marker) :]
    assert dropdown_marker in source


def test_native_export_rejects_wrist_endpoint_as_gripper_frame(
    tmp_path: Path,
) -> None:
    module = _load_module()
    export_path = tmp_path / "native.yaml"
    document = {
        "format": "isaac_grasp",
        "format_version": 1.0,
        "object_frame": module.BOTTLE_SESSION_PATH,
        "gripper_frame": module.GRIPPER_LINK_FRAME_PATH,
        "grasps": {
            "grasp_0": {
                "confidence": 1.0,
                "position": [0.0, 0.0, 0.069],
                "orientation": {
                    "w": 1.0,
                    "xyz": [0.0, 0.0, 0.0],
                },
                "cspace_position": {"left_finger": 0.03},
                "pregrasp_cspace_position": {"left_finger": 0.057},
            }
        },
    }
    export_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="gripper frame"):
        module._validate_native_export(  # noqa: SLF001
            export_path,
            yaml=yaml,
        )


def test_apply_variant_b_keeps_right_finger_as_fixed_observer() -> None:
    module = _load_module()
    state = _FakeJointState()
    articulation = _FakeArticulation()
    positions = np.asarray(
        [-0.1, 0.5, 0.0, -0.3, 0.9, -0.1, 0.0, 0.057, -0.057]
    )

    result = module.apply_variant_b_joint_settings(
        state,
        articulation,
        positions,
    )

    assert state.active == [
        ("left_finger", 0.057, 0.021, 0.02, 5.0)
    ]
    assert ("right_finger", -0.057) in state.fixed
    assert len(state.fixed) == 8
    assert result["active_joints"] == ["left_finger"]
    assert result["observer_joints"] == ["right_finger"]
    assert result["native_export_joint_policy"] == (
        "GRASP_EDITOR_NATIVE_VARIANT_B_LEFT_ONLY"
    )


def test_world_from_object_closes_object_from_gripper_chain() -> None:
    module = _load_module()
    world_from_gripper = np.eye(4)
    world_from_gripper[:3, 3] = [0.4, -0.1, 0.2]
    object_from_gripper = np.eye(4)
    object_from_gripper[:3, 3] = [-0.05, 0.0, 0.074]

    world_from_object = module.compute_world_from_object(
        world_from_gripper,
        object_from_gripper,
    )

    assert np.allclose(
        world_from_object @ object_from_gripper,
        world_from_gripper,
        atol=1e-12,
    )
    assert np.linalg.det(world_from_object[:3, :3]) == pytest.approx(1.0)


def test_kinematic_contact_reference_requires_readback_and_pose_invariance() -> None:
    module = _load_module()

    result = module.validate_object_authoring_mode(
        requested_mode="kinematic_contact_reference",
        kinematic_readbacks=[True, True, True],
        translation_drift_m=2e-8,
        rotation_drift_rad=3e-8,
        target_translation_residual_m=4e-8,
        target_rotation_residual_rad=5e-8,
        fixed_joint_used=False,
        surface_gripper_used=False,
        parent_attachment_used=False,
    )

    assert result["status"] == "PASS"
    assert result["classification"] == (
        "KINEMATIC_CONTACT_REFERENCE_NOT_DYNAMIC_HOLD"
    )
    assert result["eligible_as_static_hold_evidence"] is False

    with pytest.raises(RuntimeError, match="kinematic readback"):
        module.validate_object_authoring_mode(
            requested_mode="kinematic_contact_reference",
            kinematic_readbacks=[True, False],
            translation_drift_m=0.0,
            rotation_drift_rad=0.0,
            target_translation_residual_m=0.0,
            target_rotation_residual_rad=0.0,
            fixed_joint_used=False,
            surface_gripper_used=False,
            parent_attachment_used=False,
        )

    with pytest.raises(RuntimeError, match="pose drift"):
        module.validate_object_authoring_mode(
            requested_mode="kinematic_contact_reference",
            kinematic_readbacks=[True, True],
            translation_drift_m=2e-4,
            rotation_drift_rad=0.0,
            target_translation_residual_m=0.0,
            target_rotation_residual_rad=0.0,
            fixed_joint_used=False,
            surface_gripper_used=False,
            parent_attachment_used=False,
        )

    with pytest.raises(RuntimeError, match="target pose residual"):
        module.validate_object_authoring_mode(
            requested_mode="kinematic_contact_reference",
            kinematic_readbacks=[True, True],
            translation_drift_m=0.0,
            rotation_drift_rad=0.0,
            target_translation_residual_m=0.2,
            target_rotation_residual_rad=0.0,
            fixed_joint_used=False,
            surface_gripper_used=False,
            parent_attachment_used=False,
        )


def test_runtime_has_explicit_kinematic_authoring_cli_and_session_readback() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert '"--object-authoring-mode"' in source
    assert '"kinematic_contact_reference"' in source
    assert "CreateKinematicEnabledAttr" in source
    assert "GetKinematicEnabledAttr().Get()" in source
    assert "bottle_xformable.ClearXformOpOrder()" in source
    assert "bottle_xformable.AddTransformOp().Set(authored_matrix)" in source
    assert "KINEMATIC_CONTACT_REFERENCE_NOT_DYNAMIC_HOLD" in source
    assert "eligible_as_static_hold_evidence" in source


def test_kinematic_native_export_must_match_frozen_candidate_pose() -> None:
    module = _load_module()
    candidate = np.eye(4)
    candidate[:3, 3] = [-0.025, -0.003, 0.069]
    exported = candidate.copy()
    exported[:3, 3] += [1e-8, -2e-8, 3e-8]

    result = module.validate_native_candidate_pose(
        requested_mode="kinematic_contact_reference",
        exported_object_from_gripper=exported,
        candidate_object_from_gripper=candidate,
        np=np,
    )

    assert result["status"] == "PASS"
    assert result["translation_residual_m"] < 1e-6

    exported[:3, 3] += [0.012, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="native export pose"):
        module.validate_native_candidate_pose(
            requested_mode="kinematic_contact_reference",
            exported_object_from_gripper=exported,
            candidate_object_from_gripper=candidate,
            np=np,
        )


def test_object_from_gripper_loader_accepts_native_single_grasp_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "native.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "grasps": {
                    "grasp_0": {
                        "position": [0.1, -0.2, 0.3],
                        "orientation": {
                            "w": 1.0,
                            "xyz": [0.0, 0.0, 0.0],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = module._load_object_from_gripper(  # noqa: SLF001
        path,
        yaml=yaml,
        np=np,
        rotation_type=Rotation,
    )

    assert result[:3, 3].tolist() == pytest.approx([0.1, -0.2, 0.3])
    assert result[:3, :3] == pytest.approx(np.eye(3))


def test_evidence_camera_pose_is_derived_from_runtime_subject_points() -> None:
    module = _load_module()
    subject_points = np.asarray(
        [
            [-0.65, -0.25, 0.00],
            [-0.35, -0.10, 0.45],
            [-0.10, 0.05, 0.18],
        ],
        dtype=float,
    )

    result = module.compute_evidence_camera_pose(subject_points)

    expected_target = (
        subject_points.min(axis=0) + subject_points.max(axis=0)
    ) / 2.0
    assert np.allclose(result["target"], expected_target)
    assert np.linalg.norm(result["eye"] - result["target"]) >= 1.2
    assert result["subject_aabb_min"].tolist() == pytest.approx(
        subject_points.min(axis=0).tolist()
    )
    assert result["subject_aabb_max"].tolist() == pytest.approx(
        subject_points.max(axis=0).tolist()
    )
    assert result["policy"] == "RUNTIME_SUBJECT_AABB_OBLIQUE_FULL_ARM"


def test_closeup_camera_is_derived_from_gripper_and_bottle_extent() -> None:
    module = _load_module()

    result = module.compute_closeup_camera_pose(
        [
            [0.00, -0.04, 0.00],
            [0.00, 0.04, 0.00],
            [0.03, 0.00, 0.00],
        ]
    )

    assert result["policy"] == "RUNTIME_GRIPPER_BOTTLE_AABB_OBLIQUE_CLOSEUP"
    assert result["target"] == pytest.approx([0.015, 0.0, 0.0])
    assert result["distance_m"] == pytest.approx(0.35)
    assert np.linalg.norm(result["eye"] - result["target"]) == pytest.approx(
        result["distance_m"]
    )


def test_runtime_enables_extension_before_importing_extension_module() -> None:
    source = TOOL.read_text(encoding="utf-8")

    enable_marker = "enabled_id, version = _enable_extension_exact("
    import_marker = (
        "from isaacsim.robot_setup.grasp_editor import "
        "ui_builder as ui_module"
    )
    assert source.index(enable_marker) < source.index(import_marker)
    assert "set_desktop_for_window" not in source
    assert '["xdotool", "set_desktop"' not in source
    assert (
        source.index("_verify_isaac_auto_routed_workspace(")
        < source.index('report["result"] = _configure_and_run_native_gui(')
    )
    assert (
        'lambda: hasattr(builder, "_gripper_selection_dropdown")'
        in source
    )
    assert "articulation_action_type(joint_positions=positions)" in source
    assert "articulation.apply_action(positions)" not in source
    assert '"--class",' in source
    assert '"--name", "Isaac Sim"' not in source
    assert "WM_CLASS" in source
    assert "capture_next_frame_swapchain" in source
    assert "wait_async_capture" in source
    assert '["xwd"' not in source
    assert '["convert"' not in source
    assert "set_camera_view(" in source
    assert "get_active_viewport()" in source
    assert '"focus-new-windows", "strict"' in source
    assert "_restore_launch_focus_policy(" in source


def test_workspace_readback_requires_auto_route_without_desktop_switch() -> None:
    module = _load_module()

    result = module.validate_workspace_assignment(
        prelaunch_current_desktop="0",
        isaac_window_desktop="1",
        postlaunch_current_desktop="0",
    )

    assert result["routing"] == "GNOME_AUTO_MOVE_WINDOWS"
    assert result["workspace_human"] == 2
    assert result["current_desktop_unchanged"] is True

    with pytest.raises(RuntimeError, match="auto-route"):
        module.validate_workspace_assignment(
            prelaunch_current_desktop="0",
            isaac_window_desktop="0",
            postlaunch_current_desktop="0",
        )

    with pytest.raises(RuntimeError, match="changed"):
        module.validate_workspace_assignment(
            prelaunch_current_desktop="0",
            isaac_window_desktop="1",
            postlaunch_current_desktop="1",
        )


def test_gripper_frame_dropdown_uses_validated_robot_frame_when_root_is_joint() -> None:
    module = _load_module()
    dropdown = _FakeDropDown(
        [
            "Select A Frame of Reference",
            "/World/follower_left/vx300s_left/root_joint",
        ]
    )

    result = module.configure_gripper_frame_dropdown(
        dropdown=dropdown,
        articulation_prim_path=(
            "/World/follower_left/vx300s_left/root_joint"
        ),
        desired_frame_path=module.GRASP_FRAME_PATH,
        desired_frame_is_valid_xformable=True,
    )

    assert dropdown.get_items() == [module.GRASP_FRAME_PATH]
    assert dropdown.get_selection() == module.GRASP_FRAME_PATH
    assert result["status"] == "PASS"
    assert result["classification"] == (
        "DIAGNOSTIC_GRASP_EDITOR_ARTICULATION_ROOT_JOINT_FRAME_SCOPE"
    )
    assert result["native_items_contained_desired_frame"] is False

    with pytest.raises(RuntimeError, match="valid Xformable"):
        module.configure_gripper_frame_dropdown(
            dropdown=_FakeDropDown([]),
            articulation_prim_path="/World/follower_left/root_joint",
            desired_frame_path=module.GRASP_FRAME_PATH,
            desired_frame_is_valid_xformable=False,
        )


def test_native_success_does_not_hide_mimic_failure() -> None:
    module = _load_module()

    assert module.classify_native_grasp_result(
        native_success=True,
        mimic_error_abs_m=0.0,
        contact_summary_status="PASS",
    ) == {
        "status": "PASS",
        "native_simulate": "PASS",
        "mimic_accuracy": "PASS",
        "contact_geometry": "PASS",
        "failure_reasons": [],
    }
    failed = module.classify_native_grasp_result(
        native_success=True,
        mimic_error_abs_m=0.008,
        contact_summary_status="PASS",
    )
    assert failed["status"] == "FAIL"
    assert failed["native_simulate"] == "PASS"
    assert failed["mimic_accuracy"] == "FAIL"
    assert failed["contact_geometry"] == "PASS"
    assert failed["failure_reasons"] == ["MIMIC_ERROR_EXCEEDS_0.001_M"]


def test_native_success_does_not_hide_internal_contact_failure() -> None:
    module = _load_module()

    failed = module.classify_native_grasp_result(
        native_success=True,
        mimic_error_abs_m=0.0,
        contact_summary_status="FAIL",
    )

    assert failed == {
        "status": "FAIL",
        "native_simulate": "PASS",
        "mimic_accuracy": "PASS",
        "contact_geometry": "FAIL",
        "failure_reasons": ["CONTACT_GEOMETRY_GATE_FAILED"],
    }


def test_mimic_checkpoint_analysis_identifies_first_failing_phase() -> None:
    module = _load_module()

    result = module.analyze_mimic_checkpoints(
        [
            {
                "phase": "SET_POSITION_IMMEDIATE",
                "left_finger_m": 0.057,
                "right_finger_m": -0.057,
            },
            {
                "phase": "PRE_BOTTLE_PHYSICS_UPDATES",
                "left_finger_m": 0.0569,
                "right_finger_m": -0.05695,
            },
            {
                "phase": "POST_BOTTLE_PLACEMENT",
                "left_finger_m": 0.055,
                "right_finger_m": -0.063,
            },
        ]
    )

    assert result["status"] == "FAIL"
    assert result["first_failing_phase"] == "POST_BOTTLE_PLACEMENT"
    assert result["maximum_residual_abs_m"] == pytest.approx(0.008)
    assert result["tolerance_m"] == module.MIMIC_ERROR_TOLERANCE_M


def test_bottle_contact_summary_rejects_internal_gripper_jam() -> None:
    module = _load_module()
    contacts = [
        {
            "collider0_path": "/World/Bottle500/Body",
            "collider1_path": "/World/Robot/left_finger/mesh",
            "separation_m": -0.0001,
            "impulse_ns": 0.01,
            "phase": "NATIVE_SIMULATE",
        },
        {
            "collider0_path": "/World/Bottle500/Body",
            "collider1_path": "/World/Robot/right_finger/mesh",
            "separation_m": -0.0002,
            "impulse_ns": 0.02,
            "phase": "NATIVE_SIMULATE",
        },
        {
            "collider0_path": "/World/Bottle500/Body",
            "collider1_path": "/World/Robot/gripper_bar/mesh",
            "separation_m": -0.001,
            "impulse_ns": 0.03,
            "phase": "NATIVE_SIMULATE",
        },
    ]

    result = module.summarize_bottle_contacts(
        contacts,
        bottle_token="/World/Bottle500",
        left_finger_token="/left_finger/",
        right_finger_token="/right_finger/",
        robot_token="/World/Robot/",
    )

    assert result["bilateral_finger_contact"] is True
    assert result["unexpected_robot_contact"] is True
    assert result["status"] == "FAIL"
    assert result["unexpected_pairs"] == [
        [
            "/World/Bottle500/Body",
            "/World/Robot/gripper_bar/mesh",
        ]
    ]


def test_bottle_contact_summary_can_exclude_pre_simulation_events() -> None:
    module = _load_module()
    contacts = [
        {
            "collider0_path": "/World/Bottle500",
            "collider1_path": "/World/Table",
            "separation_m": -1e-6,
            "impulse_ns": 0.0,
            "phase": "GRASP_EDITOR_INITIALIZATION",
        },
        {
            "collider0_path": "/World/Bottle500",
            "collider1_path": "/World/Robot/left_finger/mesh",
            "separation_m": -1e-4,
            "impulse_ns": 0.01,
            "phase": "NATIVE_SIMULATE",
        },
    ]

    result = module.summarize_bottle_contacts(
        contacts,
        bottle_token="/World/Bottle500",
        left_finger_token="/left_finger/",
        right_finger_token="/right_finger/",
        robot_token="/World/Robot/",
        accepted_phases={"NATIVE_SIMULATE"},
    )

    assert result["physical_bottle_contact_count"] == 1
    assert result["left_finger_contact"] is True
    assert result["right_finger_contact"] is False
    assert result["accepted_phases"] == ["NATIVE_SIMULATE"]

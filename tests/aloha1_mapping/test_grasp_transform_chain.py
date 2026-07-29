from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.isaac_grasp_spec import IsaacGraspFile
from tools.aloha1_mapping.task_frames import rigid_transform
from tools.validate_aloha1_grasp_transform_chain import build_default_validation
from tools.validate_aloha1_grasp_transform_chain import evaluate_transform_chain
from tools.validate_aloha1_grasp_transform_chain import validate_compatibility_gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_FROM_OBJECT = rigid_transform(np.eye(3), [0.0, -0.16, 0.033])
OBJECT_FROM_GRIPPER = rigid_transform(np.eye(3), [0.0, 0.0, 0.15])
TABLE_FROM_BASE = rigid_transform(
    np.eye(3),
    [-0.4695, -0.019, 0.1109000015258789],
)
EE_FROM_GRIPPER = rigid_transform(np.eye(3), [0.0, 0.0, 0.02])
TABLE_FROM_GRIPPER = TABLE_FROM_OBJECT @ OBJECT_FROM_GRIPPER
BASE_FROM_EE = (
    np.linalg.inv(TABLE_FROM_BASE)
    @ TABLE_FROM_GRIPPER
    @ np.linalg.inv(EE_FROM_GRIPPER)
)


def test_transform_chain_requires_grasp_editor_and_runtime_closure() -> None:
    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=BASE_FROM_EE,
        ee_from_gripper=EE_FROM_GRIPPER,
        ee_frame="follower_left_ee",
        gripper_frame="follower_left_gripper",
        length_unit="m",
        max_translation_error_m=1e-6,
        max_rotation_error_rad=1e-6,
    )

    assert result["status"] == "PASS"
    assert result["failed_gates"] == []
    assert result["world_object_gripper_closure"]["translation_m"] < 1e-6
    assert result["base_ee_gripper_closure"]["rotation_rad"] < 1e-6
    assert result["convention"] == "T_A_B maps column vectors from B into A"
    assert result["double_base_transform_applied"] is False
    assert result["length_unit"] == "m"


def test_transform_chain_rejects_assumed_identity_ee_to_gripper() -> None:
    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=BASE_FROM_EE,
        ee_from_gripper=None,
        ee_frame="follower_left_ee",
        gripper_frame="follower_left_gripper",
    )

    assert result["status"] == "FAIL"
    assert "missing_ee_from_gripper" in result["failed_gates"]


def test_identity_ee_to_gripper_is_allowed_only_for_same_frame() -> None:
    same_frame_base_from_ee = (
        np.linalg.inv(TABLE_FROM_BASE) @ TABLE_FROM_GRIPPER
    )

    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=same_frame_base_from_ee,
        ee_from_gripper=None,
        ee_frame="/World/robot/gripper_link",
        gripper_frame="/World/robot/gripper_link",
    )

    assert result["status"] == "PASS"
    assert result["ee_from_gripper_source"] == (
        "IDENTITY_ALLOWED_FRAME_PATHS_IDENTICAL"
    )


def test_duplicate_or_missing_base_transform_fails_closure() -> None:
    duplicated_base_from_ee = (
        np.linalg.inv(TABLE_FROM_BASE) @ BASE_FROM_EE
    )

    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=duplicated_base_from_ee,
        ee_from_gripper=EE_FROM_GRIPPER,
        ee_frame="follower_left_ee",
        gripper_frame="follower_left_gripper",
    )

    assert result["status"] == "FAIL"
    assert "base_ee_gripper_closure" in result["failed_gates"]


def test_transform_chain_rejects_non_meter_units() -> None:
    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=BASE_FROM_EE,
        ee_from_gripper=EE_FROM_GRIPPER,
        ee_frame="follower_left_ee",
        gripper_frame="follower_left_gripper",
        length_unit="mm",
    )

    assert result["status"] == "FAIL"
    assert "length_unit_not_meters" in result["failed_gates"]


def test_compatibility_gate_uses_structural_and_probe_fields() -> None:
    result = validate_compatibility_gate(
        {
            "status": "PARTIAL",
            "classification": "INCONCLUSIVE",
            "structural_api_classification": (
                "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
            ),
            "structural_api_probe_status": "PASS",
            "grasp_tester_execution_status": "NOT_RUN",
            "probe": {
                "synthetic_serializer_parse_probe": {
                    "synthetic": True,
                    "uses_grasp_tester_output": False,
                }
            },
        }
    )

    assert result["status"] == "PASS"
    assert result["failed_gates"] == []
    assert result["synthetic_serializer_boundary"] == {
        "synthetic": True,
        "uses_grasp_tester_output": False,
        "classification": "SYNTHETIC_SERIALIZER_ONLY_NOT_GRASP_TESTER",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"),
        ("structural_api_classification", None),
        ("structural_api_probe_status", "PARTIAL"),
    ],
)
def test_compatibility_gate_fails_closed(
    field: str,
    value: object,
) -> None:
    record = {
        "status": "PARTIAL",
        "classification": "INCONCLUSIVE",
        "structural_api_classification": (
            "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
        ),
        "structural_api_probe_status": "PASS",
        "grasp_tester_execution_status": "NOT_RUN",
        "probe": {
            "synthetic_serializer_parse_probe": {
                "synthetic": True,
                "uses_grasp_tester_output": False,
            }
        },
    }
    record[field] = value

    result = validate_compatibility_gate(record)

    assert result["status"] == "FAIL"
    assert field in result["failed_gates"]


def test_default_candidate_is_strict_pre_gui_isaac_grasp(tmp_path: Path) -> None:
    grasp_path = tmp_path / "candidate.isaac_grasp.yaml"
    payload = build_default_validation(
        project_root=PROJECT_ROOT,
        grasp_output_path=grasp_path,
    )
    spec = IsaacGraspFile.load(grasp_path)
    grasp = spec.grasp("horizontal_body_grasp")
    raw = yaml.safe_load(grasp_path.read_text(encoding="utf-8"))

    assert payload["status"] == "PARTIAL"
    assert payload["classification"] == "INCONCLUSIVE"
    assert payload["diagnostic_classification"] == (
        "DIAGNOSTIC_ONLY_PENDING_GRASP_EDITOR_SIMULATION"
    )
    assert payload["transform_chain"]["status"] == "PASS"
    assert payload["pre_ik_geometry"]["status"] == "PASS"
    assert payload["source_stage"]["immutable"] is True
    assert set(raw) == {
        "format",
        "format_version",
        "object_frame",
        "gripper_frame",
        "grasps",
    }
    assert set(raw["grasps"]) == {"horizontal_body_grasp"}
    assert grasp.confidence == 0.0
    assert grasp.cspace_position == {
        "left_finger": 0.021,
        "right_finger": -0.021,
    }
    assert grasp.pregrasp_cspace_position == {
        "left_finger": 0.057,
        "right_finger": -0.057,
    }
    assert np.linalg.det(grasp.object_from_gripper[:3, :3]) == pytest.approx(
        1.0
    )
    assert payload["ee_from_gripper"]["source"] == (
        "IDENTITY_ALLOWED_FRAME_PATHS_IDENTICAL"
    )
    assert payload["grasp_candidate"]["diagnostic_confidence"] == {
        "value": 0.0,
        "semantics": (
            "UNTESTED_CANDIDATE_GRASP_TESTER_NOT_RUN"
        ),
    }
    assert payload["grasp_candidate"]["requested_close_candidate_m"] == {
        "left_finger": 0.021,
        "right_finger": -0.021,
    }
    assert payload["grasp_candidate"]["stable_cspace_position_m"] is None
    assert payload["grasp_candidate"]["stable_cspace_position_status"] == (
        "NOT_ESTABLISHED_GRASP_TESTER_NOT_RUN"
    )
    assert payload["compatibility_gate"]["status"] == "PASS"
    assert payload["contact_region_samples"]["left_gripper_m"] == pytest.approx(
        [0.05175075, 0.03813715, 0.00505137],
        abs=5e-9,
    )
    assert payload["contact_region_samples"]["right_gripper_m"] == pytest.approx(
        [0.05175077, -0.03836090, 0.00494875],
        abs=5e-9,
    )
    assert payload["contact_region_samples"]["frame_recovery"][
        "method"
    ] == "INDEPENDENT_INTERBOTIX_ALOHA_VX300S_POE_FK"
    assert payload["task8"] == "NOT_RUN"

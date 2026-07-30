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
BASE_FROM_EE = np.linalg.inv(TABLE_FROM_BASE) @ TABLE_FROM_GRIPPER @ np.linalg.inv(EE_FROM_GRIPPER)


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
    same_frame_base_from_ee = np.linalg.inv(TABLE_FROM_BASE) @ TABLE_FROM_GRIPPER

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
    assert result["ee_from_gripper_source"] == ("IDENTITY_ALLOWED_FRAME_PATHS_IDENTICAL")


def test_duplicate_or_missing_base_transform_fails_closure() -> None:
    duplicated_base_from_ee = np.linalg.inv(TABLE_FROM_BASE) @ BASE_FROM_EE

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
            "structural_api_classification": ("FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"),
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
        "structural_api_classification": ("FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"),
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
    assert payload["diagnostic_classification"] == ("DIAGNOSTIC_ONLY_PENDING_GRASP_EDITOR_SIMULATION")
    assert payload["transform_chain"]["status"] == "PASS"
    assert payload["pre_ik_geometry"]["status"] == "PASS"
    assert payload["pre_ik_geometry"]["failed_gates"] == []
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
        "left_finger": 0.048316874538855845,
    }
    assert grasp.pregrasp_cspace_position == {"left_finger": 0.057}
    assert np.linalg.det(grasp.object_from_gripper[:3, :3]) == pytest.approx(1.0)
    assert spec.gripper_frame.endswith("/follower_left_ee_gripper_link")
    assert grasp.object_from_gripper[:3, 3] == pytest.approx(
        [-0.02473675376396886, -0.00329308906472091, 0.069],
        abs=1e-12,
    )
    object_contract = payload["object_coordinate_contract"]
    assert object_contract["origin"] == "BOTTLE_BOTTOM_CENTER"
    assert object_contract["axis_a_local_m"] == [0.0, 0.0, 0.0]
    assert object_contract["axis_b_local_m"] == [0.0, 0.0, 0.206]
    assert object_contract["grasp_axis_coordinate_m"] == pytest.approx(0.069)
    assert object_contract["origin_redefinition_allowed"] is False
    bottle_axis_center_from_grasp = np.asarray(
        payload["grasp_origin"]["bottle_axis_center_from_grasp_m"],
        dtype=np.float64,
    )
    gripper_from_contact = np.asarray(
        payload["contact_helper_transform"]["gripper_from_contact"],
        dtype=np.float64,
    )
    grasp_center_in_object = (
        grasp.object_from_gripper[:3, :3]
        @ (gripper_from_contact[:3, :3] @ bottle_axis_center_from_grasp + gripper_from_contact[:3, 3])
        + grasp.object_from_gripper[:3, 3]
    )
    assert grasp_center_in_object == pytest.approx(
        [0.0, 0.0, 0.069],
        abs=1e-9,
    )
    assert payload["ee_from_gripper"]["source"] == ("IDENTITY_CANONICAL_EE_AND_GRASP_FRAME")
    assert payload["ee_from_gripper"]["ee_frame"].endswith("/follower_left_ee_gripper_link")
    assert payload["ee_from_gripper"]["gripper_frame"].endswith("/follower_left_ee_gripper_link")
    assert np.asarray(payload["ee_from_gripper"]["matrix"])[:3, 3] == (pytest.approx([0.0, 0.0, 0.0], abs=5e-12))
    contact_helper = payload["contact_helper_transform"]
    assert contact_helper["gripper_frame"].endswith("/follower_left_ee_gripper_link")
    assert contact_helper["contact_frame"].endswith(
        "/follower_left_ee_gripper_link/aloha1_supplier_cad_clearance_grasp_frame"
    )
    assert np.asarray(contact_helper["gripper_from_contact"])[:3, 3] == (
        pytest.approx([0.028320804442829875, 0.0, 0.0], abs=5e-12)
    )
    assert contact_helper["closure"]["translation_m"] < 1e-12
    assert contact_helper["closure"]["rotation_rad"] < 1e-12
    assert payload["grasp_candidate"]["diagnostic_confidence"] == {
        "value": 0.0,
        "semantics": ("UNTESTED_CANDIDATE_GRASP_TESTER_NOT_RUN"),
    }
    assert payload["grasp_candidate"]["requested_close_candidate_m"] == {
        "left_finger": 0.048316874538855845,
    }
    assert payload["grasp_candidate"]["mimic_observer"] == {
        "joint": "right_finger",
        "multiplier": -1.0,
        "offset": 0.0,
        "yaml_active": False,
    }
    assert payload["grasp_candidate"]["stable_cspace_position_m"] is None
    assert payload["grasp_candidate"]["stable_cspace_position_status"] == ("NOT_ESTABLISHED_GRASP_TESTER_NOT_RUN")
    assert payload["compatibility_gate"]["status"] == "PASS"
    assert payload["contact_region_samples"]["left_gripper_m"] == pytest.approx(
        [0.0, 0.03202360503085841, 0.0],
        abs=5e-12,
    )
    assert payload["contact_region_samples"]["right_gripper_m"] == pytest.approx(
        [0.0, -0.03202360503085841, 0.0],
        abs=5e-12,
    )
    assert payload["contact_region_samples"]["disposition"] == ("ACCEPTED_COMPLETE_GRIPPER_CLEARANCE_CONTACT_SOLUTION")
    assert payload["grasp_origin"]["policy"] == ("CHEBYSHEV_COMPLETE_GRIPPER_CLEARANCE_PAD_CONTACT_FRAME")
    assert payload["grasp_origin"]["gripper_from_contact_translation_m"] == (
        pytest.approx([0.028320804442829875, 0.0, 0.0], abs=5e-12)
    )
    assert payload["grasp_origin"]["static_clearance_gate"]["status"] == "PASS"
    assert payload["grasp_origin"]["screenshot_gate"]["status"] == "PASS"
    assert payload["grasp_origin"]["official_ee_helper_semantics"] == ("NOT_GRASP_CENTER")
    assert payload["grasp_origin"]["whole_pad_face_centroid_use"] == "REJECTED"
    assert (
        payload["source_manifest"]["generated_follower_left_urdf"]["sha256"]
        == "d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd"
    )
    assert (
        payload["source_manifest"]["vx300s_srdf"]["sha256"]
        == "39658212772f0432398e61d6b05b3bfbfac059c7ef7e3b12b5df584e9c76493b"
    )
    assert payload["contact_region_samples"]["frame_recovery"]["method"] == "INDEPENDENT_INTERBOTIX_ALOHA_VX300S_POE_FK"
    assert payload["task8"] == "NOT_RUN"

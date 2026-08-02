from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.aloha1_mapping.aloha1_model_math import OFFICIAL_JOINT_ORDER
from tools.aloha1_mapping.aloha1_model_math import is_rigid_transform
from tools.aloha1_mapping.aloha1_model_math import quaternion_wxyz_to_matrix
from tools.derive_aloha1_kinematic_contract import build_contract

ROOT = Path(__file__).resolve().parents[2]
LEFT_URDF = ROOT / "generated/urdf/follower_left.urdf"
RIGHT_URDF = ROOT / "generated/urdf/follower_right.urdf"
SOURCE_MANIFEST = ROOT / "configs/aloha1_official_parameter_sources.yaml"
REPORT = ROOT / "reports/aloha1_mapping/aloha1_kinematic_contract.json"


def test_rigid_transform_validation_rejects_reflection_and_non_orthonormality() -> None:
    identity = np.eye(4)
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    scaled = np.eye(4)
    scaled[1, 1] = 2.0

    assert is_rigid_transform(identity)
    assert not is_rigid_transform(reflection)
    assert not is_rigid_transform(scaled)


def test_quaternion_contract_is_explicitly_wxyz() -> None:
    rotation = quaternion_wxyz_to_matrix([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert np.allclose(rotation, expected, atol=1e-15)


def test_official_joint_order_is_not_alphabetical() -> None:
    assert OFFICIAL_JOINT_ORDER == (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    assert tuple(sorted(OFFICIAL_JOINT_ORDER)) != OFFICIAL_JOINT_ORDER


def test_urdf_fk_matches_independent_official_poe_and_jacobian() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCE_MANIFEST,
    )

    assert contract["status"] == "PASS"
    assert contract["id67_conflict_gate"] == "PASS_RESOLVED_WITH_CONFLICT_RETAINED"
    assert contract["left_right_robot_local_identity"]["status"] == "PASS"
    assert contract["left_right_robot_local_identity"]["mirrored"] is False
    assert contract["fk_comparison"]["max_translation_error_m"] <= contract["tolerances"]["translation_m"]
    assert contract["fk_comparison"]["max_rotation_error_rad"] <= contract["tolerances"]["rotation_rad"]
    assert contract["jacobian_comparison"]["max_abs_error"] <= contract["tolerances"]["jacobian"]
    assert len(contract["samples"]) >= 5

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["deterministic_signature"] == contract["deterministic_signature"]


def test_all_sample_transforms_are_proper_and_finite() -> None:
    contract = build_contract(
        left_urdf=LEFT_URDF,
        right_urdf=RIGHT_URDF,
        source_manifest_path=SOURCE_MANIFEST,
    )

    for sample in contract["samples"]:
        assert np.isfinite(np.asarray(sample["urdf_transform"], dtype=float)).all()
        assert is_rigid_transform(np.asarray(sample["urdf_transform"], dtype=float))
        assert is_rigid_transform(np.asarray(sample["poe_transform"], dtype=float))

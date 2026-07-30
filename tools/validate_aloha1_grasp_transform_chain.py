#!/usr/bin/env python3
"""Validate the pre-IK Bottle500 grasp transform chain.

Matrix convention: ``T_A_B`` maps homogeneous column vectors expressed in
frame ``B`` into frame ``A``.  This tool is deliberately independent of
Isaac runtime and IK.  It generates a pre-GUI diagnostic grasp candidate,
not an accepted Grasp Editor simulation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from tools.aloha1_mapping.aloha_kinematics_reference import SOURCE_FILE
from tools.aloha1_mapping.aloha_kinematics_reference import SOURCE_SHA256
from tools.aloha1_mapping.aloha_kinematics_reference import fk_space
from tools.aloha1_mapping.grasp_frame_contract import convert_contact_pose_to_gripper_pose
from tools.aloha1_mapping.grasp_frame_contract import derive_urdf_fixed_transform
from tools.aloha1_mapping.grasp_pose_geometry import derive_gripper_pose
from tools.aloha1_mapping.grasp_pose_geometry import evaluate_pre_ik_grasp
from tools.aloha1_mapping.isaac_grasp_spec import IsaacGrasp
from tools.aloha1_mapping.isaac_grasp_spec import IsaacGraspFile
from tools.aloha1_mapping.task_frames import closure_error
from tools.aloha1_mapping.task_frames import rigid_transform
from tools.aloha1_mapping.task_frames import validate_rigid_transform

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASP_OUTPUT = ROOT / "configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml"
DEFAULT_REPORT_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_grasp_transform_validation.json"
TASK_FRAME_CONFIG = Path("configs/aloha1_table_task_frame.yaml")
HORIZONTAL_KINEMATICS_REPORT = Path("reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json")
BASELINE_CONFIG = Path("configs/aloha1_stationary_user_confirmed_baseline_v1.yaml")
ORIENTATION_REPORT = Path("reports/aloha1_mapping/gripper_orientation_confirmation.json")
GRASP_EDITOR_REPORT = Path("reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json")
GENERATED_FOLLOWER_LEFT_URDF = Path("generated/urdf/follower_left.urdf")
VX300S_SRDF = Path(
    "external/ros2-essentials/aloha_ws/src/"
    "interbotix_ros_manipulators/interbotix_ros_xsarms/"
    "interbotix_xsarm_moveit/config/srdf/vx300s.srdf.xacro"
)
EXPECTED_STAGE_SHA256 = "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
EXPECTED_BOTTLE_USD_SHA256 = "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
GRIPPER_LINK_FRAME = "/World/follower_left/vx300s_left/follower_left_gripper_link"
GRIPPER_FRAME = "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
CONTACT_FRAME = (
    "/World/follower_left/vx300s_left/follower_left_ee_gripper_link/aloha1_supplier_cad_clearance_grasp_frame"
)
SUPPLIER_CAD_CLEARANCE_REPORT = Path("reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance.json")
EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256 = "9f23974af362dc92134a38633180360bfff8b54bc0a5eaefae8032e2240b91bc"
SUPPLIER_CAD_SCREENSHOT_REVIEW = Path(
    "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance_screenshot_review.json"
)
EXPECTED_SUPPLIER_CAD_SCREENSHOT_REVIEW_SHA256 = "c7097b05654a3966c976690e5f0f79c3b2be69eaa51727f430998efae2bbe0f3"
BOTTLE_OBJECT_FRAME = "/World/ALOHA1GraspEditorSession/Bottle500"
GRASP_NAME = "horizontal_body_grasp"
OPEN_FINGER_STATE_M = {
    "left_finger": 0.057,
}
CLOSED_FINGER_STATE_M = {
    "left_finger": 0.048316874538855845,
}
RIGHT_FINGER_OPEN_M = -OPEN_FINGER_STATE_M["left_finger"]
RIGHT_FINGER_CLOSED_M = -CLOSED_FINGER_STATE_M["left_finger"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_json(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=np.float64)]


def _closure_json(expected: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = closure_error(expected, observed)
    return {
        "translation_m": error.translation_m,
        "rotation_rad": error.rotation_rad,
    }


def _inverse_closure(matrix: np.ndarray) -> dict[str, float]:
    return _closure_json(np.eye(4), matrix @ np.linalg.inv(matrix))


def _passes(
    error: dict[str, float],
    *,
    max_translation_error_m: float,
    max_rotation_error_rad: float,
) -> bool:
    return bool(error["translation_m"] <= max_translation_error_m and error["rotation_rad"] <= max_rotation_error_rad)


def evaluate_transform_chain(
    *,
    table_from_object: np.ndarray,
    object_from_gripper: np.ndarray,
    table_from_base: np.ndarray,
    base_from_ee: np.ndarray,
    ee_from_gripper: np.ndarray | None,
    ee_frame: str | None = None,
    gripper_frame: str | None = None,
    length_unit: str = "m",
    max_translation_error_m: float = 1e-6,
    max_rotation_error_rad: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate both object/gripper and base/EE/gripper paths fail-closed."""

    failed_gates: list[str] = []
    if length_unit != "m":
        failed_gates.append("length_unit_not_meters")

    transforms = {
        "table_from_object": table_from_object,
        "object_from_gripper": object_from_gripper,
        "table_from_base": table_from_base,
        "base_from_ee": base_from_ee,
    }
    validated: dict[str, np.ndarray] = {}
    for name, matrix in transforms.items():
        try:
            validated[name] = validate_rigid_transform(matrix)
        except ValueError:
            failed_gates.append(f"invalid_{name}")

    ee_source = "EXPLICIT_MATRIX"
    if ee_from_gripper is None:
        if isinstance(ee_frame, str) and ee_frame and ee_frame == gripper_frame:
            validated["ee_from_gripper"] = np.eye(4, dtype=np.float64)
            ee_source = "IDENTITY_ALLOWED_FRAME_PATHS_IDENTICAL"
        else:
            failed_gates.append("missing_ee_from_gripper")
    else:
        try:
            validated["ee_from_gripper"] = validate_rigid_transform(ee_from_gripper)
        except ValueError:
            failed_gates.append("invalid_ee_from_gripper")

    required = {
        "table_from_object",
        "object_from_gripper",
        "table_from_base",
        "base_from_ee",
        "ee_from_gripper",
    }
    if not required.issubset(validated):
        return {
            "status": "FAIL",
            "failed_gates": sorted(set(failed_gates)),
            "convention": "T_A_B maps column vectors from B into A",
            "length_unit": length_unit,
            "ee_from_gripper_source": ee_source,
            "double_base_transform_applied": False,
        }

    table_from_gripper = validated["table_from_object"] @ validated["object_from_gripper"]
    base_from_gripper = np.linalg.inv(validated["table_from_base"]) @ table_from_gripper
    base_from_ee_gripper = validated["base_from_ee"] @ validated["ee_from_gripper"]
    table_from_base_ee_gripper = validated["table_from_base"] @ base_from_ee_gripper

    world_object_gripper_closure = _closure_json(
        table_from_gripper,
        validated["table_from_object"] @ validated["object_from_gripper"],
    )
    base_ee_gripper_closure = _closure_json(
        table_from_gripper,
        table_from_base_ee_gripper,
    )
    base_target_closure = _closure_json(
        base_from_gripper,
        base_from_ee_gripper,
    )
    if not _passes(
        world_object_gripper_closure,
        max_translation_error_m=max_translation_error_m,
        max_rotation_error_rad=max_rotation_error_rad,
    ):
        failed_gates.append("world_object_gripper_closure")
    if not _passes(
        base_ee_gripper_closure,
        max_translation_error_m=max_translation_error_m,
        max_rotation_error_rad=max_rotation_error_rad,
    ):
        failed_gates.append("base_ee_gripper_closure")
    if not _passes(
        base_target_closure,
        max_translation_error_m=max_translation_error_m,
        max_rotation_error_rad=max_rotation_error_rad,
    ):
        failed_gates.append("base_target_closure")

    inverse_closures = {
        name: _inverse_closure(matrix)
        for name, matrix in {
            **validated,
            "table_from_gripper": table_from_gripper,
            "base_from_gripper": base_from_gripper,
        }.items()
    }
    for name, error in inverse_closures.items():
        if not _passes(
            error,
            max_translation_error_m=max_translation_error_m,
            max_rotation_error_rad=max_rotation_error_rad,
        ):
            failed_gates.append(f"{name}_inverse_closure")

    matrices = {
        **validated,
        "table_from_gripper": table_from_gripper,
        "base_from_gripper": base_from_gripper,
        "table_from_base_ee_gripper": table_from_base_ee_gripper,
    }
    determinants = {name: float(np.linalg.det(matrix[:3, :3])) for name, matrix in matrices.items()}
    return {
        "status": "PASS" if not failed_gates else "FAIL",
        "failed_gates": sorted(set(failed_gates)),
        "convention": "T_A_B maps column vectors from B into A",
        "multiplication_order": {
            "table_from_gripper": ("table_from_object @ object_from_gripper"),
            "base_from_gripper": ("inverse(table_from_base) @ table_from_gripper"),
            "base_from_ee": ("base_from_gripper @ inverse(ee_from_gripper)"),
        },
        "length_unit": length_unit,
        "ee_frame": ee_frame,
        "gripper_frame": gripper_frame,
        "ee_from_gripper_source": ee_source,
        "world_object_gripper_closure": (world_object_gripper_closure),
        "base_ee_gripper_closure": base_ee_gripper_closure,
        "base_target_closure": base_target_closure,
        "inverse_closures": inverse_closures,
        "determinants": determinants,
        "matrices": {name: _matrix_json(matrix) for name, matrix in matrices.items()},
        "double_base_transform_applied": False,
        "double_base_transform_gate": (
            "PASS_SINGLE_INVERSE_TABLE_FROM_BASE_APPLICATION" if not failed_gates else "FAIL_CHAIN_CLOSURE"
        ),
        "thresholds": {
            "max_translation_error_m": float(max_translation_error_m),
            "max_rotation_error_rad": float(max_rotation_error_rad),
        },
    }


def validate_compatibility_gate(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate the structural/probe boundary without claiming simulation."""

    probe = record.get("probe")
    if not isinstance(probe, dict):
        probe = {}
    serializer = probe.get("synthetic_serializer_parse_probe")
    if not isinstance(serializer, dict):
        serializer = {}
    serializer_is_synthetic = serializer.get("synthetic") is True
    serializer_uses_grasp_tester = serializer.get("uses_grasp_tester_output") is True
    expected = {
        "classification": "INCONCLUSIVE",
        "structural_api_classification": ("FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"),
        "structural_api_probe_status": "PASS",
    }
    failed = [field for field, expected_value in expected.items() if record.get(field) != expected_value]
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_gates": failed,
        "readback": {
            field: record.get(field)
            for field in (
                "status",
                "classification",
                "structural_api_classification",
                "structural_api_probe_status",
                "grasp_tester_execution_status",
                "gui_evidence_status",
                "actual_isaac_grasp_export_status",
                "evidence_scope",
            )
        },
        "synthetic_serializer_boundary": {
            "synthetic": serializer_is_synthetic,
            "uses_grasp_tester_output": serializer_uses_grasp_tester,
            "classification": (
                "SYNTHETIC_SERIALIZER_ONLY_NOT_GRASP_TESTER"
                if (serializer_is_synthetic and not serializer_uses_grasp_tester)
                else "INCONCLUSIVE"
            ),
        },
        "meaning": ("STRUCTURAL_API_AND_NON_EXECUTING_COMPATIBILITY_ONLY"),
        "grasp_tester_execution_claim": False,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return data


def _point(transform: np.ndarray, value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point must be a finite 3-vector")
    return (transform @ np.asarray([*point, 1.0]))[:3]


def _source_record(project_root: Path, relative_path: Path) -> dict[str, Any]:
    path = (project_root / relative_path).resolve(strict=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
    }


def build_default_validation(
    *,
    project_root: Path,
    grasp_output_path: Path,
) -> dict[str, Any]:
    """Generate and validate the frozen pre-GUI diagnostic grasp candidate."""

    root = Path(project_root).resolve(strict=True)
    task_config_path = (root / TASK_FRAME_CONFIG).resolve(strict=True)
    kinematics_path = (root / HORIZONTAL_KINEMATICS_REPORT).resolve(strict=True)
    baseline_path = (root / BASELINE_CONFIG).resolve(strict=True)
    orientation_path = (root / ORIENTATION_REPORT).resolve(strict=True)
    editor_path = (root / GRASP_EDITOR_REPORT).resolve(strict=True)
    task_config = _load_yaml(task_config_path)
    kinematics = _load_json(kinematics_path)
    baseline = _load_yaml(baseline_path)
    orientation = _load_json(orientation_path)
    editor = _load_json(editor_path)

    stage_path = (root / str(task_config["stage"]["path"])).resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    if stage_hash_before != EXPECTED_STAGE_SHA256:
        raise RuntimeError("frozen source Stage SHA-256 mismatch")
    if task_config["stage"]["sha256"] != EXPECTED_STAGE_SHA256:
        raise RuntimeError("task-frame Stage SHA-256 mismatch")
    compatibility_gate = validate_compatibility_gate(editor)
    if compatibility_gate["status"] != "PASS":
        raise RuntimeError(
            f"Grasp Editor structural/probe compatibility gate failed: {compatibility_gate['failed_gates']}"
        )

    bottle_usd_path = (root / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd").resolve(strict=True)
    if _sha256(bottle_usd_path) != EXPECTED_BOTTLE_USD_SHA256:
        raise RuntimeError("Bottle500 USD SHA-256 mismatch")

    table_world_from_task = rigid_transform(
        np.eye(3),
        task_config["task_world"]["world_from_task_translation_m"],
    )
    task_from_world = np.linalg.inv(table_world_from_task)

    placement = kinematics["placement"]
    world_from_object = validate_rigid_transform(placement["placement_matrix"])
    task_from_object = task_from_world @ world_from_object

    lift_onset_frame = int(kinematics["lift_detection"]["lift_onset_frame"])
    left_base = baseline["followers"]["follower_left"]
    world_from_base = rigid_transform(
        Rotation.from_euler(
            "xyz",
            left_base["rotation_rpy_rad"],
        ).as_matrix(),
        left_base["translation_m"],
    )
    runtime_q = np.asarray(
        kinematics["episode_fk"]["lift_onset_runtime_readback_arm_6d"],
        dtype=np.float64,
    )
    base_from_gripper_reference = fk_space(runtime_q)
    base_from_gripper_reference = validate_rigid_transform(base_from_gripper_reference)
    world_from_gripper_reference = world_from_base @ base_from_gripper_reference

    from tools.aloha1_mapping.supplier_cad_grasp_frame import load_verified_clearance_grasp_frame

    clearance_frame = load_verified_clearance_grasp_frame(
        clearance_report_path=root / SUPPLIER_CAD_CLEARANCE_REPORT,
        screenshot_review_path=root / SUPPLIER_CAD_SCREENSHOT_REVIEW,
        expected_clearance_sha256=(EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256),
        expected_screenshot_sha256=(EXPECTED_SUPPLIER_CAD_SCREENSHOT_REVIEW_SHA256),
    )
    gripper_link_from_contact = validate_rigid_transform(clearance_frame["reference_from_grasp"])
    gripper_link_from_gripper = derive_urdf_fixed_transform(
        root / GENERATED_FOLLOWER_LEFT_URDF,
        source_link="follower_left_gripper_link",
        target_link="follower_left_ee_gripper_link",
    )
    gripper_from_contact = validate_rigid_transform(
        np.linalg.inv(gripper_link_from_gripper) @ gripper_link_from_contact
    )
    contact_from_gripper_link = np.linalg.inv(gripper_link_from_contact)
    left_reference = np.asarray(
        clearance_frame["contact_points_reference_m"]["left"],
        dtype=np.float64,
    )
    right_reference = np.asarray(
        clearance_frame["contact_points_reference_m"]["right"],
        dtype=np.float64,
    )
    pad_center_reference = np.asarray(
        clearance_frame["origin_reference_m"],
        dtype=np.float64,
    )
    if not np.allclose(
        gripper_link_from_contact[:3, 3],
        pad_center_reference,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("clearance-frame origin and transform disagree")
    left_gripper = _point(contact_from_gripper_link, left_reference)
    right_gripper = _point(contact_from_gripper_link, right_reference)
    left_open_reference = left_reference + np.asarray(
        [
            0.0,
            OPEN_FINGER_STATE_M["left_finger"] - CLOSED_FINGER_STATE_M["left_finger"],
            0.0,
        ]
    )
    right_open_reference = right_reference + np.asarray(
        [
            0.0,
            RIGHT_FINGER_OPEN_M - RIGHT_FINGER_CLOSED_M,
            0.0,
        ]
    )
    open_pad_center_gap_m = float(np.linalg.norm(right_open_reference - left_open_reference))
    gripper_from_world_reference = np.linalg.inv(world_from_gripper_reference)
    coordinate_frame = orientation["runtime_readback"]["coordinate_frame"]
    if "+X forward" not in coordinate_frame:
        raise RuntimeError("gripper +X approach-axis evidence is missing")
    approach_reference_world = world_from_gripper_reference[:3, :3] @ np.asarray([1.0, 0.0, 0.0])
    approach_reference_to_down_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(
                        np.dot(
                            approach_reference_world,
                            [0.0, 0.0, -1.0],
                        )
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    bottle_axis = placement["bottle_axis"]
    grasp_point_task = _point(
        task_from_world,
        bottle_axis["grasp_point_world_m"],
    )
    task_from_reference_candidate = derive_gripper_pose(
        left_contact_gripper_m=left_gripper,
        right_contact_gripper_m=right_gripper,
        gripper_approach_axis=[1.0, 0.0, 0.0],
        bottle_axis_world=bottle_axis["unit_world"],
        grasp_point_world_m=grasp_point_task,
        table_up_world=[0.0, 0.0, 1.0],
    )
    bottle_axis_center_from_grasp = np.asarray(
        clearance_frame["bottle_axis_center_from_grasp_m"],
        dtype=np.float64,
    )
    task_from_reference_candidate[:3, 3] = (
        grasp_point_task - task_from_reference_candidate[:3, :3] @ bottle_axis_center_from_grasp
    )
    task_from_contact = validate_rigid_transform(task_from_reference_candidate)
    object_from_contact = validate_rigid_transform(np.linalg.inv(task_from_object) @ task_from_contact)
    object_from_gripper = convert_contact_pose_to_gripper_pose(
        object_from_contact=object_from_contact,
        gripper_from_contact=gripper_from_contact,
    )
    task_from_gripper = validate_rigid_transform(task_from_object @ object_from_gripper)
    contact_pose_closure = _closure_json(
        object_from_contact,
        object_from_gripper @ gripper_from_contact,
    )

    grasp_file = IsaacGraspFile(
        object_frame=BOTTLE_OBJECT_FRAME,
        gripper_frame=GRIPPER_FRAME,
        grasps={
            GRASP_NAME: IsaacGrasp(
                name=GRASP_NAME,
                confidence=0.0,
                object_from_gripper=object_from_gripper,
                cspace_position=dict(CLOSED_FINGER_STATE_M),
                pregrasp_cspace_position=dict(OPEN_FINGER_STATE_M),
            )
        },
    )
    grasp_output = Path(grasp_output_path).resolve()
    grasp_file.write(grasp_output)
    reloaded = IsaacGraspFile.load(grasp_output)
    reloaded_grasp = reloaded.grasp(GRASP_NAME)

    left_task = _point(task_from_reference_candidate, left_gripper)
    right_task = _point(task_from_reference_candidate, right_gripper)
    axis_a_task = _point(
        task_from_world,
        bottle_axis["a_world_m"],
    )
    axis_b_task = _point(
        task_from_world,
        bottle_axis["b_world_m"],
    )
    pad_geometry = evaluate_pre_ik_grasp(
        left_contact_world_m=left_task,
        right_contact_world_m=right_task,
        bottle_axis_a_world_m=axis_a_task,
        bottle_axis_b_world_m=axis_b_task,
        expected_axis_coordinate_m=float(bottle_axis["grasp_coordinate_m"]),
        open_aperture_m=open_pad_center_gap_m,
        section_diameter_m=float(placement["bottle_collision_envelope"]["cad_maximum_diameter_m"]),
        table_up_world=[0.0, 0.0, 1.0],
        body_interval_m=[0.018, 0.120],
        axial_tolerance_m=0.005,
        perpendicular_tolerance_deg=3.0,
        contact_envelope_allowance_m=0.0,
    )

    table_from_base = task_from_world @ world_from_base
    base_from_ee = np.linalg.inv(table_from_base) @ task_from_gripper
    transform_chain = evaluate_transform_chain(
        table_from_object=task_from_object,
        object_from_gripper=reloaded_grasp.object_from_gripper,
        table_from_base=table_from_base,
        base_from_ee=base_from_ee,
        ee_from_gripper=np.eye(4, dtype=np.float64),
        ee_frame=GRIPPER_FRAME,
        gripper_frame=GRIPPER_FRAME,
        length_unit="m",
        max_translation_error_m=1e-9,
        max_rotation_error_rad=1e-9,
    )
    if transform_chain["status"] != "PASS":
        raise RuntimeError("generated grasp transform chain failed")
    stage_hash_after = _sha256(stage_path)
    if stage_hash_after != stage_hash_before:
        raise RuntimeError("frozen source Stage changed")

    source_manifest = {
        "task_frame_config": _source_record(root, TASK_FRAME_CONFIG),
        "historical_horizontal_kinematics_geometry": _source_record(
            root,
            HORIZONTAL_KINEMATICS_REPORT,
        ),
        "stationary_baseline": _source_record(root, BASELINE_CONFIG),
        "gripper_orientation_confirmation": _source_record(
            root,
            ORIENTATION_REPORT,
        ),
        "grasp_editor_compatibility": _source_record(
            root,
            GRASP_EDITOR_REPORT,
        ),
        "generated_follower_left_urdf": _source_record(
            root,
            GENERATED_FOLLOWER_LEFT_URDF,
        ),
        "vx300s_srdf": _source_record(root, VX300S_SRDF),
        "interbotix_aloha_vx300s_kinematics": {
            **_source_record(root, Path(SOURCE_FILE)),
            "expected_sha256": SOURCE_SHA256,
            "class": "aloha_vx300s",
        },
        "bottle_usd": {
            "path": str(bottle_usd_path),
            "sha256": _sha256(bottle_usd_path),
        },
        "supplier_cad_complete_gripper_clearance": (clearance_frame["clearance_report"]),
        "supplier_cad_screenshot_user_gate": (clearance_frame["screenshot_gate"]),
    }
    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "classification": "INCONCLUSIVE",
        "diagnostic_classification": ("DIAGNOSTIC_ONLY_PENDING_GRASP_EDITOR_SIMULATION"),
        "scope": ("pre-IK numeric Bottle500 horizontal-body grasp candidate"),
        "source_stage": {
            "path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "expected_sha256": EXPECTED_STAGE_SHA256,
            "immutable": True,
            "root_prim": task_config["usd_world"]["prim_path"],
        },
        "task_frame": {
            "name": task_config["task_world"]["name"],
            "status": task_config["status"],
            "origin_policy": task_config["task_world"]["origin_policy"],
            "tabletop_top_world_m": (task_config["task_world"]["world_from_task_translation_m"]),
            "world_from_task": _matrix_json(table_world_from_task),
        },
        "object_coordinate_contract": {
            "frame": BOTTLE_OBJECT_FRAME,
            "origin": "BOTTLE_BOTTOM_CENTER",
            "axis_a_local_m": [0.0, 0.0, 0.0],
            "axis_b_local_m": [0.0, 0.0, 0.206],
            "local_positive_axis": ("BOTTLE_BOTTOM_TO_MOUTH_LOCAL_POSITIVE_Z"),
            "grasp_axis_coordinate_m": float(bottle_axis["grasp_coordinate_m"]),
            "origin_redefinition_allowed": False,
        },
        "grasp_candidate": {
            "status": "PRE_GUI_DIAGNOSTIC_CANDIDATE",
            "path": str(grasp_output),
            "sha256": _sha256(grasp_output),
            "format": "isaac_grasp",
            "format_version": 1.0,
            "name": GRASP_NAME,
            "object_frame": BOTTLE_OBJECT_FRAME,
            "gripper_frame": GRIPPER_FRAME,
            "diagnostic_confidence": {
                "value": 0.0,
                "semantics": ("UNTESTED_CANDIDATE_GRASP_TESTER_NOT_RUN"),
            },
            "yaml_cspace_position_semantics": ("REQUESTED_CLOSE_CANDIDATE_NOT_STABLE_CSPACE"),
            "requested_close_candidate_m": dict(CLOSED_FINGER_STATE_M),
            "pregrasp_cspace_position_m": dict(OPEN_FINGER_STATE_M),
            "mimic_observer": {
                "joint": "right_finger",
                "multiplier": -1.0,
                "offset": 0.0,
                "yaml_active": False,
            },
            "stable_cspace_position_m": None,
            "stable_cspace_position_status": ("NOT_ESTABLISHED_GRASP_TESTER_NOT_RUN"),
            "authoritative_after_gui_simulation": False,
        },
        "compatibility_gate": compatibility_gate,
        "contact_region_samples": {
            "status": "COMPLETE_GRIPPER_CLEARANCE_CONTACTS_VERIFIED",
            "source_method": ("CHEBYSHEV_COMPLETE_GRIPPER_CLEARANCE_WITH_SUPPLIER_BREP_PAD_NORMAL_OFFSET"),
            "reference_frame": CONTACT_FRAME,
            "disposition": ("ACCEPTED_COMPLETE_GRIPPER_CLEARANCE_CONTACT_SOLUTION"),
            "left_gripper_m": [float(value) for value in left_gripper],
            "right_gripper_m": [float(value) for value in right_gripper],
            "open_aperture_m": open_pad_center_gap_m,
            "left_pad_center_reference_m": [float(value) for value in left_reference],
            "right_pad_center_reference_m": [float(value) for value in right_reference],
            "pad_center_reference_m": [float(value) for value in pad_center_reference],
            "clearance_report": clearance_frame["clearance_report"],
            "screenshot_gate": clearance_frame["screenshot_gate"],
            "bottle_axis_center_from_grasp_m": [float(value) for value in bottle_axis_center_from_grasp],
            "gripper_approach_axis": {
                "axis_gripper": [1.0, 0.0, 0.0],
                "source": ("USER_CONFIRMED_RUNTIME_FRAME_+X_FORWARD_FROM_GRIPPER_ORIENTATION_CONFIRMATION"),
            },
            "frame_recovery": {
                "method": ("INDEPENDENT_INTERBOTIX_ALOHA_VX300S_POE_FK"),
                "runtime_readback_q_rad": [float(value) for value in runtime_q],
                "runtime_frame": lift_onset_frame,
                "base_from_gripper_reference": _matrix_json(base_from_gripper_reference),
                "world_from_base": _matrix_json(world_from_base),
                "world_from_gripper_reference": _matrix_json(world_from_gripper_reference),
                "approach_axis_world_at_reference": [float(value) for value in approach_reference_world],
                "approach_axis_to_world_negative_z_deg": (approach_reference_to_down_deg),
                "source_file": SOURCE_FILE,
                "source_sha256": SOURCE_SHA256,
            },
        },
        "bottle_geometry": {
            "axis_a_task_m": [float(value) for value in axis_a_task],
            "axis_b_task_m": [float(value) for value in axis_b_task],
            "axis_unit_task": [float(value) for value in np.asarray(bottle_axis["unit_world"])],
            "grasp_coordinate_m": float(bottle_axis["grasp_coordinate_m"]),
            "grasp_point_task_m": [float(value) for value in grasp_point_task],
            "body_interval_m": [0.018, 0.120],
            "section_diameter_m": float(placement["bottle_collision_envelope"]["cad_maximum_diameter_m"]),
        },
        "pre_ik_geometry": {
            "status": pad_geometry.status,
            "failed_gates": list(pad_geometry.failed_gates),
            "metrics": pad_geometry.metrics,
            "legacy_global_closest_point_gate": {
                "status": "NOT_EVALUATED",
                "failed_gates": ["global_closest_points_not_effective_pad_surface"],
                "acceptance_use": "REJECTED_HISTORICAL_INPUT",
            },
            "left_contact_task_m": [float(value) for value in left_task],
            "right_contact_task_m": [float(value) for value in right_task],
        },
        "transform_chain": transform_chain,
        "ee_from_gripper": {
            "source": "IDENTITY_CANONICAL_EE_AND_GRASP_FRAME",
            "ee_frame": GRIPPER_FRAME,
            "gripper_frame": GRIPPER_FRAME,
            "matrix": _matrix_json(np.eye(4, dtype=np.float64)),
            "hard_blocker": False,
        },
        "contact_helper_transform": {
            "status": ("DIAGNOSTIC_GEOMETRY_HELPER_NOT_GRASP_EDITOR_OR_IK_FRAME"),
            "gripper_link_frame": GRIPPER_LINK_FRAME,
            "gripper_frame": GRIPPER_FRAME,
            "contact_frame": CONTACT_FRAME,
            "gripper_link_from_gripper": _matrix_json(gripper_link_from_gripper),
            "gripper_link_from_contact": _matrix_json(gripper_link_from_contact),
            "gripper_from_contact": _matrix_json(gripper_from_contact),
            "closure": contact_pose_closure,
            "source": clearance_frame["classification"],
        },
        "grasp_origin": {
            "policy": ("CHEBYSHEV_COMPLETE_GRIPPER_CLEARANCE_PAD_CONTACT_FRAME"),
            "prim_path": CONTACT_FRAME,
            "reference_frame": GRIPPER_FRAME,
            "gripper_from_contact_translation_m": [float(value) for value in gripper_from_contact[:3, 3]],
            "source": (
                "frozen supplier-CAD complete-gripper clearance report plus user-confirmed static screenshot gate"
            ),
            "global_closest_collider_midpoint_use": "REJECTED",
            "whole_pad_face_centroid_use": "REJECTED",
            "official_ee_helper_use": ("PARENT_REFERENCE_ONLY_NOT_PHYSICAL_PAD_CENTER"),
            "official_ee_helper_semantics": "NOT_GRASP_CENTER",
            "static_clearance_gate": (clearance_frame["clearance_report"]),
            "screenshot_gate": clearance_frame["screenshot_gate"],
            "bottle_axis_center_from_grasp_m": [float(value) for value in bottle_axis_center_from_grasp],
        },
        "matrices": {
            "world_from_gripper_reference": _matrix_json(world_from_gripper_reference),
            "gripper_reference_from_world": _matrix_json(gripper_from_world_reference),
            "task_from_object": _matrix_json(task_from_object),
            "task_from_gripper": _matrix_json(task_from_gripper),
            "task_from_contact": _matrix_json(task_from_contact),
            "object_from_contact": _matrix_json(object_from_contact),
            "object_from_gripper": _matrix_json(reloaded_grasp.object_from_gripper),
            "task_from_base": _matrix_json(table_from_base),
            "base_from_ee": _matrix_json(base_from_ee),
            "ee_from_gripper": _matrix_json(np.eye(4, dtype=np.float64)),
            "gripper_from_contact": _matrix_json(gripper_from_contact),
        },
        "evidence_boundaries": {
            "numeric_transform_chain": "PASS",
            "pre_ik_geometry": "PASS",
            "actual_grasp_editor_import_preview_export": "NOT_RUN",
            "grasp_editor_simulation_stability": "NOT_RUN",
            "ik": "NOT_RUN",
            "physics_grasp": "NOT_RUN",
            "real_calibration": "HARD_BLOCKER_NOT_MEASURED",
            "historical_report_use": ("GEOMETRY_INPUT_ONLY_NOT_HOLD_ACCEPTANCE"),
        },
        "source_manifest": source_manifest,
        "source_categories": {
            "runtime_readback": [
                "table/base transforms",
                "reference gripper pose",
                "supplier-CAD collider samples",
            ],
            "numeric_calculation": [
                "task-frame conversion",
                "T_WT_G",
                "T_O_G",
                "T_B_G",
                "closure and determinant gates",
            ],
            "engineering_diagnostic": [
                "5 mm axial tolerance",
                "3 degree perpendicular tolerance",
            ],
            "hard_blocker": ["physical tabletop/base calibration not measured"],
        },
        "task8": "NOT_RUN",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Generate and validate the pre-GUI ALOHA Bottle500 grasp transform candidate.")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--grasp-output",
        type=Path,
        default=DEFAULT_GRASP_OUTPUT,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
    )
    args = parser.parse_args()

    payload = build_default_validation(
        project_root=args.project_root,
        grasp_output_path=args.grasp_output,
    )
    _atomic_json(args.report_output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "transform_chain": payload["transform_chain"]["status"],
                "pre_ik_geometry": payload["pre_ik_geometry"]["status"],
                "grasp_output": payload["grasp_candidate"]["path"],
                "report_output": str(Path(args.report_output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run native Isaac Sim 5.1 Grasp Editor Variant B in an isolated GUI session."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0"
    / "aloha1_table_support_aligned_workcell.usda"
)
EXPECTED_STAGE_SHA256 = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)
BOTTLE_USD = ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
EXPECTED_BOTTLE_SHA256 = (
    "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
)
GRASP_CANDIDATE_PATH = (
    ROOT
    / "configs/aloha1_grasps/"
    "bottle500_horizontal_body_grasp.isaac_grasp.yaml"
)
EXPECTED_GRASP_CANDIDATE_SHA256 = (
    "b3307c86a44101eadd6ed2151722e7668bb7d644422378765d98eac906835cca"
)
ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
ARTICULATION_SELECTION_PREFIX = "/World/follower_left/vx300s_left"
GRIPPER_LINK_FRAME_PATH = (
    "/World/follower_left/vx300s_left/follower_left_gripper_link"
)
GRIPPER_FRAME_PATH = (
    "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
)
GRASP_FRAME_PATH = (
    f"{GRIPPER_FRAME_PATH}/aloha1_supplier_cad_clearance_grasp_frame"
)
SUPPLIER_CAD_MESH_ROOT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_angular_controlled/run_a"
)
SUPPLIER_CAD_BREP_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_frame_brep.json"
)
EXPECTED_SUPPLIER_CAD_BREP_REPORT_SHA256 = (
    "18f026f0d3cf778eab79cc1d04ba05efcdb40316ce82ff50b55cd6dd6ba4d6f5"
)
SUPPLIER_CAD_CLEARANCE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_supplier_cad_grasp_clearance.json"
)
EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256 = (
    "9f23974af362dc92134a38633180360bfff8b54bc0a5eaefae8032e2240b91bc"
)
SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_supplier_cad_grasp_clearance_screenshot_review.json"
)
EXPECTED_SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW_SHA256 = (
    "c7097b05654a3966c976690e5f0f79c3b2be69eaa51727f430998efae2bbe0f3"
)
URDF_EE_ARM_OFFSET_M = 0.042825
URDF_EE_BAR_OFFSET_M = 0.025875
URDF_EE_GRIPPER_OFFSET_M = 0.0385
CAD_LEFT_INNER_SAMPLE_GRIPPER_M = (
    0.05175075359603422,
    0.038137152917932765,
    0.005051372018781507,
)
CAD_RIGHT_INNER_SAMPLE_GRIPPER_M = (
    0.05175077136357113,
    -0.03836090331427272,
    0.004948747214066514,
)
BASE_FRAME_PATH = "/World/follower_left/vx300s_left/follower_left_base_link"
BOTTLE_SESSION_PATH = "/World/ALOHA1GraspEditorSession/Bottle500"
FOLLOWER_LEFT_ROOT_PATH = "/World/follower_left/vx300s_left"
LEFT_FINGER_COLLIDER_PATH = (
    f"{FOLLOWER_LEFT_ROOT_PATH}/follower_left_left_finger_link/collisions/"
    "diagnostic_supplier_cad_left_finger/mesh"
)
RIGHT_FINGER_COLLIDER_PATH = (
    f"{FOLLOWER_LEFT_ROOT_PATH}/follower_left_right_finger_link/collisions/"
    "diagnostic_supplier_cad_right_finger/mesh"
)
GRASP_EDITOR_EXTENSION_ID = "isaacsim.robot_setup.grasp_editor"
GRASP_EDITOR_VERSION = "2.0.20"
WINDOW_TITLE = "Grasp Editor"
ISAAC_WORKSPACE_INDEX = 1
GNOME_AUTO_MOVE_UUID = (
    "auto-move-windows@gnome-shell-extensions.gcampax.github.com"
)
GNOME_DESKTOP_ENTRY_ID = "aloha-isaac-grasp.desktop"
GNOME_DESKTOP_ENTRY_PATH = (
    Path.home() / ".local/share/applications" / GNOME_DESKTOP_ENTRY_ID
)
GNOME_EXTENSION_DIR = (
    Path.home() / ".local/share/gnome-shell/extensions" / GNOME_AUTO_MOVE_UUID
)
EXPECTED_ISAAC_WM_CLASS = "Isaac Sim Python 5.1.0"
EXPECTED_ISAAC_WM_CLASS_REGEX = r"^Isaac Sim Python 5\.1\.0$"
GNOME_WM_SCHEMA = "org.gnome.desktop.wm.preferences"
GNOME_FOCUS_KEY = "focus-new-windows"
# Existing Task 5 engineering gate from gripper_validation.json. This is
# distinct from Grasp Editor's 1e-4 rolling-window stability threshold.
MIMIC_ERROR_TOLERANCE_M = 0.001
MIMIC_SETTLE_OBSERVATION_FRAMES = (0, 1, 2, 4, 8, 12, 30, 60, 120)
GRASP_FRAME_TRANSLATION_TOLERANCE_M = 1e-6
GRASP_FRAME_ROTATION_TOLERANCE_ABS = 1e-6
EPISODE18_LIFT_ONSET_ARM_Q_RAD = (
    -0.16720470786094666,
    0.5324101448059082,
    -0.017540352419018745,
    -0.3624092638492584,
    0.9591664671897888,
    -0.11042828112840652,
)
TASK7A_VALIDATED_HOME_ARM_Q_RAD = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
VARIANT_B: dict[str, Any] = {
    "active_joint": "left_finger",
    "observer_joint": "right_finger",
    "open_position_m": 0.057,
    "fully_closed_position_m": 0.021,
    "clearance_contact_position_m": 0.048316874538855845,
    "observer_open_position_m": -0.057,
    "observer_clearance_contact_position_m": -0.048316874538855845,
    "max_speed_m_s": 0.02,
    "max_effort_n": 5.0,
    "fully_closed_source": (
        "USD_AND_RUNTIME_LEFT_FINGER_LOWER_LIMIT_READBACK"
    ),
    "contact_position_source": (
        "SUPPLIER_CAD_CLEARANCE_GRASP_CANDIDATE"
    ),
}


def build_external_close_targets(
    *,
    open_position_m: float,
    contact_target_m: float,
    speed_m_s: float,
    physics_dt_s: float,
) -> list[float]:
    """Build a deterministic, monotonic close path ending at CAD contact."""
    if not math.isfinite(open_position_m) or not math.isfinite(
        contact_target_m
    ):
        raise ValueError("finger positions must be finite")
    if speed_m_s <= 0.0 or not math.isfinite(speed_m_s):
        raise ValueError("speed_m_s must be finite and positive")
    if physics_dt_s <= 0.0 or not math.isfinite(physics_dt_s):
        raise ValueError("physics_dt_s must be finite and positive")
    if contact_target_m >= open_position_m:
        raise ValueError(
            "contact_target_m must be below open_position_m for left_finger"
        )

    maximum_step_m = speed_m_s * physics_dt_s
    distance_m = open_position_m - contact_target_m
    step_count = max(1, math.ceil(distance_m / maximum_step_m))
    return [
        open_position_m
        - distance_m * float(index) / float(step_count)
        for index in range(1, step_count + 1)
    ]


def derive_skip_sim_yaml_document(
    raw_document: dict[str, Any],
    *,
    open_position_m: float,
) -> dict[str, Any]:
    """Restore only the verified open pregrasp after native Skip Sim export."""
    if not math.isfinite(open_position_m):
        raise ValueError("open_position_m must be finite")
    document = copy.deepcopy(raw_document)
    grasps = document.get("grasps")
    if not isinstance(grasps, dict) or not grasps:
        raise ValueError("native Skip Sim YAML has no grasps")
    for grasp in grasps.values():
        if not isinstance(grasp, dict):
            raise ValueError("native Skip Sim YAML grasp must be a mapping")
        cspace = grasp.get("cspace_position")
        pregrasp = grasp.get("pregrasp_cspace_position")
        if not isinstance(cspace, dict) or not isinstance(pregrasp, dict):
            raise ValueError("native Skip Sim YAML lacks cspace mappings")
        if set(cspace) != {VARIANT_B["active_joint"]}:
            raise ValueError(
                "native Skip Sim YAML must contain only the active finger DOF"
            )
        if set(pregrasp) != set(cspace):
            raise ValueError("pregrasp and cspace joint names differ")
        grasp["pregrasp_cspace_position"] = {
            VARIANT_B["active_joint"]: float(open_position_m)
        }
    return document


def classify_external_skip_sim_result(
    *,
    mimic_error_abs_m: float,
    contact_summary_status: str,
    raw_export_status: str,
    derived_export_status: str,
) -> dict[str, object]:
    """Gate external close + Skip Sim without trusting Skip Sim success alone."""
    bilateral_contact = (
        "PASS" if contact_summary_status == "PASS" else "FAIL"
    )
    mimic_accuracy = (
        "PASS"
        if mimic_error_abs_m <= MIMIC_ERROR_TOLERANCE_M
        else "FAIL"
    )
    failures: list[str] = []
    if bilateral_contact != "PASS":
        failures.append("BILATERAL_PHYSICAL_CONTACT_FAILED")
    if mimic_accuracy != "PASS":
        failures.append("MIMIC_ACCURACY_FAILED")
    if raw_export_status != "PASS":
        failures.append("RAW_NATIVE_SKIP_SIM_EXPORT_FAILED")
    if derived_export_status != "PASS":
        failures.append("DERIVED_PREGRASP_EXPORT_FAILED")
    return {
        "status": "PASS" if not failures else "FAIL",
        "execution_mode": "EXTERNAL_CONTACT_SKIP_SIM",
        "bilateral_contact": bilateral_contact,
        "mimic_accuracy": mimic_accuracy,
        "raw_export": raw_export_status,
        "derived_export": derived_export_status,
        "failure_reasons": failures,
    }


def select_grasp_editor_authoring_pose() -> dict[str, object]:
    """Use the validated collision-free home pose only for relative-grasp authoring."""
    return {
        "arm_q_rad": list(TASK7A_VALIDATED_HOME_ARM_Q_RAD),
        "classification": (
            "COLLISION_FREE_ROBOT_LOCAL_AUTHORING_POSE_NOT_TASK_IK_TARGET"
        ),
        "source": "TASK7A_VALIDATED_HOME_ARM",
    }


def classify_mimic_settle_trace(
    trace: Sequence[dict[str, object]],
    *,
    tolerance_m: float,
) -> dict[str, object]:
    """Classify whether a post-SIMULATE mimic residual is transient or persistent."""
    passing = [
        int(sample["frame"])
        for sample in trace
        if float(sample["residual_abs_m"]) <= tolerance_m
    ]
    return {
        "classification": (
            "SETTLES_WITHIN_TOLERANCE"
            if passing
            else "PERSISTENT_STEADY_STATE_ERROR"
        ),
        "first_passing_frame": passing[0] if passing else None,
        "tolerance_m": tolerance_m,
    }


def select_mimic_load_case(name: str) -> dict[str, object]:
    """Return the one-variable bottle-position diagnostic contract."""
    cases = {
        "bottle_contact": {
            "name": "bottle_contact",
            "bottle_translation_delta_world_m": [0.0, 0.0, 0.0],
            "expected_bottle_contact": True,
            "native_export_policy": "VALIDATE_NATIVE_EXPORT",
        },
        "no_object_contact": {
            "name": "no_object_contact",
            "bottle_translation_delta_world_m": [0.0, 1.5, 0.0],
            "expected_bottle_contact": False,
            "native_export_policy": (
                "SKIP_DIAGNOSTIC_RELATIVE_POSE_CHANGED"
            ),
        },
    }
    if name not in cases:
        raise ValueError(f"unsupported mimic load case: {name}")
    return cases[name]


def classify_mimic_load_comparison(
    *,
    contact_residual_m: float,
    no_contact_residual_m: float,
    tolerance_m: float,
) -> dict[str, object]:
    """Separate contact-load compliance from a persistent mimic error."""
    contact_fails = contact_residual_m > tolerance_m
    no_contact_fails = no_contact_residual_m > tolerance_m
    amplification_ratio = (
        contact_residual_m / no_contact_residual_m
        if no_contact_residual_m > 0.0
        else float("inf")
    )
    if contact_fails and not no_contact_fails:
        classification = "LOAD_INDUCED_COMPLIANT_MIMIC_ERROR"
        status = "PASS"
    elif contact_fails and no_contact_fails:
        classification = (
            "OBJECT_CONTACT_AMPLIFIES_PERSISTENT_MIMIC_ERROR"
            if amplification_ratio >= 2.0
            else "OBJECT_CONTACT_NOT_PRIMARY_PERSISTENT_MIMIC_ERROR"
        )
        status = "FAIL"
    elif not contact_fails and no_contact_fails:
        classification = "INCONSISTENT_NO_CONTACT_ONLY_ERROR"
        status = "FAIL"
    else:
        classification = "MIMIC_WITHIN_TOLERANCE_IN_BOTH_CASES"
        status = "PASS"
    return {
        "status": status,
        "classification": classification,
        "contact_residual_m": contact_residual_m,
        "no_contact_residual_m": no_contact_residual_m,
        "tolerance_m": tolerance_m,
        "amplification_ratio": amplification_ratio,
    }


def classify_no_contact_diagnostic_result(
    *,
    native_success: bool,
    mimic_error_abs_m: float,
    physical_bottle_contact_count: int,
) -> dict[str, object]:
    """Validate the unloaded mimic probe without calling it a grasp."""
    failure_reasons: list[str] = []
    warnings: list[str] = []
    if native_success:
        warnings.append(
            "NATIVE_FALSE_POSITIVE_WITHOUT_PHYSICAL_CONTACT"
        )
    if physical_bottle_contact_count:
        failure_reasons.append("UNEXPECTED_BOTTLE_CONTACT")
    return {
        "status": "PASS" if not failure_reasons else "FAIL",
        "native_simulate": (
            "FALSE_POSITIVE_NO_OBJECT_CONTACT"
            if native_success
            else "EXPECTED_FAIL_NO_OBJECT_CONTACT"
        ),
        "mimic_accuracy": (
            "PASS"
            if mimic_error_abs_m <= MIMIC_ERROR_TOLERANCE_M
            else "FAIL"
        ),
        "contact_geometry": (
            "PASS_EXPECTED_ABSENT"
            if physical_bottle_contact_count == 0
            else "FAIL_UNEXPECTED_PRESENT"
        ),
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "diagnostic_completion_not_grasp_acceptance": True,
    }


def compute_supplier_cad_grasp_frame_definition() -> dict[str, object]:
    """Load the user-approved complete-gripper clearance frame."""
    from tools.aloha1_mapping.supplier_cad_grasp_frame import compare_brep_mesh_pad_evidence
    from tools.aloha1_mapping.supplier_cad_grasp_frame import derive_supplier_cad_grasp_frame
    from tools.aloha1_mapping.supplier_cad_grasp_frame import load_verified_clearance_grasp_frame

    if (
        _sha256(SUPPLIER_CAD_BREP_REPORT)
        != EXPECTED_SUPPLIER_CAD_BREP_REPORT_SHA256
    ):
        raise RuntimeError("supplier-CAD B-Rep grasp report hash mismatch")
    mesh_report = derive_supplier_cad_grasp_frame(
        left_obj_path=SUPPLIER_CAD_MESH_ROOT / "left_finger.obj",
        right_obj_path=SUPPLIER_CAD_MESH_ROOT / "right_finger.obj",
    )
    brep_report = json.loads(
        SUPPLIER_CAD_BREP_REPORT.read_text(encoding="utf-8")
    )
    brep_mesh_comparison = compare_brep_mesh_pad_evidence(
        mesh_report=mesh_report,
        brep_report=brep_report,
    )
    if brep_mesh_comparison["status"] != "PASS":
        raise RuntimeError("B-Rep and mesh pad-center evidence disagree")
    clearance_frame = load_verified_clearance_grasp_frame(
        clearance_report_path=SUPPLIER_CAD_CLEARANCE_REPORT,
        screenshot_review_path=SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW,
        expected_clearance_sha256=(
            EXPECTED_SUPPLIER_CAD_CLEARANCE_REPORT_SHA256
        ),
        expected_screenshot_sha256=(
            EXPECTED_SUPPLIER_CAD_CLEARANCE_SCREENSHOT_REVIEW_SHA256
        ),
    )

    pad_midpoint = list(clearance_frame["origin_reference_m"])
    rejected_center = [
        (left + right) / 2.0
        for left, right in zip(
            CAD_LEFT_INNER_SAMPLE_GRIPPER_M,
            CAD_RIGHT_INNER_SAMPLE_GRIPPER_M,
            strict=True,
        )
    ]
    fingers_link_offset = URDF_EE_ARM_OFFSET_M + URDF_EE_BAR_OFFSET_M
    end_effector_offset = fingers_link_offset + URDF_EE_GRIPPER_OFFSET_M
    return {
        "status": "SUPPLIER_CAD_PAD_CENTER_FRAME_DERIVED",
        "source_classification": clearance_frame["classification"],
        "reference_path": GRIPPER_LINK_FRAME_PATH,
        "parent_path": GRIPPER_FRAME_PATH,
        "prim_path": GRASP_FRAME_PATH,
        "translation_reference_m": pad_midpoint,
        "translation_parent_m": [
            pad_midpoint[0] - end_effector_offset,
            pad_midpoint[1],
            pad_midpoint[2],
        ],
        "rotation_parent_xyzw": [0.0, 0.0, 0.0, 1.0],
        "fixed_chain": [
            {
                "joint": "ee_arm",
                "translation_m": [URDF_EE_ARM_OFFSET_M, 0.0, 0.0],
            },
            {
                "joint": "ee_bar",
                "translation_m": [URDF_EE_BAR_OFFSET_M, 0.0, 0.0],
            },
            {
                "joint": "ee_gripper",
                "translation_m": [URDF_EE_GRIPPER_OFFSET_M, 0.0, 0.0],
            },
        ],
        "approach_axis_grasp": [1.0, 0.0, 0.0],
        "finger_line_axis_grasp": [0.0, 1.0, 0.0],
        "ee_endpoint_is_grasp_center": False,
        "grasp_center_offset_from_reference_m": math.dist(
            pad_midpoint,
            [0.0, 0.0, 0.0],
        ),
        "official_ee_offset_from_reference_m": end_effector_offset,
        "pad_center_offset_from_official_ee_m": (
            math.dist(
                pad_midpoint,
                [end_effector_offset, 0.0, 0.0],
            )
        ),
        "left_pad_center_reference_m": (
            clearance_frame["contact_points_reference_m"]["left"]
        ),
        "right_pad_center_reference_m": (
            clearance_frame["contact_points_reference_m"]["right"]
        ),
        "static_clearance_gate": clearance_frame["clearance_report"],
        "screenshot_gate": clearance_frame["screenshot_gate"],
        "whole_pad_face_centroid_use": (
            clearance_frame["whole_pad_face_centroid_use"]
        ),
        "bottle_axis_center_from_grasp_m": (
            clearance_frame["bottle_axis_center_from_grasp_m"]
        ),
        "brep_mesh_comparison": brep_mesh_comparison,
        "rejected_global_closest_collider_midpoint": {
            "status": "REJECTED_NOT_EFFECTIVE_FINGERTIP_CONTACT_REGION",
            "translation_reference_m": rejected_center,
            "x_before_fingers_link_origin_m": (
                fingers_link_offset - rejected_center[0]
            ),
            "reason": (
                "The global closest collider vertices select an internal or "
                "root region before the fingers_link origin; they do not "
                "define the effective distal grasp frame."
            ),
        },
        "sources": {
            "generated_urdf": (
                "generated/urdf/follower_left.urdf"
            ),
            "srdf": (
                "external/ros2-essentials/aloha_ws/src/"
                "interbotix_ros_manipulators/interbotix_ros_xsarms/"
                "interbotix_xsarm_moveit/config/srdf/vx300s.srdf.xacro"
            ),
            "supplier_cad_brep_report": str(
                SUPPLIER_CAD_BREP_REPORT
            ),
            "supplier_cad_brep_report_sha256": (
                EXPECTED_SUPPLIER_CAD_BREP_REPORT_SHA256
            ),
        },
    }


def author_session_supplier_cad_grasp_frame(
    *,
    stage: Any,
) -> dict[str, object]:
    """Author only the derived fixed pad-center frame in the session sublayer."""
    from pxr import Gf
    from pxr import UsdGeom

    definition = compute_supplier_cad_grasp_frame_definition()
    parent = stage.GetPrimAtPath(GRIPPER_FRAME_PATH)
    if not parent.IsValid() or not UsdGeom.Xformable(parent):
        raise RuntimeError("official EE helper parent is not Xformable")
    edit_target_layer = stage.GetEditTarget().GetLayer()
    if edit_target_layer == stage.GetRootLayer():
        raise RuntimeError("refusing to author CAD grasp frame in root layer")
    xform = UsdGeom.Xform.Define(stage, GRASP_FRAME_PATH)
    xform.AddTranslateOp().Set(
        Gf.Vec3d(*definition["translation_parent_m"])
    )
    prim = xform.GetPrim()
    prim.SetCustomDataByKey(
        "aloha1:classification",
        "SESSION_ONLY_SUPPLIER_CAD_PAD_CENTER_NOT_FINAL_ASSET",
    )
    prim.SetCustomDataByKey(
        "aloha1:source",
        "SUPPLIER_STEP_BREP_PLUS_DETERMINISTIC_MESH",
    )
    return {
        "status": "PASS",
        "prim_path": GRASP_FRAME_PATH,
        "parent_path": GRIPPER_FRAME_PATH,
        "translation_parent_m": definition["translation_parent_m"],
        "edit_target_identifier": edit_target_layer.identifier,
        "session_layer_authored": True,
        "source_stage_modified": False,
    }


def validate_grasp_frame_runtime_readback(
    *,
    definition: dict[str, object],
    local_translation_m: Sequence[float],
    local_rotation_xyzw: Sequence[float],
    source_prim_path: str,
) -> dict[str, object]:
    """Require the composed supplier-CAD pad-center frame."""
    expected_translation = definition["translation_reference_m"]
    expected_rotation = definition["rotation_parent_xyzw"]
    translation_error = math.dist(
        local_translation_m,
        expected_translation,
    )
    rotation_error = min(
        math.dist(local_rotation_xyzw, expected_rotation),
        math.dist(local_rotation_xyzw, [-value for value in expected_rotation]),
    )
    if translation_error > GRASP_FRAME_TRANSLATION_TOLERANCE_M:
        raise RuntimeError(
            "supplier-CAD grasp-frame translation readback mismatch: "
            f"{translation_error} m"
        )
    if rotation_error > GRASP_FRAME_ROTATION_TOLERANCE_ABS:
        raise RuntimeError(
            "supplier-CAD grasp-frame rotation readback mismatch: "
            f"{rotation_error}"
        )
    if source_prim_path != GRASP_FRAME_PATH:
        raise RuntimeError(
            f"unexpected composed grasp frame path: {source_prim_path}"
        )
    return {
        "status": "PASS",
        "classification": (
            "COMPOSED_SESSION_SUPPLIER_CAD_CLEARANCE_GRASP_FRAME"
        ),
        "translation_error_m": translation_error,
        "rotation_error_abs": rotation_error,
        "translation_tolerance_m": (
            GRASP_FRAME_TRANSLATION_TOLERANCE_M
        ),
        "rotation_tolerance_abs": (
            GRASP_FRAME_ROTATION_TOLERANCE_ABS
        ),
        "source_prim_path": source_prim_path,
        "session_layer_authored": True,
        "source_stage_modified": False,
    }


def validate_existing_grasp_frame_runtime(
    *,
    get_world_pose: Any,
    np: Any,
    rotation_type: Any,
) -> dict[str, object]:
    definition = compute_supplier_cad_grasp_frame_definition()
    reference_world = _matrix_from_pose(
        *get_world_pose(GRIPPER_LINK_FRAME_PATH),
        np=np,
        rotation_type=rotation_type,
    )
    grasp_world = _matrix_from_pose(
        *get_world_pose(GRASP_FRAME_PATH),
        np=np,
        rotation_type=rotation_type,
    )
    reference_from_grasp = np.linalg.inv(reference_world) @ grasp_world
    local_translation = reference_from_grasp[:3, 3].tolist()
    local_rotation_xyzw = rotation_type.from_matrix(
        reference_from_grasp[:3, :3]
    ).as_quat().tolist()
    validation = validate_grasp_frame_runtime_readback(
        definition=definition,
        local_translation_m=local_translation,
        local_rotation_xyzw=local_rotation_xyzw,
        source_prim_path=GRASP_FRAME_PATH,
    )
    return {
        **definition,
        **validation,
        "local_translation_readback_m": local_translation,
        "local_rotation_readback_xyzw": local_rotation_xyzw,
        "world_from_reference": reference_world.tolist(),
        "world_from_grasp": grasp_world.tolist(),
        "reference_from_grasp": reference_from_grasp.tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_variant_b_joint_settings(
    joint_state: Any,
    articulation: Any,
    positions: Sequence[float],
) -> dict[str, object]:
    """Apply native Variant B: left active, right retained as observer."""
    if tuple(articulation.dof_names) != (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ):
        raise RuntimeError(
            f"unexpected follower-left DOF order: {articulation.dof_names}"
        )
    if len(positions) != len(articulation.dof_names):
        raise ValueError("joint position count does not match DOF count")

    for index, dof_name in enumerate(articulation.dof_names):
        if dof_name == VARIANT_B["active_joint"]:
            joint_state.set_active_dof(
                articulation,
                dof_name,
                open_position=VARIANT_B["open_position_m"],
                close_position=VARIANT_B["fully_closed_position_m"],
                max_speed=VARIANT_B["max_speed_m_s"],
                max_effort=VARIANT_B["max_effort_n"],
            )
            continue
        fixed_position = float(positions[index])
        if dof_name == VARIANT_B["observer_joint"]:
            fixed_position = float(VARIANT_B["observer_open_position_m"])
        joint_state.set_fixed_dof(
            articulation,
            dof_name,
            fixed_position=fixed_position,
        )
    return {
        "active_joints": [VARIANT_B["active_joint"]],
        "observer_joints": [VARIANT_B["observer_joint"]],
        "native_export_joint_policy": (
            "GRASP_EDITOR_NATIVE_VARIANT_B_LEFT_ONLY"
        ),
    }


def compute_world_from_object(
    world_from_gripper: Any,
    object_from_gripper: Any,
) -> Any:
    """Place the object so its authored grasp closes at the live gripper."""
    import numpy as np

    world_from_gripper_array = np.asarray(
        world_from_gripper,
        dtype=float,
    )
    object_from_gripper_array = np.asarray(
        object_from_gripper,
        dtype=float,
    )
    if (
        world_from_gripper_array.shape != (4, 4)
        or object_from_gripper_array.shape != (4, 4)
        or not np.isfinite(world_from_gripper_array).all()
        or not np.isfinite(object_from_gripper_array).all()
    ):
        raise ValueError("grasp transforms must be finite 4x4 matrices")
    result = world_from_gripper_array @ np.linalg.inv(
        object_from_gripper_array
    )
    determinant = float(np.linalg.det(result[:3, :3]))
    if not np.isclose(determinant, 1.0, atol=1e-9):
        raise ValueError(
            f"world-from-object rotation determinant is {determinant}"
        )
    closure = result @ object_from_gripper_array
    if not np.allclose(
        closure,
        world_from_gripper_array,
        atol=1e-12,
    ):
        raise RuntimeError("object/gripper transform closure failed")
    return result


def compute_evidence_camera_pose(subject_points: Any) -> dict[str, Any]:
    """Derive a full-arm evidence view from live world-space subjects."""
    import numpy as np

    points = np.asarray(subject_points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 2
        or not np.isfinite(points).all()
    ):
        raise ValueError("subject points must be a finite Nx3 array")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    target = (minimum + maximum) / 2.0
    diagonal = float(np.linalg.norm(maximum - minimum))
    distance = max(1.2, diagonal * 2.8)
    view_direction = np.asarray([1.2, -1.4, 0.9], dtype=float)
    view_direction /= np.linalg.norm(view_direction)
    eye = target + view_direction * distance
    return {
        "policy": "RUNTIME_SUBJECT_AABB_OBLIQUE_FULL_ARM",
        "subject_aabb_min": minimum,
        "subject_aabb_max": maximum,
        "target": target,
        "eye": eye,
        "distance_m": distance,
    }


def compute_closeup_camera_pose(subject_points: Any) -> dict[str, Any]:
    """Derive a fixed close-up view of both fingers and the bottle."""
    import numpy as np

    points = np.asarray(subject_points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("close-up subject points must be a finite Nx3 array")
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    target = (minimum + maximum) / 2.0
    diagonal = float(np.linalg.norm(maximum - minimum))
    distance = max(0.35, diagonal * 3.0)
    # Look across the finger-line axis, not along it, so neither finger is
    # hidden behind the bottle in the close-up evidence.
    view_direction = np.asarray([1.0, 0.0, 0.35], dtype=float)
    view_direction /= np.linalg.norm(view_direction)
    eye = target + view_direction * distance
    return {
        "policy": "RUNTIME_GRIPPER_BOTTLE_AABB_OBLIQUE_CLOSEUP",
        "subject_aabb_min": minimum,
        "subject_aabb_max": maximum,
        "target": target,
        "eye": eye,
        "distance_m": distance,
    }


def _matrix_from_pose(
    position: Sequence[float],
    quaternion_wxyz: Sequence[float],
    *,
    np: Any,
    rotation_type: Any,
) -> Any:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_type.from_quat(
        [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ]
    ).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _pose_from_matrix(
    matrix: Any,
    *,
    np: Any,
    rotation_type: Any,
) -> tuple[Any, Any]:
    homogeneous = np.asarray(matrix, dtype=float)
    quaternion_xyzw = rotation_type.from_matrix(
        homogeneous[:3, :3]
    ).as_quat()
    return (
        homogeneous[:3, 3].copy(),
        np.asarray(
            [
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            ],
            dtype=float,
        ),
    )


def _load_object_from_gripper(
    path: Path,
    *,
    yaml: Any,
    np: Any,
    rotation_type: Any,
) -> Any:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    grasp = document["grasps"]["horizontal_body_grasp"]
    return _matrix_from_pose(
        grasp["position"],
        [grasp["orientation"]["w"], *grasp["orientation"]["xyz"]],
        np=np,
        rotation_type=rotation_type,
    )


def _pump_until(
    app: Any,
    predicate: Any,
    *,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while app.is_running() and time.monotonic() < deadline:
        app.update()
        if predicate():
            return
    raise RuntimeError(f"timed out waiting for {label}")


def _find_exact_isaac_window_ids() -> list[str]:
    query = subprocess.run(
        [
            "xdotool",
            "search",
            "--class",
            EXPECTED_ISAAC_WM_CLASS_REGEX,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    candidates = [
        value.strip()
        for value in query.stdout.splitlines()
        if value.strip()
    ]
    exact: list[str] = []
    expected_xprop = (
        f'WM_CLASS(STRING) = "{EXPECTED_ISAAC_WM_CLASS}", '
        f'"{EXPECTED_ISAAC_WM_CLASS}"'
    )
    for window_id in candidates:
        wm_class = subprocess.run(
            ["xprop", "-id", window_id, "WM_CLASS"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if wm_class == expected_xprop:
            exact.append(window_id)
    return exact


def _find_isaac_window_id() -> str:
    exact = _find_exact_isaac_window_ids()
    if len(exact) != 1:
        raise RuntimeError(
            "expected exactly one Isaac Sim window with exact WM_CLASS "
            f"{EXPECTED_ISAAC_WM_CLASS!r}, got {exact}"
        )
    return exact[0]


def validate_workspace_assignment(
    *,
    prelaunch_current_desktop: str,
    isaac_window_desktop: str,
    postlaunch_current_desktop: str,
) -> dict[str, object]:
    """Validate GNOME auto-routing without changing the user's desktop."""
    if isaac_window_desktop != str(ISAAC_WORKSPACE_INDEX):
        raise RuntimeError(
            "Isaac Sim GNOME auto-route mismatch: "
            f"expected {ISAAC_WORKSPACE_INDEX}, got {isaac_window_desktop}"
        )
    if postlaunch_current_desktop != prelaunch_current_desktop:
        raise RuntimeError(
            "user current desktop changed during Isaac Sim launch: "
            f"{prelaunch_current_desktop} -> {postlaunch_current_desktop}"
        )
    return {
        "routing": "GNOME_AUTO_MOVE_WINDOWS",
        "workspace_zero_based": ISAAC_WORKSPACE_INDEX,
        "workspace_human": ISAAC_WORKSPACE_INDEX + 1,
        "readback": isaac_window_desktop,
        "prelaunch_current_desktop": prelaunch_current_desktop,
        "postlaunch_current_desktop": postlaunch_current_desktop,
        "current_desktop_unchanged": True,
    }


def _validate_gnome_auto_route_prelaunch() -> dict[str, object]:
    extension_info = subprocess.run(
        ["gnome-extensions", "info", GNOME_AUTO_MOVE_UUID],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "State: ACTIVE" not in extension_info:
        raise RuntimeError("GNOME Auto Move Windows extension is not ACTIVE")

    schema_dir = GNOME_EXTENSION_DIR / "schemas"
    rules = subprocess.run(
        [
            "gsettings",
            "--schemadir",
            str(schema_dir),
            "get",
            "org.gnome.shell.extensions.auto-move-windows",
            "application-list",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_rule = f"{GNOME_DESKTOP_ENTRY_ID}:2"
    if expected_rule not in rules:
        raise RuntimeError(
            f"GNOME Auto Move rule is missing {expected_rule}: {rules}"
        )
    if not GNOME_DESKTOP_ENTRY_PATH.is_file():
        raise RuntimeError(
            f"GNOME desktop entry is missing: {GNOME_DESKTOP_ENTRY_PATH}"
        )
    desktop_entry = GNOME_DESKTOP_ENTRY_PATH.read_text(encoding="utf-8")
    expected_class_line = f"StartupWMClass={EXPECTED_ISAAC_WM_CLASS}"
    if expected_class_line not in desktop_entry:
        raise RuntimeError(
            f"GNOME desktop entry is missing {expected_class_line}"
        )
    existing_windows = _find_exact_isaac_window_ids()
    if existing_windows:
        raise RuntimeError("an Isaac Sim window already exists before launch")
    current_desktop = subprocess.run(
        ["xdotool", "get_desktop"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_desktop != "0":
        raise RuntimeError(
            "user must remain on desktop 1 before launch; "
            f"readback was {current_desktop}"
        )
    return {
        "extension_uuid": GNOME_AUTO_MOVE_UUID,
        "desktop_entry_id": GNOME_DESKTOP_ENTRY_ID,
        "desktop_entry_path": str(GNOME_DESKTOP_ENTRY_PATH),
        "startup_wm_class": EXPECTED_ISAAC_WM_CLASS,
        "rule_readback": rules,
        "current_desktop": current_desktop,
    }


def _read_focus_new_windows() -> str:
    value = subprocess.run(
        [
            "gsettings",
            "get",
            GNOME_WM_SCHEMA,
            GNOME_FOCUS_KEY,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    normalized = value.strip("'\"")
    if normalized not in {"smart", "strict"}:
        raise RuntimeError(
            f"unexpected GNOME focus-new-windows value: {value}"
        )
    return normalized


def _enable_launch_focus_guard() -> dict[str, object]:
    original = _read_focus_new_windows()
    if original != "strict":
        subprocess.run(
            [
                "gsettings",
                "set",
                GNOME_WM_SCHEMA,
                "focus-new-windows", "strict",
            ],
            check=True,
        )
    readback = _read_focus_new_windows()
    if readback != "strict":
        raise RuntimeError(
            f"GNOME strict launch-focus guard did not stick: {readback}"
        )
    return {
        "original": original,
        "launch_value": readback,
        "restored": False,
        "restore_readback": None,
    }


def _restore_launch_focus_policy(
    policy: dict[str, object],
) -> dict[str, object]:
    if bool(policy["restored"]):
        return policy
    original = str(policy["original"])
    subprocess.run(
        [
            "gsettings",
            "set",
            GNOME_WM_SCHEMA,
            GNOME_FOCUS_KEY,
            original,
        ],
        check=True,
    )
    readback = _read_focus_new_windows()
    if readback != original:
        raise RuntimeError(
            "GNOME launch-focus policy restoration failed: "
            f"{readback} != {original}"
        )
    policy["restored"] = True
    policy["restore_readback"] = readback
    return policy


def _verify_isaac_auto_routed_workspace(
    prelaunch: dict[str, object],
) -> dict[str, object]:
    window_id = _find_isaac_window_id()
    window_desktop = subprocess.run(
        ["xdotool", "get_desktop_for_window", window_id],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_desktop = subprocess.run(
        ["xdotool", "get_desktop"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = validate_workspace_assignment(
        prelaunch_current_desktop=str(prelaunch["current_desktop"]),
        isaac_window_desktop=window_desktop,
        postlaunch_current_desktop=current_desktop,
    )
    result.update(prelaunch)
    result["window_id"] = window_id
    return result


def _capture_app_swapchain(
    *,
    app: Any,
    renderer_capture: Any,
    output_path: Path,
    phase: str,
) -> dict[str, object]:
    renderer_capture.capture_next_frame_swapchain(str(output_path))
    app.update()
    renderer_capture.wait_async_capture()
    for _ in range(3):
        app.update()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"GUI screenshot was not written: {output_path}")
    return {
        "capture_method": "KIT_SWAPCHAIN_CAPTURE",
        "phase": phase,
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def _configure_evidence_view(
    *,
    app: Any,
    get_world_pose: Any,
    get_active_viewport: Any,
    set_camera_view: Any,
) -> dict[str, object]:
    subject_paths = [
        BASE_FRAME_PATH,
        GRASP_FRAME_PATH,
        BOTTLE_SESSION_PATH,
    ]
    subject_positions = []
    for path in subject_paths:
        position, _ = get_world_pose(path)
        subject_positions.append(position)
    camera = compute_evidence_camera_pose(subject_positions)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active Isaac viewport is unavailable")
    set_camera_view(
        eye=camera["eye"],
        target=camera["target"],
        viewport_api=viewport,
    )
    for _ in range(30):
        app.update()
    return {
        "status": "PASS",
        "policy": camera["policy"],
        "subject_paths": subject_paths,
        "subject_world_positions_m": [
            [float(value) for value in position]
            for position in subject_positions
        ],
        "subject_aabb_min_m": camera["subject_aabb_min"].tolist(),
        "subject_aabb_max_m": camera["subject_aabb_max"].tolist(),
        "eye_world_m": camera["eye"].tolist(),
        "target_world_m": camera["target"].tolist(),
        "distance_m": camera["distance_m"],
        "viewport_camera_path": str(viewport.camera_path),
    }


def _configure_closeup_evidence_view(
    *,
    app: Any,
    get_world_pose: Any,
    get_active_viewport: Any,
    set_camera_view: Any,
) -> dict[str, object]:
    subject_paths = [
        LEFT_FINGER_COLLIDER_PATH,
        RIGHT_FINGER_COLLIDER_PATH,
        BOTTLE_SESSION_PATH,
    ]
    subject_positions = []
    for path in subject_paths:
        position, _ = get_world_pose(path)
        subject_positions.append(position)
    camera = compute_closeup_camera_pose(subject_positions)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active Isaac viewport is unavailable")
    set_camera_view(
        eye=camera["eye"],
        target=camera["target"],
        viewport_api=viewport,
    )
    for _ in range(30):
        app.update()
    return {
        "status": "PASS",
        "policy": camera["policy"],
        "subject_paths": subject_paths,
        "subject_world_positions_m": [
            [float(value) for value in position]
            for position in subject_positions
        ],
        "subject_aabb_min_m": camera["subject_aabb_min"].tolist(),
        "subject_aabb_max_m": camera["subject_aabb_max"].tolist(),
        "eye_world_m": camera["eye"].tolist(),
        "target_world_m": camera["target"].tolist(),
        "distance_m": camera["distance_m"],
        "viewport_camera_path": str(viewport.camera_path),
    }


def _select_only_follower_left(items: Sequence[str]) -> str:
    matches = [
        item
        for item in items
        if item.startswith(ARTICULATION_SELECTION_PREFIX)
        and "follower_right" not in item
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Grasp Editor articulation dropdown did not resolve exactly one "
            f"follower_left: items={list(items)}, matches={matches}"
        )
    return matches[0]


def configure_gripper_frame_dropdown(
    *,
    dropdown: Any,
    articulation_prim_path: str,
    desired_frame_path: str,
    desired_frame_is_valid_xformable: bool,
) -> dict[str, object]:
    """Correct the 2.0.20 frame scope for a joint-root articulation."""
    if not desired_frame_is_valid_xformable:
        raise RuntimeError(
            f"gripper frame is not a valid Xformable: {desired_frame_path}"
        )
    native_items = list(dropdown.get_items())
    native_contains_desired = desired_frame_path in native_items
    classification = "NATIVE_GRASP_EDITOR_FRAME_SCOPE"
    if not native_contains_desired:
        if not articulation_prim_path.endswith("/root_joint"):
            raise RuntimeError(
                "Grasp Editor frame is absent for an unclassified "
                f"articulation prim: {articulation_prim_path}"
            )
        dropdown.set_populate_fn(
            lambda: [desired_frame_path],
            repopulate=True,
        )
        classification = (
            "DIAGNOSTIC_GRASP_EDITOR_ARTICULATION_ROOT_JOINT_FRAME_SCOPE"
        )
    dropdown.set_selection(desired_frame_path)
    if dropdown.get_selection() != desired_frame_path:
        raise RuntimeError(
            f"Grasp Editor gripper frame selection did not stick: "
            f"{dropdown.get_items()}"
        )
    return {
        "status": "PASS",
        "classification": classification,
        "articulation_prim_path": articulation_prim_path,
        "desired_frame_path": desired_frame_path,
        "desired_frame_is_valid_xformable": True,
        "native_items": native_items,
        "native_items_contained_desired_frame": native_contains_desired,
        "final_items": list(dropdown.get_items()),
        "final_selection": dropdown.get_selection(),
        "stage_or_extension_source_modified": False,
    }


def classify_native_grasp_result(
    *,
    native_success: bool,
    mimic_error_abs_m: float,
    contact_summary_status: str,
) -> dict[str, object]:
    """Keep native, mimic, and physical contact gates independent."""
    failure_reasons: list[str] = []
    native_status = "PASS" if native_success else "FAIL"
    mimic_status = (
        "PASS"
        if mimic_error_abs_m <= MIMIC_ERROR_TOLERANCE_M
        else "FAIL"
    )
    contact_status = (
        "PASS" if contact_summary_status == "PASS" else "FAIL"
    )
    if not native_success:
        failure_reasons.append("NATIVE_GRASP_EDITOR_SIMULATE_FAILED")
    if mimic_status == "FAIL":
        failure_reasons.append(
            "MIMIC_ERROR_EXCEEDS_"
            f"{MIMIC_ERROR_TOLERANCE_M:.3f}_M"
        )
    if contact_status == "FAIL":
        failure_reasons.append("CONTACT_GEOMETRY_GATE_FAILED")
    return {
        "status": "PASS" if not failure_reasons else "FAIL",
        "native_simulate": native_status,
        "mimic_accuracy": mimic_status,
        "contact_geometry": contact_status,
        "failure_reasons": failure_reasons,
    }


def analyze_mimic_checkpoints(
    checkpoints: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Find the first phase where the authored symmetric mimic diverges."""
    analyzed = []
    first_failing_phase = None
    maximum_residual = 0.0
    for checkpoint in checkpoints:
        left = float(checkpoint["left_finger_m"])
        right = float(checkpoint["right_finger_m"])
        residual = abs(left + right)
        maximum_residual = max(maximum_residual, residual)
        status = (
            "PASS"
            if residual <= MIMIC_ERROR_TOLERANCE_M
            else "FAIL"
        )
        if status == "FAIL" and first_failing_phase is None:
            first_failing_phase = str(checkpoint["phase"])
        analyzed.append(
            {
                **checkpoint,
                "residual_abs_m": residual,
                "status": status,
            }
        )
    return {
        "status": "PASS" if first_failing_phase is None else "FAIL",
        "tolerance_m": MIMIC_ERROR_TOLERANCE_M,
        "first_failing_phase": first_failing_phase,
        "maximum_residual_abs_m": maximum_residual,
        "checkpoints": analyzed,
    }


def summarize_bottle_contacts(
    contacts: Sequence[dict[str, object]],
    *,
    bottle_token: str,
    left_finger_token: str,
    right_finger_token: str,
    robot_token: str,
    accepted_phases: set[str] | None = None,
) -> dict[str, object]:
    """Classify physical bottle contacts without trusting native confidence."""
    bottle_contacts = []
    left_contact = False
    right_contact = False
    unexpected_pairs: set[tuple[str, str]] = set()
    for contact in contacts:
        if (
            accepted_phases is not None
            and str(contact["phase"]) not in accepted_phases
        ):
            continue
        collider0 = str(contact["collider0_path"])
        collider1 = str(contact["collider1_path"])
        if bottle_token not in f"{collider0}\n{collider1}":
            continue
        if float(contact["separation_m"]) > 0.0:
            continue
        bottle_contacts.append(contact)
        pair_text = f"{collider0}\n{collider1}"
        left_contact = left_contact or left_finger_token in pair_text
        right_contact = right_contact or right_finger_token in pair_text
        if (
            robot_token in pair_text
            and left_finger_token not in pair_text
            and right_finger_token not in pair_text
        ):
            unexpected_pairs.add(tuple(sorted((collider0, collider1))))
    bilateral = left_contact and right_contact
    unexpected = bool(unexpected_pairs)
    impulses = [
        float(contact["impulse_ns"]) for contact in bottle_contacts
    ]
    separations = [
        float(contact["separation_m"]) for contact in bottle_contacts
    ]
    return {
        "status": "PASS" if bilateral and not unexpected else "FAIL",
        "physical_bottle_contact_count": len(bottle_contacts),
        "left_finger_contact": left_contact,
        "right_finger_contact": right_contact,
        "bilateral_finger_contact": bilateral,
        "unexpected_robot_contact": unexpected,
        "unexpected_pairs": [
            list(pair) for pair in sorted(unexpected_pairs)
        ],
        "impulses_finite": all(
            value == value and abs(value) != float("inf")
            for value in impulses
        ),
        "maximum_impulse_ns": max(impulses, default=0.0),
        "minimum_separation_m": min(separations, default=None),
        "accepted_phases": (
            sorted(accepted_phases)
            if accepted_phases is not None
            else None
        ),
    }


def _apply_contact_reporting(
    *,
    stage: Any,
    root_paths: Sequence[str],
    usd: Any,
    usd_physics: Any,
    physx_schema: Any,
) -> list[str]:
    applied_paths = []
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(
                f"contact-report root is missing: {root_path}"
            )
        for prim in usd.PrimRange(root):
            if not prim.HasAPI(usd_physics.RigidBodyAPI):
                continue
            physx_schema.PhysxContactReportAPI.Apply(
                prim
            ).CreateThresholdAttr().Set(0.0)
            applied_paths.append(str(prim.GetPath()))
    if not applied_paths:
        raise RuntimeError("no rigid bodies received contact reporting")
    return sorted(set(applied_paths))


def _serialize_contact_events(
    headers: Sequence[Any],
    data: Sequence[Any],
    *,
    phase: str,
    path_from_id: Any,
    np: Any,
) -> list[dict[str, object]]:
    records = []
    for header in headers:
        collider0 = path_from_id(header.collider0)
        collider1 = path_from_id(header.collider1)
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        for index in range(start, end):
            item = data[index]
            impulse = np.asarray(item.impulse, dtype=float)
            records.append(
                {
                    "event_type": str(header.type),
                    "phase": phase,
                    "actor0_path": path_from_id(header.actor0),
                    "actor1_path": path_from_id(header.actor1),
                    "collider0_path": collider0,
                    "collider1_path": collider1,
                    "position_world_m": [
                        float(value) for value in item.position
                    ],
                    "normal_world": [
                        float(value) for value in item.normal
                    ],
                    "impulse_ns": float(np.linalg.norm(impulse)),
                    "impulse_vector_ns": impulse.tolist(),
                    "separation_m": float(item.separation),
                    "material0_path": path_from_id(item.material0),
                    "material1_path": path_from_id(item.material1),
                }
            )
    return records


def _validate_native_export(
    path: Path,
    *,
    yaml: Any,
) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("format") != "isaac_grasp":
        raise RuntimeError("native Grasp Editor export format mismatch")
    if document.get("gripper_frame") != GRIPPER_FRAME_PATH:
        raise RuntimeError(
            "native Grasp Editor gripper frame mismatch: "
            f"{document.get('gripper_frame')}"
        )
    if document.get("object_frame") != BOTTLE_SESSION_PATH:
        raise RuntimeError(
            "native Grasp Editor object frame mismatch: "
            f"{document.get('object_frame')}"
        )
    grasps = document.get("grasps", {})
    if list(grasps) != ["grasp_0"]:
        raise RuntimeError(
            f"native Grasp Editor grasp names changed: {list(grasps)}"
        )
    grasp = grasps["grasp_0"]
    active_joints = list(grasp.get("cspace_position", {}))
    pregrasp_joints = list(grasp.get("pregrasp_cspace_position", {}))
    if active_joints != ["left_finger"] or pregrasp_joints != [
        "left_finger"
    ]:
        raise RuntimeError(
            "Variant B native export must contain only active left_finger"
        )
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "grasp_name": "grasp_0",
        "active_joints": active_joints,
        "object_frame": document.get("object_frame"),
        "gripper_frame": document.get("gripper_frame"),
        "right_finger_policy": "RUNTIME_OBSERVER_NOT_NATIVE_EXPORT_FIELD",
    }


def _configure_and_run_native_gui(
    *,
    app: Any,
    builder: Any,
    export_path: Path,
    output_root: Path,
    np: Any,
    rotation_type: Any,
    yaml: Any,
    get_world_pose: Any,
    get_active_viewport: Any,
    articulation_action_type: Any,
    renderer_capture: Any,
    set_camera_view: Any,
    stage: Any,
    usd_geom: Any,
    contact_records: list[dict[str, object]],
    contact_state: dict[str, str],
    contact_report_paths: Sequence[str],
    grasp_frame_authoring: dict[str, object],
    mimic_load_case: dict[str, object],
    execution_mode: str,
    timeline: Any,
) -> dict[str, object]:
    grasp_frame_readback = validate_existing_grasp_frame_runtime(
        get_world_pose=get_world_pose,
        np=np,
        rotation_type=rotation_type,
    )
    contact_state["phase"] = "GRASP_EDITOR_INITIALIZATION"
    builder._gripper_selection_dropdown.repopulate()  # noqa: SLF001
    articulation_items = (
        builder._gripper_selection_dropdown.get_items()  # noqa: SLF001
    )
    articulation_selection = _select_only_follower_left(articulation_items)
    builder._gripper_selection_dropdown.set_selection(  # noqa: SLF001
        articulation_selection
    )
    builder._rb_conversion_stringfield.set_value(  # noqa: SLF001
        BOTTLE_SESSION_PATH
    )
    builder._export_path.set_value(str(export_path))  # noqa: SLF001
    builder._selection_ready_btn.trigger_click_if_a_state()  # noqa: SLF001
    _pump_until(
        app,
        lambda: builder._articulation is not None,  # noqa: SLF001
        timeout_s=60.0,
        label="Grasp Editor articulation initialization",
    )

    articulation = builder._articulation  # noqa: SLF001
    mimic_checkpoints: list[dict[str, object]] = []
    positions = np.asarray(
        articulation.get_joint_positions(),
        dtype=float,
    )
    if tuple(articulation.dof_names) != (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ):
        raise RuntimeError(
            f"unexpected runtime DOF order: {articulation.dof_names}"
        )
    authoring_pose = select_grasp_editor_authoring_pose()
    positions[:6] = np.asarray(authoring_pose["arm_q_rad"], dtype=float)
    positions[7] = float(VARIANT_B["open_position_m"])
    positions[8] = float(VARIANT_B["observer_open_position_m"])
    articulation.set_joint_positions(positions)
    articulation.set_joint_velocities(np.zeros_like(positions))
    articulation.apply_action(
        articulation_action_type(joint_positions=positions)
    )

    def record_mimic_checkpoint(phase: str) -> None:
        readback = np.asarray(
            articulation.get_joint_positions(),
            dtype=float,
        )
        mimic_checkpoints.append(
            {
                "phase": phase,
                "left_finger_m": float(readback[7]),
                "right_finger_m": float(readback[8]),
            }
        )

    record_mimic_checkpoint("SET_POSITION_IMMEDIATE")
    for _ in range(4):
        app.update()
    record_mimic_checkpoint("PRE_BOTTLE_PHYSICS_UPDATES")

    contact_state["phase"] = "POSITION_BOTTLE_AT_INPUT_CANDIDATE"
    world_gripper_position, world_gripper_quaternion = get_world_pose(
        GRIPPER_FRAME_PATH
    )
    world_from_gripper = _matrix_from_pose(
        world_gripper_position,
        world_gripper_quaternion,
        np=np,
        rotation_type=rotation_type,
    )
    object_from_gripper = _load_object_from_gripper(
        GRASP_CANDIDATE_PATH,
        yaml=yaml,
        np=np,
        rotation_type=rotation_type,
    )
    world_from_object = compute_world_from_object(
        world_from_gripper,
        object_from_gripper,
    )
    bottle_translation_delta = np.asarray(
        mimic_load_case["bottle_translation_delta_world_m"],
        dtype=float,
    )
    world_from_object[:3, 3] += bottle_translation_delta
    object_position, object_quaternion = _pose_from_matrix(
        world_from_object,
        np=np,
        rotation_type=rotation_type,
    )
    builder._rigid_body.set_world_poses(  # noqa: SLF001
        object_position[np.newaxis, :],
        object_quaternion[np.newaxis, :],
    )
    builder.stop_rigid_body()
    for _ in range(4):
        app.update()
        builder.stop_rigid_body()
    record_mimic_checkpoint("POST_BOTTLE_PLACEMENT")

    contact_state["phase"] = "CONFIGURE_NATIVE_GRASP_EDITOR"
    gripper_frame_dropdown = configure_gripper_frame_dropdown(
        dropdown=builder._gripper_subframe,  # noqa: SLF001
        articulation_prim_path=str(articulation.prim.GetPath()),
        desired_frame_path=GRIPPER_FRAME_PATH,
        desired_frame_is_valid_xformable=bool(
            usd_geom.Xformable(stage.GetPrimAtPath(GRIPPER_FRAME_PATH))
        ),
    )
    builder._rb_subframe_filter.set_value(  # noqa: SLF001
        BOTTLE_SESSION_PATH
    )
    builder._rb_subframe.repopulate()  # noqa: SLF001
    builder._rb_subframe.set_selection(  # noqa: SLF001
        BOTTLE_SESSION_PATH
    )
    if (
        builder._gripper_subframe.get_selection()  # noqa: SLF001
        != GRIPPER_FRAME_PATH
        or builder._rb_subframe.get_selection()  # noqa: SLF001
        != BOTTLE_SESSION_PATH
    ):
        raise RuntimeError("Grasp Editor frame selection did not stick")
    builder._finalize_frame_btn.trigger_click()  # noqa: SLF001

    variant_contract = apply_variant_b_joint_settings(
        builder._joint_settings_ui_state,  # noqa: SLF001
        articulation,
        positions,
    )
    left_index = articulation.get_dof_index("left_finger")
    right_index = articulation.get_dof_index("right_finger")
    articulation.get_articulation_controller().set_max_efforts(
        [float(VARIANT_B["max_effort_n"])],
        [left_index],
    )
    for frame in builder._robot_joint_frames:  # noqa: SLF001
        frame.rebuild()
    builder._test_frame.rebuild()  # noqa: SLF001
    for index, frame in enumerate(
        builder._robot_joint_frames  # noqa: SLF001
    ):
        if articulation.dof_names[index] not in {
            "left_finger",
            "right_finger",
        }:
            frame.collapsed = True

    evidence_view = _configure_evidence_view(
        app=app,
        get_world_pose=get_world_pose,
        get_active_viewport=get_active_viewport,
        set_camera_view=set_camera_view,
    )
    record_mimic_checkpoint("POST_GRASP_EDITOR_CONFIGURATION")
    for _ in range(12):
        app.update()
    record_mimic_checkpoint("BEFORE_NATIVE_SIMULATE")
    configured_capture = _capture_app_swapchain(
        app=app,
        renderer_capture=renderer_capture,
        output_path=(
            output_root / "grasp_editor_variant_b_configured_raw.png"
        ),
        phase="CONFIGURED_BEFORE_SIMULATE",
    )
    closeup_evidence_view = _configure_closeup_evidence_view(
        app=app,
        get_world_pose=get_world_pose,
        get_active_viewport=get_active_viewport,
        set_camera_view=set_camera_view,
    )
    configured_closeup_capture = _capture_app_swapchain(
        app=app,
        renderer_capture=renderer_capture,
        output_path=(
            output_root
            / "grasp_editor_variant_b_configured_closeup_raw.png"
        ),
        phase="CONFIGURED_OPEN_CLOSEUP",
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active Isaac viewport is unavailable")
    set_camera_view(
        eye=evidence_view["eye_world_m"],
        target=evidence_view["target_world_m"],
        viewport_api=viewport,
    )
    for _ in range(30):
        app.update()

    before_test = np.asarray(
        articulation.get_joint_positions(),
        dtype=float,
    )
    external_close_trace: list[dict[str, object]] = []
    if execution_mode == "native_simulate":
        contact_state["phase"] = "NATIVE_SIMULATE"
        builder._test_state_btn.trigger_click_if_a_state()  # noqa: SLF001
        _pump_until(
            app,
            lambda: builder._last_grasp_test_results is not None,  # noqa: SLF001
            timeout_s=120.0,
            label="native Grasp Editor SIMULATE result",
        )
        result = builder._last_grasp_test_results  # noqa: SLF001
        result_phase = "POST_NATIVE_SIMULATE"
        contact_phases = {"NATIVE_SIMULATE"}
        result_capture_name = "grasp_editor_variant_b_simulated_raw.png"
        result_capture_phase = "SIMULATED_RESULT"
    elif execution_mode == "external_contact_skip_sim":
        if mimic_load_case["name"] != "bottle_contact":
            raise RuntimeError(
                "external_contact_skip_sim requires bottle_contact load case"
            )
        close_targets = build_external_close_targets(
            open_position_m=float(VARIANT_B["open_position_m"]),
            contact_target_m=float(
                VARIANT_B["clearance_contact_position_m"]
            ),
            speed_m_s=float(VARIANT_B["max_speed_m_s"]),
            physics_dt_s=1.0 / 60.0,
        )
        contact_state["phase"] = "EXTERNAL_PROGRAMMATIC_CLOSE"
        timeline.play()
        app.update()
        for frame, target_m in enumerate(close_targets, start=1):
            articulation.apply_action(
                articulation_action_type(
                    joint_positions=np.asarray([target_m], dtype=float),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            app.update()
            readback = np.asarray(
                articulation.get_joint_positions(),
                dtype=float,
            )
            external_close_trace.append(
                {
                    "frame": frame,
                    "target_left_finger_m": float(target_m),
                    "readback_left_finger_m": float(readback[left_index]),
                    "readback_right_finger_m": float(readback[right_index]),
                    "mimic_residual_abs_m": abs(
                        float(readback[right_index])
                        + float(readback[left_index])
                    ),
                }
            )
        for hold_frame in range(120):
            articulation.apply_action(
                articulation_action_type(
                    joint_positions=np.asarray(
                        [close_targets[-1]],
                        dtype=float,
                    ),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            app.update()
            if hold_frame in {0, 29, 59, 119}:
                readback = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=float,
                )
                external_close_trace.append(
                    {
                        "hold_frame": hold_frame,
                        "target_left_finger_m": float(close_targets[-1]),
                        "readback_left_finger_m": float(
                            readback[left_index]
                        ),
                        "readback_right_finger_m": float(
                            readback[right_index]
                        ),
                        "mimic_residual_abs_m": abs(
                            float(readback[right_index])
                            + float(readback[left_index])
                        ),
                    }
                )
        timeline.pause()
        app.update()
        builder._last_grasp_test_results = None  # noqa: SLF001
        builder._test_skip_btn.trigger_click()  # noqa: SLF001
        if builder._last_grasp_test_results is None:  # noqa: SLF001
            raise RuntimeError("native Grasp Editor SKIP SIM returned no result")
        result = builder._last_grasp_test_results  # noqa: SLF001
        result_phase = "POST_EXTERNAL_CONTACT_SKIP_SIM"
        contact_phases = {"EXTERNAL_PROGRAMMATIC_CLOSE"}
        result_capture_name = (
            "grasp_editor_variant_b_external_contact_raw.png"
        )
        result_capture_phase = "EXTERNAL_CONTACT_SKIP_SIM_RESULT"
    else:
        raise ValueError(f"unsupported execution mode: {execution_mode}")

    after_test = np.asarray(
        articulation.get_joint_positions(),
        dtype=float,
    )
    contact_state["phase"] = result_phase
    record_mimic_checkpoint(result_phase)
    mimic_settle_trace: list[dict[str, object]] = []
    for frame in range(MIMIC_SETTLE_OBSERVATION_FRAMES[-1] + 1):
        if frame in MIMIC_SETTLE_OBSERVATION_FRAMES:
            readback = np.asarray(
                articulation.get_joint_positions(),
                dtype=float,
            )
            mimic_settle_trace.append(
                {
                    "frame": frame,
                    "left_finger_m": float(readback[left_index]),
                    "right_finger_m": float(readback[right_index]),
                    "residual_abs_m": abs(
                        float(readback[right_index])
                        + float(readback[left_index])
                    ),
                }
            )
        if frame == MIMIC_SETTLE_OBSERVATION_FRAMES[-1]:
            break
        app.update()
    settled_after_test = np.asarray(
        articulation.get_joint_positions(),
        dtype=float,
    )
    result_capture = _capture_app_swapchain(
        app=app,
        renderer_capture=renderer_capture,
        output_path=(
            output_root / result_capture_name
        ),
        phase=result_capture_phase,
    )
    set_camera_view(
        eye=closeup_evidence_view["eye_world_m"],
        target=closeup_evidence_view["target_world_m"],
        viewport_api=viewport,
    )
    for _ in range(30):
        app.update()
    result_closeup_capture = _capture_app_swapchain(
        app=app,
        renderer_capture=renderer_capture,
        output_path=(
            output_root
            / result_capture_name.replace("_raw.png", "_closeup_raw.png")
        ),
        phase=f"{result_capture_phase}_CLOSEUP",
    )
    if mimic_load_case["native_export_policy"] == "VALIDATE_NATIVE_EXPORT":
        builder._export_btn.trigger_click()  # noqa: SLF001
        for _ in range(4):
            app.update()
        native_export = _validate_native_export(export_path, yaml=yaml)
    else:
        native_export = {
            "status": "NOT_RUN",
            "reason": mimic_load_case["native_export_policy"],
        }
    contact_path = output_root / "grasp_editor_variant_b_contacts.json"
    contact_path.write_text(
        json.dumps(contact_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contact_summary = summarize_bottle_contacts(
        contact_records,
        bottle_token=BOTTLE_SESSION_PATH,
        left_finger_token=LEFT_FINGER_COLLIDER_PATH,
        right_finger_token=RIGHT_FINGER_COLLIDER_PATH,
        robot_token=FOLLOWER_LEFT_ROOT_PATH,
        accepted_phases=contact_phases,
    )

    left_after = float(after_test[left_index])
    right_after = float(after_test[right_index])
    mimic_error = abs(right_after + left_after)
    derived_export: dict[str, object] = {
        "status": "NOT_RUN",
        "reason": "NATIVE_SIMULATE_DOES_NOT_REQUIRE_SKIP_SIM_DERIVATION",
    }
    if execution_mode == "external_contact_skip_sim":
        raw_document = yaml.safe_load(export_path.read_text(encoding="utf-8"))
        derived_document = derive_skip_sim_yaml_document(
            raw_document,
            open_position_m=float(VARIANT_B["open_position_m"]),
        )
        derived_path = (
            output_root / "grasp_editor_variant_b_skip_sim_derived.yaml"
        )
        derived_path.write_text(
            yaml.safe_dump(derived_document, sort_keys=False),
            encoding="utf-8",
        )
        derived_export = _validate_native_export(derived_path, yaml=yaml)
        derived_export["classification"] = (
            "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
        )
        gate = classify_external_skip_sim_result(
            mimic_error_abs_m=mimic_error,
            contact_summary_status=str(contact_summary["status"]),
            raw_export_status=str(native_export["status"]),
            derived_export_status=str(derived_export["status"]),
        )
    elif bool(mimic_load_case["expected_bottle_contact"]):
        gate = classify_native_grasp_result(
            native_success=bool(result.success),
            mimic_error_abs_m=mimic_error,
            contact_summary_status=str(contact_summary["status"]),
        )
    else:
        gate = classify_no_contact_diagnostic_result(
            native_success=bool(result.success),
            mimic_error_abs_m=mimic_error,
            physical_bottle_contact_count=int(
                contact_summary["physical_bottle_contact_count"]
            ),
        )

    return {
        "status": gate["status"],
        "gate": gate,
        "mimic_load_case": mimic_load_case,
        "execution_mode": execution_mode,
        "variant": variant_contract,
        "authoring_pose": authoring_pose,
        "articulation_dropdown_items": list(articulation_items),
        "articulation_selection": articulation_selection,
        "dof_order": list(articulation.dof_names),
        "frames": {
            "gripper_link_reference": GRIPPER_LINK_FRAME_PATH,
            "gripper": GRIPPER_FRAME_PATH,
            "cad_contact_helper": GRASP_FRAME_PATH,
            "object": BOTTLE_SESSION_PATH,
            "grasp_frame_authoring": grasp_frame_authoring,
            "grasp_frame_readback": grasp_frame_readback,
            "gripper_dropdown_compatibility": gripper_frame_dropdown,
        },
        "placement": {
            "formula": "T_W_O = T_W_G @ inverse(T_O_G)",
            "diagnostic_bottle_translation_delta_world_m": (
                bottle_translation_delta.tolist()
            ),
            "world_from_gripper": world_from_gripper.tolist(),
            "object_from_gripper_input": object_from_gripper.tolist(),
            "world_from_object": world_from_object.tolist(),
            "closure_max_abs": float(
                np.max(
                    np.abs(
                        world_from_object @ object_from_gripper
                        - world_from_gripper
                    )
                )
            ),
        },
        "joint_readback": {
            "before_test": before_test.tolist(),
            "after_test": after_test.tolist(),
            "settled_after_test": settled_after_test.tolist(),
            "left_finger_after_m": left_after,
            "right_finger_after_m": right_after,
            "mimic_error_abs_m": mimic_error,
            "post_simulate_settle_trace": mimic_settle_trace,
            "post_simulate_settle_classification": (
                classify_mimic_settle_trace(
                    mimic_settle_trace,
                    tolerance_m=MIMIC_ERROR_TOLERANCE_M,
                )
            ),
            "checkpoint_analysis": analyze_mimic_checkpoints(
                mimic_checkpoints
            ),
        },
        "simulate": {
            "success": bool(result.success),
            "suggested_confidence": float(result.suggested_confidence),
            "stable_positions": np.asarray(
                result.stable_positions,
                dtype=float,
            ).tolist(),
        },
        "native_export": native_export,
        "derived_export": derived_export,
        "external_close_trace": external_close_trace,
        "contacts": {
            "reporting_rigid_body_paths": list(contact_report_paths),
            "raw_path": str(contact_path.resolve()),
            "raw_sha256": _sha256(contact_path),
            "raw_event_point_count": len(contact_records),
            "summary": contact_summary,
        },
        "evidence_view": evidence_view,
        "closeup_evidence_view": closeup_evidence_view,
        "screenshots": [
            configured_capture,
            configured_closeup_capture,
            result_capture,
            result_closeup_capture,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mimic-load-case",
        choices=("bottle_contact", "no_object_contact"),
        default="bottle_contact",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("native_simulate", "external_contact_skip_sim"),
        default="native_simulate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mimic_load_case = select_mimic_load_case(args.mimic_load_case)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    export_path = output_root / "grasp_editor_variant_b_native_raw.yaml"
    report_path = output_root / "grasp_editor_variant_b_gui_report.json"

    frozen_inputs = {
        "stage": (STAGE_PATH, EXPECTED_STAGE_SHA256),
        "bottle_usd": (BOTTLE_USD, EXPECTED_BOTTLE_SHA256),
        "grasp_candidate": (
            GRASP_CANDIDATE_PATH,
            EXPECTED_GRASP_CANDIDATE_SHA256,
        ),
    }
    input_manifest: dict[str, dict[str, object]] = {}
    for label, (path, expected_sha256) in frozen_inputs.items():
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{label} hash mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        input_manifest[label] = {
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "size_bytes": path.stat().st_size,
        }

    workspace_prelaunch = _validate_gnome_auto_route_prelaunch()
    launch_focus_policy = _enable_launch_focus_guard()

    import isaacsim

    try:
        app = isaacsim.SimulationApp(
            {
                "headless": False,
                "sync_loads": True,
                "fast_shutdown": False,
            }
        )
    except BaseException:
        _restore_launch_focus_policy(launch_focus_policy)
        raise
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "FAIL",
        "classification": "NATIVE_GRASP_EDITOR_GUI_VARIANT_B",
        "mimic_load_case": mimic_load_case,
        "inputs": input_manifest,
        "task8": "NOT_RUN",
    }
    stage = None
    root_layer = None
    root_specs_before = None
    root_dirty_before = None
    session_layer = None
    diagnostic_layer_identifier = None
    previous_edit_target = None
    contact_subscription = None
    try:
        import gc

        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.core.utils.xforms import get_world_pose
        import numpy as np
        import omni.kit.actions.core
        import omni.kit.app
        import omni.kit.renderer_capture
        from omni.kit.viewport.utility import get_active_viewport
        from omni.physx import get_physx_simulation_interface
        import omni.timeline
        import omni.usd
        from pxr import Gf
        from pxr import PhysicsSchemaTools
        from pxr import PhysxSchema
        from pxr import Sdf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics
        from scipy.spatial.transform import Rotation
        import yaml

        from tools.open_aloha1_grasp_editor_diagnostic import _add_external_reference
        from tools.open_aloha1_grasp_editor_diagnostic import _enable_extension_exact
        from tools.open_aloha1_grasp_editor_diagnostic import _remove_exact_session_sublayer
        from tools.open_aloha1_grasp_editor_diagnostic import _restore_previous_edit_target
        from tools.open_aloha1_grasp_editor_diagnostic import _validate_loaded_stage_contract

        manager = omni.kit.app.get_app().get_extension_manager()
        if manager.is_extension_enabled(GRASP_EDITOR_EXTENSION_ID):
            manager.set_extension_enabled_immediate(
                GRASP_EDITOR_EXTENSION_ID,
                False,  # noqa: FBT003 - local Kit API is positional.
            )
            app.update()
        enabled_id, version = _enable_extension_exact(
            manager,
            extension_id=GRASP_EDITOR_EXTENSION_ID,
            expected_version=GRASP_EDITOR_VERSION,
        )
        app.update()

        from isaacsim.robot_setup.grasp_editor import ui_builder as ui_module

        captured_builders = [
            item
            for item in gc.get_objects()
            if isinstance(item, ui_module.UIBuilder)
        ]
        if len(captured_builders) != 1:
            raise RuntimeError(
                "could not resolve exactly one native Grasp Editor UIBuilder: "
                f"{len(captured_builders)}"
            )
        builder = captured_builders[0]

        if not open_stage(str(STAGE_PATH.resolve())):
            raise RuntimeError(f"failed to open approved Stage: {STAGE_PATH}")
        for _ in range(8):
            app.update()
        stage = omni.usd.get_context().get_stage()
        stage_contract = _validate_loaded_stage_contract(stage)
        root_layer = stage.GetRootLayer()
        root_specs_before = root_layer.ExportToString()
        root_dirty_before = root_layer.dirty
        session_layer = stage.GetSessionLayer()
        previous_edit_target = stage.GetEditTarget()
        diagnostic_layer = Sdf.Layer.CreateAnonymous(
            "ALOHA1GraspEditorVariantBNativeGui"
        )
        diagnostic_layer_identifier = diagnostic_layer.identifier
        session_layer.subLayerPaths.append(diagnostic_layer_identifier)
        stage.SetEditTarget(diagnostic_layer)

        grasp_frame_authoring = author_session_supplier_cad_grasp_frame(
            stage=stage,
        )
        UsdGeom.Xform.Define(stage, "/World/ALOHA1GraspEditorSession")
        task_frame = UsdGeom.Xform.Define(
            stage,
            "/World/ALOHA1GraspEditorSession/W_T",
        )
        task_frame.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        bottle_prim = UsdGeom.Xform.Define(
            stage,
            BOTTLE_SESSION_PATH,
        ).GetPrim()
        _add_external_reference(
            bottle_prim,
            BOTTLE_USD,
            Sdf.Path("/Bottle500"),
        )
        bottle_prim.SetCustomDataByKey(
            "aloha1:classification",
            "DIAGNOSTIC_SESSION_ONLY_NOT_FINAL",
        )
        contact_report_paths = _apply_contact_reporting(
            stage=stage,
            root_paths=[
                FOLLOWER_LEFT_ROOT_PATH,
                BOTTLE_SESSION_PATH,
            ],
            usd=Usd,
            usd_physics=UsdPhysics,
            physx_schema=PhysxSchema,
        )
        contact_records: list[dict[str, object]] = []
        contact_state = {"phase": "SESSION_SETUP"}

        def path_from_id(value: Any) -> str:
            return str(PhysicsSchemaTools.intToSdfPath(value))

        def on_contact(
            headers: Sequence[Any],
            data: Sequence[Any],
        ) -> None:
            contact_records.extend(
                _serialize_contact_events(
                    headers,
                    data,
                    phase=contact_state["phase"],
                    path_from_id=path_from_id,
                    np=np,
                )
            )

        contact_subscription = (
            get_physx_simulation_interface()
            .subscribe_contact_report_events(on_contact)
        )
        app.update()

        action_id = f"CreateUIExtension:{WINDOW_TITLE}"
        action = omni.kit.actions.core.get_action_registry().get_action(
            enabled_id,
            action_id,
        )
        if action is None:
            action = omni.kit.actions.core.get_action_registry().get_action(
                GRASP_EDITOR_EXTENSION_ID,
                action_id,
            )
        if action is None:
            raise RuntimeError("native Grasp Editor action was not registered")
        action.execute()
        _pump_until(
            app,
            lambda: hasattr(builder, "_gripper_selection_dropdown"),
            timeout_s=30.0,
            label="native Grasp Editor selection controls",
        )
        try:
            workspace = _verify_isaac_auto_routed_workspace(
                workspace_prelaunch
            )
        finally:
            _restore_launch_focus_policy(launch_focus_policy)
        report["runtime"] = {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "grasp_editor_extension_id": enabled_id,
            "grasp_editor_version": version,
            "stage_contract": stage_contract,
            "task_frame_world_translation_m": [0.0, 0.0, 0.0],
            "gui_workspace": workspace,
            "launch_focus_policy": launch_focus_policy,
        }
        report["result"] = _configure_and_run_native_gui(
            app=app,
            builder=builder,
            export_path=export_path,
            output_root=output_root,
            np=np,
            rotation_type=Rotation,
            yaml=yaml,
            get_world_pose=get_world_pose,
            get_active_viewport=get_active_viewport,
            articulation_action_type=ArticulationAction,
            renderer_capture=(
                omni.kit.renderer_capture
                .acquire_renderer_capture_interface()
            ),
            set_camera_view=set_camera_view,
            stage=stage,
            usd_geom=UsdGeom,
            contact_records=contact_records,
            contact_state=contact_state,
            contact_report_paths=contact_report_paths,
            grasp_frame_authoring=grasp_frame_authoring,
            mimic_load_case=mimic_load_case,
            execution_mode=args.execution_mode,
            timeline=omni.timeline.get_timeline_interface(),
        )
        if report["result"]["status"] != "PASS":
            failure_reasons = report["result"]["gate"]["failure_reasons"]
            raise RuntimeError(
                "native Grasp Editor GUI gates failed: "
                + ", ".join(failure_reasons)
            )
        report["status"] = "PASS"
    except BaseException as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        if contact_subscription is not None:
            del contact_subscription
        cleanup_errors: list[str] = []
        try:
            _restore_launch_focus_policy(launch_focus_policy)
        except BaseException as error:
            cleanup_errors.append(
                "launch focus policy: "
                f"{type(error).__name__}: {error}"
            )
        try:
            if stage is not None and previous_edit_target is not None:
                _restore_previous_edit_target(stage, previous_edit_target)
            if (
                session_layer is not None
                and diagnostic_layer_identifier is not None
            ):
                _remove_exact_session_sublayer(
                    session_layer,
                    diagnostic_layer_identifier,
                )
            if (
                root_layer is not None
                and root_specs_before is not None
                and root_layer.ExportToString() != root_specs_before
            ):
                cleanup_errors.append("source root specs changed")
            if (
                root_layer is not None
                and root_dirty_before is not None
                and root_layer.dirty != root_dirty_before
            ):
                cleanup_errors.append("source root dirty state changed")
        except BaseException as error:
            cleanup_errors.append(f"{type(error).__name__}: {error}")
        report["cleanup_errors"] = cleanup_errors
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        app.close()
        for label, (path, expected_sha256) in frozen_inputs.items():
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"{label} changed during native GUI run: "
                    f"{actual_sha256}"
                )
        if cleanup_errors:
            raise RuntimeError(
                "native Grasp Editor cleanup failed: "
                + "; ".join(cleanup_errors)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

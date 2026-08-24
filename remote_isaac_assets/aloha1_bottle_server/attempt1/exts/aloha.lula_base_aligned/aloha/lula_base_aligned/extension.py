from __future__ import annotations

import math
import os
import traceback
import csv
import asyncio
import json
import gc
import weakref
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import carb
import numpy as np
import omni.ext
import omni.kit.app
import omni.physx
import omni.timeline
import omni.ui as ui
import omni.usd
import yaml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.numpy import rot_matrices_to_quats
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.gui.components.menu import make_menu_item_description
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics


EXTENSION_NAME = "ALOHA Lula Base Aligned"
# Build marker: repeatable random grasp/lift plus atomic canonical Bottle reset.

LEFT_ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
LEFT_BASE_PATH = "/World/follower_left/vx300s_left/follower_left_base_link"
LEFT_EE_PATH = "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
LEFT_EE_FRAME = "follower_left_ee_gripper_link"
TARGET_PATH = "/World/ALOHAAlignedIKTarget"
BOTTLE_PATH = "/World/ALOHA1RemoteBottleSession/Bottle500"
BOTTLE_CAP_PATH = "/World/ALOHA1RemoteBottleSession/BottleCap"
BOTTLE_THREAD_SLIDER_PATH = "/World/ALOHA1RemoteBottleSession/BottleThreadSlider"
BOTTLE_THREAD_PRISMATIC_PATH = "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadPrismatic"
BOTTLE_THREAD_REVOLUTE_PATH = "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadRevolute"
BOTTLE_THREAD_COUPLING_PATH = "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/RightHandThreadCoupling"
LEFT_FINGER_LINK_PATH = "/World/follower_left/vx300s_left/follower_left_left_finger_link"
RIGHT_FINGER_LINK_PATH = "/World/follower_left/vx300s_left/follower_left_right_finger_link"
LEFT_ROBOT_PATH = "/World/follower_left/vx300s_left"

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
LEFT_SLEEP_ARM_RAD = np.asarray([0.0, -1.8, 1.55, 0.0, -1.57, 0.0], dtype=np.float64)
SLEEP_READBACK_GATE_RAD = 0.02

DEFAULT_BUNDLE = os.environ.get(
    "ALOHA1_REMOTE_BUNDLE",
    "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1",
)
DEFAULT_DESCRIPTION = os.path.join(DEFAULT_BUNDLE, "configs", "aloha1_lula_follower_left.yaml")
DEFAULT_URDF = os.path.join(
    DEFAULT_BUNDLE,
    "assets",
    "Trossen",
    "ALOHA1",
    "1.0",
    "follower_vx300s",
    "follower_left",
    "source",
    "follower_left.urdf",
)
DEFAULT_LOG_DIR = os.path.join(DEFAULT_BUNDLE, "reports", "lula_joint_diagnostics")
ALIGNMENT_DIAGNOSTIC_PATH = os.path.join(DEFAULT_LOG_DIR, "latest_alignment_diagnostic.json")
DEFAULT_GRASP = os.path.join(
    DEFAULT_BUNDLE,
    "configs",
    "aloha1_grasps",
    "bottle500_horizontal_body_grasp_isolated_20260817.yaml",
)
AUTO_REQUEST_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_z5_acceptance_request.json")
AUTO_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_z5_acceptance_result.json")
AUTO_HOVER_REQUEST_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_hover_acceptance_request.json")
AUTO_HOVER_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_hover_acceptance_result.json")
AUTO_HOVER_PROGRESS_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_hover_acceptance_progress.json")
AUTO_RESET_SLEEP_REQUEST_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_reset_sleep_request.json")
AUTO_RESET_SLEEP_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_reset_sleep_result.json")
AUTO_OPEN_GRIPPER_REQUEST_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_open_left_gripper_request.json")
AUTO_OPEN_GRIPPER_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_open_left_gripper_result.json")
AUTO_DYNAMIC_SELF_CENTER_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_dynamic_self_center_request.json"
)
AUTO_DYNAMIC_SELF_CENTER_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_dynamic_self_center_result.json"
)
BOTTLE_ROTATE_REQUEST_PATH = os.path.join(DEFAULT_LOG_DIR, "rotate_bottle_midpoint_request.json")
BOTTLE_ROTATE_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "rotate_bottle_midpoint_result.json")
THREAD_BASE_JOINT_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "create_bottle_thread_base_joints_request.json"
)
THREAD_BASE_JOINT_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "create_bottle_thread_base_joints_result.json"
)
THREAD_COUPLING_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "create_bottle_thread_coupling_request.json"
)
THREAD_COUPLING_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "create_bottle_thread_coupling_result.json"
)
THREAD_RELEASE_TEST_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "bottle_thread_release_transition_request.json"
)
THREAD_RELEASE_TEST_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "bottle_thread_release_transition_result.json"
)
THREAD_RELEASE_VERIFY_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "bottle_thread_release_verification_request.json"
)
THREAD_RELEASE_VERIFY_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "bottle_thread_release_verification_result.json"
)
HOVER_REACHABILITY_DIAGNOSTIC_PATH = os.path.join(
    DEFAULT_LOG_DIR, "hover_reachability_diagnostic.json"
)
RANDOM_BOTTLE_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "random_bottle_pose_latest.json")
AUTO_GRASP_LIFT_RESULT_PATH = os.path.join(DEFAULT_LOG_DIR, "auto_random_grasp_lift_latest.json")
AUTO_RANDOM_BOTTLE_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_random_bottle_pose_request.json"
)
AUTO_GRASP_LIFT_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_random_grasp_lift_request.json"
)
AUTO_RESET_BOTTLE_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_reset_bottle_initial_pose_request.json"
)
AUTO_RESET_BOTTLE_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "auto_reset_bottle_initial_pose_result.json"
)
HORIZONTAL_CYLINDER_BENCHMARK_REQUEST_PATH = os.path.join(
    DEFAULT_LOG_DIR, "horizontal_cylinder_benchmark_request.json"
)
HORIZONTAL_CYLINDER_BENCHMARK_RESULT_PATH = os.path.join(
    DEFAULT_LOG_DIR, "horizontal_cylinder_benchmark_result.json"
)
VERTICAL_LIFT_DIAGNOSTIC_PATH = os.path.join(
    DEFAULT_LOG_DIR, "vertical_lift_rotate_diagnostic.json"
)

POSITION_GATE_M = 0.001
ORIENTATION_GATE_RAD = math.radians(0.5)
RUNTIME_POSITION_GATE_M = 0.005
RUNTIME_ORIENTATION_GATE_RAD = math.radians(2.0)
MAX_TARGET_TRANSLATION_STEP_M = 0.005
MAX_TARGET_ROTATION_STEP_RAD = math.radians(2.0)
MAX_JOINT_STEP_RAD = 0.02
PLANNED_JOINT_STEP_DEFAULT_RAD = 0.03
PLANNED_JOINT_STEP_MIN_RAD = 0.005
PLANNED_JOINT_STEP_MAX_RAD = 0.03
AUTO_APPROACH_JOINT_STEP_RAD = 0.01
AUTO_GRASP_CLEARANCE_M = 0.0
HOVER_PLAN_REACHED_RAD = 0.001
HOVER_PLAN_CONTROL_PERIOD_S = 0.020
HOVER_PLAN_SETTLED_VELOCITY_RAD_S = 0.020
HOVER_PLAN_MIN_EE_Z_M = 0.050
HOVER_PLAN_MAX_EE_STEP_M = 0.030
GUIDED_ROUTE_MAX_LATERAL_DEVIATION_M = 0.020
ARM_RUNTIME_STIFFNESS = 1600.0
ARM_RUNTIME_DAMPING = 100.0

# The orange Target is a grasp-tool frame, not the native Lula/URDF EE frame.
# In ALOHA's URDF, EE +X runs from the wrist toward the fingertips.  For a
# top-down grasp EE +X points along world -Z, while the operator-facing Target
# convention is Target +Z along world +Z.  Therefore:
#
#     R_W_EE = R_W_Target @ R_Target_EE
#     R_Target_EE = RotY(+90 deg)
#
# The two frames share the same origin; only orientation is converted.
TARGET_TO_EE_ORIENTATION_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
)
TOP_DOWN_IK_WARM_STARTS = (
    np.asarray([0.05823425, 0.64196199, -0.68369853, 0.0, 1.61278450, 0.05771287], dtype=np.float64),
    np.asarray([0.22278139, 1.15937018, -0.52136779, -2.48255348, 2.22179914, 0.19523042], dtype=np.float64),
)

# The Grasp Editor candidate was authored at Bottle-local axial z=69 mm.
# Full-articulation validation selected a reproducible diagnostic center with
# radial correction (-5.5, -1.5) mm and axial correction -10 mm.  These are
# kept explicit so the UI never silently renames the contact frame C as the
# Lula/URDF end-effector frame G.
GRASP_OBJECT_LOCAL_CORRECTION_M = np.asarray([-0.0055, -0.0015, -0.0100], dtype=np.float64)
BOTTLE_LENGTH_M = 0.206
GRASP_AXIAL_FRACTION_FROM_BOTTOM = 1.0 / 3.0
HOVER_CLEARANCE_M = 0.160
HOVER_CLEARANCE_CANDIDATES_M = (0.160, 0.180, 0.140, 0.200, 0.120, 0.220, 0.250)
HOVER_AXIAL_FRACTION_CANDIDATES = (1.0 / 3.0, 0.30, 0.36, 0.27, 0.40, 0.25, 0.45, 0.50)
PREGRASP_CLEARANCE_M = 0.120
NEAR_CLEARANCE_M = 0.010
VERTICAL_LIFT_ENDPOINT_PROBE_M = 0.180
VERTICAL_LIFT_MAX_CARTESIAN_STEP_M = 0.004
VERTICAL_LIFT_MAX_ORIENTATION_STEP_RAD = math.radians(1.0)
# Loaded lift and rotation use the last repeatably accepted 0.0075 rad / 50 Hz
# reference increment.  The 0.010 rad experiment intermittently lost unilateral
# contact and therefore is not retained as the operational setting.
VERTICAL_LIFT_MAX_JOINT_STEP_RAD = 0.0075
VERTICAL_CENTER_ROTATION_MAX_JOINT_STEP_RAD = 0.005
VERTICAL_LIFT_ROTATION_START_CLEARANCE_M = 0.150
VERTICAL_BOTTLE_TARGET_LIFT_M = 0.150
VERTICAL_BOTTLE_LIFT_TOLERANCE_M = 0.005
VERTICAL_LIFT_MIN_BOTTLE_BOTTOM_Z_M = 0.020
VERTICAL_LIFT_CAP_AXIS_GATE_RAD = math.radians(10.0)
APPROACH_STEP_M = 0.005
LEFT_GRIPPER_OPEN_POSITION_M = 0.057
LEFT_GRIPPER_MIN_POSITION_M = 0.021
GRIPPER_CALIBRATION_STEP_M = 0.001
GRIPPER_MAX_SETTLE_UPDATES = 600
GRIPPER_TARGET_TOLERANCE_M = 0.00015
GRIPPER_TARGET_STABLE_UPDATES = 5
GRIPPER_CONTACT_STABLE_UPDATES = 3
# Do not lift on the first transient contact patch.  Fifteen consecutive
# contact-report samples give the compliant fingers time to finish closing and
# settle before the loaded arm route begins.
GRIPPER_BILATERAL_STABLE_STEPS = 15
GRIPPER_MIMIC_RESIDUAL_GATE_M = 0.0025
GRIPPER_MIMIC_BAD_STEPS = 5
# Continuous Dynamic self-centering intentionally permits transient/loaded
# compliant displacement above the 2.5 mm paused-calibration gate.  A larger
# structural guard remains active so a jammed finger cannot run indefinitely.
GRIPPER_DYNAMIC_MIMIC_RESIDUAL_GATE_M = 0.010
GRIPPER_DYNAMIC_MIMIC_BAD_STEPS = 10
VERTICAL_LOADED_MIMIC_RESIDUAL_GATE_M = 0.0125
RANDOM_BOTTLE_X_RANGE_M = (-0.30, 0.05)
RANDOM_BOTTLE_Y_RANGE_M = (-0.20, 0.20)
RANDOM_BOTTLE_CENTER_Z_M = 0.034
RANDOM_BOTTLE_MAX_ATTEMPTS = 40
CANONICAL_BOTTLE_CENTER_WORLD_M = np.asarray([0.0, 0.0, 0.034], dtype=np.float64)
BOTTLE_ROOT_FROM_CENTER_LOCAL_M = np.asarray([0.0, 0.0, -0.103], dtype=np.float64)
CAP_ROOT_FROM_CENTER_LOCAL_M = np.asarray([0.0, 0.0, 0.085], dtype=np.float64)
CANONICAL_BOTTLE_ORIENTATION_WXYZ = np.asarray(
    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    return quat / norm


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def _quat_step(current: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
    current = _quat_normalize(current)
    target = _quat_normalize(target)
    if float(np.dot(current, target)) < 0.0:
        target = -target
    angle = _quat_angle(current, target)
    if angle <= max_angle:
        return target
    alpha = max_angle / max(angle, 1e-12)
    return _quat_normalize((1.0 - alpha) * current + alpha * target)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two scalar-first quaternions: R(a*b) = R(a) @ R(b)."""

    aw, ax, ay, az = _quat_normalize(a)
    bw, bx, by, bz = _quat_normalize(b)
    return _quat_normalize(
        np.asarray(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        if abs(float(angle_rad)) <= 1e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        raise ValueError("nonzero axis-angle rotation requires a finite axis")
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    return _quat_normalize(
        np.asarray(
            [math.cos(half), *(axis * math.sin(half))], dtype=np.float64
        )
    )


def _quat_slerp(start: np.ndarray, goal: np.ndarray, alpha: float) -> np.ndarray:
    start = _quat_normalize(start)
    goal = _quat_normalize(goal)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(start, goal))
    if dot < 0.0:
        goal = -goal
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize((1.0 - alpha) * start + alpha * goal)
    angle = math.acos(dot)
    scale = math.sin(angle)
    return _quat_normalize(
        math.sin((1.0 - alpha) * angle) / scale * start
        + math.sin(alpha * angle) / scale * goal
    )


class AlohaLulaBaseAlignedExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._window: Optional[ui.Window] = None
        self._articulation: Optional[SingleArticulation] = None
        self._lula: Optional[LulaKinematicsSolver] = None
        self._art_ik: Optional[ArticulationKinematicsSolver] = None
        self._target: Optional[VisualCuboid] = None
        self._follow_enabled = False
        self._aligned = False
        self._step_count = 0
        self._last_position_error = float("inf")
        self._last_orientation_error = float("inf")
        self._joint_log_enabled = False
        self._joint_log_rows: List[Dict[str, float]] = []
        self._joint_log_run_id = ""
        self._last_joint_log_path = ""
        self._joint_log_elapsed_s = 0.0
        self._auto_task = None
        self._open_window_task = None
        self._reset_task = None
        self._gripper_task = None
        self._request_poll_counter = 0
        self._gripper_abort_requested = False
        self._gripper_command_target_m = float("nan")
        self._grasp_contact_pairs: Dict[Tuple[str, ...], str] = {}
        self._recent_contact_paths: List[Tuple[str, ...]] = []
        self._grasp_left_contact = False
        self._grasp_right_contact = False
        self._grasp_nonfinger_contact = False
        self._grasp_bilateral_streak = 0
        self._grasp_mimic_bad_streak = 0
        self._contact_ui_counter = 0
        self._grasp_loaded = False
        self._grasp_name = ""
        self._grasp_world_position: Optional[np.ndarray] = None
        self._grasp_world_orientation: Optional[np.ndarray] = None
        self._grasp_object_position: Optional[np.ndarray] = None
        self._grasp_object_orientation: Optional[np.ndarray] = None
        self._grasp_preopen_position = float("nan")
        self._grasp_closed_position = float("nan")
        self._active_waypoint = "none"
        self._hover_plan_positions: List[np.ndarray] = []
        self._hover_plan_index = 0
        self._hover_plan_goal_position: Optional[np.ndarray] = None
        self._hover_plan_goal_orientation: Optional[np.ndarray] = None
        self._hover_plan_metrics: Dict[str, object] = {}
        self._hover_plan_elapsed_s = 0.0
        self._hover_reached_reported = False
        self._planned_route_name = ""
        self._locked_auto_joint_step_rad: Optional[float] = None
        self._auto_abort_requested = False
        self._last_random_bottle_pose: Optional[Dict[str, object]] = None
        self._last_auto_recovery: Optional[Dict[str, object]] = None
        self._requested_random_pose_override: Optional[Tuple[np.ndarray, float]] = None
        self._rng = np.random.default_rng()

        self._timeline = omni.timeline.get_timeline_interface()
        self._physx = omni.physx.get_physx_interface()
        self._request_poll_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._poll_external_requests)
        )
        self._physics_subscription = self._physx.subscribe_physics_step_events(self._on_physics_step)
        self._contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report_event
        )

        self._description_model = ui.SimpleStringModel(DEFAULT_DESCRIPTION)
        self._urdf_model = ui.SimpleStringModel(DEFAULT_URDF)
        self._grasp_model = ui.SimpleStringModel(DEFAULT_GRASP)
        self._planned_joint_step_model = ui.SimpleFloatModel(
            PLANNED_JOINT_STEP_DEFAULT_RAD
        )
        self._status_label = None
        self._workflow_label = None
        self._grasp_label = None
        self._target_label = None
        self._position_error_label = None
        self._orientation_error_label = None
        self._joint_log_label = None
        self._gripper_state_label = None

        self._menu_items = [
            make_menu_item_description(
                ext_id,
                EXTENSION_NAME,
                lambda extension=weakref.proxy(self): extension._toggle_window(),
            ),
        ]
        add_menu_items(self._menu_items, "Tools")
        self._build_window()
        self._open_window_task = asyncio.ensure_future(self._show_window_next_update())
        carb.log_info(f"[{EXTENSION_NAME}] started in SAFE IDLE state")
        if os.path.isfile(HORIZONTAL_CYLINDER_BENCHMARK_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(
                self._run_external_horizontal_cylinder_benchmark()
            )
        elif os.path.isfile(AUTO_RESET_BOTTLE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_reset_bottle_request())
        elif os.path.isfile(AUTO_RANDOM_BOTTLE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_random_bottle_request())
        elif os.path.isfile(AUTO_GRASP_LIFT_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_grasp_lift_request())
        elif os.path.isfile(AUTO_DYNAMIC_SELF_CENTER_REQUEST_PATH):
            self._gripper_task = asyncio.ensure_future(self._run_auto_dynamic_self_center())
        elif os.path.isfile(AUTO_OPEN_GRIPPER_REQUEST_PATH):
            self._gripper_task = asyncio.ensure_future(self._run_auto_open_left_gripper())
        elif os.path.isfile(AUTO_HOVER_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_auto_hover_acceptance())
        elif os.path.isfile(AUTO_RESET_SLEEP_REQUEST_PATH):
            self._reset_task = asyncio.ensure_future(self._run_auto_reset_sleep())
        elif os.path.isfile(AUTO_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_auto_z5_acceptance())
        elif os.path.isfile(BOTTLE_ROTATE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_rotate_bottle_midpoint())
        elif os.path.isfile(THREAD_BASE_JOINT_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_create_thread_base_joints())
        elif os.path.isfile(THREAD_COUPLING_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_create_thread_coupling())
        elif os.path.isfile(THREAD_RELEASE_TEST_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_thread_release_test_request())
        elif os.path.isfile(THREAD_RELEASE_VERIFY_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_thread_release_verify_request())

    def on_shutdown(self) -> None:
        self._follow_enabled = False
        if getattr(self, "_auto_task", None) is not None:
            self._auto_task.cancel()
            self._auto_task = None
        if getattr(self, "_open_window_task", None) is not None:
            self._open_window_task.cancel()
            self._open_window_task = None
        if getattr(self, "_reset_task", None) is not None:
            self._reset_task.cancel()
            self._reset_task = None
        if getattr(self, "_gripper_task", None) is not None:
            self._gripper_task.cancel()
            self._gripper_task = None
        self._request_poll_subscription = None
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Tools")
            self._menu_items = []
        self._physics_subscription = None
        self._contact_subscription = None
        self._target = None
        self._art_ik = None
        self._lula = None
        self._articulation = None
        if self._window is not None:
            self._window.visible = False
            self._window = None
        gc.collect()

    def _poll_external_requests(self, _event) -> None:
        """Consume simulation requests while Kit is running, including while paused."""

        self._request_poll_counter += 1
        if self._request_poll_counter % 10 != 0:
            return
        if self._gripper_task is not None and not self._gripper_task.done():
            return
        if self._auto_task is not None and not self._auto_task.done():
            return
        if self._reset_task is not None and not self._reset_task.done():
            return
        if os.path.isfile(HORIZONTAL_CYLINDER_BENCHMARK_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(
                self._run_external_horizontal_cylinder_benchmark()
            )
        elif os.path.isfile(AUTO_RESET_BOTTLE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_reset_bottle_request())
        elif os.path.isfile(AUTO_RANDOM_BOTTLE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_random_bottle_request())
        elif os.path.isfile(AUTO_GRASP_LIFT_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_external_grasp_lift_request())
        elif os.path.isfile(AUTO_DYNAMIC_SELF_CENTER_REQUEST_PATH):
            self._gripper_task = asyncio.ensure_future(self._run_auto_dynamic_self_center())
        elif os.path.isfile(AUTO_OPEN_GRIPPER_REQUEST_PATH):
            self._gripper_task = asyncio.ensure_future(self._run_auto_open_left_gripper())
        elif os.path.isfile(AUTO_HOVER_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_auto_hover_acceptance())
        elif os.path.isfile(AUTO_RESET_SLEEP_REQUEST_PATH):
            self._reset_task = asyncio.ensure_future(self._run_auto_reset_sleep())
        elif os.path.isfile(AUTO_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_auto_z5_acceptance())
        elif os.path.isfile(BOTTLE_ROTATE_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_rotate_bottle_midpoint())
        elif os.path.isfile(THREAD_BASE_JOINT_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_create_thread_base_joints())
        elif os.path.isfile(THREAD_COUPLING_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_create_thread_coupling())
        elif os.path.isfile(THREAD_RELEASE_TEST_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_thread_release_test_request())
        elif os.path.isfile(THREAD_RELEASE_VERIFY_REQUEST_PATH):
            self._auto_task = asyncio.ensure_future(self._run_thread_release_verify_request())

    async def _show_window_next_update(self) -> None:
        """Open and focus the panel after Kit has restored its workspace."""

        await omni.kit.app.get_app().next_update_async()
        if self._window is not None:
            self._window.visible = True
            self._window.focus()

    def _toggle_window(self) -> None:
        if self._window is None:
            self._build_window()
            self._window.visible = True
            self._window.focus()
        else:
            self._window.visible = not self._window.visible
            if self._window.visible:
                self._window.focus()

    def _build_window(self) -> None:
        self._window = ui.Window(
            EXTENSION_NAME,
            width=520,
            height=700,
            visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                # Keep the content bounded by the dock width.  Without an explicit
                # fractional width, long controls can establish a minimum width
                # larger than a narrow dock while the horizontal scrollbar is
                # disabled, leaving their visible text and hit area off-screen.
                with ui.VStack(spacing=6, height=0, width=ui.Fraction(1)):
                    self._build_window_contents()

    def _build_window_contents(self) -> None:
        ui.Label("ALOHA Left-Arm Guided Bottle Approach", height=24)
        ui.Label(
            "Operator-guided workflow. Buttons set a guarded IK target; they never command gripper DOFs. Do not save the Stage.",
            word_wrap=True,
            height=42,
        )

        with ui.CollapsableFrame("Workflow status", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                self._workflow_label = ui.Label(
                    "Step 1/4 - Prepare the paused robot.", word_wrap=True, height=36
                )
                ui.Label(
                    "Sequence: Prepare robot -> Validate frames -> Load grasp -> Arm target -> Set one waypoint -> Play -> Pause and inspect.",
                    word_wrap=True,
                    height=52,
                )
                ui.Button(
                    "ABORT: Pause and Hold Current EE",
                    clicked_fn=self._on_abort_and_hold,
                    width=ui.Fraction(1),
                )

        with ui.CollapsableFrame("0. Repeatable random Bottle test", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Random center: x [-0.30, 0.05] m, y [-0.20, 0.20] m; horizontal yaw is randomized. The complete Bottle/Cap/thread assembly is moved atomically while Paused.",
                    word_wrap=True,
                    height=54,
                )
                with ui.HStack(spacing=6, height=28):
                    ui.Label("Planned joint step at 50 Hz (rad)", width=ui.Fraction(3))
                    ui.FloatField(
                        model=self._planned_joint_step_model,
                        width=ui.Fraction(1),
                    )
                ui.Label(
                    "Allowed 0.005-0.030 rad; default 0.030. Auto sequence: Plan Hover -> Near (+10 mm) -> close -> lift to Plan Hover. The Near leg remains limited to 0.010 rad.",
                    word_wrap=True,
                    height=42,
                )
                self._reset_bottle_button = ui.Button(
                    "RESET BOTTLE TO INITIAL POSE",
                    clicked_fn=self._on_reset_bottle_initial_pose,
                    width=ui.Fraction(1),
                    height=34,
                )
                self._randomize_bottle_button = ui.Button(
                    "RANDOMIZE BOTTLE POSE",
                    clicked_fn=self._on_randomize_bottle,
                    width=ui.Fraction(1),
                    height=34,
                )
                self._auto_grasp_lift_button = ui.Button(
                    "GRASP: SLEEP -> HOVER -> NEAR +10 -> CLOSE -> LIFT",
                    clicked_fn=self._on_auto_grasp_lift,
                    width=ui.Fraction(1),
                    height=34,
                )
                self._auto_grasp_vertical_button = ui.Button(
                    "GRASP VERTICAL: LIFT 15 cm -> ROTATE CAP TO +Z (0.0075 rad)",
                    clicked_fn=self._on_auto_grasp_vertical,
                    width=ui.Fraction(1),
                    height=34,
                )
                self._random_test_label = ui.Label(
                    "Random Bottle: not generated in this extension session.",
                    word_wrap=True,
                    height=48,
                )

        with ui.CollapsableFrame("1. Prepare robot", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Timeline must be Paused, not Stopped. Reset preserves all gripper DOFs.", word_wrap=True, height=36)
                self._reset_button = ui.Button(
                    "1A. Reset Left Arm to Sleep (Paused)",
                    clicked_fn=self._on_reset_left_sleep,
                    width=ui.Fraction(1),
                )
                ui.Button(
                    "1B. Load Left Arm and Runtime Drive Profile",
                    clicked_fn=self._on_load_left_arm,
                    width=ui.Fraction(1),
                )

        with ui.CollapsableFrame("2. Align and validate frames", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                self._sync_button = ui.Button(
                    "2A. Sync Lula Base Pose", clicked_fn=self._on_sync_base, width=ui.Fraction(1)
                )
                self._validate_button = ui.Button(
                    "2B. Validate EE Alignment",
                    clicked_fn=self._on_validate_alignment,
                    width=ui.Fraction(1),
                )
                self._position_error_label = ui.Label("Position error: not measured", word_wrap=True)
                self._orientation_error_label = ui.Label("Orientation error: not measured", word_wrap=True)

        with ui.CollapsableFrame("3. Define Bottle grasp", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Isaac grasp YAML (object O -> gripper frame G)", word_wrap=True)
                ui.StringField(model=self._grasp_model, width=ui.Fraction(1))
                self._load_grasp_button = ui.Button(
                    "3A. Load and Validate Bottle Grasp",
                    clicked_fn=self._on_load_bottle_grasp,
                    width=ui.Fraction(1),
                )
                self._grasp_label = ui.Label(
                    "Not loaded. No target will move until the grasp frame contract is validated.",
                    word_wrap=True,
                    height=72,
                )

        with ui.CollapsableFrame("4. Guided target sequence", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "First create the target at the current EE and arm IK while Paused. Then set exactly one waypoint and press Play. Pause before selecting the next waypoint.",
                    word_wrap=True,
                    height=58,
                )
                self._create_target_button = ui.Button(
                    "4A. Create Target at Current EE",
                    clicked_fn=self._on_create_target,
                    width=ui.Fraction(1),
                    height=30,
                )
                self._follow_button = ui.Button(
                    "4B. Arm IK Follow",
                    clicked_fn=self._on_toggle_follow,
                    width=ui.Fraction(1),
                    height=30,
                )
                # Waypoints are deliberately single-column.  Their labels are too
                # long for a two-column dock and Kit can otherwise place part of a
                # button's hit area outside the visible ScrollingFrame.
                self._hover_button = ui.Button(
                    "PLAN HOVER (Bottle bottom L/3)",
                    clicked_fn=self._on_plan_hover_route,
                    width=ui.Fraction(1),
                    height=30,
                )
                self._pregrasp_button = ui.Button(
                    "SAFE PREGRASP (+120 mm)",
                    clicked_fn=lambda: self._on_set_grasp_waypoint("PREGRASP", PREGRASP_CLEARANCE_M),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._near_button = ui.Button(
                    "NEAR (+10 mm)",
                    clicked_fn=lambda: self._on_set_grasp_waypoint("NEAR", NEAR_CLEARANCE_M),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._grasp_pose_button = ui.Button(
                    "GRASP POSE (open, +0 mm)",
                    clicked_fn=lambda: self._on_set_grasp_waypoint("GRASP_POSE", 0.0),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._step_button = ui.Button(
                    "STEP APPROACH (-5 mm)",
                    clicked_fn=self._on_step_approach,
                    width=ui.Fraction(1),
                    height=30,
                )
                self._return_button = ui.Button(
                    "RETURN TO HOVER",
                    clicked_fn=lambda: self._on_set_grasp_waypoint("HOVER", HOVER_CLEARANCE_M),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._target_label = ui.Label(
                    "Target: not created | grasp waypoint: none", word_wrap=True, height=50
                )
                ui.Label(
                    "Guided waypoints solve and validate a bounded joint route while Paused; Play executes it. They do not close fingers or move Bottle500.",
                    word_wrap=True,
                    height=58,
                )

        with ui.CollapsableFrame("5. Gripper preparation", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Open at SAFE PREGRASP before descending. Only left_finger is commanded; right_finger remains Mimic-driven.",
                    word_wrap=True,
                    height=42,
                )
                ui.Button(
                    "5A. OPEN LEFT GRIPPER (0.057 m)",
                    clicked_fn=self._on_open_left_gripper,
                    width=ui.Fraction(1),
                )

        with ui.CollapsableFrame("6. Grasp contact calibration", collapsed=False, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label(
                    "Use only after APPROACH +0 mm is REACHED and the Timeline is Paused. Bottle500 must remain kinematic. Only left_finger is commanded; right_finger remains Mimic-driven.",
                    word_wrap=True,
                    height=58,
                )
                self._gripper_close_step_button = ui.Button(
                    "6A. CLOSE 1 mm",
                    clicked_fn=lambda: self._on_gripper_calibration_step(-GRIPPER_CALIBRATION_STEP_M),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._gripper_open_step_button = ui.Button(
                    "6B. OPEN 1 mm",
                    clicked_fn=lambda: self._on_gripper_calibration_step(GRIPPER_CALIBRATION_STEP_M),
                    width=ui.Fraction(1),
                    height=30,
                )
                self._gripper_auto_close_button = ui.Button(
                    "6C. AUTO CLOSE UNTIL BILATERAL CONTACT",
                    clicked_fn=self._on_auto_close_gripper,
                    width=ui.Fraction(1),
                    height=30,
                )
                self._gripper_abort_button = ui.Button(
                    "6D. ABORT GRIPPER MOTION",
                    clicked_fn=self._on_abort_gripper_motion,
                    width=ui.Fraction(1),
                    height=30,
                )
                self._gripper_state_label = ui.Label(
                    "Finger state: not measured\nContacts: left=no | right=no | non-finger Bottle contact=no",
                    word_wrap=True,
                    height=64,
                )
                self._gripper_readiness_label = ui.Label(
                    "6A/6B/6C readiness: checking current workflow state...",
                    word_wrap=True,
                    height=42,
                )
                ui.Label(
                    "Automatic closure advances the active finger by 1 mm at a time and stops on stable bilateral contact, non-finger Bottle contact, Mimic residual above 2.5 mm, or the 0.021 m lower limit.",
                    word_wrap=True,
                    height=58,
                )

        with ui.CollapsableFrame("Diagnostics and cleanup", collapsed=True, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Button("Disable IK Follow", clicked_fn=self._disable_follow, width=ui.Fraction(1))
                ui.Button("Remove Extension Target", clicked_fn=self._on_remove_target, width=ui.Fraction(1))
                ui.Label(
                    "Records each physics-step actual position, velocity, raw IK request, and bounded position target for all six arm joints.",
                    word_wrap=True,
                    height=42,
                )
                ui.Button("Start New Joint Log", clicked_fn=self._on_start_joint_log, width=ui.Fraction(1))
                ui.Button(
                    "Stop and Save Joint Log CSV",
                    clicked_fn=self._on_stop_joint_log,
                    width=ui.Fraction(1),
                )
                self._joint_log_label = ui.Label(
                    f"Joint log: idle\nOutput: {DEFAULT_LOG_DIR}", word_wrap=True, height=52
                )

        with ui.CollapsableFrame("Advanced configuration", collapsed=True, height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label(f"Articulation: {LEFT_ARTICULATION_PATH}", word_wrap=True)
                ui.Label(f"Base: {LEFT_BASE_PATH}", word_wrap=True)
                ui.Label(f"End effector: {LEFT_EE_FRAME}", word_wrap=True)
                ui.Label("Robot Description YAML")
                ui.StringField(model=self._description_model, width=ui.Fraction(1))
                ui.Label("Robot URDF")
                ui.StringField(model=self._urdf_model, width=ui.Fraction(1))
                ui.Label(
                    "Runtime online IK limits: target 5 mm / 2 deg; arm joints 0.02 rad. Preplanned 50 Hz routes use the configurable 0.005-0.030 rad step. The gripper section commands only left_finger; right_finger remains Mimic-driven.",
                    word_wrap=True,
                    height=42,
                )

        ui.Label("Status")
        self._status_label = ui.Label(
            "SAFE IDLE: load the left arm while the timeline is paused, not stopped.",
            word_wrap=True,
            height=88,
        )
        self._refresh_workflow_ui()

    def _set_status(self, text: str, warn: bool = False) -> None:
        if self._status_label is not None:
            self._status_label.text = text
        if warn:
            carb.log_warn(f"[{EXTENSION_NAME}] {text}")
        else:
            carb.log_info(f"[{EXTENSION_NAME}] {text}")
        self._refresh_workflow_ui()

    def _refresh_workflow_ui(self) -> None:
        if self._workflow_label is None:
            return
        loaded = self._articulation is not None and self._art_ik is not None
        automatic_busy = self._auto_task is not None and not self._auto_task.done()
        for name in (
            "_reset_bottle_button",
            "_randomize_bottle_button",
            "_auto_grasp_lift_button",
            "_auto_grasp_vertical_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.enabled = not automatic_busy
        target_exists = self._target is not None and is_prim_path_valid(TARGET_PATH)
        if not loaded:
            step = "Step 1/4 - Prepare and load the paused left arm."
        elif not self._aligned:
            step = "Step 2/4 - Sync the base and pass EE alignment gates."
        elif not self._grasp_loaded:
            step = "Step 3/4 - Load and validate the Bottle grasp definition."
        elif not target_exists or not self._follow_enabled:
            step = "Step 4/4 - Create the current-EE target, then arm IK Follow while Paused."
        else:
            step = "READY - While Paused, select one waypoint; then Play, Pause, and inspect."
        self._workflow_label.text = step

        for name in ("_sync_button", "_validate_button"):
            button = getattr(self, name, None)
            if button is not None:
                button.enabled = loaded
        if getattr(self, "_create_target_button", None) is not None:
            self._create_target_button.enabled = loaded and self._aligned
        if getattr(self, "_follow_button", None) is not None:
            self._follow_button.enabled = loaded and self._aligned and target_exists
        waypoint_ready = self._grasp_loaded and target_exists and self._follow_enabled
        for name in (
            "_hover_button",
            "_pregrasp_button",
            "_near_button",
            "_grasp_pose_button",
            "_step_button",
            "_return_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.enabled = waypoint_ready
        at_zero_reached = (
            self._active_waypoint.endswith("/ REACHED")
            and ("+0 mm" in self._active_waypoint or self._active_waypoint.startswith("GRASP_POSE"))
        )
        gripper_busy = self._gripper_task is not None and not self._gripper_task.done()
        calibration_ready = waypoint_ready and at_zero_reached and not self._timeline.is_playing() and not gripper_busy
        for name in ("_gripper_close_step_button", "_gripper_open_step_button", "_gripper_auto_close_button"):
            button = getattr(self, name, None)
            if button is not None:
                # Keep the controls clickable while idle.  The callback performs
                # the authoritative safety checks and reports the exact failed
                # gate.  Silently disabling a button made a missed click and a
                # rejected command indistinguishable to the operator.
                button.enabled = not gripper_busy
        if getattr(self, "_gripper_abort_button", None) is not None:
            self._gripper_abort_button.enabled = gripper_busy
        if getattr(self, "_gripper_readiness_label", None) is not None:
            failed_gates = []
            if not waypoint_ready:
                failed_gates.append("load/align/grasp/target/IK Follow")
            if not at_zero_reached:
                failed_gates.append("APPROACH +0 mm / REACHED")
            if self._timeline.is_playing():
                failed_gates.append("Timeline Paused")
            if gripper_busy:
                readiness = "BUSY - a gripper command is running; use 6D to abort"
            elif calibration_ready:
                readiness = "READY - click 6A, 6B, or 6C"
            else:
                readiness = "BLOCKED - missing: " + "; ".join(failed_gates)
            self._gripper_readiness_label.text = f"6A/6B/6C readiness: {readiness}"
        if self._target_label is not None:
            target_state = "created" if target_exists else "not created"
            follow_state = "ARMED" if self._follow_enabled else "disabled"
            self._target_label.text = (
                f"Target: {target_state} | IK Follow: {follow_state} | waypoint: {self._active_waypoint}"
            )
        self._refresh_gripper_state_label()

    def _on_randomize_bottle(self) -> None:
        if self._auto_task is not None and not self._auto_task.done():
            self._set_status("A random Bottle or automatic grasp task is already running.", warn=True)
            return
        self._auto_abort_requested = False
        self._auto_task = asyncio.ensure_future(self._randomize_bottle_transaction())
        self._refresh_workflow_ui()

    def _on_reset_bottle_initial_pose(self) -> None:
        if self._auto_task is not None and not self._auto_task.done():
            self._set_status("A Bottle or automatic grasp task is already running.", warn=True)
            return
        if self._gripper_task is not None and not self._gripper_task.done():
            self._set_status("Stop the active gripper task before resetting Bottle.", warn=True)
            return
        self._auto_abort_requested = False
        self._auto_task = asyncio.ensure_future(
            self._reset_bottle_initial_pose_transaction()
        )
        self._refresh_workflow_ui()

    def _on_auto_grasp_lift(self) -> None:
        if self._auto_task is not None and not self._auto_task.done():
            self._set_status("A random Bottle or automatic grasp task is already running.", warn=True)
            return
        self._auto_abort_requested = False
        self._auto_task = asyncio.ensure_future(self._auto_grasp_lift_transaction())
        self._refresh_workflow_ui()

    def _on_auto_grasp_vertical(self) -> None:
        if self._auto_task is not None and not self._auto_task.done():
            self._set_status("A random Bottle or automatic grasp task is already running.", warn=True)
            return
        self._auto_abort_requested = False
        self._auto_task = asyncio.ensure_future(
            self._auto_grasp_lift_transaction(orient_cap_positive_z=True)
        )
        self._refresh_workflow_ui()

    async def _read_stable_external_request(self, running_path: str) -> Dict[str, object]:
        """Wait for an scp-authored request to become non-empty and stable."""

        app = omni.kit.app.get_app()
        previous_size = -1
        stable_reads = 0
        for _ in range(60):
            current_size = os.path.getsize(running_path)
            if current_size > 0 and current_size == previous_size:
                stable_reads += 1
                if stable_reads >= 2:
                    with open(running_path, "r", encoding="utf-8") as stream:
                        payload = json.load(stream)
                    if not isinstance(payload, dict):
                        raise RuntimeError("external request root must be a JSON object")
                    return payload
            else:
                stable_reads = 0
            previous_size = current_size
            await app.next_update_async()
        raise RuntimeError("external request did not become stable and non-empty")

    async def _run_external_random_bottle_request(self) -> None:
        running_path = AUTO_RANDOM_BOTTLE_REQUEST_PATH + ".running"
        try:
            os.replace(AUTO_RANDOM_BOTTLE_REQUEST_PATH, running_path)
            request = await self._read_stable_external_request(running_path)
            if request.get("simulation_only") is not True:
                raise RuntimeError("external random Bottle request must set simulation_only=true")
            self._planned_joint_step_model.set_value(
                float(request.get("planned_joint_step_rad", PLANNED_JOINT_STEP_DEFAULT_RAD))
            )
            if "center_world_m" in request:
                center = np.asarray(request["center_world_m"], dtype=np.float64)
                if center.shape != (3,):
                    raise RuntimeError("center_world_m override must contain three values")
                self._requested_random_pose_override = (
                    center,
                    math.radians(float(request.get("yaw_deg", 0.0))),
                )
            self._auto_abort_requested = False
            await self._randomize_bottle_transaction()
        finally:
            self._requested_random_pose_override = None
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_external_reset_bottle_request(self) -> None:
        running_path = AUTO_RESET_BOTTLE_REQUEST_PATH + ".running"
        try:
            os.replace(AUTO_RESET_BOTTLE_REQUEST_PATH, running_path)
            request = await self._read_stable_external_request(running_path)
            if request.get("simulation_only") is not True:
                raise RuntimeError("external Bottle reset request must set simulation_only=true")
            self._auto_abort_requested = False
            await self._reset_bottle_initial_pose_transaction()
        finally:
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_external_grasp_lift_request(self) -> None:
        running_path = AUTO_GRASP_LIFT_REQUEST_PATH + ".running"
        try:
            os.replace(AUTO_GRASP_LIFT_REQUEST_PATH, running_path)
            request = await self._read_stable_external_request(running_path)
            if request.get("simulation_only") is not True:
                raise RuntimeError("external grasp request must set simulation_only=true")
            self._planned_joint_step_model.set_value(
                float(request.get("planned_joint_step_rad", PLANNED_JOINT_STEP_DEFAULT_RAD))
            )
            self._auto_abort_requested = False
            await self._auto_grasp_lift_transaction(
                orient_cap_positive_z=bool(
                    request.get("orient_cap_positive_z", False)
                )
            )
        except BaseException as exc:
            # asyncio cancellation does not derive from Exception on all
            # supported Python versions.  An external acceptance request must
            # always leave a readable result instead of disappearing after it
            # has been renamed to .running.done.
            if not os.path.isfile(AUTO_GRASP_LIFT_RESULT_PATH):
                outer_result = {
                    "status": "EXCEPTION",
                    "classification": "EXTERNAL_GRASP_REQUEST_WRAPPER",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-40:],
                    "stage_saved": False,
                    "ros_used": False,
                    "real_robot_touched": False,
                }
                os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
                with open(AUTO_GRASP_LIFT_RESULT_PATH, "w", encoding="utf-8") as stream:
                    json.dump(outer_result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            carb.log_error(
                "External automatic grasp request stopped before its transaction "
                f"report was written: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
        finally:
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_external_horizontal_cylinder_benchmark(self) -> None:
        """Run the isolated horizontal-cylinder contact benchmark.

        The implementation lives in a separate module so the production Bottle
        workflow stays readable.  It only authors the Session Layer, restores
        every touched runtime value, never saves the Stage, and never uses ROS
        or a real robot.
        """

        running_path = HORIZONTAL_CYLINDER_BENCHMARK_REQUEST_PATH + ".running"
        try:
            os.replace(HORIZONTAL_CYLINDER_BENCHMARK_REQUEST_PATH, running_path)
            request = await self._read_stable_external_request(running_path)
            if request.get("simulation_only") is not True:
                raise RuntimeError(
                    "horizontal Cylinder benchmark must set simulation_only=true"
                )
            if request.get("horizontal_on_table") is not True:
                raise RuntimeError(
                    "horizontal Cylinder benchmark requires horizontal_on_table=true"
                )
            from .horizontal_cylinder_benchmark import run_benchmark

            await run_benchmark(
                self,
                request,
                HORIZONTAL_CYLINDER_BENCHMARK_RESULT_PATH,
            )
        except BaseException as exc:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            with open(
                HORIZONTAL_CYLINDER_BENCHMARK_RESULT_PATH,
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    {
                        "status": "EXCEPTION",
                        "classification": "HORIZONTAL_CYLINDER_BENCHMARK_WRAPPER",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-50:],
                        "stage_saved": False,
                        "ros_used": False,
                        "real_robot_touched": False,
                    },
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")
            carb.log_error(
                "Horizontal Cylinder benchmark failed: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
        finally:
            self._timeline.pause()
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")
            self._auto_task = None
            self._refresh_workflow_ui()

    async def _ensure_initialized_paused(self) -> None:
        app = omni.kit.app.get_app()
        self._timeline.pause()
        if self._timeline.is_stopped():
            self._timeline.play()
            await app.next_update_async()
            self._timeline.pause()
            await app.next_update_async()

    def _bottle_assembly_rigid_apis(self) -> Dict[str, UsdPhysics.RigidBodyAPI]:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        result = {}
        for path in (BOTTLE_PATH, BOTTLE_CAP_PATH, BOTTLE_THREAD_SLIDER_PATH):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Bottle assembly prim is missing: {path}")
            api = UsdPhysics.RigidBodyAPI(prim)
            if not api:
                raise RuntimeError(f"RigidBodyAPI is missing: {path}")
            result[path] = api
        return result

    def _set_bottle_assembly_kinematic(self, enabled: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        rigid_apis = self._bottle_assembly_rigid_apis()
        # A PhysX Joint cannot be instantiated between two kinematic bodies.
        # While the assembly is held for repeatable placement, each component
        # is positioned explicitly and the thread Joints stay disabled.  The
        # locked Joints are restored only after all components become Dynamic.
        if enabled:
            self._set_thread_locked_enabled(False)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for api in rigid_apis.values():
                api.CreateKinematicEnabledAttr().Set(bool(enabled))
        readback = {
            path: bool(api.GetKinematicEnabledAttr().Get())
            for path, api in rigid_apis.items()
        }
        if any(value != bool(enabled) for value in readback.values()):
            raise RuntimeError(f"Bottle assembly Kinematic readback mismatch: {readback}")
        if not enabled:
            self._set_thread_locked_enabled(True)

    def _set_thread_locked_enabled(self, enabled: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in (BOTTLE_THREAD_PRISMATIC_PATH, BOTTLE_THREAD_REVOLUTE_PATH):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"thread Joint is missing: {path}")
                prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(
                    bool(enabled)
                )
            coupling = stage.GetPrimAtPath(BOTTLE_THREAD_COUPLING_PATH)
            if not coupling or not coupling.IsValid():
                raise RuntimeError(f"thread coupling is missing: {BOTTLE_THREAD_COUPLING_PATH}")
            coupling.CreateAttribute(
                "physics:jointEnabled", Sdf.ValueTypeNames.Bool
            ).Set(False)

    def _restore_threaded_locked_startup_state(self) -> Dict[str, object]:
        """Restore the authored Dynamic + locked-THREADED runtime contract."""

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        rigid_apis = self._bottle_assembly_rigid_apis()
        prismatic = stage.GetPrimAtPath(BOTTLE_THREAD_PRISMATIC_PATH)
        revolute = stage.GetPrimAtPath(BOTTLE_THREAD_REVOLUTE_PATH)
        coupling = stage.GetPrimAtPath(BOTTLE_THREAD_COUPLING_PATH)
        joint_scope = stage.GetPrimAtPath(
            BOTTLE_THREAD_PRISMATIC_PATH.rsplit("/", 1)[0]
        )
        for prim, path in (
            (prismatic, BOTTLE_THREAD_PRISMATIC_PATH),
            (revolute, BOTTLE_THREAD_REVOLUTE_PATH),
            (coupling, BOTTLE_THREAD_COUPLING_PATH),
            (joint_scope, str(joint_scope.GetPath()) if joint_scope else "BottleThreadJoints"),
        ):
            if not prim or not prim.IsValid():
                raise RuntimeError(f"thread reset prim is missing: {path}")

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for api in rigid_apis.values():
                api.CreateKinematicEnabledAttr().Set(False)
            for prim in (prismatic, revolute):
                prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(0.0)
                prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(0.0)
                prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            coupling.CreateAttribute(
                "physics:jointEnabled", Sdf.ValueTypeNames.Bool
            ).Set(False)
            angular_drive = UsdPhysics.DriveAPI.Get(revolute, "angular")
            if angular_drive:
                revolute.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            for prim in (joint_scope, coupling):
                prim.SetCustomDataByKey("threadState", "THREADED")
                prim.SetCustomDataByKey("transitionInProgress", False)
                prim.SetCustomDataByKey("tightHoldMode", "LOCKED_JOINT_LIMITS")
                prim.SetCustomDataByKey("tightHoldCalibrationStatus", "NOT_APPLICABLE")
                for key in ("releaseExtensionM", "releaseThresholdM"):
                    prim.ClearCustomDataByKey(key)

        return {
            "thread_state": joint_scope.GetCustomDataByKey("threadState"),
            "bottle_kinematic": bool(rigid_apis[BOTTLE_PATH].GetKinematicEnabledAttr().Get()),
            "cap_kinematic": bool(rigid_apis[BOTTLE_CAP_PATH].GetKinematicEnabledAttr().Get()),
            "slider_kinematic": bool(
                rigid_apis[BOTTLE_THREAD_SLIDER_PATH].GetKinematicEnabledAttr().Get()
            ),
            "prismatic_enabled": bool(
                prismatic.GetAttribute("physics:jointEnabled").Get()
            ),
            "prismatic_limits_m": [
                float(prismatic.GetAttribute(name).Get())
                for name in ("physics:lowerLimit", "physics:upperLimit")
            ],
            "revolute_enabled": bool(
                revolute.GetAttribute("physics:jointEnabled").Get()
            ),
            "revolute_limits_deg": [
                float(revolute.GetAttribute(name).Get())
                for name in ("physics:lowerLimit", "physics:upperLimit")
            ],
            "coupling_enabled": bool(
                coupling.GetAttribute("physics:jointEnabled").Get()
            ),
            "angular_drive_present": bool(UsdPhysics.DriveAPI.Get(revolute, "angular")),
        }

    def _clear_bottle_visible_pose_overrides(self) -> None:
        """Let Dynamic PhysX motion drive the viewport instead of session Xforms."""

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in (
                BOTTLE_PATH,
                BOTTLE_CAP_PATH,
                BOTTLE_THREAD_SLIDER_PATH,
            ):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"Bottle component is missing: {path}")
                for attribute_name in (
                    "xformOp:translate",
                    "xformOp:orient",
                    "xformOpOrder",
                ):
                    attribute = prim.GetAttribute(attribute_name)
                    if attribute and attribute.IsValid():
                        attribute.Clear()

    def _set_bottle_tensor_poses(
        self, component_poses: Dict[str, Tuple[np.ndarray, np.ndarray]], name_prefix: str
    ) -> Dict[str, SingleRigidPrim]:
        bodies: Dict[str, SingleRigidPrim] = {}
        for index, path in enumerate(
            (BOTTLE_PATH, BOTTLE_CAP_PATH, BOTTLE_THREAD_SLIDER_PATH)
        ):
            if path not in component_poses:
                raise RuntimeError(f"missing requested Bottle component pose: {path}")
            position, orientation = component_poses[path]
            body = SingleRigidPrim(
                path,
                name=f"{name_prefix}_{index}",
                reset_xform_properties=False,
            )
            body.initialize()
            body.set_world_pose(
                position=np.asarray(position, dtype=np.float64),
                orientation=_quat_normalize(orientation),
            )
            bodies[path] = body
        return bodies

    def _author_bottle_visible_poses(
        self, component_poses: Dict[str, Tuple[np.ndarray, np.ndarray]], name_prefix: str
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for index, path in enumerate(
                (BOTTLE_PATH, BOTTLE_CAP_PATH, BOTTLE_THREAD_SLIDER_PATH)
            ):
                if path not in component_poses:
                    raise RuntimeError(f"missing visible Bottle component pose: {path}")
                position, orientation = component_poses[path]
                visible_prim = SingleXFormPrim(
                    path,
                    name=f"{name_prefix}_{index}",
                    reset_xform_properties=False,
                )
                visible_prim.set_world_pose(
                    position=np.asarray(position, dtype=np.float64),
                    orientation=_quat_normalize(orientation),
                )

    def _bottle_component_poses_from_center(
        self, center_world_m: np.ndarray, yaw_rad: float
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        yaw_orientation = np.asarray(
            [math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)],
            dtype=np.float64,
        )
        orientation = _quat_multiply(
            yaw_orientation, CANONICAL_BOTTLE_ORIENTATION_WXYZ
        )
        rotation = _quat_to_rotation(orientation)
        return {
            BOTTLE_PATH: (
                center_world_m + rotation @ BOTTLE_ROOT_FROM_CENTER_LOCAL_M,
                orientation,
            ),
            BOTTLE_CAP_PATH: (
                center_world_m + rotation @ CAP_ROOT_FROM_CENTER_LOCAL_M,
                orientation,
            ),
            BOTTLE_THREAD_SLIDER_PATH: (
                center_world_m + rotation @ CAP_ROOT_FROM_CENTER_LOCAL_M,
                orientation,
            ),
        }

    async def _place_bottle_assembly(
        self, center_world_m: np.ndarray, yaw_rad: float
    ) -> Dict[str, object]:
        app = omni.kit.app.get_app()
        self._timeline.pause()
        center_world_m = np.asarray(center_world_m, dtype=np.float64)
        if center_world_m.shape != (3,) or not np.all(np.isfinite(center_world_m)):
            raise RuntimeError(f"invalid Bottle center: {center_world_m}")
        if not RANDOM_BOTTLE_X_RANGE_M[0] <= float(center_world_m[0]) <= RANDOM_BOTTLE_X_RANGE_M[1]:
            raise RuntimeError("Bottle center x is outside the configured random region")
        if not RANDOM_BOTTLE_Y_RANGE_M[0] <= float(center_world_m[1]) <= RANDOM_BOTTLE_Y_RANGE_M[1]:
            raise RuntimeError("Bottle center y is outside the configured random region")

        self._set_bottle_assembly_kinematic(True)
        requested = self._bottle_component_poses_from_center(center_world_m, yaw_rad)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")

        # SingleRigidPrim.set_world_pose() updates the PhysX tensor view, but
        # while Timeline is Paused that pose is not guaranteed to propagate to
        # the USD/Fabric transform consumed by the viewport.  Author the same
        # world pose into the non-persistent session layer first so the visible
        # meshes move immediately, then mirror it into PhysX below.  All three
        # rigid components must move together; moving only their parent Xform
        # would allow stale child physics state to overwrite the placement on
        # the next Play.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for index, (path, (position, orientation)) in enumerate(requested.items()):
                visible_prim = SingleXFormPrim(
                    path,
                    name=f"random_bottle_visible_component_{index}",
                    reset_xform_properties=False,
                )
                visible_prim.set_world_pose(
                    position=position,
                    orientation=orientation,
                )
        bodies: Dict[str, SingleRigidPrim] = {}
        for index, (path, (position, orientation)) in enumerate(requested.items()):
            body = SingleRigidPrim(
                path,
                name=f"random_bottle_component_{index}",
                reset_xform_properties=False,
            )
            body.initialize()
            body.set_world_pose(position=position, orientation=orientation)
            bodies[path] = body
        for _ in range(8):
            await app.next_update_async()

        readback = {}
        for path, body in bodies.items():
            position, orientation = body.get_world_pose()
            expected_position, expected_orientation = requested[path]
            position = np.asarray(position, dtype=np.float64)
            orientation = _quat_normalize(orientation)
            position_error_m = float(np.linalg.norm(position - expected_position))
            orientation_error_rad = _quat_angle(orientation, expected_orientation)
            if position_error_m > 1.0e-5 or orientation_error_rad > math.radians(0.01):
                raise RuntimeError(
                    f"Bottle assembly pose readback failed for {path}: "
                    f"{position_error_m * 1000.0:.4f} mm / "
                    f"{math.degrees(orientation_error_rad):.6f} deg"
                )
            usd_position, usd_orientation = get_world_pose(path)
            fabric_position, fabric_orientation = get_world_pose(path, fabric=True)
            usd_position = np.asarray(usd_position, dtype=np.float64)
            usd_orientation = _quat_normalize(usd_orientation)
            fabric_position = np.asarray(fabric_position, dtype=np.float64)
            fabric_orientation = _quat_normalize(fabric_orientation)
            usd_position_error_m = float(
                np.linalg.norm(usd_position - expected_position)
            )
            usd_orientation_error_rad = _quat_angle(
                usd_orientation, expected_orientation
            )
            fabric_position_error_m = float(
                np.linalg.norm(fabric_position - expected_position)
            )
            fabric_orientation_error_rad = _quat_angle(
                fabric_orientation, expected_orientation
            )
            if (
                usd_position_error_m > 1.0e-5
                or usd_orientation_error_rad > math.radians(0.01)
            ):
                raise RuntimeError(
                    f"visible USD pose readback failed for {path}: "
                    f"{usd_position_error_m * 1000.0:.4f} mm / "
                    f"{math.degrees(usd_orientation_error_rad):.6f} deg"
                )
            if (
                fabric_position_error_m > 1.0e-5
                or fabric_orientation_error_rad > math.radians(0.01)
            ):
                raise RuntimeError(
                    f"visible Fabric pose readback failed for {path}: "
                    f"{fabric_position_error_m * 1000.0:.4f} mm / "
                    f"{math.degrees(fabric_orientation_error_rad):.6f} deg"
                )
            readback[path] = {
                "position_m": position.tolist(),
                "orientation_wxyz": orientation.tolist(),
                "physx_tensor_position_error_m": position_error_m,
                "physx_tensor_orientation_error_rad": orientation_error_rad,
                "usd_visible_position_m": usd_position.tolist(),
                "usd_visible_orientation_wxyz": usd_orientation.tolist(),
                "usd_visible_position_error_m": usd_position_error_m,
                "usd_visible_orientation_error_rad": usd_orientation_error_rad,
                "fabric_visible_position_m": fabric_position.tolist(),
                "fabric_visible_orientation_wxyz": fabric_orientation.tolist(),
                "fabric_visible_position_error_m": fabric_position_error_m,
                "fabric_visible_orientation_error_rad": fabric_orientation_error_rad,
            }
        return {
            "center_world_m": center_world_m.tolist(),
            "yaw_deg": math.degrees(yaw_rad),
            "component_readback": readback,
            "assembly_kinematic": True,
            "thread_prismatic_enabled": False,
            "thread_revolute_enabled": False,
            "thread_coupling_enabled": False,
        }

    async def _prepare_robot_for_automatic_grasp(
        self, update_grasp_world_pose: bool = True
    ) -> None:
        await self._ensure_initialized_paused()

        # A failed loaded trial can leave the Bottle between the fingers and
        # leave active contact pairs in the event tracker.  Opening the gripper
        # in that state can push the arm far away from its drive targets before
        # the Sleep reset begins.  Preserve the requested test pose, park the
        # complete kinematic Bottle assembly well above the workcell, then
        # release/reset the robot without any Bottle contact.
        if self._last_random_bottle_pose is None:
            target_center = CANONICAL_BOTTLE_CENTER_WORLD_M.copy()
            target_yaw_rad = 0.0
            target_source = "CANONICAL_DEFAULT_AFTER_CLEAN_START"
        else:
            target_center = np.asarray(
                self._last_random_bottle_pose["center_world_m"], dtype=np.float64
            )
            target_yaw_rad = math.radians(
                float(self._last_random_bottle_pose["yaw_deg"])
            )
            target_source = str(
                self._last_random_bottle_pose.get("source", "VERIFIED_PLACEMENT")
            )
        parked = await self._place_bottle_assembly(
            np.asarray([0.0, 0.0, 0.75], dtype=np.float64), 0.0
        )
        self._ensure_grasp_contact_monitor()
        await self._open_left_gripper_transaction(require_arm_stationary=False)
        await self._reset_left_sleep_from_button()
        restored = await self._place_bottle_assembly(target_center, target_yaw_rad)
        restored["source"] = target_source
        self._last_random_bottle_pose = restored
        # Drop both the stale pre-recovery pairs and transient events generated
        # while teleporting the kinematic assembly.  Any contact reported from
        # HOVER onward is therefore part of the current transaction.
        self._ensure_grasp_contact_monitor()
        self._last_auto_recovery = {
            "status": "PASS",
            "park_center_world_m": parked["center_world_m"],
            "restored_center_world_m": restored["center_world_m"],
            "restored_yaw_deg": restored["yaw_deg"],
            "restored_source": target_source,
            "stale_contact_pairs_cleared": True,
            "robot_reset_to_sleep_after_parking": True,
        }
        self._load_left_arm()
        self._sync_base_pose()
        if not self._validate_alignment():
            raise RuntimeError(
                f"automatic preparation alignment failed: "
                f"{self._last_position_error * 1000.0:.3f} mm / "
                f"{math.degrees(self._last_orientation_error):.3f} deg"
            )
        self._load_bottle_grasp(update_world_pose=update_grasp_world_pose)
        self._create_target_at_current_ee()
        self._enable_follow()

    async def _execute_current_planned_route(
        self,
        route_name: str,
        timeout_updates: int = 3000,
        monitor_loaded_grasp: bool = False,
        bottle_body=None,
    ) -> Dict[str, object]:
        app = omni.kit.app.get_app()
        if not self._hover_plan_positions or not self._active_waypoint.endswith("/ PLAN READY"):
            raise RuntimeError(f"{route_name} has no ready prevalidated route")
        # Render updates can run faster than the 50 Hz control clock.  Scale
        # the watchdog with the prevalidated sample count so long, gentle
        # loaded routes are not aborted merely because 3000 render frames
        # elapse before all control references have been issued.
        timeout_updates = max(
            int(timeout_updates), 4 * len(self._hover_plan_positions)
        )
        planned_metrics = dict(self._hover_plan_metrics)
        self._timeline.play()
        updates = 0
        bilateral_samples = 0
        bilateral_bad_streak = 0
        mimic_bad_streak = 0
        maximum_mimic_residual_m = 0.0
        minimum_bottle_bottom_z_m = float("inf")
        loaded_start_bottle_z_m = None
        loaded_start_cap_axis = None
        rotation_start_bottle_lift_m = None
        minimum_bottle_lift_after_rotation_start_m = float("inf")
        is_vertical_loaded_route = (
            "ROTATE CAP TO +Z" in route_name
            or route_name.startswith("AUTO SAFE LIFT HORIZONTAL")
            or route_name.startswith("AUTO TRANSFER HORIZONTAL TO CENTER HOVER")
        )
        loaded_mimic_gate_m = (
            VERTICAL_LOADED_MIMIC_RESIDUAL_GATE_M
            if is_vertical_loaded_route
            else GRIPPER_DYNAMIC_MIMIC_RESIDUAL_GATE_M
        )
        if monitor_loaded_grasp:
            if bottle_body is None:
                raise RuntimeError("loaded route monitor requires a Bottle tensor body")
            loaded_start_position, loaded_start_orientation = bottle_body.get_world_pose()
            loaded_start_bottle_z_m = float(
                np.asarray(loaded_start_position, dtype=np.float64)[2]
            )
            loaded_start_cap_axis = (
                _quat_to_rotation(_quat_normalize(loaded_start_orientation))[:, 2]
            )
        try:
            for updates in range(1, timeout_updates + 1):
                await app.next_update_async()
                if self._auto_abort_requested:
                    raise RuntimeError("automatic Bottle task aborted by operator")
                if monitor_loaded_grasp:
                    if bottle_body is None:
                        raise RuntimeError("loaded route monitor requires a Bottle tensor body")
                    if self._grasp_nonfinger_contact:
                        raise RuntimeError(
                            f"non-finger robot geometry contacted Bottle during {route_name}"
                        )
                    bilateral = bool(
                        self._grasp_left_contact and self._grasp_right_contact
                    )
                    bilateral_samples += int(bilateral)
                    bilateral_bad_streak = 0 if bilateral else bilateral_bad_streak + 1
                    _, left_index, right_index, positions = self._get_gripper_state()
                    mimic_residual_m = abs(
                        float(positions[left_index]) + float(positions[right_index])
                    )
                    maximum_mimic_residual_m = max(
                        maximum_mimic_residual_m, mimic_residual_m
                    )
                    mimic_bad_streak = (
                        mimic_bad_streak + 1
                        if mimic_residual_m > loaded_mimic_gate_m
                        else 0
                    )
                    bottle_position, bottle_orientation = bottle_body.get_world_pose()
                    bottle_position = np.asarray(bottle_position, dtype=np.float64)
                    minimum_bottle_bottom_z_m = min(
                        minimum_bottle_bottom_z_m,
                        float(bottle_position[2]),
                    )
                    cap_axis_now = _quat_to_rotation(
                        _quat_normalize(bottle_orientation)
                    )[:, 2]
                    cap_rotation_from_start_rad = math.acos(
                        float(
                            np.clip(
                                np.dot(cap_axis_now, loaded_start_cap_axis),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    if (
                        rotation_start_bottle_lift_m is None
                        and cap_rotation_from_start_rad >= math.radians(2.0)
                    ):
                        rotation_start_bottle_lift_m = float(
                            bottle_position[2] - loaded_start_bottle_z_m
                        )
                    if rotation_start_bottle_lift_m is not None:
                        minimum_bottle_lift_after_rotation_start_m = min(
                            minimum_bottle_lift_after_rotation_start_m,
                            float(bottle_position[2] - loaded_start_bottle_z_m),
                        )
                    if bilateral_bad_streak >= 10:
                        raise RuntimeError(
                            f"bilateral finger contact was lost during {route_name}"
                        )
                    if mimic_bad_streak >= GRIPPER_DYNAMIC_MIMIC_BAD_STEPS:
                        raise RuntimeError(
                            f"Mimic residual exceeded the structural gate during {route_name}"
                        )
                    if (
                        minimum_bottle_bottom_z_m
                        < VERTICAL_LIFT_MIN_BOTTLE_BOTTOM_Z_M - 0.005
                    ):
                        raise RuntimeError(
                            f"Bottle bottom swept too close to the table during {route_name}: "
                            f"z={minimum_bottle_bottom_z_m:.4f} m"
                        )
                if self._active_waypoint.endswith("/ REACHED"):
                    break
            else:
                raise RuntimeError(
                    f"{route_name} timed out after {timeout_updates} render updates"
                )
        finally:
            self._timeline.pause()
        for _ in range(3):
            await app.next_update_async()
        if not self._active_waypoint.endswith("/ REACHED"):
            raise RuntimeError(f"{route_name} did not reach its final reference")
        result = {
            "route_name": route_name,
            "render_updates": updates,
            "physics_elapsed_s": float(self._hover_plan_elapsed_s),
            "metrics": planned_metrics,
        }
        if monitor_loaded_grasp:
            bilateral_fraction = (
                float(bilateral_samples) / float(updates) if updates else 0.0
            )
            if bilateral_fraction < 0.8:
                raise RuntimeError(
                    f"{route_name} bilateral-contact fraction was only "
                    f"{bilateral_fraction:.3f}"
                )
            if not (self._grasp_left_contact and self._grasp_right_contact):
                raise RuntimeError(f"{route_name} did not end with bilateral contact")
            requested_rotation_clearance_m = float(
                planned_metrics.get(
                    "rotation_clearance_m",
                    VERTICAL_LIFT_ROTATION_START_CLEARANCE_M,
                )
            )
            rotation_gate_ok = (
                rotation_start_bottle_lift_m is not None
                and (
                    (
                        requested_rotation_clearance_m >= 0.100
                        and requested_rotation_clearance_m - 0.010
                        <= rotation_start_bottle_lift_m
                        <= requested_rotation_clearance_m + 0.020
                        and minimum_bottle_lift_after_rotation_start_m
                        >= requested_rotation_clearance_m - 0.070
                    )
                    or (
                        requested_rotation_clearance_m < 0.100
                        and rotation_start_bottle_lift_m >= -0.030
                        and minimum_bottle_lift_after_rotation_start_m >= -0.080
                    )
                )
            )
            if "ROTATE CAP TO +Z" in route_name and not rotation_gate_ok:
                raise RuntimeError(
                    f"{route_name} violated the lift-before-rotate gate: "
                    f"rotation_start={rotation_start_bottle_lift_m} m, "
                    f"minimum_after_start={minimum_bottle_lift_after_rotation_start_m} m"
                )
            result["loaded_grasp_monitor"] = {
                "bilateral_contact_fraction": bilateral_fraction,
                "maximum_mimic_residual_m": maximum_mimic_residual_m,
                "mimic_residual_gate_m": loaded_mimic_gate_m,
                "minimum_bottle_bottom_z_m": minimum_bottle_bottom_z_m,
                "rotation_start_bottle_lift_m": rotation_start_bottle_lift_m,
                "minimum_bottle_lift_after_rotation_start_m": (
                    minimum_bottle_lift_after_rotation_start_m
                ),
                "final_left_contact": bool(self._grasp_left_contact),
                "final_right_contact": bool(self._grasp_right_contact),
                "nonfinger_contact": bool(self._grasp_nonfinger_contact),
            }
        return result

    async def _reset_bottle_initial_pose_transaction(self) -> None:
        result: Dict[str, object] = {
            "status": "STARTED",
            "classification": "RESET_BOTTLE_ASSEMBLY_TO_CANONICAL_STARTUP_POSE",
            "stage_saved": False,
            "arm_commanded": False,
            "gripper_commanded": False,
            "ros_used": False,
            "real_robot_touched": False,
        }
        app = omni.kit.app.get_app()
        try:
            if self._gripper_task is not None and not self._gripper_task.done():
                raise RuntimeError("an active gripper task must finish before Bottle reset")
            await self._ensure_initialized_paused()
            self._clear_hover_plan()
            requested = self._bottle_component_poses_from_center(
                CANONICAL_BOTTLE_CENTER_WORLD_M, 0.0
            )
            await self._place_bottle_assembly(
                CANONICAL_BOTTLE_CENTER_WORLD_M, 0.0
            )
            self._clear_bottle_visible_pose_overrides()
            bodies = self._set_bottle_tensor_poses(
                requested, "reset_bottle_initial_component"
            )
            # Velocity writes are only legal for Dynamic PhysX bodies. Restore
            # the Dynamic + locked-THREADED contract before clearing any stale
            # motion left by the previous grasp trial.
            threaded_readback = self._restore_threaded_locked_startup_state()
            zero_velocity = np.zeros(3, dtype=np.float32)
            for body in bodies.values():
                body.set_linear_velocity(zero_velocity)
                body.set_angular_velocity(zero_velocity)

            # Publish the Dynamic bodies and locked joints, then return to a
            # genuinely Paused state. No arm or gripper target is changed.
            self._timeline.play()
            for _ in range(12):
                await app.next_update_async()
                if self._auto_abort_requested:
                    raise RuntimeError("Bottle reset aborted by operator")
            self._timeline.pause()
            for _ in range(4):
                await app.next_update_async()

            component_readback: Dict[str, object] = {}
            maximum_position_error_m = 0.0
            maximum_orientation_error_rad = 0.0
            for path, body in bodies.items():
                position, orientation = body.get_world_pose()
                position = np.asarray(position, dtype=np.float64)
                orientation = _quat_normalize(orientation)
                expected_position, expected_orientation = requested[path]
                position_error_m = float(np.linalg.norm(position - expected_position))
                orientation_error_rad = _quat_angle(orientation, expected_orientation)
                maximum_position_error_m = max(
                    maximum_position_error_m, position_error_m
                )
                maximum_orientation_error_rad = max(
                    maximum_orientation_error_rad, orientation_error_rad
                )
                component_readback[path] = {
                    "position_m": position.tolist(),
                    "orientation_wxyz": orientation.tolist(),
                    "position_error_m": position_error_m,
                    "orientation_error_rad": orientation_error_rad,
                }
            if maximum_position_error_m > 0.001:
                raise RuntimeError(
                    "Bottle startup position readback exceeded 1 mm: "
                    f"{maximum_position_error_m * 1000.0:.3f} mm"
                )
            if maximum_orientation_error_rad > math.radians(0.5):
                raise RuntimeError(
                    "Bottle startup orientation readback exceeded 0.5 deg: "
                    f"{math.degrees(maximum_orientation_error_rad):.3f} deg"
                )
            checks = {
                "all_bodies_dynamic": not any(
                    bool(threaded_readback[key])
                    for key in (
                        "bottle_kinematic",
                        "cap_kinematic",
                        "slider_kinematic",
                    )
                ),
                "thread_state_is_threaded": threaded_readback["thread_state"]
                == "THREADED",
                "prismatic_locked_zero": bool(
                    threaded_readback["prismatic_enabled"]
                )
                and threaded_readback["prismatic_limits_m"] == [0.0, 0.0],
                "revolute_locked_zero": bool(threaded_readback["revolute_enabled"])
                and threaded_readback["revolute_limits_deg"] == [0.0, 0.0],
                "coupling_disabled": not bool(threaded_readback["coupling_enabled"]),
                "no_angular_drive": not bool(
                    threaded_readback["angular_drive_present"]
                ),
                "timeline_paused": not self._timeline.is_playing(),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Bottle startup-state readback failed: {checks}")

            # Treat the canonical reset pose as a verified grasp source too;
            # the automatic button must work after RESET, not only after a
            # random-placement transaction.
            self._last_random_bottle_pose = {
                "center_world_m": CANONICAL_BOTTLE_CENTER_WORLD_M.tolist(),
                "yaw_deg": 0.0,
                "component_readback": component_readback,
                "assembly_kinematic": False,
                "source": "CANONICAL_RESET_POSE",
            }
            self._grasp_loaded = False
            self._active_waypoint = "BOTTLE INITIAL POSE / robot unchanged"
            result.update(
                {
                    "status": "PASS",
                    "canonical_center_world_m": CANONICAL_BOTTLE_CENTER_WORLD_M.tolist(),
                    "canonical_yaw_deg": 0.0,
                    "component_readback": component_readback,
                    "maximum_position_error_m": maximum_position_error_m,
                    "maximum_orientation_error_rad": maximum_orientation_error_rad,
                    "threaded_readback": threaded_readback,
                    "checks": checks,
                }
            )
            if self._random_test_label is not None:
                self._random_test_label.text = (
                    "INITIAL: center=(0.000, 0.000, 0.034) m, yaw=0.0 deg; "
                    "Bottle assembly=Dynamic; thread=THREADED; arm unchanged."
                )
            self._set_status(
                "Bottle reset PASS: Bottle/Cap/thread Slider returned atomically to the "
                "canonical startup pose; velocities are zero, bodies are Dynamic, thread "
                "joints are locked THREADED, and Timeline is Paused."
            )
        except Exception as exc:
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Bottle initial-pose reset failed safely: {exc}", warn=True)
        finally:
            self._timeline.pause()
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            with open(AUTO_RESET_BOTTLE_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self._auto_task = None
            self._refresh_workflow_ui()

    async def _randomize_bottle_transaction(self) -> None:
        result: Dict[str, object] = {
            "status": "STARTED",
            "classification": "RANDOM_REACHABLE_BOTTLE_POSE",
            "stage_saved": False,
            "ros_used": False,
            "real_robot_touched": False,
        }
        try:
            speed = float(self._planned_joint_step_model.get_value_as_float())
            if not PLANNED_JOINT_STEP_MIN_RAD <= speed <= PLANNED_JOINT_STEP_MAX_RAD:
                raise RuntimeError(
                    f"speed must be {PLANNED_JOINT_STEP_MIN_RAD:.3f}-"
                    f"{PLANNED_JOINT_STEP_MAX_RAD:.3f} rad"
                )
            self._locked_auto_joint_step_rad = speed
            # A previous lifted trial may leave the Bottle tilted.  Load and
            # validate the object-local Grasp Editor record now, but defer its
            # world-pose projection until after this transaction has authored
            # and tensor-verified the new horizontal random Bottle pose.
            await self._prepare_robot_for_automatic_grasp(
                update_grasp_world_pose=False
            )
            attempts = []
            accepted = None
            attempt_limit = (
                1
                if self._requested_random_pose_override is not None
                else RANDOM_BOTTLE_MAX_ATTEMPTS
            )
            for attempt_index in range(1, attempt_limit + 1):
                if self._auto_abort_requested:
                    raise RuntimeError("random Bottle task aborted by operator")
                if self._requested_random_pose_override is not None:
                    center, yaw_rad = self._requested_random_pose_override
                    center = np.asarray(center, dtype=np.float64).copy()
                    yaw_rad = float(yaw_rad)
                else:
                    center = np.asarray(
                        [
                            self._rng.uniform(*RANDOM_BOTTLE_X_RANGE_M),
                            self._rng.uniform(*RANDOM_BOTTLE_Y_RANGE_M),
                            RANDOM_BOTTLE_CENTER_Z_M,
                        ],
                        dtype=np.float64,
                    )
                    yaw_rad = float(self._rng.uniform(-math.pi, math.pi))
                placement = await self._place_bottle_assembly(center, yaw_rad)
                self._clear_hover_plan()
                current_position, current_orientation = self._current_lula_ee_pose()
                self._set_target_from_ee_pose(current_position, current_orientation)
                self._active_waypoint = "current EE"
                try:
                    bottle_readback = placement["component_readback"][BOTTLE_PATH]
                    self._plan_hover_route(
                        bottle_pose=(
                            np.asarray(bottle_readback["position_m"], dtype=np.float64),
                            np.asarray(
                                bottle_readback["orientation_wxyz"], dtype=np.float64
                            ),
                        )
                    )
                    chosen_fraction = float(
                        self._hover_plan_metrics["chosen_axial_fraction_from_bottom"]
                    )
                    chosen_clearance = float(
                        self._hover_plan_metrics["chosen_clearance_m"]
                    )
                    if abs(chosen_fraction - GRASP_AXIAL_FRACTION_FROM_BOTTOM) > 1.0e-9:
                        raise RuntimeError(
                            "random pose HOVER required a fallback axial grasp station"
                        )
                    if abs(chosen_clearance - HOVER_CLEARANCE_M) > 1.0e-9:
                        raise RuntimeError(
                            "random pose HOVER required a fallback clearance"
                        )
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "center_world_m": center.tolist(),
                            "yaw_deg": math.degrees(yaw_rad),
                            "reachable": True,
                        }
                    )
                    accepted = placement
                    break
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "center_world_m": center.tolist(),
                            "yaw_deg": math.degrees(yaw_rad),
                            "reachable": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if accepted is None:
                result["attempts"] = attempts
                raise RuntimeError(
                    f"no reachable random Bottle pose after {attempt_limit} attempts"
                )
            self._clear_hover_plan()
            current_position, current_orientation = self._current_lula_ee_pose()
            self._set_target_from_ee_pose(current_position, current_orientation)
            self._active_waypoint = "RANDOM BOTTLE READY / arm at sleep"
            self._last_random_bottle_pose = accepted
            result.update(
                {
                    "status": "PASS",
                    "configured_joint_step_rad": speed,
                    "control_hz": 1.0 / HOVER_PLAN_CONTROL_PERIOD_S,
                    "accepted_pose": accepted,
                    "attempts": attempts,
                    "timeline_paused": not self._timeline.is_playing(),
                }
            )
            if self._random_test_label is not None:
                center = accepted["center_world_m"]
                self._random_test_label.text = (
                    f"READY: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m, "
                    f"yaw={accepted['yaw_deg']:.1f} deg; arm=sleep; Bottle assembly=Kinematic."
                )
            self._set_status(
                "Random reachable Bottle pose PASS. Arm remains at sleep, gripper is open, "
                "Bottle/Cap/thread assembly is Kinematic, and Timeline is Paused. Click GRASP."
            )
        except Exception as exc:
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Random Bottle placement failed safely: {exc}", warn=True)
        finally:
            self._timeline.pause()
            self._locked_auto_joint_step_rad = None
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            with open(RANDOM_BOTTLE_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self._auto_task = None
            self._refresh_workflow_ui()

    async def _auto_grasp_lift_transaction(
        self, orient_cap_positive_z: bool = False
    ) -> None:
        result: Dict[str, object] = {
            "status": "STARTED",
            "classification": (
                "SLEEP_HOVER_NEAR10_CLOSE_CONTINUOUS_LIFT_ROTATE_CAP_POSITIVE_Z"
                if orient_cap_positive_z
                else "SLEEP_HOVER_NEAR10_CLOSE_LIFT_TO_PLAN_HOVER"
            ),
            "phase_sequence": [
                "SLEEP_TO_PLAN_HOVER",
                "PLAN_HOVER_TO_NEAR_PLUS_10_MM",
                "DYNAMIC_CONTINUOUS_CLOSE",
                (
                    "LIFT_HORIZONTAL_TRANSFER_CENTER_ROTATE_CAP_TO_POSITIVE_Z"
                    if orient_cap_positive_z
                    else "LIFT_TO_PLAN_HOVER"
                ),
            ],
            "stage_saved": False,
            "ros_used": False,
            "real_robot_touched": False,
        }
        app = omni.kit.app.get_app()
        bottle_body = None
        use_center_transfer = False
        self._last_auto_recovery = None
        try:
            speed = float(self._planned_joint_step_model.get_value_as_float())
            if not PLANNED_JOINT_STEP_MIN_RAD <= speed <= PLANNED_JOINT_STEP_MAX_RAD:
                raise RuntimeError(
                    f"speed must be {PLANNED_JOINT_STEP_MIN_RAD:.3f}-"
                    f"{PLANNED_JOINT_STEP_MAX_RAD:.3f} rad"
                )
            self._locked_auto_joint_step_rad = speed
            # The ordinary USD transform may still contain the previous
            # dynamic trial's authored pose.  Automatic grasping projects the
            # Grasp Editor record from the tensor-verified pose captured by the
            # random-placement transaction below, so do not pre-project from
            # stale USD state here.
            await self._prepare_robot_for_automatic_grasp(
                update_grasp_world_pose=False
            )
            routes = []

            random_bottle_pose = None
            if self._last_random_bottle_pose is not None:
                bottle_readback = self._last_random_bottle_pose[
                    "component_readback"
                ][BOTTLE_PATH]
                random_bottle_pose = (
                    np.asarray(bottle_readback["position_m"], dtype=np.float64),
                    np.asarray(
                        bottle_readback["orientation_wxyz"], dtype=np.float64
                    ),
                )
                requested_center = np.asarray(
                    self._last_random_bottle_pose.get(
                        "center_world_m", bottle_readback["position_m"]
                    ),
                    dtype=np.float64,
                )
                # Loaded lift+rotation is contact-sensitive even near the
                # workspace centre.  Always separate vertical Bottle tasks
                # into horizontal lift, high transfer, and centre rotation;
                # the old radial threshold incorrectly sent near-centre poses
                # through the less reliable simultaneous lift+rotate route.
                use_center_transfer = bool(orient_cap_positive_z)
                result["center_transfer_selected"] = use_center_transfer
            self._plan_hover_route(bottle_pose=random_bottle_pose)
            self._planned_route_name = "AUTO HOVER"
            self._active_waypoint = "AUTO HOVER / PLAN READY"
            routes.append(await self._execute_current_planned_route("AUTO HOVER"))
            # The operator-requested automatic approach contains one and only
            # one second motion leg after HOVER.  Its internal 50 Hz samples
            # remain continuous; PREGRASP and the old +80/+40/+20/+0 stops are
            # intentionally absent.
            self._locked_auto_joint_step_rad = min(speed, AUTO_APPROACH_JOINT_STEP_RAD)
            near_route_name = "AUTO NEAR +10 mm"
            self._plan_guided_waypoint_route(
                near_route_name,
                NEAR_CLEARANCE_M,
                bottle_pose=random_bottle_pose,
            )
            routes.append(await self._execute_current_planned_route(near_route_name))
            if self._grasp_nonfinger_contact:
                raise RuntimeError(
                    f"non-finger Bottle contact detected during {near_route_name}"
                )

            self._ensure_grasp_contact_monitor()
            if self._last_random_bottle_pose is None:
                raise RuntimeError("automatic grasp has no verified random Bottle pose")
            release_component_poses: Dict[
                str, Tuple[np.ndarray, np.ndarray]
            ] = {}
            for component_path, component_readback in self._last_random_bottle_pose[
                "component_readback"
            ].items():
                release_component_poses[component_path] = (
                    np.asarray(component_readback["position_m"], dtype=np.float64),
                    np.asarray(
                        component_readback["orientation_wxyz"], dtype=np.float64
                    ),
                )
            # Paused random placement uses strong session-layer Xforms so it is
            # immediately visible.  Remove those opinions before Dynamic Play;
            # otherwise the viewport remains pinned to the table even while the
            # PhysX tensor body is lifted.  Reassert every component tensor pose
            # atomically before the first Dynamic physics update.
            self._clear_bottle_visible_pose_overrides()
            dynamic_component_bodies = self._set_bottle_tensor_poses(
                release_component_poses,
                "auto_dynamic_release_component",
            )
            bottle_body = dynamic_component_bodies[BOTTLE_PATH]
            bottle_before_release_position, bottle_before_release_orientation = (
                bottle_body.get_world_pose()
            )
            self._set_bottle_assembly_kinematic(False)

            articulation, left_index, right_index, close_before_positions = (
                self._get_gripper_state()
            )
            close_arm_before = close_before_positions[: len(ARM_JOINTS)].copy()
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(
                        [LEFT_GRIPPER_MIN_POSITION_M], dtype=np.float32
                    ),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            self._gripper_command_target_m = LEFT_GRIPPER_MIN_POSITION_M
            self._grasp_bilateral_streak = 0
            self._timeline.play()
            dynamic_updates = 0
            dynamic_mimic_bad_streak = 0
            maximum_dynamic_mimic_residual_m = 0.0
            dynamic_close_samples: List[Dict[str, object]] = []
            try:
                for dynamic_updates in range(1, 601):
                    await app.next_update_async()
                    if self._auto_abort_requested:
                        raise RuntimeError("automatic grasp aborted by operator")
                    if self._grasp_nonfinger_contact:
                        raise RuntimeError("non-finger robot geometry contacted Bottle")
                    _, _, _, live_close_positions = self._get_gripper_state()
                    live_mimic_residual_m = abs(
                        float(live_close_positions[left_index])
                        + float(live_close_positions[right_index])
                    )
                    maximum_dynamic_mimic_residual_m = max(
                        maximum_dynamic_mimic_residual_m,
                        live_mimic_residual_m,
                    )
                    if live_mimic_residual_m > GRIPPER_DYNAMIC_MIMIC_RESIDUAL_GATE_M:
                        dynamic_mimic_bad_streak += 1
                    else:
                        dynamic_mimic_bad_streak = 0
                    live_bottle_position, live_bottle_orientation = (
                        bottle_body.get_world_pose()
                    )
                    if dynamic_updates <= 100 or dynamic_updates % 10 == 0:
                        dynamic_close_samples.append(
                            {
                                "update": dynamic_updates,
                                "left_actual_m": float(
                                    live_close_positions[left_index]
                                ),
                                "right_actual_m": float(
                                    live_close_positions[right_index]
                                ),
                                "mimic_residual_m": live_mimic_residual_m,
                                "left_contact": bool(self._grasp_left_contact),
                                "right_contact": bool(self._grasp_right_contact),
                                "bilateral_streak": int(
                                    self._grasp_bilateral_streak
                                ),
                                "bottle_position_m": np.asarray(
                                    live_bottle_position, dtype=np.float64
                                ).tolist(),
                                "bottle_orientation_wxyz": _quat_normalize(
                                    live_bottle_orientation
                                ).tolist(),
                            }
                        )
                    if (
                        dynamic_mimic_bad_streak
                        >= GRIPPER_DYNAMIC_MIMIC_BAD_STEPS
                    ):
                        raise RuntimeError(
                            "Dynamic self-centering Mimic residual exceeded "
                            f"{GRIPPER_DYNAMIC_MIMIC_RESIDUAL_GATE_M * 1000.0:.1f} mm "
                            f"for {GRIPPER_DYNAMIC_MIMIC_BAD_STEPS} consecutive updates"
                        )
                    if self._grasp_bilateral_streak >= GRIPPER_BILATERAL_STABLE_STEPS:
                        break
                else:
                    raise RuntimeError("dynamic self-centering did not reach bilateral contact")
            finally:
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            bottle_grasp_position, bottle_grasp_orientation = bottle_body.get_world_pose()
            _, _, _, close_after_positions = self._get_gripper_state()
            close_arm_change = float(
                np.max(
                    np.abs(
                        close_after_positions[: len(ARM_JOINTS)] - close_arm_before
                    )
                )
            )
            close_readback = {
                "target_m": LEFT_GRIPPER_MIN_POSITION_M,
                "left_actual_m": float(close_after_positions[left_index]),
                "right_actual_m": float(close_after_positions[right_index]),
                "mimic_residual_m": abs(
                    float(close_after_positions[left_index])
                    + float(close_after_positions[right_index])
                ),
                "left_contact": bool(self._grasp_left_contact),
                "right_contact": bool(self._grasp_right_contact),
                "bilateral_stable_steps": int(self._grasp_bilateral_streak),
                "nonfinger_contact": bool(self._grasp_nonfinger_contact),
                "maximum_arm_joint_change_rad": close_arm_change,
                "settle_updates": dynamic_updates,
                "settled_reason": "stable_bilateral_dynamic_contact",
            }
            dynamic_bottle_displacement = (
                np.asarray(bottle_grasp_position, dtype=np.float64)
                - np.asarray(bottle_before_release_position, dtype=np.float64)
            )
            recenter_result = {
                "method": "DYNAMIC_CONTINUOUS_SELF_CENTER",
                "status": "STABLE_BILATERAL_CONTACT",
                "single_continuous_close_command": True,
                "gripper_reopen_count": 0,
                "gripper_lateral_adjustment_count": 0,
                "dynamic_updates": dynamic_updates,
                "dynamic_mimic_residual_gate_m": (
                    GRIPPER_DYNAMIC_MIMIC_RESIDUAL_GATE_M
                ),
                "maximum_dynamic_mimic_residual_m": (
                    maximum_dynamic_mimic_residual_m
                ),
                "bottle_displacement_during_close_m": (
                    dynamic_bottle_displacement.tolist()
                ),
                "bottle_displacement_during_close_norm_m": float(
                    np.linalg.norm(dynamic_bottle_displacement)
                ),
                "samples": dynamic_close_samples,
                "final_close_readback": close_readback,
            }
            before_lift_position = np.asarray(bottle_grasp_position, dtype=np.float64)
            self._locked_auto_joint_step_rad = speed
            if orient_cap_positive_z:
                if use_center_transfer:
                    self._locked_auto_joint_step_rad = min(
                        speed, VERTICAL_CENTER_ROTATION_MAX_JOINT_STEP_RAD
                    )
                    safe_lift_name = "AUTO SAFE LIFT HORIZONTAL +170 mm"
                    self._plan_guided_waypoint_route(
                        safe_lift_name,
                        0.170,
                        bottle_pose=(
                            np.asarray(bottle_grasp_position, dtype=np.float64),
                            _quat_normalize(bottle_grasp_orientation),
                        ),
                    )
                    routes.append(
                        await self._execute_current_planned_route(
                            safe_lift_name,
                            monitor_loaded_grasp=True,
                            bottle_body=bottle_body,
                        )
                    )
                    lifted_position, lifted_orientation = bottle_body.get_world_pose()
                    lifted_position = np.asarray(lifted_position, dtype=np.float64)
                    lifted_orientation = _quat_normalize(lifted_orientation)
                    lifted_center = lifted_position - (
                        _quat_to_rotation(lifted_orientation)
                        @ BOTTLE_ROOT_FROM_CENTER_LOCAL_M
                    )
                    center_offset = np.asarray(
                        [-lifted_center[0], -lifted_center[1], 0.0],
                        dtype=np.float64,
                    )
                    center_transfer_name = "AUTO TRANSFER HORIZONTAL TO CENTER HOVER"
                    self._plan_guided_waypoint_route(
                        center_transfer_name,
                        0.0,
                        bottle_pose=(lifted_position, lifted_orientation),
                        world_position_offset=center_offset,
                        maximum_lateral_deviation_m=0.040,
                    )
                    routes.append(
                        await self._execute_current_planned_route(
                            center_transfer_name,
                            monitor_loaded_grasp=True,
                            bottle_body=bottle_body,
                        )
                    )
                    centered_position, centered_orientation = bottle_body.get_world_pose()
                    lift_route_name = "AUTO ROTATE CAP TO +Z AT CENTER HOVER"
                    self._plan_continuous_lift_rotate_cap_up_route(
                        (
                            np.asarray(centered_position, dtype=np.float64),
                            _quat_normalize(centered_orientation),
                        ),
                        additional_bottle_lift_m=0.0,
                        rotation_clearance_m=0.0,
                        maximum_joint_step_rad=(
                            VERTICAL_CENTER_ROTATION_MAX_JOINT_STEP_RAD
                        ),
                    )
                else:
                    lift_route_name = "AUTO LIFT + ROTATE CAP TO +Z"
                    self._plan_continuous_lift_rotate_cap_up_route(
                        (
                            np.asarray(bottle_grasp_position, dtype=np.float64),
                            _quat_normalize(bottle_grasp_orientation),
                        )
                    )
            else:
                lift_route_name = "AUTO LIFT TO PLAN HOVER"
                self._plan_guided_waypoint_route(
                    lift_route_name,
                    HOVER_CLEARANCE_M,
                    bottle_pose=(
                        np.asarray(bottle_grasp_position, dtype=np.float64),
                        _quat_normalize(bottle_grasp_orientation),
                    ),
                )
            routes.append(
                await self._execute_current_planned_route(
                    lift_route_name,
                    monitor_loaded_grasp=True,
                    bottle_body=bottle_body,
                )
            )
            # The configured 0.03 rad reference cap intentionally makes the
            # lift brisk.  Do not score contact on the first paused frame at
            # the end of that transient.  Hold the final targets and require
            # five consecutive bilateral-contact samples.  Dynamic closure
            # uses its bounded 10 mm structural guard because transient and
            # loaded compliant displacement is part of physical self-centering;
            # after lift, record that displacement rather than misclassifying
            # elastic pad deflection as loss of grasp.
            post_lift_stable_updates = 0
            post_lift_settle_updates = 0
            self._timeline.play()
            try:
                for post_lift_settle_updates in range(1, 301):
                    await app.next_update_async()
                    if self._auto_abort_requested:
                        raise RuntimeError("automatic grasp aborted during post-lift settling")
                    if self._grasp_nonfinger_contact:
                        raise RuntimeError(
                            "non-finger robot geometry contacted Bottle during post-lift settling"
                        )
                    _, settle_left_index, settle_right_index, settle_positions = (
                        self._get_gripper_state()
                    )
                    settle_residual = abs(
                        float(settle_positions[settle_left_index])
                        + float(settle_positions[settle_right_index])
                    )
                    if (
                        self._grasp_left_contact
                        and self._grasp_right_contact
                    ):
                        post_lift_stable_updates += 1
                    else:
                        post_lift_stable_updates = 0
                    if post_lift_stable_updates >= GRIPPER_BILATERAL_STABLE_STEPS:
                        break
                else:
                    raise RuntimeError(
                        "post-lift grasp did not maintain stable bilateral contact"
                    )
            finally:
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            final_component_tensor_poses: Dict[
                str, Tuple[np.ndarray, np.ndarray]
            ] = {}
            for component_path, component_body in dynamic_component_bodies.items():
                component_position, component_orientation = (
                    component_body.get_world_pose()
                )
                final_component_tensor_poses[component_path] = (
                    np.asarray(component_position, dtype=np.float64),
                    _quat_normalize(component_orientation),
                )
            after_lift_position, after_lift_orientation = final_component_tensor_poses[
                BOTTLE_PATH
            ]
            after_lift_position = np.asarray(after_lift_position, dtype=np.float64)
            lift_delta = after_lift_position - before_lift_position
            _, _, _, final_positions = self._get_gripper_state()
            final_mimic_residual = abs(
                float(final_positions[left_index]) + float(final_positions[right_index])
            )
            final_cap_axis_world = (
                _quat_to_rotation(after_lift_orientation)
                @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            )
            final_cap_axis_error_rad = math.acos(
                float(
                    np.clip(
                        np.dot(
                            final_cap_axis_world,
                            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            if float(lift_delta[2]) < (
                VERTICAL_BOTTLE_TARGET_LIFT_M - VERTICAL_BOTTLE_LIFT_TOLERANCE_M
            ):
                raise RuntimeError(
                    f"Bottle lift height was only {lift_delta[2] * 1000.0:.1f} mm; "
                    f"required at least "
                    f"{(VERTICAL_BOTTLE_TARGET_LIFT_M - VERTICAL_BOTTLE_LIFT_TOLERANCE_M) * 1000.0:.1f} mm"
                )
            if (
                orient_cap_positive_z
                and final_cap_axis_error_rad > VERTICAL_LIFT_CAP_AXIS_GATE_RAD
            ):
                raise RuntimeError(
                    "Bottle cap axis did not finish at world +Z: "
                    f"error={math.degrees(final_cap_axis_error_rad):.3f} deg"
                )
            # Timeline is paused now.  Publish the final tensor poses back into
            # the session layer so the viewport and Stage tree visibly remain
            # at the lifted result instead of snapping to the authored startup
            # pose.  A PASS is not reported until all visible component poses
            # match their final physical poses.
            self._author_bottle_visible_poses(
                final_component_tensor_poses,
                "auto_lift_final_visible_component",
            )
            for _ in range(5):
                await app.next_update_async()
            final_visible_readback: Dict[str, object] = {}
            for component_path, (
                expected_visible_position,
                expected_visible_orientation,
            ) in final_component_tensor_poses.items():
                visible_position, visible_orientation = get_world_pose(component_path)
                visible_position = np.asarray(visible_position, dtype=np.float64)
                visible_orientation = _quat_normalize(visible_orientation)
                visible_position_error_m = float(
                    np.linalg.norm(visible_position - expected_visible_position)
                )
                visible_orientation_error_rad = _quat_angle(
                    visible_orientation, expected_visible_orientation
                )
                if (
                    visible_position_error_m > 1.0e-5
                    or visible_orientation_error_rad > math.radians(0.01)
                ):
                    raise RuntimeError(
                        f"final visible lifted pose mismatch for {component_path}: "
                        f"{visible_position_error_m * 1000.0:.4f} mm / "
                        f"{math.degrees(visible_orientation_error_rad):.6f} deg"
                    )
                final_visible_readback[component_path] = {
                    "position_m": visible_position.tolist(),
                    "orientation_wxyz": visible_orientation.tolist(),
                    "position_error_m": visible_position_error_m,
                    "orientation_error_rad": visible_orientation_error_rad,
                }
            result.update(
                {
                    "status": "PASS",
                    "configured_joint_step_rad": speed,
                    "approach_joint_step_rad": min(speed, AUTO_APPROACH_JOINT_STEP_RAD),
                    "control_hz": 1.0 / HOVER_PLAN_CONTROL_PERIOD_S,
                    "automatic_start_recovery": self._last_auto_recovery,
                    "random_pose": self._last_random_bottle_pose,
                    "routes": routes,
                    "close_readback": close_readback,
                    "dynamic_continuous_self_center": recenter_result,
                    "dynamic_self_center_updates": dynamic_updates,
                    "post_lift_settle_updates": post_lift_settle_updates,
                    "post_lift_stable_updates": post_lift_stable_updates,
                    "post_lift_mimic_residual_classification": (
                        "LOADED_COMPLIANT_DEFLECTION_RECORDED_NOT_CLOSE_GATE"
                    ),
                    "bottle_before_release_position_m": np.asarray(
                        bottle_before_release_position, dtype=np.float64
                    ).tolist(),
                    "bottle_before_release_orientation_wxyz": _quat_normalize(
                        bottle_before_release_orientation
                    ).tolist(),
                    "bottle_before_lift_position_m": before_lift_position.tolist(),
                    "bottle_after_lift_position_m": after_lift_position.tolist(),
                    "bottle_after_lift_orientation_wxyz": _quat_normalize(
                        after_lift_orientation
                    ).tolist(),
                    "bottle_lift_delta_m": lift_delta.tolist(),
                    "bottle_cap_axis_after_lift_world": final_cap_axis_world.tolist(),
                    "bottle_cap_axis_error_from_positive_z_rad": (
                        final_cap_axis_error_rad
                    ),
                    "bottle_cap_axis_gate_rad": VERTICAL_LIFT_CAP_AXIS_GATE_RAD,
                    "bottle_cap_positive_z_required": bool(
                        orient_cap_positive_z
                    ),
                    "final_visible_pose_verified": True,
                    "final_visible_component_readback": final_visible_readback,
                    "final_left_contact": bool(self._grasp_left_contact),
                    "final_right_contact": bool(self._grasp_right_contact),
                    "final_nonfinger_contact": bool(self._grasp_nonfinger_contact),
                    "final_mimic_residual_m": final_mimic_residual,
                    "timeline_paused": not self._timeline.is_playing(),
                    "bottle_assembly_dynamic": True,
                }
            )
            self._active_waypoint = (
                "AUTO VERTICAL HOVER / REACHED WITH CAP +Z"
                if orient_cap_positive_z
                else "AUTO PLAN HOVER / REACHED WITH BOTTLE"
            )
            if self._random_test_label is not None:
                self._random_test_label.text = (
                    f"GRASP PASS: Bottle lifted {lift_delta[2] * 1000.0:.1f} mm; "
                    + (
                        f"cap +Z error={math.degrees(final_cap_axis_error_rad):.2f} deg; "
                        if orient_cap_positive_z
                        else "final pose=PLAN HOVER; "
                    )
                    + f"speed={speed:.3f} rad/50 Hz; Timeline Paused."
                )
            self._set_status(
                f"Automatic random Bottle grasp PASS: lifted {lift_delta[2] * 1000.0:.1f} mm "
                + (
                    f"while continuously rotating cap to world +Z "
                    f"(error {math.degrees(final_cap_axis_error_rad):.2f} deg) "
                    if orient_cap_positive_z
                    else "to PLAN HOVER "
                )
                + f"at {speed:.3f} rad / 50 Hz. Timeline Paused; gripper holds its closing target."
            )
        except Exception as exc:
            exception_pose_diagnostic: Dict[str, object] = {}
            try:
                actual_ee_position, actual_ee_orientation = (
                    self._current_lula_ee_pose()
                )
                target_ee_position, target_ee_orientation = self._target_pose()
                exception_pose_diagnostic.update(
                    {
                        "actual_ee_position_m": actual_ee_position.tolist(),
                        "actual_ee_orientation_wxyz": actual_ee_orientation.tolist(),
                        "target_ee_position_m": target_ee_position.tolist(),
                        "target_ee_orientation_wxyz": target_ee_orientation.tolist(),
                        "actual_to_target_position_error_m": (
                            actual_ee_position - target_ee_position
                        ).tolist(),
                        "actual_to_target_orientation_error_rad": _quat_angle(
                            actual_ee_orientation, target_ee_orientation
                        ),
                    }
                )
                if self._grasp_world_position is not None:
                    exception_pose_diagnostic[
                        "computed_grasp_position_m"
                    ] = self._grasp_world_position.tolist()
            except Exception as diagnostic_exc:
                exception_pose_diagnostic["pose_diagnostic_error"] = (
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
            try:
                _, diagnostic_left_index, diagnostic_right_index, diagnostic_positions = (
                    self._get_gripper_state()
                )
                exception_pose_diagnostic.update(
                    {
                        "left_finger_actual_m": float(
                            diagnostic_positions[diagnostic_left_index]
                        ),
                        "right_finger_actual_m": float(
                            diagnostic_positions[diagnostic_right_index]
                        ),
                        "mimic_residual_m": abs(
                            float(diagnostic_positions[diagnostic_left_index])
                            + float(diagnostic_positions[diagnostic_right_index])
                        ),
                    }
                )
            except Exception as diagnostic_exc:
                exception_pose_diagnostic["finger_diagnostic_error"] = (
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
            result.update(
                {
                    "status": "EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-40:],
                    "configured_joint_step_rad": self._locked_auto_joint_step_rad,
                    "automatic_start_recovery": self._last_auto_recovery,
                    "random_pose": self._last_random_bottle_pose,
                    "active_waypoint": self._active_waypoint,
                    "completed_routes": routes if "routes" in locals() else [],
                    "close_readback": close_readback
                    if "close_readback" in locals()
                    else None,
                    "dynamic_continuous_self_center": recenter_result
                    if "recenter_result" in locals()
                    else None,
                    "exception_pose_diagnostic": exception_pose_diagnostic,
                    "left_contact": bool(self._grasp_left_contact),
                    "right_contact": bool(self._grasp_right_contact),
                    "nonfinger_contact": bool(self._grasp_nonfinger_contact),
                    "active_contact_pairs": [
                        {"paths": list(paths), "classification": classification}
                        for paths, classification in self._grasp_contact_pairs.items()
                    ],
                    "recent_contact_paths": [
                        list(paths) for paths in self._recent_contact_paths[-20:]
                    ],
                    "timeline_paused": not self._timeline.is_playing(),
                }
            )
            self._set_status(f"Automatic random Bottle grasp stopped safely: {exc}", warn=True)
        finally:
            self._timeline.pause()
            self._locked_auto_joint_step_rad = None
            result["timeline_paused"] = True
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            timestamped_path = os.path.join(
                DEFAULT_LOG_DIR,
                f"auto_random_grasp_lift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            for output_path in (timestamped_path, AUTO_GRASP_LIFT_RESULT_PATH):
                with open(output_path, "w", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            self._auto_task = None
            self._refresh_workflow_ui()

    def _on_load_bottle_grasp(self) -> None:
        self._run_guarded("Load Bottle Grasp", self._load_bottle_grasp)

    def _load_bottle_grasp(self, update_world_pose: bool = True) -> None:
        if self._timeline.is_playing():
            raise RuntimeError("pause the timeline before loading a grasp definition")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        bottle_prim = stage.GetPrimAtPath(BOTTLE_PATH)
        if not bottle_prim or not bottle_prim.IsValid():
            raise RuntimeError(f"Bottle prim is missing: {BOTTLE_PATH}")

        grasp_path = self._grasp_model.get_value_as_string().strip()
        if not os.path.isfile(grasp_path):
            raise FileNotFoundError(grasp_path)
        with open(grasp_path, "r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict) or payload.get("format") != "isaac_grasp":
            raise ValueError("file is not an isaac_grasp YAML")
        object_frame = payload.get("object_frame") or payload.get("object_frame_link")
        gripper_frame = payload.get("gripper_frame") or payload.get("gripper_frame_link")
        if object_frame != BOTTLE_PATH:
            raise ValueError(f"object frame must be {BOTTLE_PATH}, got {object_frame}")
        if gripper_frame != LEFT_EE_PATH:
            raise ValueError(f"gripper frame must be {LEFT_EE_PATH}, got {gripper_frame}")
        grasps = payload.get("grasps")
        if not isinstance(grasps, dict) or len(grasps) != 1:
            raise ValueError("this guided panel requires exactly one named grasp")
        grasp_name, grasp = next(iter(grasps.items()))
        if not isinstance(grasp, dict):
            raise ValueError("named grasp entry is not a mapping")
        object_position = np.asarray(grasp.get("position"), dtype=np.float64)
        orientation = grasp.get("orientation")
        if object_position.shape != (3,) or not isinstance(orientation, dict):
            raise ValueError("grasp position/orientation fields are malformed")
        xyz = np.asarray(orientation.get("xyz"), dtype=np.float64)
        if xyz.shape != (3,) or "w" not in orientation:
            raise ValueError("grasp orientation must use scalar-first {w, xyz}")
        object_orientation = _quat_normalize(
            np.asarray([float(orientation["w"]), xyz[0], xyz[1], xyz[2]], dtype=np.float64)
        )
        cspace = grasp.get("cspace_position", {})
        pregrasp = grasp.get("pregrasp_cspace_position", {})
        if set(cspace) != {"left_finger"} or set(pregrasp) != {"left_finger"}:
            raise ValueError("grasp must expose only the active left_finger DOF")

        self._grasp_name = str(grasp_name)
        self._grasp_object_position = object_position + GRASP_OBJECT_LOCAL_CORRECTION_M
        expected_axial_position = BOTTLE_LENGTH_M * GRASP_AXIAL_FRACTION_FROM_BOTTOM
        if abs(float(self._grasp_object_position[2]) - expected_axial_position) > 1e-9:
            raise ValueError(
                "corrected Bottle grasp station must be exactly L_bot/3 from the bottom: "
                f"{self._grasp_object_position[2]:.9f} m != {expected_axial_position:.9f} m"
            )
        self._grasp_object_orientation = object_orientation
        self._grasp_closed_position = float(cspace["left_finger"])
        self._grasp_preopen_position = float(pregrasp["left_finger"])
        if update_world_pose:
            self._update_grasp_world_pose()
        self._grasp_loaded = True
        self._active_waypoint = "none"
        confidence = float(grasp.get("confidence", 0.0))
        self._grasp_label.text = (
            f"Loaded: {self._grasp_name} (confidence={confidence:.3f}; diagnostic seed)\n"
            f"Corrected O->G p=({self._grasp_object_position[0]:.6f}, "
            f"{self._grasp_object_position[1]:.6f}, {self._grasp_object_position[2]:.6f}) m\n"
            f"Axial station: L_bot/3 from bottom = {expected_axial_position:.6f} m\n"
            f"Finger metadata: preopen={self._grasp_preopen_position:.6f} m, "
            f"authored contact={self._grasp_closed_position:.6f} m; this panel will not command either value."
        )
        self._set_status(
            "Bottle grasp frame contract PASS. Applied the audited local correction "
            "(-5.5, -1.5, -10.0) mm. Next create the Target at the current EE and arm IK Follow."
        )

    def _update_grasp_world_pose(
        self,
        bottle_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._grasp_object_position is None or self._grasp_object_orientation is None:
            raise RuntimeError("load the Bottle grasp first")
        if bottle_pose is None:
            bottle_position, bottle_orientation = get_world_pose(BOTTLE_PATH)
        else:
            bottle_position, bottle_orientation = bottle_pose
        bottle_position = np.asarray(bottle_position, dtype=np.float64)
        bottle_orientation = _quat_normalize(bottle_orientation)
        self._grasp_world_position = (
            bottle_position + _quat_to_rotation(bottle_orientation) @ self._grasp_object_position
        )
        # Position remains object-local, but the guided top-down orientation is
        # a world-frame task constraint.  The orange grasp Target is exactly
        # world aligned (Target +Z == world +Z); Lula receives the equivalent
        # native EE orientation through the fixed Target->EE transform.
        bottle_axis_world = _quat_to_rotation(bottle_orientation) @ np.asarray(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        horizontal_norm = float(np.linalg.norm(bottle_axis_world[:2]))
        if horizontal_norm < 0.999:
            raise RuntimeError(
                "random Bottle grasp requires a horizontal Bottle axis"
            )
        bottle_yaw_rad = math.atan2(
            float(bottle_axis_world[1]), float(bottle_axis_world[0])
        )
        bottle_aligned_target_orientation = np.asarray(
            [
                math.cos(0.5 * bottle_yaw_rad),
                0.0,
                0.0,
                math.sin(0.5 * bottle_yaw_rad),
            ],
            dtype=np.float64,
        )
        self._grasp_world_orientation = _quat_multiply(
            bottle_aligned_target_orientation, TARGET_TO_EE_ORIENTATION_WXYZ
        )
        return self._grasp_world_position.copy(), self._grasp_world_orientation.copy()

    def _target_orientation_from_ee(self, ee_orientation: np.ndarray) -> np.ndarray:
        """Convert native Lula EE orientation into the visible grasp Target frame."""

        return _quat_multiply(
            ee_orientation, _quat_conjugate(TARGET_TO_EE_ORIENTATION_WXYZ)
        )

    def _ee_orientation_from_target(self, target_orientation: np.ndarray) -> np.ndarray:
        """Convert the visible grasp Target orientation into native Lula EE orientation."""

        return _quat_multiply(target_orientation, TARGET_TO_EE_ORIENTATION_WXYZ)

    def _set_target_from_ee_pose(self, position: np.ndarray, ee_orientation: np.ndarray) -> None:
        if self._target is None:
            raise RuntimeError("extension target does not exist")
        self._target.set_world_pose(
            position=np.asarray(position, dtype=np.float64),
            orientation=self._target_orientation_from_ee(ee_orientation),
        )

    def _require_guided_target_ready(self) -> None:
        if self._timeline.is_stopped():
            raise RuntimeError("timeline is stopped; press Play once, then Pause")
        if self._timeline.is_playing():
            raise RuntimeError("pause the timeline before changing a guided waypoint")
        if not self._grasp_loaded:
            raise RuntimeError("load and validate the Bottle grasp first")
        if self._target is None or not is_prim_path_valid(TARGET_PATH):
            raise RuntimeError("create the extension Target at the current EE first")
        if not self._follow_enabled:
            raise RuntimeError("arm IK Follow while the Target is still at the current EE")

    def _on_set_grasp_waypoint(self, name: str, clearance_m: float) -> None:
        self._run_guarded(
            f"Set {name} waypoint", lambda: self._set_grasp_waypoint(name, clearance_m)
        )

    def _clear_hover_plan(self) -> None:
        self._hover_plan_positions = []
        self._hover_plan_index = 0
        self._hover_plan_goal_position = None
        self._hover_plan_goal_orientation = None
        self._hover_plan_metrics = {}
        self._hover_plan_elapsed_s = 0.0
        self._hover_reached_reported = False
        self._planned_route_name = ""

    def _configured_planned_joint_step_rad(self) -> float:
        value = (
            float(self._locked_auto_joint_step_rad)
            if self._locked_auto_joint_step_rad is not None
            else float(self._planned_joint_step_model.get_value_as_float())
        )
        if not PLANNED_JOINT_STEP_MIN_RAD <= value <= PLANNED_JOINT_STEP_MAX_RAD:
            raise RuntimeError(
                "planned joint step must be within "
                f"[{PLANNED_JOINT_STEP_MIN_RAD:.3f}, {PLANNED_JOINT_STEP_MAX_RAD:.3f}] rad; "
                f"got {value:.6f}"
            )
        return value

    def _on_plan_hover_route(self) -> None:
        self._run_guarded("Plan HOVER route", self._plan_hover_route)

    def _plan_hover_route(
        self,
        bottle_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Solve HOVER once and validate a continuous joint-space route without moving."""

        self._require_guided_target_ready()
        if self._lula is None or self._art_ik is None:
            raise RuntimeError("load the left arm first")
        grasp_position, grasp_orientation = self._update_grasp_world_pose(bottle_pose)
        subset = self._art_ik.get_joints_subset()
        current = np.asarray(subset.get_joint_positions(), dtype=np.float64)
        joint_step_rad = self._configured_planned_joint_step_rad()
        if current.shape != (len(ARM_JOINTS),):
            raise RuntimeError(f"expected six current arm joints, got {current.shape}")

        goal = None
        goal_position = None
        chosen_clearance = None
        chosen_warm_start = None
        chosen_axial_fraction = None
        chosen_target_yaw_deg = None
        warm_starts = [("current", current)] + [
            (f"validated_top_down_{index + 1}", seed)
            for index, seed in enumerate(TOP_DOWN_IK_WARM_STARTS)
        ]
        if bottle_pose is None:
            bottle_position, bottle_orientation = get_world_pose(BOTTLE_PATH)
        else:
            bottle_position, bottle_orientation = bottle_pose
        bottle_position = np.asarray(bottle_position, dtype=np.float64)
        bottle_orientation = _quat_normalize(bottle_orientation)
        bottle_rotation = _quat_to_rotation(bottle_orientation)
        bottle_axis_world = bottle_rotation @ np.asarray(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        bottle_yaw_rad = math.atan2(
            float(bottle_axis_world[1]), float(bottle_axis_world[0])
        )
        target_yaw_candidates = []
        for yaw_offset_rad in (0.0, math.pi):
            candidate_yaw_rad = bottle_yaw_rad + yaw_offset_rad
            wrapped_yaw_deg = (
                math.degrees(candidate_yaw_rad) + 180.0
            ) % 360.0 - 180.0
            target_yaw_candidates.append(
                (
                    wrapped_yaw_deg,
                    np.asarray(
                        [
                            math.cos(0.5 * candidate_yaw_rad),
                            0.0,
                            0.0,
                            math.sin(0.5 * candidate_yaw_rad),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
        reachability_attempts = []
        seed_fk = []
        for warm_start_name, warm_start in warm_starts:
            seed_position, seed_rotation = self._lula.compute_forward_kinematics(
                LEFT_EE_FRAME, warm_start
            )
            seed_fk.append(
                {
                    "name": warm_start_name,
                    "joints_rad": np.asarray(warm_start).tolist(),
                    "fk_position_m": np.asarray(seed_position).tolist(),
                    "fk_orientation_wxyz": _quat_normalize(
                        rot_matrices_to_quats(seed_rotation)
                    ).tolist(),
                }
            )
        for axial_fraction in HOVER_AXIAL_FRACTION_CANDIDATES:
            candidate_local_grasp = self._grasp_object_position.copy()
            candidate_local_grasp[2] = BOTTLE_LENGTH_M * float(axial_fraction)
            candidate_grasp_position = bottle_position + bottle_rotation @ candidate_local_grasp
            for target_yaw_deg, target_orientation in target_yaw_candidates:
                candidate_grasp_orientation = _quat_multiply(
                    target_orientation, TARGET_TO_EE_ORIENTATION_WXYZ
                )
                for clearance_m in HOVER_CLEARANCE_CANDIDATES_M:
                    candidate_position = candidate_grasp_position + np.asarray([0.0, 0.0, clearance_m])
                    for warm_start_name, warm_start in warm_starts:
                        candidate_goal, success = self._lula.compute_inverse_kinematics(
                            LEFT_EE_FRAME,
                            candidate_position,
                            candidate_grasp_orientation,
                            warm_start=warm_start,
                            position_tolerance=0.0005,
                            orientation_tolerance=math.radians(0.5),
                        )
                        candidate_goal = np.asarray(candidate_goal, dtype=np.float64)
                        attempt = {
                            "axial_fraction_from_bottom": float(axial_fraction),
                            "clearance_m": float(clearance_m),
                            "target_yaw_deg": float(target_yaw_deg),
                            "warm_start": warm_start_name,
                            "success": bool(success),
                            "returned_joints_rad": candidate_goal.tolist(),
                        }
                        if candidate_goal.shape == current.shape and np.all(np.isfinite(candidate_goal)):
                            returned_position, returned_rotation = self._lula.compute_forward_kinematics(
                                LEFT_EE_FRAME, candidate_goal
                            )
                            returned_orientation = _quat_normalize(
                                rot_matrices_to_quats(returned_rotation)
                            )
                            attempt["fk_position_error_m"] = float(
                                np.linalg.norm(np.asarray(returned_position) - candidate_position)
                            )
                            attempt["fk_orientation_error_rad"] = _quat_angle(
                                returned_orientation, candidate_grasp_orientation
                            )
                        reachability_attempts.append(attempt)
                        if success and candidate_goal.shape == current.shape and np.all(np.isfinite(candidate_goal)):
                            goal = candidate_goal
                            goal_position = candidate_position
                            grasp_orientation = candidate_grasp_orientation
                            chosen_clearance = float(clearance_m)
                            chosen_warm_start = warm_start_name
                            chosen_axial_fraction = float(axial_fraction)
                            chosen_target_yaw_deg = float(target_yaw_deg)
                            break
                    if goal is not None:
                        break
                if goal is not None:
                    break
            if goal is not None:
                break
        if goal is None or goal_position is None or chosen_clearance is None:
            finite_attempts = [
                attempt for attempt in reachability_attempts
                if "fk_position_error_m" in attempt and "fk_orientation_error_rad" in attempt
            ]
            finite_attempts.sort(
                key=lambda attempt: (
                    float(attempt["fk_position_error_m"]),
                    float(attempt["fk_orientation_error_rad"]),
                )
            )
            diagnostic = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "bottle_position_m": bottle_position.tolist(),
                "bottle_orientation_wxyz": bottle_orientation.tolist(),
                "seed_forward_kinematics": seed_fk,
                "attempt_count": len(reachability_attempts),
                "best_attempts": finite_attempts[:20],
            }
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            with open(HOVER_REACHABILITY_DIAGNOSTIC_PATH, "w", encoding="utf-8") as stream:
                json.dump(diagnostic, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            raise RuntimeError(
                "no exact top-down HOVER IK solution was found from Bottle bottom 0.25L-0.50L "
                f"at 120-250 mm clearance; diagnostic: {HOVER_REACHABILITY_DIAGNOSTIC_PATH}; "
                "the arm was not moved and the grasp orientation was not tilted"
            )

        visible_target_orientation = self._target_orientation_from_ee(grasp_orientation)
        target_z_world = _quat_to_rotation(visible_target_orientation)[:, 2]
        target_z_world_dot = float(np.dot(target_z_world, np.asarray([0.0, 0.0, 1.0])))
        if target_z_world_dot < 1.0 - 1e-9:
            raise RuntimeError(
                f"visible Target +Z is not world +Z (dot={target_z_world_dot:.12f}); no motion planned"
            )

        sample_count = max(
            2,
            int(math.ceil(float(np.max(np.abs(goal - current))) / joint_step_rad)) + 1,
        )
        positions: List[np.ndarray] = []
        previous_ee: Optional[np.ndarray] = None
        minimum_ee_z = float("inf")
        maximum_ee_step = 0.0
        for alpha in np.linspace(0.0, 1.0, sample_count):
            joints = current + float(alpha) * (goal - current)
            ee_position, _ = self._lula.compute_forward_kinematics(LEFT_EE_FRAME, joints)
            ee_position = np.asarray(ee_position, dtype=np.float64)
            if ee_position.shape != (3,) or not np.all(np.isfinite(ee_position)):
                raise RuntimeError("non-finite Lula FK encountered while validating HOVER route")
            minimum_ee_z = min(minimum_ee_z, float(ee_position[2]))
            if previous_ee is not None:
                maximum_ee_step = max(maximum_ee_step, float(np.linalg.norm(ee_position - previous_ee)))
            previous_ee = ee_position
            positions.append(joints.copy())

        final_position, final_rotation = self._lula.compute_forward_kinematics(LEFT_EE_FRAME, goal)
        final_orientation = _quat_normalize(rot_matrices_to_quats(final_rotation))
        final_position_error = float(np.linalg.norm(np.asarray(final_position) - goal_position))
        final_orientation_error = _quat_angle(final_orientation, grasp_orientation)
        if final_position_error > 0.001 or final_orientation_error > math.radians(1.0):
            raise RuntimeError(
                f"HOVER final FK verification failed: {final_position_error * 1000.0:.3f} mm, "
                f"{math.degrees(final_orientation_error):.3f} deg"
            )
        if minimum_ee_z < HOVER_PLAN_MIN_EE_Z_M:
            raise RuntimeError(
                f"HOVER route rejected: sampled EE z fell to {minimum_ee_z:.4f} m "
                f"(< {HOVER_PLAN_MIN_EE_Z_M:.3f} m)"
            )
        if maximum_ee_step > HOVER_PLAN_MAX_EE_STEP_M:
            raise RuntimeError(
                f"HOVER route rejected: sampled FK jump {maximum_ee_step * 1000.0:.2f} mm "
                f"exceeds {HOVER_PLAN_MAX_EE_STEP_M * 1000.0:.1f} mm"
            )

        self._hover_plan_positions = positions
        self._planned_route_name = "HOVER"
        self._hover_plan_index = 0
        self._hover_plan_elapsed_s = 0.0
        self._hover_reached_reported = False
        self._hover_plan_goal_position = goal_position.copy()
        self._hover_plan_goal_orientation = grasp_orientation.copy()
        self._hover_plan_metrics = {
            "sample_count": float(sample_count),
            "chosen_clearance_m": chosen_clearance,
            "chosen_warm_start": chosen_warm_start,
            "chosen_axial_fraction_from_bottom": chosen_axial_fraction,
            "chosen_axial_distance_from_bottom_m": BOTTLE_LENGTH_M * chosen_axial_fraction,
            "chosen_target_yaw_deg": chosen_target_yaw_deg,
            "minimum_ee_z_m": minimum_ee_z,
            "maximum_sampled_ee_step_m": maximum_ee_step,
            "final_fk_position_error_m": final_position_error,
            "final_fk_orientation_error_rad": final_orientation_error,
            "visible_target_z_dot_world_z": target_z_world_dot,
            "planned_joint_step_rad": joint_step_rad,
            "control_hz": 1.0 / HOVER_PLAN_CONTROL_PERIOD_S,
        }
        self._set_target_from_ee_pose(goal_position, grasp_orientation)
        self._active_waypoint = "HOVER / PLAN READY"
        self._set_status(
            f"HOVER route PRECHECK PASS without motion: {sample_count} joint samples at "
            f"{joint_step_rad:.3f} rad / 50 Hz, "
            f"exact-vertical clearance={chosen_clearance * 1000.0:.1f} mm, "
            f"Bottle axial station={chosen_axial_fraction:.6f} L from bottom, "
            f"Target yaw={chosen_target_yaw_deg:.1f} deg, "
            f"IK warm start={chosen_warm_start}, "
            f"minimum EE z={minimum_ee_z:.4f} m, maximum sampled EE step={maximum_ee_step * 1000.0:.2f} mm, "
            f"final FK error={final_position_error * 1000.0:.3f} mm / "
            f"{math.degrees(final_orientation_error):.3f} deg, Target +Z dot world +Z={target_z_world_dot:.9f}. "
            "Press Play to execute; keep the gripper open."
        )

    def _plan_guided_waypoint_route(
        self,
        name: str,
        clearance_m: float,
        bottle_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        world_position_offset: Optional[np.ndarray] = None,
        maximum_lateral_deviation_m: float = GUIDED_ROUTE_MAX_LATERAL_DEVIATION_M,
    ) -> None:
        """Pre-solve a waypoint and validate its joint-interpolated route while paused."""

        self._require_guided_target_ready()
        self._clear_hover_plan()
        if self._lula is None or self._art_ik is None:
            raise RuntimeError("load the left arm first")
        grasp_position, grasp_orientation = self._update_grasp_world_pose(bottle_pose)
        position_offset = (
            np.zeros(3, dtype=np.float64)
            if world_position_offset is None
            else np.asarray(world_position_offset, dtype=np.float64)
        )
        if position_offset.shape != (3,) or not np.all(np.isfinite(position_offset)):
            raise RuntimeError(f"{name} world-position offset must be a finite 3-vector")
        goal_position = (
            grasp_position
            + np.asarray([0.0, 0.0, float(clearance_m)])
            + position_offset
        )
        target_visible_orientation = self._target_orientation_from_ee(grasp_orientation)
        target_z_dot = float(
            np.dot(
                _quat_to_rotation(target_visible_orientation)[:, 2],
                np.asarray([0.0, 0.0, 1.0]),
            )
        )
        if target_z_dot < 1.0 - 1e-9:
            raise RuntimeError(
                f"{name} Target +Z is not world +Z (dot={target_z_dot:.12f}); no route planned"
            )

        subset = self._art_ik.get_joints_subset()
        current = np.asarray(subset.get_joint_positions(), dtype=np.float64)
        joint_step_rad = self._configured_planned_joint_step_rad()
        if current.shape != (len(ARM_JOINTS),):
            raise RuntimeError(f"expected six current arm joints, got {current.shape}")
        start_position, _ = self._lula.compute_forward_kinematics(LEFT_EE_FRAME, current)
        start_position = np.asarray(start_position, dtype=np.float64)

        warm_starts = [("current", current)] + [
            (f"validated_top_down_{index + 1}", seed)
            for index, seed in enumerate(TOP_DOWN_IK_WARM_STARTS)
        ]
        solutions = []
        attempts = []
        for warm_start_name, warm_start in warm_starts:
            candidate, success = self._lula.compute_inverse_kinematics(
                LEFT_EE_FRAME,
                goal_position,
                grasp_orientation,
                warm_start=warm_start,
                position_tolerance=0.0005,
                orientation_tolerance=math.radians(0.5),
            )
            candidate = np.asarray(candidate, dtype=np.float64)
            attempt = {
                "warm_start": warm_start_name,
                "success": bool(success),
                "returned_joints_rad": candidate.tolist(),
            }
            if candidate.shape == current.shape and np.all(np.isfinite(candidate)):
                returned_position, returned_rotation = self._lula.compute_forward_kinematics(
                    LEFT_EE_FRAME, candidate
                )
                returned_orientation = _quat_normalize(rot_matrices_to_quats(returned_rotation))
                attempt["fk_position_error_m"] = float(
                    np.linalg.norm(np.asarray(returned_position) - goal_position)
                )
                attempt["fk_orientation_error_rad"] = _quat_angle(
                    returned_orientation, grasp_orientation
                )
                if success:
                    solutions.append(
                        (
                            float(np.linalg.norm(candidate - current)),
                            warm_start_name,
                            candidate,
                        )
                    )
            attempts.append(attempt)
        if not solutions:
            raise RuntimeError(
                f"{name} endpoint IK has no exact solution from the validated warm starts; "
                "the arm was not moved"
            )
        _, chosen_warm_start, goal = min(solutions, key=lambda row: row[0])

        sample_count = max(
            2,
            int(math.ceil(float(np.max(np.abs(goal - current))) / joint_step_rad)) + 1,
        )
        positions: List[np.ndarray] = []
        previous_ee: Optional[np.ndarray] = None
        maximum_ee_step = 0.0
        maximum_lateral_deviation = 0.0
        minimum_ee_z = float("inf")
        line = goal_position - start_position
        line_norm_squared = float(np.dot(line, line))
        for alpha in np.linspace(0.0, 1.0, sample_count):
            joints = current + float(alpha) * (goal - current)
            ee_position, _ = self._lula.compute_forward_kinematics(LEFT_EE_FRAME, joints)
            ee_position = np.asarray(ee_position, dtype=np.float64)
            if ee_position.shape != (3,) or not np.all(np.isfinite(ee_position)):
                raise RuntimeError(f"non-finite Lula FK while validating {name} route")
            minimum_ee_z = min(minimum_ee_z, float(ee_position[2]))
            if previous_ee is not None:
                maximum_ee_step = max(
                    maximum_ee_step, float(np.linalg.norm(ee_position - previous_ee))
                )
            previous_ee = ee_position
            if line_norm_squared > 1e-12:
                progress = float(np.clip(np.dot(ee_position - start_position, line) / line_norm_squared, 0.0, 1.0))
                closest = start_position + progress * line
                maximum_lateral_deviation = max(
                    maximum_lateral_deviation, float(np.linalg.norm(ee_position - closest))
                )
            positions.append(joints.copy())

        final_position, final_rotation = self._lula.compute_forward_kinematics(LEFT_EE_FRAME, goal)
        final_orientation = _quat_normalize(rot_matrices_to_quats(final_rotation))
        final_position_error = float(np.linalg.norm(np.asarray(final_position) - goal_position))
        final_orientation_error = _quat_angle(final_orientation, grasp_orientation)
        if final_position_error > 0.001 or final_orientation_error > math.radians(1.0):
            raise RuntimeError(
                f"{name} final FK verification failed: {final_position_error * 1000.0:.3f} mm, "
                f"{math.degrees(final_orientation_error):.3f} deg"
            )
        if minimum_ee_z < HOVER_PLAN_MIN_EE_Z_M:
            raise RuntimeError(f"{name} route drops below the minimum EE height")
        if maximum_ee_step > HOVER_PLAN_MAX_EE_STEP_M:
            raise RuntimeError(f"{name} route contains a discontinuous FK step")
        if maximum_lateral_deviation > float(maximum_lateral_deviation_m):
            raise RuntimeError(
                f"{name} route deviates {maximum_lateral_deviation * 1000.0:.2f} mm laterally "
                f"from the straight approach (> {float(maximum_lateral_deviation_m) * 1000.0:.1f} mm)"
            )

        self._hover_plan_positions = positions
        self._planned_route_name = name
        self._hover_plan_index = 0
        self._hover_plan_elapsed_s = 0.0
        self._hover_reached_reported = False
        self._hover_plan_goal_position = goal_position.copy()
        self._hover_plan_goal_orientation = grasp_orientation.copy()
        self._hover_plan_metrics = {
            "route_name": name,
            "sample_count": float(sample_count),
            "clearance_m": float(clearance_m),
            "world_position_offset_m": position_offset.tolist(),
            "chosen_warm_start": chosen_warm_start,
            "maximum_sampled_ee_step_m": maximum_ee_step,
            "maximum_lateral_deviation_m": maximum_lateral_deviation,
            "maximum_allowed_lateral_deviation_m": float(
                maximum_lateral_deviation_m
            ),
            "minimum_ee_z_m": minimum_ee_z,
            "final_fk_position_error_m": final_position_error,
            "final_fk_orientation_error_rad": final_orientation_error,
            "visible_target_z_dot_world_z": target_z_dot,
            "ik_attempts": attempts,
            "planned_joint_step_rad": joint_step_rad,
            "control_hz": 1.0 / HOVER_PLAN_CONTROL_PERIOD_S,
        }
        self._set_target_from_ee_pose(goal_position, grasp_orientation)
        self._active_waypoint = f"{name} / PLAN READY"
        self._set_status(
            f"{name} route PRECHECK PASS without motion: {sample_count} joint samples at "
            f"{joint_step_rad:.3f} rad / 50 Hz, "
            f"clearance={clearance_m * 1000.0:.1f} mm, warm start={chosen_warm_start}, "
            f"maximum lateral deviation={maximum_lateral_deviation * 1000.0:.2f} mm, "
            f"final FK error={final_position_error * 1000.0:.3f} mm / "
            f"{math.degrees(final_orientation_error):.3f} deg. Press Play to execute."
        )

    def _set_grasp_waypoint(self, name: str, clearance_m: float) -> None:
        self._plan_guided_waypoint_route(name, clearance_m)

    def _plan_continuous_lift_rotate_cap_up_route(
        self,
        bottle_pose: Tuple[np.ndarray, np.ndarray],
        additional_bottle_lift_m: float = VERTICAL_BOTTLE_TARGET_LIFT_M,
        rotation_clearance_m: float = VERTICAL_LIFT_ROTATION_START_CLEARANCE_M,
        maximum_joint_step_rad: float = VERTICAL_LIFT_MAX_JOINT_STEP_RAD,
    ) -> None:
        """Plan one uninterrupted loaded SE(3) route with cap axis ending at world +Z."""

        self._require_guided_target_ready()
        self._clear_hover_plan()
        if self._lula is None or self._art_ik is None:
            raise RuntimeError("load the left arm first")

        bottle_position = np.asarray(bottle_pose[0], dtype=np.float64)
        bottle_orientation = _quat_normalize(bottle_pose[1])
        additional_bottle_lift_m = float(additional_bottle_lift_m)
        rotation_clearance_m = float(rotation_clearance_m)
        maximum_joint_step_rad = float(maximum_joint_step_rad)
        if additional_bottle_lift_m < -1e-9 or rotation_clearance_m < -1e-9:
            raise RuntimeError("loaded lift and rotation clearance must be non-negative")
        if maximum_joint_step_rad <= 0.0:
            raise RuntimeError("loaded route joint step must be positive")
        start_ee_position, start_ee_orientation = self._current_lula_ee_pose()
        start_ee_position = np.asarray(start_ee_position, dtype=np.float64)
        start_ee_orientation = _quat_normalize(start_ee_orientation)
        start_ee_rotation = _quat_to_rotation(start_ee_orientation)
        bottle_rotation = _quat_to_rotation(bottle_orientation)

        # Preserve the measured physical EE->Bottle transform.  This avoids
        # assuming that the compliant, self-centred Bottle exactly matches the
        # nominal Grasp Editor transform after closure.
        ee_to_bottle_position = start_ee_rotation.T @ (
            bottle_position - start_ee_position
        )
        ee_to_bottle_rotation = start_ee_rotation.T @ bottle_rotation
        cap_axis_start = bottle_rotation[:, 2]
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        cap_dot = float(np.clip(np.dot(cap_axis_start, world_up), -1.0, 1.0))
        alignment_angle = math.acos(cap_dot)
        alignment_axis = np.cross(cap_axis_start, world_up)
        axis_norm = float(np.linalg.norm(alignment_axis))
        if axis_norm <= 1e-9:
            if cap_dot > 0.0:
                alignment_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                alignment_angle = 0.0
            else:
                # Deterministic 180-degree fallback perpendicular to the axis.
                alignment_axis = start_ee_rotation[:, 0]
                alignment_axis -= cap_axis_start * float(
                    np.dot(alignment_axis, cap_axis_start)
                )
                if float(np.linalg.norm(alignment_axis)) <= 1e-9:
                    alignment_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        alignment_axis /= max(float(np.linalg.norm(alignment_axis)), 1e-12)
        alignment_quaternion = _quat_from_axis_angle(
            alignment_axis, alignment_angle
        )
        subset = self._art_ik.get_joints_subset()
        current = np.asarray(subset.get_joint_positions(), dtype=np.float64)
        if current.shape != (len(ARM_JOINTS),):
            raise RuntimeError(f"expected six current arm joints, got {current.shape}")

        # Once the cap axis is +Z, rotation about world Z is a free task-space
        # degree of freedom.  Select the final bottle/gripper roll whose IK is
        # closest to the currently loaded arm pose; this avoids forcing a
        # needless wrist-flip singularity.
        endpoint_orientation_candidates = []
        endpoint_lift_probe_m = VERTICAL_LIFT_ENDPOINT_PROBE_M
        endpoint_position_probe = start_ee_position + np.asarray(
            [0.0, 0.0, endpoint_lift_probe_m], dtype=np.float64
        )
        for final_roll_deg in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
            roll_quaternion = _quat_from_axis_angle(
                world_up, math.radians(final_roll_deg)
            )
            total_alignment = _quat_multiply(
                roll_quaternion, alignment_quaternion
            )
            orientation_candidate = _quat_multiply(
                total_alignment, start_ee_orientation
            )
            joints_candidate, success = self._lula.compute_inverse_kinematics(
                LEFT_EE_FRAME,
                endpoint_position_probe,
                orientation_candidate,
                warm_start=current,
                position_tolerance=0.0005,
                orientation_tolerance=math.radians(0.5),
            )
            joints_candidate = np.asarray(joints_candidate, dtype=np.float64)
            endpoint_orientation_candidates.append(
                {
                    "final_roll_deg": final_roll_deg,
                    "success": bool(success),
                    "joint_distance_rad": (
                        float(np.linalg.norm(joints_candidate - current))
                        if joints_candidate.shape == current.shape
                        and np.all(np.isfinite(joints_candidate))
                        else None
                    ),
                    "orientation_wxyz": orientation_candidate.tolist(),
                }
            )
        feasible_endpoint_orientations = [
            item
            for item in endpoint_orientation_candidates
            if item["success"] and item["joint_distance_rad"] is not None
        ]
        if not feasible_endpoint_orientations:
            raise RuntimeError("cap +Z endpoint IK is unreachable for all tested wrist rolls")
        chosen_endpoint_orientation = min(
            feasible_endpoint_orientations,
            key=lambda item: float(item["joint_distance_rad"]),
        )
        chosen_final_roll_deg = float(
            chosen_endpoint_orientation["final_roll_deg"]
        )
        goal_ee_orientation = np.asarray(
            chosen_endpoint_orientation["orientation_wxyz"], dtype=np.float64
        )
        goal_ee_rotation = _quat_to_rotation(goal_ee_orientation)
        # Specify the final height in Bottle coordinates.  Rotation about the
        # grasp point changes the Bottle centre height, so an arbitrary EE lift
        # cannot guarantee the operator-requested 150 mm Bottle lift.
        target_bottle_center_z_m = float(
            bottle_position[2] + additional_bottle_lift_m
        )
        required_goal_ee_z_m = float(
            target_bottle_center_z_m
            - (goal_ee_rotation @ ee_to_bottle_position)[2]
        )
        required_ee_lift_m = float(required_goal_ee_z_m - start_ee_position[2])
        lift_distance_candidates = (required_ee_lift_m,)

        route_diagnostics: List[Dict[str, object]] = []
        selected = None
        for lift_distance_m in lift_distance_candidates:
            base_samples = max(
                3,
                int(
                    math.ceil(
                        2.0
                        * float(lift_distance_m)
                        / VERTICAL_LIFT_MAX_CARTESIAN_STEP_M
                    )
                )
                + 1,
                int(
                    math.ceil(
                        1.5
                        * float(alignment_angle)
                        / VERTICAL_LIFT_MAX_ORIENTATION_STEP_RAD
                    )
                )
                + 1,
            )
            for density_scale in (1.0, 1.5, 2.0, 3.0):
                sample_count = int(math.ceil(base_samples * density_scale))
                previous_joints = current.copy()
                previous_position = start_ee_position.copy()
                previous_orientation = start_ee_orientation.copy()
                positions: List[np.ndarray] = [current.copy()]
                maximum_joint_step = 0.0
                maximum_cartesian_step = 0.0
                maximum_orientation_step = 0.0
                maximum_fk_position_error = 0.0
                maximum_fk_orientation_error = 0.0
                minimum_predicted_bottle_bottom_z = float("inf")
                failure = None
                final_predicted_cap_axis = cap_axis_start.copy()

                for sample_index, progress in enumerate(
                    np.linspace(0.0, 1.0, sample_count)
                ):
                    progress = float(progress)
                    lift_ratio = progress
                    rotation_start_progress = min(
                        0.8,
                        rotation_clearance_m
                        / float(lift_distance_m),
                    )
                    if progress <= rotation_start_progress:
                        rotate_ratio = 0.0
                    else:
                        rotate_progress = (
                            (progress - rotation_start_progress)
                            / (1.0 - rotation_start_progress)
                        )
                        rotate_ratio = (
                            3.0 * rotate_progress * rotate_progress
                            - 2.0 * rotate_progress * rotate_progress * rotate_progress
                        )
                    desired_orientation = _quat_slerp(
                        start_ee_orientation,
                        goal_ee_orientation,
                        rotate_ratio,
                    )
                    desired_rotation = _quat_to_rotation(desired_orientation)
                    desired_position = start_ee_position + np.asarray(
                        [0.0, 0.0, float(lift_distance_m) * lift_ratio],
                        dtype=np.float64,
                    )
                    predicted_bottle_position = (
                        desired_position
                        + desired_rotation @ ee_to_bottle_position
                    )
                    predicted_bottle_rotation = (
                        desired_rotation @ ee_to_bottle_rotation
                    )
                    final_predicted_cap_axis = predicted_bottle_rotation[:, 2]
                    minimum_predicted_bottle_bottom_z = min(
                        minimum_predicted_bottle_bottom_z,
                        float(predicted_bottle_position[2]),
                    )

                    if sample_index == 0:
                        continue
                    cartesian_step = float(
                        np.linalg.norm(desired_position - previous_position)
                    )
                    orientation_step = _quat_angle(
                        previous_orientation, desired_orientation
                    )
                    maximum_cartesian_step = max(
                        maximum_cartesian_step, cartesian_step
                    )
                    maximum_orientation_step = max(
                        maximum_orientation_step, orientation_step
                    )
                    candidate, success = self._lula.compute_inverse_kinematics(
                        LEFT_EE_FRAME,
                        desired_position,
                        desired_orientation,
                        warm_start=previous_joints,
                        position_tolerance=0.0005,
                        orientation_tolerance=math.radians(0.5),
                    )
                    candidate = np.asarray(candidate, dtype=np.float64)
                    if (
                        not success
                        or candidate.shape != current.shape
                        or not np.all(np.isfinite(candidate))
                    ):
                        failure = f"IK failed at sample {sample_index}/{sample_count - 1}"
                        break
                    joint_step = float(
                        np.max(np.abs(candidate - previous_joints))
                    )
                    maximum_joint_step = max(maximum_joint_step, joint_step)
                    fk_position, fk_rotation = self._lula.compute_forward_kinematics(
                        LEFT_EE_FRAME, candidate
                    )
                    fk_orientation = _quat_normalize(
                        rot_matrices_to_quats(fk_rotation)
                    )
                    maximum_fk_position_error = max(
                        maximum_fk_position_error,
                        float(
                            np.linalg.norm(
                                np.asarray(fk_position, dtype=np.float64)
                                - desired_position
                            )
                        ),
                    )
                    maximum_fk_orientation_error = max(
                        maximum_fk_orientation_error,
                        _quat_angle(fk_orientation, desired_orientation),
                    )
                    positions.append(candidate.copy())
                    previous_joints = candidate
                    previous_position = desired_position
                    previous_orientation = desired_orientation

                final_cap_angle = math.acos(
                    float(
                        np.clip(
                            np.dot(final_predicted_cap_axis, world_up), -1.0, 1.0
                        )
                    )
                )
                diagnostic = {
                    "lift_distance_m": float(lift_distance_m),
                    "chosen_final_roll_deg": chosen_final_roll_deg,
                    "sample_count": int(sample_count),
                    "failure": failure,
                    "maximum_joint_step_rad": maximum_joint_step,
                    "maximum_cartesian_step_m": maximum_cartesian_step,
                    "maximum_orientation_step_rad": maximum_orientation_step,
                    "maximum_fk_position_error_m": maximum_fk_position_error,
                    "maximum_fk_orientation_error_rad": maximum_fk_orientation_error,
                    "minimum_predicted_bottle_bottom_z_m": (
                        minimum_predicted_bottle_bottom_z
                    ),
                    "final_predicted_cap_axis_world": final_predicted_cap_axis.tolist(),
                    "final_predicted_cap_axis_error_rad": final_cap_angle,
                }
                route_diagnostics.append(diagnostic)
                if failure is not None:
                    continue
                if maximum_joint_step > maximum_joint_step_rad + 1e-9:
                    continue
                if (
                    minimum_predicted_bottle_bottom_z
                    < VERTICAL_LIFT_MIN_BOTTLE_BOTTOM_Z_M
                ):
                    continue
                if final_cap_angle > VERTICAL_LIFT_CAP_AXIS_GATE_RAD:
                    continue
                selected = {
                    "positions": positions,
                    "goal_position": previous_position.copy(),
                    "goal_orientation": previous_orientation.copy(),
                    "diagnostic": diagnostic,
                }
                break
            if selected is not None:
                break

        # Lula 5.1 can jump between equivalent wrist branches when IK is
        # solved independently at hundreds of Cartesian samples.  If that
        # happens, solve a 150 mm orientation-preserving lift pose and the
        # physical endpoint, then time-parameterize both joint segments without
        # a held sample between them.  FK validation below enforces that the
        # Bottle stays horizontal until clearance and rotates while the second
        # segment continues lifting.
        joint_continuous_attempts: List[Dict[str, object]] = []
        if selected is None:
            clearance_position = start_ee_position + np.asarray(
                [0.0, 0.0, rotation_clearance_m],
                dtype=np.float64,
            )
            clearance_joints, clearance_success = (
                self._lula.compute_inverse_kinematics(
                    LEFT_EE_FRAME,
                    clearance_position,
                    start_ee_orientation,
                    warm_start=current,
                    position_tolerance=0.0005,
                    orientation_tolerance=math.radians(0.5),
                )
            )
            clearance_joints = np.asarray(clearance_joints, dtype=np.float64)
            if (
                not clearance_success
                or clearance_joints.shape != current.shape
                or not np.all(np.isfinite(clearance_joints))
            ):
                clearance_joints = None
            for lift_distance_m in lift_distance_candidates:
                if clearance_joints is None:
                    break
                goal_position_candidate = start_ee_position + np.asarray(
                    [0.0, 0.0, float(lift_distance_m)], dtype=np.float64
                )
                for endpoint_item in feasible_endpoint_orientations:
                    final_roll_deg = float(endpoint_item["final_roll_deg"])
                    goal_orientation_candidate = np.asarray(
                        endpoint_item["orientation_wxyz"], dtype=np.float64
                    )
                    goal_joints, success = self._lula.compute_inverse_kinematics(
                        LEFT_EE_FRAME,
                        goal_position_candidate,
                        goal_orientation_candidate,
                        warm_start=clearance_joints,
                        position_tolerance=0.0005,
                        orientation_tolerance=math.radians(0.5),
                    )
                    goal_joints = np.asarray(goal_joints, dtype=np.float64)
                    attempt: Dict[str, object] = {
                        "trajectory_type": "FK_VALIDATED_CONTINUOUS_JOINT_LIFT_ROTATE",
                        "lift_distance_m": float(lift_distance_m),
                        "final_roll_deg": final_roll_deg,
                        "endpoint_ik_success": bool(success),
                    }
                    joint_continuous_attempts.append(attempt)
                    if (
                        not success
                        or goal_joints.shape != current.shape
                        or not np.all(np.isfinite(goal_joints))
                    ):
                        attempt["failure"] = "endpoint IK failed"
                        continue
                    clearance_delta = float(
                        np.max(np.abs(clearance_joints - current))
                    )
                    loaded_delta = float(
                        np.max(np.abs(goal_joints - clearance_joints))
                    )
                    ramp_fraction = 0.0
                    peak_progress_rate = 1.0 / (1.0 - ramp_fraction)
                    clearance_sample_count = max(
                        2,
                        int(
                            math.ceil(
                                peak_progress_rate
                                * clearance_delta
                                / maximum_joint_step_rad
                            )
                        )
                        + 1,
                    )
                    loaded_sample_count = max(
                        2,
                        int(
                            math.ceil(
                                peak_progress_rate
                                * loaded_delta
                                / maximum_joint_step_rad
                            )
                        )
                        + 1,
                    )

                    def ramped_progress(progress: float) -> float:
                        progress = float(progress)
                        if progress < ramp_fraction:
                            return (
                                0.5
                                * progress
                                * progress
                                / (ramp_fraction * (1.0 - ramp_fraction))
                            )
                        if progress > 1.0 - ramp_fraction:
                            remaining = 1.0 - progress
                            return 1.0 - (
                                0.5
                                * remaining
                                * remaining
                                / (ramp_fraction * (1.0 - ramp_fraction))
                            )
                        return (
                            progress - 0.5 * ramp_fraction
                        ) / (1.0 - ramp_fraction)

                    joint_samples = [
                        current
                        + ramped_progress(progress)
                        * (clearance_joints - current)
                        for progress in np.linspace(0.0, 1.0, clearance_sample_count)
                    ]
                    joint_samples.extend(
                        clearance_joints
                        + ramped_progress(progress)
                        * (goal_joints - clearance_joints)
                        for progress in np.linspace(0.0, 1.0, loaded_sample_count)[1:]
                    )
                    sample_count = len(joint_samples)
                    positions: List[np.ndarray] = []
                    previous_joints = current.copy()
                    previous_ee_position = start_ee_position.copy()
                    previous_cap_error = alignment_angle
                    maximum_joint_step = 0.0
                    maximum_ee_step = 0.0
                    maximum_ee_z_drop_per_step = 0.0
                    maximum_cap_error_increase = 0.0
                    minimum_predicted_bottle_bottom_z = float("inf")
                    quarter_lift_m = None
                    quarter_cap_rotation_rad = None
                    maximum_rotation_before_clearance_rad = 0.0
                    rotation_start_bottle_lift_m = None
                    minimum_bottle_lift_during_rotation_m = float("inf")
                    final_predicted_bottle_lift_m = 0.0
                    final_ee_position = start_ee_position.copy()
                    final_ee_orientation = start_ee_orientation.copy()
                    final_cap_axis = cap_axis_start.copy()
                    final_cap_error = alignment_angle
                    for sample_index, joints in enumerate(joint_samples):
                        progress = float(sample_index) / float(max(1, sample_count - 1))
                        ee_position, ee_rotation = self._lula.compute_forward_kinematics(
                            LEFT_EE_FRAME, joints
                        )
                        ee_position = np.asarray(ee_position, dtype=np.float64)
                        ee_orientation = _quat_normalize(
                            rot_matrices_to_quats(ee_rotation)
                        )
                        ee_rotation = _quat_to_rotation(ee_orientation)
                        predicted_bottle_position = (
                            ee_position + ee_rotation @ ee_to_bottle_position
                        )
                        predicted_bottle_rotation = (
                            ee_rotation @ ee_to_bottle_rotation
                        )
                        cap_axis = predicted_bottle_rotation[:, 2]
                        cap_error = math.acos(
                            float(np.clip(np.dot(cap_axis, world_up), -1.0, 1.0))
                        )
                        bottle_lift_now_m = float(
                            predicted_bottle_position[2] - bottle_position[2]
                        )
                        rotation_now_rad = float(alignment_angle - cap_error)
                        if sample_index < clearance_sample_count - 1:
                            maximum_rotation_before_clearance_rad = max(
                                maximum_rotation_before_clearance_rad,
                                rotation_now_rad,
                            )
                        if (
                            rotation_start_bottle_lift_m is None
                            and rotation_now_rad >= math.radians(2.0)
                        ):
                            rotation_start_bottle_lift_m = bottle_lift_now_m
                        if sample_index >= clearance_sample_count - 1:
                            minimum_bottle_lift_during_rotation_m = min(
                                minimum_bottle_lift_during_rotation_m,
                                bottle_lift_now_m,
                            )
                        minimum_predicted_bottle_bottom_z = min(
                            minimum_predicted_bottle_bottom_z,
                            float(predicted_bottle_position[2]),
                        )
                        if sample_index:
                            maximum_joint_step = max(
                                maximum_joint_step,
                                float(np.max(np.abs(joints - previous_joints))),
                            )
                            maximum_ee_step = max(
                                maximum_ee_step,
                                float(np.linalg.norm(ee_position - previous_ee_position)),
                            )
                            maximum_ee_z_drop_per_step = max(
                                maximum_ee_z_drop_per_step,
                                float(previous_ee_position[2] - ee_position[2]),
                            )
                            maximum_cap_error_increase = max(
                                maximum_cap_error_increase,
                                float(cap_error - previous_cap_error),
                            )
                        if quarter_lift_m is None and progress >= 0.25:
                            quarter_lift_m = float(
                                ee_position[2] - start_ee_position[2]
                            )
                            quarter_cap_rotation_rad = float(
                                alignment_angle - cap_error
                            )
                        positions.append(joints.copy())
                        previous_joints = joints
                        previous_ee_position = ee_position
                        previous_cap_error = cap_error
                        final_ee_position = ee_position
                        final_ee_orientation = ee_orientation
                        final_cap_axis = cap_axis
                        final_cap_error = cap_error
                        final_predicted_bottle_lift_m = bottle_lift_now_m

                    actual_lift_m = float(
                        final_ee_position[2] - start_ee_position[2]
                    )
                    attempt.update(
                        {
                            "sample_count": int(sample_count),
                            "maximum_joint_step_rad": maximum_joint_step,
                            "maximum_ee_step_m": maximum_ee_step,
                            "maximum_ee_z_drop_per_step_m": maximum_ee_z_drop_per_step,
                            "maximum_cap_axis_error_increase_rad": maximum_cap_error_increase,
                            "minimum_predicted_bottle_bottom_z_m": (
                                minimum_predicted_bottle_bottom_z
                            ),
                            "quarter_progress_lift_m": quarter_lift_m,
                            "quarter_progress_cap_rotation_rad": quarter_cap_rotation_rad,
                            "maximum_rotation_before_clearance_rad": (
                                maximum_rotation_before_clearance_rad
                            ),
                            "rotation_start_bottle_lift_m": (
                                rotation_start_bottle_lift_m
                            ),
                            "minimum_bottle_lift_during_rotation_m": (
                                minimum_bottle_lift_during_rotation_m
                            ),
                            "actual_fk_lift_m": actual_lift_m,
                            "final_predicted_bottle_lift_m": (
                                final_predicted_bottle_lift_m
                            ),
                            "final_predicted_cap_axis_world": final_cap_axis.tolist(),
                            "final_predicted_cap_axis_error_rad": final_cap_error,
                        }
                    )
                    checks = {
                        "joint_step": maximum_joint_step
                        <= maximum_joint_step_rad + 1e-9,
                        "ee_step": maximum_ee_step
                        <= VERTICAL_LIFT_MAX_CARTESIAN_STEP_M + 1e-9,
                        "no_downward_ee_step": maximum_ee_z_drop_per_step <= 0.001,
                        "cap_error_monotonic": maximum_cap_error_increase
                        <= math.radians(1.0),
                        "bottle_bottom_sweep": minimum_predicted_bottle_bottom_z
                        >= VERTICAL_LIFT_MIN_BOTTLE_BOTTOM_Z_M,
                        "bottle_lift_height": final_predicted_bottle_lift_m
                        >= (
                            additional_bottle_lift_m
                            - VERTICAL_BOTTLE_LIFT_TOLERANCE_M
                        ),
                        "cap_axis": final_cap_error <= VERTICAL_LIFT_CAP_AXIS_GATE_RAD,
                        "clearance_then_lift_rotate": (
                            (
                                rotation_clearance_m >= 0.100
                                and maximum_rotation_before_clearance_rad
                                <= math.radians(2.0)
                                and rotation_start_bottle_lift_m is not None
                                and rotation_start_bottle_lift_m
                                >= rotation_clearance_m - 0.010
                                and rotation_start_bottle_lift_m
                                <= rotation_clearance_m + 0.020
                                and minimum_bottle_lift_during_rotation_m
                                >= rotation_clearance_m - 0.070
                            )
                            or (
                                rotation_clearance_m < 0.100
                                and rotation_start_bottle_lift_m is not None
                                and rotation_start_bottle_lift_m >= -0.030
                                and minimum_bottle_lift_during_rotation_m >= -0.080
                            )
                        ),
                    }
                    attempt["checks"] = checks
                    if not all(checks.values()):
                        attempt["failure"] = "FK sweep gate failed"
                        continue
                    selected = {
                        "positions": positions,
                        "goal_position": final_ee_position.copy(),
                        "goal_orientation": final_ee_orientation.copy(),
                        "diagnostic": attempt,
                        "trajectory_type": (
                            "FK_VALIDATED_CONTINUOUS_JOINT_LIFT_ROTATE"
                        ),
                    }
                    chosen_final_roll_deg = final_roll_deg
                    break
                if selected is not None:
                    break

        if selected is None:
            diagnostic_report = {
                "status": "NO_SAFE_ROUTE",
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "cap_axis_start_world": cap_axis_start.tolist(),
                "alignment_axis_world": alignment_axis.tolist(),
                "alignment_angle_rad": float(alignment_angle),
                "endpoint_orientation_candidates": endpoint_orientation_candidates,
                "chosen_final_roll_deg": chosen_final_roll_deg,
                "start_ee_position_m": start_ee_position.tolist(),
                "start_ee_orientation_wxyz": start_ee_orientation.tolist(),
                "bottle_position_m": bottle_position.tolist(),
                "bottle_orientation_wxyz": bottle_orientation.tolist(),
                "attempts": route_diagnostics,
                "joint_continuous_attempts": joint_continuous_attempts,
            }
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            with open(
                VERTICAL_LIFT_DIAGNOSTIC_PATH, "w", encoding="utf-8"
            ) as stream:
                json.dump(diagnostic_report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            raise RuntimeError(
                "no continuous loaded Lift+Rotate route passed IK, adaptive joint-step, "
                "150 mm Bottle lift before rotation, Bottle-bottom sweep, "
                "and cap +Z gates; diagnostic: "
                f"{VERTICAL_LIFT_DIAGNOSTIC_PATH}"
            )

        self._hover_plan_positions = selected["positions"]
        self._planned_route_name = "AUTO LIFT + ROTATE CAP TO +Z"
        self._hover_plan_index = 0
        self._hover_plan_elapsed_s = 0.0
        self._hover_reached_reported = False
        self._hover_plan_goal_position = selected["goal_position"]
        self._hover_plan_goal_orientation = selected["goal_orientation"]
        self._hover_plan_metrics = {
            "route_name": self._planned_route_name,
            "trajectory_type": selected.get(
                "trajectory_type", "SINGLE_CONTINUOUS_SE3_LIFT_AND_ROTATE"
            ),
            "lift_profile": (
                "150 mm orientation-preserving Bottle lift before rotation"
            ),
            "rotation_profile": (
                "starts only after approximately 150 mm Bottle lift; the "
                "Bottle centre height is then held while orientation changes"
            ),
            "cap_axis_start_world": cap_axis_start.tolist(),
            "requested_alignment_axis_world": alignment_axis.tolist(),
            "requested_alignment_angle_rad": float(alignment_angle),
            "additional_bottle_lift_m": additional_bottle_lift_m,
            "rotation_clearance_m": rotation_clearance_m,
            "planned_joint_step_rad": maximum_joint_step_rad,
            "endpoint_orientation_candidates": endpoint_orientation_candidates,
            "chosen_final_roll_deg": chosen_final_roll_deg,
            "measured_ee_to_bottle_position_m": ee_to_bottle_position.tolist(),
            "arm_joint_names": list(ARM_JOINTS),
            "start_arm_joints_rad": current.tolist(),
            "goal_arm_joints_rad": np.asarray(
                selected["positions"][-1], dtype=np.float64
            ).tolist(),
            "arm_joint_delta_rad": (
                np.asarray(selected["positions"][-1], dtype=np.float64) - current
            ).tolist(),
            "selected": selected["diagnostic"],
            "all_attempts": route_diagnostics,
            "joint_continuous_attempts": joint_continuous_attempts,
            "control_hz": 1.0 / HOVER_PLAN_CONTROL_PERIOD_S,
        }
        self._set_target_from_ee_pose(
            selected["goal_position"], selected["goal_orientation"]
        )
        self._active_waypoint = (
            "AUTO LIFT + ROTATE CAP TO +Z / PLAN READY"
        )
        self._set_status(
            "Continuous loaded Lift+Rotate PRECHECK PASS: position and orientation "
            "change on every 50 Hz route sample; no intermediate stop; predicted Bottle "
            f"bottom z >= {selected['diagnostic']['minimum_predicted_bottle_bottom_z_m']:.4f} m."
        )

    def _on_step_approach(self) -> None:
        self._run_guarded("Step Approach 5 mm", self._step_approach)

    def _step_approach(self) -> None:
        self._require_guided_target_ready()
        if self._active_waypoint == "none":
            raise RuntimeError("select HOVER or PREGRASP before using incremental approach")
        grasp_position, grasp_orientation = self._update_grasp_world_pose()
        current_position, _ = self._target_pose()
        current_clearance = max(0.0, float(current_position[2] - grasp_position[2]))
        next_clearance = max(0.0, current_clearance - APPROACH_STEP_M)
        self._plan_guided_waypoint_route(
            f"APPROACH +{next_clearance * 1000.0:.0f} mm", next_clearance
        )

    def _on_abort_and_hold(self) -> None:
        self._auto_abort_requested = True
        self._run_guarded("Abort and Hold", self._abort_and_hold)

    def _abort_and_hold(self) -> None:
        if self._timeline.is_playing():
            self._timeline.pause()
        self._disable_follow()
        self._clear_hover_plan()
        if self._art_ik is not None and self._target is not None and is_prim_path_valid(TARGET_PATH):
            position, orientation = self._current_lula_ee_pose()
            self._set_target_from_ee_pose(position, orientation)
        self._active_waypoint = "ABORTED / current EE"
        self._set_status(
            "ABORT complete: Timeline Paused, IK Follow disabled, and the extension Target was returned to the current EE. "
            "Arm drives retain their last targets; Bottle and gripper DOFs were not commanded."
        )

    def _summarize_auto_response(self) -> Dict[str, object]:
        if len(self._joint_log_rows) < 60:
            raise RuntimeError(f"only {len(self._joint_log_rows)} physics samples were recorded")
        results = []
        for joint_name in ("shoulder", "elbow", "wrist_angle"):
            actual = np.asarray(
                [row[f"{joint_name}_actual_position_rad"] for row in self._joint_log_rows],
                dtype=np.float64,
            )
            velocity = np.asarray(
                [row[f"{joint_name}_velocity_rad_s"] for row in self._joint_log_rows],
                dtype=np.float64,
            )
            target = float(self._joint_log_rows[-1][f"{joint_name}_position_target_rad"])
            command = target - float(actual[0])
            error = actual - target
            overshoot = 0.0
            crossings = 0
            if abs(command) >= 1e-5:
                overshoot = max(
                    0.0,
                    float(np.max(actual) - target) if command > 0.0 else float(target - np.min(actual)),
                ) / abs(command)
                deadband = 0.01 * abs(command)
                signs = np.where(error > deadband, 1, np.where(error < -deadband, -1, 0))
                signs = signs[signs != 0]
                crossings = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
            tolerance = max(0.02 * abs(command), 0.0001)
            settled = (np.abs(error) <= tolerance) & (np.abs(velocity) <= 0.01)
            settle_time_s = None
            for sample in range(len(settled)):
                if bool(np.all(settled[sample:])):
                    settle_time_s = float(sum(
                        row["physics_dt_s"] for row in self._joint_log_rows[: sample + 1]
                    ))
                    break
            result = {
                "joint": joint_name,
                "command_rad": command,
                "overshoot_fraction": overshoot,
                "target_crossings": crossings,
                "peak_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
                "settle_time_s": settle_time_s,
                "final_abs_error_rad": float(abs(error[-1])),
            }
            result["pass"] = bool(
                overshoot <= 0.02
                and crossings <= 1
                and settle_time_s is not None
                and settle_time_s <= 0.25
                and result["final_abs_error_rad"] <= 0.002
            )
            results.append(result)
        return {
            "sample_count": len(self._joint_log_rows),
            "joints": results,
            "pass": all(bool(row["pass"]) for row in results),
        }

    async def _run_auto_z5_acceptance(self) -> None:
        """Consume one request and run a guarded +Z acceptance test."""

        app = omni.kit.app.get_app()
        running_path = AUTO_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "runtime_profile": {
                "drive_type": "acceleration",
                "stiffness_rad_units": ARM_RUNTIME_STIFFNESS,
                "damping_rad_units": ARM_RUNTIME_DAMPING,
                "robot_gravity_compensation": True,
            },
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(AUTO_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                result["request"] = json.load(stream)
            request = result["request"]
            delta_z_m = float(request.get("delta_z_m", 0.005))
            if not 0.0 < delta_z_m <= 0.020:
                raise ValueError(f"delta_z_m must be in (0, 0.020], got {delta_z_m}")
            capture_viewport = bool(request.get("capture_viewport", False))
            capture_dir = ""
            viewport = None
            capture_index = 0
            if capture_viewport:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("no active viewport is available for the requested recording")
                safe_run_id = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in str(request.get("run_id", "auto_z_acceptance"))
                )
                capture_dir = os.path.join(DEFAULT_LOG_DIR, "viewport_recordings", safe_run_id)
                os.makedirs(capture_dir, exist_ok=False)
                result["viewport_recording"] = {
                    "frame_directory": capture_dir,
                    "source": "Isaac Sim active server viewport",
                    "desktop_switched": False,
                }

                async def capture_frame() -> None:
                    nonlocal capture_index
                    output = os.path.join(capture_dir, f"frame_{capture_index:05d}.png")
                    helper = capture_viewport_to_file(viewport, file_path=output, is_hdr=False)
                    await helper.wait_for_result()
                    for _ in range(50):
                        if os.path.isfile(output) and os.path.getsize(output) > 0:
                            break
                        await app.next_update_async()
                    if not os.path.isfile(output):
                        raise RuntimeError(f"viewport frame was not written: {output}")
                    if os.path.getsize(output) <= 0:
                        raise RuntimeError(f"viewport frame is empty: {output}")
                    capture_index += 1

            result["motion"] = {"delta_z_m": delta_z_m}
            for _ in range(5):
                await app.next_update_async()
            self._disable_follow()
            self._joint_log_enabled = False
            if is_prim_path_valid(TARGET_PATH):
                delete_prim(TARGET_PATH)
            self._target = None
            if self._timeline.is_stopped():
                self._timeline.play()
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            self._reset_left_sleep()
            # A paused teleport updates articulation state immediately but the
            # visible USD/Fabric link transforms need physics frames before
            # Lula-vs-USD alignment can be measured.
            self._timeline.play()
            for _ in range(30):
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(5):
                await app.next_update_async()
            self._load_left_arm()
            self._sync_base_pose()
            for _ in range(3):
                await app.next_update_async()
            if not self._validate_alignment():
                raise RuntimeError(
                    f"initial alignment failed: {self._last_position_error * 1000.0:.4f} mm, "
                    f"{math.degrees(self._last_orientation_error):.4f} deg"
                )
            self._on_create_target()
            self._on_start_joint_log()
            self._enable_follow()
            position, orientation = self._target_pose()
            if capture_viewport:
                await capture_frame()
            self._set_target_from_ee_pose(
                position + np.asarray([0.0, 0.0, delta_z_m]), orientation
            )
            self._timeline.play()

            for _ in range(600):
                await app.next_update_async()
                if capture_viewport and capture_index < 90:
                    await capture_frame()
                if not self._follow_enabled:
                    raise RuntimeError("IK Follow stopped before the automated response settled")
                if len(self._joint_log_rows) >= 120:
                    recent = self._joint_log_rows[-30:]
                    peak_recent_velocity = max(
                        abs(row[f"{joint}_velocity_rad_s"])
                        for row in recent
                        for joint in ("shoulder", "elbow", "wrist_angle")
                    )
                    if peak_recent_velocity <= 0.005:
                        break
            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            if capture_viewport:
                await capture_frame()
                result["viewport_recording"]["frame_count"] = capture_index
            response = self._summarize_auto_response()
            result["response"] = response
            result["alignment_after_motion"] = {
                "position_error_m": self._last_position_error,
                "orientation_error_rad": self._last_orientation_error,
            }
            self._on_stop_joint_log()
            result["joint_log_csv"] = self._last_joint_log_path
            result["status"] = "PASS" if response["pass"] else "FAILED_GATE"
            self._set_status(
                f"Automated +Z {delta_z_m * 1000.0:.1f} mm acceptance {result['status']}; "
                f"result: {AUTO_RESULT_PATH}",
                warn=not bool(response["pass"]),
            )
        except Exception as exc:
            self._timeline.pause()
            self._disable_follow()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Automated +Z acceptance failed: {exc}", warn=True)
        finally:
            with open(AUTO_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_rotate_bottle_midpoint(self) -> None:
        """Rotate Bottle500 180 degrees about its geometric midpoint around world Z."""

        app = omni.kit.app.get_app()
        running_path = BOTTLE_ROTATE_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "arm_commanded": False,
            "gripper_commanded": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(BOTTLE_ROTATE_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("rotation request must explicitly set simulation_only=true")
            if float(request.get("angle_deg", 0.0)) != 180.0:
                raise RuntimeError("only an exact 180 degree midpoint rotation is accepted")
            if request.get("axis") != "world_z":
                raise RuntimeError("rotation axis must be world_z")
            component = str(request.get("component", "bottle"))

            if component == "canonical_startup_pose":
                if self._timeline.is_playing():
                    self._timeline.pause()
                for _ in range(3):
                    await app.next_update_async()
                canonical_orientation = np.asarray(
                    [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float64
                )
                requested_poses = {
                    BOTTLE_PATH: np.asarray([-0.103, 0.0, 0.034], dtype=np.float64),
                    BOTTLE_CAP_PATH: np.asarray([0.085, 0.0, 0.034], dtype=np.float64),
                }
                readback = {}
                for index, (prim_path, requested_position) in enumerate(requested_poses.items()):
                    prim = SingleXFormPrim(
                        prim_path,
                        name=f"canonical_bottle_component_{index}",
                        reset_xform_properties=False,
                    )
                    prim.set_world_pose(
                        position=requested_position, orientation=canonical_orientation
                    )
                for _ in range(8):
                    await app.next_update_async()
                for prim_path, requested_position in requested_poses.items():
                    position, orientation = get_world_pose(prim_path)
                    position = np.asarray(position, dtype=np.float64)
                    orientation = _quat_normalize(orientation)
                    position_error_m = float(np.linalg.norm(position - requested_position))
                    orientation_error_rad = _quat_angle(orientation, canonical_orientation)
                    if position_error_m > 0.0001 or orientation_error_rad > math.radians(0.01):
                        raise RuntimeError(
                            f"canonical pose readback failed for {prim_path}: "
                            f"{position_error_m * 1000.0:.4f} mm / "
                            f"{math.degrees(orientation_error_rad):.6f} deg"
                        )
                    readback[prim_path] = {
                        "position_m": position.tolist(),
                        "orientation_wxyz": orientation.tolist(),
                    }
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("no active server viewport is available")
                screenshot_path = os.path.join(
                    DEFAULT_LOG_DIR, "canonical_plus_x_bottle_startup_pose.png"
                )
                helper = capture_viewport_to_file(viewport, file_path=screenshot_path, is_hdr=False)
                await helper.wait_for_result()
                result.update(
                    {
                        "status": "PASS",
                        "component": component,
                        "startup_pose_readback": readback,
                        "bottle_center_world_m": [0.0, 0.0, 0.034],
                        "bottle_mouth_world_axis": "+X",
                        "screenshot": screenshot_path,
                    }
                )
                self._set_status(
                    "Canonical +X Bottle/BottleCap startup pose applied in the current session. "
                    "Timeline remains Paused."
                )
                return

            if component == "cap_only":
                pivot = np.asarray(request.get("midpoint_world_m"), dtype=np.float64)
                if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
                    raise RuntimeError("cap_only repair requires a finite midpoint_world_m vector")
                if self._timeline.is_playing():
                    self._timeline.pause()
                for _ in range(3):
                    await app.next_update_async()
                cap_before_position, cap_before_orientation = get_world_pose(BOTTLE_CAP_PATH)
                cap_before_position = np.asarray(cap_before_position, dtype=np.float64)
                cap_before_orientation = _quat_normalize(cap_before_orientation)
                world_z_180 = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                rotation = _quat_to_rotation(world_z_180)
                cap_after_requested_position = pivot + rotation @ (cap_before_position - pivot)
                cap_after_requested_orientation = _quat_multiply(world_z_180, cap_before_orientation)
                cap = SingleXFormPrim(
                    BOTTLE_CAP_PATH, name="bottle_cap_midpoint_rotation", reset_xform_properties=False
                )
                cap.set_world_pose(
                    position=cap_after_requested_position,
                    orientation=cap_after_requested_orientation,
                )
                for _ in range(8):
                    await app.next_update_async()
                cap_after_position, cap_after_orientation = get_world_pose(BOTTLE_CAP_PATH)
                cap_after_position = np.asarray(cap_after_position, dtype=np.float64)
                cap_after_orientation = _quat_normalize(cap_after_orientation)
                position_error_m = float(
                    np.linalg.norm(cap_after_position - cap_after_requested_position)
                )
                orientation_error_rad = _quat_angle(
                    cap_after_orientation, cap_after_requested_orientation
                )
                if position_error_m > 0.0001 or orientation_error_rad > math.radians(0.01):
                    raise RuntimeError(
                        "BottleCap did not reach the rigidly transformed pose: "
                        f"{position_error_m * 1000.0:.4f} mm / "
                        f"{math.degrees(orientation_error_rad):.6f} deg"
                    )
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("no active server viewport is available")
                screenshot_path = os.path.join(
                    DEFAULT_LOG_DIR, "rotate_bottle_midpoint_with_cap_after.png"
                )
                helper = capture_viewport_to_file(viewport, file_path=screenshot_path, is_hdr=False)
                await helper.wait_for_result()
                result.update(
                    {
                        "status": "PASS",
                        "component": "cap_only repair after Bottle body rotation",
                        "midpoint_world_m": pivot.tolist(),
                        "cap_before": {
                            "position_m": cap_before_position.tolist(),
                            "orientation_wxyz": cap_before_orientation.tolist(),
                        },
                        "cap_after": {
                            "position_m": cap_after_position.tolist(),
                            "orientation_wxyz": cap_after_orientation.tolist(),
                        },
                        "position_error_m": position_error_m,
                        "orientation_error_rad": orientation_error_rad,
                        "screenshot": screenshot_path,
                    }
                )
                self._set_status(
                    "Bottle body and separate BottleCap now share the same 180 deg midpoint transform. "
                    "Timeline remains Paused; inspect the server viewport."
                )
                return

            if component != "bottle":
                raise RuntimeError(f"unsupported rotation component: {component}")
            bottle_length_m = float(request.get("bottle_length_m", 0.206))
            if not 0.05 <= bottle_length_m <= 0.50:
                raise RuntimeError(f"implausible Bottle length: {bottle_length_m}")

            if self._timeline.is_playing():
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            before_position, before_orientation = get_world_pose(BOTTLE_PATH)
            before_position = np.asarray(before_position, dtype=np.float64)
            before_orientation = _quat_normalize(before_orientation)
            local_midpoint = np.asarray([0.0, 0.0, 0.5 * bottle_length_m], dtype=np.float64)
            midpoint_world = before_position + _quat_to_rotation(before_orientation) @ local_midpoint

            world_z_180 = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            requested_orientation = _quat_multiply(world_z_180, before_orientation)
            requested_position = midpoint_world - _quat_to_rotation(requested_orientation) @ local_midpoint

            bottle = SingleXFormPrim(BOTTLE_PATH, name="bottle_midpoint_rotation", reset_xform_properties=False)
            bottle.set_world_pose(position=requested_position, orientation=requested_orientation)
            for _ in range(8):
                await app.next_update_async()

            after_position, after_orientation = get_world_pose(BOTTLE_PATH)
            after_position = np.asarray(after_position, dtype=np.float64)
            after_orientation = _quat_normalize(after_orientation)
            after_midpoint_world = after_position + _quat_to_rotation(after_orientation) @ local_midpoint
            midpoint_error_m = float(np.linalg.norm(after_midpoint_world - midpoint_world))
            rotation_error_rad = abs(math.pi - _quat_angle(before_orientation, after_orientation))
            if midpoint_error_m > 0.0001:
                raise RuntimeError(f"Bottle midpoint drifted {midpoint_error_m * 1000.0:.4f} mm")
            if rotation_error_rad > math.radians(0.01):
                raise RuntimeError(
                    f"Bottle rotation differs from 180 degrees by {math.degrees(rotation_error_rad):.6f} deg"
                )

            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("no active server viewport is available")
            screenshot_path = os.path.join(DEFAULT_LOG_DIR, "rotate_bottle_midpoint_after.png")
            helper = capture_viewport_to_file(viewport, file_path=screenshot_path, is_hdr=False)
            await helper.wait_for_result()

            result.update(
                {
                    "status": "PASS",
                    "before": {
                        "position_m": before_position.tolist(),
                        "orientation_wxyz": before_orientation.tolist(),
                    },
                    "after": {
                        "position_m": after_position.tolist(),
                        "orientation_wxyz": after_orientation.tolist(),
                    },
                    "midpoint_world_m": midpoint_world.tolist(),
                    "midpoint_error_m": midpoint_error_m,
                    "rotation_deg": math.degrees(_quat_angle(before_orientation, after_orientation)),
                    "screenshot": screenshot_path,
                }
            )
            self._set_status(
                "Bottle rotated 180 deg around its geometric midpoint about world Z. "
                "Timeline remains Paused; inspect the server viewport."
            )
        except Exception as exc:
            self._timeline.pause()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Bottle midpoint rotation failed: {exc}", warn=True)
        finally:
            with open(BOTTLE_ROTATE_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_create_thread_base_joints(self) -> None:
        """Author and verify the Bottle thread prismatic and revolute joints only."""

        app = omni.kit.app.get_app()
        running_path = THREAD_BASE_JOINT_REQUEST_PATH + ".running"
        scope_path = "/World/ALOHA1RemoteBottleSession/BottleThreadJoints"
        slider_path = "/World/ALOHA1RemoteBottleSession/BottleThreadSlider"
        prismatic_path = f"{scope_path}/ThreadPrismatic"
        revolute_path = f"{scope_path}/ThreadRevolute"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "timeline_played": False,
            "transforms_commanded": False,
            "rack_and_pinion_created": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(THREAD_BASE_JOINT_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("joint request must explicitly set simulation_only=true")
            if request.get("create") != ["ThreadPrismatic", "ThreadRevolute"]:
                raise RuntimeError("request must name exactly ThreadPrismatic and ThreadRevolute")

            if self._timeline.is_playing():
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("no active USD Stage")
            for path in (BOTTLE_PATH, BOTTLE_CAP_PATH):
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"required prim is missing: {path}")
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(f"required prim lacks PhysicsRigidBodyAPI: {path}")

            # BottleThreadSlider is a runtime-only, collisionless proxy body.  A clean
            # Stage reload intentionally removes it, so the base-joint request must be
            # self-contained instead of requiring a manually-created stale prim.
            slider_created = False
            slider_prim = stage.GetPrimAtPath(slider_path)
            if not slider_prim or not slider_prim.IsValid():
                session_path = "/World/ALOHA1RemoteBottleSession"
                session_prim = stage.GetPrimAtPath(session_path)
                cap_prim = stage.GetPrimAtPath(BOTTLE_CAP_PATH)
                if not session_prim or not session_prim.IsValid():
                    raise RuntimeError(f"required parent prim is missing: {session_path}")
                relative_transform, _ = UsdGeom.XformCache(
                    Usd.TimeCode.Default()
                ).ComputeRelativeTransform(cap_prim, session_prim)
                slider_prim = stage.DefinePrim(slider_path, "Xform")
                slider_xformable = UsdGeom.Xformable(slider_prim)
                relative_translation = relative_transform.ExtractTranslation()
                relative_rotation = relative_transform.ExtractRotationQuat()
                relative_rotation_imaginary = relative_rotation.GetImaginary()
                slider_xformable.AddTranslateOp().Set(
                    Gf.Vec3d(
                        float(relative_translation[0]),
                        float(relative_translation[1]),
                        float(relative_translation[2]),
                    )
                )
                slider_xformable.AddOrientOp().Set(
                    Gf.Quatf(
                        float(relative_rotation.GetReal()),
                        Gf.Vec3f(
                            float(relative_rotation_imaginary[0]),
                            float(relative_rotation_imaginary[1]),
                            float(relative_rotation_imaginary[2]),
                        ),
                    )
                )
                UsdPhysics.RigidBodyAPI.Apply(slider_prim).CreateKinematicEnabledAttr().Set(False)
                slider_created = True
                result["transforms_commanded"] = True
            elif slider_prim.GetTypeName() != "Xform":
                raise RuntimeError(
                    f"expected Xform at {slider_path}, found {slider_prim.GetTypeName()}"
                )
            if not slider_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"required prim lacks PhysicsRigidBodyAPI: {slider_path}")

            bottle_position, bottle_orientation = get_world_pose(BOTTLE_PATH)
            cap_position, cap_orientation = get_world_pose(BOTTLE_CAP_PATH)
            slider_position, slider_orientation = get_world_pose(slider_path)
            bottle_position = np.asarray(bottle_position, dtype=np.float64)
            cap_position = np.asarray(cap_position, dtype=np.float64)
            slider_position = np.asarray(slider_position, dtype=np.float64)
            bottle_orientation = _quat_normalize(bottle_orientation)
            cap_orientation = _quat_normalize(cap_orientation)
            slider_orientation = _quat_normalize(slider_orientation)

            prismatic_anchor0 = bottle_position + _quat_to_rotation(
                bottle_orientation
            ) @ np.asarray([0.0, 0.0, 0.188], dtype=np.float64)
            prismatic_anchor1 = slider_position.copy()
            prismatic_anchor_error_m = float(
                np.linalg.norm(prismatic_anchor0 - prismatic_anchor1)
            )
            revolute_anchor_error_m = float(np.linalg.norm(slider_position - cap_position))
            bottle_slider_orientation_error_rad = _quat_angle(
                bottle_orientation, slider_orientation
            )
            slider_cap_orientation_error_rad = _quat_angle(slider_orientation, cap_orientation)
            result["preflight"] = {
                "bottle_world_position_m": bottle_position.tolist(),
                "bottle_world_orientation_wxyz": bottle_orientation.tolist(),
                "cap_world_position_m": cap_position.tolist(),
                "cap_world_orientation_wxyz": cap_orientation.tolist(),
                "slider_world_position_m": slider_position.tolist(),
                "slider_world_orientation_wxyz": slider_orientation.tolist(),
                "prismatic_anchor0_world_m": prismatic_anchor0.tolist(),
                "prismatic_anchor1_world_m": prismatic_anchor1.tolist(),
                "prismatic_anchor_error_m": prismatic_anchor_error_m,
                "revolute_anchor_error_m": revolute_anchor_error_m,
                "bottle_slider_orientation_error_rad": bottle_slider_orientation_error_rad,
                "slider_cap_orientation_error_rad": slider_cap_orientation_error_rad,
            }
            if prismatic_anchor_error_m > 0.0001:
                raise RuntimeError(
                    f"prismatic anchors differ by {prismatic_anchor_error_m * 1000.0:.3f} mm"
                )
            if revolute_anchor_error_m > 0.0001:
                raise RuntimeError(
                    f"revolute anchors differ by {revolute_anchor_error_m * 1000.0:.3f} mm"
                )
            if bottle_slider_orientation_error_rad > math.radians(0.01):
                raise RuntimeError("Bottle and BottleThreadSlider orientations do not match")
            if slider_cap_orientation_error_rad > math.radians(0.01):
                raise RuntimeError("BottleThreadSlider and BottleCap orientations do not match")

            scope = stage.GetPrimAtPath(scope_path)
            if not scope or not scope.IsValid():
                scope = stage.DefinePrim(scope_path, "Scope")
            elif scope.GetTypeName() != "Scope":
                raise RuntimeError(f"expected Scope at {scope_path}, found {scope.GetTypeName()}")

            expected_types = {
                prismatic_path: "PhysicsPrismaticJoint",
                revolute_path: "PhysicsRevoluteJoint",
            }
            for path, expected_type in expected_types.items():
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid() and prim.GetTypeName() != expected_type:
                    raise RuntimeError(
                        f"refusing to replace {path}: expected {expected_type}, found {prim.GetTypeName()}"
                    )

            identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
            zero = Gf.Vec3f(0.0, 0.0, 0.0)

            prismatic = UsdPhysics.PrismaticJoint.Define(stage, prismatic_path)
            prismatic.CreateBody0Rel().SetTargets([Sdf.Path(BOTTLE_PATH)])
            prismatic.CreateBody1Rel().SetTargets([Sdf.Path(slider_path)])
            prismatic.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
            prismatic.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.188))
            prismatic.CreateLocalPos1Attr().Set(zero)
            prismatic.CreateLocalRot0Attr().Set(identity)
            prismatic.CreateLocalRot1Attr().Set(identity)
            prismatic.CreateLowerLimitAttr().Set(0.0)
            prismatic.CreateUpperLimitAttr().Set(0.012)
            prismatic.CreateCollisionEnabledAttr().Set(False)

            revolute = UsdPhysics.RevoluteJoint.Define(stage, revolute_path)
            revolute.CreateBody0Rel().SetTargets([Sdf.Path(slider_path)])
            revolute.CreateBody1Rel().SetTargets([Sdf.Path(BOTTLE_CAP_PATH)])
            revolute.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
            revolute.CreateLocalPos0Attr().Set(zero)
            revolute.CreateLocalPos1Attr().Set(zero)
            revolute.CreateLocalRot0Attr().Set(identity)
            revolute.CreateLocalRot1Attr().Set(identity)
            revolute.CreateCollisionEnabledAttr().Set(False)

            for _ in range(5):
                await app.next_update_async()

            prismatic_readback = UsdPhysics.PrismaticJoint.Get(stage, prismatic_path)
            revolute_readback = UsdPhysics.RevoluteJoint.Get(stage, revolute_path)
            checks = {
                "scope_valid": stage.GetPrimAtPath(scope_path).GetTypeName() == "Scope",
                "prismatic_type_valid": bool(prismatic_readback),
                "prismatic_body0_valid": prismatic_readback.GetBody0Rel().GetTargets()
                == [Sdf.Path(BOTTLE_PATH)],
                "prismatic_body1_valid": prismatic_readback.GetBody1Rel().GetTargets()
                == [Sdf.Path(slider_path)],
                "prismatic_axis_valid": prismatic_readback.GetAxisAttr().Get()
                == UsdPhysics.Tokens.z,
                "prismatic_limits_valid": abs(
                    float(prismatic_readback.GetLowerLimitAttr().Get())
                )
                < 1e-9
                and abs(float(prismatic_readback.GetUpperLimitAttr().Get()) - 0.012) < 1e-9,
                "revolute_type_valid": bool(revolute_readback),
                "revolute_body0_valid": revolute_readback.GetBody0Rel().GetTargets()
                == [Sdf.Path(slider_path)],
                "revolute_body1_valid": revolute_readback.GetBody1Rel().GetTargets()
                == [Sdf.Path(BOTTLE_CAP_PATH)],
                "revolute_axis_valid": revolute_readback.GetAxisAttr().Get()
                == UsdPhysics.Tokens.z,
                "timeline_paused": not self._timeline.is_playing(),
            }
            if not all(checks.values()):
                raise RuntimeError(f"joint readback validation failed: {checks}")
            result.update(
                {
                    "status": "PASS",
                    "checks": checks,
                    "slider_created": slider_created,
                    "edit_target_layer": stage.GetEditTarget().GetLayer().identifier,
                    "created_or_updated_paths": (
                        [slider_path, prismatic_path, revolute_path]
                        if slider_created
                        else [prismatic_path, revolute_path]
                    ),
                    "next_required_joint": "RightHandThreadCoupling",
                }
            )
            self._set_status(
                "ThreadPrismatic and ThreadRevolute created and verified. Timeline remains Paused; "
                "Rack-and-Pinion coupling has not been created."
            )
        except Exception as exc:
            self._timeline.pause()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Thread base-joint creation failed: {exc}", warn=True)
        finally:
            with open(THREAD_BASE_JOINT_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_create_thread_coupling(self) -> None:
        """Author and verify the right-hand rack-and-pinion thread coupling only."""

        app = omni.kit.app.get_app()
        running_path = THREAD_COUPLING_REQUEST_PATH + ".running"
        scope_path = "/World/ALOHA1RemoteBottleSession/BottleThreadJoints"
        slider_path = "/World/ALOHA1RemoteBottleSession/BottleThreadSlider"
        prismatic_path = f"{scope_path}/ThreadPrismatic"
        revolute_path = f"{scope_path}/ThreadRevolute"
        coupling_path = f"{scope_path}/RightHandThreadCoupling"
        pitch_m_per_turn = 0.003
        axial_travel_m = 0.012
        ratio_deg_per_m = -360.0 / pitch_m_per_turn
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "timeline_played": False,
            "transforms_commanded": False,
            "kinematic_state_changed": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(THREAD_COUPLING_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("coupling request must explicitly set simulation_only=true")
            if request.get("create") != ["RightHandThreadCoupling"]:
                raise RuntimeError("request must name exactly RightHandThreadCoupling")
            if request.get("save_stage", False):
                raise RuntimeError("this runtime operation refuses save_stage=true")
            if request.get("play_timeline", False):
                raise RuntimeError("this authoring operation refuses play_timeline=true")

            if self._timeline.is_playing():
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("no active USD Stage")
            required_types = {
                BOTTLE_PATH: None,
                BOTTLE_CAP_PATH: None,
                slider_path: None,
                prismatic_path: "PhysicsPrismaticJoint",
                revolute_path: "PhysicsRevoluteJoint",
            }
            for path, expected_type in required_types.items():
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"required prim is missing: {path}")
                if expected_type is not None and prim.GetTypeName() != expected_type:
                    raise RuntimeError(
                        f"required prim {path} has type {prim.GetTypeName()}, expected {expected_type}"
                    )
            existing = stage.GetPrimAtPath(coupling_path)
            expected_coupling_type = "PhysxPhysicsRackAndPinionJoint"
            if existing and existing.IsValid() and existing.GetTypeName() != expected_coupling_type:
                raise RuntimeError(
                    f"refusing to replace {coupling_path}: expected {expected_coupling_type}, "
                    f"found {existing.GetTypeName()}"
                )

            coupling = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(stage, coupling_path)
            prim = coupling.GetPrim()
            prim.CreateRelationship("physics:body0").SetTargets([Sdf.Path(BOTTLE_CAP_PATH)])
            prim.CreateRelationship("physics:body1").SetTargets([Sdf.Path(slider_path)])
            prim.CreateRelationship("physics:hinge").SetTargets([Sdf.Path(revolute_path)])
            prim.CreateRelationship("physics:prismatic").SetTargets([Sdf.Path(prismatic_path)])
            prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            prim.CreateAttribute("physics:ratio", Sdf.ValueTypeNames.Float).Set(ratio_deg_per_m)
            prim.SetCustomDataByKey("axialTravelM", axial_travel_m)
            prim.SetCustomDataByKey("calibrationStatus", "TEMPORARY_UNCALIBRATED")
            prim.SetCustomDataByKey("pitchMPerTurn", pitch_m_per_turn)
            prim.SetCustomDataByKey("positiveRemovalRotation", "CAP_POSITIVE_LOCAL_Z")
            prim.SetCustomDataByKey("threadHandedness", "RIGHT_HAND")

            for _ in range(5):
                await app.next_update_async()

            readback = stage.GetPrimAtPath(coupling_path)
            checks = {
                "coupling_type_valid": readback.GetTypeName() == expected_coupling_type,
                "body0_cap_valid": readback.GetRelationship("physics:body0").GetTargets()
                == [Sdf.Path(BOTTLE_CAP_PATH)],
                "body1_slider_valid": readback.GetRelationship("physics:body1").GetTargets()
                == [Sdf.Path(slider_path)],
                "hinge_relation_valid": readback.GetRelationship("physics:hinge").GetTargets()
                == [Sdf.Path(revolute_path)],
                "prismatic_relation_valid": readback.GetRelationship("physics:prismatic").GetTargets()
                == [Sdf.Path(prismatic_path)],
                "ratio_valid": abs(float(readback.GetAttribute("physics:ratio").Get()) - ratio_deg_per_m)
                < 0.001,
                "collision_disabled": readback.GetAttribute("physics:collisionEnabled").Get() is False,
                "right_hand_metadata_valid": readback.GetCustomDataByKey("threadHandedness")
                == "RIGHT_HAND",
                "pitch_metadata_valid": abs(
                    float(readback.GetCustomDataByKey("pitchMPerTurn")) - pitch_m_per_turn
                )
                < 1e-12,
                "timeline_paused": not self._timeline.is_playing(),
            }
            if not all(checks.values()):
                raise RuntimeError(f"thread-coupling readback validation failed: {checks}")
            result.update(
                {
                    "status": "PASS",
                    "checks": checks,
                    "edit_target_layer": stage.GetEditTarget().GetLayer().identifier,
                    "created_or_updated_paths": [coupling_path],
                    "pitch_m_per_turn": pitch_m_per_turn,
                    "axial_travel_m": axial_travel_m,
                    "ratio_deg_per_m": ratio_deg_per_m,
                    "positive_rotation_axis": "BOTTLE_LOCAL_POSITIVE_Z",
                    "expected_positive_translation_axis": "BOTTLE_LOCAL_POSITIVE_Z",
                    "next_step": "Run isaac_script/bottle_thread_coupling_test.py in Script Editor",
                }
            )
            self._set_status(
                "RightHandThreadCoupling created and verified. Timeline remains Paused; "
                "no transforms or kinematic state were changed."
            )
        except Exception as exc:
            self._timeline.pause()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Thread-coupling creation failed: {exc}", warn=True)
        finally:
            with open(THREAD_COUPLING_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_thread_release_test_request(self) -> None:
        """Run the standalone THREADED -> RELEASED test inside active Kit."""

        app = omni.kit.app.get_app()
        running_path = THREAD_RELEASE_TEST_REQUEST_PATH + ".running"
        script_path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_release_test.py"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "ros_used": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(THREAD_RELEASE_TEST_REQUEST_PATH, running_path)
            previous_size = -1
            stable_reads = 0
            for _ in range(30):
                current_size = os.path.getsize(running_path)
                if current_size > 0 and current_size == previous_size:
                    stable_reads += 1
                    if stable_reads >= 2:
                        break
                else:
                    stable_reads = 0
                previous_size = current_size
                await app.next_update_async()
            if stable_reads < 2:
                raise RuntimeError("release-test request file did not become stable and non-empty")
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("release-test request must set simulation_only=true")
            if request.get("transition") != "THREADED_TO_RELEASED":
                raise RuntimeError("release-test request must name THREADED_TO_RELEASED")
            if request.get("save_stage", False):
                raise RuntimeError("release test refuses save_stage=true")
            if not os.path.isfile(script_path):
                raise RuntimeError(f"release test script is missing: {script_path}")
            if self._timeline.is_playing():
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            if os.path.isfile(THREAD_RELEASE_TEST_RESULT_PATH):
                os.remove(THREAD_RELEASE_TEST_RESULT_PATH)

            namespace = {
                "__name__": "__main__",
                "__file__": script_path,
            }
            with open(script_path, "rb") as stream:
                code = compile(stream.read(), script_path, "exec")
            exec(code, namespace, namespace)

            for update in range(6000):
                await app.next_update_async()
                if update % 10 == 0 and os.path.isfile(THREAD_RELEASE_TEST_RESULT_PATH):
                    break
            if not os.path.isfile(THREAD_RELEASE_TEST_RESULT_PATH):
                raise RuntimeError("release test did not produce a result before the update limit")
            with open(THREAD_RELEASE_TEST_RESULT_PATH, "r", encoding="utf-8") as stream:
                result = json.load(stream)
            if result.get("status") == "PASS":
                self._set_status(
                    "THREADED -> RELEASED PASS. Thread joints are disabled; BottleCap is Dynamic "
                    "with gravity enabled. Timeline remains Paused."
                )
            else:
                self._set_status(
                    f"THREADED -> RELEASED test finished with {result.get('status')}: "
                    f"{result.get('error', result.get('checks', 'see report'))}",
                    warn=True,
                )
        except Exception as exc:
            self._timeline.pause()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            with open(THREAD_RELEASE_TEST_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self._set_status(f"Thread release test request failed: {exc}", warn=True)
        finally:
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_thread_release_verify_request(self) -> None:
        """Run the standalone dynamic verification for an existing RELEASED state."""

        app = omni.kit.app.get_app()
        running_path = THREAD_RELEASE_VERIFY_REQUEST_PATH + ".running"
        script_path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_release_verify.py"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "ros_used": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(THREAD_RELEASE_VERIFY_REQUEST_PATH, running_path)
            previous_size = -1
            stable_reads = 0
            for _ in range(30):
                current_size = os.path.getsize(running_path)
                if current_size > 0 and current_size == previous_size:
                    stable_reads += 1
                    if stable_reads >= 2:
                        break
                else:
                    stable_reads = 0
                previous_size = current_size
                await app.next_update_async()
            if stable_reads < 2:
                raise RuntimeError("release-verification request did not become stable")
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("release-verification request must set simulation_only=true")
            if request.get("verify") != "RELEASED_DYNAMIC":
                raise RuntimeError("release-verification request must name RELEASED_DYNAMIC")
            if request.get("save_stage", False):
                raise RuntimeError("release verification refuses save_stage=true")
            if not os.path.isfile(script_path):
                raise RuntimeError(f"release verification script is missing: {script_path}")
            if self._timeline.is_playing():
                self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            if os.path.isfile(THREAD_RELEASE_VERIFY_RESULT_PATH):
                os.remove(THREAD_RELEASE_VERIFY_RESULT_PATH)
            namespace = {"__name__": "__main__", "__file__": script_path}
            with open(script_path, "rb") as stream:
                code = compile(stream.read(), script_path, "exec")
            exec(code, namespace, namespace)
            for update in range(3000):
                await app.next_update_async()
                if update % 10 == 0 and os.path.isfile(THREAD_RELEASE_VERIFY_RESULT_PATH):
                    break
            if not os.path.isfile(THREAD_RELEASE_VERIFY_RESULT_PATH):
                raise RuntimeError("release verification did not produce a result")
            with open(THREAD_RELEASE_VERIFY_RESULT_PATH, "r", encoding="utf-8") as stream:
                result = json.load(stream)
            if result.get("status") == "PASS":
                self._set_status(
                    "RELEASED dynamic verification PASS. BottleCap moved independently; "
                    "all thread joints remain disabled and Timeline is Paused."
                )
            else:
                self._set_status(
                    f"RELEASED dynamic verification finished with {result.get('status')}: "
                    f"{result.get('error', result.get('checks', 'see report'))}",
                    warn=True,
                )
        except Exception as exc:
            self._timeline.pause()
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            with open(THREAD_RELEASE_VERIFY_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self._set_status(f"Released-state verification request failed: {exc}", warn=True)
        finally:
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    async def _run_auto_hover_acceptance(self) -> None:
        """Precheck and optionally execute the corrected HOVER route in simulation."""

        app = omni.kit.app.get_app()
        running_path = AUTO_HOVER_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "bottle_commanded": False,
            "gripper_commanded": False,
        }

        def write_progress(phase: str, **details: object) -> None:
            progress = {
                "phase": phase,
                "timeline_playing": bool(self._timeline.is_playing()),
                "timeline_time_s": float(self._timeline.get_current_time()),
                "active_waypoint": str(self._active_waypoint),
                "route_sample_index": int(self._hover_plan_index),
                "route_sample_count": int(len(self._hover_plan_positions)),
                "joint_log_samples": int(len(self._joint_log_rows)),
                **details,
            }
            temporary = AUTO_HOVER_PROGRESS_PATH + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(progress, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, AUTO_HOVER_PROGRESS_PATH)

        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(AUTO_HOVER_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            write_progress("request_loaded")
            execute = bool(request.get("execute", False))
            capture_viewport = bool(request.get("capture_viewport", False))

            # When the extension is enabled from the Kit command line it can
            # start a few frames before the --exec stage loader has finished.
            # Do not mistake that startup ordering for an IK/configuration
            # failure: wait until both required runtime prims are available.
            startup_wait_updates = 0
            startup_wait_limit = int(request.get("max_updates", 3000))
            while not (
                is_prim_path_valid(LEFT_ARTICULATION_PATH)
                and is_prim_path_valid(BOTTLE_PATH)
            ):
                if startup_wait_updates >= startup_wait_limit:
                    raise RuntimeError(
                        "timed out waiting for the loaded ALOHA articulation and Bottle prim"
                    )
                await app.next_update_async()
                startup_wait_updates += 1
            result["startup_wait_updates"] = startup_wait_updates
            write_progress("runtime_prims_ready", startup_wait_updates=startup_wait_updates)

            viewport = None
            capture_dir = ""
            capture_index = 0
            if capture_viewport:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("no active server viewport is available")
                safe_run_id = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in str(request.get("run_id", "auto_hover"))
                )
                capture_dir = os.path.join(DEFAULT_LOG_DIR, "viewport_recordings", safe_run_id)
                os.makedirs(capture_dir, exist_ok=False)
                result["viewport_recording"] = {
                    "frame_directory": capture_dir,
                    "source": "Isaac Sim active server viewport",
                    "desktop_switched": False,
                }

                async def capture_frame() -> None:
                    nonlocal capture_index
                    output = os.path.join(capture_dir, f"frame_{capture_index:05d}.png")
                    helper = capture_viewport_to_file(viewport, file_path=output, is_hdr=False)
                    await helper.wait_for_result()
                    capture_index += 1

            for _ in range(5):
                await app.next_update_async()
            self._disable_follow()
            self._joint_log_enabled = False
            if is_prim_path_valid(TARGET_PATH):
                delete_prim(TARGET_PATH)
            self._target = None
            if self._timeline.is_stopped():
                self._timeline.play()
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            self._reset_left_sleep()
            self._timeline.play()
            for _ in range(30):
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(5):
                await app.next_update_async()
            self._load_left_arm()
            self._sync_base_pose()
            for _ in range(3):
                await app.next_update_async()
            if not self._validate_alignment():
                raise RuntimeError(
                    f"initial alignment failed: {self._last_position_error * 1000.0:.4f} mm, "
                    f"{math.degrees(self._last_orientation_error):.4f} deg"
                )
            self._load_bottle_grasp()
            self._on_create_target()
            self._enable_follow()
            write_progress("hover_precheck_started")
            self._plan_hover_route()
            write_progress("hover_precheck_passed")
            result["precheck"] = dict(self._hover_plan_metrics)
            result["hover_goal_world"] = {
                "position_m": self._hover_plan_goal_position.tolist(),
                "orientation_wxyz": self._hover_plan_goal_orientation.tolist(),
            }
            if not execute:
                result["status"] = "PRECHECK_PASS"
                self._set_status(
                    f"Automated HOVER PRECHECK PASS; no motion executed. Result: {AUTO_HOVER_RESULT_PATH}"
                )
                return

            self._on_start_joint_log()
            if capture_viewport:
                await capture_frame()
            bottle_start_position, bottle_start_orientation = get_world_pose(BOTTLE_PATH)
            self._timeline.play()
            reached_samples = 0
            max_updates = int(request.get("max_updates", 6000))
            if not 1800 <= max_updates <= 12000:
                raise ValueError("max_updates must be within [1800, 12000]")
            for update_index in range(max_updates):
                await app.next_update_async()
                if update_index % 100 == 0:
                    write_progress("hover_executing", update_index=update_index)
                if capture_viewport and update_index % 10 == 0 and capture_index < 480:
                    await capture_frame()
                if not self._follow_enabled:
                    raise RuntimeError("HOVER execution stopped before reaching the route goal")
                if self._active_waypoint == "HOVER / REACHED":
                    subset = self._art_ik.get_joints_subset()
                    peak_velocity = float(np.max(np.abs(np.asarray(subset.get_joint_velocities()))))
                    reached_samples = reached_samples + 1 if peak_velocity <= 0.01 else 0
                    if reached_samples >= 25:
                        break
            self._timeline.pause()
            for _ in range(5):
                await app.next_update_async()
            if self._active_waypoint != "HOVER / REACHED":
                raise RuntimeError("HOVER route timed out before reaching the final joint sample")
            if capture_viewport:
                await capture_frame()
                result["viewport_recording"]["frame_count"] = capture_index
            write_progress("hover_reached")

            final_position, final_orientation = self._current_lula_ee_pose()
            bottle_end_position, bottle_end_orientation = get_world_pose(BOTTLE_PATH)
            static_alignment_position, static_alignment_orientation = self._compute_alignment_errors()
            result["final"] = {
                "ee_position_m": final_position.tolist(),
                "ee_orientation_wxyz": final_orientation.tolist(),
                "goal_position_error_m": float(
                    np.linalg.norm(final_position - self._hover_plan_goal_position)
                ),
                "goal_orientation_error_rad": _quat_angle(
                    final_orientation, self._hover_plan_goal_orientation
                ),
                "bottle_translation_m": float(
                    np.linalg.norm(np.asarray(bottle_end_position) - np.asarray(bottle_start_position))
                ),
                "bottle_orientation_change_rad": _quat_angle(
                    np.asarray(bottle_end_orientation), np.asarray(bottle_start_orientation)
                ),
                "static_alignment_position_error_m": static_alignment_position,
                "static_alignment_orientation_error_rad": static_alignment_orientation,
            }
            if static_alignment_position > POSITION_GATE_M or static_alignment_orientation > ORIENTATION_GATE_RAD:
                raise RuntimeError(
                    f"static final EE alignment failed: {static_alignment_position * 1000.0:.3f} mm, "
                    f"{math.degrees(static_alignment_orientation):.3f} deg"
                )
            if result["final"]["bottle_translation_m"] > 0.001:
                raise RuntimeError(
                    f"HOVER disturbed Bottle by {result['final']['bottle_translation_m'] * 1000.0:.3f} mm"
                )

            requested_sequence = request.get("waypoint_sequence")
            if requested_sequence is None:
                next_waypoint = str(request.get("next_waypoint", "")).strip().upper()
                waypoint_sequence = [next_waypoint] if next_waypoint else []
            else:
                if not isinstance(requested_sequence, list):
                    raise ValueError("waypoint_sequence must be a list")
                waypoint_sequence = [str(value).strip().upper() for value in requested_sequence]
            waypoint_clearances = {
                "PREGRASP": PREGRASP_CLEARANCE_M,
                "NEAR": NEAR_CLEARANCE_M,
                "GRASP_POSE": 0.0,
            }
            if any(name not in waypoint_clearances for name in waypoint_sequence):
                raise ValueError(f"unsupported automated waypoint sequence: {waypoint_sequence}")

            waypoint_results: List[Dict[str, object]] = []
            for waypoint_name in waypoint_sequence:
                write_progress(f"{waypoint_name.lower()}_precheck_started")
                self._plan_guided_waypoint_route(
                    waypoint_name, waypoint_clearances[waypoint_name]
                )
                waypoint_goal_position = self._hover_plan_goal_position.copy()
                waypoint_goal_orientation = self._hover_plan_goal_orientation.copy()
                waypoint_result: Dict[str, object] = {
                    "name": waypoint_name,
                    "precheck": dict(self._hover_plan_metrics),
                    "goal_world": {
                        "position_m": waypoint_goal_position.tolist(),
                        "orientation_wxyz": waypoint_goal_orientation.tolist(),
                    },
                }
                reached_samples = 0
                self._timeline.play()
                for update_index in range(max_updates):
                    await app.next_update_async()
                    if update_index % 100 == 0:
                        write_progress(
                            f"{waypoint_name.lower()}_executing",
                            update_index=update_index,
                        )
                    if capture_viewport and update_index % 10 == 0 and capture_index < 480:
                        await capture_frame()
                    if not self._follow_enabled:
                        raise RuntimeError(
                            f"{waypoint_name} execution stopped before reaching the route goal"
                        )
                    if self._active_waypoint == f"{waypoint_name} / REACHED":
                        subset = self._art_ik.get_joints_subset()
                        peak_velocity = float(
                            np.max(np.abs(np.asarray(subset.get_joint_velocities())))
                        )
                        reached_samples = reached_samples + 1 if peak_velocity <= 0.01 else 0
                        if reached_samples >= 25:
                            break
                self._timeline.pause()
                for _ in range(5):
                    await app.next_update_async()
                if self._active_waypoint != f"{waypoint_name} / REACHED":
                    raise RuntimeError(
                        f"{waypoint_name} route timed out before reaching the final joint sample"
                    )
                if capture_viewport:
                    await capture_frame()
                    result["viewport_recording"]["frame_count"] = capture_index
                waypoint_final_position, waypoint_final_orientation = self._current_lula_ee_pose()
                waypoint_bottle_position, waypoint_bottle_orientation = get_world_pose(BOTTLE_PATH)
                waypoint_result["final"] = {
                    "ee_position_m": waypoint_final_position.tolist(),
                    "ee_orientation_wxyz": waypoint_final_orientation.tolist(),
                    "goal_position_error_m": float(
                        np.linalg.norm(waypoint_final_position - waypoint_goal_position)
                    ),
                    "goal_orientation_error_rad": _quat_angle(
                        waypoint_final_orientation, waypoint_goal_orientation
                    ),
                    "bottle_translation_from_hover_start_m": float(
                        np.linalg.norm(
                            np.asarray(waypoint_bottle_position)
                            - np.asarray(bottle_start_position)
                        )
                    ),
                    "bottle_orientation_change_from_hover_start_rad": _quat_angle(
                        np.asarray(waypoint_bottle_orientation),
                        np.asarray(bottle_start_orientation),
                    ),
                }
                if waypoint_result["final"]["goal_position_error_m"] > RUNTIME_POSITION_GATE_M:
                    raise RuntimeError(f"{waypoint_name} final position error exceeds the runtime gate")
                if waypoint_result["final"]["goal_orientation_error_rad"] > RUNTIME_ORIENTATION_GATE_RAD:
                    raise RuntimeError(f"{waypoint_name} final orientation error exceeds the runtime gate")
                if waypoint_result["final"]["bottle_translation_from_hover_start_m"] > 0.001:
                    raise RuntimeError(f"{waypoint_name} disturbed Bottle by more than 1 mm")
                waypoint_results.append(waypoint_result)
                write_progress(f"{waypoint_name.lower()}_reached")

                if waypoint_name == "PREGRASP" and bool(request.get("open_at_pregrasp", False)):
                    result["gripper_open"] = await self._open_left_gripper_transaction()

            if waypoint_results:
                result["waypoints"] = waypoint_results

            if bool(request.get("calibrate_unilateral_contact", False)):
                if not waypoint_sequence or waypoint_sequence[-1] != "GRASP_POSE":
                    raise RuntimeError(
                        "unilateral contact calibration requires GRASP_POSE as the final waypoint"
                    )
                self._require_gripper_calibration_ready()
                self._ensure_grasp_contact_monitor()
                calibration_steps: List[Dict[str, object]] = []
                # A 1 mm relative target produces too little closing effort in
                # this vertical wrist pose: the finite-stiffness drive can hold
                # a static tracking offset before reaching the bottle.  Use the
                # normal position-gripper transaction instead: command the
                # validated closed target once and stop as soon as either pad
                # reports stable contact.  Bottle500 remains kinematic here.
                readback = await self._execute_gripper_target(
                    LEFT_GRIPPER_MIN_POSITION_M, stop_on_bilateral=True
                )
                calibration_steps.append(dict(readback))
                write_progress(
                    "unilateral_contact_calibrating",
                    calibration_step=1,
                    command_target_m=float(LEFT_GRIPPER_MIN_POSITION_M),
                    left_actual_m=float(readback["left_actual_m"]),
                    right_actual_m=float(readback["right_actual_m"]),
                    mimic_residual_m=float(readback["mimic_residual_m"]),
                    left_contact=bool(readback["left_contact"]),
                    right_contact=bool(readback["right_contact"]),
                )
                unilateral_reached = bool(
                    readback["left_contact"] and not readback["right_contact"]
                )
                if readback["right_contact"] and not readback["left_contact"]:
                    raise RuntimeError(
                        "right/orange finger contacted first; the required left-only state was not formed"
                    )
                if readback["left_contact"] and readback["right_contact"]:
                    raise RuntimeError(
                        "bilateral contact formed before the required unilateral release state"
                    )
                result["unilateral_calibration"] = {
                    "status": "PASS" if unilateral_reached else "FAILED_GATE",
                    "steps": calibration_steps,
                    "left_contact": bool(self._grasp_left_contact),
                    "right_contact": bool(self._grasp_right_contact),
                    "nonfinger_contact": bool(self._grasp_nonfinger_contact),
                }
                if not unilateral_reached:
                    raise RuntimeError(
                        "reached the gripper lower limit without forming left-only Bottle contact"
                    )
                write_progress("unilateral_contact_ready")

            self._on_stop_joint_log()
            result["joint_log_csv"] = self._last_joint_log_path
            result["status"] = "PASS"
            write_progress("complete", status="PASS")
            self._set_status(
                f"Automated corrected HOVER route PASS. Timeline Paused; result: {AUTO_HOVER_RESULT_PATH}"
            )
        except Exception as exc:
            self._timeline.pause()
            self._disable_follow()
            if self._joint_log_enabled and self._joint_log_rows:
                self._on_stop_joint_log()
                result["joint_log_csv"] = self._last_joint_log_path
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            write_progress("failed", status="EXCEPTION", error=result["error"])
            self._set_status(f"Automated HOVER acceptance failed: {exc}", warn=True)
        finally:
            with open(AUTO_HOVER_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    def _run_guarded(self, operation_name: str, callback) -> None:
        try:
            callback()
        except Exception as exc:
            self._disable_follow()
            message = f"{operation_name} failed: {exc}"
            self._set_status(message, warn=True)
            carb.log_error(traceback.format_exc())

    def _on_load_left_arm(self) -> None:
        self._run_guarded("Load Left Arm", self._load_left_arm)

    def _on_reset_left_sleep(self) -> None:
        if self._reset_task is not None and not self._reset_task.done():
            self._set_status(
                "Sleep reset is already running; wait for the Timeline to return to Paused.",
                warn=True,
            )
            return
        self._reset_task = asyncio.ensure_future(self._reset_left_sleep_from_button())

    async def _reset_left_sleep_from_button(self) -> None:
        """Perform a complete, visible reset transaction from the UI button."""

        app = omni.kit.app.get_app()
        try:
            if self._joint_log_enabled:
                raise RuntimeError("stop and save the active joint log before resetting the left arm")

            # Reset owns the whole transition back to the preparation state.
            # A stale HOVER Target must not turn this button into a no-op.
            self._timeline.pause()
            self._disable_follow()
            if is_prim_path_valid(TARGET_PATH):
                delete_prim(TARGET_PATH)
            self._target = None
            self._clear_hover_plan()
            self._active_waypoint = "none"

            # Initialize the physics view if the Timeline was Stopped.
            if self._timeline.is_stopped():
                self._timeline.play()
                await app.next_update_async()
                self._timeline.pause()
                await app.next_update_async()

            self._reset_left_sleep()

            # Publish the paused state write to USD/Fabric and let the Drive
            # target settle. The operation always returns to Paused.
            self._timeline.play()
            for _ in range(30):
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(5):
                await app.next_update_async()

            readback = np.asarray(
                self._articulation.get_joint_positions(), dtype=np.float64
            )[: len(ARM_JOINTS)]
            maximum_error = float(np.max(np.abs(readback - LEFT_SLEEP_ARM_RAD)))
            if maximum_error > SLEEP_READBACK_GATE_RAD:
                raise RuntimeError(
                    f"post-settle sleep readback error {maximum_error:.6f} rad exceeds "
                    f"{SLEEP_READBACK_GATE_RAD:.3f} rad"
                )
            self._set_status(
                "Sleep reset PASS: IK Follow disabled, extension Target removed, six arm joints and Drive "
                "targets restored, velocities zeroed, gripper DOFs preserved, and Timeline returned to Paused. "
                f"Maximum post-settle readback error: {maximum_error:.6f} rad."
            )
        except Exception as exc:
            self._disable_follow()
            self._set_status(f"Reset Left Arm to Sleep failed: {exc}", warn=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._timeline.pause()
            self._reset_task = None

    async def _run_auto_reset_sleep(self) -> None:
        """Consume a one-shot request and verify the same transaction as the UI button."""

        running_path = AUTO_RESET_SLEEP_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "gripper_commanded": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(AUTO_RESET_SLEEP_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                result["request"] = json.load(stream)
            await self._reset_left_sleep_from_button()
            articulation = self._articulation
            if articulation is None:
                raise RuntimeError("left articulation was not initialized by reset")
            readback = np.asarray(
                articulation.get_joint_positions(), dtype=np.float64
            )[: len(ARM_JOINTS)]
            maximum_error = float(np.max(np.abs(readback - LEFT_SLEEP_ARM_RAD)))
            result.update(
                {
                    "status": "PASS"
                    if maximum_error <= SLEEP_READBACK_GATE_RAD
                    and not is_prim_path_valid(TARGET_PATH)
                    and not self._timeline.is_playing()
                    else "FAIL",
                    "sleep_target_arm_rad": LEFT_SLEEP_ARM_RAD.tolist(),
                    "sleep_readback_arm_rad": readback.tolist(),
                    "maximum_readback_error_rad": maximum_error,
                    "target_removed": not is_prim_path_valid(TARGET_PATH),
                    "timeline_paused": not self._timeline.is_playing(),
                }
            )
        except Exception as exc:
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
        finally:
            self._timeline.pause()
            with open(AUTO_RESET_SLEEP_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")

    def _get_gripper_state(self) -> Tuple[SingleArticulation, int, int, np.ndarray]:
        articulation = self._articulation or SingleArticulation(LEFT_ARTICULATION_PATH)
        if not articulation.handles_initialized:
            articulation.initialize()
        dof_names = list(articulation.dof_names)
        expected_dofs = ARM_JOINTS + ["gripper", "left_finger", "right_finger"]
        if dof_names != expected_dofs:
            raise RuntimeError(f"unexpected left articulation DOF order: {dof_names}")
        positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        if positions.shape != (len(expected_dofs),) or not np.all(np.isfinite(positions)):
            raise RuntimeError(f"invalid left articulation joint state: {positions}")
        self._articulation = articulation
        return (
            articulation,
            dof_names.index("left_finger"),
            dof_names.index("right_finger"),
            positions,
        )

    def _refresh_gripper_state_label(self) -> None:
        if self._gripper_state_label is None:
            return
        try:
            _, left_index, right_index, positions = self._get_gripper_state()
            left = float(positions[left_index])
            right = float(positions[right_index])
            residual = abs(left + right)
            target = (
                f"{self._gripper_command_target_m:.6f}"
                if math.isfinite(self._gripper_command_target_m)
                else "--"
            )
            self._gripper_state_label.text = (
                f"Finger target/actual: left={target}/{left:.6f} m | right(Mimic)={right:.6f} m\n"
                f"Mimic residual |left+right|={residual * 1000.0:.3f} mm | "
                f"contacts: left={'yes' if self._grasp_left_contact else 'no'}, "
                f"right={'yes' if self._grasp_right_contact else 'no'}, "
                f"non-finger Bottle={'YES' if self._grasp_nonfinger_contact else 'no'}"
            )
        except Exception as exc:
            self._gripper_state_label.text = f"Finger state unavailable: {exc}"

    def _ensure_grasp_contact_monitor(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        # ContactReportAPI is not inherited from an articulation root by each
        # rigid link.  Apply it explicitly to both finger rigid bodies and to
        # Bottle500 so contact classification does not depend on which actor
        # PhysX chooses as the reporting side.
        for path in (
            LEFT_ARTICULATION_PATH,
            LEFT_FINGER_LINK_PATH,
            RIGHT_FINGER_LINK_PATH,
            BOTTLE_PATH,
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"contact-report prim is missing: {path}")
            api = (
                PhysxSchema.PhysxContactReportAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                else PhysxSchema.PhysxContactReportAPI.Apply(prim)
            )
            api.CreateThresholdAttr().Set(0.0)
        self._grasp_contact_pairs.clear()
        self._recent_contact_paths.clear()
        self._grasp_left_contact = False
        self._grasp_right_contact = False
        self._grasp_nonfinger_contact = False
        self._grasp_bilateral_streak = 0
        self._grasp_mimic_bad_streak = 0

    @staticmethod
    def _decode_contact_path(encoded_path: int) -> str:
        if not encoded_path:
            return ""
        try:
            return str(PhysicsSchemaTools.intToSdfPath(encoded_path))
        except Exception:
            return ""

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        del contact_data
        for header in contact_headers:
            paths = tuple(
                sorted(
                    {
                        path
                        for path in (
                            self._decode_contact_path(header.actor0),
                            self._decode_contact_path(header.actor1),
                            self._decode_contact_path(header.collider0),
                            self._decode_contact_path(header.collider1),
                        )
                        if path
                    }
                )
            )
            if paths:
                self._recent_contact_paths.append(paths)
                del self._recent_contact_paths[:-20]
            if not paths or not any(path.startswith(BOTTLE_PATH) for path in paths):
                continue
            robot_contact = any(path.startswith(LEFT_ROBOT_PATH) for path in paths)
            if not robot_contact:
                continue
            if any(path.startswith(LEFT_FINGER_LINK_PATH) for path in paths):
                classification = "left"
            elif any(path.startswith(RIGHT_FINGER_LINK_PATH) for path in paths):
                classification = "right"
            else:
                classification = "nonfinger"
            if header.type == ContactEventType.CONTACT_LOST:
                self._grasp_contact_pairs.pop(paths, None)
            else:
                self._grasp_contact_pairs[paths] = classification
        active = set(self._grasp_contact_pairs.values())
        self._grasp_left_contact = "left" in active
        self._grasp_right_contact = "right" in active
        self._grasp_nonfinger_contact = "nonfinger" in active

    def _require_gripper_calibration_ready(self) -> None:
        self._require_guided_target_ready()
        if not (
            self._active_waypoint.endswith("/ REACHED")
            and ("+0 mm" in self._active_waypoint or self._active_waypoint.startswith("GRASP_POSE"))
        ):
            raise RuntimeError("reach APPROACH +0 mm and Pause before calibrating gripper contact")
        stage = omni.usd.get_context().get_stage()
        bottle_prim = stage.GetPrimAtPath(BOTTLE_PATH) if stage is not None else None
        if not bottle_prim or not bottle_prim.IsValid():
            raise RuntimeError(f"Bottle prim is missing: {BOTTLE_PATH}")
        rigid_api = UsdPhysics.RigidBodyAPI(bottle_prim)
        if not rigid_api or not bool(rigid_api.GetKinematicEnabledAttr().Get()):
            raise RuntimeError("Bottle500 must remain kinematic during contact calibration")

    def _on_gripper_calibration_step(self, delta_m: float) -> None:
        if self._gripper_task is not None and not self._gripper_task.done():
            self._set_status("A gripper calibration command is already running.", warn=True)
            return
        direction = "CLOSE" if delta_m < 0.0 else "OPEN"
        self._set_status(
            f"6{'A' if delta_m < 0.0 else 'B'} received: {direction} 1 mm; "
            "validating workflow gates before any gripper command."
        )
        self._gripper_abort_requested = False
        self._gripper_task = asyncio.ensure_future(self._run_gripper_calibration_step(delta_m))
        self._refresh_workflow_ui()

    def _on_auto_close_gripper(self) -> None:
        if self._gripper_task is not None and not self._gripper_task.done():
            self._set_status("A gripper calibration command is already running.", warn=True)
            return
        self._set_status(
            "6C received: AUTO CLOSE; validating workflow gates before any gripper command."
        )
        self._gripper_abort_requested = False
        self._gripper_task = asyncio.ensure_future(self._auto_close_until_bilateral_contact())
        self._refresh_workflow_ui()

    def _on_abort_gripper_motion(self) -> None:
        self._gripper_abort_requested = True
        self._timeline.pause()
        try:
            articulation, left_index, _, positions = self._get_gripper_state()
            hold = float(positions[left_index])
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=np.asarray([hold], dtype=np.float32),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            self._gripper_command_target_m = hold
        except Exception as exc:
            self._set_status(f"Gripper ABORT paused motion but could not author the hold target: {exc}", warn=True)
            return
        self._set_status(
            "Gripper ABORT: Timeline Paused and active left_finger target held at its current position. "
            "right_finger remains Mimic-driven."
        )

    async def _execute_gripper_target(self, target_m: float, stop_on_bilateral: bool) -> Dict[str, object]:
        app = omni.kit.app.get_app()
        articulation, left_index, right_index, before = self._get_gripper_state()
        arm_before = before[: len(ARM_JOINTS)].copy()
        target_m = float(np.clip(target_m, LEFT_GRIPPER_MIN_POSITION_M, LEFT_GRIPPER_OPEN_POSITION_M))
        self._gripper_command_target_m = target_m
        self._grasp_bilateral_streak = 0
        self._grasp_mimic_bad_streak = 0
        articulation.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=np.asarray([target_m], dtype=np.float32),
                joint_indices=np.asarray([left_index], dtype=np.int32),
            )
        )
        self._timeline.play()
        target_stable_updates = 0
        any_contact_stable_updates = 0
        settle_updates = 0
        settled_reason = ""
        try:
            for settle_updates in range(1, GRIPPER_MAX_SETTLE_UPDATES + 1):
                await app.next_update_async()
                if self._gripper_abort_requested:
                    raise RuntimeError("gripper motion aborted by operator")
                if self._grasp_nonfinger_contact:
                    raise RuntimeError("non-finger robot geometry contacted Bottle500")
                if self._grasp_mimic_bad_streak >= GRIPPER_MIMIC_BAD_STEPS:
                    raise RuntimeError(
                        f"Mimic residual exceeded {GRIPPER_MIMIC_RESIDUAL_GATE_M * 1000.0:.1f} mm"
                    )

                _, _, _, live_positions = self._get_gripper_state()
                target_error_m = abs(float(live_positions[left_index]) - target_m)
                if target_error_m <= GRIPPER_TARGET_TOLERANCE_M:
                    target_stable_updates += 1
                else:
                    target_stable_updates = 0

                if self._grasp_left_contact or self._grasp_right_contact:
                    any_contact_stable_updates += 1
                else:
                    any_contact_stable_updates = 0

                # A closing calibration must return control to its caller as soon
                # as either pad has stable contact.  The caller decides whether
                # the observed state is the requested unilateral/bilateral state.
                if stop_on_bilateral and (
                    self._grasp_bilateral_streak >= GRIPPER_BILATERAL_STABLE_STEPS
                    or any_contact_stable_updates >= GRIPPER_CONTACT_STABLE_UPDATES
                ):
                    settled_reason = "contact"
                    break
                if target_stable_updates >= GRIPPER_TARGET_STABLE_UPDATES:
                    settled_reason = "target_converged"
                    break
            if not settled_reason:
                _, _, _, timed_out_positions = self._get_gripper_state()
                timed_out_error_m = abs(float(timed_out_positions[left_index]) - target_m)
                raise RuntimeError(
                    "left_finger target convergence timeout after "
                    f"{GRIPPER_MAX_SETTLE_UPDATES} updates: target={target_m:.6f} m, "
                    f"actual={float(timed_out_positions[left_index]):.6f} m, "
                    f"error={timed_out_error_m * 1000.0:.3f} mm, "
                    f"recent_contact_paths={self._recent_contact_paths[-5:]}"
                )
        finally:
            self._timeline.pause()
        for _ in range(3):
            await app.next_update_async()
        _, _, _, after = self._get_gripper_state()
        arm_change = float(np.max(np.abs(after[: len(ARM_JOINTS)] - arm_before)))
        if arm_change > 0.005:
            raise RuntimeError(f"gripper calibration changed an arm joint by {arm_change:.6f} rad")
        residual = abs(float(after[left_index]) + float(after[right_index]))
        target_error_m = abs(float(after[left_index]) - target_m)
        return {
            "target_m": target_m,
            "left_actual_m": float(after[left_index]),
            "right_actual_m": float(after[right_index]),
            "target_error_m": target_error_m,
            "settle_updates": int(settle_updates),
            "settled_reason": settled_reason,
            "mimic_residual_m": residual,
            "left_contact": self._grasp_left_contact,
            "right_contact": self._grasp_right_contact,
            "bilateral_stable_steps": self._grasp_bilateral_streak,
            "nonfinger_contact": self._grasp_nonfinger_contact,
            "maximum_arm_joint_change_rad": arm_change,
        }

    async def _run_gripper_calibration_step(self, delta_m: float) -> None:
        try:
            self._require_gripper_calibration_ready()
            self._ensure_grasp_contact_monitor()
            _, left_index, _, positions = self._get_gripper_state()
            current = float(positions[left_index])
            if delta_m < 0.0 and self._grasp_left_contact and self._grasp_right_contact:
                raise RuntimeError("bilateral Bottle contact already exists; do not close farther")
            target = current + float(delta_m)
            readback = await self._execute_gripper_target(target, stop_on_bilateral=delta_m < 0.0)
            contact_text = (
                "stable bilateral Bottle contact reached"
                if readback["bilateral_stable_steps"] >= GRIPPER_BILATERAL_STABLE_STEPS
                else "no stable bilateral Bottle contact"
            )
            self._set_status(
                f"Gripper 1 mm step complete: target={readback['target_m']:.6f} m, "
                f"left/right={readback['left_actual_m']:.6f}/{readback['right_actual_m']:.6f} m, "
                f"Mimic residual={readback['mimic_residual_m'] * 1000.0:.3f} mm; {contact_text}. "
                "Timeline returned to Paused."
            )
        except Exception as exc:
            self._set_status(f"Gripper 1 mm step stopped safely: {exc}", warn=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._timeline.pause()
            self._gripper_task = None
            self._refresh_workflow_ui()

    async def _auto_close_until_bilateral_contact(self) -> None:
        try:
            self._require_gripper_calibration_ready()
            self._ensure_grasp_contact_monitor()
            _, left_index, _, positions = self._get_gripper_state()
            current = float(positions[left_index])
            if self._grasp_left_contact and self._grasp_right_contact:
                raise RuntimeError("bilateral Bottle contact already exists")
            step_count = 0
            while current > LEFT_GRIPPER_MIN_POSITION_M + 1e-9:
                target = max(LEFT_GRIPPER_MIN_POSITION_M, current - GRIPPER_CALIBRATION_STEP_M)
                readback = await self._execute_gripper_target(target, stop_on_bilateral=True)
                step_count += 1
                current = float(readback["left_actual_m"])
                if readback["bilateral_stable_steps"] >= GRIPPER_BILATERAL_STABLE_STEPS:
                    self._set_status(
                        f"AUTO CLOSE PASS: stable bilateral Bottle contact after {step_count} x 1 mm steps; "
                        f"target={readback['target_m']:.6f} m, "
                        f"left/right={readback['left_actual_m']:.6f}/{readback['right_actual_m']:.6f} m, "
                        f"Mimic residual={readback['mimic_residual_m'] * 1000.0:.3f} mm. "
                        "Timeline Paused; inspect front, side, and top views before any dynamic test."
                    )
                    return
                if target <= LEFT_GRIPPER_MIN_POSITION_M + 1e-9:
                    break
            raise RuntimeError(
                f"reached the {LEFT_GRIPPER_MIN_POSITION_M:.3f} m lower limit without stable bilateral contact"
            )
        except Exception as exc:
            self._set_status(f"AUTO CLOSE stopped safely: {exc}", warn=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._timeline.pause()
            self._gripper_task = None
            self._refresh_workflow_ui()

    def _on_open_left_gripper(self) -> None:
        if self._gripper_task is not None and not self._gripper_task.done():
            self._set_status("Left-gripper open command is already running.", warn=True)
            return
        self._gripper_task = asyncio.ensure_future(self._open_left_gripper_from_button())
        self._refresh_workflow_ui()

    async def _open_left_gripper_transaction(
        self, require_arm_stationary: bool = True
    ) -> Dict[str, object]:
        """Open only the active left_finger DOF and verify its Mimic follower."""

        app = omni.kit.app.get_app()
        self._timeline.pause()
        if self._timeline.is_stopped():
            self._timeline.play()
            await app.next_update_async()
            self._timeline.pause()
            await app.next_update_async()

        articulation, left_index, right_index, before = self._get_gripper_state()
        arm_before = before[: len(ARM_JOINTS)].copy()

        articulation.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=np.asarray([LEFT_GRIPPER_OPEN_POSITION_M], dtype=np.float32),
                joint_indices=np.asarray([left_index], dtype=np.int32),
            )
        )
        self._gripper_command_target_m = LEFT_GRIPPER_OPEN_POSITION_M
        self._timeline.play()
        for _ in range(100):
            await app.next_update_async()
        self._timeline.pause()
        for _ in range(5):
            await app.next_update_async()

        after = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        left_error = abs(float(after[left_index]) - LEFT_GRIPPER_OPEN_POSITION_M)
        arm_change = float(np.max(np.abs(after[: len(ARM_JOINTS)] - arm_before)))
        if left_error > 0.002:
            raise RuntimeError(
                f"left_finger did not reach open target: actual={after[left_index]:.6f} m, "
                f"target={LEFT_GRIPPER_OPEN_POSITION_M:.6f} m"
            )
        if require_arm_stationary and arm_change > 0.005:
            raise RuntimeError(f"opening gripper changed an arm joint by {arm_change:.6f} rad")

        self._articulation = articulation
        return {
            "left_finger_target_m": LEFT_GRIPPER_OPEN_POSITION_M,
            "left_finger_before_m": float(before[left_index]),
            "left_finger_after_m": float(after[left_index]),
            "right_finger_before_m": float(before[right_index]),
            "right_finger_after_m": float(after[right_index]),
            "maximum_arm_joint_change_rad": arm_change,
            "arm_stationary_gate_required": bool(require_arm_stationary),
            "timeline_paused": not self._timeline.is_playing(),
            "commanded_dofs": ["left_finger"],
            "mimic_only_dofs": ["right_finger"],
        }

    async def _open_left_gripper_from_button(self) -> None:
        try:
            readback = await self._open_left_gripper_transaction()
            self._set_status(
                "Left gripper OPEN PASS: left_finger reached "
                f"{readback['left_finger_after_m']:.6f} m; right_finger followed through Mimic; "
                "six arm joints were not commanded and Timeline returned to Paused."
            )
        except Exception as exc:
            self._set_status(f"Open Left Gripper failed: {exc}", warn=True)
            carb.log_error(traceback.format_exc())
        finally:
            self._timeline.pause()
            self._gripper_task = None
            self._refresh_workflow_ui()

    async def _run_auto_open_left_gripper(self) -> None:
        running_path = AUTO_OPEN_GRIPPER_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "stage_saved": False,
            "bottle_commanded": False,
            "arm_commanded": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(AUTO_OPEN_GRIPPER_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                result["request"] = json.load(stream)
            result.update(await self._open_left_gripper_transaction())
            result["status"] = "PASS"
        except Exception as exc:
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
        finally:
            self._timeline.pause()
            with open(AUTO_OPEN_GRIPPER_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")
            self._gripper_task = None

    async def _run_auto_dynamic_self_center(self) -> None:
        """Release Bottle500 and test whether a bounded close produces bilateral contact."""

        app = omni.kit.app.get_app()
        running_path = AUTO_DYNAMIC_SELF_CENTER_REQUEST_PATH + ".running"
        result: Dict[str, object] = {
            "status": "STARTED",
            "real_robot_touched": False,
            "ros_used": False,
            "stage_saved": False,
            "arm_commanded": False,
            "bottle_pose_directly_commanded": False,
        }
        try:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            os.replace(AUTO_DYNAMIC_SELF_CENTER_REQUEST_PATH, running_path)
            with open(running_path, "r", encoding="utf-8") as stream:
                request = json.load(stream)
            result["request"] = request
            if request.get("simulation_only") is not True:
                raise RuntimeError("dynamic self-centering request must set simulation_only=true")

            startup_wait_updates = 0
            startup_wait_limit = int(request.get("startup_wait_updates", 3000))
            while not (
                is_prim_path_valid(LEFT_ARTICULATION_PATH)
                and is_prim_path_valid(BOTTLE_PATH)
            ):
                if startup_wait_updates >= startup_wait_limit:
                    raise RuntimeError(
                        "timed out waiting for the loaded ALOHA articulation and Bottle prim"
                    )
                await app.next_update_async()
                startup_wait_updates += 1
            result["startup_wait_updates"] = startup_wait_updates

            close_delta_m = float(request.get("close_delta_m", 0.001))
            requested_close_target = request.get("close_target_m")
            max_updates = int(request.get("max_updates", 75))
            required_bilateral_steps = int(
                request.get("required_bilateral_steps", GRIPPER_BILATERAL_STABLE_STEPS)
            )
            if not 0.0 < close_delta_m <= 0.002:
                raise RuntimeError("close_delta_m must be in (0, 0.002] m")
            if requested_close_target is not None:
                requested_close_target = float(requested_close_target)
                if not LEFT_GRIPPER_MIN_POSITION_M <= requested_close_target <= LEFT_GRIPPER_OPEN_POSITION_M:
                    raise RuntimeError(
                        "close_target_m must be within the validated finger limits "
                        f"[{LEFT_GRIPPER_MIN_POSITION_M}, {LEFT_GRIPPER_OPEN_POSITION_M}] m"
                    )
            if not 5 <= max_updates <= 200:
                raise RuntimeError("max_updates must be in [5, 200]")
            if not 1 <= required_bilateral_steps <= 20:
                raise RuntimeError("required_bilateral_steps must be in [1, 20]")

            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("no USD stage is open")
            bottle_prim = stage.GetPrimAtPath(BOTTLE_PATH)
            if not bottle_prim or not bottle_prim.IsValid():
                raise RuntimeError(f"Bottle prim is missing: {BOTTLE_PATH}")
            rigid_api = UsdPhysics.RigidBodyAPI(bottle_prim)
            if not rigid_api:
                raise RuntimeError("Bottle500 has no RigidBodyAPI")

            self._ensure_grasp_contact_monitor()
            kinematic_before = bool(rigid_api.GetKinematicEnabledAttr().Get())
            if not kinematic_before:
                raise RuntimeError("Bottle500 must be kinematic at the start of the release test")
            self._timeline.play()
            for _ in range(8):
                await app.next_update_async()
            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            if not self._grasp_left_contact or self._grasp_right_contact:
                raise RuntimeError(
                    "release test requires the calibrated unilateral state "
                    f"left=yes/right=no; observed left={self._grasp_left_contact}, "
                    f"right={self._grasp_right_contact}"
                )

            articulation, left_index, right_index, positions = self._get_gripper_state()
            arm_before = positions[: len(ARM_JOINTS)].copy()
            left_before = float(positions[left_index])
            right_before = float(positions[right_index])
            target_m = (
                float(requested_close_target)
                if requested_close_target is not None
                else max(LEFT_GRIPPER_MIN_POSITION_M, left_before - close_delta_m)
            )
            if target_m >= left_before:
                raise RuntimeError(
                    f"dynamic close target {target_m:.6f} m is not below current left_finger "
                    f"position {left_before:.6f} m"
                )

            # Read Bottle motion from the initialized PhysX tensor view. Both
            # the default USD query and the optional usdrt/Fabric query can
            # remain at the authored startup xform in this streaming Stage.
            bottle_body = SingleRigidPrim(
                BOTTLE_PATH,
                name="aloha_dynamic_self_center_bottle",
                reset_xform_properties=False,
            )
            bottle_body.initialize()
            bottle_before_position, bottle_before_orientation = bottle_body.get_world_pose()
            left_finger_position, _ = get_world_pose(LEFT_FINGER_LINK_PATH, fabric=True)
            right_finger_position, _ = get_world_pose(RIGHT_FINGER_LINK_PATH, fabric=True)
            bottle_before_position = np.asarray(bottle_before_position, dtype=np.float64)
            bottle_before_orientation = np.asarray(bottle_before_orientation, dtype=np.float64)
            finger_axis = np.asarray(right_finger_position, dtype=np.float64) - np.asarray(
                left_finger_position, dtype=np.float64
            )
            finger_axis_norm = float(np.linalg.norm(finger_axis))
            if finger_axis_norm <= 1e-9:
                raise RuntimeError("left/right finger origins are coincident")
            toward_orange_axis = finger_axis / finger_axis_norm

            capture_viewport = bool(request.get("capture_viewport", True))
            capture_dir = ""
            viewport = None
            capture_index = 0
            if capture_viewport:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("no active viewport is available")
                safe_run_id = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in str(request.get("run_id", "dynamic_self_center"))
                )
                capture_dir = os.path.join(DEFAULT_LOG_DIR, "viewport_recordings", safe_run_id)
                os.makedirs(capture_dir, exist_ok=False)
                result["viewport_recording"] = {
                    "frame_directory": capture_dir,
                    "source": "Isaac Sim active server viewport",
                    "desktop_switched": False,
                }

                async def capture_frame(label: str) -> None:
                    nonlocal capture_index
                    output = os.path.join(capture_dir, f"frame_{capture_index:02d}_{label}.png")
                    helper = capture_viewport_to_file(viewport, file_path=output, is_hdr=False)
                    await helper.wait_for_result()
                    if not os.path.isfile(output) or os.path.getsize(output) <= 0:
                        raise RuntimeError(f"viewport frame was not written: {output}")
                    capture_index += 1

                await capture_frame("pre_dynamic")

            with Usd.EditContext(stage, stage.GetSessionLayer()):
                rigid_api.CreateKinematicEnabledAttr().Set(False)
            result["bottle_kinematic_before"] = kinematic_before
            result["bottle_kinematic_after_release"] = False

            self._gripper_command_target_m = target_m
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=np.asarray([target_m], dtype=np.float32),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            self._timeline.play()

            samples: List[Dict[str, object]] = []
            bilateral_reached = False
            for update_index in range(max_updates):
                await app.next_update_async()
                _, _, _, current_positions = self._get_gripper_state()
                bottle_position, _ = bottle_body.get_world_pose()
                samples.append(
                    {
                        "update": update_index + 1,
                        "timeline_time_s": float(self._timeline.get_current_time()),
                        "left_actual_m": float(current_positions[left_index]),
                        "right_actual_m": float(current_positions[right_index]),
                        "mimic_residual_m": abs(
                            float(current_positions[left_index])
                            + float(current_positions[right_index])
                        ),
                        "bottle_position_m": np.asarray(
                            bottle_position, dtype=np.float64
                        ).tolist(),
                        "left_contact": bool(self._grasp_left_contact),
                        "right_contact": bool(self._grasp_right_contact),
                        "bilateral_streak": int(self._grasp_bilateral_streak),
                        "nonfinger_contact": bool(self._grasp_nonfinger_contact),
                    }
                )
                if self._grasp_nonfinger_contact:
                    raise RuntimeError("non-finger robot geometry contacted Bottle500")
                if self._grasp_bilateral_streak >= required_bilateral_steps:
                    bilateral_reached = True
                    break

            self._timeline.pause()
            for _ in range(3):
                await app.next_update_async()
            _, _, _, final_positions = self._get_gripper_state()
            bottle_after_position, bottle_after_orientation = bottle_body.get_world_pose()
            bottle_after_position = np.asarray(bottle_after_position, dtype=np.float64)
            bottle_after_orientation = np.asarray(bottle_after_orientation, dtype=np.float64)
            displacement = bottle_after_position - bottle_before_position
            toward_orange_displacement = float(np.dot(displacement, toward_orange_axis))
            arm_change = float(
                np.max(np.abs(final_positions[: len(ARM_JOINTS)] - arm_before))
            )

            hold_m = float(final_positions[left_index])
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(
                    # Preserve the closing-force target after pausing. Writing
                    # the actual blocked position here would unload the grasp
                    # as soon as the Timeline resumes.
                    joint_positions=np.asarray([target_m], dtype=np.float32),
                    joint_indices=np.asarray([left_index], dtype=np.int32),
                )
            )
            self._gripper_command_target_m = target_m
            if capture_viewport:
                await capture_frame("post_dynamic")
                result["viewport_recording"]["frame_count"] = capture_index

            result.update(
                {
                    "status": "PASS"
                    if bilateral_reached and toward_orange_displacement > 0.0001
                    else "FAILED_GATE",
                    "left_finger_before_m": left_before,
                    "right_finger_before_m": right_before,
                    "command_target_m": target_m,
                    "left_finger_after_m": hold_m,
                    "right_finger_after_m": float(final_positions[right_index]),
                    "bottle_position_before_m": bottle_before_position.tolist(),
                    "bottle_orientation_before_wxyz": bottle_before_orientation.tolist(),
                    "bottle_position_after_m": bottle_after_position.tolist(),
                    "bottle_orientation_after_wxyz": bottle_after_orientation.tolist(),
                    "bottle_displacement_m": displacement.tolist(),
                    "bottle_displacement_norm_m": float(np.linalg.norm(displacement)),
                    "toward_orange_axis_world": toward_orange_axis.tolist(),
                    "toward_orange_displacement_m": toward_orange_displacement,
                    "orange_assumed_right_finger": True,
                    "bilateral_contact_reached": bilateral_reached,
                    "final_left_contact": bool(self._grasp_left_contact),
                    "final_right_contact": bool(self._grasp_right_contact),
                    "final_nonfinger_contact": bool(self._grasp_nonfinger_contact),
                    "maximum_arm_joint_change_rad": arm_change,
                    "timeline_paused": not self._timeline.is_playing(),
                    "samples": samples,
                }
            )
            self._set_status(
                "Dynamic self-centering test "
                f"{result['status']}: toward-orange displacement="
                f"{toward_orange_displacement * 1000.0:.3f} mm, "
                f"bilateral_contact={bilateral_reached}. Timeline Paused; Bottle500 remains dynamic."
            )
        except Exception as exc:
            result["status"] = "EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-30:]
            self._set_status(f"Dynamic self-centering test failed: {exc}", warn=True)
        finally:
            self._timeline.pause()
            with open(AUTO_DYNAMIC_SELF_CENTER_RESULT_PATH, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            if os.path.exists(running_path):
                os.replace(running_path, running_path + ".done")
            self._gripper_task = None

    def _apply_runtime_arm_profile(self, articulation: SingleArticulation) -> None:
        """Apply the validated simulation-only IK profile without saving USD."""

        arm_indices = np.arange(len(ARM_JOINTS), dtype=np.int64)
        articulation._articulation_view.set_gains(
            kps=np.full(len(ARM_JOINTS), ARM_RUNTIME_STIFFNESS, dtype=np.float64),
            kds=np.full(len(ARM_JOINTS), ARM_RUNTIME_DAMPING, dtype=np.float64),
            joint_indices=arm_indices,
            save_to_usd=False,
        )
        # Isaac Cortex uses disabled articulation gravity to model the gravity
        # compensation supplied by a real robot controller.  This affects the
        # arm only; bottle and environment gravity remain enabled.
        articulation.disable_gravity()
        kps, kds = articulation._articulation_view.get_gains()
        kps = np.asarray(kps, dtype=np.float64).reshape(-1)[: len(ARM_JOINTS)]
        kds = np.asarray(kds, dtype=np.float64).reshape(-1)[: len(ARM_JOINTS)]
        if not np.allclose(kps, ARM_RUNTIME_STIFFNESS, atol=1e-3):
            raise RuntimeError(f"runtime arm stiffness readback mismatch: {kps.tolist()}")
        if not np.allclose(kds, ARM_RUNTIME_DAMPING, atol=1e-3):
            raise RuntimeError(f"runtime arm damping readback mismatch: {kds.tolist()}")

    def _reset_left_sleep(self) -> None:
        if self._timeline.is_stopped():
            raise RuntimeError("timeline is stopped; press Play once, then Pause, and retry")
        if self._timeline.is_playing():
            raise RuntimeError("pause the timeline before resetting the left arm")
        if self._follow_enabled:
            raise RuntimeError("disable IK Follow before resetting the left arm")
        if self._joint_log_enabled:
            raise RuntimeError("stop and save the active joint log before resetting the left arm")
        if is_prim_path_valid(TARGET_PATH):
            raise RuntimeError("remove the extension Target before resetting the left arm")

        articulation = self._articulation or SingleArticulation(LEFT_ARTICULATION_PATH)
        if not articulation.handles_initialized:
            articulation.initialize()
        self._apply_runtime_arm_profile(articulation)
        dof_names = list(articulation.dof_names)
        if dof_names[: len(ARM_JOINTS)] != ARM_JOINTS:
            raise RuntimeError(f"unexpected arm DOF order: {dof_names}")

        current_positions = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        if current_positions.shape != (len(dof_names),):
            raise RuntimeError(
                f"unexpected articulation position shape {current_positions.shape}; expected {(len(dof_names),)}"
            )
        reset_positions = current_positions.copy()
        reset_positions[: len(ARM_JOINTS)] = LEFT_SLEEP_ARM_RAD
        zero_velocities = np.zeros_like(reset_positions)
        arm_indices = np.arange(len(ARM_JOINTS), dtype=np.int32)

        articulation.set_joints_default_state(positions=reset_positions, velocities=zero_velocities)
        articulation.set_joint_positions(reset_positions)
        articulation.set_joint_velocities(zero_velocities)
        articulation.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=LEFT_SLEEP_ARM_RAD.astype(np.float32),
                joint_velocities=np.zeros(len(ARM_JOINTS), dtype=np.float32),
                joint_indices=arm_indices,
            )
        )

        readback = np.asarray(articulation.get_joint_positions(), dtype=np.float64)[: len(ARM_JOINTS)]
        maximum_error = float(np.max(np.abs(readback - LEFT_SLEEP_ARM_RAD)))
        if maximum_error > SLEEP_READBACK_GATE_RAD:
            raise RuntimeError(
                f"sleep readback error {maximum_error:.6f} rad exceeds {SLEEP_READBACK_GATE_RAD:.3f} rad"
            )

        self._articulation = articulation
        self._aligned = False
        self._step_count = 0
        self._last_position_error = float("inf")
        self._last_orientation_error = float("inf")
        if self._position_error_label is not None:
            self._position_error_label.text = "Position error: not measured after sleep reset"
        if self._orientation_error_label is not None:
            self._orientation_error_label.text = "Orientation error: not measured after sleep reset"
        self._set_status(
            "Left arm reset to sleep while paused; six arm velocities were zeroed and gripper DOFs were preserved. "
            f"Maximum readback error: {maximum_error:.6f} rad. Next click Load Left Arm, Sync Base Pose, and Validate EE Alignment."
        )

    def _load_left_arm(self) -> None:
        self._disable_follow()
        if self._timeline.is_stopped():
            raise RuntimeError("timeline is stopped; press Play once, then Pause, and retry")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        for path in (LEFT_ARTICULATION_PATH, LEFT_BASE_PATH, LEFT_EE_PATH):
            if not stage.GetPrimAtPath(path):
                raise RuntimeError(f"required Prim is missing: {path}")

        description_path = self._description_model.get_value_as_string().strip()
        urdf_path = self._urdf_model.get_value_as_string().strip()
        if not os.path.isfile(description_path):
            raise FileNotFoundError(description_path)
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(urdf_path)

        articulation = SingleArticulation(LEFT_ARTICULATION_PATH)
        if not articulation.handles_initialized:
            articulation.initialize()
        self._apply_runtime_arm_profile(articulation)
        dof_names = list(articulation.dof_names)
        if dof_names[: len(ARM_JOINTS)] != ARM_JOINTS:
            raise RuntimeError(f"unexpected arm DOF order: {dof_names}")

        lula = LulaKinematicsSolver(robot_description_path=description_path, urdf_path=urdf_path)
        if LEFT_EE_FRAME not in lula.get_all_frame_names():
            raise RuntimeError(f"URDF does not contain end-effector frame {LEFT_EE_FRAME}")
        art_ik = ArticulationKinematicsSolver(articulation, lula, LEFT_EE_FRAME)
        subset = art_ik.get_joints_subset()
        if list(subset.joint_names) != ARM_JOINTS:
            raise RuntimeError(f"Lula c-space is not arm-only: {list(subset.joint_names)}")

        self._articulation = articulation
        self._lula = lula
        self._art_ik = art_ik
        self._aligned = False
        self._set_status(
            "Left arm loaded with runtime gravity compensation and validated IK gains "
            f"kp={ARM_RUNTIME_STIFFNESS:.1f}, kd={ARM_RUNTIME_DAMPING:.1f}. "
            "USD was not saved. IK remains disabled. Next click Sync Base Pose."
        )

    def _read_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        position, orientation = get_world_pose(LEFT_BASE_PATH)
        return np.asarray(position, dtype=np.float64), _quat_normalize(orientation)

    def _sync_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lula is None:
            raise RuntimeError("load the left arm first")
        position, orientation = self._read_base_pose()
        self._lula.set_robot_base_pose(position, orientation)
        return position, orientation

    def _on_sync_base(self) -> None:
        def operation() -> None:
            position, orientation = self._sync_base_pose()
            self._aligned = False
            self._set_status(
                "Base pose synchronized: "
                f"p=({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}), "
                f"q=({orientation[0]:.6f}, {orientation[1]:.6f}, {orientation[2]:.6f}, {orientation[3]:.6f}). "
                "Next validate EE alignment."
            )

        self._run_guarded("Sync Base Pose", operation)

    def _compute_alignment_errors(self) -> Tuple[float, float]:
        if self._art_ik is None:
            raise RuntimeError("load the left arm first")
        if self._articulation is None:
            raise RuntimeError("left-arm articulation is unavailable")
        base_position, base_orientation = self._sync_base_pose()
        actual_positions = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        actual_velocities = np.asarray(self._articulation.get_joint_velocities(), dtype=np.float64)
        applied_targets = None
        try:
            applied_action = self._articulation.get_applied_action()
            if applied_action is not None and applied_action.joint_positions is not None:
                applied_targets = np.asarray(applied_action.joint_positions, dtype=np.float64)
        except Exception:
            # Target readback is diagnostic-only and must never prevent the
            # alignment safety gate from running.
            applied_targets = None
        lula_position, lula_rotation = self._art_ik.compute_end_effector_pose()
        lula_orientation = _quat_normalize(rot_matrices_to_quats(lula_rotation))
        usd_position, usd_orientation = get_world_pose(LEFT_EE_PATH)
        usd_position = np.asarray(usd_position, dtype=np.float64)
        usd_orientation = _quat_normalize(usd_orientation)
        lula_position = np.asarray(lula_position, dtype=np.float64)
        position_delta = usd_position - lula_position
        position_error = float(np.linalg.norm(position_delta))
        orientation_error = _quat_angle(lula_orientation, usd_orientation)
        diagnostic = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "articulation_path": LEFT_ARTICULATION_PATH,
            "base_path": LEFT_BASE_PATH,
            "ee_path": LEFT_EE_PATH,
            "ee_frame": LEFT_EE_FRAME,
            "joint_names": list(self._articulation.dof_names),
            "actual_positions_rad": actual_positions.tolist(),
            "actual_velocities_rad_s": actual_velocities.tolist(),
            "applied_position_targets_rad": None if applied_targets is None else applied_targets.tolist(),
            "lula_base_world": {
                "position_m": base_position.tolist(),
                "orientation_wxyz": base_orientation.tolist(),
            },
            "lula_ee_world": {
                "position_m": lula_position.tolist(),
                "orientation_wxyz": lula_orientation.tolist(),
            },
            "usd_ee_world": {
                "position_m": usd_position.tolist(),
                "orientation_wxyz": usd_orientation.tolist(),
            },
            "usd_minus_lula_position_m": position_delta.tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        with open(ALIGNMENT_DIAGNOSTIC_PATH, "w", encoding="utf-8") as stream:
            json.dump(diagnostic, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        carb.log_info(
            "[ALOHA Lula Base Aligned] Alignment diagnostic written to "
            f"{ALIGNMENT_DIAGNOSTIC_PATH}; delta_m={position_delta.tolist()}"
        )
        self._last_position_error = position_error
        self._last_orientation_error = orientation_error
        return position_error, orientation_error

    def _update_error_labels(self, position_error: float, orientation_error: float) -> None:
        if self._position_error_label is not None:
            self._position_error_label.text = (
                f"Position error: {position_error * 1000.0:.4f} mm (gate <= 1.0 mm)"
            )
        if self._orientation_error_label is not None:
            self._orientation_error_label.text = (
                f"Orientation error: {math.degrees(orientation_error):.4f} deg (gate <= 0.5 deg)"
            )

    def _validate_alignment(self) -> bool:
        position_error, orientation_error = self._compute_alignment_errors()
        self._update_error_labels(position_error, orientation_error)
        self._aligned = position_error <= POSITION_GATE_M and orientation_error <= ORIENTATION_GATE_RAD
        return self._aligned

    def _on_validate_alignment(self) -> None:
        def operation() -> None:
            if self._validate_alignment():
                self._set_status("EE alignment PASS. IK remains disabled. Next create the Target at the current EE pose.")
            else:
                self._disable_follow()
                self._set_status("EE alignment FAIL. Do not create or follow a target; inspect the base and URDF mapping.", warn=True)

        self._run_guarded("Validate EE Alignment", operation)

    def _current_lula_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._art_ik is None:
            raise RuntimeError("load the left arm first")
        self._sync_base_pose()
        position, rotation = self._art_ik.compute_end_effector_pose()
        return np.asarray(position, dtype=np.float64), _quat_normalize(rot_matrices_to_quats(rotation))

    def _on_create_target(self) -> None:
        self._run_guarded("Create Target At Current EE", self._create_target_at_current_ee)

    def _create_target_at_current_ee(self) -> None:
        self._disable_follow()
        if not self._validate_alignment():
            self._update_error_labels(self._last_position_error, self._last_orientation_error)
            raise RuntimeError("EE alignment gates have not passed")
        if is_prim_path_valid(TARGET_PATH):
            delete_prim(TARGET_PATH)
        position, orientation = self._current_lula_ee_pose()
        self._target = VisualCuboid(
            prim_path=TARGET_PATH,
            name="aloha_aligned_ik_target",
            position=position,
            orientation=self._target_orientation_from_ee(orientation),
            size=0.04,
            color=np.array([1.0, 0.15, 0.05]),
        )
        self._active_waypoint = "current EE"
        self._set_status(
            f"Target created at current EE pose: {TARGET_PATH}. IK remains disabled. "
            "Select the target in Stage, but do not move it until IK Follow is enabled."
        )

    def _target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the visible Target pose converted to the native Lula EE frame."""

        if self._target is None or not is_prim_path_valid(TARGET_PATH):
            raise RuntimeError("extension target does not exist")
        position, orientation = self._target.get_world_pose()
        return (
            np.asarray(position, dtype=np.float64),
            self._ee_orientation_from_target(_quat_normalize(orientation)),
        )

    def _on_toggle_follow(self) -> None:
        self._run_guarded("Enable IK Follow", self._enable_follow)

    def _enable_follow(self) -> None:
        if self._follow_enabled:
            self._disable_follow()
            return
        if self._articulation is None or self._art_ik is None or self._lula is None:
            raise RuntimeError("load the left arm first")
        if not self._validate_alignment():
            raise RuntimeError("EE alignment gates do not pass")
        current_position, current_orientation = self._current_lula_ee_pose()
        target_position, target_orientation = self._target_pose()
        if float(np.linalg.norm(target_position - current_position)) > POSITION_GATE_M:
            raise RuntimeError("target is not at the current EE position; recreate it before enabling")
        if _quat_angle(current_orientation, target_orientation) > ORIENTATION_GATE_RAD:
            raise RuntimeError("target orientation is not aligned with the current EE; recreate it before enabling")
        self._follow_enabled = True
        self._follow_button.text = "IK Follow ARMED (click to disable)"
        self._set_status(
            "IK Follow ARMED. Keep Timeline Paused, then choose one guided waypoint. "
            "Only the six arm joints will be commanded after Play."
        )

    def _disable_follow(self) -> None:
        self._follow_enabled = False
        self._clear_hover_plan()
        self._aligned = False if self._art_ik is None else self._aligned
        if hasattr(self, "_follow_button") and self._follow_button is not None:
            self._follow_button.text = "4B. Arm IK Follow"
        self._refresh_workflow_ui()

    def _on_remove_target(self) -> None:
        self._disable_follow()
        if is_prim_path_valid(TARGET_PATH):
            delete_prim(TARGET_PATH)
        self._target = None
        self._clear_hover_plan()
        self._active_waypoint = "none"
        self._set_status("Extension target removed. IK Follow disabled.")

    def _on_start_joint_log(self) -> None:
        def operation() -> None:
            if self._articulation is None or self._art_ik is None:
                raise RuntimeError("load the left arm first")
            if self._timeline.is_playing():
                raise RuntimeError("pause the timeline before starting a new joint log")
            if self._target is None or not is_prim_path_valid(TARGET_PATH):
                raise RuntimeError("create the extension target first")
            self._joint_log_rows = []
            self._joint_log_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._last_joint_log_path = ""
            self._joint_log_elapsed_s = 0.0
            self._joint_log_enabled = True
            self._update_joint_log_label()
            self._set_status(
                "Joint logging ARMED. Enable IK Follow, move the Target while paused, then Play. "
                "Pause and click Stop and Save after the response settles."
            )

        self._run_guarded("Start New Joint Log", operation)

    def _on_stop_joint_log(self) -> None:
        def operation() -> None:
            self._joint_log_enabled = False
            if not self._joint_log_rows:
                self._update_joint_log_label()
                raise RuntimeError("no joint samples were recorded; run IK Follow while the timeline is playing")
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            run_id = self._joint_log_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DEFAULT_LOG_DIR, f"aloha_lula_joint_response_{run_id}.csv")
            with open(output_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self._joint_log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._joint_log_rows)
            self._last_joint_log_path = output_path
            sample_count = len(self._joint_log_rows)
            self._update_joint_log_label()
            self._set_status(f"Joint log saved: {sample_count} samples to {output_path}")

        self._run_guarded("Stop and Save Joint Log CSV", operation)

    def _update_joint_log_label(self) -> None:
        if self._joint_log_label is None:
            return
        state = "RECORDING" if self._joint_log_enabled else "idle"
        destination = self._last_joint_log_path or DEFAULT_LOG_DIR
        self._joint_log_label.text = (
            f"Joint log: {state}; samples: {len(self._joint_log_rows)}\nOutput: {destination}"
        )

    def _record_joint_sample(
        self,
        physics_dt: float,
        raw_target_position: np.ndarray,
        raw_target_orientation: np.ndarray,
        bounded_target_position: np.ndarray,
        bounded_target_orientation: np.ndarray,
        actual_positions: np.ndarray,
        actual_velocities: np.ndarray,
        ik_requested_positions: np.ndarray,
        position_targets: np.ndarray,
    ) -> None:
        if not self._joint_log_enabled:
            return
        self._joint_log_elapsed_s += float(physics_dt)
        row: Dict[str, float] = {
            "sample": float(len(self._joint_log_rows)),
            "elapsed_physics_time_s": self._joint_log_elapsed_s,
            "timeline_time_s": float(self._timeline.get_current_time()),
            "physics_dt_s": float(physics_dt),
            "raw_target_x_m": float(raw_target_position[0]),
            "raw_target_y_m": float(raw_target_position[1]),
            "raw_target_z_m": float(raw_target_position[2]),
            "raw_target_qw": float(raw_target_orientation[0]),
            "raw_target_qx": float(raw_target_orientation[1]),
            "raw_target_qy": float(raw_target_orientation[2]),
            "raw_target_qz": float(raw_target_orientation[3]),
            "bounded_target_x_m": float(bounded_target_position[0]),
            "bounded_target_y_m": float(bounded_target_position[1]),
            "bounded_target_z_m": float(bounded_target_position[2]),
            "bounded_target_qw": float(bounded_target_orientation[0]),
            "bounded_target_qx": float(bounded_target_orientation[1]),
            "bounded_target_qy": float(bounded_target_orientation[2]),
            "bounded_target_qz": float(bounded_target_orientation[3]),
        }
        for index, joint_name in enumerate(ARM_JOINTS):
            row[f"{joint_name}_actual_position_rad"] = float(actual_positions[index])
            row[f"{joint_name}_velocity_rad_s"] = float(actual_velocities[index])
            row[f"{joint_name}_ik_requested_position_rad"] = float(ik_requested_positions[index])
            row[f"{joint_name}_position_target_rad"] = float(position_targets[index])
        self._joint_log_rows.append(row)
        if len(self._joint_log_rows) % 30 == 0:
            self._update_joint_log_label()

    def _bounded_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        current_position, current_orientation = self._current_lula_ee_pose()
        target_position, target_orientation = self._target_pose()
        delta = target_position - current_position
        distance = float(np.linalg.norm(delta))
        if distance > MAX_TARGET_TRANSLATION_STEP_M:
            target_position = current_position + delta * (MAX_TARGET_TRANSLATION_STEP_M / distance)
        target_orientation = _quat_step(current_orientation, target_orientation, MAX_TARGET_ROTATION_STEP_RAD)
        return target_position, target_orientation

    def _apply_bounded_ik_step(self, physics_dt: float) -> None:
        if self._articulation is None or self._art_ik is None:
            raise RuntimeError("left arm is not loaded")
        raw_target_position, raw_target_orientation = self._target_pose()
        # HOVER is an already prevalidated, time-parameterized joint route.
        # Do not compare asynchronous USD/Fabric link transforms against Lula
        # FK on every moving physics frame; validate the static endpoints
        # instead. This also avoids writing one alignment JSON per substep.
        if self._hover_plan_positions:
            self._apply_hover_plan_step(
                physics_dt, raw_target_position, raw_target_orientation
            )
            return
        # The strict 1 mm / 0.5 deg gate is required before arming. During
        # motion, Lula FK and USD/Fabric link transforms can differ by one
        # physics frame, so use a bounded runtime gate equal to the maximum
        # permitted Cartesian target step. Large base/URDF errors still stop.
        position_error, orientation_error = self._compute_alignment_errors()
        self._update_error_labels(position_error, orientation_error)
        if (
            position_error > RUNTIME_POSITION_GATE_M
            or orientation_error > RUNTIME_ORIENTATION_GATE_RAD
        ):
            raise RuntimeError(
                f"runtime EE alignment gate failed: {self._last_position_error * 1000.0:.3f} mm, "
                f"{math.degrees(self._last_orientation_error):.3f} deg"
            )
        target_position, target_orientation = self._bounded_target_pose()
        action, success = self._art_ik.compute_inverse_kinematics(
            target_position,
            target_orientation,
            position_tolerance=0.0005,
            orientation_tolerance=math.radians(0.5),
        )
        if not success:
            raise RuntimeError("Lula IK did not converge; no action was applied")
        subset = self._art_ik.get_joints_subset()
        if list(subset.joint_names) != ARM_JOINTS:
            raise RuntimeError("runtime joint subset changed; refusing to command")
        current_positions = np.asarray(subset.get_joint_positions(), dtype=np.float64)
        current_velocities = np.asarray(subset.get_joint_velocities(), dtype=np.float64)
        requested_positions = np.asarray(action.joint_positions, dtype=np.float64)
        if current_positions.shape != (6,) or current_velocities.shape != (6,) or requested_positions.shape != (6,):
            raise RuntimeError(
                "expected six arm joint values, got "
                f"position={current_positions.shape}, velocity={current_velocities.shape}, "
                f"requested={requested_positions.shape}"
            )
        bounded_positions = current_positions + np.clip(
            requested_positions - current_positions,
            -MAX_JOINT_STEP_RAD,
            MAX_JOINT_STEP_RAD,
        )
        bounded_action = subset.make_articulation_action(bounded_positions, None)
        self._articulation.get_articulation_controller().apply_action(bounded_action)
        self._record_joint_sample(
            physics_dt,
            raw_target_position,
            raw_target_orientation,
            target_position,
            target_orientation,
            current_positions,
            current_velocities,
            requested_positions,
            bounded_positions,
        )

    def _apply_hover_plan_step(
        self,
        physics_dt: float,
        raw_target_position: np.ndarray,
        raw_target_orientation: np.ndarray,
    ) -> None:
        """Execute a prevalidated HOVER joint route without another online IK search."""

        if self._articulation is None or self._art_ik is None:
            raise RuntimeError("left arm is not loaded")
        subset = self._art_ik.get_joints_subset()
        current = np.asarray(subset.get_joint_positions(), dtype=np.float64)
        velocity = np.asarray(subset.get_joint_velocities(), dtype=np.float64)
        if current.shape != (len(ARM_JOINTS),) or velocity.shape != current.shape:
            raise RuntimeError("unexpected arm state shape during HOVER route")

        index = min(self._hover_plan_index, len(self._hover_plan_positions) - 1)
        self._hover_plan_elapsed_s += float(physics_dt)
        index = min(
            int(self._hover_plan_elapsed_s / HOVER_PLAN_CONTROL_PERIOD_S),
            len(self._hover_plan_positions) - 1,
        )
        self._hover_plan_index = index
        requested = np.asarray(self._hover_plan_positions[index], dtype=np.float64)
        error = requested - current
        # The route itself is sampled at the configured bounded joint step every 20 ms. Applying a
        # second limit relative to the lagging actual position creates a
        # second actual-position limit can make forearm_roll crawl
        # indefinitely. Send the already bounded 50 Hz reference directly.
        bounded = requested.copy()
        self._articulation.get_articulation_controller().apply_action(
            subset.make_articulation_action(bounded, None)
        )
        bounded_position, bounded_rotation = self._lula.compute_forward_kinematics(
            LEFT_EE_FRAME, bounded
        )
        bounded_orientation = _quat_normalize(rot_matrices_to_quats(bounded_rotation))
        self._record_joint_sample(
            physics_dt,
            raw_target_position,
            raw_target_orientation,
            np.asarray(bounded_position, dtype=np.float64),
            bounded_orientation,
            current,
            velocity,
            requested,
            bounded,
        )
        if (
            index == len(self._hover_plan_positions) - 1
            and float(np.max(np.abs(error))) <= HOVER_PLAN_REACHED_RAD
            and float(np.max(np.abs(velocity))) <= HOVER_PLAN_SETTLED_VELOCITY_RAD_S
        ):
            route_name = self._planned_route_name or "GUIDED ROUTE"
            self._active_waypoint = f"{route_name} / REACHED"
            if not self._hover_reached_reported:
                self._hover_reached_reported = True
                self._set_status(
                    f"{route_name} reached on the prevalidated 50 Hz joint route. Pause the Timeline and inspect the "
                    "open gripper pose before selecting the next waypoint. IK Follow remains armed to hold the final target."
                )

    def _update_gripper_calibration_runtime_state(self) -> None:
        if self._grasp_left_contact and self._grasp_right_contact:
            self._grasp_bilateral_streak += 1
        else:
            self._grasp_bilateral_streak = 0
        if self._articulation is not None and self._articulation.handles_initialized:
            try:
                dof_names = list(self._articulation.dof_names)
                positions = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
                left = float(positions[dof_names.index("left_finger")])
                right = float(positions[dof_names.index("right_finger")])
                if abs(left + right) > GRIPPER_MIMIC_RESIDUAL_GATE_M:
                    self._grasp_mimic_bad_streak += 1
                else:
                    self._grasp_mimic_bad_streak = 0
            except Exception:
                self._grasp_mimic_bad_streak = 0
        self._contact_ui_counter += 1
        if self._contact_ui_counter % 5 == 0:
            self._refresh_gripper_state_label()

    def _on_physics_step(self, step: float) -> None:
        self._update_gripper_calibration_runtime_state()
        if not self._follow_enabled:
            return
        try:
            self._apply_bounded_ik_step(step)
            self._step_count += 1
            if self._step_count % 30 == 0:
                self._update_error_labels(self._last_position_error, self._last_orientation_error)
        except Exception as exc:
            self._disable_follow()
            self._set_status(f"IK Follow stopped safely: {exc}", warn=True)
            carb.log_error(traceback.format_exc())

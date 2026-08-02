from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping import grasp_20cm_isaac_bindings as isaac_bindings
from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import IsaacGrasp20cmBindings
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import arm_phase_target_reached
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import arm_phase_timeout_reached
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import bilateral_observation_contact
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import bottle_tensor_lifecycle_plan
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import build_lula_cspace_phase_targets
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import derive_gripper_closeup_camera_geometry
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import derive_overview_camera_geometry
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import derive_subject_bounding_closeup_camera_geometry
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import formal_phase_bottle_dynamic
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import initial_pose_hold_complete
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import open_pregrasp_evidence_ready
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import physics_sample_duration_s
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import preload_solver_contact_ready
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import required_collider_phase_labels
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import reset_body_transition_plan
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import single_body_tensor_indices
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import solver_active_contacts
from tools.aloha1_mapping.grasp_20cm_runtime import EXPECTED_DOF_ORDER
from tools.aloha1_mapping.grasp_20cm_runtime import FrozenInputError
from tools.aloha1_mapping.grasp_20cm_runtime import Grasp20cmRuntimeAdapter
from tools.aloha1_mapping.grasp_20cm_runtime import apply_verified_session_sublayers
from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage
from tools.aloha1_mapping.grasp_20cm_runtime import verify_frozen_file
from tools.run_aloha1_grasp_20cm_gui import _load_frozen_bottle_transform
from tools.run_aloha1_grasp_20cm_gui import evaluate_abort_reset_evidence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_gui.yaml"
GUI_SCRIPT = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"


class _FakePrim:
    def __init__(self, *, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:  # noqa: N802 - matches the USD API.
        return self._valid


class _FakeSessionLayer:
    def __init__(self) -> None:
        self.subLayerPaths: list[str] = []


class _FakeStageWithSession:
    def __init__(self) -> None:
        self.session = _FakeSessionLayer()

    def GetSessionLayer(self) -> _FakeSessionLayer:  # noqa: N802
        return self.session


def test_verified_session_sublayer_is_inserted_once() -> None:
    stage = _FakeStageWithSession()
    records = [
        {
            "absolute_path": "/tmp/finger_source_limits.usda",
            "sha256": "a" * 64,
        }
    ]

    first = apply_verified_session_sublayers(stage=stage, records=records)
    second = apply_verified_session_sublayers(stage=stage, records=records)

    assert stage.session.subLayerPaths == [
        "/tmp/finger_source_limits.usda"
    ]
    assert first["inserted_paths"] == ["/tmp/finger_source_limits.usda"]
    assert second["inserted_paths"] == []
    assert second["already_present_paths"] == [
        "/tmp/finger_source_limits.usda"
    ]


class _FakeContinuousTrajectory:
    start_time = 0.0
    end_time = 0.5

    def get_joint_targets(
        self,
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        fraction = float(time_s) / self.end_time
        position = np.full(6, fraction, dtype=np.float64)
        velocity = np.full(6, 2.0, dtype=np.float64)
        if time_s in {self.start_time, self.end_time}:
            velocity[:] = 0.0
        return position, velocity


class _FakeLulaTrajectoryGenerator:
    def __init__(
        self,
        robot_description_path: str,
        urdf_path: str,
    ) -> None:
        self.paths = (robot_description_path, urdf_path)
        self.velocity_limits: np.ndarray | None = None
        self.acceleration_limits: np.ndarray | None = None
        self.waypoints: np.ndarray | None = None

    def get_active_joints(self) -> list[str]:
        return [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]

    def set_c_space_velocity_limits(self, limits: np.ndarray) -> None:
        self.velocity_limits = limits

    def set_c_space_acceleration_limits(
        self,
        limits: np.ndarray,
    ) -> None:
        self.acceleration_limits = limits

    def compute_c_space_trajectory(
        self,
        waypoints: np.ndarray,
    ) -> _FakeContinuousTrajectory:
        self.waypoints = waypoints
        return _FakeContinuousTrajectory()


def test_lula_phase_targets_apply_limits_and_include_stopped_endpoint() -> None:
    created: list[_FakeLulaTrajectoryGenerator] = []

    def factory(
        robot_description_path: str,
        urdf_path: str,
    ) -> _FakeLulaTrajectoryGenerator:
        generator = _FakeLulaTrajectoryGenerator(
            robot_description_path,
            urdf_path,
        )
        created.append(generator)
        return generator

    result = build_lula_cspace_phase_targets(
        generator_factory=factory,
        robot_description_path="/tmp/robot.yaml",
        urdf_path="/tmp/robot.urdf",
        waypoint_positions=[
            [0.0, -0.9, 1.1, 0.0, -0.3, 0.0],
            [0.1, -0.8, 1.0, 0.1, -0.2, 0.1],
        ],
        physics_dt_s=0.1,
        velocity_limits_rad_s=[3.0] * 6,
        acceleration_limits_rad_s2=[5.0] * 6,
    )

    assert len(created) == 1
    assert created[0].paths == ("/tmp/robot.yaml", "/tmp/robot.urdf")
    assert created[0].velocity_limits == pytest.approx([3.0] * 6)
    assert created[0].acceleration_limits == pytest.approx([5.0] * 6)
    np.testing.assert_allclose(
        created[0].waypoints,
        np.asarray(
            [
            [0.0, -0.9, 1.1, 0.0, -0.3, 0.0],
            [0.1, -0.8, 1.0, 0.1, -0.2, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    assert len(result["targets"]) == 6
    assert result["targets"][0]["velocity_rad_s"] == pytest.approx(
        [0.0] * 6
    )
    assert result["targets"][-1]["position_rad"] == pytest.approx(
        [1.0] * 6
    )
    assert result["targets"][-1]["velocity_rad_s"] == pytest.approx(
        [0.0] * 6
    )
    assert result["audit"]["duration_s"] == pytest.approx(0.5)
    assert result["audit"]["endpoint_velocity_zero"] is True
    assert result["audit"]["finite"] is True


def test_runtime_defaults_preserve_translation_only_baseline() -> None:
    signature = inspect.signature(IsaacGrasp20cmBindings)

    assert signature.parameters["bottle_world_from_object"].default is None
    assert signature.parameters["initial_arm_q_rad"].default is None
    assert signature.parameters["initial_pose_hold_frames"].default == 60


def test_gui_accepts_frozen_bottle_pose_and_initial_arm_q() -> None:
    source = GUI_SCRIPT.read_text(encoding="utf-8")

    assert '"--bottle-world-from-object-json"' in source
    assert '"--initial-arm-q-rad"' in source
    assert '"--initial-pose-hold-frames"' in source
    assert "nargs=6" in source


def test_initial_pose_hold_completes_on_exact_required_frame() -> None:
    assert not initial_pose_hold_complete(
        observed_frame_count=59,
        required_frame_count=60,
    )
    assert initial_pose_hold_complete(
        observed_frame_count=60,
        required_frame_count=60,
    )


def test_dynamic_gate_excludes_kinematic_setup_but_not_formal_phases() -> None:
    observations = [
        type("Observation", (), {"bottle_dynamic": False})(),
        type("Observation", (), {"bottle_dynamic": True})(),
        type("Observation", (), {"bottle_dynamic": True})(),
    ]
    telemetry = [
        {"phase": Phase.SETUP_KINEMATIC.value},
        {"phase": Phase.RELEASE_DYNAMIC.value},
        {"phase": Phase.HOLD.value},
    ]

    assert formal_phase_bottle_dynamic(observations, telemetry)

    observations[-1].bottle_dynamic = False
    assert not formal_phase_bottle_dynamic(observations, telemetry)


def test_arm_phase_requires_trajectory_exhaustion_and_readback() -> None:
    target = np.array([0.2, -0.4, 0.8, 0.1, -0.2, 0.3])

    assert not arm_phase_target_reached(
        trajectory_exhausted=False,
        joint_readback=target,
        joint_target=target,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
        tolerance_rad=0.020,
    )
    assert not arm_phase_target_reached(
        trajectory_exhausted=True,
        joint_readback=target
        + np.array([0.46, 0.0, 0.0, 0.0, -0.298, 0.0]),
        joint_target=target,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
        tolerance_rad=0.020,
    )
    assert arm_phase_target_reached(
        trajectory_exhausted=True,
        joint_readback=target + 0.019,
        joint_target=target,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
        tolerance_rad=0.020,
    )


def test_arm_phase_timeout_starts_after_trajectory_exhaustion() -> None:
    assert not arm_phase_timeout_reached(
        phase_frame_count=900,
        trajectory_sample_count=1922,
        readback_settle_timeout_frames=900,
        trajectory_exhausted=False,
    )
    assert not arm_phase_timeout_reached(
        phase_frame_count=2822,
        trajectory_sample_count=1922,
        readback_settle_timeout_frames=900,
        trajectory_exhausted=True,
    )
    assert arm_phase_timeout_reached(
        phase_frame_count=2823,
        trajectory_sample_count=1922,
        readback_settle_timeout_frames=900,
        trajectory_exhausted=True,
    )


def test_overview_camera_includes_random_start_in_derived_target() -> None:
    result = derive_overview_camera_geometry(
        base_position_world_m=[-0.45, 0.0, 0.0],
        initial_ee_position_world_m=[-0.12, 0.15, 0.775],
        grasp_position_world_m=[-0.18, 0.0, 0.056],
        lift_position_world_m=[-0.18, 0.0, 0.266],
    )

    assert result["distance_m"] >= 3.6
    assert result["anchor_min_world_m"][2] == pytest.approx(0.0)
    assert result["anchor_max_world_m"][2] == pytest.approx(0.775)
    assert result["target_world_m"][2] > 0.25
    assert result["derivation"] == (
        "RUNTIME_BASE_INITIAL_EE_GRASP_LIFT_ANCHOR_BOUNDS"
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_frozen_bottle_transform_accepts_rigid_4x4_json(
    tmp_path: Path,
    *,
    wrapped: bool,
) -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.03]
    path = tmp_path / "pose.json"
    payload: object = (
        {"world_from_object": transform.tolist()}
        if wrapped
        else transform.tolist()
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_frozen_bottle_transform(path) == pytest.approx(transform)


class _FakeLayer:
    def __init__(self, sublayers: list[str]) -> None:
        self.subLayerPaths = sublayers


class _FakeStage:
    def __init__(
        self,
        *,
        valid_prims: set[str],
        sublayers: list[str] | None = None,
    ) -> None:
        self._valid_prims = valid_prims
        self._layer = _FakeLayer(sublayers or [])

    def GetPrimAtPath(self, path: str) -> _FakePrim:  # noqa: N802
        return _FakePrim(valid=path in self._valid_prims)

    def GetRootLayer(self) -> _FakeLayer:  # noqa: N802
        return self._layer


class _FakeBindings:
    def __init__(self) -> None:
        self.prepared = 0
        self.observed = 0
        self.applied_phases: list[Phase] = []
        self.kinematic_updates: list[bool] = []
        self.finalized: list[tuple[Phase, str]] = []
        self.reset_count = 0
        self.observation = RunObservation(
            frame=1,
            time_s=1.0 / 60.0,
            clearance_m=0.0,
            bottle_dynamic=True,
            support_contact=True,
            bottle_linear_speed_m_s=0.0,
            bottle_angular_speed_rad_s=0.0,
            stage_contract_valid=True,
            setup_complete=True,
            open_target_reached=True,
            descent_complete=True,
            bilateral_contact=True,
            preload_complete=True,
            lift_waypoint_exhausted=False,
            hold_drop_m=0.0,
            finite_state=True,
            persistent_penetration=False,
            numerical_ejection=False,
            forbidden_constraint=False,
            phase_timed_out=False,
            ee_vertical_displacement_m=0.0,
        )

    def prepare_run(self) -> None:
        self.prepared += 1

    def read_observation(
        self,
        *,
        frame: int,
        time_s: float,
    ) -> RunObservation:
        self.observed += 1
        return RunObservation(
            **{
                **self.observation.__dict__,
                "frame": frame,
                "time_s": time_s,
            }
        )

    def apply_phase_target(self, phase: Phase) -> None:
        self.applied_phases.append(phase)

    def set_bottle_kinematic(self, *, enabled: bool) -> None:
        self.kinematic_updates.append(enabled)

    def finalize_run(self, phase: Phase, reason: str) -> None:
        self.finalized.append((phase, reason))

    def reset_session(self) -> None:
        self.reset_count += 1


def test_config_freezes_local_runtime_and_height_semantics() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    assert config["classification"] == (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    )
    assert config["target"]["measurement"] == (
        "BOTTLE_COLLISION_MIN_WORLD_Z_MINUS_TABLE_TOP_WORLD_Z"
    )
    assert config["target"]["clearance_m"] == pytest.approx(0.200)
    assert config["target"]["hold_duration_s"] == pytest.approx(2.0)
    assert config["target"]["hold_drop_gate_m"] == pytest.approx(0.010)
    assert config["target"][
        "support_contact_latch_clearance_m"
    ] == pytest.approx(0.0005)
    assert config["physics"] == {
        "frequency_hz": 60,
        "mass_kg": 0.020,
        "friction": 0.7,
        "restitution": 0.0,
        "solve_articulation_contact_last": True,
        "finger_drive_type": "force",
        "preload_delta_m": 0.0,
    }
    assert config["boundaries"]["task8"] == "NOT_RUN"
    assert config["boundaries"]["real_robot"] is False
    assert config["boundaries"]["remote_103"] is False


def test_config_freezes_approved_stage_bottle_and_joint_order() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["stage"] == {
        "path": (
            "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "table_support_alignment/1.0/"
            "aloha1_table_support_aligned_workcell.usda"
        ),
        "sha256": (
            "2b3f76365ed67532f478d995ae859a88"
            "b5639975ac07cb7ac8a53ac679e8205c"
        ),
        "root_prim": "/World",
        "articulation_prim": (
            "/World/follower_left/vx300s_left/root_joint"
        ),
        "table_prim": (
            "/World/environment/worldBody/user_confirmed_table"
        ),
    }
    assert config["bottle"] == {
        "path": "assets/bottle_500ml/isaac/bottle_500ml_sim.usd",
        "sha256": (
            "16427135f152ec951de2321fd689366d"
            "745a2dd389cbe260976631783952533e"
        ),
        "reference_prim": "/Bottle500",
        "session_prim": "/World/ALOHA1Grasp20cmSession/Bottle500",
    }
    assert config["robot"]["dof_order"] == EXPECTED_DOF_ORDER
    assert config["evidence"]["collider_overlay"] == {
        "display_setting": (
            "/persistent/physics/visualizationDisplayColliders"
        ),
        "display_value": 2,
        "authored_geometry_clone": True,
        "semantics": (
            "ISAAC_PHYSICS_DEBUG_DISPLAY_PLUS_SESSION_AUTHORED_"
            "COLLIDER_CLONE_NOT_COOKED_HULL_READBACK"
        ),
        "finger_colliders": {
            "left": {
                "link": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_left_finger_link"
                ),
                "visual": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_left_finger_link/visuals/"
                    "diagnostic_supplier_cad_left_finger/mesh"
                ),
                "collider": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_left_finger_link/collisions/"
                    "diagnostic_supplier_cad_left_finger/mesh"
                ),
            },
            "right": {
                "link": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_right_finger_link"
                ),
                "visual": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_right_finger_link/visuals/"
                    "diagnostic_supplier_cad_right_finger/mesh"
                ),
                "collider": (
                    "/World/follower_left/vx300s_left/"
                    "follower_left_right_finger_link/collisions/"
                    "diagnostic_supplier_cad_right_finger/mesh"
                ),
            },
        },
    }


def test_all_frozen_files_match_config_hashes() -> None:
    profile = load_and_verify_config(CONFIG, project_root=ROOT)
    assert profile["config_path"] == str(CONFIG.resolve())
    assert profile["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert set(profile["frozen_inputs"]) == {
        "stage",
        "bottle",
        "task7b2_runtime_profile",
        "lula_descriptor",
        "grasp_editor_semantics",
        "grasp_editor_variant_b_raw",
        "kinematics_report",
        "ik_correspondence_report",
        "coupling_report",
        "follower_left_urdf",
        "joint_map",
    }


def test_frozen_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "stage.usda"
    candidate.write_text("changed", encoding="utf-8")
    with pytest.raises(FrozenInputError, match="sha256 mismatch"):
        verify_frozen_file(candidate, "2b3f" + "0" * 60)


def test_validate_composed_stage_records_sublayers_and_required_prims() -> None:
    stage = _FakeStage(
        valid_prims={"/World", "/World/robot", "/World/table"},
        sublayers=["configuration.usda", "physics.usda"],
    )
    result = validate_composed_stage(
        stage=stage,
        expected_root_prim="/World",
        required_prims=["/World/robot", "/World/table"],
    )
    assert result == {
        "root_prim": "/World",
        "sublayers": ["configuration.usda", "physics.usda"],
        "required_prims": ["/World/robot", "/World/table"],
    }


def test_validate_composed_stage_fails_on_missing_prim() -> None:
    stage = _FakeStage(valid_prims={"/World", "/World/robot"})
    with pytest.raises(FrozenInputError, match="/World/table"):
        validate_composed_stage(
            stage=stage,
            expected_root_prim="/World",
            required_prims=["/World/robot", "/World/table"],
        )


def test_runtime_adapter_advances_exactly_one_physics_step() -> None:
    bindings = _FakeBindings()
    adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
    adapter.start()

    transition = adapter.on_physics_step(1.0 / 60.0)

    assert transition is not None
    assert transition.current is Phase.SETUP_KINEMATIC
    assert adapter.physics_step_count == 1
    assert bindings.observed == 1
    assert bindings.applied_phases == [Phase.SETUP_KINEMATIC]


def test_release_transition_makes_bottle_dynamic_once() -> None:
    bindings = _FakeBindings()
    adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
    adapter.start()
    adapter.on_physics_step(1.0 / 60.0)
    adapter.on_physics_step(1.0 / 60.0)

    assert adapter.phase is Phase.RELEASE_DYNAMIC
    assert bindings.kinematic_updates == [False]


def test_abort_stops_new_targets_without_freezing_bottle() -> None:
    bindings = _FakeBindings()
    adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
    adapter.start()
    adapter.on_physics_step(1.0 / 60.0)
    applied_before_abort = len(bindings.applied_phases)

    transition = adapter.abort()
    assert transition.current is Phase.ABORTED
    assert adapter.on_physics_step(1.0 / 60.0) is None
    assert len(bindings.applied_phases) == applied_before_abort
    assert bindings.kinematic_updates == []
    assert bindings.finalized == [(Phase.ABORTED, "user_abort")]


def test_runtime_exception_marks_adapter_terminal_without_finalizing() -> None:
    bindings = _FakeBindings()
    adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
    adapter.start()
    adapter.on_physics_step(1.0 / 60.0)
    applied_before_failure = len(bindings.applied_phases)

    transition = adapter.fail_due_to_exception("runtime_exception")

    assert transition.current is Phase.FAIL
    assert adapter.is_running is False
    assert adapter.on_physics_step(1.0 / 60.0) is None
    assert len(bindings.applied_phases) == applied_before_failure
    assert bindings.finalized == []


def test_preload_uses_active_solver_contacts_not_penetration_sign() -> None:
    assert preload_solver_contact_ready(
        close_exhausted=True,
        left_solver_active=True,
        right_solver_active=True,
        coupling_residual_m=0.0002,
        coupling_gate_m=0.001,
    )
    assert not preload_solver_contact_ready(
        close_exhausted=True,
        left_solver_active=True,
        right_solver_active=False,
        coupling_residual_m=0.0002,
        coupling_gate_m=0.001,
    )


def test_reset_calls_session_cleanup_and_returns_idle() -> None:
    bindings = _FakeBindings()
    adapter = Grasp20cmRuntimeAdapter(bindings=bindings)
    adapter.start()
    adapter.abort()

    transition = adapter.reset()

    assert transition.current is Phase.IDLE
    assert bindings.reset_count == 1
    assert adapter.physics_step_count == 0


def test_abort_reset_evidence_requires_no_abort_target_write() -> None:
    report = evaluate_abort_reset_evidence(
        requested_abort_phase="VERTICAL_DESCENT",
        before_abort={
            "phase": "VERTICAL_DESCENT",
            "target_write_count": 42,
            "telemetry_count": 41,
            "bottle_kinematic_enabled": False,
            "stage_sha256": "a" * 64,
        },
        after_abort={
            "phase": "ABORTED",
            "target_write_count": 42,
            "telemetry_count": 41,
            "bottle_kinematic_enabled": False,
            "stage_sha256": "a" * 64,
        },
        after_reset={
            "phase": "IDLE",
            "target_write_count": 43,
            "telemetry_count": 0,
            "bottle_kinematic_enabled": True,
            "stage_sha256": "a" * 64,
        },
        machine_report={"status": "ABORTED", "reason": "user_abort"},
        stage_sha256_before="a" * 64,
        stage_sha256_after="a" * 64,
    )

    assert report["status"] == "PASS"
    assert report["gates"]["no_target_write_after_abort"] is True
    assert report["gates"]["bottle_remained_dynamic_after_abort"] is True
    assert report["gates"]["reset_returned_idle"] is True
    assert report["task8"] == "NOT_RUN"


def test_abort_reset_evidence_fails_if_abort_writes_target() -> None:
    report = evaluate_abort_reset_evidence(
        requested_abort_phase="VERTICAL_DESCENT",
        before_abort={
            "phase": "VERTICAL_DESCENT",
            "target_write_count": 42,
            "telemetry_count": 41,
            "bottle_kinematic_enabled": False,
            "stage_sha256": "a" * 64,
        },
        after_abort={
            "phase": "ABORTED",
            "target_write_count": 43,
            "telemetry_count": 41,
            "bottle_kinematic_enabled": False,
            "stage_sha256": "a" * 64,
        },
        after_reset={
            "phase": "IDLE",
            "target_write_count": 44,
            "telemetry_count": 0,
            "bottle_kinematic_enabled": True,
            "stage_sha256": "a" * 64,
        },
        machine_report={"status": "ABORTED", "reason": "user_abort"},
        stage_sha256_before="a" * 64,
        stage_sha256_after="a" * 64,
    )

    assert report["status"] == "FAIL"
    assert report["gates"]["no_target_write_after_abort"] is False


def test_single_body_tensor_indices_are_explicit_int32() -> None:
    indices = single_body_tensor_indices(count=1)

    assert indices.tolist() == [0]
    assert indices.dtype == np.int32
    with pytest.raises(ValueError, match="exactly one"):
        single_body_tensor_indices(count=2)


def test_reset_body_transition_restores_state_while_dynamic() -> None:
    assert reset_body_transition_plan(initially_kinematic=True) == (
        "set_dynamic",
        "set_transform",
        "set_velocity",
        "set_kinematic",
    )
    assert reset_body_transition_plan(initially_kinematic=False) == (
        "set_transform",
        "set_velocity",
        "set_kinematic",
    )


def test_solver_active_contact_requires_matching_pair_and_finite_impulse() -> None:
    contacts = [
        {
            "actor0_path": "/World/left_finger_link",
            "actor1_path": "/World/Bottle500",
            "collider0_path": "/World/left_finger_link/mesh",
            "collider1_path": "/World/Bottle500/COL",
            "separation_m": 0.000031,
            "impulse_ns": 0.010,
        },
        {
            "actor0_path": "/World/left_finger_link",
            "actor1_path": "/World/Bottle500",
            "collider0_path": "/World/left_finger_link/mesh",
            "collider1_path": "/World/Bottle500/COL",
            "separation_m": 0.009,
            "impulse_ns": 0.0,
        },
        {
            "actor0_path": "/World/right_finger_link",
            "actor1_path": "/World/Bottle500",
            "collider0_path": "/World/right_finger_link/mesh",
            "collider1_path": "/World/Bottle500/COL",
            "separation_m": 0.0,
            "impulse_ns": float("nan"),
        },
    ]
    assert solver_active_contacts(
        contacts,
        tokens=("Bottle500", "left_finger_link"),
    ) == [contacts[0]]


def test_bilateral_observation_gate_uses_force_carrying_contact_report() -> None:
    assert bilateral_observation_contact(
        bilateral_geometric=False,
        bilateral_solver_active=True,
    )
    assert not bilateral_observation_contact(
        bilateral_geometric=True,
        bilateral_solver_active=False,
    )


def test_open_pregrasp_collider_evidence_waits_for_open_target() -> None:
    assert not open_pregrasp_evidence_ready(
        open_target_reached=False,
        already_captured=False,
    )
    assert open_pregrasp_evidence_ready(
        open_target_reached=True,
        already_captured=False,
    )
    assert not open_pregrasp_evidence_ready(
        open_target_reached=True,
        already_captured=True,
    )


@pytest.mark.parametrize(
    ("phase", "terminal", "observation", "contact", "expected"),
    [
        ("RELEASE_DYNAMIC", False, {}, {}, ["RELEASE_DYNAMIC"]),
        (
            "OPEN_PREGRASP",
            False,
            {"open_target_reached": True},
            {},
            ["OPEN_PREGRASP"],
        ),
        (
            "BILATERAL_CONTACT",
            False,
            {},
            {"bilateral_solver_active_contact": True},
            ["BILATERAL_CONTACT"],
        ),
        (
            "VERTICAL_LIFT",
            False,
            {"clearance_m": 0.0011},
            {},
            ["FIRST_SUPPORT_CLEARANCE"],
        ),
        ("HEIGHT_REACHED", False, {}, {}, ["HEIGHT_REACHED"]),
        ("HOLD", True, {}, {}, ["HOLD_END"]),
    ],
)
def test_required_collider_phase_labels_selects_sparse_milestones(
    phase: str,
    terminal: bool,  # noqa: FBT001 - parametrized input.
    observation: dict[str, object],
    contact: dict[str, object],
    expected: list[str],
) -> None:
    assert required_collider_phase_labels(
        phase=phase,
        terminal=terminal,
        observation=observation,
        contact=contact,
        captured=set(),
    ) == expected


def test_required_collider_phase_labels_does_not_recapture() -> None:
    assert required_collider_phase_labels(
        phase="HEIGHT_REACHED",
        terminal=False,
        observation={},
        contact={},
        captured={"HEIGHT_REACHED"},
    ) == []


def test_hold_duration_counts_physics_samples_not_sample_intervals() -> None:
    assert physics_sample_duration_s(
        sample_count=120,
        physics_dt_s=1.0 / 60.0,
    ) == pytest.approx(2.0)


def test_closeup_camera_looks_along_ab_and_centers_lift_interval() -> None:
    axis = np.asarray([0.9912548881, 0.1319611564, 0.0])
    grasp = np.asarray([0.0049432018, -0.1597277926, 0.0329096618])
    result = derive_gripper_closeup_camera_geometry(
        grasp_point_world_m=grasp,
        bottle_axis_world=axis,
        nominal_lift_m=0.210,
    )
    target = np.asarray(result["target_world_m"])
    position = np.asarray(result["position_world_m"])
    forward = np.asarray(result["camera_forward_world"])
    axis /= np.linalg.norm(axis)
    finger_line = np.asarray([-axis[1], axis[0], 0.0])

    assert target[2] == pytest.approx(grasp[2] + 0.105)
    assert np.dot(forward[:2], axis[:2]) < 0.0
    assert abs(float(np.dot(forward, finger_line))) < 1e-12
    assert np.linalg.norm(position - target) == pytest.approx(
        np.hypot(1.25, 0.75)
    )
    assert result["derivation"] == (
        "LOOK_ALONG_BOTTLE_AB_AND_CENTER_NOMINAL_VERTICAL_LIFT"
    )
    assert result["axial_side"] == 1


def test_closeup_camera_can_use_opposite_ab_side_without_changing_target() -> None:
    axis = np.asarray([0.9912548881, 0.1319611564, 0.0])
    grasp = np.asarray([0.0049432018, -0.1597277926, 0.0329096618])
    baseline = derive_gripper_closeup_camera_geometry(
        grasp_point_world_m=grasp,
        bottle_axis_world=axis,
        nominal_lift_m=0.210,
    )
    opposite = derive_gripper_closeup_camera_geometry(
        grasp_point_world_m=grasp,
        bottle_axis_world=axis,
        nominal_lift_m=0.210,
        axial_side=-1,
    )
    axis /= np.linalg.norm(axis)
    baseline_target = np.asarray(baseline["target_world_m"])
    opposite_target = np.asarray(opposite["target_world_m"])
    opposite_position = np.asarray(opposite["position_world_m"])
    opposite_forward = np.asarray(opposite["camera_forward_world"])

    assert opposite_target == pytest.approx(baseline_target)
    assert opposite_position[:2] - opposite_target[:2] == pytest.approx(
        -axis[:2] * 1.25
    )
    assert np.dot(opposite_forward[:2], axis[:2]) > 0.0
    assert np.linalg.norm(opposite_position - opposite_target) == pytest.approx(
        np.hypot(1.25, 0.75)
    )
    assert opposite["axial_side"] == -1


def test_subject_bounding_closeup_frames_release_bottle_and_both_fingers() -> None:
    points = np.asarray(
        [
            [0.0437, -0.0179, 0.0317],
            [-0.0437, 0.1687, 0.0334],
            [-0.2135, -0.1755, 0.3970],
            [-0.3169, -0.2232, 0.3930],
        ],
        dtype=np.float64,
    )
    result = derive_subject_bounding_closeup_camera_geometry(
        subject_points_world_m=points,
        bottle_axis_world=[-0.4244923228, 0.9054315368, 0.0],
        horizontal_fov_rad=np.deg2rad(65.0),
        vertical_fov_rad=np.deg2rad(40.0),
        near_clipping_m=1.0,
        frame_margin_fraction=0.15,
    )

    target = np.asarray(result["target_world_m"])
    position = np.asarray(result["position_world_m"])
    radius = max(float(np.linalg.norm(point - target)) for point in points)
    distance = float(np.linalg.norm(position - target))
    usable_half_fov = np.deg2rad(40.0) * 0.5 * (1.0 - 0.15)

    assert target == pytest.approx((points.min(axis=0) + points.max(axis=0)) / 2.0)
    assert distance >= 1.0 + radius
    assert np.arctan2(radius, distance) <= usable_half_fov
    assert result["subject_point_count"] == 4
    assert result["derivation"] == (
        "CURRENT_FRAME_BOTTLE_AB_AND_BILATERAL_FINGER_BOUNDING_SPHERE"
    )


def test_subject_bounding_closeup_rejects_non_horizontal_bottle_axis() -> None:
    with pytest.raises(ValueError, match="horizontal"):
        derive_subject_bounding_closeup_camera_geometry(
            subject_points_world_m=[[0, 0, 0], [1, 0, 0]],
            bottle_axis_world=[1, 0, 0.1],
            horizontal_fov_rad=1.0,
            vertical_fov_rad=0.8,
            near_clipping_m=0.1,
        )


def test_pending_evidence_camera_uses_full_finger_collider_geometry() -> None:
    source = inspect.getsource(
        IsaacGrasp20cmBindings.prepare_pending_evidence_cameras
    )

    assert "self._finger_collider_world_points()" in source
    assert 'finger_points["left"]' in source
    assert 'finger_points["right"]' in source


def test_gui_exposes_run_abort_reset_and_workspace_two() -> None:
    source = GUI_SCRIPT.read_text(encoding="utf-8")
    assert 'ui.Button("Run: Grasp + Lift 20 cm"' in source
    assert 'ui.Button("Abort"' in source
    assert 'ui.Button("Reset"' in source
    assert "_move_current_process_window_to_workspace(2)" in source
    assert "subscribe_physics_on_step_events" in source
    assert "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING" in source
    assert '"--closeup-axial-side"' in source
    assert '"--bottle-tensor-lifecycle"' in source
    assert '"--bottle-usd-velocity-readback"' in source


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("BASELINE", ("create_initial_view",)),
        (
            "INITIALIZE_KINEMATIC_BODIES",
            ("initialize_kinematic_bodies", "create_initial_view"),
        ),
        (
            "RECREATE_AFTER_DYNAMIC",
            ("create_initial_view", "recreate_after_dynamic"),
        ),
        (
            "RECREATE_AFTER_DYNAMIC_STEP",
            (
                "create_initial_view",
                "wait_one_dynamic_physics_step",
                "recreate_after_dynamic_step",
            ),
        ),
    ],
)
def test_bottle_tensor_lifecycle_plan_changes_one_operation(
    mode: str,
    expected: tuple[str, ...],
) -> None:
    assert bottle_tensor_lifecycle_plan(mode) == expected


def test_bottle_tensor_lifecycle_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported bottle tensor lifecycle"):
        bottle_tensor_lifecycle_plan("GUESS")


def test_delayed_tensor_recreation_decision_is_available() -> None:
    assert hasattr(isaac_bindings, "delayed_tensor_recreation_due")


def test_delayed_tensor_recreation_waits_for_next_physics_frame() -> None:
    decision = isaac_bindings.delayed_tensor_recreation_due

    assert not decision(
        mode="RECREATE_AFTER_DYNAMIC_STEP",
        pending=True,
        current_frame=61,
        transition_frame=61,
    )
    assert decision(
        mode="RECREATE_AFTER_DYNAMIC_STEP",
        pending=True,
        current_frame=62,
        transition_frame=61,
    )
    assert not decision(
        mode="RECREATE_AFTER_DYNAMIC",
        pending=True,
        current_frame=62,
        transition_frame=61,
    )
    assert not decision(
        mode="RECREATE_AFTER_DYNAMIC_STEP",
        pending=False,
        current_frame=62,
        transition_frame=61,
    )


def test_tensor_view_identity_requires_exact_bottle_path() -> None:
    class FakeView:
        count = 1
        prim_paths = ("/World/Session/Bottle500",)

    record = isaac_bindings.tensor_view_identity_record(
        FakeView(),
        expected_prim_path="/World/Session/Bottle500",
    )

    assert record == {
        "count": 1,
        "prim_paths": ["/World/Session/Bottle500"],
        "expected_prim_path": "/World/Session/Bottle500",
        "exact_path_match": True,
    }
    with pytest.raises(ValueError, match="does not bind exact bottle path"):
        isaac_bindings.tensor_view_identity_record(
            FakeView(),
            expected_prim_path="/World/Session/Other",
        )


def test_direct_physx_transform_readback_is_numeric() -> None:
    record = isaac_bindings.normalize_direct_physx_transform(
        {
            "ret_val": True,
            "position": (1.0, 2.0, 3.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }
    )

    assert record == {
        "available": True,
        "position_world_m": [1.0, 2.0, 3.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(ValueError, match="unavailable"):
        isaac_bindings.normalize_direct_physx_transform({"ret_val": False})


def test_usd_velocity_readback_converts_angular_degrees_to_radians() -> None:
    record = isaac_bindings.normalize_usd_velocity_readback(
        linear_velocity=(1.0, 2.0, 3.0),
        angular_velocity_deg_s=(0.0, 180.0, -90.0),
    )

    assert record["linear_velocity_world_m_s"] == [1.0, 2.0, 3.0]
    assert np.allclose(
        record["angular_velocity_world_rad_s"],
        [0.0, np.pi, -np.pi / 2.0],
    )
    assert record["angular_source_units"] == "degrees_per_second"


def test_runtime_telemetry_records_pose_finite_difference_velocity() -> None:
    source = (
        ROOT / "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    assert '"pose_finite_difference_velocity"' in source
    assert '"center_of_mass_pose_finite_difference_velocity"' in source
    assert '"bottle_tensor_lifecycle"' in source


def test_runtime_telemetry_records_post_step_com_velocity_sample() -> None:
    source = (
        ROOT / "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    assert "from tools.aloha1_mapping.bottle_com_velocity import build_sample" in source
    assert "self._build_com_velocity_sample = build_sample" in source
    assert '"synchronized_com_velocity_sample"' in source
    assert '"callback_phase": "POST_PHYSICS_STEP"' in source
    assert '"sampling_phase": "POST_PHYSICS_STEP"' in source
    assert "self.physics_context.get_physics_dt()" in source


def test_isaac_binding_uses_pose_synchronized_render_only_colliders() -> None:
    source = (
        ROOT / "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    assert "_create_bottle_render_evidence" in source
    assert "_update_bottle_render_evidence" in source
    assert "_create_finger_render_evidence" in source
    assert "_update_finger_render_evidence" in source
    assert "authored_geometry_clone" in source
    assert "physics_schemas_copied" in source
    assert "collision_schemas_copied" in source
    assert "COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES = 20" in source
    assert "for _ in range(COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES)" in source


def test_isaac_binding_enforces_initialization_and_per_frame_finger_safety() -> None:
    source = (
        ROOT / "tools/aloha1_mapping/grasp_20cm_isaac_bindings.py"
    ).read_text(encoding="utf-8")

    assert "evaluate_finger_initialization" in source
    assert "canonical_initialization_signature" in source
    assert "evaluate_finger_runtime_frame" in source
    assert "self.articulation._articulation_view.get_dof_limits()" in source
    assert "self.articulation.get_dof_limits()" not in source
    assert '"initialization_contract"' in source
    assert '"finger_safety"' in source
    assert '"first_violation"' in source
    assert "abort_on_first_runtime_violation" in source
    assert '"session_sublayer_application"' in source


def test_button_callbacks_do_not_contain_blocking_loops() -> None:
    tree = ast.parse(GUI_SCRIPT.read_text(encoding="utf-8"))
    callback_names = {
        "on_run_clicked",
        "on_abort_clicked",
        "on_reset_clicked",
    }
    callbacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in callback_names
    ]
    assert {callback.name for callback in callbacks} == callback_names
    for callback in callbacks:
        assert not any(
            isinstance(node, ast.For | ast.While)
            for node in ast.walk(callback)
        )

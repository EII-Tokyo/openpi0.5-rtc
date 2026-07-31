from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import derive_gripper_closeup_camera_geometry
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import physics_sample_duration_s
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import preload_solver_contact_ready
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import reset_body_transition_plan
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import single_body_tensor_indices
from tools.aloha1_mapping.grasp_20cm_isaac_bindings import solver_active_contacts
from tools.aloha1_mapping.grasp_20cm_runtime import EXPECTED_DOF_ORDER
from tools.aloha1_mapping.grasp_20cm_runtime import FrozenInputError
from tools.aloha1_mapping.grasp_20cm_runtime import Grasp20cmRuntimeAdapter
from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage
from tools.aloha1_mapping.grasp_20cm_runtime import verify_frozen_file
from tools.run_aloha1_grasp_20cm_gui import evaluate_abort_reset_evidence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_gui.yaml"
GUI_SCRIPT = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"


class _FakePrim:
    def __init__(self, *, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:  # noqa: N802 - matches the USD API.
        return self._valid


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


def test_gui_exposes_run_abort_reset_and_workspace_two() -> None:
    source = GUI_SCRIPT.read_text(encoding="utf-8")
    assert 'ui.Button("Run: Grasp + Lift 20 cm"' in source
    assert 'ui.Button("Abort"' in source
    assert 'ui.Button("Reset"' in source
    assert "_move_current_process_window_to_workspace(2)" in source
    assert "subscribe_physics_on_step_events" in source
    assert "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING" in source


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

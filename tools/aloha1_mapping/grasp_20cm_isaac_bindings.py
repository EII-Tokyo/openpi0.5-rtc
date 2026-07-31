"""Isaac Sim 5.1 bindings for the ALOHA Bottle500 20 cm diagnostic."""

# Isaac Sim 5.1 native APIs use positional boolean arguments.
# ruff: noqa: FBT003

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_controller import canonical_run_signature
from tools.aloha1_mapping.grasp_20cm_runtime import EXPECTED_DOF_ORDER
from tools.aloha1_mapping.grasp_20cm_runtime import sha256_file

PHASE_TIMEOUT_FRAMES = {
    Phase.VALIDATE: 60,
    Phase.SETUP_KINEMATIC: 60,
    Phase.RELEASE_DYNAMIC: 60,
    Phase.SETTLE: 600,
    Phase.OPEN_PREGRASP: 900,
    Phase.VERTICAL_DESCENT: 900,
    Phase.BILATERAL_CONTACT: 300,
    Phase.CLOSE_PRELOAD: 300,
    Phase.VERTICAL_LIFT: 1800,
    Phase.HEIGHT_REACHED: 60,
    Phase.HOLD: 240,
}
COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES = 20


def single_body_tensor_indices(*, count: int) -> np.ndarray:
    """Return the explicit index required by local PhysX tensor setters."""

    if int(count) != 1:
        raise ValueError("diagnostic rigid-body view must contain exactly one")
    return np.asarray([0], dtype=np.int32)


def reset_body_transition_plan(
    *,
    initially_kinematic: bool,
) -> tuple[str, ...]:
    """Describe the local PhysX-safe Reset ordering."""

    prefix = ("set_dynamic",) if initially_kinematic else ()
    return (
        *prefix,
        "set_transform",
        "set_velocity",
        "set_kinematic",
    )


def derive_gripper_closeup_camera_geometry(
    *,
    grasp_point_world_m: Sequence[float],
    bottle_axis_world: Sequence[float],
    nominal_lift_m: float,
    axial_distance_m: float = 1.25,
    elevation_m: float = 0.75,
) -> dict[str, list[float] | float | str]:
    """Derive an evidence view along AB, centered on the lift interval."""

    grasp = np.asarray(grasp_point_world_m, dtype=np.float64)
    axis = np.asarray(bottle_axis_world, dtype=np.float64)
    if grasp.shape != (3,) or axis.shape != (3,):
        raise ValueError("grasp point and bottle axis must be 3-vectors")
    if not np.isfinite(grasp).all() or not np.isfinite(axis).all():
        raise ValueError("camera geometry inputs must be finite")
    if abs(float(axis[2])) > 1e-6:
        raise ValueError("Bottle500 AB must be horizontal")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("bottle axis must be nonzero")
    axis /= norm
    if (
        nominal_lift_m <= 0.0
        or axial_distance_m <= 0.0
        or elevation_m <= 0.0
    ):
        raise ValueError("camera distances must be positive")
    target = grasp + np.asarray(
        [0.0, 0.0, nominal_lift_m / 2.0],
        dtype=np.float64,
    )
    position = (
        target
        + axis * float(axial_distance_m)
        + np.asarray([0.0, 0.0, elevation_m])
    )
    forward = target - position
    forward /= np.linalg.norm(forward)
    return {
        "position_world_m": position.tolist(),
        "target_world_m": target.tolist(),
        "camera_forward_world": forward.tolist(),
        "bottle_axis_world": axis.tolist(),
        "nominal_lift_m": float(nominal_lift_m),
        "axial_distance_m": float(axial_distance_m),
        "elevation_m": float(elevation_m),
        "derivation": (
            "LOOK_ALONG_BOTTLE_AB_AND_CENTER_NOMINAL_VERTICAL_LIFT"
        ),
    }


def solver_active_contacts(
    contacts: Sequence[Mapping[str, Any]],
    *,
    tokens: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Return reported pairs carrying a finite nonzero solver impulse.

    Positive separation is retained here because PhysX can generate and
    solve contacts inside the contact-offset envelope.  Callers must keep
    this distinct from geometric contact (`separation <= 0`).
    """

    records: list[Mapping[str, Any]] = []
    for contact in contacts:
        pair_text = "\n".join(
            str(contact.get(key, ""))
            for key in (
                "actor0_path",
                "actor1_path",
                "collider0_path",
                "collider1_path",
            )
        )
        if not all(token in pair_text for token in tokens):
            continue
        try:
            separation_m = float(contact["separation_m"])
            impulse_ns = float(contact["impulse_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(separation_m)
            and math.isfinite(impulse_ns)
            and impulse_ns > 0.0
        ):
            records.append(contact)
    return records


def physics_sample_duration_s(
    *,
    sample_count: int,
    physics_dt_s: float,
) -> float:
    """Measure evidence duration represented by fixed-rate physics samples."""

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    return float(sample_count) * float(physics_dt_s)


def preload_solver_contact_ready(
    *,
    close_exhausted: bool,
    left_solver_active: bool,
    right_solver_active: bool,
    coupling_residual_m: float,
    coupling_gate_m: float,
) -> bool:
    """Require bilateral force-carrying PhysX contacts for preload."""

    residual = float(coupling_residual_m)
    gate = float(coupling_gate_m)
    return bool(
        close_exhausted
        and left_solver_active
        and right_solver_active
        and math.isfinite(residual)
        and math.isfinite(gate)
        and gate >= 0.0
        and residual <= gate
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class IsaacGrasp20cmBindings:
    """Own session-only USD, PhysX, trajectory, and machine evidence state."""

    def __init__(
        self,
        *,
        app: Any,
        profile: Mapping[str, Any],
        artifact_root: Path,
        delegate_readback: Mapping[str, Any],
        bottle_xy_offset_m: Sequence[float] = (0.0, 0.0),
        additional_lift_margin_m: float = 0.0,
        capture_collider_evidence: bool = True,
    ) -> None:
        import carb.settings
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.xforms import get_world_pose
        from omni.physx import get_physx_interface
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics

        from tools.aloha1_mapping.grasp_20cm_sampling import translate_horizontal_bottle_profile
        from tools.audit_aloha1_bottle_collision_runtime import _create_bottle_render_evidence
        from tools.audit_aloha1_bottle_collision_runtime import _update_bottle_render_evidence
        from tools.run_aloha1_grasp_editor_variant_b_gui import build_external_close_targets
        from tools.validate_aloha1_follower_finger_collision_runtime import _create_finger_render_evidence
        from tools.validate_aloha1_follower_finger_collision_runtime import _update_finger_render_evidence
        from tools.validate_aloha1_gripper_coupling_ab import author_coupling_variant
        from tools.validate_aloha1_task7b2_horizontal_grasp import DIAGNOSTIC_COUPLING_CLASSIFICATION
        from tools.validate_aloha1_task7b2_horizontal_grasp import _author_session_finger_drive_type
        from tools.validate_aloha1_task7b2_horizontal_grasp import _command_positions
        from tools.validate_aloha1_task7b2_horizontal_grasp import _create_session_bottle
        from tools.validate_aloha1_task7b2_horizontal_grasp import _load_profile
        from tools.validate_aloha1_task7b2_horizontal_grasp import _look_at_quaternion
        from tools.validate_aloha1_task7b2_horizontal_grasp import _physical_contacts
        from tools.validate_aloha1_task7b2_horizontal_grasp import _serialize_contacts
        from tools.validate_aloha1_task7b2_horizontal_grasp import _solve_settled_bottle_runtime_ik
        from tools.validate_aloha1_task7b2_horizontal_grasp import _world_bounds
        from tools.validate_aloha1_task7b2_horizontal_grasp import read_physx_bottle_state
        from tools.validate_aloha1_task7b2_horizontal_grasp import transform_local_points_to_world_bounds

        self.app = app
        self.profile = dict(profile)
        self.config = self.profile["config"]
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.report_path = (
            self.artifact_root / "aloha1_grasp_20cm_runtime.json"
        )
        self.telemetry_path = (
            self.artifact_root / "aloha1_grasp_20cm_telemetry.jsonl"
        )
        self.delegate_readback = dict(delegate_readback)
        self.stage_path = Path(
            self.profile["frozen_inputs"]["stage"]["absolute_path"]
        )
        self.stage_hash_before = sha256_file(self.stage_path)
        self.dt = 1.0 / float(self.config["physics"]["frequency_hz"])
        self._get_world_pose = get_world_pose
        self._command_positions = _command_positions
        self._physical_contacts = _physical_contacts
        self._serialize_contacts = _serialize_contacts
        self._solve_settled_ik = _solve_settled_bottle_runtime_ik
        self._read_bottle_state = read_physx_bottle_state
        self._transform_collision_bounds = (
            transform_local_points_to_world_bounds
        )
        self._physx = get_physx_interface()
        self._physx_sim = get_physx_simulation_interface()
        self._settings = carb.settings.get_settings()
        self._collider_display_setting = str(
            self.config["evidence"]["collider_overlay"][
                "display_setting"
            ]
        )
        self._collider_display_value = int(
            self.config["evidence"]["collider_overlay"][
                "display_value"
            ]
        )
        self._collider_display_before = int(
            self._settings.get(self._collider_display_setting) or 0
        )

        task_profile_path = Path(
            self.profile["frozen_inputs"][
                "task7b2_runtime_profile"
            ]["absolute_path"]
        )
        self.task_profile = _load_profile(task_profile_path)
        self.task_profile = translate_horizontal_bottle_profile(
            self.task_profile,
            offset_xy_m=bottle_xy_offset_m,
        )
        self.bottle_xy_offset_m = [
            float(value) for value in bottle_xy_offset_m
        ]
        self.additional_lift_margin_m = float(
            additional_lift_margin_m
        )
        if (
            not math.isfinite(self.additional_lift_margin_m)
            or self.additional_lift_margin_m < 0.0
        ):
            raise ValueError(
                "additional lift margin must be finite and non-negative"
            )
        self.capture_collider_evidence_enabled = bool(
            capture_collider_evidence
        )
        self.task_profile["config"]["bottle"]["session_path"] = str(
            self.config["bottle"]["session_prim"]
        )
        self.task_profile["diagnostic_preload_delta_m"] = float(
            self.config["physics"]["preload_delta_m"]
        )
        self.task_profile["diagnostic_finger_drive_type"] = str(
            self.config["physics"]["finger_drive_type"]
        )
        if (
            self.task_profile["hashes"]["task7a_stage"]
            != self.stage_hash_before
        ):
            raise RuntimeError("Task7B.2 profile does not bind approved Stage")
        verified_lula = Path(
            self.profile["frozen_inputs"]["lula_descriptor"][
                "absolute_path"
            ]
        )
        if (
            self.task_profile["inputs"]["lula_descriptor"]
            != verified_lula
            or self.task_profile["hashes"]["lula_descriptor"]
            != sha256_file(verified_lula)
        ):
            raise RuntimeError("runtime IK does not bind frozen Lula descriptor")

        self.stage = get_current_stage()
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        session_root = str(
            Path(self.config["bottle"]["session_prim"]).parent
        ).replace("\\", "/")
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            if self.stage.GetPrimAtPath(session_root).IsValid():
                self.stage.RemovePrim(session_root)
            coupling = author_coupling_variant(
                stage=self.stage,
                variant="official_symmetric_adapter",
                physx_schema=PhysxSchema,
                usd_physics=UsdPhysics,
            )
            if (
                coupling["classification"]
                != DIAGNOSTIC_COUPLING_CLASSIFICATION
            ):
                raise RuntimeError("unexpected diagnostic coupling")
            self.coupling_readback = coupling
            self.drive_readback = _author_session_finger_drive_type(
                stage=self.stage,
                usd_physics=UsdPhysics,
                requested_type=str(
                    self.config["physics"]["finger_drive_type"]
                ),
            )
            (
                self.bottle_prim,
                self.bottle_session,
                self.bottle_collision_points_local,
            ) = _create_session_bottle(self.stage, self.task_profile)
            (
                self._bottle_render_evidence,
                self._bottle_render_handles,
            ) = _create_bottle_render_evidence(
                self.stage,
                bottle_path=str(self.bottle_prim.GetPath()),
            )
            (
                self._finger_render_evidence,
                self._finger_render_handles,
            ) = _create_finger_render_evidence(
                self.stage,
                finger_paths=self.config["evidence"][
                    "collider_overlay"
                ]["finger_colliders"],
            )
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["visual_root"]
                )
            ).MakeInvisible()
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeInvisible()
            pusher_visual = self.stage.GetPrimAtPath(
                self._bottle_render_evidence["pusher_visual_prim"]
            )
            if pusher_visual.IsValid():
                UsdGeom.Imageable(pusher_visual).MakeInvisible()
            for side, paths in self.config["evidence"][
                "collider_overlay"
            ]["finger_colliders"].items():
                clone_base = (
                    f"{self._finger_render_evidence['root']}/{side}"
                )
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{clone_base}/ExactVisualAtPhysxPose"
                    )
                ).MakeInvisible()
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{clone_base}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeInvisible()
                source_visual_parent = self.stage.GetPrimAtPath(
                    str(paths["visual"])
                ).GetParent()
                if source_visual_parent.IsValid():
                    UsdGeom.Imageable(
                        source_visual_parent
                    ).MakeVisible()
            self._render_evidence = {
                "authored_geometry_clone": bool(
                    self.config["evidence"]["collider_overlay"][
                        "authored_geometry_clone"
                    ]
                ),
                "semantics": str(
                    self.config["evidence"]["collider_overlay"][
                        "semantics"
                    ]
                ),
                "bottle": self._bottle_render_evidence,
                "fingers": self._finger_render_evidence,
                "physics_schemas_copied": False,
                "collision_schemas_copied": False,
            }
            self._update_bottle_render_evidence = (
                _update_bottle_render_evidence
            )
            self._update_finger_render_evidence = (
                _update_finger_render_evidence
            )

        World.clear_instance()
        self.world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=self.dt,
            rendering_dt=self.dt,
        )
        self.physics_context = self.world.get_physics_context()
        self.physics_context.set_solve_articulation_contact_last(True)
        self.articulation = SingleArticulation(
            prim_path=str(self.config["robot"]["articulation_prim"]),
            name="aloha1_grasp_20cm_follower_left",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.articulation)
        self.world.reset()
        if list(self.articulation.dof_names) != EXPECTED_DOF_ORDER:
            raise RuntimeError(
                f"unexpected DOF order: {self.articulation.dof_names}"
            )
        controller = self.articulation.get_articulation_controller()
        finger_indices = np.asarray([7, 8], dtype=np.int32)
        copied_max_force = float(
            coupling["copied_left_drive_parameters"]["max_force"]
        )
        controller.set_max_efforts(
            np.asarray(
                [copied_max_force, copied_max_force],
                dtype=np.float32,
            ),
            finger_indices,
        )
        max_efforts = np.asarray(
            controller.get_max_efforts(),
            dtype=np.float64,
        )
        if not np.allclose(
            max_efforts[finger_indices],
            copied_max_force,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("finger max-force readback mismatch")

        frozen_waypoints = self.task_profile["kinematics"]["ik"][
            "waypoints"
        ]
        pregrasp = [
            waypoint
            for waypoint in frozen_waypoints
            if waypoint["phase"] == "move_to_pregrasp"
        ]
        if not pregrasp:
            raise RuntimeError("frozen pregrasp waypoint missing")
        initial_arm = np.asarray(
            pregrasp[-1]["joint_positions_rad"],
            dtype=np.float64,
        )
        self.initial_command = np.asarray(
            [
                *initial_arm,
                0.0,
                *self.config["robot"].get(
                    "open_targets_m",
                    self.task_profile["config"]["robot"][
                        "open_targets_m"
                    ],
                ),
            ],
            dtype=np.float64,
        )
        self.command = self.initial_command.copy()
        self._target_write_count = 0
        self.articulation.set_joint_positions(self.command)
        self.articulation.set_joint_velocities(
            np.zeros_like(self.command)
        )
        self._write_joint_command()

        simulation_view = SimulationManager.get_physics_sim_view()
        if simulation_view is None or not simulation_view.is_valid:
            raise RuntimeError("PhysX tensor SimulationView unavailable")
        bottle_path = str(self.config["bottle"]["session_prim"])
        self.bottle = simulation_view.create_rigid_body_view(bottle_path)
        if self.bottle is None or int(self.bottle.count) != 1:
            raise RuntimeError("Bottle500 PhysX rigid-body view unavailable")
        self._finger_link_views = {
            side: simulation_view.create_rigid_body_view(
                str(paths["link"])
            )
            for side, paths in self.config["evidence"][
                "collider_overlay"
            ]["finger_colliders"].items()
        }
        if any(
            view is None or int(view.count) != 1
            for view in self._finger_link_views.values()
        ):
            raise RuntimeError(
                "finger PhysX rigid-body view unavailable for "
                "collider evidence"
            )
        table_bounds = _world_bounds(
            self.stage,
            str(self.config["stage"]["table_prim"]),
        )
        self.table_top_z_m = float(table_bounds["maximum"][2])
        self.base_position, self.base_orientation = self._get_world_pose(
            "/World/follower_left/vx300s_left/follower_left_base_link"
        )
        self._look_at_quaternion = _look_at_quaternion
        self._capture_attempt_index = 0
        self._initialize_video_capture()

        clearance_report = json.loads(
            self.task_profile["inputs"][
                "supplier_cad_clearance_report"
            ].read_text(encoding="utf-8")
        )
        contact_target = float(
            clearance_report["contact_solution"]["left_finger_q_m"]
        )
        close_target = contact_target - float(
            self.config["physics"]["preload_delta_m"]
        )
        self.close_targets = build_external_close_targets(
            open_position_m=float(
                self.task_profile["config"]["robot"][
                    "open_targets_m"
                ][0]
            ),
            contact_target_m=close_target,
            speed_m_s=0.02,
            physics_dt_s=self.dt,
        )

        self._contact_buffer: list[dict[str, Any]] = []
        self.all_contacts: list[dict[str, Any]] = []
        self._contact_frame = 0
        self._contact_phase = Phase.IDLE

        def on_contact(
            headers: Sequence[Any],
            data: Sequence[Any],
        ) -> None:
            records = self._serialize_contacts(
                headers,
                data,
                frame=self._contact_frame,
                time_s=self._contact_frame * self.dt,
                phase=self._contact_phase.value,
                dt=self.dt,
            )
            self._contact_buffer.extend(records)
            self.all_contacts.extend(records)

        self.contact_subscription = (
            self._physx_sim.subscribe_contact_report_events(on_contact)
        )
        self._initial_bottle_transform = np.asarray(
            self.bottle.get_transforms()[0],
            dtype=np.float64,
        )
        self._reset_runtime_records()

    def _initialize_video_capture(self) -> None:
        """Create independent local-5.1 camera/render-product pairs."""

        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        from isaacsim.sensors.camera import Camera

        grasp = np.asarray(
            self.task_profile["kinematics"]["placement"][
                "bottle_axis"
            ]["grasp_point_world_m"],
            dtype=np.float64,
        )
        base = np.asarray(self.base_position, dtype=np.float64)
        overview_target = (base + grasp) / 2.0
        overview_target[2] += 0.08
        specs = {
            "overview": {
                "position": (
                    overview_target
                    + np.asarray([1.85, -1.75, 1.45])
                ),
                "target": overview_target,
            },
        }
        closeup = derive_gripper_closeup_camera_geometry(
            grasp_point_world_m=grasp,
            bottle_axis_world=self.task_profile["kinematics"][
                "placement"
            ]["bottle_axis"]["unit_world"],
            nominal_lift_m=(
                float(self.config["target"]["clearance_m"])
                + float(self.config["target"]["hold_drop_gate_m"])
            ),
        )
        specs["gripper_closeup"] = {
            "position": np.asarray(
                closeup["position_world_m"],
                dtype=np.float64,
            ),
            "target": np.asarray(
                closeup["target_world_m"],
                dtype=np.float64,
            ),
            "derivation": closeup,
        }
        self.video_cameras: dict[str, dict[str, Any]] = {}
        for view, spec in specs.items():
            orientation = self._look_at_quaternion(
                spec["position"],
                spec["target"],
            )
            camera = Camera(
                prim_path=(
                    "/World/ALOHA1Grasp20cmSession/Cameras/"
                    f"{view}"
                ),
                name=f"aloha1_grasp_20cm_{view}",
                position=spec["position"],
                orientation=orientation,
                resolution=(960, 540),
                annotator_device="cpu",
            )
            camera.initialize(attach_rgb_annotator=True)
            camera.set_world_pose(
                position=spec["position"],
                orientation=orientation,
                camera_axes="usd",
            )
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = quats_to_rot_matrices(orientation)
            matrix[:3, 3] = spec["position"]
            self.video_cameras[view] = {
                "camera": camera,
                "position_world_m": spec["position"].tolist(),
                "target_world_m": spec["target"].tolist(),
                "orientation_wxyz": orientation.tolist(),
                "camera_world_matrix": matrix.tolist(),
                "resolution": [960, 540],
                "render_product_path": str(
                    camera.get_render_product_path()
                ),
                "view_status": (
                    "ENGINEERING_EVIDENCE_VIEW_NOT_CALIBRATED"
                ),
                "derivation": spec.get(
                    "derivation",
                    {
                        "derivation": (
                            "FULL_ARM_OVERVIEW_FROM_FROZEN_BASE_AND_GRASP"
                        )
                    },
                ),
            }
        render_products = {
            record["render_product_path"]
            for record in self.video_cameras.values()
        }
        if len(render_products) != 2:
            raise RuntimeError(
                "video views do not have independent render products"
            )

    def _reset_runtime_records(self) -> None:
        self.started_at = time.perf_counter()
        self.observations: list[RunObservation] = []
        self.telemetry: list[dict[str, Any]] = []
        self._phase = Phase.IDLE
        self._phase_frames = 0
        self._trajectory: dict[Phase, list[np.ndarray]] = {}
        self._trajectory_cursor: dict[Phase, int] = {}
        self._ik_report: dict[str, Any] = {"status": "NOT_RUN"}
        self._setup_complete = True
        self._preload_stable_frames = 0
        self._bilateral_before_lift = False
        self._bilateral_through_hold = True
        self._support_contact_ever = False
        self._height_reached = False
        self._hold_reference_clearance_m: float | None = None
        self._maximum_clearance_m = -math.inf
        self._initial_ee_z_m: float | None = None
        self._deep_penetration_frames: list[int] = []
        self._last_snapshot = {
            "clearance_m": 0.0,
            "maximum_clearance_m": 0.0,
            "left_contact": False,
            "right_contact": False,
            "ee_position_world_m": None,
            "ik": "NOT_RUN",
            "fingers": None,
            "bottle_velocity": None,
            "hold_drop_m": 0.0,
        }
        self._pending_capture_frames: list[dict[str, Any]] = []
        self._captured_frame_records: list[dict[str, Any]] = []
        self._video_capture_finalized = False
        self._video_capture_error: str | None = None
        self._video_attempt_root: Path | None = None
        self._collider_overlay_records: list[dict[str, Any]] = []
        self._captured_overlay_phases: set[str] = set()

    def prepare_run(self) -> None:
        if sha256_file(self.stage_path) != self.stage_hash_before:
            raise RuntimeError("approved Stage hash changed before Run")
        if not bool(
            self.physics_context.get_solve_articulation_contact_last()
        ):
            raise RuntimeError(
                "solve_articulation_contact_last readback is false"
            )
        self._reset_runtime_records()
        while True:
            self._capture_attempt_index += 1
            candidate = (
                self.artifact_root
                / f"video_attempt_{self._capture_attempt_index:03d}"
            )
            if not candidate.exists():
                self._video_attempt_root = candidate
                break
        assert self._video_attempt_root is not None
        self._video_attempt_root.mkdir(parents=True, exist_ok=False)
        self._phase = Phase.VALIDATE

    def _set_phase(self, phase: Phase) -> None:
        if phase is not self._phase:
            self._phase = phase
            self._phase_frames = 0
            self._contact_phase = phase
        self._phase_frames += 1

    def _build_runtime_trajectories(self) -> None:
        if self._ik_report.get("status") == "PASS":
            return
        bottle_state = self._read_bottle_state(self.bottle)
        ee_position, ee_orientation = self._get_world_pose(
            str(self.config["robot"]["end_effector_prim"])
        )
        current_q = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        nominal_lift_m = float(
            self.config["target"]["clearance_m"]
            + self.config["target"]["hold_drop_gate_m"]
            + self.additional_lift_margin_m
        )
        from tools.aloha1_mapping.grasp_20cm_sampling import extend_profile_for_clearance_lift

        extended_profile = extend_profile_for_clearance_lift(
            self.task_profile,
            target_clearance_m=float(
                self.config["target"]["clearance_m"]
            ),
            hold_drop_gate_m=float(
                self.config["target"]["hold_drop_gate_m"]
            ),
            additional_lift_margin_m=self.additional_lift_margin_m,
        )
        result = self._solve_settled_ik(
            extended_profile,
            base_position=np.asarray(
                self.base_position,
                dtype=np.float64,
            ),
            base_orientation=np.asarray(
                self.base_orientation,
                dtype=np.float64,
            ),
            bottle_state=bottle_state,
            current_ee_position=np.asarray(
                ee_position,
                dtype=np.float64,
            ),
            current_ee_orientation=np.asarray(
                ee_orientation,
                dtype=np.float64,
            ),
            current_arm_q=current_q[:6],
        )
        self._ik_report = {
            **result,
            "nominal_vertical_lift_m": nominal_lift_m,
            "actual_gate": (
                "BOTTLE_COLLISION_MIN_WORLD_Z_MINUS_TABLE_TOP_WORLD_Z"
            ),
        }
        if result["status"] != "PASS":
            raise RuntimeError(
                "settled Bottle500 runtime IK failed: "
                f"{result.get('failure_phase')}"
            )
        phase_map = {
            "move_to_pregrasp": Phase.OPEN_PREGRASP,
            "vertical_descent": Phase.VERTICAL_DESCENT,
            "vertical_lift": Phase.VERTICAL_LIFT,
        }
        for phase_name, phase in phase_map.items():
            self._trajectory[phase] = [
                np.asarray(
                    waypoint["joint_positions_rad"],
                    dtype=np.float64,
                )
                for waypoint in result["waypoints"]
                if waypoint["phase"] == phase_name
            ]
            self._trajectory_cursor[phase] = 0
            if not self._trajectory[phase]:
                raise RuntimeError(
                    f"runtime IK has no {phase_name} waypoints"
                )
        self._trajectory[Phase.BILATERAL_CONTACT] = []
        self._trajectory_cursor[Phase.BILATERAL_CONTACT] = 0
        self._trajectory_cursor[Phase.CLOSE_PRELOAD] = 0

    def _advance_arm(self, phase: Phase) -> None:
        targets = self._trajectory.get(phase, [])
        cursor = self._trajectory_cursor.get(phase, 0)
        if cursor >= len(targets):
            return
        self.command[:6] = targets[cursor]
        self._trajectory_cursor[phase] = cursor + 1
        self._write_joint_command()

    def _advance_close(self, phase: Phase) -> None:
        cursor = self._trajectory_cursor.get(
            Phase.BILATERAL_CONTACT,
            0,
        )
        if cursor < len(self.close_targets):
            left = float(self.close_targets[cursor])
            self.command[7] = left
            self.command[8] = -left
            self._trajectory_cursor[Phase.BILATERAL_CONTACT] = cursor + 1
        self._trajectory_cursor[phase] = self._trajectory_cursor.get(
            phase,
            0,
        ) + 1
        self._write_joint_command()

    def _write_joint_command(self) -> None:
        self._command_positions(self.articulation, self.command)
        self._target_write_count += 1

    def apply_phase_target(self, phase: Phase) -> None:
        self._set_phase(phase)
        if phase in {
            Phase.VALIDATE,
            Phase.SETUP_KINEMATIC,
            Phase.RELEASE_DYNAMIC,
            Phase.SETTLE,
            Phase.HEIGHT_REACHED,
            Phase.HOLD,
        }:
            self._write_joint_command()
        elif phase is Phase.OPEN_PREGRASP:
            self._build_runtime_trajectories()
            self._advance_arm(phase)
        elif phase is Phase.VERTICAL_DESCENT:
            self._advance_arm(phase)
        elif phase in {
            Phase.BILATERAL_CONTACT,
            Phase.CLOSE_PRELOAD,
        }:
            self._advance_close(phase)
        elif phase is Phase.VERTICAL_LIFT:
            self._advance_arm(phase)
        if phase is Phase.HEIGHT_REACHED:
            self._height_reached = True
            self._hold_reference_clearance_m = float(
                self._last_snapshot["clearance_m"]
            )

    def set_bottle_kinematic(self, *, enabled: bool) -> None:
        from pxr import UsdPhysics

        rigid = UsdPhysics.RigidBodyAPI(self.bottle_prim)
        rigid.GetKinematicEnabledAttr().Set(bool(enabled))
        self._physx_sim.flush_changes()

    def _phase_done(self, phase: Phase) -> bool:
        targets = self._trajectory.get(phase, [])
        return bool(targets) and self._trajectory_cursor.get(
            phase,
            0,
        ) >= len(targets)

    def read_observation(
        self,
        *,
        frame: int,
        time_s: float,
    ) -> RunObservation:
        from pxr import UsdPhysics

        self._contact_frame = frame
        self._physx.update_transformations(True, True, False, False)
        bottle_state = self._read_bottle_state(self.bottle)
        position = np.asarray(
            bottle_state["position_world_m"],
            dtype=np.float64,
        )
        orientation = np.asarray(
            bottle_state["orientation_wxyz"],
            dtype=np.float64,
        )
        collision_bounds = self._transform_collision_bounds(
            local_points=self.bottle_collision_points_local,
            position_world=position,
            orientation_world_wxyz=orientation,
        )
        clearance_m = (
            float(collision_bounds["minimum"][2])
            - self.table_top_z_m
        )
        self._maximum_clearance_m = max(
            self._maximum_clearance_m,
            clearance_m,
        )
        current_contacts = self._contact_buffer
        self._contact_buffer = []
        bottle_token = str(self.config["bottle"]["session_prim"])
        left_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_left_finger",
            ),
        )
        right_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_right_finger",
            ),
        )
        left_solver_contacts = solver_active_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_left_finger",
            ),
        )
        right_solver_contacts = solver_active_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                "diagnostic_supplier_cad_right_finger",
            ),
        )
        support_contacts = self._physical_contacts(
            current_contacts,
            tokens=(
                bottle_token,
                str(self.config["stage"]["table_prim"]).rsplit(
                    "/",
                    maxsplit=1,
                )[-1],
            ),
        )
        if support_contacts:
            self._support_contact_ever = True
        support_contact = bool(support_contacts) or bool(
            self._support_contact_ever
            and clearance_m
            <= float(
                self.config["target"][
                    "support_contact_latch_clearance_m"
                ]
            )
        )
        bilateral_geometric = bool(left_contacts and right_contacts)
        bilateral_solver_active = bool(
            left_solver_contacts and right_solver_contacts
        )
        bilateral = (
            bilateral_solver_active
            if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
            else bilateral_geometric
        )
        if bilateral_geometric and self._phase in {
            Phase.BILATERAL_CONTACT,
            Phase.CLOSE_PRELOAD,
        }:
            self._bilateral_before_lift = True
        if self._phase is Phase.HOLD:
            self._bilateral_through_hold &= bilateral

        qpos = np.asarray(
            self.articulation.get_joint_positions(),
            dtype=np.float64,
        )
        qvel = np.asarray(
            self.articulation.get_joint_velocities(),
            dtype=np.float64,
        )
        ee_position, _ = self._get_world_pose(
            str(self.config["robot"]["end_effector_prim"])
        )
        ee_position = np.asarray(ee_position, dtype=np.float64)
        if self._initial_ee_z_m is None:
            self._initial_ee_z_m = float(ee_position[2])
        ee_displacement = (
            float(ee_position[2]) - self._initial_ee_z_m
        )
        coupling_residual = abs(float(qpos[7] + qpos[8]))
        close_exhausted = (
            self._trajectory_cursor.get(
                Phase.BILATERAL_CONTACT,
                0,
            )
            >= len(self.close_targets)
        )
        if self._phase is Phase.CLOSE_PRELOAD and preload_solver_contact_ready(
            close_exhausted=close_exhausted,
            left_solver_active=bool(left_solver_contacts),
            right_solver_active=bool(right_solver_contacts),
            coupling_residual_m=coupling_residual,
            coupling_gate_m=0.001,
        ):
            self._preload_stable_frames += 1
        elif self._phase is Phase.CLOSE_PRELOAD:
            self._preload_stable_frames = 0
        preload_complete = self._preload_stable_frames >= 5
        hold_drop_m = (
            max(
                0.0,
                self._hold_reference_clearance_m - clearance_m,
            )
            if self._hold_reference_clearance_m is not None
            else 0.0
        )
        deep = [
            contact
            for contact in current_contacts
            if bottle_token in (
                f"{contact.get('collider0_path', '')} "
                f"{contact.get('collider1_path', '')}"
            )
            and float(contact["separation_m"]) < -0.005
        ]
        if deep:
            self._deep_penetration_frames.append(frame)
        persistent_penetration = all(
            frame - offset in self._deep_penetration_frames
            for offset in (0, 1, 2)
        )
        finite_values = np.concatenate(
            (
                qpos,
                qvel,
                position,
                np.asarray(
                    bottle_state["linear_velocity_world_m_s"],
                    dtype=np.float64,
                ),
                np.asarray(
                    bottle_state["angular_velocity_world_rad_s"],
                    dtype=np.float64,
                ),
            )
        )
        maximum_speed = float(
            np.linalg.norm(
                bottle_state["linear_velocity_world_m_s"]
            )
        )
        maximum_angular = float(
            bottle_state["angular_speed_rad_s"]
        )
        phase_timed_out = self._phase_frames > PHASE_TIMEOUT_FRAMES.get(
            self._phase,
            600,
        )
        dynamic = not bool(
            UsdPhysics.RigidBodyAPI(self.bottle_prim)
            .GetKinematicEnabledAttr()
            .Get()
        )
        observation = RunObservation(
            frame=frame,
            time_s=time_s,
            clearance_m=clearance_m,
            bottle_dynamic=dynamic,
            support_contact=support_contact,
            bottle_linear_speed_m_s=maximum_speed,
            bottle_angular_speed_rad_s=maximum_angular,
            stage_contract_valid=(
                sha256_file(self.stage_path) == self.stage_hash_before
            ),
            setup_complete=self._setup_complete,
            open_target_reached=self._phase_done(
                Phase.OPEN_PREGRASP
            ),
            descent_complete=self._phase_done(
                Phase.VERTICAL_DESCENT
            ),
            bilateral_contact=bilateral,
            preload_complete=preload_complete,
            lift_waypoint_exhausted=self._phase_done(
                Phase.VERTICAL_LIFT
            ),
            hold_drop_m=hold_drop_m,
            finite_state=bool(np.isfinite(finite_values).all()),
            persistent_penetration=persistent_penetration,
            numerical_ejection=bool(
                maximum_speed > 5.0 or maximum_angular > 50.0
            ),
            forbidden_constraint=False,
            phase_timed_out=phase_timed_out,
            ee_vertical_displacement_m=ee_displacement,
        )
        self.observations.append(observation)
        record = {
            "frame": frame,
            "time_s": time_s,
            "phase": self._phase.value,
            "observation": asdict(observation),
            "joint_target": self.command.tolist(),
            "joint_readback": qpos.tolist(),
            "joint_velocity": qvel.tolist(),
            "bottle": {
                **bottle_state,
                "collision_bounds": collision_bounds,
            },
            "contact_semantics": {
                "geometric_contact_definition": "separation_m <= 0",
                "solver_active_definition": (
                    "finite impulse_ns > 0 inside reported contact pair"
                ),
                "left_geometric_contact": bool(left_contacts),
                "right_geometric_contact": bool(right_contacts),
                "bilateral_geometric_contact": bilateral_geometric,
                "left_solver_active_contact": bool(
                    left_solver_contacts
                ),
                "right_solver_active_contact": bool(
                    right_solver_contacts
                ),
                "bilateral_solver_active_contact": (
                    bilateral_solver_active
                ),
                "observation_contact_gate": (
                    "SOLVER_ACTIVE_AFTER_HEIGHT_REACHED"
                    if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
                    else "GEOMETRIC_BEFORE_HEIGHT_REACHED"
                ),
            },
            "contacts": current_contacts,
        }
        self.telemetry.append(record)
        self._pending_capture_frames.append(
            {
                "physics_frame": frame,
                "time_s": time_s,
                "phase": self._phase.value,
                "telemetry_index": len(self.telemetry) - 1,
            }
        )
        self._last_snapshot = {
            "clearance_m": clearance_m,
            "maximum_clearance_m": self._maximum_clearance_m,
            "left_contact": (
                bool(left_solver_contacts)
                if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
                else bool(left_contacts)
            ),
            "right_contact": (
                bool(right_solver_contacts)
                if self._phase in {Phase.HEIGHT_REACHED, Phase.HOLD}
                else bool(right_contacts)
            ),
            "left_geometric_contact": bool(left_contacts),
            "right_geometric_contact": bool(right_contacts),
            "left_solver_active_contact": bool(left_solver_contacts),
            "right_solver_active_contact": bool(right_solver_contacts),
            "ee_position_world_m": ee_position.tolist(),
            "ik": self._ik_report.get("status", "NOT_RUN"),
            "fingers": {
                "target_m": self.command[[7, 8]].tolist(),
                "readback_m": qpos[[7, 8]].tolist(),
                "coupling_residual_m": coupling_residual,
            },
            "bottle_velocity": {
                "linear_world_m_s": bottle_state[
                    "linear_velocity_world_m_s"
                ],
                "angular_world_rad_s": bottle_state[
                    "angular_velocity_world_rad_s"
                ],
            },
            "hold_drop_m": hold_drop_m,
        }
        return observation

    @staticmethod
    def _jsonable_render_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): IsaacGrasp20cmBindings._jsonable_render_value(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def capture_pending_render_frame(self) -> bool:
        """Save both camera products after the enclosing app update."""

        if not self._pending_capture_frames:
            return False
        if len(self._pending_capture_frames) != 1:
            raise RuntimeError(
                "render capture fell behind physics: "
                f"{len(self._pending_capture_frames)} pending frames"
            )
        if self._video_attempt_root is None:
            raise RuntimeError("video attempt root is not initialized")
        from tools.validate_aloha1_task7b2_horizontal_grasp import FULL_ARM_LINK_PRIMS
        from tools.validate_aloha1_task7b2_horizontal_grasp import _full_arm_framing_evidence

        pending = self._pending_capture_frames[0]
        frame = int(pending["physics_frame"])
        telemetry = self.telemetry[int(pending["telemetry_index"])]
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices

        bottle = telemetry["bottle"]
        bottle_position = np.asarray(
            bottle["position_world_m"],
            dtype=np.float64,
        )
        bottle_rotation = quats_to_rot_matrices(
            np.asarray(
                bottle["orientation_wxyz"],
                dtype=np.float64,
            )
        )
        bottle_axis_config = self.task_profile["config"]["bottle"][
            "axis"
        ]
        a_world = (
            bottle_rotation
            @ np.asarray(
                bottle_axis_config["a_local_m"],
                dtype=np.float64,
            )
            + bottle_position
        )
        b_world = (
            bottle_rotation
            @ np.asarray(
                bottle_axis_config["b_local_m"],
                dtype=np.float64,
            )
            + bottle_position
        )
        left_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"][
                    "left_finger_collider"
                ]
            )
        )
        right_origin, _ = self._get_world_pose(
            str(
                self.task_profile["config"]["robot"][
                    "right_finger_collider"
                ]
            )
        )
        projection_points = {
            "bottle_a": a_world.tolist(),
            "bottle_b": b_world.tolist(),
            "left_finger_collider_origin": np.asarray(
                left_origin,
                dtype=np.float64,
            ).tolist(),
            "right_finger_collider_origin": np.asarray(
                right_origin,
                dtype=np.float64,
            ).tolist(),
        }
        selected_contacts: dict[str, dict[str, Any] | None] = {}
        for side, token in (
            ("left", "diagnostic_supplier_cad_left_finger"),
            ("right", "diagnostic_supplier_cad_right_finger"),
        ):
            candidates = [
                item
                for item in telemetry["contacts"]
                if token
                in (
                    f"{item.get('collider0_path', '')} "
                    f"{item.get('collider1_path', '')}"
                )
                and math.isfinite(float(item["impulse_ns"]))
            ]
            selected = (
                max(candidates, key=lambda item: float(item["impulse_ns"]))
                if candidates
                else None
            )
            selected_contacts[side] = selected
            if selected is not None:
                point = np.asarray(
                    selected["position_world_m"],
                    dtype=np.float64,
                )
                normal = np.asarray(
                    selected["normal_world"],
                    dtype=np.float64,
                )
                projection_points[f"{side}_contact"] = point.tolist()
                projection_points[f"{side}_normal_endpoint"] = (
                    point + 0.030 * normal
                ).tolist()
        camera_samples: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for view, camera_record in self.video_cameras.items():
            camera = camera_record["camera"]
            rgba = camera.get_rgba(device="cpu")
            if rgba is None:
                return False
            pixels = np.asarray(rgba)
            if pixels.size == 0:
                return False
            current = camera.get_current_frame(clone=True)
            rendering_time = float(current.get("rendering_time", -1.0))
            if rendering_time < 0.0:
                return False
            camera_samples[view] = (pixels, current)
        view_records: dict[str, dict[str, Any]] = {}
        rendering_times: list[float] = []
        for view, camera_record in self.video_cameras.items():
            camera = camera_record["camera"]
            pixels, current = camera_samples[view]
            if pixels.shape != (540, 960, 4):
                raise RuntimeError(
                    f"{view} unexpected RGBA shape {pixels.shape}"
                )
            if pixels.dtype != np.uint8:
                if not np.isfinite(pixels).all():
                    raise RuntimeError(f"{view} contains non-finite pixels")
                pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            destination = (
                self._video_attempt_root
                / "frames"
                / view
                / f"{frame:06d}.png"
            ).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels, mode="RGBA").convert("RGB").save(
                destination
            )
            rendering_time = float(current.get("rendering_time", -1.0))
            if not math.isfinite(rendering_time):
                raise RuntimeError(
                    f"{view} has invalid rendering_time"
                )
            rendering_times.append(rendering_time)
            record = {
                "absolute_path": str(destination),
                "sha256": sha256_file(destination),
                "resolution": [960, 540],
                "physics_frame": frame,
                "time_s": float(pending["time_s"]),
                "runtime_signature": None,
                "rendering_time_s": rendering_time,
                "rendering_frame": self._jsonable_render_value(
                    current.get("rendering_frame")
                ),
                "camera_prim_path": str(camera.prim_path),
                "render_product_path": str(
                    camera.get_render_product_path()
                ),
                "camera_world_matrix": camera_record[
                    "camera_world_matrix"
                ],
                "camera_pose": {
                    "position_world_m": camera_record[
                        "position_world_m"
                    ],
                    "orientation_wxyz": camera_record[
                        "orientation_wxyz"
                    ],
                    "target_world_m": camera_record["target_world_m"],
                },
                "view_status": camera_record["view_status"],
                "camera_derivation": camera_record["derivation"],
                "projection_world_points": projection_points,
                "projection_pixels_xy": {
                    label: pixel.tolist()
                    for label, pixel in zip(
                        projection_points,
                        np.asarray(
                            camera.get_image_coords_from_world_points(
                                np.asarray(
                                    list(projection_points.values()),
                                    dtype=np.float64,
                                )
                            ),
                            dtype=np.float64,
                        ),
                        strict=True,
                    )
                    if np.isfinite(pixel).all()
                },
                "selected_contact_records": selected_contacts,
            }
            if view == "overview":
                framing = _full_arm_framing_evidence(
                    stage=self.stage,
                    camera=camera,
                    camera_world_matrix=camera_record[
                        "camera_world_matrix"
                    ],
                    resolution=(960, 540),
                    required_link_prims=FULL_ARM_LINK_PRIMS,
                    required_scene_prims=(
                        str(self.config["bottle"]["session_prim"]),
                        str(self.config["stage"]["table_prim"]),
                    ),
                )
                record["framing_evidence"] = {
                    **framing,
                    "required_full_arm_links_in_frame": list(
                        framing["projected_in_frame_links"]
                    ),
                    "occlusion_status": (
                        "PENDING_VISUAL_MODEL_REVIEW"
                    ),
                }
            view_records[view] = record
        if max(rendering_times) - min(rendering_times) > 1e-9:
            raise RuntimeError(
                "two camera products are not from the same render time: "
                f"{rendering_times}"
            )
        self._captured_frame_records.append(
            {
                "physics_frame": frame,
                "time_s": float(pending["time_s"]),
                "phase": str(pending["phase"]),
                "telemetry_index": int(pending["telemetry_index"]),
                "views": view_records,
            }
        )
        self._pending_capture_frames.pop(0)
        return True

    @property
    def video_capture_finalized(self) -> bool:
        return self._video_capture_finalized

    @property
    def has_pending_video_frame(self) -> bool:
        return bool(self._pending_capture_frames)

    def capture_required_collider_evidence(
        self,
        *,
        terminal: bool,
    ) -> list[str]:
        """Capture exact-pose viewport overlays for required milestones."""

        if not self.capture_collider_evidence_enabled:
            return []
        if not self._captured_frame_records or self._video_attempt_root is None:
            return []
        frame_record = self._captured_frame_records[-1]
        telemetry = self.telemetry[int(frame_record["telemetry_index"])]
        phase = str(frame_record["phase"])
        observation = telemetry["observation"]
        contact = telemetry["contact_semantics"]
        labels: list[str] = []
        if (
            phase == Phase.RELEASE_DYNAMIC.value
            and "RELEASE_DYNAMIC" not in self._captured_overlay_phases
        ):
            labels.append("RELEASE_DYNAMIC")
        if (
            phase == Phase.OPEN_PREGRASP.value
            and "OPEN_PREGRASP" not in self._captured_overlay_phases
        ):
            labels.append("OPEN_PREGRASP")
        if (
            phase == Phase.BILATERAL_CONTACT.value
            and bool(contact["bilateral_geometric_contact"])
            and "BILATERAL_CONTACT"
            not in self._captured_overlay_phases
        ):
            labels.append("BILATERAL_CONTACT")
        if (
            phase == Phase.VERTICAL_LIFT.value
            and float(observation["clearance_m"]) > 0.001
            and "FIRST_SUPPORT_CLEARANCE"
            not in self._captured_overlay_phases
        ):
            labels.append("FIRST_SUPPORT_CLEARANCE")
        if (
            phase == Phase.HEIGHT_REACHED.value
            and "HEIGHT_REACHED" not in self._captured_overlay_phases
        ):
            labels.append("HEIGHT_REACHED")
        if (
            terminal
            and phase == Phase.HOLD.value
            and "HOLD_END" not in self._captured_overlay_phases
        ):
            labels.append("HOLD_END")
        if not labels:
            return []

        from omni.kit.viewport.utility import get_active_viewport

        from tools.validate_aloha1_task7b2_horizontal_grasp import _capture_viewport_png

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("active viewport unavailable for collider capture")
        frame = int(frame_record["physics_frame"])
        captured: list[str] = []
        try:
            from pxr import UsdGeom

            bottle_state = self._read_bottle_state(self.bottle)
            self._update_bottle_render_evidence(
                self.stage,
                handles=self._bottle_render_handles,
                position_world=bottle_state["position_world_m"],
                orientation_world_wxyz=bottle_state[
                    "orientation_wxyz"
                ],
            )
            self._update_finger_render_evidence(
                self.stage,
                handles=self._finger_render_handles,
                link_transforms={
                    side: np.asarray(
                        view.get_transforms()[0],
                        dtype=np.float64,
                    ).tolist()
                    for side, view in self._finger_link_views.items()
                },
            )
            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeVisible()
            for side in self._finger_link_views:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{self._finger_render_evidence['root']}/"
                        f"{side}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeVisible()
            self._settings.set_int(
                self._collider_display_setting,
                self._collider_display_value,
            )
            readback = int(
                self._settings.get(self._collider_display_setting) or 0
            )
            if readback != self._collider_display_value:
                raise RuntimeError(
                    "collider visualization setting readback mismatch"
                )
            for label in labels:
                for view, camera_record in self.video_cameras.items():
                    destination = (
                        self._video_attempt_root
                        / "collider_overlay"
                        / label
                        / f"{view}_raw.png"
                    ).resolve()
                    width, height = _capture_viewport_png(
                        self.app,
                        viewport,
                        camera_path=str(camera_record["camera"].prim_path),
                        destination=destination,
                    )
                    normal = Path(
                        frame_record["views"][view]["absolute_path"]
                    )
                    self._collider_overlay_records.append(
                        {
                            "phase_label": label,
                            "runtime_phase": phase,
                            "physics_frame": frame,
                            "time_s": float(frame_record["time_s"]),
                            "view": view,
                            "normal_absolute_path": str(normal),
                            "normal_sha256": sha256_file(normal),
                            "collider_overlay_absolute_path": str(
                                destination
                            ),
                            "collider_overlay_sha256": sha256_file(
                                destination
                            ),
                            "resolution": [width, height],
                            "camera_prim_path": str(
                                camera_record["camera"].prim_path
                            ),
                            "camera_world_matrix": camera_record[
                                "camera_world_matrix"
                            ],
                            "setting": {
                                "path": self._collider_display_setting,
                                "requested": (
                                    self._collider_display_value
                                ),
                                "readback": readback,
                            },
                            "render_evidence": {
                                "semantics": self._render_evidence[
                                    "semantics"
                                ],
                                "authored_geometry_clone": True,
                                "physics_schemas_copied": False,
                                "collision_schemas_copied": False,
                                "bottle_collider_mesh_count": (
                                    self._bottle_render_evidence[
                                        "collider_mesh_count"
                                    ]
                                ),
                                "finger_collider_mesh_count": sum(
                                    1
                                    for record in (
                                        self._finger_render_evidence[
                                            "records"
                                        ]
                                    )
                                    if record["category"]
                                    == "collider"
                                ),
                            },
                        }
                    )
                self._captured_overlay_phases.add(label)
                captured.append(label)
        finally:
            from pxr import UsdGeom

            UsdGeom.Imageable(
                self.stage.GetPrimAtPath(
                    self._bottle_render_evidence["collider_root"]
                )
            ).MakeInvisible()
            for side in self._finger_link_views:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath(
                        f"{self._finger_render_evidence['root']}/"
                        f"{side}/AuthoredColliderAtPhysxPose"
                    )
                ).MakeInvisible()
            self._settings.set_int(
                self._collider_display_setting,
                self._collider_display_before,
            )
            for _ in range(COLLIDER_OVERLAY_RENDER_FLUSH_UPDATES):
                self.app.update()
            restored = int(
                self._settings.get(self._collider_display_setting) or 0
            )
            if restored != self._collider_display_before:
                raise RuntimeError(
                    "collider visualization setting was not restored"
                )
        return captured

    def finalize_video_capture(self) -> dict[str, Any]:
        """Bind frames to the terminal signature and build candidates."""

        if self._video_capture_finalized:
            candidate_path = (
                self._video_attempt_root / "video" / "candidate_manifest.json"
            )
            return json.loads(candidate_path.read_text(encoding="utf-8"))
        if self._pending_capture_frames:
            raise RuntimeError("cannot finalize with pending video frames")
        if self._video_attempt_root is None:
            raise RuntimeError("video attempt root is not initialized")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        signature = str(report["deterministic_signature"])
        for record in self._captured_frame_records:
            for view_record in record["views"].values():
                view_record["runtime_signature"] = signature
        manifest = {
            "schema_version": 1,
            "runtime_signature": signature,
            "required_full_arm_links": [
                "base",
                "shoulder",
                "elbow",
                "forearm",
                "wrist",
                "gripper",
            ],
            "capture_api": {
                "class": "isaacsim.sensors.camera.Camera",
                "extension": "isaacsim.sensors.camera",
                "version_scope": "ISAAC_SIM_5_1_0_0_KIT_107_3_3",
                "independent_render_products": True,
                "capture_location": (
                    "OUTER_APP_LOOP_AFTER_PHYSICS_AND_RENDER_UPDATE"
                ),
            },
            "collision_evidence": {
                "enabled": self.capture_collider_evidence_enabled,
                "purpose": (
                    "PAIRED_COLLIDER_SCREENSHOT_REPEAT"
                    if self.capture_collider_evidence_enabled
                    else "PRIMARY_CLEAN_VIDEO"
                ),
                "setting_path": self._collider_display_setting,
                "setting_before": self._collider_display_before,
                "setting_after": int(
                    self._settings.get(
                        self._collider_display_setting
                    )
                    or 0
                ),
                "required_phase_labels": (
                    [
                        "RELEASE_DYNAMIC",
                        "OPEN_PREGRASP",
                        "BILATERAL_CONTACT",
                        "FIRST_SUPPORT_CLEARANCE",
                        "HEIGHT_REACHED",
                        "HOLD_END",
                    ]
                    if self.capture_collider_evidence_enabled
                    else []
                ),
                "captured_phase_labels": sorted(
                    self._captured_overlay_phases
                ),
                "render_evidence": self._render_evidence,
                "records": self._collider_overlay_records,
            },
            "frames": self._captured_frame_records,
        }
        manifest_path = (
            self._video_attempt_root / "frame_manifest.json"
        )
        _atomic_json(manifest_path, manifest)
        from tools.build_aloha1_grasp_20cm_video import build_video_evidence

        candidate = build_video_evidence(
            report_path=self.report_path,
            telemetry_path=self.telemetry_path,
            frame_manifest_path=manifest_path,
            output_root=self._video_attempt_root / "video",
        )
        self._video_capture_finalized = True
        return candidate

    def save_video_capture_exception(self, exception_text: str) -> None:
        self._video_capture_error = exception_text[-12000:]
        root = self._video_attempt_root or self.artifact_root
        _atomic_json(
            root / "video_capture_error.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "reason": "video_capture_exception",
                "exception": self._video_capture_error,
                "captured_frame_count": len(
                    self._captured_frame_records
                ),
                "pending_frame_count": len(
                    self._pending_capture_frames
                ),
                "machine_report_preserved": str(self.report_path),
                "task8": "NOT_RUN",
            },
        )

    def ui_snapshot(self) -> dict[str, Any]:
        return dict(self._last_snapshot)

    def abort_reset_snapshot(self, *, phase: str) -> dict[str, Any]:
        """Read the minimal mutation audit state for Abort/Reset."""

        from pxr import UsdPhysics

        kinematic = bool(
            UsdPhysics.RigidBodyAPI(self.bottle_prim)
            .GetKinematicEnabledAttr()
            .Get()
        )
        return {
            "phase": str(phase),
            "target_write_count": int(self._target_write_count),
            "telemetry_count": len(self.telemetry),
            "joint_target": self.command.tolist(),
            "bottle_kinematic_enabled": kinematic,
            "stage_sha256": sha256_file(self.stage_path),
        }

    def _terminal_metrics(self, phase: Phase) -> dict[str, Any]:
        dynamic_formal = all(
            item.bottle_dynamic
            for item in self.observations
            if item.frame > 2
        )
        hold_samples = [
            item
            for item in self.observations
            if self.telemetry[item.frame - 1]["phase"] == Phase.HOLD.value
        ]
        hold_records = [
            self.telemetry[item.frame - 1]
            for item in hold_samples
        ]
        hold_duration = physics_sample_duration_s(
            sample_count=len(hold_samples),
            physics_dt_s=self.dt,
        )
        return {
            "status": phase.value,
            "aborted": phase is Phase.ABORTED,
            "forbidden_constraint": False,
            "finite_state": all(
                item.finite_state for item in self.observations
            ),
            "persistent_penetration": any(
                item.persistent_penetration
                for item in self.observations
            ),
            "numerical_ejection": any(
                item.numerical_ejection for item in self.observations
            ),
            "dynamic_during_formal_phases": dynamic_formal,
            "bilateral_contact_before_lift": (
                self._bilateral_before_lift
            ),
            "bilateral_contact_through_hold": (
                self._bilateral_through_hold
            ),
            "height_reached": self._height_reached,
            "maximum_clearance_m": self._maximum_clearance_m,
            "hold_duration_s": hold_duration,
            "hold_physics_frame_count": len(hold_samples),
            "hold_bilateral_geometric_frame_count": sum(
                bool(
                    record["contact_semantics"][
                        "bilateral_geometric_contact"
                    ]
                )
                for record in hold_records
            ),
            "hold_bilateral_solver_active_frame_count": sum(
                bool(
                    record["contact_semantics"][
                        "bilateral_solver_active_contact"
                    ]
                )
                for record in hold_records
            ),
            "hold_drop_m": float(
                self._last_snapshot["hold_drop_m"]
            ),
            "ee_vertical_displacement_m": (
                self.observations[-1].ee_vertical_displacement_m
                if self.observations
                else 0.0
            ),
        }

    def finalize_run(self, phase: Phase, reason: str) -> None:
        metrics = self._terminal_metrics(phase)
        signature = canonical_run_signature(
            self.observations,
            metrics,
        )
        report = {
            "schema_version": 1,
            "status": phase.value,
            "reason": reason,
            "classification": (
                "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            ),
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "delegate": self.delegate_readback,
                "solve_articulation_contact_last": bool(
                    self.physics_context
                    .get_solve_articulation_contact_last()
                ),
                "dof_order": list(self.articulation.dof_names),
                "ik": self._ik_report,
                "coupling": self.coupling_readback,
                "finger_drive": self.drive_readback,
                "trajectory": {
                    "additional_lift_margin_m": (
                        self.additional_lift_margin_m
                    ),
                    "classification": (
                        "DIAGNOSTIC_CALCULATED_RUNTIME_CLEARANCE_"
                        "COMPENSATION"
                        if self.additional_lift_margin_m > 0.0
                        else "BASELINE_ZERO_ADDITIONAL_MARGIN"
                    ),
                },
            },
            "stage": {
                "absolute_path": str(self.stage_path),
                "sha256_before": self.stage_hash_before,
                "sha256_after": sha256_file(self.stage_path),
                "root_prim": str(
                    self.stage.GetDefaultPrim().GetPath()
                ),
                "sublayers": list(
                    self.stage.GetRootLayer().subLayerPaths
                ),
                "session_only": True,
            },
            "bottle": self.bottle_session,
            "bottle_random_position": {
                "offset_xy_m": self.bottle_xy_offset_m,
                "changed_variable": (
                    "BOTTLE_INITIAL_WORLD_XY_TRANSLATION_ONLY"
                ),
                "rotation_unchanged": True,
                "object_from_gripper_unchanged": True,
            },
            "table_top_z_m": self.table_top_z_m,
            "target_clearance_m": float(
                self.config["target"]["clearance_m"]
            ),
            "metrics": metrics,
            "deterministic_signature": signature,
            "telemetry_absolute_path": str(self.telemetry_path),
            "runtime_seconds": time.perf_counter() - self.started_at,
            "boundaries": {
                "real_robot": False,
                "remote_103": False,
                "surface_gripper": False,
                "fixed_joint": False,
                "parent_attachment": False,
                "source_stage_modified": False,
                "final_collider_modified": False,
                "task8": "NOT_RUN",
            },
        }
        if report["stage"]["sha256_after"] != self.stage_hash_before:
            report["status"] = "FAIL"
            report["reason"] = "approved_stage_hash_changed"
        _atomic_json(self.report_path, report)
        self.telemetry_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in self.telemetry
            ),
            encoding="utf-8",
        )

    def save_exception(self, exception_text: str) -> None:
        _atomic_json(
            self.report_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "reason": "exception",
                "exception": exception_text[-12000:],
                "stage": {
                    "absolute_path": str(self.stage_path),
                    "sha256_before": self.stage_hash_before,
                    "sha256_after": sha256_file(self.stage_path),
                },
                "boundaries": {
                    "real_robot": False,
                    "remote_103": False,
                    "surface_gripper": False,
                    "fixed_joint": False,
                    "parent_attachment": False,
                    "source_stage_modified": False,
                    "final_collider_modified": False,
                    "task8": "NOT_RUN",
                },
            },
        )

    def reset_session(self) -> None:
        from pxr import UsdPhysics

        self.world.pause()
        rigid = UsdPhysics.RigidBodyAPI(self.bottle_prim)
        kinematic_attr = rigid.GetKinematicEnabledAttr()
        initially_kinematic = bool(kinematic_attr.Get())
        operation_plan = reset_body_transition_plan(
            initially_kinematic=initially_kinematic
        )
        if operation_plan[0] == "set_dynamic":
            kinematic_attr.Set(False)
            self._physx_sim.flush_changes()
        bottle_indices = single_body_tensor_indices(
            count=int(self.bottle.count)
        )
        self.bottle.set_transforms(
            self._initial_bottle_transform.reshape(1, 7),
            bottle_indices,
        )
        self.bottle.set_velocities(
            np.zeros((1, 6), dtype=np.float32),
            bottle_indices,
        )
        kinematic_attr.Set(True)
        self._physx_sim.flush_changes()
        self.command = self.initial_command.copy()
        self.articulation.set_joint_positions(self.command)
        self.articulation.set_joint_velocities(
            np.zeros_like(self.command)
        )
        self._write_joint_command()
        self._contact_buffer = []
        self.all_contacts = []
        self._reset_runtime_records()

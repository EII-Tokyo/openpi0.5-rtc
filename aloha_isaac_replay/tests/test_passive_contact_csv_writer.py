from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_calibration_config
from aloha_isaac_replay.scripts.create_table_to_base_calibration import build_evidence_record
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _audit_required_table_frame
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _active_target_contact_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _guard_final_contact_stage_namespace
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _guard_support_plane_calibration_mode
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _finger_targets
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_grasp_transform
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _passive_contact_geometry_sanity
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _parse_vec3
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _apply_replay_target_and_step
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _controller_tracking_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _load_support_plane_config
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _non_target_contact_gate
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _resolve_support_plane_options
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _resolve_side_arm_dof_name
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _should_disable_workcell_environment_collision
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_contact_pairs
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_target_limit_violations
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _summarize_tracking_errors
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _target_from_standard_qpos
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _target_limit_step_violations
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _targets_from_hdf5_qpos
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _tracking_groups
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _tracking_step_errors
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import _write_csv
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_dof_names_for_side
from aloha_isaac_replay.validation.contact_proxy_profiles import finger_qpos_limits_for_side

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeWorld:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self, *, render: bool) -> None:
        assert render is False
        self.step_calls += 1


class _FakeArticulation:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.25, -0.5], dtype=np.float64)

    def get_joint_positions(self) -> np.ndarray:
        return self.qpos.copy()


class _FakeSceneBaseLeftArticulation:
    dof_names = ["left_waist", "left_left_finger", "left_right_finger"]

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)


class _FakeSceneBaseLeftArmArticulation:
    dof_names = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_left_finger",
        "left_right_finger",
    ]

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(len(self.dof_names), dtype=np.float64)


def test_finger_targets_can_use_same_sign_right_close(monkeypatch) -> None:
    art = _FakeSceneBaseLeftArticulation()
    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._get_limits",
        lambda _art: np.asarray([[-1.0, 1.0], [0.0, 0.08], [0.0, 0.08]], dtype=np.float64),
    )
    finger_dofs = {"left_finger": "left_left_finger", "right_finger": "left_right_finger"}

    legacy_target, legacy_values = _finger_targets(art, -0.014, 0.001, finger_dofs)
    spatial_target, spatial_values = _finger_targets(
        art, -0.014, 0.001, finger_dofs, right_finger_sign=1.0
    )

    assert legacy_values["left_finger"] == spatial_values["left_finger"]
    assert legacy_values["right_finger"] > 0.04
    assert spatial_values["right_finger"] < 0.04
    assert legacy_target[2] != spatial_target[2]


def test_load_grasp_transform_reads_scalar_first_quaternion(tmp_path: Path) -> None:
    path = tmp_path / "grasps.yaml"
    path.write_text(
        "\n".join(
            [
                "format: isaac_grasp",
                "object_frame: /World/Bottle500",
                "gripper_frame: /scene/left_base_link/left_gripper_link",
                "grasps:",
                "  grasp_mid:",
                "    position: [0.0, 0.052, 0.105]",
                "    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}",
            ]
        )
        + "\n"
    )

    info = _load_grasp_transform(path, "grasp_mid")

    assert info["object_frame"] == "/World/Bottle500"
    assert info["gripper_frame"] == "/scene/left_base_link/left_gripper_link"
    np.testing.assert_allclose(info["t_object_gripper"][:3, 3], [0.0, 0.052, 0.105])
    np.testing.assert_allclose(info["t_object_gripper"][:3, :3], np.eye(3))


def test_parse_vec3_accepts_center_offset_and_rejects_bad_values() -> None:
    np.testing.assert_allclose(_parse_vec3([0.1, -0.2, 0.0], name="offset"), [0.1, -0.2, 0.0])
    np.testing.assert_allclose(_parse_vec3(None, name="offset"), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exactly three"):
        _parse_vec3([1.0, 2.0], name="offset")
    with pytest.raises(ValueError, match="NaN/Inf"):
        _parse_vec3([1.0, float("nan"), 0.0], name="offset")


def test_passive_contact_csv_writer_preserves_late_diagnostic_columns(tmp_path) -> None:
    path = tmp_path / "contact.csv"
    _write_csv(
        path,
        [
            {"phase": "settle", "step": 0, "object_center_x": 0.0},
            {"phase": "close", "step": 0, "tracking_controlled_max_abs_error": 0.12},
        ],
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert "tracking_controlled_max_abs_error" in rows[0]
    assert rows[1]["tracking_controlled_max_abs_error"] == "0.12"


def test_replay_actuation_mode_drive_target_steps_with_joint_target(monkeypatch) -> None:
    calls: list[str] = []

    def fake_set_target(art, target) -> None:
        calls.append("target")

    def fake_set_state(art, target) -> None:
        calls.append("state")

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_target",
        fake_set_target,
    )
    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_state",
        fake_set_state,
    )

    world = _FakeWorld()
    pre_step_qpos = _apply_replay_target_and_step(world, _FakeArticulation(), np.zeros(2), actuation_mode="drive_target")

    assert calls == ["target"]
    np.testing.assert_allclose(pre_step_qpos, [0.25, -0.5])
    assert world.step_calls == 1


def test_replay_actuation_mode_drive_target_can_hold_target(monkeypatch) -> None:
    calls: list[str] = []

    def fake_set_target(art, target) -> None:
        calls.append("target")

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_target",
        fake_set_target,
    )

    world = _FakeWorld()
    pre_step_qpos = _apply_replay_target_and_step(
        world,
        _FakeArticulation(),
        np.zeros(2),
        actuation_mode="drive_target",
        target_hold_steps=3,
    )

    assert calls == ["target", "target", "target"]
    np.testing.assert_allclose(pre_step_qpos, [0.25, -0.5])
    assert world.step_calls == 3


def test_replay_actuation_mode_state_teleport_sets_state_then_target(monkeypatch) -> None:
    calls: list[str] = []

    def fake_set_target(art, target) -> None:
        calls.append("target")

    def fake_set_state(art, target) -> None:
        calls.append("state")
        art.qpos = np.asarray(target, dtype=np.float64)

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_target",
        fake_set_target,
    )
    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_state",
        fake_set_state,
    )

    world = _FakeWorld()
    pre_step_qpos = _apply_replay_target_and_step(
        world,
        _FakeArticulation(),
        np.asarray([1.0, 2.0], dtype=np.float64),
        actuation_mode="state_teleport",
    )

    assert calls == ["state", "target"]
    np.testing.assert_allclose(pre_step_qpos, [1.0, 2.0])
    assert world.step_calls == 1


def test_replay_actuation_mode_state_teleport_reapplies_state_during_hold(monkeypatch) -> None:
    calls: list[str] = []

    def fake_set_target(art, target) -> None:
        calls.append("target")

    def fake_set_state(art, target) -> None:
        calls.append("state")
        art.qpos = np.asarray(target, dtype=np.float64)

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_target",
        fake_set_target,
    )
    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact._set_full_state",
        fake_set_state,
    )

    world = _FakeWorld()
    pre_step_qpos = _apply_replay_target_and_step(
        world,
        _FakeArticulation(),
        np.asarray([1.0, 2.0], dtype=np.float64),
        actuation_mode="state_teleport",
        target_hold_steps=2,
    )

    assert calls == ["state", "target", "state", "target"]
    np.testing.assert_allclose(pre_step_qpos, [1.0, 2.0])
    assert world.step_calls == 2


def test_replay_actuation_mode_rejects_non_positive_hold_steps() -> None:
    with pytest.raises(ValueError, match="target_hold_steps must be positive"):
        _apply_replay_target_and_step(
            _FakeWorld(),
            _FakeArticulation(),
            np.zeros(2),
            actuation_mode="drive_target",
            target_hold_steps=0,
        )


def test_tracking_groups_accept_scene_base_link_prefixed_left_arm_dofs() -> None:
    groups = _tracking_groups(
        [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_left_finger",
            "left_right_finger",
        ],
        replay_mode="left_arm_and_gripper",
        finger_dof_names={"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
        side="left",
    )

    assert groups["left_arm"] == [0, 1, 2, 3, 4, 5]
    assert groups["controlled"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_tracking_groups_include_arm_for_hdf5_arm_start_then_gripper_only() -> None:
    groups = _tracking_groups(
        _FakeSceneBaseLeftArmArticulation.dof_names,
        replay_mode="hdf5_arm_start_then_gripper_only",
        finger_dof_names={"left_finger": "left_left_finger", "right_finger": "left_right_finger"},
        side="left",
    )

    assert groups["left_arm"] == [0, 1, 2, 3, 4, 5]
    assert groups["gripper"] == [6, 7]
    assert groups["controlled"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_tracking_summary_records_max_error_dof_name_and_step() -> None:
    groups = {"left_arm": [0, 1, 2]}
    dof_names = ["left_waist", "left_shoulder", "left_elbow"]
    rows = [
        {
            "phase": "settle",
            "step": 0,
            "groups": _tracking_step_errors(
                target=np.asarray([0.0, 0.0, 0.0]),
                actual=np.asarray([0.1, -0.2, 0.05]),
                groups=groups,
            ),
        },
        {
            "phase": "close",
            "step": 3,
            "groups": _tracking_step_errors(
                target=np.asarray([0.0, 0.0, 0.0]),
                actual=np.asarray([0.1, 0.05, -0.7]),
                groups=groups,
            ),
        },
    ]

    summary = _summarize_tracking_errors(rows, groups, dof_names)

    assert summary["groups"]["left_arm"]["max_abs_error"] == 0.7
    assert summary["groups"]["left_arm"]["max_abs_error_dof_name"] == "left_elbow"
    assert summary["groups"]["left_arm"]["max_abs_error_signed"] == -0.7
    assert summary["groups"]["left_arm"]["max_abs_error_phase"] == "close"
    assert summary["groups"]["left_arm"]["max_abs_error_step"] == 3


def test_controller_tracking_gate_fails_when_post_step_error_exceeds_threshold() -> None:
    tracking_summary = {
        "groups": {
            "controlled": {
                "max_abs_error": 1.49,
                "max_abs_error_dof_name": "left_wrist_rotate",
                "max_abs_error_phase": "close",
                "max_abs_error_step": 62,
            }
        }
    }

    gate = _controller_tracking_gate(tracking_summary=tracking_summary, max_controlled_error=0.02)

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_POST_STEP_TRACKING_EXCEEDS_THRESHOLD"
    assert gate["max_controlled_error"] == pytest.approx(1.49)
    assert gate["max_controlled_error_dof_name"] == "left_wrist_rotate"


def test_controller_tracking_gate_passes_when_post_step_error_is_small() -> None:
    tracking_summary = {"groups": {"controlled": {"max_abs_error": 0.00043}}}

    gate = _controller_tracking_gate(tracking_summary=tracking_summary, max_controlled_error=0.02)

    assert gate["pass"] is True
    assert gate["status"] == "PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD"


def test_target_limit_summary_records_clamped_dof_name_and_step() -> None:
    groups = {"left_arm": [0, 1, 2], "controlled": [0, 1, 2]}
    dof_names = ["left_waist", "left_shoulder", "left_elbow"]
    limits = np.asarray([[-1.0, 1.0], [0.0, 1.25], [-2.0, 2.0]], dtype=np.float64)
    rows = [
        {
            "phase": "settle",
            "step": 0,
            "groups": _target_limit_step_violations(
                target=np.asarray([0.0, 1.0, 0.0]),
                limits=limits,
                groups=groups,
            ),
        },
        {
            "phase": "close",
            "step": 4,
            "groups": _target_limit_step_violations(
                target=np.asarray([0.0, 2.13, 0.0]),
                limits=limits,
                groups=groups,
            ),
        },
    ]

    summary = _summarize_target_limit_violations(rows, groups, dof_names)

    assert summary["status"] == "FAIL_TARGET_OUTSIDE_RUNTIME_LIMITS"
    assert summary["controller_ready"] is False
    assert summary["groups"]["left_arm"]["max_violation"] == pytest.approx(0.88)
    assert summary["groups"]["left_arm"]["max_violation_dof_name"] == "left_shoulder"
    assert summary["groups"]["left_arm"]["max_violation_signed"] == pytest.approx(0.88)
    assert summary["groups"]["left_arm"]["max_violation_phase"] == "close"
    assert summary["groups"]["left_arm"]["max_violation_step"] == 4
    assert summary["groups"]["left_arm"]["target_at_max_violation"] == pytest.approx(2.13)
    assert summary["groups"]["left_arm"]["upper_at_max_violation"] == pytest.approx(1.25)


def test_scene_base_link_hdf5_gripper_qpos_maps_both_fingers_positive() -> None:
    qpos = np.zeros(14, dtype=np.float64)
    qpos[6] = 0.5

    target = _target_from_standard_qpos(
        art=_FakeSceneBaseLeftArticulation(),
        side="left",
        qpos_frame=qpos,
        mapping=None,
        replay_mode="gripper_only",
        finger_dof_names=finger_dof_names_for_side("scene_base_link", "left"),
        finger_qpos_limits=finger_qpos_limits_for_side("scene_base_link", "left"),
    )

    assert target[1] == pytest.approx(0.039)
    assert target[2] == pytest.approx(0.039)
    assert target[2] > 0.0


def test_hdf5_arm_start_then_gripper_only_holds_start_arm_and_replays_gripper(monkeypatch) -> None:
    class _ArmTarget:
        def __init__(self, name: str, value: float) -> None:
            self.isaac_dof_name = name
            self.value = value

    def fake_arm_targets(frame: np.ndarray, _mapping: dict[str, object], *, side: str) -> list[_ArmTarget]:
        assert side == "left"
        return [
            _ArmTarget("left/waist", float(frame[0])),
            _ArmTarget("left/shoulder", float(frame[1])),
            _ArmTarget("left/elbow", float(frame[2])),
            _ArmTarget("left/forearm_roll", float(frame[3])),
            _ArmTarget("left/wrist_angle", float(frame[4])),
            _ArmTarget("left/wrist_rotate", float(frame[5])),
        ]

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact.arm_only_targets_from_standard_qpos",
        fake_arm_targets,
    )
    qpos = np.zeros((2, 14), dtype=np.float64)
    qpos[0, :6] = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    qpos[1, :6] = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    qpos[0, 6] = 1.0
    qpos[1, 6] = 0.0

    targets, summary = _targets_from_hdf5_qpos(
        art=_FakeSceneBaseLeftArmArticulation(),
        side="left",
        qpos=qpos,
        mapping={"dummy": True},
        replay_mode="hdf5_arm_start_then_gripper_only",
        finger_dof_names=finger_dof_names_for_side("scene_base_link", "left"),
        finger_qpos_limits=finger_qpos_limits_for_side("scene_base_link", "left"),
    )

    np.testing.assert_allclose(targets[0][:6], qpos[0, :6])
    np.testing.assert_allclose(targets[1][:6], qpos[0, :6])
    assert targets[0][6] > targets[1][6]
    assert targets[0][7] > targets[1][7]
    assert summary["formal_full_hdf5_replay"] is False
    assert summary["arm_initialized_from_hdf5"] is True
    assert summary["hdf5_arm_targets_after_start_used"] is False
    assert summary["arm_target_behavior"] == "constant_hdf5_start_frame_hold"
    assert summary["arm_qpos_delta"]["max_abs_net_delta"] == pytest.approx(8.89)


def test_hdf5_arm_start_then_gripper_only_can_use_action_gripper_source(monkeypatch) -> None:
    class _ArmTarget:
        def __init__(self, name: str, value: float) -> None:
            self.isaac_dof_name = name
            self.value = value

    def fake_arm_targets(frame: np.ndarray, _mapping: dict[str, object], *, side: str) -> list[_ArmTarget]:
        assert side == "left"
        return [
            _ArmTarget("left/waist", float(frame[0])),
            _ArmTarget("left/shoulder", float(frame[1])),
            _ArmTarget("left/elbow", float(frame[2])),
            _ArmTarget("left/forearm_roll", float(frame[3])),
            _ArmTarget("left/wrist_angle", float(frame[4])),
            _ArmTarget("left/wrist_rotate", float(frame[5])),
        ]

    monkeypatch.setattr(
        "aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact.arm_only_targets_from_standard_qpos",
        fake_arm_targets,
    )
    qpos = np.zeros((2, 14), dtype=np.float64)
    qpos[0, :6] = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    qpos[1, :6] = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    qpos[:, 6] = 1.0
    action = qpos.copy()
    action[0, 6] = 1.0
    action[1, 6] = 0.0

    targets, summary = _targets_from_hdf5_qpos(
        art=_FakeSceneBaseLeftArmArticulation(),
        side="left",
        qpos=qpos,
        gripper_sequence=action,
        gripper_source="action",
        mapping={"dummy": True},
        replay_mode="hdf5_arm_start_then_gripper_only",
        finger_dof_names=finger_dof_names_for_side("scene_base_link", "left"),
        finger_qpos_limits=finger_qpos_limits_for_side("scene_base_link", "left"),
    )

    np.testing.assert_allclose(targets[0][:6], qpos[0, :6])
    np.testing.assert_allclose(targets[1][:6], qpos[0, :6])
    assert targets[0][6] > targets[1][6]
    assert targets[0][7] > targets[1][7]
    assert summary["source"] == "action"
    assert summary["arm_source"] == "observations/qpos"
    assert summary["raw_start"] == pytest.approx(1.0)
    assert summary["raw_end"] == pytest.approx(0.0)


def test_resolve_side_arm_dof_name_rejects_missing_precisely() -> None:
    with pytest.raises(ValueError, match="Could not resolve mapped DOF"):
        _resolve_side_arm_dof_name(
            "shoulder",
            dof_names=["left_waist", "left_elbow"],
            side="left",
            source_name="left/shoulder",
        )


def test_workcell_environment_collision_filter_is_diagnostic_and_does_not_match_robot_or_target_paths() -> None:
    assert _should_disable_workcell_environment_collision("/scene/worldBody/table/collisions/table/table/table")
    assert _should_disable_workcell_environment_collision("/scene/worldBody/__27/collisions/__27/__27/extrusion_1000")
    assert _should_disable_workcell_environment_collision("/World/Table/collisions/table")

    assert not _should_disable_workcell_environment_collision("/scene/left_base_link/left_wrist_link/collisions/wrist")
    assert not _should_disable_workcell_environment_collision(
        "/scene/left_base_link/left_left_finger_link/bbox_collision_proxy"
    )
    assert not _should_disable_workcell_environment_collision(
        "/World/phase43_passive_contact_cube/Collisions/COL_Body_14/COL_Body_14Mesh"
    )
    assert not _should_disable_workcell_environment_collision("/World/phase58_static_support_plane/Collision")


def test_passive_contact_geometry_sanity_rejects_implausible_open_gap() -> None:
    sanity = _passive_contact_geometry_sanity(
        finger_surface_gap_stage_units=0.745959177295105,
        object_side_length_stage_units=0.44757550637706295,
        stage_units_in_meters=1.0,
        max_finger_surface_gap_meters=0.12,
        max_generated_object_side_meters=0.08,
    )

    assert sanity["status"] == "FAIL_IMPLAUSIBLE_FINGER_GAP"
    assert sanity["pass"] is False
    assert sanity["finger_surface_gap_open_meters"] == pytest.approx(0.745959177295105)


def test_passive_contact_geometry_sanity_rejects_implausible_object_size() -> None:
    sanity = _passive_contact_geometry_sanity(
        finger_surface_gap_stage_units=0.09,
        object_side_length_stage_units=0.081,
        stage_units_in_meters=1.0,
        max_finger_surface_gap_meters=0.12,
        max_generated_object_side_meters=0.08,
    )

    assert sanity["status"] == "FAIL_IMPLAUSIBLE_OBJECT_SIZE"
    assert sanity["pass"] is False
    assert sanity["object_side_length_meters"] == pytest.approx(0.081)


def test_contact_summary_classifies_diagnostic_support_contacts() -> None:
    object_path = "/World/object"
    finger_path = "/World/left_finger"
    support_path = "/World/support"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "close",
                "step": 1,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": support_path,
                "sorted_pair": [f"{object_path}/body", support_path],
            },
            {
                "phase": "close",
                "step": 2,
                "type_name": "CONTACT_FOUND",
                "collider0": support_path,
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [support_path, f"{finger_path}/proxy"],
            },
            {
                "phase": "close",
                "step": 3,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{object_path}/body",
                "collider1": f"{finger_path}/proxy",
                "sorted_pair": [f"{object_path}/body", f"{finger_path}/proxy"],
            },
        ],
        object_path=object_path,
        expected_finger_paths=[finger_path],
        diagnostic_contact_paths=[support_path],
    )

    support_summary = summary["diagnostic_contact_summaries"][support_path]
    assert summary["target_contact_pair_found"] is True
    assert support_summary["contact_pair_count"] == 2
    assert support_summary["object_contact_pair_count"] == 1
    assert support_summary["expected_finger_contact_pair_count"] == 1
    assert support_summary["other_contact_pair_count"] == 0


def test_contact_summary_accepts_collision_descendants_under_finger_link() -> None:
    object_path = "/World/object"
    finger_link = "/scene/left_base_link/left_left_finger_link"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{finger_link}/collisions/left_left_g0/left_left_g0",
                "collider1": object_path,
                "sorted_pair": [object_path, f"{finger_link}/collisions/left_left_g0/left_left_g0"],
            }
        ],
        object_path=object_path,
        expected_finger_paths=[finger_link],
    )

    assert summary["target_contact_pair_found"] is True
    assert summary["target_contact_found_event"] is True
    assert summary["target_contact_finger_hits"][finger_link] is True
    assert summary["first_target_contact_found_phase"] == "settle"


def test_active_target_contact_gate_requires_close_contact_found_event() -> None:
    gate = _active_target_contact_gate(
        contact_summary={
            "target_contact_found_phases": ["settle"],
            "first_target_contact_phase": "settle",
            "first_target_contact_found_phase": "settle",
        },
        require_active_target_contact=True,
        already_in_contact_setup=False,
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE"


def test_active_target_contact_gate_rejects_settle_first_even_if_close_later() -> None:
    gate = _active_target_contact_gate(
        contact_summary={
            "target_contact_found_phases": ["close", "settle"],
            "first_target_contact_phase": "settle",
            "first_target_contact_found_phase": "settle",
        },
        require_active_target_contact=True,
        already_in_contact_setup=False,
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE"


def test_active_target_contact_gate_passes_close_contact_found_event() -> None:
    gate = _active_target_contact_gate(
        contact_summary={
            "target_contact_found_phases": ["close"],
            "first_target_contact_phase": "close",
            "first_target_contact_found_phase": "close",
        },
        require_active_target_contact=True,
        already_in_contact_setup=False,
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE"


def test_active_target_contact_gate_documents_already_contacting_setup() -> None:
    gate = _active_target_contact_gate(
        contact_summary={
            "target_contact_found_phases": ["settle"],
            "first_target_contact_phase": "settle",
            "first_target_contact_found_phase": "settle",
        },
        require_active_target_contact=False,
        already_in_contact_setup=True,
    )

    assert gate["pass"] is True
    assert gate["status"] == "SKIPPED_ALREADY_IN_CONTACT_SETUP"


def test_contact_summary_classifies_non_target_object_contacts() -> None:
    object_path = "/World/object"
    finger_link = "/scene/left_base_link/left_left_finger_link"
    support_path = "/World/support_plane"

    summary = _summarize_contact_pairs(
        contact_pair_rows=[
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": f"{finger_link}/collisions/left_left_g0/left_left_g0",
                "collider1": object_path,
                "sorted_pair": [object_path, f"{finger_link}/collisions/left_left_g0/left_left_g0"],
            },
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": "/scene/left_base_link/left_wrist_link/collisions/wrist",
                "collider1": object_path,
                "sorted_pair": [object_path, "/scene/left_base_link/left_wrist_link/collisions/wrist"],
            },
            {
                "phase": "settle",
                "step": 0,
                "type_name": "CONTACT_FOUND",
                "collider0": support_path,
                "collider1": object_path,
                "sorted_pair": [object_path, support_path],
            },
        ],
        object_path=object_path,
        expected_finger_paths=[finger_link],
        diagnostic_contact_paths=[support_path],
        same_side_robot_root="/scene/left_base_link",
        other_side_robot_root="/scene/right_base_link",
    )

    assert summary["object_contact_categories"]["target_finger"]["contact_pair_count"] == 1
    assert summary["object_contact_categories"]["same_side_robot_non_target"]["contact_pair_count"] == 1
    assert summary["object_contact_categories"]["diagnostic_support"]["contact_pair_count"] == 1
    assert summary["non_target_object_contact_found"] is True
    assert summary["non_target_object_contact_pair_count"] == 2


def test_non_target_contact_gate_allows_declared_workcell_support_category() -> None:
    summary = {"non_target_object_contact_categories": ["workcell_or_environment"]}

    gate = _non_target_contact_gate(
        contact_summary=summary,
        fail_on_non_target=True,
        allowed_categories=["workcell_or_environment"],
    )

    assert gate["pass"] is True
    assert gate["status"] == "PASS_NON_TARGET_CONTACTS_ALLOWED"
    assert gate["blocking_categories"] == []


def test_non_target_contact_gate_rejects_undeclared_robot_body_category() -> None:
    summary = {
        "non_target_object_contact_categories": [
            "same_side_robot_non_target",
            "workcell_or_environment",
        ]
    }

    gate = _non_target_contact_gate(
        contact_summary=summary,
        fail_on_non_target=True,
        allowed_categories=["workcell_or_environment"],
    )

    assert gate["pass"] is False
    assert gate["status"] == "FAIL_NON_TARGET_OBJECT_CONTACT"
    assert gate["blocking_categories"] == ["same_side_robot_non_target"]


def test_phase63_fixed_table_candidate_config_is_explicit_and_diagnostic() -> None:
    cfg = _load_support_plane_config(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml")

    assert cfg["mode"] == "fixed_box"
    assert cfg["center"] == [0.593227851197621, 0.7853100288947757, -0.3171450733686908]
    assert cfg["size"] == [1.22, 0.625, 0.04]
    assert cfg["provenance"]["table_size"]["source"] == "user_measured"
    assert cfg["provenance"]["center_xy"]["source"] == "phase60_diagnostic_object_bottom"
    assert cfg["table_frame"]["T_table_left_base"]["status"] == "not_calibrated"
    assert cfg["table_frame"]["T_table_right_base"]["status"] == "not_calibrated"


def test_support_plane_config_resolves_fixed_box_options() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        support_plane_mode="none",
        support_plane_center=None,
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
        support_plane_patch_margin=0.04,
    )

    resolved = _resolve_support_plane_options(args)

    assert resolved["mode"] == "fixed_box"
    assert resolved["center"] == [0.593227851197621, 0.7853100288947757, -0.3171450733686908]
    assert resolved["size_x"] == 1.22
    assert resolved["size_y"] == 0.625
    assert resolved["thickness"] == 0.04
    assert resolved["config"] == "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"
    assert resolved["table_frame"]["T_world_table"]["status"] == "diagnostic_candidate"


def test_support_plane_config_rejects_object_bottom_mix() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        support_plane_mode="object_bottom",
        support_plane_center=None,
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
        support_plane_patch_margin=0.04,
    )

    try:
        _resolve_support_plane_options(args)
    except ValueError as exc:
        assert "object_bottom" in str(exc)
    else:
        raise AssertionError("expected object_bottom/config combination to be rejected")


def test_require_calibrated_table_frame_requires_config() -> None:
    import argparse

    args = argparse.Namespace(require_calibrated_table_frame=True, support_plane_config=None)

    with pytest.raises(ValueError, match="requires --support-plane-config"):
        _audit_required_table_frame(args)


def test_support_plane_config_requires_calibrated_gate_or_explicit_diagnostic_opt_in() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        require_calibrated_table_frame=False,
        allow_diagnostic_support_plane_config=False,
    )

    with pytest.raises(ValueError, match="--allow-diagnostic-support-plane-config"):
        _guard_support_plane_calibration_mode(args)


def test_support_plane_config_allows_explicit_diagnostic_opt_in() -> None:
    import argparse

    args = argparse.Namespace(
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        require_calibrated_table_frame=False,
        allow_diagnostic_support_plane_config=True,
    )

    _guard_support_plane_calibration_mode(args)


def test_require_calibrated_table_frame_rejects_diagnostic_config() -> None:
    import argparse

    args = argparse.Namespace(
        require_calibrated_table_frame=True,
        support_plane_config=str(REPO_ROOT / "examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml"),
        stage_units_in_meters=1.0,
    )

    with pytest.raises(ValueError, match="BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM"):
        _audit_required_table_frame(args)


def test_require_calibrated_table_frame_accepts_measured_config(tmp_path: Path) -> None:
    import argparse

    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "measured.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = argparse.Namespace(
        require_calibrated_table_frame=True, support_plane_config=str(path), stage_units_in_meters=1.0
    )

    audit = _audit_required_table_frame(args)

    assert audit is not None
    assert audit["status"] == "PASS_TABLE_TO_BASE_CALIBRATION_READY"


def test_require_calibrated_table_frame_rejects_support_plane_cli_overrides(tmp_path: Path) -> None:
    import argparse

    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "measured.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = argparse.Namespace(
        require_calibrated_table_frame=True,
        support_plane_config=str(path),
        stage_units_in_meters=1.0,
        support_plane_center=[0.593227851197621, 0.7853100288947757, -0.3171450733686908],
        support_plane_size=2.0,
        support_plane_size_x=None,
        support_plane_size_y=None,
        support_plane_thickness=0.02,
    )

    with pytest.raises(ValueError, match="cannot combine --support-plane-config with support-plane CLI overrides"):
        _audit_required_table_frame(args)


def test_require_calibrated_table_frame_rejects_legacy_centimeter_world_units(tmp_path: Path) -> None:
    import argparse

    evidence_path = tmp_path / "measurement_evidence.yaml"
    evidence_path.write_text("measurement: synthetic\n")
    cfg = build_calibration_config(
        table_top_center=[1.0, 2.0, 0.5],
        table_size=[1.22, 0.625, 0.04],
        table_yaw_deg=0.0,
        left_base_in_table=[-0.3, 0.1, 0.0],
        left_yaw_deg=0.0,
        right_base_in_table=[0.3, 0.1, 0.0],
        right_yaw_deg=180.0,
        source="user_measured",
        status="measured",
        calibration_evidence=build_evidence_record(
            evidence_path,
            evidence_type="unit_test",
            real_robot_touched=False,
            remote_103_touched=False,
        ),
    )
    path = tmp_path / "measured.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = argparse.Namespace(
        require_calibrated_table_frame=True, support_plane_config=str(path), stage_units_in_meters=0.01
    )

    with pytest.raises(ValueError, match="requires --stage-units-in-meters 1.0"):
        _audit_required_table_frame(args)


def test_final_contact_validation_rejects_scene_overlay_with_legacy_puppet_proxy_paths(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "scene_overlay.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "scene"',
                "{",
                '    over "left_base_link" {}',
                '    over "right_base_link" {}',
                "}",
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="legacy_puppet"
    )

    with pytest.raises(ValueError, match="contact validator uses legacy /puppet_"):
        _guard_final_contact_stage_namespace(args)


def test_final_contact_validation_allows_scene_overlay_with_scene_proxy_profile(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "scene_overlay.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "scene"',
                "{",
                '    over "left_base_link" {}',
                '    over "right_base_link" {}',
                '    def Cube "bbox_collision_proxy" {}',
                "}",
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="scene_base_link"
    )

    summary = _guard_final_contact_stage_namespace(args)

    assert summary["stage_namespace_hints"]["uses_scene_namespace"] is True
    assert summary["finger_proxy_namespace_roots"] == ["scene"]


def test_final_contact_validation_allows_legacy_puppet_runtime_stage_namespace(tmp_path: Path) -> None:
    import argparse

    stage = tmp_path / "legacy_puppet_runtime.usda"
    stage.write_text(
        "\n".join(
            [
                "#usda 1.0",
                'over "puppet_left_vx300s" {}',
                'over "puppet_right_vx300s" {}',
                'def Cube "bbox_collision_proxy" {}',
            ]
        )
    )
    args = argparse.Namespace(
        require_calibrated_table_frame=True, stage_usd=str(stage), contact_proxy_profile="legacy_puppet"
    )

    summary = _guard_final_contact_stage_namespace(args)

    assert summary["stage_namespace_hints"]["uses_legacy_puppet_namespace"] is True

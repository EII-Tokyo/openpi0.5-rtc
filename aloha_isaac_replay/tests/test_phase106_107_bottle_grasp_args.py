from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aloha_isaac_replay.scripts.run_phase106_bottleusd_already_grasped_gate import _phase106_args
from aloha_isaac_replay.scripts.run_phase107_bottleusd_hdf5_drive_target_gate import _phase107_args
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import (
    _active_grasp_geometry_precondition,
    _closing_axis_gap_centering_solver,
    _contact_projection_model_for_args,
    _derived_tabletop_top_z_from_open_finger,
    _loaded_gripper_soft_bottle_calibration_diagnostic,
    _nominal_object_axis_length_stage_units,
    _object_width_stop_target,
    _oriented_cylinder_projection_model,
    _open_finger_object_height_alignment,
    _tabletop_reference_contract,
    _tabletop_z_shift_from_bboxes,
    _target_contact_hits_for_phase,
)
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import main as validate_contact_main


ROT45_Z = [
    [0.70710678, -0.70710678, 0.0, 0.0],
    [0.70710678, 0.70710678, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_phase106_keeps_legacy_gap_center_smoke_placement() -> None:
    args = _phase106_args(Path("out"))

    assert args[args.index("--object-placement") + 1] == "gap_center"
    offset_index = args.index("--object-center-offset")
    assert args[offset_index + 1 : offset_index + 4] == ["0.08", "0.0", "0.0"]


def test_phase107_uses_runtime_finger_rear_quarter_placement() -> None:
    args = _phase107_args(Path("out"))

    assert args[args.index("--object-placement") + 1] == "finger_rear_quarter"
    assert args[args.index("--object-grasp-name") + 1] == "grasp_rear_quarter"
    offset_index = args.index("--object-center-offset")
    assert args[offset_index + 1 : offset_index + 4] == ["0.0", "0.0", "0.0"]
    assert args[args.index("--object-rear-quarter-fraction") + 1] == "0.25"
    assert "--save-debug-stage" not in args


def test_phase107_can_opt_in_to_debug_stage_export() -> None:
    args = _phase107_args(Path("out"), save_debug_stage=True)

    assert "--save-debug-stage" in args


def test_validator_accepts_hdf5_close_finger_rear_quarter_mode(monkeypatch, tmp_path) -> None:
    seen: dict[str, list[str]] = {}

    class ParserSpy:
        def __init__(self, *args, **kwargs):
            self.choices = None

        def add_argument(self, *args, **kwargs):
            if args == ("--object-placement",):
                self.choices = tuple(kwargs["choices"])
                seen["choices"] = list(self.choices)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", ParserSpy)
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert "hdf5_close_finger_rear_quarter" in seen["choices"]
    assert "hdf5_close_finger_rear_quarter_tabletop" in seen["choices"]
    assert "hdf5_open_finger_rear_quarter" in seen["choices"]
    assert "hdf5_open_finger_rear_quarter_tabletop" in seen["choices"]


def test_tabletop_reference_contract_passes_calibrated_table_and_finger_height() -> None:
    tabletop_adjustment = {
        "pass": True,
        "table_path": "/World/Table",
        "table_top_z_m": 0.0,
        "tabletop_clearance_m": 0.001,
        "tabletop_gap_after_m": 0.001,
        "object_bottom_z_after_m": 0.001,
        "table_bbox": {"bbox_valid": True, "max": [0.5, 0.5, 0.0]},
    }
    collision_audit = {"enabled_collision_prim_count": 1, "status": "PASS_TABLETOP_REFERENCE_HAS_ENABLED_COLLIDER"}
    open_left_box = {"bbox_valid": True, "center": [0.0, -0.03, 0.04]}
    open_right_box = {"bbox_valid": True, "center": [0.0, 0.03, 0.04]}
    object_box = {"bbox_valid": True, "center": [0.0, 0.0, 0.035]}

    row = _tabletop_reference_contract(
        required=True,
        tabletop_adjustment=tabletop_adjustment,
        table_collision_audit=collision_audit,
        open_left_box=open_left_box,
        open_right_box=open_right_box,
        object_box=object_box,
        max_finger_object_center_height_error=0.02,
    )

    assert row["pass"] is True
    assert row["status"] == "PASS_CALIBRATED_TABLETOP_REFERENCE"


def test_derived_tabletop_top_z_aligns_bottle_center_to_open_finger_midpoint() -> None:
    open_left_box = {"bbox_valid": True, "center": [0.02, -0.21, 0.101646]}
    open_right_box = {"bbox_valid": True, "center": [0.01, -0.11, 0.065646]}

    row = _derived_tabletop_top_z_from_open_finger(
        open_left_box=open_left_box,
        open_right_box=open_right_box,
        object_contact_radius=0.026,
        clearance=0.001,
    )

    assert row["pass"] is True
    assert row["status"] == "PASS_DERIVED_TABLETOP_TOP_Z_FROM_OPEN_FINGER"
    assert row["open_finger_contact_midpoint_z_m"] == pytest.approx(0.083646)
    assert row["derived_table_top_z_m"] == pytest.approx(0.056646)


def test_tabletop_reference_contract_fails_table_robot_frame_mismatch() -> None:
    tabletop_adjustment = {
        "pass": True,
        "table_path": "/World/Table",
        "table_top_z_m": 0.0,
        "tabletop_clearance_m": 0.001,
        "tabletop_gap_after_m": 0.001,
        "object_bottom_z_after_m": 0.001,
        "table_bbox": {"bbox_valid": True, "max": [0.5, 0.5, 0.0]},
    }
    collision_audit = {"enabled_collision_prim_count": 1, "status": "PASS_TABLETOP_REFERENCE_HAS_ENABLED_COLLIDER"}
    open_left_box = {"bbox_valid": True, "center": [0.0, -0.03, 0.16]}
    open_right_box = {"bbox_valid": True, "center": [0.0, 0.03, 0.16]}
    object_box = {"bbox_valid": True, "center": [0.0, 0.0, 0.035]}

    row = _tabletop_reference_contract(
        required=True,
        tabletop_adjustment=tabletop_adjustment,
        table_collision_audit=collision_audit,
        open_left_box=open_left_box,
        open_right_box=open_right_box,
        object_box=object_box,
        max_finger_object_center_height_error=0.04,
    )

    assert row["pass"] is False
    assert row["status"] == "FAIL_TABLE_ROBOT_FRAME_MISMATCH"
    assert row["finger_object_center_height_error_m"] == 0.125


def test_validator_accepts_bottle_visual_cylinder_proxy_shape(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--object-shape",):
                seen["choices"] = list(kwargs["choices"])

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert "bottle_usd_cylinder_proxy" in seen["choices"]


def test_validator_accepts_min_object_lift_gate(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--min-object-lift",):
                seen["kwargs"] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["kwargs"]["type"] is float
    assert seen["kwargs"]["default"] == 0.0


def test_validator_accepts_explicit_object_side_length(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--object-side-length",):
                seen["kwargs"] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["kwargs"]["type"] is float
    assert seen["kwargs"]["default"] is None


def test_validator_accepts_finger_gap_projection_model(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--finger-gap-projection-model",):
                seen["kwargs"] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert tuple(seen["kwargs"]["choices"]) == ("world_aabb", "oriented_box")
    assert seen["kwargs"]["default"] == "world_aabb"


def test_validator_accepts_soft_bottle_effective_contact_width(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args in {
                ("--object-effective-contact-width",),
                ("--object-effective-contact-width-source",),
            }:
                seen[args[0]] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["--object-effective-contact-width"]["type"] is float
    assert seen["--object-effective-contact-width"]["default"] is None
    assert seen["--object-effective-contact-width-source"]["default"] == ""


def test_bottle_usd_contact_proxy_nominal_length_stays_true_bottle_length() -> None:
    class Args:
        object_shape = "bottle_usd_cylinder_proxy"
        stage_units_in_meters = 1.0
        object_length_multiplier = 99.0

    assert _nominal_object_axis_length_stage_units(Args(), 0.055) == pytest.approx(0.206)


def test_bottle_usd_cylinder_proxy_projection_uses_static_axis_and_effective_width() -> None:
    class Args:
        object_shape = "bottle_usd_cylinder_proxy"
        stage_units_in_meters = 1.0
        object_length_multiplier = 4.0

    object_box = {
        "bbox_valid": True,
        "center": [0.0, 0.0, 0.0],
        "size": [0.206, 0.052, 0.052],
    }
    projection = _contact_projection_model_for_args(
        args=Args(),
        object_box=object_box,
        object_axis_unit_world=[1.0, 0.0, 0.0],
        projection_unit_world=[0.0, 1.0, 0.0],
        side_length=0.052,
    )

    assert projection["valid"] is True
    assert projection["source"] == "bottle_usd_cylinder_proxy_oriented_contact_proxy"
    assert projection["projected_width_m"] == pytest.approx(0.052)
    assert projection["half_length_m"] == pytest.approx(0.103)


def test_validator_accepts_object_contact_material_parameters(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args in {
                ("--object-static-friction",),
                ("--object-dynamic-friction",),
                ("--object-restitution",),
                ("--finger-static-friction",),
                ("--finger-dynamic-friction",),
                ("--finger-restitution",),
            }:
                seen[args[0]] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["--object-static-friction"]["type"] is float
    assert seen["--object-dynamic-friction"]["type"] is float
    assert seen["--object-restitution"]["type"] is float
    assert seen["--finger-static-friction"]["type"] is float
    assert seen["--finger-dynamic-friction"]["type"] is float
    assert seen["--finger-restitution"]["type"] is float


def test_validator_accepts_object_width_finger_stop(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--enforce-object-width-finger-stop",):
                seen["kwargs"] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["kwargs"]["action"] == "store_true"


def test_validator_accepts_contact_triggered_diagnostic_hold_mode(monkeypatch) -> None:
    seen: dict[str, list[str]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--diagnostic-held-object-mode",):
                seen["choices"] = list(kwargs["choices"])

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert "follow_after_bilateral_contact" in seen["choices"]


def test_contact_triggered_diagnostic_hold_requires_expected_close_contact_pairs() -> None:
    rows = [
        {
            "phase": "settle",
            "type_name": "CONTACT_FOUND",
            "collider0": "/World/Bottle",
            "collider1": "/World/LeftFinger",
            "sorted_pair": ["/World/Bottle", "/World/LeftFinger"],
        },
        {
            "phase": "close",
            "type_name": "ContactEventType.CONTACT_PERSIST",
            "collider0": "/World/Bottle",
            "collider1": "/World/LeftFinger",
            "sorted_pair": ["/World/Bottle", "/World/LeftFinger"],
        },
    ]

    one_finger = _target_contact_hits_for_phase(
        rows=rows,
        object_path="/World/Bottle",
        expected_finger_paths=["/World/LeftFinger", "/World/RightFinger"],
        phase="close",
    )
    assert one_finger["triggered"] is False
    assert one_finger["finger_hits"] == {"/World/LeftFinger": True, "/World/RightFinger": False}
    assert one_finger["finger_found_hits"] == {"/World/LeftFinger": False, "/World/RightFinger": False}

    rows.append(
        {
            "phase": "close",
            "type_name": "CONTACT_FOUND",
            "collider0": "/World/Bottle",
            "collider1": "/World/RightFinger",
            "sorted_pair": ["/World/Bottle", "/World/RightFinger"],
        }
    )
    both_fingers = _target_contact_hits_for_phase(
        rows=rows,
        object_path="/World/Bottle",
        expected_finger_paths=["/World/LeftFinger", "/World/RightFinger"],
        phase="close",
    )
    assert both_fingers["triggered"] is True
    assert both_fingers["contact_pair_count"] == 2
    assert both_fingers["contact_found_pair_count"] == 1


def test_validator_accepts_open_finger_object_height_gate(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class ParserSpy:
        def add_argument(self, *args, **kwargs):
            if args == ("--max-open-finger-object-center-height-error",):
                seen["kwargs"] = dict(kwargs)

        def parse_args(self):
            raise RuntimeError("stop after parser construction")

    monkeypatch.setattr("argparse.ArgumentParser", lambda *args, **kwargs: ParserSpy())
    try:
        validate_contact_main()
    except RuntimeError as exc:
        assert str(exc) == "stop after parser construction"

    assert seen["kwargs"]["type"] is float
    assert seen["kwargs"]["default"] == 0.04


def test_object_width_stop_target_holds_finger_targets_when_surface_gap_reaches_bottle_width() -> None:
    dof_names = ["joint0", "left_finger", "right_finger"]
    finger_dof_names = {"left_finger": "left_finger", "right_finger": "right_finger"}
    current_qpos = np.asarray([0.1, 0.21, -0.22], dtype=np.float64)
    target = np.asarray([0.2, 0.02, -0.03], dtype=np.float64)
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.054, 0.0],
        "max": [0.01, -0.034, 0.02],
        "center": [0.005, -0.044, 0.01],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.034, 0.0],
        "max": [0.01, 0.054, 0.02],
        "center": [0.005, 0.044, 0.01],
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.068, 0.068]}

    guarded, row = _object_width_stop_target(
        enabled=True,
        current_qpos=current_qpos,
        target=target,
        dof_names=dof_names,
        finger_dof_names=finger_dof_names,
        left_box=left_box,
        right_box=right_box,
        object_box=object_box,
        clearance=0.001,
    )

    assert row["active"] is True
    assert row["status"] == "ACTIVE_HOLD_FINGER_TARGETS_AT_OBJECT_WIDTH"
    assert row["current_surface_gap_m"] == pytest.approx(0.068)
    assert row["stop_surface_gap_m"] == pytest.approx(0.069)
    assert guarded.tolist() == [0.2, 0.21, -0.22]


def test_object_width_stop_target_allows_close_target_while_gap_is_wider_than_bottle() -> None:
    dof_names = ["joint0", "left_finger", "right_finger"]
    finger_dof_names = {"left_finger": "left_finger", "right_finger": "right_finger"}
    current_qpos = np.asarray([0.1, 0.21, -0.22], dtype=np.float64)
    target = np.asarray([0.2, 0.02, -0.03], dtype=np.float64)
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.090, 0.0],
        "max": [0.01, -0.070, 0.02],
        "center": [0.005, -0.080, 0.01],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.070, 0.0],
        "max": [0.01, 0.090, 0.02],
        "center": [0.005, 0.080, 0.01],
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.068, 0.068]}

    guarded, row = _object_width_stop_target(
        enabled=True,
        current_qpos=current_qpos,
        target=target,
        dof_names=dof_names,
        finger_dof_names=finger_dof_names,
        left_box=left_box,
        right_box=right_box,
        object_box=object_box,
        clearance=0.001,
    )

    assert row["active"] is False
    assert row["status"] == "OBSERVED_FINGER_GAP_ABOVE_OBJECT_WIDTH"
    assert row["current_surface_gap_m"] == pytest.approx(0.14)
    assert guarded.tolist() == target.tolist()


def test_object_width_stop_target_uses_projected_interval_before_aabb_fallback() -> None:
    dof_names = ["joint0", "left_finger", "right_finger"]
    finger_dof_names = {"left_finger": "left_finger", "right_finger": "right_finger"}
    current_qpos = np.asarray([0.1, 0.21, -0.22], dtype=np.float64)
    target = np.asarray([0.2, 0.02, -0.03], dtype=np.float64)
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.060, 0.0],
        "max": [0.01, -0.040, 0.02],
        "center": [0.005, -0.050, 0.01],
        "size": [0.01, 0.02, 0.02],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.040, 0.0],
        "max": [0.01, 0.060, 0.02],
        "center": [0.005, 0.050, 0.01],
        "size": [0.01, 0.02, 0.02],
    }
    # The world AABB is deliberately much wider than the true contact
    # projection, as with a rotated bottle proxy.
    object_box = {"bbox_valid": True, "size": [0.206, 0.120, 0.120]}

    guarded, row = _object_width_stop_target(
        enabled=True,
        current_qpos=current_qpos,
        target=target,
        dof_names=dof_names,
        finger_dof_names=finger_dof_names,
        left_box=left_box,
        right_box=right_box,
        object_box=object_box,
        clearance=0.001,
        object_projected_interval=(-0.015, 0.015),
    )

    assert row["mode"] == "closing_axis_projected_inner_gap"
    assert row["projected_inner_gap"]["finger_inner_gap_m"] == pytest.approx(0.08)
    assert row["projected_stop_gap_m"] == pytest.approx(0.031)
    assert row["active"] is False
    assert guarded.tolist() == target.tolist()


def test_object_width_stop_target_can_use_oriented_finger_box_gap() -> None:
    dof_names = ["joint0", "left_finger", "right_finger"]
    finger_dof_names = {"left_finger": "left_finger", "right_finger": "right_finger"}
    current_qpos = np.asarray([0.1, 0.21, -0.22], dtype=np.float64)
    target = np.asarray([0.2, 0.02, -0.03], dtype=np.float64)
    left_box = {
        "bbox_valid": True,
        "min": [-0.06, -0.08, -0.01],
        "max": [0.06, -0.02, 0.01],
        "center": [0.0, -0.050, 0.0],
        "size": [0.12, 0.06, 0.02],
        "oriented_size": [0.012, 0.028, 0.035],
        "oriented_world_matrix": ROT45_Z,
    }
    right_box = {
        "bbox_valid": True,
        "min": [-0.06, 0.02, -0.01],
        "max": [0.06, 0.08, 0.01],
        "center": [0.0, 0.050, 0.0],
        "size": [0.12, 0.06, 0.02],
        "oriented_size": [0.012, 0.028, 0.035],
        "oriented_world_matrix": ROT45_Z,
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.120, 0.120]}

    guarded_aabb, row_aabb = _object_width_stop_target(
        enabled=True,
        current_qpos=current_qpos,
        target=target,
        dof_names=dof_names,
        finger_dof_names=finger_dof_names,
        left_box=left_box,
        right_box=right_box,
        object_box=object_box,
        clearance=0.001,
        object_projected_interval=(-0.034, 0.034),
        use_oriented_finger_boxes=False,
    )
    guarded_obb, row_obb = _object_width_stop_target(
        enabled=True,
        current_qpos=current_qpos,
        target=target,
        dof_names=dof_names,
        finger_dof_names=finger_dof_names,
        left_box=left_box,
        right_box=right_box,
        object_box=object_box,
        clearance=0.001,
        object_projected_interval=(-0.034, 0.034),
        use_oriented_finger_boxes=True,
    )

    assert row_aabb["active"] is True
    assert guarded_aabb.tolist() == [0.2, 0.21, -0.22]
    assert row_obb["active"] is False
    assert row_obb["finger_projection_model"] == "oriented_box_support"
    assert row_obb["projected_inner_gap"]["finger_inner_gap_m"] > row_aabb["projected_inner_gap"][
        "finger_inner_gap_m"
    ]
    assert guarded_obb.tolist() == target.tolist()


def test_loaded_gripper_soft_bottle_calibration_reports_qpos_residual_without_pass_gate() -> None:
    diagnostic = _loaded_gripper_soft_bottle_calibration_diagnostic(
        final_alignment={
            "closing_axis_projected_inner_gap": {
                "valid": True,
                "object_gap_to_lower_finger_m": 0.00349,
                "object_gap_to_upper_finger_m": 0.01846,
                "object_interval_m": [0.1569, 0.2095],
            },
            "object_projection_model": {
                "valid": True,
                "projected_width_m": 0.0526,
            },
        },
        hdf5_gripper_summary={
            "source": "observations/qpos",
            "raw_start": 0.947,
            "raw_end": 0.570,
            "raw_range": 0.377,
            "sample_count": 37,
        },
        reachability_audit={"status": "FAIL_NO_GEOMETRIC_REACH_TO_TARGET_COLLIDER"},
        contact_distance_m=0.002,
        object_effective_contact_width_m=0.052,
        visual_bottle_outer_diameter_m=0.068,
        moving_fingers="both",
        controller_tracking_gate={"pass": True},
        positive_control_gate={"status": "PASS_FORCED_OVERLAP_CONTACT_PIPELINE_REPORTED"},
    )

    assert diagnostic["status"] == "COMPUTED_FORMAL_QPOS_LOADED_CONTACT_RESIDUAL"
    assert diagnostic["formal_gate_result_preserved"] is True
    assert diagnostic["may_set_overall_pass"] is False
    assert diagnostic["qpos_source_is_loaded_gap_calibrated"] is False
    assert diagnostic["requires_raw_finger_or_spacer_calibration"] is True
    assert diagnostic["nearest_surface_gap_m"] == pytest.approx(0.00349)
    assert diagnostic["missing_to_contact_distance_m"] == pytest.approx(0.00149)
    assert diagnostic["per_finger_loaded_closure_deficit_to_zero_gap_m"] == pytest.approx(0.001745)
    assert diagnostic["per_finger_loaded_closure_deficit_to_contact_distance_m"] == pytest.approx(0.000745)
    assert diagnostic["implied_effective_contact_widths_if_explained_as_soft_deformation"][
        "nearest_touch_m"
    ] == pytest.approx(0.05958)


def test_loaded_gripper_soft_bottle_calibration_refuses_invalid_gap() -> None:
    diagnostic = _loaded_gripper_soft_bottle_calibration_diagnostic(
        final_alignment={"closing_axis_projected_inner_gap": {"valid": False}},
        hdf5_gripper_summary={"source": "observations/qpos"},
        reachability_audit={},
        contact_distance_m=0.002,
        object_effective_contact_width_m=0.052,
        visual_bottle_outer_diameter_m=0.068,
        moving_fingers="both",
    )

    assert diagnostic["status"] == "NOT_COMPUTED_INVALID_FINAL_PROJECTED_GAP"
    assert diagnostic["formal_gate_result_preserved"] is True
    assert diagnostic["may_set_overall_pass"] is False


def test_open_finger_object_height_alignment_rejects_airborne_open_gripper() -> None:
    left_box = {"bbox_valid": True, "center": [0.0, -0.04, 0.164]}
    right_box = {"bbox_valid": True, "center": [0.0, 0.04, 0.164]}
    object_box = {"bbox_valid": True, "center": [0.0, 0.0, 0.039]}

    row = _open_finger_object_height_alignment(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        max_error=0.04,
    )

    assert row["pass"] is False
    assert row["status"] == "FAIL_OPEN_FINGER_OBJECT_HEIGHT_MISMATCH"
    assert row["height_error_m"] == pytest.approx(0.125)


def test_open_finger_object_height_alignment_accepts_bottle_body_height() -> None:
    left_box = {"bbox_valid": True, "center": [0.0, -0.04, 0.048]}
    right_box = {"bbox_valid": True, "center": [0.0, 0.04, 0.048]}
    object_box = {"bbox_valid": True, "center": [0.0, 0.0, 0.039]}

    row = _open_finger_object_height_alignment(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        max_error=0.04,
    )

    assert row["pass"] is True
    assert row["status"] == "PASS_OPEN_FINGER_OBJECT_HEIGHT_ALIGNMENT"
    assert row["height_error_m"] == pytest.approx(0.009)


def test_active_grasp_geometry_precondition_fails_when_object_wider_than_open_gap() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [0.0, 0.0, 0.0],
        "max": [0.01, 0.01, 0.02],
        "center": [0.005, 0.005, 0.01],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.0, 0.07],
        "max": [0.01, 0.01, 0.08],
        "center": [0.005, 0.005, 0.075],
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.068, 0.068]}

    result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=2,
        clearance=0.001,
    )

    assert result["pass"] is False
    assert result["status"] == "FAIL_ACTIVE_FREE_SPACE_CENTERLINE_GEOMETRY_PRECONDITION"
    assert result["open_finger_center_gap_m"] == 0.065
    assert result["open_finger_surface_gap_m"] == 0.05
    assert result["object_width_along_gap_axis_m"] == 0.068


def test_active_grasp_geometry_precondition_rejects_true_axis_gap_even_when_centerline_passes() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.060, 0.0],
        "max": [0.01, -0.020, 0.02],
        "center": [0.005, -0.040, 0.01],
        "size": [0.01, 0.04, 0.02],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.020, 0.0],
        "max": [0.01, 0.060, 0.02],
        "center": [0.005, 0.040, 0.01],
        "size": [0.01, 0.04, 0.02],
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.068, 0.068]}

    result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.001,
    )

    assert result["pass"] is False
    assert result["status"] == "FAIL_ACTIVE_FREE_SPACE_TRUE_CLOSING_AXIS_GEOMETRY_PRECONDITION"
    assert result["open_finger_center_gap_m"] == 0.08
    assert result["open_finger_surface_gap_m"] == 0.04
    assert result["centerline_gap_pass"] is True
    assert result["true_closing_axis_gap_pass"] is False


def test_active_grasp_geometry_precondition_accepts_true_axis_inner_gap() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.060, 0.0],
        "max": [0.01, -0.020, 0.02],
        "center": [0.005, -0.040, 0.01],
        "size": [0.01, 0.04, 0.02],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.020, 0.0],
        "max": [0.01, 0.060, 0.02],
        "center": [0.005, 0.040, 0.01],
        "size": [0.01, 0.04, 0.02],
    }
    object_box = {
        "bbox_valid": True,
        "min": [-0.006, -0.010, 0.004],
        "max": [0.006, 0.010, 0.016],
        "center": [0.0, 0.0, 0.010],
        "size": [0.012, 0.020, 0.012],
    }

    result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.001,
    )

    assert result["pass"] is True
    assert result["status"] == "PASS_ACTIVE_GRASP_GEOMETRY_PRECONDITION"
    assert result["centerline_gap_pass"] is True
    assert result["true_closing_axis_gap_pass"] is True


def test_active_grasp_geometry_precondition_uses_oriented_cylinder_projection_override() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [-0.010, 0.040, -0.010],
        "max": [0.010, 0.060, 0.010],
        "center": [0.0, 0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    right_box = {
        "bbox_valid": True,
        "min": [-0.010, -0.060, -0.010],
        "max": [0.010, -0.040, 0.010],
        "center": [0.0, -0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    # This world AABB is intentionally wider than the true finger gap along Y.
    # It represents the failure mode for a rotated long bottle proxy: AABB
    # projection includes the long axis even when the authored cylinder axis is
    # perpendicular to the gripper closing direction.
    object_box = {
        "bbox_valid": True,
        "min": [-0.100, -0.060, -0.020],
        "max": [0.100, 0.060, 0.020],
        "center": [0.0, 0.0, 0.0],
        "size": [0.200, 0.120, 0.040],
    }
    projection = _oriented_cylinder_projection_model(
        object_box=object_box,
        object_axis_unit_world=[1.0, 0.0, 0.0],
        projection_unit_world=[0.0, 1.0, 0.0],
        radius_m=0.025,
        half_length_m=0.100,
        source="test_cylinder_proxy",
    )

    result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.005,
        object_projected_interval=tuple(projection["object_interval_m"]),
        object_projection_model=projection,
    )

    assert projection["projected_width_m"] == pytest.approx(0.05)
    assert result["pass"] is True
    assert result["true_closing_axis_gap_pass"] is True
    assert result["object_projection_model"]["source"] == "test_cylinder_proxy"


def test_active_grasp_geometry_precondition_can_use_oriented_finger_box_gap() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [-0.06, 0.02, -0.01],
        "max": [0.06, 0.08, 0.01],
        "center": [0.0, 0.050, 0.0],
        "size": [0.12, 0.06, 0.02],
        "oriented_size": [0.012, 0.028, 0.035],
        "oriented_world_matrix": ROT45_Z,
    }
    right_box = {
        "bbox_valid": True,
        "min": [-0.06, -0.08, -0.01],
        "max": [0.06, -0.02, 0.01],
        "center": [0.0, -0.050, 0.0],
        "size": [0.12, 0.06, 0.02],
        "oriented_size": [0.012, 0.028, 0.035],
        "oriented_world_matrix": ROT45_Z,
    }
    object_box = {
        "bbox_valid": True,
        "min": [-0.103, -0.060, -0.034],
        "max": [0.103, 0.060, 0.034],
        "center": [0.0, 0.0, 0.0],
        "size": [0.206, 0.120, 0.068],
    }
    projection = _oriented_cylinder_projection_model(
        object_box=object_box,
        object_axis_unit_world=[1.0, 0.0, 0.0],
        projection_unit_world=[0.0, 1.0, 0.0],
        radius_m=0.034,
        half_length_m=0.103,
        source="test_bottle500_proxy",
    )

    aabb_result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.001,
        object_projected_interval=tuple(projection["object_interval_m"]),
        object_projection_model=projection,
        use_oriented_finger_boxes=False,
    )
    obb_result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.001,
        object_projected_interval=tuple(projection["object_interval_m"]),
        object_projection_model=projection,
        use_oriented_finger_boxes=True,
    )

    assert aabb_result["pass"] is False
    assert obb_result["pass"] is True
    assert obb_result["finger_projection_model"] == "oriented_box_support"
    assert obb_result["closing_axis_projected_inner_gap"]["finger_inner_gap_m"] > 0.068


def test_closing_axis_gap_centering_solver_centers_feasible_bottle_without_z_shift() -> None:
    lower_box = {
        "bbox_valid": True,
        "min": [-0.010, -0.060, -0.010],
        "max": [0.010, -0.040, 0.010],
        "center": [0.0, -0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    upper_box = {
        "bbox_valid": True,
        "min": [-0.010, 0.040, -0.010],
        "max": [0.010, 0.060, 0.010],
        "center": [0.0, 0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    object_box = {
        "bbox_valid": True,
        "center": [0.0, -0.020, 0.012],
        "size": [0.200, 0.120, 0.040],
    }
    projection = _oriented_cylinder_projection_model(
        object_box=object_box,
        object_axis_unit_world=[1.0, 0.0, 0.0],
        projection_unit_world=[0.0, 1.0, 0.0],
        radius_m=0.025,
        half_length_m=0.100,
        source="test_cylinder_proxy",
    )

    result = _closing_axis_gap_centering_solver(
        lower_box=lower_box,
        upper_box=upper_box,
        object_projection_model=projection,
        projection_unit_world=[0.0, 1.0, 0.0],
        clearance=0.005,
    )

    assert result["pass"] is True
    assert result["status"] == "PASS_CLOSING_AXIS_GAP_CENTERING_SHIFT_COMPUTED"
    assert result["delta_world_m"] == pytest.approx([0.0, 0.020, 0.0])
    assert result["gap_after_expected"]["object_gap_to_lower_finger_m"] == pytest.approx(0.015)
    assert result["gap_after_expected"]["object_gap_to_upper_finger_m"] == pytest.approx(0.015)


def test_closing_axis_gap_centering_solver_rejects_infeasible_bottle_width() -> None:
    lower_box = {
        "bbox_valid": True,
        "min": [-0.010, -0.060, -0.010],
        "max": [0.010, -0.040, 0.010],
        "center": [0.0, -0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    upper_box = {
        "bbox_valid": True,
        "min": [-0.010, 0.040, -0.010],
        "max": [0.010, 0.060, 0.010],
        "center": [0.0, 0.050, 0.0],
        "size": [0.020, 0.020, 0.020],
    }
    object_box = {
        "bbox_valid": True,
        "center": [0.0, 0.0, 0.012],
        "size": [0.200, 0.120, 0.040],
    }
    projection = _oriented_cylinder_projection_model(
        object_box=object_box,
        object_axis_unit_world=[1.0, 0.0, 0.0],
        projection_unit_world=[0.0, 1.0, 0.0],
        radius_m=0.040,
        half_length_m=0.100,
        source="test_too_wide_cylinder_proxy",
    )

    result = _closing_axis_gap_centering_solver(
        lower_box=lower_box,
        upper_box=upper_box,
        object_projection_model=projection,
        projection_unit_world=[0.0, 1.0, 0.0],
        clearance=0.005,
    )

    assert result["pass"] is False
    assert result["status"] == "FAIL_CLOSING_AXIS_INNER_GAP_INFEASIBLE"
    assert result["shortfall_m"] == pytest.approx(0.010)


def test_active_grasp_geometry_precondition_rejects_skewed_true_axis_overlap() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [0.096, -0.051, 0.0],
        "max": [0.110, -0.031, 0.02],
        "center": [0.103, -0.041, 0.01],
    }
    right_box = {
        "bbox_valid": True,
        "min": [-0.007, -0.010, 0.0],
        "max": [0.007, 0.010, 0.02],
        "center": [0.0, 0.0, 0.01],
    }
    object_box = {"bbox_valid": True, "size": [0.206, 0.068, 0.068]}

    result = _active_grasp_geometry_precondition(
        require_active_target_contact=True,
        already_in_contact_setup=False,
        open_left_box=left_box,
        open_right_box=right_box,
        object_box=object_box,
        gap_axis=1,
        clearance=0.001,
    )

    assert result["pass"] is False
    assert result["status"] == "FAIL_ACTIVE_FREE_SPACE_TRUE_CLOSING_AXIS_GEOMETRY_PRECONDITION"
    assert result["open_finger_center_gap_m"] > 0.11
    assert result["open_finger_surface_gap_m"] < 0.021
    assert result["object_width_centerline_m"] == 0.068
    assert result["centerline_gap_pass"] is True
    assert result["true_closing_axis_gap_pass"] is False


def test_tabletop_z_shift_places_object_bottom_on_table_top_with_clearance() -> None:
    result = _tabletop_z_shift_from_bboxes(
        table_box={"bbox_valid": True, "min": [-0.5, -0.3, -0.04], "max": [0.5, 0.3, 0.0]},
        object_box={"bbox_valid": True, "min": [-0.1, -0.034, -0.020], "max": [0.1, 0.034, 0.048]},
        clearance=0.001,
    )

    assert result["pass"] is True
    assert result["table_top_z_m"] == 0.0
    assert result["object_bottom_z_before_m"] == -0.02
    assert result["target_object_bottom_z_m"] == 0.001
    assert result["z_shift_m"] == 0.021

from __future__ import annotations

from pathlib import Path

from aloha_isaac_replay.scripts.run_phase106_bottleusd_already_grasped_gate import _phase106_args
from aloha_isaac_replay.scripts.run_phase107_bottleusd_hdf5_drive_target_gate import _phase107_args
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import (
    _active_grasp_geometry_precondition,
    _tabletop_z_shift_from_bboxes,
)
from aloha_isaac_replay.scripts.validate_aloha1_gripper_passive_contact import main as validate_contact_main


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
    assert result["status"] == "FAIL_ACTIVE_FREE_SPACE_GEOMETRY_PRECONDITION"
    assert result["open_finger_center_gap_m"] == 0.065
    assert result["open_finger_surface_gap_m"] == 0.05
    assert result["object_width_along_gap_axis_m"] == 0.068


def test_active_grasp_geometry_precondition_uses_proxy_center_gap_not_surface_gap_only() -> None:
    left_box = {
        "bbox_valid": True,
        "min": [0.0, -0.060, 0.0],
        "max": [0.01, -0.020, 0.02],
        "center": [0.005, -0.040, 0.01],
    }
    right_box = {
        "bbox_valid": True,
        "min": [0.0, 0.020, 0.0],
        "max": [0.01, 0.060, 0.02],
        "center": [0.005, 0.040, 0.01],
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

    assert result["pass"] is True
    assert result["status"] == "PASS_ACTIVE_GRASP_GEOMETRY_PRECONDITION"
    assert result["open_finger_center_gap_m"] == 0.08
    assert result["open_finger_surface_gap_m"] == 0.04
    assert result["surface_gap_is_diagnostic_only"] is True


def test_active_grasp_geometry_precondition_uses_centerline_distance_for_skewed_fingers() -> None:
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

    assert result["pass"] is True
    assert result["open_finger_center_gap_m"] > 0.11
    assert result["open_finger_surface_gap_m"] < 0.021
    assert result["object_width_centerline_m"] == 0.068


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

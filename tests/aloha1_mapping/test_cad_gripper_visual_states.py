from __future__ import annotations

from pathlib import Path

import pytest

from tools.aloha1_mapping.cad_gripper_visual_states import capture_plan
from tools.aloha1_mapping.cad_gripper_visual_states import infer_static_cad_state
from tools.aloha1_mapping.cad_gripper_visual_states import orthographic_frame
from tools.aloha1_mapping.cad_gripper_visual_states import points_mm_to_m
from tools.aloha1_mapping.cad_gripper_visual_states import required_capture_inventory
from tools.aloha1_mapping.cad_gripper_visual_states import state_translations_mm
from tools.aloha1_mapping.cad_gripper_visual_states import view_basis


def test_static_simple_viper_pose_is_closer_to_urdf_closed_than_open() -> None:
    result = infer_static_cad_state(
        cad_positive_center_mm=16.0776544657,
        cad_negative_center_mm=-15.8778739504,
        urdf_closed_positive_center_mm=17.75,
        urdf_open_positive_center_mm=53.75,
    )

    assert result["status"] == "PASS"
    assert result["classification"] == "CLOSED_REFERENCE"
    assert result["cad_half_separation_mm"] == 15.97776420805
    assert result["closed_residual_mm"] < 1.78
    assert result["open_residual_mm"] > 37.7


def test_open_state_moves_each_handed_brep_outward_by_urdf_travel() -> None:
    translations = state_translations_mm(open_delta_mm=36.0)

    assert translations["closed"]["cad_positive_x_finger"] == [0.0, 0.0, 0.0]
    assert translations["closed"]["cad_negative_x_finger"] == [0.0, 0.0, 0.0]
    assert translations["open"]["cad_positive_x_finger"] == [36.0, 0.0, 0.0]
    assert translations["open"]["cad_negative_x_finger"] == [-36.0, 0.0, 0.0]


def test_screenshot_inventory_is_four_views_times_two_states_times_two_versions() -> None:
    inventory = required_capture_inventory()

    assert len(inventory) == 16
    assert set(inventory) == {
        f"{state}_{view}_{version}.png"
        for state in ("closed", "open")
        for view in ("true_top", "true_bottom", "tip_end", "base_oblique")
        for version in ("raw", "annotated")
    }


@pytest.mark.parametrize(
    ("view_id", "camera_forward", "image_up", "image_right"),
    [
        ("true_top", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        (
            "true_bottom",
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
        ("tip_end", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        (
            "base_oblique",
            (0.0, -(2**-0.5), -(2**-0.5)),
            (0.0, -(2**-0.5), 2**-0.5),
            (-1.0, 0.0, 0.0),
        ),
    ],
)
def test_view_basis_has_proven_cad_axis_direction(
    view_id: str,
    camera_forward: tuple[float, float, float],
    image_up: tuple[float, float, float],
    image_right: tuple[float, float, float],
) -> None:
    result = view_basis(view_id)

    assert result["camera_forward"] == camera_forward
    assert result["image_up"] == image_up
    assert result["image_right"] == image_right


def test_open_and_closed_capture_plan_share_camera_per_view(tmp_path: Path) -> None:
    plan = capture_plan(output_root=tmp_path)

    assert len(plan) == 8
    for view_id in ("true_top", "true_bottom", "tip_end", "base_oblique"):
        pair = [record for record in plan if record["view_id"] == view_id]
        assert {record["state_id"] for record in pair} == {"open", "closed"}
        assert len({record["camera_key"] for record in pair}) == 1
        assert all(Path(record["raw_path"]).is_absolute() for record in pair)
        assert all(
            Path(record["annotated_path"]).is_absolute() for record in pair
        )


def test_orthographic_frame_contains_union_bounds_for_paired_states() -> None:
    points = [
        (-75.0, -590.0, 396.0),
        (75.0, -430.0, 462.0),
        (-66.0, -589.0, 400.0),
        (66.0, -493.0, 454.0),
    ]

    frame = orthographic_frame(
        points_mm=points,
        view_id="true_top",
        resolution=(1280, 900),
        margin=1.12,
    )

    assert frame["target_mm"] == pytest.approx((0.0, -510.0, 429.0))
    assert frame["ortho_height_mm"] >= 160.0 * 1.12
    assert frame["ortho_width_mm"] >= 150.0 * 1.12
    assert frame["camera_location_mm"][2] > 462.0


def test_blender_render_copy_converts_millimetres_to_metres() -> None:
    assert points_mm_to_m(
        [[1000.0, -500.0, 25.0], [0.0, 0.0, 0.0]]
    ) == [[1.0, -0.5, 0.025], [0.0, 0.0, 0.0]]

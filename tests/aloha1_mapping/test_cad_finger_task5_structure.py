from __future__ import annotations

from tools.aloha1_mapping.cad_finger_task5_structure import GLOBAL_SESSION_HIDDEN_VISUALS
from tools.aloha1_mapping.cad_finger_task5_structure import LEGAL_POSES_M
from tools.aloha1_mapping.cad_finger_task5_structure import POSE_ALIASES
from tools.aloha1_mapping.cad_finger_task5_structure import VIEW_HIDDEN_VISUALS
from tools.aloha1_mapping.cad_finger_task5_structure import VIEW_RADII_M
from tools.aloha1_mapping.cad_finger_task5_structure import drive_mimic_status
from tools.aloha1_mapping.cad_finger_task5_structure import hide_non_target_robot_gprim
from tools.aloha1_mapping.cad_finger_task5_structure import hide_non_target_robot_visual
from tools.aloha1_mapping.cad_finger_task5_structure import hide_robot_debug_container
from tools.aloha1_mapping.cad_finger_task5_structure import summarize_image_projection
from tools.aloha1_mapping.cad_finger_task5_structure import validate_pose_records


def _records() -> list[dict[str, object]]:
    gaps = {
        "closed": 0.004,
        "partial": 0.040,
        "maximum_legal_aperture": 0.076,
    }
    return [
        {
            "state": state,
            "readback_m": list(target),
            "limits_m": {
                "left": [0.021, 0.057],
                "right": [-0.057, -0.021],
            },
            "surface_gap_m": gaps[state],
        }
        for state, target in LEGAL_POSES_M.items()
    ]


def test_pose_plan_uses_only_source_derived_limits_and_midpoint() -> None:
    assert LEGAL_POSES_M == {
        "closed": (0.021, -0.021),
        "partial": (0.039, -0.039),
        "maximum_legal_aperture": (0.057, -0.057),
    }
    assert POSE_ALIASES == {"open": "maximum_legal_aperture"}


def test_legal_pose_readback_and_gap_are_machine_gated() -> None:
    result = validate_pose_records(_records())
    assert result["status"] == "PASS"
    assert all(result["gates"].values())
    assert "does not prove drive tracking" in result["acceptance_boundary"]


def test_zero_or_nonmonotonic_pose_cannot_pass() -> None:
    records = _records()
    records[0]["readback_m"] = [0.0, 0.0]
    records[1]["surface_gap_m"] = 0.080
    result = validate_pose_records(records)
    assert result["status"] == "FAIL"
    assert result["gates"]["all_readbacks_within_limits"] is False
    assert result["gates"]["aperture_monotonicity"] is False


def test_missing_mimic_and_zero_max_force_are_not_greenwashed() -> None:
    result = drive_mimic_status(
        physx_mimic_api_present=False,
        left_max_force=0.0,
        right_max_force=0.0,
    )
    assert result["status"] == "FAIL"
    assert result["teleport_or_pose_injection_counts_as_dynamic_pass"] is False


def test_evidence_views_expose_fingers_without_changing_asset_geometry() -> None:
    prop_visuals = (
        "/workcell/vx300s_left/"
        "vx300s_left_gripper_prop_link/visuals"
    )
    prop_link = (
        "/workcell/vx300s_left/"
        "vx300s_left_gripper_prop_link"
    )
    camera_focus = (
        "/workcell/vx300s_left/"
        "vx300s_left_camera_focus"
    )
    shell = (
        "/workcell/vx300s_left/"
        "vx300s_left_gripper_link/visuals"
    )
    assert VIEW_RADII_M["true_top"] >= 0.9
    assert VIEW_RADII_M["true_bottom"] >= 0.9
    assert VIEW_RADII_M["base_oblique_tool"] == 0.38
    assert VIEW_RADII_M["base_oblique_top"] == 0.25
    assert VIEW_RADII_M["base_oblique_closing"] == 0.14
    assert (
        prop_link,
        camera_focus,
    ) == GLOBAL_SESSION_HIDDEN_VISUALS
    assert prop_visuals not in GLOBAL_SESSION_HIDDEN_VISUALS
    assert VIEW_HIDDEN_VISUALS["tip_end"] == ()
    assert VIEW_HIDDEN_VISUALS["base_oblique"] == (shell,)
    assert VIEW_HIDDEN_VISUALS["true_top"] == ()
    assert VIEW_HIDDEN_VISUALS["true_bottom"] == ()


def test_camera_focus_visual_is_hidden_but_gripper_context_is_kept() -> None:
    prefix = "/workcell/vx300s_left/"
    assert hide_non_target_robot_visual(
        prefix + "vx300s_left_camera_focus/visuals",
        "visuals",
    )
    for link in (
        "vx300s_left_gripper_link",
        "vx300s_left_gripper_prop_link",
        "vx300s_left_left_finger_link",
        "vx300s_left_right_finger_link",
    ):
        assert not hide_non_target_robot_visual(
            prefix + link + "/visuals",
            "visuals",
        )
    assert not hide_non_target_robot_visual(
        "/workcell/table/visuals",
        "visuals",
    )


def test_non_target_robot_gprim_filter_keeps_only_fingers_and_shell() -> None:
    prefix = "/workcell/vx300s_left/"
    assert hide_non_target_robot_gprim(
        prefix + "vx300s_left_camera_focus/visuals/cross/mesh"
    )
    assert hide_non_target_robot_gprim(
        prefix + "vx300s_left_gripper_prop_link/visuals/prop/mesh"
    )
    for path in (
        prefix
        + "vx300s_left_left_finger_link/visuals/"
        "diagnostic_supplier_cad_left_finger/mesh",
        prefix
        + "vx300s_left_right_finger_link/visuals/"
        "diagnostic_supplier_cad_right_finger/mesh",
        prefix
        + "vx300s_left_gripper_link/visuals/shell/mesh",
    ):
        assert not hide_non_target_robot_gprim(path)
    assert not hide_non_target_robot_gprim("/workcell/table/mesh")


def test_collision_and_site_debug_geometry_is_hidden_for_render_only() -> None:
    prefix = "/workcell/vx300s_left/"
    assert hide_robot_debug_container(
        prefix + "vx300s_left_gripper_link/collisions",
        "collisions",
    )
    assert hide_robot_debug_container(
        prefix + "vx300s_left_gripper_link/sites",
        "sites",
    )
    assert not hide_robot_debug_container(
        prefix + "vx300s_left_gripper_link/visuals",
        "visuals",
    )
    assert not hide_robot_debug_container(
        "/workcell/table/collisions",
        "collisions",
    )


def test_projection_summary_records_pixel_bbox_and_center() -> None:
    result = summarize_image_projection(
        [[100.0, 200.0], [300.0, 500.0], [200.0, 350.0]],
        width=1280,
        height=900,
    )
    assert result["finite_point_count"] == 3
    assert result["bbox_min_px"] == [100.0, 200.0]
    assert result["bbox_max_px"] == [300.0, 500.0]
    assert result["bbox_center_px"] == [200.0, 350.0]
    assert result["fully_in_frame"] is True

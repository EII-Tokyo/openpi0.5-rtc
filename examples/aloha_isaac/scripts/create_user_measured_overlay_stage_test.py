from __future__ import annotations

from pathlib import Path

from examples.aloha_isaac.scripts import create_user_measured_overlay_stage as overlay


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_overlay_defaults_live_next_to_confirmed_aloha_stage() -> None:
    confirmed_dir = REPO_ROOT / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"

    assert overlay.DEFAULT_BASE_USD.parent == confirmed_dir
    assert overlay.DEFAULT_OUTPUT_USD.parent == confirmed_dir
    assert overlay.DEFAULT_BASE_USD.name == "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
    assert overlay.DEFAULT_OUTPUT_USD.name == (
        "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda"
    )


def test_overlay_uses_user_measured_workcell_config() -> None:
    assert overlay.DEFAULT_CONFIG == REPO_ROOT / "examples/aloha_isaac/config/workcell_user_measured.yaml"


def test_industrial_office_lighting_profile_uses_local_work_area_highlight() -> None:
    assert overlay.LIGHTING_ROOT == "/World/Lighting/industrial_office_lab"
    assert overlay.ALOHA_SATIN_BLACK_MATERIAL.startswith(overlay.LIGHTING_ROOT)
    assert overlay.WINDOW_KEY_LIGHT["name"] == "soft_window_key_from_right"
    assert overlay.WINDOW_KEY_LIGHT["width"] > 1.0
    assert overlay.WINDOW_KEY_LIGHT["height"] > 1.0

    names = [spec["name"] for spec in overlay.CEILING_STRIP_LIGHTS]
    assert len(names) == 8
    assert len(set(names)) == 8
    assert sum(name.startswith("ceiling_strip_workbench_") for name in names) == 4
    assert sum(name.startswith("ceiling_strip_background_") for name in names) == 4
    assert all(spec["width"] > spec["height"] for spec in overlay.CEILING_STRIP_LIGHTS)
    assert 90.0 <= overlay.FILL_DOME_INTENSITY <= 240.0


def test_window_daylight_has_visible_depth_layers_not_a_single_flat_panel() -> None:
    layer_names = {spec["name"] for spec in overlay.WINDOW_DAYLIGHT_DEPTH_LAYERS}

    assert {"exterior_sky_gradient_top", "exterior_sunlit_floor_band", "exterior_soft_shadow_band"} <= layer_names
    assert len(overlay.WINDOW_DAYLIGHT_DEPTH_LAYERS) >= 3
    assert all(spec["translation"][0] > overlay.OFFICE_WINDOW_PANEL["translation"][0] for spec in overlay.WINDOW_DAYLIGHT_DEPTH_LAYERS)
    assert all(spec["collision"] is False for spec in overlay.WINDOW_DAYLIGHT_DEPTH_LAYERS)

    colors = [tuple(spec["color"]) for spec in overlay.WINDOW_DAYLIGHT_DEPTH_LAYERS]
    assert len(set(colors)) == len(colors)
    assert max(max(color) for color in colors) <= 0.92
    assert min(min(color) for color in colors) >= 0.34


def test_stage_spotlights_hit_aloha_from_multiple_angles_without_flattening_scene() -> None:
    names = {spec["name"] for spec in overlay.ALOHA_STAGE_SPOT_LIGHTS}

    assert {
        "left_front_arm_highlight",
        "right_front_arm_highlight",
        "rear_rim_light",
        "pipe_task_highlight",
        "low_cross_fill",
    } <= names
    assert len(overlay.ALOHA_STAGE_SPOT_LIGHTS) >= 5

    positions = [spec["position"] for spec in overlay.ALOHA_STAGE_SPOT_LIGHTS]
    assert any(pos[0] < -0.5 and pos[1] > 0.4 for pos in positions)
    assert any(pos[0] > 0.5 and pos[1] > 0.4 for pos in positions)
    assert any(pos[1] < -0.8 and pos[2] > 1.0 for pos in positions)
    assert all(0.10 <= spec["width"] <= 0.85 for spec in overlay.ALOHA_STAGE_SPOT_LIGHTS)
    assert all(0.10 <= spec["height"] <= 0.85 for spec in overlay.ALOHA_STAGE_SPOT_LIGHTS)
    assert all(450.0 <= spec["intensity"] <= 2600.0 for spec in overlay.ALOHA_STAGE_SPOT_LIGHTS)


def test_aloha_beauty_lights_prioritize_specular_highlights_over_room_wash() -> None:
    all_beauty_lights = overlay.ALOHA_STAGE_SPOT_LIGHTS + [overlay.ALOHA_VIEW_BEAUTY_LIGHT]

    assert overlay.ALOHA_VIEW_BEAUTY_LIGHT["name"] == "camera_angle_aloha_beauty_key"
    assert overlay.ALOHA_VIEW_BEAUTY_LIGHT["position"][1] > 1.6
    assert overlay.ALOHA_VIEW_BEAUTY_LIGHT["target"][2] < 0.40

    assert all(0.08 <= spec["diffuse"] <= 0.45 for spec in all_beauty_lights)
    assert all(1.15 <= spec["specular"] <= 3.0 for spec in all_beauty_lights)
    assert all(spec["normalize"] is True for spec in all_beauty_lights)
    assert all(-1.0 <= spec["exposure"] <= 1.2 for spec in all_beauty_lights)


def test_aloha_beauty_lights_are_light_linked_to_robot_and_task_objects() -> None:
    assert "/scene/left_base_link" in overlay.ALOHA_BEAUTY_LIGHT_LINK_TARGETS
    assert "/scene/right_base_link" in overlay.ALOHA_BEAUTY_LIGHT_LINK_TARGETS
    assert "/World/PipePlaceholder" in overlay.ALOHA_BEAUTY_LIGHT_LINK_TARGETS

    all_beauty_lights = overlay.ALOHA_STAGE_SPOT_LIGHTS + [overlay.ALOHA_VIEW_BEAUTY_LIGHT]
    assert all(spec["light_link_targets"] == overlay.ALOHA_BEAUTY_LIGHT_LINK_TARGETS for spec in all_beauty_lights)
    assert all(spec["light_link_include_root"] is False for spec in all_beauty_lights)


def test_room_base_lights_stay_below_camera_light_like_wash_levels() -> None:
    non_robot_rect_lights = [
        overlay.WINDOW_KEY_LIGHT,
        overlay.SURGICAL_SOFTBOX_LIGHT,
        overlay.FRONT_FILL_LIGHT,
        *overlay.BOUNCE_FILL_LIGHTS,
        *overlay.CEILING_STRIP_LIGHTS,
    ]

    assert overlay.FILL_DOME_INTENSITY <= 120.0
    assert all(spec.get("diffuse", 1.0) <= 0.82 for spec in non_robot_rect_lights)
    assert all(spec.get("specular", 1.0) <= 1.15 for spec in non_robot_rect_lights)
    assert overlay.SURGICAL_SOFTBOX_LIGHT["intensity"] <= 3600.0
    assert overlay.WINDOW_KEY_LIGHT["intensity"] <= 1200.0


def test_lab_environment_keeps_startup_camera_side_open_and_avoids_white_box() -> None:
    assert overlay.OFFICE_ENV_ROOT == "/World/OfficeEnvironment/industrial_lab_corner"
    assert overlay.OFFICE_ENV_MATERIAL_ROOT.startswith(overlay.OFFICE_ENV_ROOT)

    surface_names = {spec["name"] for spec in overlay.OFFICE_ROOM_SURFACES}
    assert {"matte_epoxy_floor", "rear_acoustic_wall", "left_side_wall", "ceiling_baffle"} <= surface_names
    assert all(spec["collision"] is False for spec in overlay.OFFICE_ROOM_SURFACES)
    assert all(max(spec["color"]) < 0.88 for spec in overlay.OFFICE_ROOM_SURFACES)

    rear_wall = next(spec for spec in overlay.OFFICE_ROOM_SURFACES if spec["name"] == "rear_acoustic_wall")
    assert rear_wall["translation"][1] < 0.0

    ceiling = next(spec for spec in overlay.OFFICE_ROOM_SURFACES if spec["name"] == "ceiling_baffle")
    assert ceiling["translation"][2] > max(spec["position"][2] for spec in overlay.CEILING_STRIP_LIGHTS)

    assert overlay.OFFICE_WINDOW_PANEL["translation"][0] > 0.0
    assert len(overlay.OFFICE_VERTICAL_BLINDS) >= 6
    assert all(blind["translation"][0] > 0.0 for blind in overlay.OFFICE_VERTICAL_BLINDS)


def test_stage_lighting_makes_aloha_work_area_bright_without_overexposed_white_surfaces() -> None:
    floor = next(spec for spec in overlay.OFFICE_ROOM_SURFACES if spec["name"] == "matte_epoxy_floor")
    rear_wall = next(spec for spec in overlay.OFFICE_ROOM_SURFACES if spec["name"] == "rear_acoustic_wall")
    left_wall = next(spec for spec in overlay.OFFICE_ROOM_SURFACES if spec["name"] == "left_side_wall")

    assert floor["size"][0] >= 4.0
    assert floor["size"][1] >= 3.2
    assert rear_wall["translation"][1] <= -1.65
    assert left_wall["translation"][0] <= -2.0

    assert overlay.OPERATING_SURFACE["name"] == "warm_maple_workbench_top"
    assert 0.45 <= max(overlay.OPERATING_SURFACE["color"]) <= 0.78

    assert overlay.SURGICAL_SOFTBOX_LIGHT["name"] == "large_softbox_over_workbench"
    assert overlay.SURGICAL_SOFTBOX_LIGHT["width"] >= 1.4
    assert overlay.SURGICAL_SOFTBOX_LIGHT["height"] >= 0.7
    assert 1800.0 <= overlay.SURGICAL_SOFTBOX_LIGHT["intensity"] <= 5200.0

    assert overlay.FRONT_FILL_LIGHT["name"] == "front_fill_for_black_aloha"
    assert 600.0 <= overlay.FRONT_FILL_LIGHT["intensity"] <= 1600.0
    assert overlay.FILL_DOME_INTENSITY <= 240.0


def test_neutral_lab_surfaces_have_material_contrast_not_white_mode() -> None:
    assert 0.58 <= min(overlay.WALL_DIFFUSE_NEUTRAL) <= 0.78
    assert 0.50 <= min(overlay.CEILING_DIFFUSE_NEUTRAL) <= 0.76
    assert 0.34 <= min(overlay.FLOOR_DIFFUSE_EPOXY) <= 0.58
    assert overlay.WALL_ROUGHNESS >= 0.72
    assert overlay.ALOHA_SATIN_BLACK_DIFFUSE[0] >= 0.018

    assert len(overlay.WALL_GRAIN_MARKS) >= 72
    assert all(min(mark["size"]) <= 0.003 for mark in overlay.WALL_GRAIN_MARKS)
    assert all(max(mark["color"]) < 0.86 for mark in overlay.WALL_GRAIN_MARKS)


def test_explicit_fill_lights_are_moderate_and_do_not_make_a_white_stage() -> None:
    names = {spec["name"] for spec in overlay.BOUNCE_FILL_LIGHTS}
    assert {"left_cabinet_soft_fill", "rear_wall_gentle_fill"} <= names
    assert all(spec["width"] >= 1.5 for spec in overlay.BOUNCE_FILL_LIGHTS)
    assert all(350.0 <= spec["intensity"] <= 1400.0 for spec in overlay.BOUNCE_FILL_LIGHTS)
    assert overlay.SURGICAL_SOFTBOX_LIGHT["intensity"] <= 5200.0
    assert overlay.FRONT_FILL_LIGHT["intensity"] <= 1600.0


def test_environment_uses_downloaded_nvidia_office_and_warehouse_assets() -> None:
    assert overlay.OFFICIAL_ENV_ASSET_ROOT.exists()

    reference_names = {spec["name"] for spec in overlay.OFFICIAL_BACKGROUND_PROPS}
    assert {"office_file_cabinet", "office_blinds", "warehouse_cardboard_boxes", "warehouse_bottle_reference"} <= reference_names

    for spec in overlay.OFFICIAL_BACKGROUND_PROPS:
        asset_path = overlay.REPO_ROOT / spec["asset"]
        assert asset_path.exists(), f"missing official Isaac asset: {asset_path}"
        assert str(spec["asset"]).startswith("local_eval_assets/nvidia_isaac_5_1_environments/")


def test_visible_lab_furniture_adds_office_depth_without_covering_aloha() -> None:
    names = {spec["name"] for spec in overlay.LAB_BACKGROUND_FURNITURE}
    assert {"rear_low_cabinet", "left_equipment_cabinet", "rear_storage_shelf", "warm_archive_boxes"} <= names
    assert all(spec["translation"][1] < -0.95 for spec in overlay.LAB_BACKGROUND_FURNITURE)
    assert all(max(spec["color"]) < 0.78 for spec in overlay.LAB_BACKGROUND_FURNITURE)
    assert all(spec["collision"] is False for spec in overlay.LAB_BACKGROUND_FURNITURE)

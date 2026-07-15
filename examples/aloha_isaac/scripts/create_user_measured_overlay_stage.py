from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from examples.aloha_isaac.scripts.apply_aloha_black_material import should_apply_robot_material
from examples.aloha_isaac.scripts import create_basic_workcell_stage as basic_stage


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "examples/aloha_isaac/config/workcell_user_measured.yaml"
DEFAULT_BASE_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose.usd"
)
DEFAULT_OUTPUT_USD = (
    REPO_ROOT
    / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
    / "aloha2_menagerie_scene_deep_black_real_start_pose_with_user_table_pipe.usda"
)
OFFICIAL_ENV_ASSET_ROOT = REPO_ROOT / "local_eval_assets/nvidia_isaac_5_1_environments/Isaac/Environments"
OFFICE_ENV_ROOT = "/World/OfficeEnvironment/industrial_lab_corner"
OFFICE_ENV_MATERIAL_ROOT = f"{OFFICE_ENV_ROOT}/Materials"
WALL_DIFFUSE_NEUTRAL = (0.70, 0.71, 0.68)
CEILING_DIFFUSE_NEUTRAL = (0.62, 0.64, 0.61)
FLOOR_DIFFUSE_EPOXY = (0.42, 0.44, 0.43)
WORKBENCH_WARM_MAPLE = (0.66, 0.52, 0.34)
WALL_ROUGHNESS = 0.82
ALOHA_SATIN_BLACK_DIFFUSE = (0.022, 0.021, 0.019)
ALOHA_SATIN_BLACK_ROUGHNESS = 0.42
OFFICE_ROOM_SURFACES = [
    {
        "name": "matte_epoxy_floor",
        "translation": [0.0, 0.15, -0.055],
        "size": [4.80, 4.00, 0.04],
        "color": list(FLOOR_DIFFUSE_EPOXY),
        "material": "floor_matte_epoxy",
        "collision": False,
    },
    {
        "name": "ceiling_baffle",
        "translation": [0.0, 0.15, 2.55],
        "size": [4.80, 4.00, 0.035],
        "color": list(CEILING_DIFFUSE_NEUTRAL),
        "material": "ceiling_neutral",
        "collision": False,
    },
    {
        "name": "rear_acoustic_wall",
        "translation": [0.0, -1.86, 1.22],
        "size": [4.80, 0.035, 2.48],
        "color": list(WALL_DIFFUSE_NEUTRAL),
        "material": "wall_acoustic_neutral",
        "collision": False,
    },
    {
        "name": "left_side_wall",
        "translation": [-2.38, 0.15, 1.22],
        "size": [0.035, 4.00, 2.48],
        "color": list(WALL_DIFFUSE_NEUTRAL),
        "material": "wall_acoustic_neutral",
        "collision": False,
    },
]
OFFICE_WINDOW_PANEL = {
    "name": "right_side_daylight_window",
    "translation": [2.355, 0.15, 1.22],
    "size": [0.018, 2.45, 1.34],
    "color": [0.46, 0.62, 0.82],
    "material": "window_soft_blue",
    "collision": False,
}
WINDOW_DAYLIGHT_DEPTH_LAYERS = [
    {
        "name": "exterior_sky_gradient_top",
        "translation": [2.405, 0.15, 1.68],
        "size": [0.014, 2.25, 0.34],
        "color": [0.56, 0.70, 0.90],
        "material": "exterior_sky_gradient",
        "collision": False,
    },
    {
        "name": "exterior_sunlit_floor_band",
        "translation": [2.415, 0.18, 0.88],
        "size": [0.014, 2.18, 0.28],
        "color": [0.86, 0.80, 0.58],
        "material": "exterior_warm_daylight",
        "collision": False,
    },
    {
        "name": "exterior_soft_shadow_band",
        "translation": [2.425, -0.42, 1.18],
        "size": [0.014, 0.72, 0.94],
        "color": [0.38, 0.48, 0.58],
        "material": "exterior_soft_shadow",
        "collision": False,
    },
]
OFFICE_VERTICAL_BLINDS = [
    {
        "name": f"vertical_blind_{idx + 1:02d}",
        "translation": [2.335, -0.88 + idx * 0.25, 1.22],
        "size": [0.025, 0.018, 1.22],
        "color": [0.78, 0.76, 0.68],
        "material": "blind_tan",
        "collision": False,
    }
    for idx in range(8)
]
OPERATING_SURFACE = {
    "name": "warm_maple_workbench_top",
    "translation": [0.0, 0.0, 0.004],
    "size": [1.16, 0.66, 0.008],
    "color": list(WORKBENCH_WARM_MAPLE),
    "material": "workbench_warm_maple",
    "collision": False,
}
OFFICIAL_BACKGROUND_PROPS = [
    {
        "name": "office_file_cabinet",
        "asset": "local_eval_assets/nvidia_isaac_5_1_environments/Isaac/Environments/Office/Props/SM_FileCabinet_01.usd",
        "translation": [-1.78, -1.60, -0.30],
        "rotation": [0.0, 0.0, 4.0],
        "scale": [0.42, 0.42, 0.42],
    },
    {
        "name": "office_blinds",
        "asset": "local_eval_assets/nvidia_isaac_5_1_environments/Isaac/Environments/Office/Props/SM_BlindsBigOpen.usd",
        "translation": [2.42, -0.45, 1.12],
        "rotation": [0.0, 0.0, 90.0],
        "scale": [0.54, 0.54, 0.54],
    },
    {
        "name": "warehouse_cardboard_boxes",
        "asset": "local_eval_assets/nvidia_isaac_5_1_environments/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
        "translation": [1.62, -1.54, -0.23],
        "rotation": [0.0, 0.0, -12.0],
        "scale": [0.34, 0.34, 0.34],
    },
    {
        "name": "warehouse_bottle_reference",
        "asset": "local_eval_assets/nvidia_isaac_5_1_environments/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_01.usd",
        "translation": [1.18, -1.50, -0.01],
        "rotation": [0.0, 0.0, 20.0],
        "scale": [0.28, 0.28, 0.28],
    },
]
LAB_BACKGROUND_FURNITURE = [
    {
        "name": "rear_low_cabinet",
        "translation": [-0.62, -1.72, 0.30],
        "size": [1.28, 0.18, 0.58],
        "color": [0.36, 0.38, 0.37],
        "material": "cabinet_dark_gray",
        "collision": False,
    },
    {
        "name": "left_equipment_cabinet",
        "translation": [-2.06, -1.12, 0.72],
        "size": [0.34, 0.42, 1.34],
        "color": [0.40, 0.42, 0.41],
        "material": "cabinet_dark_gray",
        "collision": False,
    },
    {
        "name": "rear_storage_shelf",
        "translation": [1.38, -1.72, 0.62],
        "size": [0.95, 0.12, 1.06],
        "color": [0.46, 0.47, 0.45],
        "material": "shelf_powder_coated_metal",
        "collision": False,
    },
    {
        "name": "warm_archive_boxes",
        "translation": [1.30, -1.62, 0.32],
        "size": [0.54, 0.24, 0.32],
        "color": [0.62, 0.46, 0.27],
        "material": "cardboard_warm",
        "collision": False,
    },
]


def _make_wall_grain_marks() -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for idx in range(36):
        x = -2.05 + (idx % 12) * 0.36
        z = 0.32 + (idx // 12) * 0.42
        shade = 0.675 + (idx % 3) * 0.018
        marks.append(
            {
                "name": f"rear_wall_grain_{idx + 1:02d}",
                "translation": [round(x, 3), -1.839, round(z, 3)],
                "size": [0.030, 0.002, 0.018],
                "color": [shade, shade * 1.01, shade * 0.965],
                "material": "wall_grain_matte",
                "collision": False,
            }
        )
    for idx in range(36):
        y = -1.42 + (idx % 12) * 0.28
        z = 0.36 + (idx // 12) * 0.39
        shade = 0.665 + (idx % 4) * 0.014
        marks.append(
            {
                "name": f"left_wall_grain_{idx + 1:02d}",
                "translation": [-2.359, round(y, 3), round(z, 3)],
                "size": [0.002, 0.024, 0.016],
                "color": [shade, shade * 1.01, shade * 0.965],
                "material": "wall_grain_matte",
                "collision": False,
            }
        )
    return marks


WALL_GRAIN_MARKS = _make_wall_grain_marks()
LIGHTING_ROOT = "/World/Lighting/industrial_office_lab"
ALOHA_SATIN_BLACK_MATERIAL = f"{LIGHTING_ROOT}/Materials/aloha_satin_black_for_indoor_light"
CEILING_STRIP_LIGHTS = [
    {
        "name": "ceiling_strip_workbench_1",
        "position": [-0.66, -0.42, 1.86],
        "target": [-0.38, -0.06, 0.18],
        "width": 0.46,
        "height": 0.065,
        "intensity": 360.0,
    },
    {
        "name": "ceiling_strip_workbench_2",
        "position": [-0.22, -0.42, 1.90],
        "target": [-0.13, -0.04, 0.18],
        "width": 0.46,
        "height": 0.065,
        "intensity": 360.0,
    },
    {
        "name": "ceiling_strip_workbench_3",
        "position": [0.22, -0.42, 1.90],
        "target": [0.13, -0.04, 0.18],
        "width": 0.46,
        "height": 0.065,
        "intensity": 360.0,
    },
    {
        "name": "ceiling_strip_workbench_4",
        "position": [0.66, -0.42, 1.86],
        "target": [0.38, -0.06, 0.18],
        "width": 0.46,
        "height": 0.065,
        "intensity": 360.0,
    },
    {
        "name": "ceiling_strip_background_1",
        "position": [-0.72, 0.48, 1.94],
        "target": [-0.38, 0.08, 0.20],
        "width": 0.50,
        "height": 0.070,
        "intensity": 260.0,
    },
    {
        "name": "ceiling_strip_background_2",
        "position": [-0.24, 0.48, 1.98],
        "target": [-0.13, 0.08, 0.20],
        "width": 0.50,
        "height": 0.070,
        "intensity": 260.0,
    },
    {
        "name": "ceiling_strip_background_3",
        "position": [0.24, 0.48, 1.98],
        "target": [0.13, 0.08, 0.20],
        "width": 0.50,
        "height": 0.070,
        "intensity": 260.0,
    },
    {
        "name": "ceiling_strip_background_4",
        "position": [0.72, 0.48, 1.94],
        "target": [0.38, 0.08, 0.20],
        "width": 0.50,
        "height": 0.070,
        "intensity": 260.0,
    },
]
for _ceiling_spec in CEILING_STRIP_LIGHTS:
    _ceiling_spec.setdefault("diffuse", 0.72)
    _ceiling_spec.setdefault("specular", 0.88)
    _ceiling_spec.setdefault("exposure", -0.30)
    _ceiling_spec.setdefault("normalize", True)
WINDOW_KEY_LIGHT = {
    "name": "soft_window_key_from_right",
    "position": [2.05, 0.05, 1.28],
    "target": [0.00, 0.00, 0.28],
    "width": 2.45,
    "height": 1.34,
    "intensity": 1120.0,
    "diffuse": 0.74,
    "specular": 0.92,
    "exposure": -0.25,
    "normalize": True,
}
SURGICAL_SOFTBOX_LIGHT = {
    "name": "large_softbox_over_workbench",
    "position": [0.00, -0.02, 2.20],
    "target": [0.00, 0.00, 0.16],
    "width": 1.65,
    "height": 0.95,
    "intensity": 3400.0,
    "diffuse": 0.68,
    "specular": 1.05,
    "exposure": -0.20,
    "normalize": True,
}
FRONT_FILL_LIGHT = {
    "name": "front_fill_for_black_aloha",
    "position": [0.00, 1.45, 0.95],
    "target": [0.00, 0.00, 0.22],
    "width": 1.70,
    "height": 0.85,
    "intensity": 860.0,
    "diffuse": 0.55,
    "specular": 1.00,
    "exposure": -0.25,
    "normalize": True,
}
ALOHA_BEAUTY_LIGHT_LINK_TARGETS = [
    "/scene/left_base_link",
    "/scene/right_base_link",
    "/World/PipePlaceholder",
]
ALOHA_VIEW_BEAUTY_LIGHT = {
    "name": "camera_angle_aloha_beauty_key",
    "position": [-0.82, 2.18, 1.12],
    "target": [0.00, 0.02, 0.24],
    "width": 0.52,
    "height": 0.34,
    "intensity": 2050.0,
    "color": (0.96, 0.98, 1.0),
    "diffuse": 0.20,
    "specular": 2.45,
    "exposure": 0.35,
    "normalize": True,
    "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
    "light_link_include_root": False,
}
ALOHA_STAGE_SPOT_LIGHTS = [
    {
        "name": "left_front_arm_highlight",
        "position": [-1.05, 1.04, 1.18],
        "target": [-0.34, 0.03, 0.32],
        "width": 0.54,
        "height": 0.42,
        "intensity": 1650.0,
        "color": (1.0, 0.91, 0.80),
        "diffuse": 0.28,
        "specular": 2.20,
        "exposure": 0.20,
        "normalize": True,
        "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
        "light_link_include_root": False,
    },
    {
        "name": "right_front_arm_highlight",
        "position": [1.05, 1.04, 1.18],
        "target": [0.34, 0.03, 0.32],
        "width": 0.54,
        "height": 0.42,
        "intensity": 1650.0,
        "color": (1.0, 0.92, 0.82),
        "diffuse": 0.28,
        "specular": 2.20,
        "exposure": 0.20,
        "normalize": True,
        "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
        "light_link_include_root": False,
    },
    {
        "name": "rear_rim_light",
        "position": [0.00, -1.22, 1.58],
        "target": [0.00, -0.03, 0.48],
        "width": 0.68,
        "height": 0.36,
        "intensity": 1420.0,
        "color": (0.82, 0.91, 1.0),
        "diffuse": 0.18,
        "specular": 2.80,
        "exposure": 0.45,
        "normalize": True,
        "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
        "light_link_include_root": False,
    },
    {
        "name": "pipe_task_highlight",
        "position": [0.72, 0.72, 1.38],
        "target": [0.05, 0.30, 0.32],
        "width": 0.34,
        "height": 0.24,
        "intensity": 1180.0,
        "color": (1.0, 0.96, 0.84),
        "diffuse": 0.34,
        "specular": 1.75,
        "exposure": 0.10,
        "normalize": True,
        "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
        "light_link_include_root": False,
    },
    {
        "name": "low_cross_fill",
        "position": [-0.88, 0.78, 0.58],
        "target": [0.18, -0.02, 0.24],
        "width": 0.78,
        "height": 0.38,
        "intensity": 740.0,
        "color": (0.92, 0.96, 1.0),
        "diffuse": 0.32,
        "specular": 1.40,
        "exposure": -0.10,
        "normalize": True,
        "light_link_targets": ALOHA_BEAUTY_LIGHT_LINK_TARGETS,
        "light_link_include_root": False,
    },
]
BOUNCE_FILL_LIGHTS = [
    {
        "name": "left_cabinet_soft_fill",
        "position": [-1.88, 0.05, 0.92],
        "target": [-0.04, 0.00, 0.24],
        "width": 1.75,
        "height": 1.25,
        "intensity": 650.0,
        "diffuse": 0.64,
        "specular": 0.90,
        "exposure": -0.30,
        "normalize": True,
    },
    {
        "name": "rear_wall_gentle_fill",
        "position": [0.00, -1.46, 1.08],
        "target": [0.00, 0.00, 0.24],
        "width": 2.40,
        "height": 1.35,
        "intensity": 470.0,
        "diffuse": 0.58,
        "specular": 0.82,
        "exposure": -0.35,
        "normalize": True,
    },
]
FILL_DOME_INTENSITY = 105.0


def _require_isaac() -> None:
    basic_stage._require_isaac()


def _define_emissive_material(stage: Any, material_path: str, color: tuple[float, float, float]) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.18)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _define_preview_material(
    stage: Any,
    material_path: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.55,
    emissive: tuple[float, float, float] | None = None,
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _define_satin_black_material(stage: Any, material_path: str) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*ALOHA_SATIN_BLACK_DIFFUSE))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(ALOHA_SATIN_BLACK_ROUGHNESS)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _add_reference_prop(stage: Any, root_path: str, spec: dict[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    asset_path = (REPO_ROOT / spec["asset"]).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"official Isaac environment asset is missing: {asset_path}")
    prim_path = Sdf.Path(f"{root_path}/official_props/{spec['name']}")
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))
    basic_stage._set_xform(prim, spec["translation"], spec["rotation"])
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*spec["scale"]))
    prim.SetCustomDataByKey("official_asset_source", str(asset_path))


def _bind_satin_black_material_to_aloha_visuals(stage: Any, material: Any) -> int:
    from pxr import Gf, UsdGeom

    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        path_lower = path.lower()
        if "/visuals" not in path_lower:
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        if not should_apply_robot_material(path):
            continue
        _bind_material(prim, material)
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*ALOHA_SATIN_BLACK_DIFFUSE)])
            gprim.CreateDisplayOpacityAttr().Set([1.0])
        bound += 1
    return bound


def add_lightweight_office_environment(stage: Any) -> None:
    """Add non-physical room context so lighting has surfaces to bounce from."""
    from pxr import Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, Sdf.Path("/World/OfficeEnvironment"))
    UsdGeom.Xform.Define(stage, Sdf.Path(OFFICE_ENV_ROOT))
    UsdGeom.Xform.Define(stage, Sdf.Path(OFFICE_ENV_MATERIAL_ROOT))
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{OFFICE_ENV_ROOT}/official_props"))

    materials = {
        "wall_acoustic_neutral": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/wall_acoustic_neutral",
            WALL_DIFFUSE_NEUTRAL,
            roughness=WALL_ROUGHNESS,
        ),
        "ceiling_neutral": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/ceiling_neutral",
            CEILING_DIFFUSE_NEUTRAL,
            roughness=0.78,
        ),
        "floor_matte_epoxy": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/floor_matte_epoxy",
            FLOOR_DIFFUSE_EPOXY,
            roughness=0.70,
        ),
        "window_soft_blue": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/window_soft_blue",
            (0.46, 0.62, 0.82),
            roughness=0.22,
            emissive=(0.08, 0.13, 0.20),
        ),
        "exterior_sky_gradient": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/exterior_sky_gradient",
            (0.56, 0.70, 0.90),
            roughness=0.38,
            emissive=(0.20, 0.27, 0.36),
        ),
        "exterior_warm_daylight": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/exterior_warm_daylight",
            (0.86, 0.80, 0.58),
            roughness=0.52,
            emissive=(0.32, 0.25, 0.13),
        ),
        "exterior_soft_shadow": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/exterior_soft_shadow",
            (0.38, 0.48, 0.58),
            roughness=0.68,
            emissive=(0.08, 0.10, 0.13),
        ),
        "blind_tan": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/blind_tan",
            (0.78, 0.76, 0.68),
            roughness=0.72,
        ),
        "workbench_warm_maple": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/workbench_warm_maple",
            WORKBENCH_WARM_MAPLE,
            roughness=0.50,
        ),
        "wall_grain_matte": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/wall_grain_matte",
            (0.68, 0.69, 0.66),
            roughness=0.92,
        ),
        "cabinet_dark_gray": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/cabinet_dark_gray",
            (0.36, 0.38, 0.37),
            roughness=0.66,
        ),
        "shelf_powder_coated_metal": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/shelf_powder_coated_metal",
            (0.46, 0.47, 0.45),
            roughness=0.54,
        ),
        "cardboard_warm": _define_preview_material(
            stage,
            f"{OFFICE_ENV_MATERIAL_ROOT}/cardboard_warm",
            (0.62, 0.46, 0.27),
            roughness=0.86,
        ),
    }

    for spec in OFFICE_ROOM_SURFACES:
        path = f"{OFFICE_ENV_ROOT}/{spec['name']}"
        basic_stage._add_cube(
            stage,
            path,
            spec["translation"],
            spec["size"],
            spec["color"],
            collision=bool(spec["collision"]),
        )
        _bind_material(stage.GetPrimAtPath(path), materials[spec["material"]])

    for spec in WALL_GRAIN_MARKS:
        path = f"{OFFICE_ENV_ROOT}/{spec['name']}"
        basic_stage._add_cube(
            stage,
            path,
            spec["translation"],
            spec["size"],
            spec["color"],
            collision=bool(spec["collision"]),
        )
        _bind_material(stage.GetPrimAtPath(path), materials[spec["material"]])

    for spec in LAB_BACKGROUND_FURNITURE:
        path = f"{OFFICE_ENV_ROOT}/{spec['name']}"
        basic_stage._add_cube(
            stage,
            path,
            spec["translation"],
            spec["size"],
            spec["color"],
            collision=bool(spec["collision"]),
        )
        _bind_material(stage.GetPrimAtPath(path), materials[spec["material"]])

    operating_surface_path = f"{OFFICE_ENV_ROOT}/{OPERATING_SURFACE['name']}"
    basic_stage._add_cube(
        stage,
        operating_surface_path,
        OPERATING_SURFACE["translation"],
        OPERATING_SURFACE["size"],
        OPERATING_SURFACE["color"],
        collision=bool(OPERATING_SURFACE["collision"]),
    )
    _bind_material(stage.GetPrimAtPath(operating_surface_path), materials[OPERATING_SURFACE["material"]])

    window_path = f"{OFFICE_ENV_ROOT}/{OFFICE_WINDOW_PANEL['name']}"
    basic_stage._add_cube(
        stage,
        window_path,
        OFFICE_WINDOW_PANEL["translation"],
        OFFICE_WINDOW_PANEL["size"],
        OFFICE_WINDOW_PANEL["color"],
        collision=bool(OFFICE_WINDOW_PANEL["collision"]),
    )
    _bind_material(stage.GetPrimAtPath(window_path), materials[OFFICE_WINDOW_PANEL["material"]])

    for spec in WINDOW_DAYLIGHT_DEPTH_LAYERS:
        path = f"{OFFICE_ENV_ROOT}/{spec['name']}"
        basic_stage._add_cube(
            stage,
            path,
            spec["translation"],
            spec["size"],
            spec["color"],
            collision=bool(spec["collision"]),
        )
        _bind_material(stage.GetPrimAtPath(path), materials[spec["material"]])

    for spec in OFFICE_VERTICAL_BLINDS:
        path = f"{OFFICE_ENV_ROOT}/{spec['name']}"
        basic_stage._add_cube(
            stage,
            path,
            spec["translation"],
            spec["size"],
            spec["color"],
            collision=bool(spec["collision"]),
        )
        _bind_material(stage.GetPrimAtPath(path), materials[spec["material"]])

    for spec in OFFICIAL_BACKGROUND_PROPS:
        _add_reference_prop(stage, OFFICE_ENV_ROOT, spec)

    root = stage.GetPrimAtPath(OFFICE_ENV_ROOT)
    root.SetCustomDataByKey("environment_intent", "industrial office lab corner with official Isaac props")
    root.SetCustomDataByKey("startup_camera_side", "open_positive_y")
    root.SetCustomDataByKey("reference_photo", "/home/eii/Downloads/iphone/IMG_5334.JPG")
    root.SetCustomDataByKey("official_asset_root", str(OFFICIAL_ENV_ASSET_ROOT))
    root.SetCustomDataByKey("window_depth_layer_count", len(WINDOW_DAYLIGHT_DEPTH_LAYERS))


def _set_look_at_transform(prim: Any, position: list[float], target: list[float]) -> None:
    from pxr import Gf, UsdGeom

    eye = Gf.Vec3d(*position)
    look_at = Gf.Vec3d(*target)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    matrix = Gf.Matrix4d().SetLookAt(eye, look_at, up).GetInverse()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)


def _add_rect_light(stage: Any, root_path: str, spec: dict[str, Any], color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdLux

    light = UsdLux.RectLight.Define(stage, Sdf.Path(f"{root_path}/{spec['name']}"))
    light.CreateWidthAttr(float(spec["width"]))
    light.CreateHeightAttr(float(spec["height"]))
    light.CreateIntensityAttr(float(spec["intensity"]))
    light.CreateColorAttr(Gf.Vec3f(*color))
    light.CreateDiffuseAttr(float(spec.get("diffuse", 1.0)))
    light.CreateSpecularAttr(float(spec.get("specular", 1.0)))
    light.CreateExposureAttr(float(spec.get("exposure", 0.0)))
    light.CreateNormalizeAttr(bool(spec.get("normalize", False)))
    if "light_link_targets" in spec:
        prim = light.GetPrim()
        prim.SetCustomDataByKey("intended_light_link_targets", ", ".join(spec["light_link_targets"]))
        prim.SetCustomDataByKey("intended_light_link_include_root", bool(spec.get("light_link_include_root", False)))
    _set_look_at_transform(light.GetPrim(), spec["position"], spec["target"])


def _add_luminous_strip(stage: Any, root_path: str, spec: dict[str, Any], material: Any) -> None:
    from pxr import Gf, Sdf, UsdGeom

    panel = UsdGeom.Cube.Define(stage, Sdf.Path(f"{root_path}/{spec['name']}_visible_panel"))
    panel.CreateSizeAttr(1.0)
    panel.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.96, 0.86)])
    basic_stage._set_xform(panel.GetPrim(), spec["position"], [0.0, 0.0, 0.0])
    panel.AddScaleOp().Set(Gf.Vec3f(float(spec["width"]), float(spec["height"]), 0.018))
    _bind_material(panel.GetPrim(), material)


def add_indoor_photo_lighting(stage: Any) -> None:
    """Add a lighting rig modeled on the user's indoor ALOHA room photo."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    UsdGeom.Xform.Define(stage, Sdf.Path(LIGHTING_ROOT))
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{LIGHTING_ROOT}/Materials"))
    material = _define_emissive_material(stage, f"{LIGHTING_ROOT}/Materials/warm_fluorescent_panel", (1.0, 0.94, 0.78))
    satin_black = _define_satin_black_material(stage, ALOHA_SATIN_BLACK_MATERIAL)

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{LIGHTING_ROOT}/ambient_dome"))
    dome.CreateIntensityAttr(FILL_DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.83, 0.88))

    _add_rect_light(stage, LIGHTING_ROOT, WINDOW_KEY_LIGHT, (0.86, 0.94, 1.0))
    _add_rect_light(stage, LIGHTING_ROOT, SURGICAL_SOFTBOX_LIGHT, (1.0, 0.98, 0.92))
    _add_luminous_strip(stage, LIGHTING_ROOT, SURGICAL_SOFTBOX_LIGHT, material)
    _add_rect_light(stage, LIGHTING_ROOT, FRONT_FILL_LIGHT, (1.0, 0.97, 0.90))
    _add_rect_light(stage, LIGHTING_ROOT, ALOHA_VIEW_BEAUTY_LIGHT, ALOHA_VIEW_BEAUTY_LIGHT["color"])
    for spec in ALOHA_STAGE_SPOT_LIGHTS:
        _add_rect_light(stage, LIGHTING_ROOT, spec, spec["color"])
    for spec in BOUNCE_FILL_LIGHTS:
        _add_rect_light(stage, LIGHTING_ROOT, spec, (1.0, 0.98, 0.93))
    for spec in CEILING_STRIP_LIGHTS:
        _add_rect_light(stage, LIGHTING_ROOT, spec, (1.0, 0.96, 0.88))
        _add_luminous_strip(stage, LIGHTING_ROOT, spec, material)

    bound_visuals = _bind_satin_black_material_to_aloha_visuals(stage, satin_black)
    root = stage.GetPrimAtPath(LIGHTING_ROOT)
    root.SetCustomDataByKey("lighting_reference_photo", "/home/eii/Downloads/iphone/IMG_5334.JPG")
    root.SetCustomDataByKey(
        "lighting_intent",
        "layered window light plus stage-style multi-angle highlights for black ALOHA",
    )
    root.SetCustomDataByKey("stage_spot_light_count", len(ALOHA_STAGE_SPOT_LIGHTS) + 1)
    root.SetCustomDataByKey("aloha_light_link_targets", ", ".join(ALOHA_BEAUTY_LIGHT_LINK_TARGETS))
    root.SetCustomDataByKey("satin_black_visuals_bound", bound_visuals)


def build_overlay_stage(config_path: Path, base_usd: Path, output_usd: Path, headless: bool = True) -> Path:
    """Create a non-destructive USD layer over the confirmed ALOHA scene."""
    _require_isaac()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})
    try:
        from pxr import Sdf, Usd, UsdGeom

        cfg = basic_stage._load_config(config_path)
        base_usd = base_usd.resolve()
        output_usd = output_usd.resolve()
        if not base_usd.exists():
            raise FileNotFoundError(f"base ALOHA USD does not exist: {base_usd}")
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        if output_usd.exists():
            output_usd.unlink()

        stage = Usd.Stage.CreateNew(str(output_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.GetRootLayer().subLayerPaths.append(os.path.relpath(base_usd, output_usd.parent))

        # Keep measured workcell objects under /World so the confirmed /scene
        # ALOHA asset remains untouched and easy to compare against.
        UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        table_cfg = cfg["table"]
        basic_stage._add_cube(
            stage,
            table_cfg["prim_path"],
            table_cfg["pose"]["translation"],
            table_cfg["size"],
            table_cfg["color"],
            collision=True,
        )
        basic_stage._add_table_frame(stage, cfg["table_frame"])
        basic_stage._add_pipe_support_and_axis(stage, cfg, None)
        add_lightweight_office_environment(stage)
        add_indoor_photo_lighting(stage)
        for name, camera_cfg in cfg["cameras"].items():
            basic_stage._add_camera(stage, name, camera_cfg)

        root = stage.GetPrimAtPath("/World")
        root.SetCustomDataByKey("overlay_base_usd", str(base_usd))
        root.SetCustomDataByKey("overlay_config", str(config_path.resolve()))
        root.SetCustomDataByKey("overlay_scope", "user_measured_table_pipe_over_confirmed_aloha")
        stage.GetRootLayer().Save()
        return output_usd
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a measured table/pipe overlay over the confirmed ALOHA Isaac startup stage."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-usd", type=Path, default=DEFAULT_BASE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    output = build_overlay_stage(args.config, args.base_usd, args.output_usd, headless=args.headless)
    print(f"usd={output}")


if __name__ == "__main__":
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    main()

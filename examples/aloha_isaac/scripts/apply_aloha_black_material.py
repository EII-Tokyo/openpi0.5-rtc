from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_USD = REPO_ROOT / "local_eval_assets/aloha_isaac_menagerie/aloha2_menagerie_scene.usd"
DEFAULT_OUTPUT_USD = REPO_ROOT / "local_eval_assets/aloha_isaac_menagerie_black/aloha2_menagerie_scene_black.usd"
DEFAULT_KEYFRAME_XML = REPO_ROOT / "external/mujoco_menagerie/aloha/keyframe_ctrl.xml"

ALOHA_NEUTRAL_JOINT_PATHS = (
    ("/scene/joints/left_waist", "angular"),
    ("/scene/joints/left_shoulder", "angular"),
    ("/scene/joints/left_elbow", "angular"),
    ("/scene/joints/left_forearm_roll", "angular"),
    ("/scene/joints/left_wrist_angle", "angular"),
    ("/scene/joints/left_wrist_rotate", "angular"),
    ("/scene/joints/left_left_finger", "linear"),
    ("/scene/joints/left_right_finger", "linear"),
    ("/scene/joints/right_waist", "angular"),
    ("/scene/joints/right_shoulder", "angular"),
    ("/scene/joints/right_elbow", "angular"),
    ("/scene/joints/right_forearm_roll", "angular"),
    ("/scene/joints/right_wrist_angle", "angular"),
    ("/scene/joints/right_wrist_rotate", "angular"),
    ("/scene/joints/right_left_finger", "linear"),
    ("/scene/joints/right_right_finger", "linear"),
)


def should_apply_robot_material(prim_path: str) -> bool:
    """Return True for ALOHA follower arm visual meshes, but not cameras/table/frame."""
    path = prim_path.lower()
    if not ("/left_" in path or "/right_" in path):
        return False
    excluded = (
        "camera",
        "d405",
        "table",
        "frame",
        "floor",
        "mount",
        "extrusion",
        "bracket",
        "collision",
        "sites",
    )
    return not any(token in path for token in excluded)


def should_apply_robot_prototype_material(prim_path: str) -> bool:
    path = prim_path.lower()
    if "vx300s" not in path:
        return False
    excluded = ("d405", "camera", "collision")
    return not any(token in path for token in excluded)


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def _prepare_output_bundle(source_usd: Path, output_usd: Path) -> Path:
    if source_usd.resolve() == output_usd.resolve():
        return output_usd

    source_dir = source_usd.parent
    output_dir = output_usd.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != output_dir.resolve():
        shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    copied_source = output_dir / source_usd.name
    if copied_source.resolve() != output_usd.resolve():
        shutil.copy2(copied_source, output_usd)
    return output_usd


def _bind_robot_visuals_to_material(stage, material_path) -> int:
    from pxr import Gf, UsdGeom, UsdShade

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise RuntimeError(f"stage is missing expected black material: {material_path}")
    material = UsdShade.Material(material_prim)

    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        path_lower = path.lower()
        if not prim.IsA(UsdGeom.Imageable):
            continue
        if "/visuals" not in path_lower:
            continue
        if not should_apply_robot_material(path):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.005, 0.005, 0.005)])
            gprim.CreateDisplayOpacityAttr().Set([1.0])
        bound += 1

    return bound


def _define_robot_black_material(stage, material_path):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.005, 0.005, 0.005))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def read_neutral_qpos(keyframe_xml: Path) -> tuple[float, ...]:
    root = ET.parse(keyframe_xml).getroot()
    for key in root.findall(".//key"):
        if key.get("name") == "neutral_pose":
            qpos = key.get("qpos")
            if not qpos:
                raise ValueError(f"neutral_pose in {keyframe_xml} does not define qpos")
            values = tuple(float(part) for part in qpos.split())
            if len(values) != len(ALOHA_NEUTRAL_JOINT_PATHS):
                raise ValueError(
                    f"neutral_pose qpos has {len(values)} values; expected {len(ALOHA_NEUTRAL_JOINT_PATHS)}"
                )
            return values
    raise ValueError(f"{keyframe_xml} does not contain keyframe named neutral_pose")


def qpos_to_usd_joint_positions(qpos: tuple[float, ...]) -> tuple[float, ...]:
    """Convert MJCF qpos into USD joint position units: degrees for revolute, meters for prismatic."""
    converted: list[float] = []
    for value, (_, drive_type) in zip(qpos, ALOHA_NEUTRAL_JOINT_PATHS, strict=True):
        if drive_type == "angular":
            converted.append(math.degrees(value))
        elif drive_type == "linear":
            converted.append(value)
        else:
            raise ValueError(f"unsupported drive type: {drive_type}")
    return tuple(converted)


def _apply_neutral_pose(stage, neutral_qpos: tuple[float, ...]) -> int:
    from pxr import PhysxSchema, UsdPhysics

    positions = qpos_to_usd_joint_positions(neutral_qpos)
    applied = 0
    for (joint_path, drive_type), position in zip(ALOHA_NEUTRAL_JOINT_PATHS, positions, strict=True):
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            raise RuntimeError(f"stage is missing expected ALOHA joint: {joint_path}")

        if drive_type == "angular":
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            joint_state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
        else:
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            joint_state = PhysxSchema.JointStateAPI.Apply(prim, "linear")

        drive.CreateTargetPositionAttr().Set(position)
        drive.CreateTargetVelocityAttr().Set(0.0)
        joint_state.CreatePositionAttr().Set(position)
        joint_state.CreateVelocityAttr().Set(0.0)
        applied += 1
    return applied


def apply_black_material(source_usd: Path, output_usd: Path, neutral_keyframe_xml: Path | None = DEFAULT_KEYFRAME_XML) -> tuple[int, int]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    if not source_usd.exists():
        raise FileNotFoundError(f"source USD does not exist: {source_usd}")

    output_usd = _prepare_output_bundle(source_usd, output_usd)

    stage = Usd.Stage.Open(str(output_usd))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {output_usd}")

    material_path = Sdf.Path("/scene/Looks/material_robot_deep_black")
    _define_robot_black_material(stage, str(material_path))
    bound = _bind_robot_visuals_to_material(stage, material_path)
    neutral_applied = 0
    neutral_qpos = None
    if neutral_keyframe_xml is not None:
        if not neutral_keyframe_xml.exists():
            raise FileNotFoundError(f"neutral keyframe XML does not exist: {neutral_keyframe_xml}")
        neutral_qpos = read_neutral_qpos(neutral_keyframe_xml)
        neutral_applied = _apply_neutral_pose(stage, neutral_qpos)

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/scene/Looks/aloha_view_dome_light"))
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/scene/Looks/aloha_view_key_light"))
    sun.CreateIntensityAttr(650.0)
    sun.CreateAngleAttr(0.35)
    xform = UsdGeom.Xformable(sun.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    stage.GetRootLayer().Save()

    base_layer = output_usd.parent / "configuration" / "aloha2_menagerie_scene_base.usd"
    if base_layer.exists():
        base_stage = Usd.Stage.Open(str(base_layer))
        if base_stage is None:
            raise RuntimeError(f"failed to open base layer: {base_layer}")
        _define_robot_black_material(base_stage, str(material_path))
        bound += _bind_robot_visuals_to_material(base_stage, material_path)
        if neutral_qpos is not None:
            neutral_applied += _apply_neutral_pose(base_stage, neutral_qpos)
        base_stage.GetRootLayer().Save()

    return bound, neutral_applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind a real-ALOHA black material to converted ALOHA arm meshes.")
    parser.add_argument("--source-usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--output-usd", type=Path, default=DEFAULT_OUTPUT_USD)
    parser.add_argument("--neutral-keyframe-xml", type=Path, default=DEFAULT_KEYFRAME_XML)
    parser.add_argument(
        "--skip-neutral-pose",
        action="store_true",
        help="Only restore the black material, leaving imported joint targets unchanged.",
    )
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = _start_isaac_headless()
    try:
        bound, neutral_applied = apply_black_material(
            args.source_usd.resolve(),
            args.output_usd.resolve(),
            None if args.skip_neutral_pose else args.neutral_keyframe_xml.resolve() if args.neutral_keyframe_xml else None,
        )
        print(f"bound_black_material_meshes={bound}")
        print(f"neutral_pose_joints={neutral_applied}")
        print(f"output_usd={args.output_usd.resolve()}")
        if bound == 0:
            raise RuntimeError("no robot meshes were matched; refusing to produce an apparently unchanged stage")
        if args.neutral_keyframe_xml and neutral_applied == 0:
            raise RuntimeError("neutral keyframe was provided but no joints were updated")
    finally:
        app.close()


if __name__ == "__main__":
    main()

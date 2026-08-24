from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SCRIPT = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT.parents[1]
MASS_KG = 0.004
OUTER_RADIUS_M = 0.017
INNER_RADIUS_M = 0.0154
HEIGHT_M = 0.022
TOP_THICKNESS_M = 0.002
SIDE_COLLIDER_COUNT = 16
VISUAL_RIB_COUNT = 32
CAP_STATIC_FRICTION = 0.90
CAP_DYNAMIC_FRICTION = 0.75


def read_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            points.append(Gf.Vec3f(*(float(value) / 1000.0 for value in fields[1:4])))
        elif fields[0] == "f" and len(fields) >= 4:
            face = [int(value.split("/")[0]) - 1 for value in fields[1:]]
            counts.append(len(face))
            indices.extend(face)
    if not points or not counts:
        raise RuntimeError(f"OBJ contains no usable mesh: {path}")
    return points, counts, indices


def bind_physics_material(prim: Usd.Prim, material_path: Sdf.Path) -> None:
    prim.CreateRelationship("material:binding:physics").SetTargets([material_path])


def append_api_schema(prim: Usd.Prim, schema_name: str) -> None:
    schemas = list(prim.GetAppliedSchemas())
    if schema_name not in schemas:
        schemas.append(schema_name)
    operation = Sdf.TokenListOp()
    operation.explicitItems = schemas
    prim.SetMetadata("apiSchemas", operation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root_dir = args.asset_root.resolve()
    obj_path = root_dir / "geometry/visual/bottle_cap_visual.obj"
    output_path = root_dir / "usd/bottle_cap_diagnostic_v1.usda"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points, counts, indices = read_obj(obj_path)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/BottleCap")
    stage.SetDefaultPrim(root.GetPrim())

    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(MASS_KG)
    com_z = 0.011
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, com_z))
    radial_sum = OUTER_RADIUS_M**2 + INNER_RADIUS_M**2
    inertia_z = 0.5 * MASS_KG * radial_sum
    inertia_xy = MASS_KG * (3.0 * radial_sum + HEIGHT_M**2) / 12.0
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_xy, inertia_xy, inertia_z))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    append_api_schema(root.GetPrim(), "PhysxRigidBodyAPI")
    root.GetPrim().CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    root.GetPrim().CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.05)
    root.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.05)

    visuals = UsdGeom.Xform.Define(stage, "/BottleCap/Visuals")
    mesh = UsdGeom.Mesh.Define(stage, "/BottleCap/Visuals/VIS_BottleCap")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(-OUTER_RADIUS_M, -OUTER_RADIUS_M, 0.0), Gf.Vec3f(OUTER_RADIUS_M, OUTER_RADIUS_M, HEIGHT_M)])

    visual_material = UsdShade.Material.Define(stage, "/BottleCap/Looks/MAT_Cap_DeepBlue_Plastic")
    shader = UsdShade.Shader.Define(stage, "/BottleCap/Looks/MAT_Cap_DeepBlue_Plastic/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.025, 0.09, 0.32))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.30)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.18)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    visual_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual_material)

    rib_material = UsdShade.Material.Define(stage, "/BottleCap/Looks/MAT_Cap_Rib_Highlight")
    rib_shader = UsdShade.Shader.Define(stage, "/BottleCap/Looks/MAT_Cap_Rib_Highlight/PreviewSurface")
    rib_shader.CreateIdAttr("UsdPreviewSurface")
    rib_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.07, 0.32, 0.95))
    rib_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    rib_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rib_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.20)
    rib_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    rib_material.CreateSurfaceOutput().ConnectToSource(rib_shader.ConnectableAPI(), "surface")

    rib_depth = 0.001
    rib_width = 0.0015
    rib_height = 0.0175
    rib_center_radius = OUTER_RADIUS_M - rib_depth / 2.0
    for index in range(VISUAL_RIB_COUNT):
        angle = 2.0 * math.pi * index / VISUAL_RIB_COUNT
        rib = UsdGeom.Cube.Define(stage, f"/BottleCap/Visuals/Ribs/VIS_Rib_{index:02d}")
        rib.CreateSizeAttr(1.0)
        rib.AddTranslateOp().Set(
            Gf.Vec3d(
                rib_center_radius * math.cos(angle),
                rib_center_radius * math.sin(angle),
                0.01075,
            )
        )
        rib.AddRotateZOp().Set(math.degrees(angle))
        rib.AddScaleOp().Set(Gf.Vec3f(rib_depth, rib_width, rib_height))
        UsdShade.MaterialBindingAPI.Apply(rib.GetPrim()).Bind(rib_material)

    top_material = UsdShade.Material.Define(stage, "/BottleCap/Looks/MAT_Cap_Top_Accent")
    top_shader = UsdShade.Shader.Define(stage, "/BottleCap/Looks/MAT_Cap_Top_Accent/PreviewSurface")
    top_shader.CreateIdAttr("UsdPreviewSurface")
    top_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.10, 0.42, 1.0))
    top_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.24)
    top_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    top_shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.45)
    top_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    top_material.CreateSurfaceOutput().ConnectToSource(top_shader.ConnectableAPI(), "surface")
    top_accent = UsdGeom.Cylinder.Define(stage, "/BottleCap/Visuals/VIS_TopAccent")
    top_accent.CreateAxisAttr(UsdGeom.Tokens.z)
    top_accent.CreateRadiusAttr(0.0125)
    top_accent.CreateHeightAttr(0.00025)
    top_accent.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.022125))
    UsdShade.MaterialBindingAPI.Apply(top_accent.GetPrim()).Bind(top_material)

    physics_material = UsdShade.Material.Define(stage, "/BottleCap/Looks/MAT_Cap_Physics_TEMP")
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(CAP_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr(CAP_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr(0.0)
    append_api_schema(physics_material.GetPrim(), "PhysxMaterialAPI")
    physics_material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    physics_material.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")

    collisions = UsdGeom.Xform.Define(stage, "/BottleCap/Collisions")
    radial_thickness = 0.0012
    ring_center_radius = OUTER_RADIUS_M - radial_thickness / 2.0
    tangential_width = 2.0 * ring_center_radius * math.tan(math.pi / SIDE_COLLIDER_COUNT) * 0.98
    side_height = HEIGHT_M - TOP_THICKNESS_M
    for index in range(SIDE_COLLIDER_COUNT):
        angle = 2.0 * math.pi * index / SIDE_COLLIDER_COUNT
        cube = UsdGeom.Cube.Define(stage, f"/BottleCap/Collisions/COL_Side_{index:02d}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(ring_center_radius * math.cos(angle), ring_center_radius * math.sin(angle), side_height / 2.0))
        cube.AddRotateZOp().Set(math.degrees(angle))
        cube.AddScaleOp().Set(Gf.Vec3f(radial_thickness, tangential_width, side_height))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        bind_physics_material(cube.GetPrim(), physics_material.GetPath())

    top = UsdGeom.Cylinder.Define(stage, "/BottleCap/Collisions/COL_Top")
    top.CreateAxisAttr(UsdGeom.Tokens.z)
    top.CreateRadiusAttr(OUTER_RADIUS_M)
    top.CreateHeightAttr(TOP_THICKNESS_M)
    top.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, HEIGHT_M - TOP_THICKNESS_M / 2.0))
    UsdPhysics.CollisionAPI.Apply(top.GetPrim()).CreateCollisionEnabledAttr(True)
    bind_physics_material(top.GetPrim(), physics_material.GetPath())

    mount = UsdGeom.Xform.Define(stage, "/BottleCap/Frames/CapAxisFrame")
    mount.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, HEIGHT_M / 2.0))
    root.GetPrim().SetCustomDataByKey("assetStatus", "TEMPORARY_UNCALIBRATED")
    root.GetPrim().SetCustomDataByKey("sourceBottleThreadOuterDiameterMm", 30.0)
    root.GetPrim().SetCustomDataByKey("capInnerDiameterMm", 30.8)
    root.GetPrim().SetCustomDataByKey("massStatus", "TEMPORARY_UNCALIBRATED")
    root.GetPrim().SetCustomDataByKey("frictionStatus", "TEMPORARY_UNCALIBRATED")
    root.GetPrim().SetCustomDataByKey("visualStyle", "DEEP_BLUE_PLASTIC_32_GRIP_RIBS_TOP_ACCENT")
    stage.GetRootLayer().Save()
    print(json.dumps({"status": "PASS", "usd": str(output_path), "points": len(points), "faces": len(counts), "collision_count": SIDE_COLLIDER_COUNT + 1, "visual_rib_count": VISUAL_RIB_COUNT, "visual_style": "DEEP_BLUE_PLASTIC"}, indent=2))


if __name__ == "__main__":
    main()

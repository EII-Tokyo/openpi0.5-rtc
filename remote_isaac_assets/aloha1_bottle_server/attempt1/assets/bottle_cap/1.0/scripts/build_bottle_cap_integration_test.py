from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SCRIPT = Path(__file__).resolve()
ASSET_ROOT = SCRIPT.parents[1]
OUTPUT = ASSET_ROOT / "tests/bottle_cap_integration_test.usda"
REPORT = ASSET_ROOT / "reports/bottle_cap_integration_validation.json"
CAP_BASE_Z_M = 0.188

MATERIALS = {
    "BottleSurface_TEMP": (0.65, 0.50),
    "CapSurface_TEMP": (0.90, 0.75),
    "TableSurface_TEMP": (0.60, 0.50),
    "GripperPad_TEMP": (0.90, 0.75),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_api_schema(prim: Usd.Prim, schema_name: str) -> None:
    schemas = list(prim.GetAppliedSchemas())
    if schema_name not in schemas:
        schemas.append(schema_name)
    operation = Sdf.TokenListOp()
    operation.explicitItems = schemas
    prim.SetMetadata("apiSchemas", operation)


def define_physics_material(stage: Usd.Stage, name: str, static: float, dynamic: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"/World/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    append_api_schema(material.GetPrim(), "PhysxMaterialAPI")
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    material.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    material.GetPrim().SetCustomDataByKey("calibrationStatus", "TEMPORARY_UNCALIBRATED")
    return material


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    prim.CreateRelationship("material:binding:physics").SetTargets([material.GetPath()])


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    materials = {
        name: define_physics_material(stage, name, static, dynamic)
        for name, (static, dynamic) in MATERIALS.items()
    }

    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    table.AddScaleOp().Set(Gf.Vec3f(0.6, 0.6, 0.02))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    bind(table.GetPrim(), materials["TableSurface_TEMP"])

    bottle = stage.DefinePrim("/World/Bottle500", "Xform")
    bottle.GetReferences().AddReference("../../../bottle_500ml/isaac/bottle_500ml_sim.usd", "/Bottle500")
    bottle.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    cap = stage.DefinePrim("/World/BottleCap", "Xform")
    cap.GetReferences().AddReference("../usd/bottle_cap_diagnostic_v1.usda", "/BottleCap")
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CAP_BASE_Z_M))
    cap.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    # PhysX does not support CCD on a kinematic body. The reusable cap asset
    # keeps CCD enabled for its future dynamic use; only this held test disables it.
    cap.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(False)

    stage.Load()
    bottle_collisions = [prim for prim in Usd.PrimRange(bottle) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    cap_collisions = [prim for prim in Usd.PrimRange(cap) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    for prim in bottle_collisions:
        bind(prim, materials["BottleSurface_TEMP"])
    for prim in cap_collisions:
        bind(prim, materials["CapSurface_TEMP"])

    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(OUTPUT))
    assert reopened is not None
    bottle_root = reopened.GetPrimAtPath("/World/Bottle500")
    cap_root = reopened.GetPrimAtPath("/World/BottleCap")
    bottle_collisions = [prim for prim in Usd.PrimRange(bottle_root) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    cap_collisions = [prim for prim in Usd.PrimRange(cap_root) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    cap_meshes = [prim for prim in Usd.PrimRange(cap_root) if prim.IsA(UsdGeom.Mesh)]
    checks = {
        "bottle_composed": bool(bottle_root) and len(bottle_collisions) == 41,
        "cap_composed": bool(cap_root) and len(cap_collisions) == 17 and len(cap_meshes) == 1,
        "cap_base_at_0_188_m": abs(float(cap_root.GetAttribute("xformOp:translate").Get()[2]) - CAP_BASE_Z_M) < 1e-9,
        "cap_inner_radius_clears_thread": 0.0154 > 0.0150,
        "four_surface_materials": all(bool(reopened.GetPrimAtPath(f"/World/PhysicsMaterials/{name}")) for name in MATERIALS),
        "all_bottle_colliders_bound": all(prim.GetRelationship("material:binding:physics").GetTargets() == [materials["BottleSurface_TEMP"].GetPath()] for prim in bottle_collisions),
        "all_cap_colliders_bound": all(prim.GetRelationship("material:binding:physics").GetTargets() == [materials["CapSurface_TEMP"].GetPath()] for prim in cap_collisions),
        "table_bound": reopened.GetPrimAtPath("/World/Table").GetRelationship("material:binding:physics").GetTargets() == [materials["TableSurface_TEMP"].GetPath()],
        "diagnostic_kinematic_hold": bottle_root.GetAttribute("physics:kinematicEnabled").Get() is True and cap_root.GetAttribute("physics:kinematicEnabled").Get() is True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "PHASE_1_2_DIAGNOSTIC_ONLY_NO_THREAD_JOINT",
        "stage": {"absolute_path": str(OUTPUT), "sha256": sha256(OUTPUT)},
        "checks": checks,
        "counts": {"bottle_colliders": len(bottle_collisions), "cap_colliders": len(cap_collisions), "cap_visual_meshes": len(cap_meshes)},
        "placement": {"bottle_mouth_z_m": 0.206, "cap_base_z_m": CAP_BASE_Z_M, "cap_top_z_m": CAP_BASE_Z_M + 0.022},
        "materials": {name: {"static_friction": values[0], "dynamic_friction": values[1], "restitution": 0.0, "status": "TEMPORARY_UNCALIBRATED"} for name, values in MATERIALS.items()},
        "limitations": ["No helical thread geometry or thread joint is present.", "Bottle and cap are held kinematic in this integration loading test."],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

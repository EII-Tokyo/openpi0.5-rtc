from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


SCRIPT = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT.parents[1]


def authored_api_schemas(prim: Usd.Prim) -> list[str]:
    operation = prim.GetMetadata("apiSchemas")
    return list(operation.explicitItems) if operation is not None else []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root_dir = args.asset_root.resolve()
    usd_path = root_dir / "usd/bottle_cap_diagnostic_v1.usda"
    report_path = root_dir / "reports/bottle_cap_asset_validation.json"
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open {usd_path}")
    root = stage.GetPrimAtPath("/BottleCap")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = bbox_cache.ComputeWorldBound(root).ComputeAlignedRange()
    size = bounds.GetSize()
    collision_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    rib_prims = [prim for prim in stage.Traverse() if str(prim.GetPath()).startswith("/BottleCap/Visuals/Ribs/VIS_Rib_")]
    top_accent = stage.GetPrimAtPath("/BottleCap/Visuals/VIS_TopAccent")
    plastic_shader = stage.GetPrimAtPath("/BottleCap/Looks/MAT_Cap_DeepBlue_Plastic/PreviewSurface")
    material = stage.GetPrimAtPath("/BottleCap/Looks/MAT_Cap_Physics_TEMP")
    material_api = UsdPhysics.MaterialAPI(material)
    mass_api = UsdPhysics.MassAPI(root)
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    finite_inertia = inertia is not None and all(math.isfinite(float(value)) and float(value) > 0.0 for value in inertia)
    material_bindings_ok = all(
        prim.GetRelationship("material:binding:physics").GetTargets() == [material.GetPath()]
        for prim in collision_prims
    )
    checks = {
        "default_prim": bool(stage.GetDefaultPrim()) and stage.GetDefaultPrim().GetPath() == root.GetPath(),
        "meters_per_unit": abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-12,
        "up_axis_z": UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
        "root_exists": bool(root),
        "rigid_body_api": root.HasAPI(UsdPhysics.RigidBodyAPI),
        "mass_api": root.HasAPI(UsdPhysics.MassAPI),
        "physx_rigid_body_api": "PhysxRigidBodyAPI" in authored_api_schemas(root),
        "ccd_enabled": root.GetAttribute("physxRigidBody:enableCCD").Get() is True,
        "mass_positive": float(mass_api.GetMassAttr().Get() or 0.0) > 0.0,
        "inertia_finite_positive": finite_inertia,
        "visual_mesh_count": len(mesh_prims) == 1,
        "visual_rib_count": len(rib_prims) == 32,
        "top_accent_present": bool(top_accent) and top_accent.IsA(UsdGeom.Cylinder),
        "plastic_roughness": abs(float(plastic_shader.GetAttribute("inputs:roughness").Get()) - 0.30) < 1e-6,
        "plastic_clearcoat": abs(float(plastic_shader.GetAttribute("inputs:clearcoat").Get()) - 0.35) < 1e-6,
        "collision_prim_count": len(collision_prims) == 17,
        "material_api": material.HasAPI(UsdPhysics.MaterialAPI),
        "physx_material_api": "PhysxMaterialAPI" in authored_api_schemas(material),
        "friction_combine_average": material.GetAttribute("physxMaterial:frictionCombineMode").Get() == "average",
        "material_bindings": material_bindings_ok,
        "static_friction": abs(float(material_api.GetStaticFrictionAttr().Get()) - 0.90) < 1e-6,
        "dynamic_friction": abs(float(material_api.GetDynamicFrictionAttr().Get()) - 0.75) < 1e-6,
        "restitution": abs(float(material_api.GetRestitutionAttr().Get())) < 1e-9,
        "outer_diameter": abs(float(size[0]) - 0.034) < 5e-5 and abs(float(size[1]) - 0.034) < 5e-5,
        "visual_envelope_height": abs(float(size[2]) - 0.02225) < 5e-5,
        "neck_radial_clearance_positive": 0.0154 > 0.0150,
        "classification_present": root.GetCustomDataByKey("assetStatus") == "TEMPORARY_UNCALIBRATED",
        "visual_style_present": root.GetCustomDataByKey("visualStyle") == "DEEP_BLUE_PLASTIC_32_GRIP_RIBS_TOP_ACCENT",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "TEMPORARY_UNCALIBRATED_DIAGNOSTIC_CAP",
        "usd": {"absolute_path": str(usd_path), "sha256": sha256(usd_path)},
        "checks": checks,
        "computed": {
            "bounds_min_m": [float(value) for value in bounds.GetMin()],
            "bounds_max_m": [float(value) for value in bounds.GetMax()],
            "size_m": [float(value) for value in size],
            "mass_kg": float(mass_api.GetMassAttr().Get()),
            "center_of_mass_m": [float(value) for value in mass_api.GetCenterOfMassAttr().Get()],
            "diagonal_inertia_kg_m2": [float(value) for value in inertia],
            "visual_mesh_paths": [str(prim.GetPath()) for prim in mesh_prims],
            "visual_rib_paths": [str(prim.GetPath()) for prim in rib_prims],
            "collision_paths": [str(prim.GetPath()) for prim in collision_prims],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

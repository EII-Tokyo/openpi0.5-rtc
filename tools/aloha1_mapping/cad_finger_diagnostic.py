"""Build the visual-only supplier-CAD finger diagnostic composition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.cad_finger_installation import CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION
from tools.aloha1_mapping.cad_finger_installation import FINGER_LINK_CLOSED_ORIGIN_M
from tools.aloha1_mapping.cad_finger_installation import cad_global_to_finger_link_matrix
from tools.aloha1_mapping.cad_finger_installation import determinant3
from tools.aloha1_mapping.cad_finger_installation import transform_point

DIAGNOSTIC_COLLISION_POLICY = {
    "source_collision_branches": "UNCHANGED",
    "cad_mesh_role": "VISUAL_ONLY",
    "new_collision_api_applied": False,
    "final_default_collider_modified": False,
}

SOURCE_STAGE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
SOURCE_CAD_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)

FINGER_PATHS = {
    "left": {
        "link": (
            "/workcell/vx300s_left/"
            "vx300s_left_left_finger_link"
        ),
        "old_visual_root": "vx300s_10_gripper_finger_left",
        "color": (0.08, 0.24, 0.82),
        "cad_product": "Part__Feature007",
        "cad_side": "+X",
    },
    "right": {
        "link": (
            "/workcell/vx300s_left/"
            "vx300s_left_right_finger_link"
        ),
        "old_visual_root": "vx300s_10_gripper_finger_right",
        "color": (0.92, 0.28, 0.04),
        "cad_product": "Part__Feature008",
        "cad_side": "-X",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mesh_payload(side: str, path: Path) -> dict[str, Any]:
    source_points = []
    triangles = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            source_points.append(
                tuple(float(value) for value in line.split()[1:4])
            )
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                index = int(token.split("/", 1)[0])
                if index <= 0:
                    raise ValueError(
                        f"negative/zero OBJ index is unsupported: {path}"
                    )
                indices.append(index - 1)
            if len(indices) != 3:
                raise ValueError(f"non-triangle OBJ face: {path}")
            triangles.append(tuple(indices))
    if not source_points or not triangles:
        raise ValueError(f"OBJ has no mesh data: {path}")
    points = [transform_point(side, point) for point in source_points]
    axes = list(zip(*points, strict=True))
    bounds = [
        (min(values), max(values))
        for values in axes
    ]
    aabb = {
        "x_min": bounds[0][0],
        "x_max": bounds[0][1],
        "x_size": round(bounds[0][1] - bounds[0][0], 12),
        "y_min": bounds[1][0],
        "y_max": bounds[1][1],
        "y_size": round(bounds[1][1] - bounds[1][0], 12),
        "z_min": bounds[2][0],
        "z_max": bounds[2][1],
        "z_size": round(bounds[2][1] - bounds[2][0], 12),
    }
    closed_origin = FINGER_LINK_CLOSED_ORIGIN_M[side]
    closed_gripper_points = [
        tuple(
            point[index] + closed_origin[index]
            for index in range(3)
        )
        for point in points
    ]
    closed_axes = list(zip(*closed_gripper_points, strict=True))
    closed_bounds = [
        (min(values), max(values))
        for values in closed_axes
    ]
    closed_gripper_aabb = {
        axis: value
        for index, name in enumerate(("x", "y", "z"))
        for axis, value in (
            (f"{name}_min", closed_bounds[index][0]),
            (f"{name}_max", closed_bounds[index][1]),
            (
                f"{name}_size",
                round(
                    closed_bounds[index][1] - closed_bounds[index][0],
                    12,
                ),
            ),
        )
    }
    return {
        "side": side,
        "source_obj_path": str(path.resolve()),
        "source_obj_sha256": sha256_file(path),
        "source_points_cad_global_m": source_points,
        "points_finger_link_m": points,
        "triangles": triangles,
        "point_count": len(points),
        "triangle_count": len(triangles),
        "aabb_m": aabb,
        "closed_gripper_aabb_m": closed_gripper_aabb,
    }


def _relative_asset_path(path: Path, owner_layer: Path) -> str:
    return os.path.relpath(path.resolve(), owner_layer.resolve().parent)


def _collision_inventory(stage: Any) -> list[dict[str, Any]]:
    from pxr import Usd
    from pxr import UsdPhysics

    records = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim)
                .GetApproximationAttr()
                .Get()
            )
        records.append(
            {
                "path": str(prim.GetPath()),
                "type_name": prim.GetTypeName(),
                "is_instance_proxy": prim.IsInstanceProxy(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "approximation": approximation,
            }
        )
    return sorted(records, key=lambda record: record["path"])


def _articulation_roots(stage: Any) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange.Stage(stage)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )


def _author_geometry_layer(
    path: Path,
    payloads: dict[str, dict[str, Any]],
) -> None:
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/CadFingerGeometry").GetPrim()
    stage.SetDefaultPrim(root)
    for side, payload in payloads.items():
        container = UsdGeom.Xform.Define(
            stage,
            f"/CadFingerGeometry/{side}_finger",
        )
        mesh = UsdGeom.Mesh.Define(
            stage,
            f"/CadFingerGeometry/{side}_finger/mesh",
        )
        points = [
            Gf.Vec3f(*point)
            for point in payload["points_finger_link_m"]
        ]
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(
            [3] * payload["triangle_count"]
        )
        mesh.CreateFaceVertexIndicesAttr(
            [
                index
                for triangle in payload["triangles"]
                for index in triangle
            ]
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
        mesh.CreateDoubleSidedAttr().Set(False)  # noqa: FBT003
        mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(points))
        mesh.CreateDisplayColorAttr(
            [Gf.Vec3f(*FINGER_PATHS[side]["color"])]
        )
        container.GetPrim().SetCustomDataByKey(
            "aloha1:sourceObjSha256",
            payload["source_obj_sha256"],
        )
        container.GetPrim().SetCustomDataByKey(
            "aloha1:cadProduct",
            FINGER_PATHS[side]["cad_product"],
        )
    stage.GetRootLayer().Save()


def _author_configuration_layer(
    path: Path,
    geometry_path: Path,
) -> None:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    geometry_asset = _relative_asset_path(geometry_path, path)
    for side, paths in FINGER_PATHS.items():
        visuals_path = f"{paths['link']}/visuals"
        visuals = stage.OverridePrim(visuals_path)
        visuals.SetInstanceable(False)  # noqa: FBT003
        old_root = stage.OverridePrim(
            f"{visuals_path}/{paths['old_visual_root']}"
        )
        old_root.SetActive(False)  # noqa: FBT003
        replacement_path = (
            f"{visuals_path}/diagnostic_supplier_cad_{side}_finger"
        )
        replacement = UsdGeom.Xform.Define(
            stage,
            replacement_path,
        ).GetPrim()
        reference_ok = replacement.GetReferences().AddReference(
            geometry_asset,
            Sdf.Path(f"/CadFingerGeometry/{side}_finger"),
        )
        if not reference_ok:
            raise RuntimeError(
                f"failed to reference geometry for {side}"
            )
        replacement.SetCustomDataByKey(
            "aloha1:diagnosticRole",
            "VISUAL_ONLY_DIAGNOSTIC_NOT_FINAL",
        )
    stage.GetRootLayer().Save()


def _author_wrapper_layer(
    path: Path,
    configuration_path: Path,
    source_stage_path: Path,
) -> None:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/workcell").GetPrim()
    reference_ok = root.GetReferences().AddReference(
        _relative_asset_path(source_stage_path, path),
        Sdf.Path("/workcell"),
    )
    if not reference_ok:
        raise RuntimeError("failed to reference authorized /workcell")
    stage.GetRootLayer().subLayerPaths = [
        _relative_asset_path(configuration_path, path)
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()


def _authored_spec_paths(layer_path: Path) -> list[str]:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        raise RuntimeError(f"failed to open layer: {layer_path}")
    paths = []

    def visit(path: Any) -> bool:
        paths.append(str(path))
        return True

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return sorted(paths)


def create_diagnostic_asset(
    *,
    source_stage_path: Path,
    left_obj_path: Path,
    right_obj_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    source_stage_path = source_stage_path.resolve(strict=True)
    if sha256_file(source_stage_path) != SOURCE_STAGE_SHA256:
        raise RuntimeError("authorized source Stage hash mismatch")
    if output_root.exists():
        raise FileExistsError(
            f"diagnostic output root already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    geometry_dir = output_root / "geometry"
    configuration_dir = output_root / "configuration"
    geometry_dir.mkdir()
    configuration_dir.mkdir()
    geometry_path = geometry_dir / "supplier_cad_finger_visual.usda"
    configuration_path = (
        configuration_dir / "supplier_cad_finger_installation.usda"
    )
    wrapper_path = output_root / "aloha_viperx_cad_finger_diagnostic.usda"

    source_sha256_before = sha256_file(source_stage_path)
    payloads = {
        "left": build_mesh_payload("left", left_obj_path),
        "right": build_mesh_payload("right", right_obj_path),
    }
    _author_geometry_layer(geometry_path, payloads)
    _author_configuration_layer(configuration_path, geometry_path)
    _author_wrapper_layer(
        wrapper_path,
        configuration_path,
        source_stage_path,
    )

    source_stage = Usd.Stage.Open(str(source_stage_path), Usd.Stage.LoadAll)
    diagnostic_stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if source_stage is None or diagnostic_stage is None:
        raise RuntimeError("failed to open source or diagnostic Stage")

    visual_records = {}
    for side, paths in FINGER_PATHS.items():
        visuals_path = f"{paths['link']}/visuals"
        old_path = f"{visuals_path}/{paths['old_visual_root']}"
        replacement_path = (
            f"{visuals_path}/diagnostic_supplier_cad_{side}_finger"
        )
        mesh_path = f"{replacement_path}/mesh"
        visuals = diagnostic_stage.GetPrimAtPath(visuals_path)
        old_root = diagnostic_stage.GetPrimAtPath(old_path)
        replacement = diagnostic_stage.GetPrimAtPath(replacement_path)
        mesh_prim = diagnostic_stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        visual_records[side] = {
            "visuals_path": visuals_path,
            "visuals_is_instance": visuals.IsInstance(),
            "old_visual_root": old_path,
            "old_visual_active": old_root.IsActive(),
            "replacement_path": replacement_path,
            "replacement_valid": replacement.IsValid(),
            "mesh_path": mesh_path,
            "mesh_valid": mesh_prim.IsValid(),
            "mesh_is_instance_proxy": mesh_prim.IsInstanceProxy(),
            "mesh_point_count": len(mesh.GetPointsAttr().Get() or []),
            "mesh_face_count": len(
                mesh.GetFaceVertexCountsAttr().Get() or []
            ),
            "mesh_has_collision_api": mesh_prim.HasAPI(
                UsdPhysics.CollisionAPI
            ),
            "mesh_has_rigid_body_api": mesh_prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ),
        }

    source_collision_inventory = _collision_inventory(source_stage)
    diagnostic_collision_inventory = _collision_inventory(diagnostic_stage)
    source_sha256_after = sha256_file(source_stage_path)
    configuration_specs = _authored_spec_paths(configuration_path)
    forbidden_configuration_specs = [
        path
        for path in configuration_specs
        if "/collisions" in path
        or "/joints" in path
        or "physics:" in path
        or "physx:" in path
    ]
    gates = {
        "source_stage_immutable": (
            source_sha256_before
            == source_sha256_after
            == SOURCE_STAGE_SHA256
        ),
        "default_prim": (
            str(diagnostic_stage.GetDefaultPrim().GetPath()) == "/workcell"
        ),
        "units": (
            UsdGeom.GetStageMetersPerUnit(diagnostic_stage) == 1.0
            and UsdGeom.GetStageUpAxis(diagnostic_stage)
            == UsdGeom.Tokens.z
        ),
        "visual_replacement": all(
            not record["visuals_is_instance"]
            and not record["old_visual_active"]
            and record["replacement_valid"]
            and record["mesh_valid"]
            and not record["mesh_is_instance_proxy"]
            and not record["mesh_has_collision_api"]
            and not record["mesh_has_rigid_body_api"]
            and record["mesh_point_count"]
            == payloads[side]["point_count"]
            and record["mesh_face_count"]
            == payloads[side]["triangle_count"]
            for side, record in visual_records.items()
        ),
        "collision_inventory_identical": (
            source_collision_inventory == diagnostic_collision_inventory
        ),
        "articulation_roots_identical": (
            _articulation_roots(source_stage)
            == _articulation_roots(diagnostic_stage)
        ),
        "configuration_has_no_collision_joint_physics_specs": (
            not forbidden_configuration_specs
        ),
        "proper_rotation_no_mirror": (
            determinant3(CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION) == 1.0
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "ISOLATED_SUPPLIER_CAD_FINGER_VISUAL_DIAGNOSTIC; "
            "NOT_FINAL_ASSET; COLLIDERS_UNCHANGED"
        ),
        "source_stage": {
            "absolute_path": str(source_stage_path),
            "sha256_before": source_sha256_before,
            "sha256_after": source_sha256_after,
            "explicit_reference_prim": "/workcell",
        },
        "diagnostic_outputs": {
            "root_usd": {
                "absolute_path": str(wrapper_path.resolve()),
                "sha256": sha256_file(wrapper_path),
            },
            "configuration_layer": {
                "absolute_path": str(configuration_path.resolve()),
                "sha256": sha256_file(configuration_path),
                "authored_spec_paths": configuration_specs,
            },
            "geometry_layer": {
                "absolute_path": str(geometry_path.resolve()),
                "sha256": sha256_file(geometry_path),
            },
        },
        "mapping": {
            "source_cad_sha256": SOURCE_CAD_SHA256,
            "matrix_convention": (
                "column-vector math; points baked from CAD global metres "
                "into finger-link local metres"
            ),
            "cad_global_to_finger_link_matrix": {
                side: [
                    list(row)
                    for row in cad_global_to_finger_link_matrix(side)
                ]
                for side in ("left", "right")
            },
            "linear_determinant": determinant3(
                CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION
            ),
            "mirror_used": False,
            "unit_conversion": "OBJ already metres; scale 1.0",
            "left": "Part__Feature007 / CAD +X / left_finger",
            "right": "Part__Feature008 / CAD -X / right_finger",
        },
        "meshes": {
            side: {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "source_points_cad_global_m",
                    "points_finger_link_m",
                    "triangles",
                }
            }
            for side, payload in payloads.items()
        },
        "visual_records": visual_records,
        "collision_policy": DIAGNOSTIC_COLLISION_POLICY,
        "collision_inventory": {
            "source": source_collision_inventory,
            "diagnostic": diagnostic_collision_inventory,
            "difference": []
            if source_collision_inventory
            == diagnostic_collision_inventory
            else "NONEMPTY",
        },
        "articulation_roots": {
            "source": _articulation_roots(source_stage),
            "diagnostic": _articulation_roots(diagnostic_stage),
        },
        "forbidden_configuration_specs": forbidden_configuration_specs,
        "gates": gates,
        "license": {
            "status": "UNKNOWN_HARD_BLOCKER",
            "redistribution_allowed": False,
            "git_commit_diagnostic_geometry": False,
        },
        "task8": "NOT_RUN",
    }


def write_diagnostic_report(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ALOHA Viper Supplier-CAD Finger Diagnostic Asset",
        "",
        f"- Status: `{report['status']}`",
        f"- Root USD: `{report['diagnostic_outputs']['root_usd']['absolute_path']}`",
        f"- Source Stage: `{report['source_stage']['absolute_path']}`",
        f"- Source Stage immutable: `{report['gates']['source_stage_immutable']}`",
        "- Default/final collider modified: `false`",
        "- CAD visual promoted to collider: `false`",
        "- Task 8: `NOT_RUN`",
        "",
        "## Machine gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            "## Provenance boundary",
            "",
            (
                "The supplier CAD license remains `UNKNOWN_HARD_BLOCKER`; "
                "the derived diagnostic geometry is local-only and must not "
                "be committed or redistributed."
            ),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

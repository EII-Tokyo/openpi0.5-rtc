"""Build an isolated supplier-CAD convex-hull Task 5 diagnostic asset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.cad_finger_diagnostic import SOURCE_STAGE_SHA256
from tools.aloha1_mapping.cad_finger_diagnostic import _articulation_roots
from tools.aloha1_mapping.cad_finger_diagnostic import _author_geometry_layer
from tools.aloha1_mapping.cad_finger_diagnostic import _collision_inventory
from tools.aloha1_mapping.cad_finger_diagnostic import _relative_asset_path
from tools.aloha1_mapping.cad_finger_diagnostic import build_mesh_payload
from tools.aloha1_mapping.cad_finger_diagnostic import sha256_file

TASK5_COLLISION_POLICY = {
    "approximation": "convexHull",
    "source_generic_finger_colliders": "DEACTIVATED_IN_DIAGNOSTIC_ONLY",
    "supplier_cad_mesh_role": "VISUAL_AND_DIAGNOSTIC_COLLISION",
    "source_stage_modified": False,
    "default_configuration_modified": False,
    "final_default_collider_modified": False,
    "task8": "NOT_RUN",
}


def task5_finger_paths() -> dict[str, dict[str, dict[str, str]]]:
    return {
        robot: {
            side: {
                "link": (
                    f"/workcell/{stage_robot}/"
                    f"{stage_robot}_{side}_finger_link"
                ),
                "old_root": f"vx300s_10_gripper_finger_{side}",
                "cad_product": (
                    "Part__Feature007"
                    if side == "left"
                    else "Part__Feature008"
                ),
                "cad_side": "+X" if side == "left" else "-X",
            }
            for side in ("left", "right")
        }
        for robot, stage_robot in (
            ("follower_left", "vx300s_left"),
            ("follower_right", "vx300s_right"),
        )
    }


def _author_configuration_layer(
    *,
    path: Path,
    geometry_path: Path,
    finger_paths: dict[str, dict[str, dict[str, str]]],
) -> dict[str, list[str]]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    geometry_asset = _relative_asset_path(geometry_path, path)
    deactivated_visuals = []
    deactivated_collisions = []
    replacements = []
    for sides in finger_paths.values():
        for side, paths in sides.items():
            for branch in ("visuals", "collisions"):
                branch_path = f"{paths['link']}/{branch}"
                branch_prim = stage.OverridePrim(branch_path)
                branch_prim.SetInstanceable(False)  # noqa: FBT003
                old_path = f"{branch_path}/{paths['old_root']}"
                stage.OverridePrim(old_path).SetActive(False)  # noqa: FBT003
                replacement_path = (
                    f"{branch_path}/diagnostic_supplier_cad_{side}_finger"
                )
                replacement = UsdGeom.Xform.Define(
                    stage,
                    replacement_path,
                ).GetPrim()
                if not replacement.GetReferences().AddReference(
                    geometry_asset,
                    Sdf.Path(f"/CadFingerGeometry/{side}_finger"),
                ):
                    raise RuntimeError(
                        f"failed geometry reference: {replacement_path}"
                    )
                replacement.SetCustomDataByKey(
                    "aloha1:diagnosticRole",
                    (
                        "VISUAL_ONLY_DIAGNOSTIC_NOT_FINAL"
                        if branch == "visuals"
                        else "CONVEX_HULL_COLLISION_DIAGNOSTIC_NOT_FINAL"
                    ),
                )
                replacements.append(replacement_path)
                if branch == "visuals":
                    deactivated_visuals.append(old_path)
                else:
                    deactivated_collisions.append(old_path)
    stage.GetRootLayer().Save()
    return {
        "deactivated_visuals": deactivated_visuals,
        "deactivated_collisions": deactivated_collisions,
        "replacements": replacements,
    }


def _author_wrapper(
    *,
    path: Path,
    configuration_path: Path,
    physics_path: Path,
    source_stage_path: Path,
) -> None:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/workcell").GetPrim()
    if not root.GetReferences().AddReference(
        _relative_asset_path(source_stage_path, path),
        Sdf.Path("/workcell"),
    ):
        raise RuntimeError("failed to reference approved source /workcell")
    stage.GetRootLayer().subLayerPaths = [
        _relative_asset_path(physics_path, path),
        _relative_asset_path(configuration_path, path),
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()


def _author_physics_layer(
    *,
    wrapper_path: Path,
    physics_path: Path,
    finger_paths: dict[str, dict[str, dict[str, str]]],
) -> list[str]:
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to compose wrapper: {wrapper_path}")
    layer = stage.GetLayerStack(includeSessionLayers=False)
    target = next(
        item
        for item in layer
        if Path(item.realPath).resolve() == physics_path.resolve()
    )
    stage.SetEditTarget(target)
    collider_paths = []
    for sides in finger_paths.values():
        for side, paths in sides.items():
            mesh_path = (
                f"{paths['link']}/collisions/"
                f"diagnostic_supplier_cad_{side}_finger/mesh"
            )
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim.IsValid():
                raise RuntimeError(f"replacement mesh missing: {mesh_path}")
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh_prim)
            mesh_prim.SetCustomDataByKey(
                "aloha1:diagnosticCollider",
                "SUPPLIER_CAD_CONVEX_HULL_NOT_FINAL",
            )
            collider_paths.append(mesh_path)
    target.Save()
    return collider_paths


def _nonfinger_colliders(
    inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record
        for record in inventory
        if "_finger_link/collisions/" not in record["path"]
    ]


def _source_follower_presence(
    stage: Any,
) -> dict[str, bool]:
    from pxr import Usd
    from pxr import UsdPhysics

    presence = {}
    for robot, sides in task5_finger_paths().items():
        robot_root = sides["left"]["link"].rsplit("/", 1)[0]
        robot_prim = stage.GetPrimAtPath(robot_root)
        articulation_roots = []
        if robot_prim.IsValid():
            articulation_roots = [
                prim
                for prim in Usd.PrimRange(robot_prim)
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
        presence[robot] = bool(
            robot_prim.IsValid()
            and articulation_roots
            and all(
                stage.GetPrimAtPath(paths["link"]).IsValid()
                for paths in sides.values()
            )
        )
    return presence


def create_task5_diagnostic_asset(
    *,
    source_stage_path: Path,
    left_obj_path: Path,
    right_obj_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    source_stage_path = source_stage_path.resolve(strict=True)
    source_hash_before = sha256_file(source_stage_path)
    if source_hash_before != SOURCE_STAGE_SHA256:
        raise RuntimeError("approved source Stage hash mismatch")
    source_stage = Usd.Stage.Open(str(source_stage_path), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError("approved source Stage did not open")
    source_follower_presence = _source_follower_presence(source_stage)
    all_finger_paths = task5_finger_paths()
    present_finger_paths = {
        robot: all_finger_paths[robot]
        for robot, present in source_follower_presence.items()
        if present
    }
    if not present_finger_paths:
        raise RuntimeError("approved source Stage has no usable follower")
    hard_blockers = []
    if not source_follower_presence["follower_right"]:
        hard_blockers.append(
            {
                "code": (
                    "HARD_BLOCKER_APPROVED_STAGE_MISSING_FOLLOWER_RIGHT"
                ),
                "evidence": "/workcell/vx300s_right is absent",
                "scope": "follower_right runtime Task 5",
            }
        )
    if output_root.exists():
        raise FileExistsError(
            f"Task 5 diagnostic output already exists: {output_root}"
        )
    geometry_dir = output_root / "geometry"
    configuration_dir = output_root / "configuration"
    physics_dir = output_root / "physics"
    geometry_dir.mkdir(parents=True)
    configuration_dir.mkdir()
    physics_dir.mkdir()
    geometry_path = geometry_dir / "supplier_cad_finger_mesh.usda"
    configuration_path = (
        configuration_dir / "supplier_cad_finger_task5.usda"
    )
    physics_path = physics_dir / "supplier_cad_finger_convex_hull.usda"
    wrapper_path = output_root / "aloha_viperx_supplier_cad_task5.usda"
    physics_layer = Sdf.Layer.CreateNew(str(physics_path))
    if physics_layer is None:
        raise RuntimeError(f"failed to create physics layer: {physics_path}")
    physics_layer.Save()

    payloads = {
        "left": build_mesh_payload("left", left_obj_path),
        "right": build_mesh_payload("right", right_obj_path),
    }
    _author_geometry_layer(geometry_path, payloads)
    authored = _author_configuration_layer(
        path=configuration_path,
        geometry_path=geometry_path,
        finger_paths=present_finger_paths,
    )
    _author_wrapper(
        path=wrapper_path,
        configuration_path=configuration_path,
        physics_path=physics_path,
        source_stage_path=source_stage_path,
    )
    authored_collider_paths = _author_physics_layer(
        wrapper_path=wrapper_path,
        physics_path=physics_path,
        finger_paths=present_finger_paths,
    )

    diagnostic_stage = Usd.Stage.Open(str(wrapper_path), Usd.Stage.LoadAll)
    if diagnostic_stage is None:
        raise RuntimeError("Task 5 diagnostic Stage did not open")
    source_inventory = _collision_inventory(source_stage)
    diagnostic_inventory = _collision_inventory(diagnostic_stage)
    new_colliders = []
    for path in authored_collider_paths:
        prim = diagnostic_stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        new_colliders.append(
            {
                "path": path,
                "valid": prim.IsValid(),
                "collision_api": prim.HasAPI(UsdPhysics.CollisionAPI),
                "mesh_collision_api": prim.HasAPI(
                    UsdPhysics.MeshCollisionAPI
                ),
                "approximation": (
                    UsdPhysics.MeshCollisionAPI(prim)
                    .GetApproximationAttr()
                    .Get()
                ),
                "point_count": len(mesh.GetPointsAttr().Get() or []),
                "face_count": len(
                    mesh.GetFaceVertexCountsAttr().Get() or []
                ),
                "rigid_body_api": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            }
        )
    deactivated = []
    for path in authored["deactivated_collisions"]:
        prim = diagnostic_stage.GetPrimAtPath(path)
        deactivated.append(
            {
                "path": path,
                "valid": prim.IsValid(),
                "active": prim.IsActive() if prim.IsValid() else None,
            }
        )
    nonfinger_unchanged = (
        _nonfinger_colliders(source_inventory)
        == _nonfinger_colliders(diagnostic_inventory)
    )
    source_hash_after = sha256_file(source_stage_path)
    gates = {
        "source_stage_immutable": (
            source_hash_before
            == source_hash_after
            == SOURCE_STAGE_SHA256
        ),
        "expected_new_finger_colliders": (
            len(new_colliders) == 2 * len(present_finger_paths)
        ),
        "all_new_colliders_convex_hull": all(
            item["valid"]
            and item["collision_api"]
            and item["mesh_collision_api"]
            and item["approximation"] == "convexHull"
            and item["point_count"] == payloads[
                "left" if "left_finger_link" in item["path"] else "right"
            ]["point_count"]
            and item["face_count"] == payloads[
                "left" if "left_finger_link" in item["path"] else "right"
            ]["triangle_count"]
            and not item["rigid_body_api"]
            for item in new_colliders
        ),
        "expected_generic_finger_colliders_deactivated": (
            len(deactivated) == 2 * len(present_finger_paths)
            and all(item["valid"] and not item["active"] for item in deactivated)
        ),
        "only_existing_follower_fingers_authored": all(
            any(
                item["path"].startswith(paths["left"]["link"].rsplit("/", 1)[0])
                for paths in present_finger_paths.values()
            )
            for item in new_colliders
        ),
        "nonfinger_collision_inventory_unchanged": nonfinger_unchanged,
        "articulation_roots_unchanged": (
            _articulation_roots(source_stage)
            == _articulation_roots(diagnostic_stage)
        ),
        "proper_units": (
            UsdGeom.GetStageMetersPerUnit(diagnostic_stage) == 1.0
            and UsdGeom.GetStageUpAxis(diagnostic_stage)
            == UsdGeom.Tokens.z
        ),
    }
    return {
        "schema_version": 2,
        "status": (
            "FAIL"
            if not all(gates.values())
            else "PARTIAL"
            if hard_blockers
            else "PASS"
        ),
        "scope": (
            "ISOLATED_SUPPLIER_CAD_FINGER_TASK5_CONVEX_HULL_DIAGNOSTIC; "
            "NOT_FINAL_ASSET"
        ),
        "source_stage": {
            "absolute_path": str(source_stage_path),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
        },
        "source_follower_presence": source_follower_presence,
        "hard_blockers": hard_blockers,
        "outputs": {
            name: {
                "absolute_path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("root_usd", wrapper_path),
                ("geometry_layer", geometry_path),
                ("configuration_layer", configuration_path),
                ("physics_layer", physics_path),
            )
        },
        "collision_policy": TASK5_COLLISION_POLICY,
        "source_meshes": {
            side: {
                "cad_product": (
                    "Part__Feature007"
                    if side == "left"
                    else "Part__Feature008"
                ),
                "absolute_path": payload["source_obj_path"],
                "sha256": payload["source_obj_sha256"],
                "point_count": payload["point_count"],
                "face_count": payload["triangle_count"],
                "aabb_m": payload["aabb_m"],
            }
            for side, payload in payloads.items()
        },
        "new_finger_colliders": new_colliders,
        "deactivated_generic_finger_colliders": deactivated,
        "deactivated_generic_finger_visuals": authored[
            "deactivated_visuals"
        ],
        "nonfinger_collision_inventory_unchanged": nonfinger_unchanged,
        "articulation_roots": _articulation_roots(diagnostic_stage),
        "gates": gates,
        "license": {
            "status": "UNKNOWN_HARD_BLOCKER",
            "blocks": [
                "commit_or_redistribute_supplier_CAD_derived_USD_without_license"
            ],
            "does_not_block": ["local_isolated_diagnostic"],
        },
        "task8": "NOT_RUN",
    }


def write_task5_asset_report(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ALOHA Viper supplier-CAD Task 5 diagnostic asset",
        "",
        f"- Status: `{report['status']}`",
        f"- Root USD: `{report['outputs']['root_usd']['absolute_path']}`",
        "- Approximation: `convexHull`",
        (
            "- New supplier-CAD finger colliders: "
            f"`{len(report['new_finger_colliders'])}`"
        ),
        (
            "- Source generic finger colliders deactivated in diagnostic: "
            f"`{len(report['deactivated_generic_finger_colliders'])}`"
        ),
        (
            "- Source follower presence: "
            f"`{report['source_follower_presence']}`"
        ),
        (
            "- Non-finger collision inventory unchanged: "
            f"`{report['nonfinger_collision_inventory_unchanged']}`"
        ),
        "- Source/default/final collider modified: `false / false / false`",
        "- Task 8: `NOT_RUN`",
        "",
        "This asset is an isolated physics diagnostic. Its supplier-CAD mesh "
        "is not promoted to the final/default collider by this report.",
        "",
        "## HARD_BLOCKER",
        "",
        *[
            (
                f"- `{item['code']}`: {item['evidence']} "
                f"(scope: {item['scope']})"
            )
            for item in report["hard_blockers"]
        ],
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

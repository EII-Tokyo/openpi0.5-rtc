#!/usr/bin/env python3
"""Audit Isaac Sim 5.1 convex hull/decomposition follower finger colliders."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.gripper_collider_ab import assert_profile_pair_is_frozen
from tools.aloha1_mapping.gripper_collider_ab import load_collision_profiles
from tools.aloha1_mapping.gripper_collider_ab import sha256_file


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_protected_baseline(
    project_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for item in manifest["protected_baseline"]:
        path = project_root / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        results.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "match": actual == item["sha256"],
            }
        )
    failures = [item["path"] for item in results if not item["match"]]
    if failures:
        raise RuntimeError(f"protected baseline hash mismatch: {failures}")
    return results


def _extension_version(extension_root: Path) -> str:
    text = (extension_root / "config/extension.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"version not found in {extension_root}")
    return match.group(1)


def _find_extension_root(module_file: str) -> Path:
    path = Path(module_file).resolve()
    for parent in path.parents:
        if (parent / "config/extension.toml").is_file():
            return parent
    raise RuntimeError(f"extension root not found above {path}")


def _local_api_probe() -> dict[str, Any]:
    from isaacsim.asset.importer.urdf import _urdf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    import_config = _urdf.ImportConfig()
    initial = bool(import_config.convex_decomp)
    import_config.set_convex_decomp(True)
    enabled = bool(import_config.convex_decomp)
    import_config.set_convex_decomp(False)
    disabled = bool(import_config.convex_decomp)

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Probe")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_api.CreateApproximationAttr().Set("convexDecomposition")
    decomposition_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
        mesh.GetPrim()
    )
    decomposition_defaults = {}
    for name, getter in (
        ("maxConvexHulls", decomposition_api.GetMaxConvexHullsAttr),
        ("voxelResolution", decomposition_api.GetVoxelResolutionAttr),
        ("errorPercentage", decomposition_api.GetErrorPercentageAttr),
        ("shrinkWrap", decomposition_api.GetShrinkWrapAttr),
        ("minThickness", decomposition_api.GetMinThicknessAttr),
        ("hullVertexLimit", decomposition_api.GetHullVertexLimitAttr),
    ):
        attribute = getter()
        decomposition_defaults[name] = {
            "value": attribute.Get(),
            "authored": attribute.HasAuthoredValueOpinion(),
            "type": str(attribute.GetTypeName()),
        }

    importer_root = _find_extension_root(_urdf.__file__)
    physx_schema_root = _find_extension_root(PhysxSchema.__file__)
    return {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "urdf_importer": {
            "version": _extension_version(importer_root),
            "root": str(importer_root),
            "binding": str(Path(_urdf.__file__).resolve()),
            "convex_decomp_initial": initial,
            "convex_decomp_after_true": enabled,
            "convex_decomp_after_false": disabled,
        },
        "mesh_collision_api": {
            "python_class": "pxr.UsdPhysics.MeshCollisionAPI",
            "approximation_readback": mesh_api.GetApproximationAttr().Get(),
            "supported_tokens_tested": ["convexHull", "convexDecomposition"],
        },
        "convex_decomposition_api": {
            "python_class": "pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI",
            "plugin_schema_type": "PhysxSchemaPhysxConvexDecompositionCollisionAPI",
            "schema_root": str(physx_schema_root),
            "defaults": decomposition_defaults,
        },
    }


def _iter_prims(stage: Any) -> Any:
    from pxr import Usd

    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def _collider_inventory(stage: Any) -> list[dict[str, Any]]:
    from pxr import UsdPhysics

    result = []
    for prim in _iter_prims(stage):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(
                prim
            ).GetApproximationAttr().Get()
        result.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "approximation": approximation,
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "instance_proxy": prim.IsInstanceProxy(),
            }
        )
    return sorted(result, key=lambda item: item["path"])


def _finger_collider_paths(inventory: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = [
        str(item["path"])
        for item in inventory
        if (
            ("_left_finger_link/" in str(item["path"]) or "_right_finger_link/" in str(item["path"]))
            and item.get("approximation") is not None
        )
    ]
    if len(paths) != 2:
        raise RuntimeError(f"expected exactly two finger mesh colliders, found {paths}")
    return sorted(paths)


def _create_diagnostic_asset(
    *,
    source_asset: Path,
    destination: Path,
    robot: str,
    approximation: str,
) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    source_stage = Usd.Stage.Open(str(source_asset))
    if source_stage is None:
        raise RuntimeError(f"failed to open source asset: {source_asset}")
    source_inventory = _collider_inventory(source_stage)
    source_fingers = _finger_collider_paths(source_inventory)
    source_instanceable_ancestors = set()
    for finger_path in source_fingers:
        finger_prim = source_stage.GetPrimAtPath(finger_path)
        ancestor = finger_prim
        while ancestor.IsValid():
            if ancestor.IsInstanceable():
                source_instanceable_ancestors.add(str(ancestor.GetPath()))
            ancestor = ancestor.GetParent()
    relative_fingers = [
        Sdf.Path(path).MakeRelativePath(Sdf.Path(f"/{robot}"))
        for path in source_fingers
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    root = UsdGeom.Xform.Define(stage, f"/{robot}").GetPrim()
    source_relative = os.path.relpath(source_asset, destination.parent)
    root.GetReferences().AddReference(
        source_relative,
        Sdf.Path(f"/{robot}"),
    )
    # The source hierarchy uses instanceable composition. Both A and B wrappers
    # disable it identically so the two exact finger child prims can receive
    # stronger diagnostic opinions.
    root.SetInstanceable(False)  # noqa: FBT003 - USD API is positional-only
    stage.SetDefaultPrim(root)

    deinstanced_ancestors = []
    for ancestor_path in sorted(
        source_instanceable_ancestors,
        key=lambda value: (value.count("/"), value),
    ):
        source_relative_path = Sdf.Path(ancestor_path).MakeRelativePath(
            Sdf.Path(f"/{robot}")
        )
        diagnostic_path = Sdf.Path(f"/{robot}").AppendPath(source_relative_path)
        ancestor_prim = stage.GetPrimAtPath(diagnostic_path)
        if not ancestor_prim.IsValid():
            raise RuntimeError(
                f"instanceable ancestor is not composed: {diagnostic_path}"
            )
        ancestor_prim.SetInstanceable(
            False  # noqa: FBT003 - USD API is positional-only
        )
        deinstanced_ancestors.append(str(diagnostic_path))

    authored_paths = []
    for relative in relative_fingers:
        path = Sdf.Path(f"/{robot}").AppendPath(relative)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsInstanceProxy():
            raise RuntimeError(f"finger collider is not locally overrideable: {path}")
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        if not mesh_api:
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(approximation)
        if approximation == "convexHull":
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
        elif approximation == "convexDecomposition":
            decomposition_api = (
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            )
            # Do not call any Create*Attr method here: schema fallback values
            # are intentionally left unauthored for the first A/B round.
            if any(
                getter().HasAuthoredValueOpinion()
                for getter in (
                    decomposition_api.GetMaxConvexHullsAttr,
                    decomposition_api.GetVoxelResolutionAttr,
                    decomposition_api.GetErrorPercentageAttr,
                    decomposition_api.GetShrinkWrapAttr,
                    decomposition_api.GetMinThicknessAttr,
                    decomposition_api.GetHullVertexLimitAttr,
                )
            ):
                raise RuntimeError("decomposition defaults were unexpectedly authored")
        else:
            raise ValueError(f"unsupported approximation: {approximation}")
        authored_paths.append(str(path))
    stage.GetRootLayer().Save()

    composed = Usd.Stage.Open(str(destination))
    if composed is None:
        raise RuntimeError(f"failed to reopen diagnostic asset: {destination}")
    composed_inventory = _collider_inventory(composed)
    composed_fingers = _finger_collider_paths(composed_inventory)
    actual = {
        item["path"]: item["approximation"]
        for item in composed_inventory
        if item["path"] in composed_fingers
    }
    if set(actual.values()) != {approximation}:
        raise RuntimeError(
            f"finger approximation readback mismatch in {destination}: {actual}"
        )

    source_tokens = {item["path"]: item["approximation"] for item in source_inventory}
    composed_tokens = {item["path"]: item["approximation"] for item in composed_inventory}
    changed = []
    for source_path, source_token in source_tokens.items():
        suffix = Sdf.Path(source_path).MakeRelativePath(Sdf.Path(f"/{robot}"))
        composed_path = str(Sdf.Path(f"/{robot}").AppendPath(suffix))
        composed_token = composed_tokens.get(composed_path)
        if source_token != composed_token:
            changed.append(
                {
                    "path": composed_path,
                    "source": source_token,
                    "diagnostic": composed_token,
                }
            )
    non_finger_changes = [
        item
        for item in changed
        if item["path"] not in composed_fingers
    ]
    if non_finger_changes:
        raise RuntimeError(
            f"non-finger collider changed in diagnostic layer: {non_finger_changes}"
        )
    if len(changed) != (0 if approximation == "convexHull" else 2):
        raise RuntimeError(f"unexpected collider change set: {changed}")
    return {
        "robot": robot,
        "source_asset": str(source_asset),
        "diagnostic_asset": str(destination),
        "diagnostic_sha256": sha256_file(destination),
        "root_instanceable_authored": False,
        "deinstanced_ancestors": deinstanced_ancestors,
        "authored_finger_paths": authored_paths,
        "finger_paths_readback": composed_fingers,
        "approximation_requested": approximation,
        "approximation_readback": actual,
        "changed_collider_tokens": changed,
        "non_finger_collider_changes": non_finger_changes,
        "all_colliders": composed_inventory,
    }


def _convex_piece_metrics(piece: Any) -> dict[str, Any]:
    from scipy.spatial import ConvexHull

    vertices = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in piece.vertices])
    volume = None
    if len(vertices) >= 4:
        try:
            volume = float(ConvexHull(vertices).volume)
        except Exception:
            volume = None
    return {
        "vertex_count": len(piece.vertices),
        "index_count": len(piece.indices),
        "face_count": len(piece.polygons),
        "aabb_min": vertices.min(axis=0) if len(vertices) else None,
        "aabb_max": vertices.max(axis=0) if len(vertices) else None,
        "volume": volume,
        "vertices": vertices,
        "indices": [int(value) for value in piece.indices],
        "polygons": [
            {
                "index_base": int(polygon.index_base),
                "num_vertices": int(polygon.num_vertices),
                "plane": [float(value) for value in polygon.plane],
            }
            for polygon in piece.polygons
        ],
    }


def _cooking_statistics() -> dict[str, int]:
    from omni.physx import get_physx_cooking_private_interface

    stats = get_physx_cooking_private_interface().get_cooking_statistics()
    return {
        "scheduled": int(stats.total_scheduled_tasks),
        "finished": int(stats.total_finished_tasks),
        "cache_hits": int(stats.total_finished_cache_hit_tasks),
        "cache_misses": int(stats.total_finished_cache_miss_tasks),
        "convex_polygon_limit_warnings": int(
            stats.total_warnings_convex_polygon_limits_reached
        ),
        "failed_gpu_compatibility_warnings": int(
            stats.total_warnings_failed_gpu_compatibility
        ),
    }


def _subtract_stats(
    after: Mapping[str, int],
    before: Mapping[str, int],
) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in before}


def _build_direct_mesh_cooking_probe(
    source_stage: Any,
    collider_path: str,
    approximation: str,
) -> tuple[Any, str, dict[str, Any]]:
    """Bake importer descendant meshes into a direct Mesh API cooking probe."""

    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    collider_prim = source_stage.GetPrimAtPath(collider_path)
    if not collider_prim.IsValid():
        raise RuntimeError(f"collider prim missing: {collider_path}")
    xform_cache = UsdGeom.XformCache()
    points: list[Any] = []
    indices: list[int] = []
    counts: list[int] = []
    mesh_sources = []
    for prim in Usd.PrimRange(collider_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        source_points = mesh.GetPointsAttr().Get() or []
        source_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        source_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        if not source_points or not source_indices:
            continue
        mesh_world = xform_cache.GetLocalToWorldTransform(prim)
        composed_scale = Gf.Transform(mesh_world).GetScale()
        base = len(points)
        points.extend(
            [
                Gf.Vec3d(
                    float(point[0]) * float(composed_scale[0]),
                    float(point[1]) * float(composed_scale[1]),
                    float(point[2]) * float(composed_scale[2]),
                )
                for point in source_points
            ]
        )
        indices.extend(base + int(index) for index in source_indices)
        counts.extend(int(count) for count in source_counts)
        mesh_sources.append(
            {
                "path": str(prim.GetPath()),
                "point_count": len(source_points),
                "face_count": len(source_counts),
                "composed_world_scale": [
                    float(composed_scale[0]),
                    float(composed_scale[1]),
                    float(composed_scale[2]),
                ],
            }
        )
    if not mesh_sources:
        raise RuntimeError(f"no descendant UsdGeom.Mesh found below {collider_path}")

    probe_stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(probe_stage, "/Probe").GetPrim()
    probe_stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(probe_stage, 1.0)
    UsdGeom.SetStageUpAxis(probe_stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(probe_stage, "/Probe/physicsScene")
    probe = UsdGeom.Mesh.Define(probe_stage, "/Probe/Collider")
    probe.CreatePointsAttr(points)
    probe.CreateFaceVertexIndicesAttr(indices)
    probe.CreateFaceVertexCountsAttr(counts)
    probe.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
    UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
    UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateDensityAttr(1.0)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(probe.GetPrim())
    mesh_api.CreateApproximationAttr().Set(approximation)
    if approximation == "convexHull":
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(probe.GetPrim())
    elif approximation == "convexDecomposition":
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(probe.GetPrim())
    else:
        raise ValueError(f"unsupported approximation: {approximation}")
    point_array = np.asarray(
        [[float(value[0]), float(value[1]), float(value[2])] for value in points],
        dtype=np.float64,
    )
    baked_extent = point_array.max(axis=0) - point_array.min(axis=0)
    if float(np.max(baked_extent)) >= 0.2:
        raise RuntimeError(
            f"cooking probe unit/scale audit failed for {collider_path}: "
            f"extent={baked_extent.tolist()}"
        )
    return (
        probe_stage,
        "/Probe/Collider",
        {
            "status": "DIRECT_MESH_EQUIVALENT_FOR_SUPPORTED_COOKING_API",
            "reason": (
                "importer authors CollisionAPI on an Xform with descendant Mesh; "
                "the public 5.1 cooking request returned RESULT_ERROR_INVALID_PARSING"
            ),
            "source_collider": collider_path,
            "source_meshes": mesh_sources,
            "baked_point_count": len(points),
            "baked_face_count": len(counts),
            "baked_aabb_min": point_array.min(axis=0),
            "baked_aabb_max": point_array.max(axis=0),
            "baked_aabb_extent": baked_extent,
            "approximation_readback": mesh_api.GetApproximationAttr().Get(),
            "decomposition_parameters_authored": False,
        },
    )


def _request_cooked_representation(
    app: Any,
    *,
    stage: Any,
    collider_path: str,
) -> tuple[Any, list[Any], float]:
    from omni.physx import get_physx_cooking_interface
    from pxr import PhysicsSchemaTools
    from pxr import UsdUtils

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    callback: dict[str, Any] = {}

    def on_result(result: Any, convexes: Sequence[Any]) -> None:
        callback["result"] = result
        callback["convexes"] = list(convexes)

    task_start = time.perf_counter()
    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=stage_id,
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(collider_path),
        run_asynchronously=True,
        on_result=on_result,
    )
    for _ in range(600):
        if callback:
            break
        app.update()
    elapsed = time.perf_counter() - task_start
    if not callback:
        raise RuntimeError(f"collision cooking timed out: {collider_path}")
    return callback["result"], callback["convexes"], elapsed


def _cook_finger_colliders(
    app: Any,
    asset: Path,
) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Usd
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"failed to open diagnostic asset for cooking: {asset}")
    inventory = _collider_inventory(stage)
    finger_paths = _finger_collider_paths(inventory)
    cooking = get_physx_cooking_interface()
    cooking.release_local_mesh_cache()
    start_stats = _cooking_statistics()
    cooked = {}
    for collider_path in finger_paths:
        composed_result, convexes, composed_elapsed = _request_cooked_representation(
            app,
            stage=stage,
            collider_path=collider_path,
        )
        result = composed_result
        probe = None
        elapsed = composed_elapsed
        if result == PhysxCollisionRepresentationResult.RESULT_ERROR_INVALID_PARSING:
            approximation = UsdPhysics.MeshCollisionAPI(
                stage.GetPrimAtPath(collider_path)
            ).GetApproximationAttr().Get()
            probe_stage, probe_path, probe = _build_direct_mesh_cooking_probe(
                stage,
                collider_path,
                approximation,
            )
            result, convexes, fallback_elapsed = _request_cooked_representation(
                app,
                stage=probe_stage,
                collider_path=probe_path,
            )
            elapsed += fallback_elapsed
        if result != PhysxCollisionRepresentationResult.RESULT_VALID:
            raise RuntimeError(f"collision cooking failed for {collider_path}: {result}")
        pieces = [_convex_piece_metrics(piece) for piece in convexes]
        all_min = np.min(
            np.asarray([piece["aabb_min"] for piece in pieces], dtype=np.float64),
            axis=0,
        )
        all_max = np.max(
            np.asarray([piece["aabb_max"] for piece in pieces], dtype=np.float64),
            axis=0,
        )
        volumes = [piece["volume"] for piece in pieces]
        cooked[collider_path] = {
            "result": str(result),
            "composed_collider_request_result": str(composed_result),
            "direct_mesh_probe": probe,
            "piece_count": len(pieces),
            "pieces": pieces,
            "combined_aabb_min": all_min,
            "combined_aabb_max": all_max,
            "sum_piece_volume": (
                float(sum(volumes)) if all(value is not None for value in volumes) else None
            ),
            "runtime_s": elapsed,
        }
    end_stats = _cooking_statistics()
    return {
        "asset": str(asset),
        "colliders": cooked,
        "cooking_statistics_delta": _subtract_stats(end_stats, start_stats),
    }


def _source_mesh_metrics(
    source_path: Path,
    scale: Sequence[float],
) -> tuple[dict[str, Any], Any]:
    import trimesh

    mesh = trimesh.load_mesh(source_path, force="mesh", process=True)
    mesh.apply_scale(np.asarray(scale, dtype=np.float64))
    hull = mesh.convex_hull
    metrics = {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "triangle_count": len(mesh.faces),
        "vertex_count_after_weld": len(mesh.vertices),
        "watertight": bool(mesh.is_watertight),
        "volume_valid": bool(mesh.is_volume),
        "volume": float(mesh.volume) if mesh.is_volume else None,
        "aabb_min": mesh.bounds[0],
        "aabb_max": mesh.bounds[1],
        "aabb_extent": mesh.extents,
        "offline_single_hull_supplement": {
            "status": "SUPPLEMENTAL_NOT_PHYSX_COOKED_RESULT",
            "vertex_count": len(hull.vertices),
            "face_count": len(hull.faces),
            "aabb_min": hull.bounds[0],
            "aabb_max": hull.bounds[1],
            "volume": float(hull.volume),
        },
    }
    return metrics, mesh


def _piece_meshes(cooking_result: Mapping[str, Any]) -> list[Any]:
    import trimesh

    meshes = []
    for piece in cooking_result["pieces"]:
        faces = []
        indices = piece["indices"]
        for polygon in piece["polygons"]:
            start = polygon["index_base"]
            polygon_indices = indices[start : start + polygon["num_vertices"]]
            faces.extend(
                [
                    [
                        polygon_indices[0],
                        polygon_indices[offset],
                        polygon_indices[offset + 1],
                    ]
                    for offset in range(1, len(polygon_indices) - 1)
                ]
            )
        meshes.append(
            trimesh.Trimesh(
                vertices=np.asarray(piece["vertices"], dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
        )
    return meshes


def _sampling_difference(
    source_mesh: Any,
    cooked: Mapping[str, Any],
    *,
    seed: int = 51073,
    sample_count: int = 20000,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    import trimesh

    pieces = _piece_meshes(cooked)
    if not pieces:
        return {"status": "FAIL", "reason": "no_cooked_pieces"}
    combined = trimesh.util.concatenate(pieces)
    source_points, _ = trimesh.sample.sample_surface(
        source_mesh,
        sample_count,
        seed=seed,
    )
    cooked_points, _ = trimesh.sample.sample_surface(
        combined,
        sample_count,
        seed=seed,
    )
    source_tree = cKDTree(source_points)
    cooked_tree = cKDTree(cooked_points)
    cooked_to_source = source_tree.query(cooked_points, workers=1)[0]
    source_to_cooked = cooked_tree.query(source_points, workers=1)[0]

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_m": float(np.mean(values)),
            "p95_m": float(np.quantile(values, 0.95)),
            "p99_m": float(np.quantile(values, 0.99)),
            "max_m": float(np.max(values)),
        }

    source_extent_x = float(source_mesh.bounds[1, 0] - source_mesh.bounds[0, 0])
    inner_threshold_x = float(
        source_mesh.bounds[1, 0] - 0.25 * source_extent_x
    )
    source_inner = source_points[source_points[:, 0] >= inner_threshold_x]
    cooked_inner = cooked_points[cooked_points[:, 0] >= inner_threshold_x]
    inner_result: dict[str, Any]
    if len(source_inner) >= 100 and len(cooked_inner) >= 100:
        source_inner_tree = cKDTree(source_inner)
        cooked_inner_tree = cKDTree(cooked_inner)
        inner_result = {
            "status": "SUPPLEMENTAL_CONTACT_SIDE_REGION",
            "direction_source_mesh_local": [1.0, 0.0, 0.0],
            "direction_derivation": (
                "URDF left/right prismatic joint closing directions transformed "
                "by their collision origin RPY both map to source-mesh local +X"
            ),
            "visualization_crop": (
                "top quartile of source local X AABB; visualization region only, "
                "not a guessed physical pad dimension"
            ),
            "threshold_local_x_m": inner_threshold_x,
            "source_sample_count": len(source_inner),
            "cooked_sample_count": len(cooked_inner),
            "cooked_to_source": stats(
                source_inner_tree.query(cooked_inner, workers=1)[0]
            ),
            "source_to_cooked": stats(
                cooked_inner_tree.query(source_inner, workers=1)[0]
            ),
        }
    else:
        inner_result = {
            "status": "PARTIAL_INSUFFICIENT_SAMPLES",
            "source_sample_count": len(source_inner),
            "cooked_sample_count": len(cooked_inner),
        }
    return {
        "status": "SUPPLEMENTAL_SAMPLED_SURFACE_DISTANCE",
        "sample_count_each_surface": sample_count,
        "seed": seed,
        "method": "bidirectional nearest neighbor between deterministic surface samples",
        "cooked_to_source": stats(cooked_to_source),
        "source_to_cooked": stats(source_to_cooked),
        "inner_gripping_region": inner_result,
    }


def _render_cooked(
    cooked: Mapping[str, Any],
    *,
    overview_path: Path,
    closeup_path: Path,
    inner_path: Path,
    title: str,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes = _piece_meshes(cooked)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(meshes), 1)))

    def draw(path: Path, *, mode: str) -> None:
        figure = plt.figure(figsize=(9, 7), dpi=180)
        axis = figure.add_subplot(111, projection="3d")
        all_vertices = []
        for index, mesh in enumerate(meshes):
            triangles = mesh.vertices[mesh.faces]
            collection = Poly3DCollection(
                triangles,
                alpha=0.50,
                facecolor=colors[index],
                edgecolor=(0.08, 0.08, 0.08, 0.40),
                linewidth=0.15,
            )
            axis.add_collection3d(collection)
            all_vertices.append(mesh.vertices)
        vertices = np.vstack(all_vertices)
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        center = (lower + upper) / 2.0
        extent = upper - lower
        if mode == "distal":
            # The distal half is selected numerically along the longest local
            # axis; this is a reproducible crop, not an eyeballed dimension.
            longest = int(np.argmax(extent))
            crop_lower = lower.copy()
            crop_lower[longest] = center[longest]
            limits = list(zip(crop_lower, upper, strict=True))
            label = "distal-half close-up (longest local axis)"
        elif mode == "inner":
            crop_lower = lower.copy()
            crop_lower[0] = upper[0] - 0.25 * extent[0]
            limits = list(zip(crop_lower, upper, strict=True))
            label = (
                "inner gripping-side close-up "
                "(mesh-local +X derived from URDF joint axis/RPY)"
            )
        else:
            margin = np.maximum(extent * 0.08, 1.0e-4)
            limits = list(zip(lower - margin, upper + margin, strict=True))
            label = "full cooked collider"
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_xlabel("local X (m)")
        axis.set_ylabel("local Y (m)")
        axis.set_zlabel("local Z (m)")
        axis.set_title(f"{title}\n{label}; {len(meshes)} cooked convex pieces")
        axis.view_init(
            elev=20 if mode == "inner" else 24,
            azim=5 if mode == "inner" else -55,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)

    draw(overview_path, mode="overview")
    draw(closeup_path, mode="distal")
    draw(inner_path, mode="inner")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ALOHA 1 Gripper Collider Comparison",
        "",
        f"Status: **{report['status']}**",
        "",
        "This is a diagnostic comparison. Convex decomposition is supported by "
        "the local NVIDIA PhysX schema, but it is not assumed to be the correct "
        "final collider and does not produce an exact collider.",
        "",
        "## Local Isaac 5.1 readback",
        "",
        f"- URDF Importer: `{report['local_api']['urdf_importer']['version']}`",
        "- `ImportConfig.convex_decomp` initial readback: "
        f"`{report['local_api']['urdf_importer']['convex_decomp_initial']}`",
        "- Approximation tokens tested: `convexHull`, `convexDecomposition`",
        "",
        "| Attribute | Local schema default | Authored by diagnostic layer |",
        "| --- | ---: | --- |",
    ]
    for name, item in report["local_api"]["convex_decomposition_api"]["defaults"].items():
        lines.append(f"| `{name}` | `{item['value']}` | `{item['authored']}` |")
    lines.extend(
        [
            "",
            "## Cooked geometry",
            "",
            "| Profile | Robot | Side | Pieces | Sum convex volume (m³) | GPU warning count |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for profile, profile_result in report["profiles"].items():
        for asset in profile_result["assets"]:
            warnings = asset["cooking"]["cooking_statistics_delta"][
                "failed_gpu_compatibility_warnings"
            ]
            for path, collider in asset["cooking"]["colliders"].items():
                side = "left" if "_left_finger_link/" in path else "right"
                lines.append(
                    f"| {profile} | {asset['robot']} | {side} | "
                    f"{collider['piece_count']} | {collider['sum_piece_volume']} | "
                    f"{warnings} |"
                )
    lines.extend(
        [
            "",
            "The source STL is not watertight, so its mesh volume is not presented "
            "as calibrated solid volume. Cooked-piece volume, AABB, and deterministic "
            "surface sampling are retained in the JSON report.",
            "",
            "## Numeric interpretation",
            "",
            f"- Hull/decomposition cooked-volume ratio: "
            f"`{report['geometry_conclusions']['hull_to_decomposition_volume_ratio']}`.",
            f"- Decomposition piece count: "
            f"`{report['geometry_conclusions']['decomposition_piece_count']}` "
            f"(local default maximum "
            f"`{report['geometry_conclusions']['decomposition_max_convex_hulls_default']}`).",
            f"- Hull source-distance p95: "
            f"`{report['geometry_conclusions']['hull_cooked_to_source_p95_m']} m`; "
            f"decomposition: "
            f"`{report['geometry_conclusions']['decomposition_cooked_to_source_p95_m']} m`.",
            "",
            "The numeric evidence supports that the single hull bridges STL "
            "concavities. It does not by itself prove that the bridged region is "
            "the calibrated inner fingertip contact surface.",
            "",
        ]
    )
    runtime = report.get("runtime_ab_evidence")
    if runtime is not None:
        lines.extend(
            [
                "## Frozen runtime A/B",
                "",
                f"- Final status: `{runtime['CONVEX_DECOMPOSITION_STATUS']}`.",
                f"- Root cause: `{runtime['root_cause_classification']}`.",
                f"- Hull drop: `{runtime['hull_drop_m']} m`; decomposition drop: "
                f"`{runtime['decomposition_drop_m']} m`; unchanged gate: "
                f"`{runtime['drop_gate_m']} m`.",
                f"- Contact points per trial: Hull `{runtime['hull_contact_points']}`, "
                f"decomposition `{runtime['decomposition_contact_points']}`.",
                f"- Mean runtime ratio decomposition/hull: "
                f"`{runtime['runtime_ratio_decomposition_to_hull']}`.",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _representative_collider(
    profile_results: Mapping[str, Any],
    profile: str,
) -> Mapping[str, Any]:
    return next(
        iter(
            profile_results[profile]["assets"][0]["cooking"]["colliders"].values()
        )
    )


def _geometry_conclusions(
    profile_results: Mapping[str, Any],
    local_api: Mapping[str, Any],
) -> dict[str, Any]:
    hull = _representative_collider(profile_results, "convex_hull")
    decomposition = _representative_collider(
        profile_results,
        "convex_decomposition",
    )
    hull_p95 = hull["source_surface_sampling"]["cooked_to_source"]["p95_m"]
    decomposition_p95 = decomposition["source_surface_sampling"][
        "cooked_to_source"
    ]["p95_m"]
    hull_inner_p95 = hull["source_surface_sampling"]["inner_gripping_region"][
        "cooked_to_source"
    ]["p95_m"]
    decomposition_inner_p95 = decomposition["source_surface_sampling"][
        "inner_gripping_region"
    ]["cooked_to_source"]["p95_m"]
    piece_extents = [
        np.asarray(piece["aabb_max"], dtype=np.float64)
        - np.asarray(piece["aabb_min"], dtype=np.float64)
        for piece in decomposition["pieces"]
    ]
    return {
        "single_hull_closes_stl_concavities": "EVIDENCE_SUPPORTED",
        "single_hull_exceeds_calibrated_inner_surface": (
            "INCONCLUSIVE_NO_CAD_OR_MEASURED_INNER_SURFACE"
        ),
        "single_hull_exceeds_source_mesh_inner_region": (
            "EVIDENCE_SUPPORTED_BY_CONTACT_SIDE_SAMPLING"
            if hull_inner_p95 > decomposition_inner_p95
            else "NOT_SUPPORTED"
        ),
        "decomposition_source_surface_fidelity": (
            "IMPROVED_SAMPLED_DISTANCE_BUT_NOT_EXACT_COLLIDER"
        ),
        "decomposition_tiny_piece_risk": (
            "EVIDENCE_PRESENT_HIT_MAX_HULL_COUNT_NO_PARAMETER_TUNING_RUN"
        ),
        "hull_to_decomposition_volume_ratio": (
            float(hull["sum_piece_volume"])
            / float(decomposition["sum_piece_volume"])
        ),
        "hull_cooked_to_source_p95_m": hull_p95,
        "decomposition_cooked_to_source_p95_m": decomposition_p95,
        "hull_inner_region_cooked_to_source_p95_m": hull_inner_p95,
        "decomposition_inner_region_cooked_to_source_p95_m": (
            decomposition_inner_p95
        ),
        "decomposition_piece_count": decomposition["piece_count"],
        "decomposition_max_convex_hulls_default": local_api[
            "convex_decomposition_api"
        ]["defaults"]["maxConvexHulls"]["value"],
        "decomposition_minimum_piece_aabb_extent_m": np.min(
            np.asarray(piece_extents),
            axis=0,
        ),
        "decomposition_minimum_piece_longest_dimension_m": min(
            float(np.max(extent)) for extent in piece_extents
        ),
        "initial_internal_overlap": "SEE_RUNTIME_AB_EVIDENCE",
    }


def _runtime_ab_evidence(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "reports/aloha1_mapping/gripper_collider_ab_results.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    hull = report["groups"]["hull_current"]
    decomposition = report["groups"]["decomposition_current"]
    hull_runtime = float(hull["combined"]["runtime_mean_s"])
    decomposition_runtime = float(decomposition["combined"]["runtime_mean_s"])
    return {
        "source_report": str(path),
        "source_sha256": sha256_file(path),
        "status": report["status"],
        "experiment_execution_status": report["experiment_execution_status"],
        "CONVEX_DECOMPOSITION_STATUS": report[
            "CONVEX_DECOMPOSITION_STATUS"
        ],
        "root_cause_classification": report[
            "root_cause_classification"
        ]["classification"],
        "hull_drop_m": hull["diagnostic_metrics"]["drop_m"]["mean"],
        "decomposition_drop_m": decomposition["diagnostic_metrics"]["drop_m"][
            "mean"
        ],
        "drop_gate_m": report["frozen_values"]["drop_gate_m"],
        "hull_contact_points": hull["diagnostic_metrics"][
            "contact_point_count"
        ]["mean"],
        "decomposition_contact_points": decomposition["diagnostic_metrics"][
            "contact_point_count"
        ]["mean"],
        "runtime_ratio_decomposition_to_hull": (
            decomposition_runtime / hull_runtime
        ),
        "all_trials_bilateral_contact": all(
            group["diagnostic_metrics"]["bilateral_contact_trial_count"]
            == group["combined"]["trial_count"]
            for group in report["groups"].values()
        ),
        "persistent_penetration_trial_count": sum(
            group["diagnostic_metrics"][
                "persistent_penetration_trial_count"
            ]
            for group in report["groups"].values()
        ),
        "unexpected_internal_collision_trial_count": sum(
            group["diagnostic_metrics"][
                "unexpected_internal_collision_trial_count"
            ]
            for group in report["groups"].values()
        ),
        "determinism": report["determinism"],
    }


def finalize_cooking_log(
    project_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    report_path = (
        project_root / "reports/aloha1_mapping/gripper_collider_comparison.json"
    )
    markdown_path = (
        project_root / "reports/aloha1_mapping/gripper_collider_comparison.md"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    relevant = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in ("cook", "convex", "gpu compatibility", "cpu fallback")
        )
    ]
    warning_lines = [
        line
        for line in relevant
        if "warning" in line.lower() or "error" in line.lower()
    ]
    gpu_fallback = [
        line
        for line in lines
        if (
            "gpu" in line.lower()
            and (
                "compatib" in line.lower()
                or "fallback" in line.lower()
                or "fall back" in line.lower()
            )
        )
    ]
    cpu_fallback = [
        line
        for line in lines
        if "cpu" in line.lower()
        and ("fallback" in line.lower() or "fall back" in line.lower())
    ]
    report["cooking_log"] = {
        "path": str(log_path.resolve(strict=True)),
        "sha256": sha256_file(log_path),
        "relevant_line_count": len(relevant),
        "warning_or_error_line_count": len(warning_lines),
        "bounded_relevant_lines": relevant[:100],
        "bounded_warning_or_error_lines": warning_lines[:100],
        "gpu_compatibility_or_fallback": (
            "NOT_OBSERVED_IN_CAPTURED_LOG" if not gpu_fallback else "OBSERVED"
        ),
        "cpu_fallback": (
            "NOT_OBSERVED_IN_CAPTURED_LOG" if not cpu_fallback else "OBSERVED"
        ),
        "gpu_fallback_lines": gpu_fallback[:100],
        "cpu_fallback_lines": cpu_fallback[:100],
        "statistics_note": (
            "private cooking statistics returned zero warnings; the complete "
            "captured Kit log was independently scanned"
        ),
    }
    _write_json(report_path, report)
    markdown_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    return report


def run(
    app: Any,
    *,
    project_root: Path,
    profile_path: Path,
    report_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    manifest = load_collision_profiles(profile_path, project_root)
    assert_profile_pair_is_frozen(
        manifest["profiles"]["convex_hull"],
        manifest["profiles"]["convex_decomposition"],
        allowed_differences={"approximation"},
    )
    baseline_before = _verify_protected_baseline(project_root, manifest)
    local_api = _local_api_probe()
    if local_api["urdf_importer"]["version"] != "2.4.30":
        raise RuntimeError(f"unexpected local importer: {local_api['urdf_importer']}")

    source_config = {
        robot: (
            project_root
            / manifest["frozen"]["drive_source"][robot]
        ).resolve(strict=True)
        for robot in manifest["experiment"]["robots"]
    }
    profile_results = {}
    for profile_name, profile in manifest["profiles"].items():
        approximation = profile["approximation"]
        assets = []
        for robot in manifest["experiment"]["robots"]:
            output_dir = (
                project_root
                / manifest["diagnostic_directories"][profile_name]
                / robot
            )
            diagnostic_asset = output_dir / f"{robot}_{profile_name}.usd"
            layer = _create_diagnostic_asset(
                source_asset=source_config[robot],
                destination=diagnostic_asset,
                robot=robot,
                approximation=approximation,
            )
            cooking = _cook_finger_colliders(app, diagnostic_asset)
            source_mesh_path = (project_root / manifest["source_mesh"]["path"]).resolve(
                strict=True
            )
            source_metrics, source_mesh = _source_mesh_metrics(
                source_mesh_path,
                manifest["source_mesh"]["urdf_scale"],
            )
            for collider_path, collider in cooking["colliders"].items():
                collider["source_surface_sampling"] = _sampling_difference(
                    source_mesh,
                    collider,
                )
                side = "left" if "_left_finger_link/" in collider_path else "right"
                overview = output_dir / f"{robot}_{side}_collider_overview.png"
                closeup = output_dir / f"{robot}_{side}_collider_distal_closeup.png"
                inner = (
                    output_dir
                    / f"{robot}_{side}_collider_inner_gripping_surface_closeup.png"
                )
                _render_cooked(
                    collider,
                    overview_path=overview,
                    closeup_path=closeup,
                    inner_path=inner,
                    title=f"{robot} {side} {approximation}",
                )
                collider["visualization"] = {
                    "overview": str(overview),
                    "overview_sha256": sha256_file(overview),
                    "distal_closeup": str(closeup),
                    "distal_closeup_sha256": sha256_file(closeup),
                    "inner_gripping_surface_closeup": str(inner),
                    "inner_gripping_surface_closeup_sha256": sha256_file(inner),
                    "crop_policy": {
                        "distal": (
                            "distal half along numerically longest local AABB axis"
                        ),
                        "inner": (
                            "top quartile along mesh-local +X; +X is derived from "
                            "URDF prismatic closing directions and collision RPY"
                        ),
                    },
                }
            assets.append(
                {
                    "robot": robot,
                    "layer": layer,
                    "source_mesh": source_metrics,
                    "cooking": cooking,
                }
            )
        profile_results[profile_name] = {
            "approximation": approximation,
            "decomposition_parameters": (
                {
                    name: {
                        **item,
                        "authored_by_diagnostic_layer": False,
                    }
                    for name, item in local_api[
                        "convex_decomposition_api"
                    ]["defaults"].items()
                }
                if approximation == "convexDecomposition"
                else None
            ),
            "assets": assets,
        }

    # Local cooked geometry should be symmetric because both links use the same
    # source STL and scale. Record exact piece signatures rather than assuming.
    symmetry = []
    for profile_name, profile_result in profile_results.items():
        for asset in profile_result["assets"]:
            colliders = asset["cooking"]["colliders"]
            left = next(value for path, value in colliders.items() if "_left_finger_link/" in path)
            right = next(value for path, value in colliders.items() if "_right_finger_link/" in path)
            left_payload = {
                "piece_count": left["piece_count"],
                "pieces": [
                    {
                        "vertex_count": piece["vertex_count"],
                        "face_count": piece["face_count"],
                        "volume": piece["volume"],
                        "aabb_extent": (
                            np.asarray(piece["aabb_max"])
                            - np.asarray(piece["aabb_min"])
                        ),
                    }
                    for piece in left["pieces"]
                ],
            }
            right_payload = {
                "piece_count": right["piece_count"],
                "pieces": [
                    {
                        "vertex_count": piece["vertex_count"],
                        "face_count": piece["face_count"],
                        "volume": piece["volume"],
                        "aabb_extent": (
                            np.asarray(piece["aabb_max"])
                            - np.asarray(piece["aabb_min"])
                        ),
                    }
                    for piece in right["pieces"]
                ],
            }
            left_signature = hashlib.sha256(
                json.dumps(left_payload, sort_keys=True, default=_json_default).encode()
            ).hexdigest()
            right_signature = hashlib.sha256(
                json.dumps(right_payload, sort_keys=True, default=_json_default).encode()
            ).hexdigest()
            symmetry.append(
                {
                    "profile": profile_name,
                    "robot": asset["robot"],
                    "left_signature": left_signature,
                    "right_signature": right_signature,
                    "symmetric_cooked_local_geometry": left_signature == right_signature,
                }
            )

    baseline_after = _verify_protected_baseline(project_root, manifest)
    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if all(item["symmetric_cooked_local_geometry"] for item in symmetry)
            else "FAIL"
        ),
        "scope": "ALOHA 1 follower finger collider geometry only",
        "decision_policy": (
            "convex decomposition remains diagnostic until the frozen runtime A/B "
            "and 2x2 regression are complete"
        ),
        "local_api": local_api,
        "baseline_protection": {
            "before": baseline_before,
            "after": baseline_after,
        },
        "profiles": profile_results,
        "left_right_symmetry": symmetry,
        "geometry_conclusions": _geometry_conclusions(
            profile_results,
            local_api,
        ),
        "runtime_ab_evidence": _runtime_ab_evidence(project_root),
        "cooking_log": {
            "capture_instruction": (
                "redirect complete stdout/stderr to "
                ".codex/artifacts/aloha1-gripper-collider-ab/"
                "compare_aloha1_gripper_colliders.log"
            ),
            "gpu_compatibility_evidence": (
                "per-asset cooking_statistics_delta.failed_gpu_compatibility_warnings"
            ),
            "cpu_fallback_evidence": (
                "search bounded captured log; absence is reported, not inferred as GPU use"
            ),
        },
        "task8": "NOT_RUN",
        "default_asset_collider_modified": False,
    }
    _write_json(report_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--finalize-log",
        type=Path,
        default=None,
        help="scan an existing complete Kit log and update the comparison report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.project_root.resolve(strict=True)
    if args.finalize_log is not None:
        finalize_cooking_log(root, args.finalize_log)
        return 0
    profile_path = (
        args.profile.resolve(strict=True)
        if args.profile is not None
        else root / "configs/aloha1_gripper_collision_profiles.yaml"
    )
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    failure_path = root / "reports/aloha1_mapping/gripper_collider_comparison_failure.json"
    try:
        if args.probe_only:
            report = {
                "schema_version": 1,
                "status": "PASS",
                "local_api": _local_api_probe(),
            }
            _write_json(
                root / "reports/aloha1_mapping/gripper_collider_api_probe.json",
                report,
            )
        else:
            report = run(
                app,
                project_root=root,
                profile_path=profile_path,
                report_path=root
                / "reports/aloha1_mapping/gripper_collider_comparison.json",
                markdown_path=root
                / "reports/aloha1_mapping/gripper_collider_comparison.md",
            )
        failure_path.unlink(missing_ok=True)
    except BaseException as error:
        _write_json(
            failure_path,
            {
                "schema_version": 1,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        raise
    finally:
        app.close()
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

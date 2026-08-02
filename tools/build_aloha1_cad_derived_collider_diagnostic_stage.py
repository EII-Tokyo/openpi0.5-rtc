#!/usr/bin/env python3
"""Author an isolated OpenUSD layer stack for CAD-derived ALOHA colliders.

Run this script with the OpenUSD Python bindings bundled with Isaac Sim 5.1.
It never edits the source Stage or any final/default robot asset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics
from pxr import Vt

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
SOURCE_STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
SOURCE_STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
GEOMETRY_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json"
SEMANTICS_REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_link_collision_semantics.json"
MCP_SUMMARY = (
    ROOT / ".codex/artifacts/20260802-aloha1-cad-derived-colliders/commands/"
    "phase4_direct_nvidia_mcp_and_local_api_summary.json"
)
GEOMETRY_LAYER = ASSET_ROOT / "geometry/cad_derived_colliders.usda"
PHYSICS_LAYER = ASSET_ROOT / "physics/cad_derived_colliders_physics.usda"
ROOT_LAYER = ASSET_ROOT / "aloha1_cad_derived_full_body_collider_diagnostic.usda"
GRIPPER_DECOMP_PHYSICS_LAYER = ASSET_ROOT / "physics/cad_derived_colliders_gripper_decomposition_physics.usda"
GRIPPER_DECOMP_ROOT_LAYER = ASSET_ROOT / "aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda"
TABLETOP_ZERO_ROOT_LAYER = (
    ASSET_ROOT
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)
Z_UP_METERS_ROOT_LAYER = (
    ASSET_ROOT
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
)
TABLETOP_ZERO_LAYER = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/"
    "configuration/aloha1_tabletop_world_zero.usda"
)
REPORT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_collider_stage.json"
SUPPORTED_SUFFIXES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "gripper_link",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_obj(path: Path) -> tuple[list[Gf.Vec3f], list[int]]:
    points: list[Gf.Vec3f] = []
    indices: list[int] = []
    for line in path.resolve(strict=True).read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            values = [float(value) for value in line.split()[1:4]]
            points.append(Gf.Vec3f(*values))
        elif line.startswith("f "):
            face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            if len(face) != 3:
                raise ValueError(f"non-triangle OBJ face in {path}")
            indices.extend(face)
    if not points or not indices:
        raise ValueError(f"empty OBJ: {path}")
    return points, indices


def _connected_mesh_components(points: list[Gf.Vec3f], indices: list[int]) -> list[tuple[list[Gf.Vec3f], list[int]]]:
    """Split a triangle mesh into deterministic vertex-connected pieces."""

    parent = list(range(len(points)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    faces = [indices[index : index + 3] for index in range(0, len(indices), 3)]
    for first, second, third in faces:
        union(first, second)
        union(second, third)
    grouped: dict[int, list[list[int]]] = {}
    for face in faces:
        grouped.setdefault(find(face[0]), []).append(face)

    components = []
    for _, component_faces in sorted(grouped.items()):
        used = sorted({index for face in component_faces for index in face})
        remap = {source: target for target, source in enumerate(used)}
        component_points = [points[index] for index in used]
        component_indices = [remap[index] for face in component_faces for index in face]
        components.append((component_points, component_indices))
    components.sort(
        key=lambda item: (
            min(tuple(float(value) for value in point) for point in item[0]),
            len(item[0]),
            len(item[1]),
        )
    )
    return components


def _source_layer_hashes(stage: Usd.Stage) -> dict[str, str]:
    records = {}
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        real_path = Path(layer.realPath)
        if real_path.is_file():
            records[str(real_path.resolve())] = _sha256(real_path)
    return dict(sorted(records.items()))


def _api_signature(stage: Usd.Stage, api: type) -> list[dict[str, Any]]:
    records = []
    for prim in stage.Traverse():
        if not prim.HasAPI(api):
            continue
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "attributes": {
                    attr.GetName(): repr(attr.Get()) for attr in prim.GetAttributes() if attr.HasAuthoredValueOpinion()
                },
                "relationships": {
                    rel.GetName(): [str(target) for target in rel.GetTargets()]
                    for rel in prim.GetRelationships()
                    if rel.HasAuthoredTargets()
                },
            }
        )
    return records


def _joint_signature(stage: Usd.Stage) -> list[dict[str, Any]]:
    return [
        {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "attributes": {
                attr.GetName(): repr(attr.Get()) for attr in prim.GetAttributes() if attr.HasAuthoredValueOpinion()
            },
            "relationships": {
                rel.GetName(): [str(target) for target in rel.GetTargets()]
                for rel in prim.GetRelationships()
                if rel.HasAuthoredTargets()
            },
        }
        for prim in stage.Traverse()
        if prim.GetTypeName()
        in {
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
            "PhysicsFixedJoint",
        }
    ]


def _define_mesh(
    stage: Usd.Stage,
    mesh_path: str,
    points: list[Gf.Vec3f],
    indices: list[int],
) -> None:
    parent = Sdf.Path(mesh_path).GetParentPath()
    for ancestor in parent.GetPrefixes():
        stage.OverridePrim(ancestor)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)


def _author_geometry(records: list[dict[str, Any]], fallback_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage = Usd.Stage.CreateInMemory("cad_derived_colliders_geometry")
    authored_records = []
    for record in records:
        points, indices = _load_obj(Path(record["source_obj_path"]))
        components = _connected_mesh_components(points, indices)
        if len(components) != record["expected_piece_count"]:
            raise RuntimeError(
                f"component count drift for {record['robot']}/"
                f"{record['link_suffix']}: {len(components)} != "
                f"{record['expected_piece_count']}"
            )
        for piece_index, (piece_points, piece_indices) in enumerate(components):
            mesh_path = f"{record['prim_root_path']}/piece_{piece_index:03d}/mesh"
            _define_mesh(stage, mesh_path, piece_points, piece_indices)
            authored_records.append(
                {
                    **record,
                    "prim_path": mesh_path,
                    "piece_index": piece_index,
                    "piece_count": len(components),
                    "source_kind": "CAD_DERIVED",
                }
            )
    for record in fallback_records:
        _define_mesh(
            stage,
            record["prim_path"],
            record["points"],
            record["indices"],
        )
        authored_records.append({key: value for key, value in record.items() if key not in {"points", "indices"}})
    if not stage.GetRootLayer().Export(str(GEOMETRY_LAYER)):
        raise RuntimeError(f"failed to export {GEOMETRY_LAYER}")
    return authored_records


def _author_physics(
    mesh_records: list[dict[str, Any]],
    deactivated_source_paths: list[str],
    *,
    output: Path,
    decompose_gripper: bool,
) -> None:
    stage = Usd.Stage.CreateInMemory("cad_derived_colliders_physics")
    for source_path in deactivated_source_paths:
        stage.OverridePrim(source_path).SetActive(False)  # noqa: FBT003
    for record in mesh_records:
        mesh_path = record["prim_path"]
        prim = stage.OverridePrim(mesh_path)
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(True)  # noqa: FBT003
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        approximation = (
            UsdPhysics.Tokens.convexDecomposition
            if decompose_gripper
            and record.get("source_kind") == "CAD_DERIVED"
            and record.get("link_suffix") == "gripper_link"
            else UsdPhysics.Tokens.convexHull
        )
        mesh_collision.CreateApproximationAttr().Set(approximation)
    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"failed to export {output}")


def _author_root_layer(*, output: Path, physics_sublayer: str) -> list[str]:
    relative_paths = [
        physics_sublayer,
        "geometry/cad_derived_colliders.usda",
        "../../signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda",
    ]
    layer = Sdf.Layer.CreateAnonymous("cad_derived_collider_root")
    world = Sdf.CreatePrimInLayer(layer, "/World")
    world.specifier = Sdf.SpecifierOver
    layer.defaultPrim = "World"
    layer.subLayerPaths = relative_paths
    if not layer.Export(str(output)):
        raise RuntimeError(f"failed to export {output}")
    return relative_paths


def _author_tabletop_zero_runtime_wrapper() -> list[str]:
    """Compose the confirmed tabletop origin over the frozen collider Stage."""

    relative_paths = [
        (
            "../../table_support_alignment/1.0/configuration/"
            "aloha1_tabletop_world_zero.usda"
        ),
        GRIPPER_DECOMP_ROOT_LAYER.name,
    ]
    layer = Sdf.Layer.CreateAnonymous("cad_derived_collider_tabletop_zero_root")
    world = Sdf.CreatePrimInLayer(layer, "/World")
    world.specifier = Sdf.SpecifierOver
    layer.defaultPrim = "World"
    layer.subLayerPaths = relative_paths
    if not layer.Export(str(TABLETOP_ZERO_ROOT_LAYER)):
        raise RuntimeError(f"failed to export {TABLETOP_ZERO_ROOT_LAYER}")
    return relative_paths


def _author_z_up_meters_runtime_wrapper() -> list[str]:
    """Author explicit world-axis and unit metadata over the frozen Stage."""

    relative_paths = [TABLETOP_ZERO_ROOT_LAYER.name]
    temporary = Z_UP_METERS_ROOT_LAYER.with_suffix(".tmp.usda")
    if temporary.exists():
        raise RuntimeError(f"stale temporary layer exists: {temporary}")
    layer = Sdf.Layer.CreateNew(str(temporary))
    if layer is None:
        raise RuntimeError(f"failed to create temporary layer: {temporary}")
    world = Sdf.CreatePrimInLayer(layer, "/World")
    world.specifier = Sdf.SpecifierOver
    layer.defaultPrim = "World"
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise RuntimeError("failed to open Z-up/meters wrapper Stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    layer.subLayerPaths = relative_paths
    if not layer.Save():
        raise RuntimeError(f"failed to save {temporary}")
    temporary.replace(Z_UP_METERS_ROOT_LAYER)
    return relative_paths


def _readback_z_up_meters_composition() -> dict[str, Any]:
    old_stage = Usd.Stage.Open(str(TABLETOP_ZERO_ROOT_LAYER), load=Usd.Stage.LoadAll)
    new_stage = Usd.Stage.Open(str(Z_UP_METERS_ROOT_LAYER), load=Usd.Stage.LoadAll)
    if old_stage is None or new_stage is None:
        raise RuntimeError("failed to open Stage contract comparison inputs")
    old_paths = [str(prim.GetPath()) for prim in old_stage.Traverse()]
    new_paths = [str(prim.GetPath()) for prim in new_stage.Traverse()]
    maximum_delta = 0.0
    xformable_count = 0
    for path in old_paths:
        old_prim = old_stage.GetPrimAtPath(path)
        new_prim = new_stage.GetPrimAtPath(path)
        old_xform = UsdGeom.Xformable(old_prim)
        new_xform = UsdGeom.Xformable(new_prim)
        if not old_xform or not new_xform:
            continue
        old_matrix = old_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        new_matrix = new_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xformable_count += 1
        maximum_delta = max(
            maximum_delta,
            *(
                abs(float(old_matrix[row][column]) - float(new_matrix[row][column]))
                for row in range(4)
                for column in range(4)
            ),
        )
    return {
        "old_up_axis": str(UsdGeom.GetStageUpAxis(old_stage)),
        "old_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(old_stage)),
        "new_up_axis": str(UsdGeom.GetStageUpAxis(new_stage)),
        "new_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(new_stage)),
        "prim_paths_identical": old_paths == new_paths,
        "xformable_prim_count": xformable_count,
        "maximum_world_matrix_absolute_delta": maximum_delta,
    }


def _collider_records_from_geometry_report() -> list[dict[str, Any]]:
    geometry = json.loads(GEOMETRY_REPORT.read_text(encoding="utf-8"))
    if geometry["profile"] != "CAD_SUBPART_COMPOUND_CONVEX_HULL":
        raise ValueError("geometry profile drift")
    by_suffix = {
        record["link_suffix"]: record
        for record in geometry["physical_link_records"]
        if record["robot"] == "follower_left" and record["kind"] == "CAD_CANDIDATE" and record["status"] == "PASS"
    }
    if set(by_suffix) != set(SUPPORTED_SUFFIXES):
        raise ValueError(f"unexpected supported CAD link set: {sorted(by_suffix)}")
    records = []
    for side in ("left", "right"):
        robot = f"follower_{side}"
        for suffix in SUPPORTED_SUFFIXES:
            source = by_suffix[suffix]
            records.append(
                {
                    "robot": robot,
                    "link_suffix": suffix,
                    "source_obj_path": source["output_obj"]["absolute_path"],
                    "source_obj_sha256": source["output_obj"]["sha256"],
                    "expected_piece_count": source["convex_piece_count"],
                    "prim_root_path": (
                        f"/World/{robot}/vx300s_{side}/{robot}_{suffix}/cad_derived_collisions/cad_derived_{suffix}"
                    ),
                }
            )
    return records


def _fallback_records_from_source(stage: Usd.Stage) -> list[dict[str, Any]]:
    """Materialize four audited instance-proxy fallback meshes in link space."""

    traverse_proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    records = []
    for side in ("left", "right"):
        robot = f"follower_{side}"
        for suffix in ("wrist_link", "gripper_prop_link"):
            owner_path = f"/World/{robot}/vx300s_{side}/{robot}_{suffix}"
            prefix = f"{owner_path}/collisions/"
            matches = [
                prim
                for prim in stage.Traverse(traverse_proxies)
                if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"expected one source fallback below {prefix}; found {len(matches)}")
            source = matches[0]
            mesh_prims = [prim for prim in Usd.PrimRange(source) if prim.IsA(UsdGeom.Mesh)]
            if len(mesh_prims) != 1:
                raise RuntimeError(f"expected one mesh below fallback {source.GetPath()}; found {len(mesh_prims)}")
            source_mesh = mesh_prims[0]
            mesh = UsdGeom.Mesh(source_mesh)
            points = mesh.GetPointsAttr().Get() or []
            counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            if not points or not indices or any(int(value) != 3 for value in counts):
                raise RuntimeError(f"fallback is not a non-empty triangle mesh: {source.GetPath()}")
            source_world = UsdGeom.Xformable(source_mesh).ComputeLocalToWorldTransform(0.0)
            owner = stage.GetPrimAtPath(owner_path)
            owner_inverse = UsdGeom.Xformable(owner).ComputeLocalToWorldTransform(0.0).GetInverse()
            local_points = [
                Gf.Vec3f(owner_inverse.Transform(source_world.Transform(Gf.Vec3d(*point)))) for point in points
            ]
            records.append(
                {
                    "robot": robot,
                    "link_suffix": suffix,
                    "source_kind": "IMPORTER_BASELINE_FALLBACK",
                    "source_instance_proxy_path": str(source.GetPath()),
                    "prim_path": (f"{owner_path}/materialized_baseline_fallback/{suffix}/mesh"),
                    "points": local_points,
                    "indices": [int(value) for value in indices],
                    "approximation": UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get(),
                    "source_point_count": len(points),
                    "source_face_count": len(counts),
                }
            )
    return records


def main() -> int:
    if _sha256(SOURCE_STAGE) != SOURCE_STAGE_SHA256:
        raise ValueError("approved source Stage hash drift")
    if json.loads(MCP_SUMMARY.read_text(encoding="utf-8"))["status"] != "PASS":
        raise ValueError("direct NVIDIA MCP/local API gate did not pass")

    source_stage = Usd.Stage.Open(str(SOURCE_STAGE), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"cannot open source Stage {SOURCE_STAGE}")
    source_hashes_before = _source_layer_hashes(source_stage)
    source_joint_signature = _joint_signature(source_stage)
    source_rigid_signature = _api_signature(source_stage, UsdPhysics.RigidBodyAPI)
    source_mass_signature = _api_signature(source_stage, UsdPhysics.MassAPI)

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    GEOMETRY_LAYER.parent.mkdir(parents=True, exist_ok=True)
    PHYSICS_LAYER.parent.mkdir(parents=True, exist_ok=True)
    records = _collider_records_from_geometry_report()
    fallback_records = _fallback_records_from_source(source_stage)
    deactivated_source_paths = [
        f"/World/follower_{side}/vx300s_{side}/follower_{side}_{suffix}/collisions"
        for side in ("left", "right")
        for suffix in (
            *SUPPORTED_SUFFIXES,
            "wrist_link",
            "gripper_prop_link",
            "gripper_bar_link",
        )
    ]
    authored_records = _author_geometry(records, fallback_records)
    _author_physics(
        authored_records,
        deactivated_source_paths,
        output=PHYSICS_LAYER,
        decompose_gripper=False,
    )
    sublayers = _author_root_layer(
        output=ROOT_LAYER,
        physics_sublayer="physics/cad_derived_colliders_physics.usda",
    )
    _author_physics(
        authored_records,
        deactivated_source_paths,
        output=GRIPPER_DECOMP_PHYSICS_LAYER,
        decompose_gripper=True,
    )
    decomp_sublayers = _author_root_layer(
        output=GRIPPER_DECOMP_ROOT_LAYER,
        physics_sublayer=("physics/cad_derived_colliders_gripper_decomposition_physics.usda"),
    )
    tabletop_zero_sublayers = _author_tabletop_zero_runtime_wrapper()
    z_up_meters_sublayers = _author_z_up_meters_runtime_wrapper()
    z_up_meters_readback = _readback_z_up_meters_composition()

    if _sha256(SOURCE_STAGE) != SOURCE_STAGE_SHA256:
        raise RuntimeError("source Stage mutated during diagnostic build")
    source_stage_after = Usd.Stage.Open(str(SOURCE_STAGE), load=Usd.Stage.LoadAll)
    source_hashes_after = _source_layer_hashes(source_stage_after)
    source_hashes_unchanged = source_hashes_before == source_hashes_after
    if not source_hashes_unchanged:
        raise RuntimeError("source layer hash drift during diagnostic build")

    stage = Usd.Stage.Open(str(ROOT_LAYER), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open diagnostic Stage {ROOT_LAYER}")
    readback = []
    materialized_fallback = []
    for record in authored_records:
        prim = stage.GetPrimAtPath(record["prim_path"])
        mesh = UsdGeom.Mesh(prim)
        collision = UsdPhysics.CollisionAPI(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        result = {
            **record,
            "type_name": prim.GetTypeName(),
            "collision_enabled": collision.GetCollisionEnabledAttr().Get(),
            "approximation": mesh_collision.GetApproximationAttr().Get(),
            "purpose": mesh.GetPurposeAttr().Get(),
            "point_count": len(mesh.GetPointsAttr().Get() or []),
            "face_count": len(mesh.GetFaceVertexCountsAttr().Get() or []),
        }
        if result["source_kind"] == "CAD_DERIVED":
            readback.append(result)
        else:
            materialized_fallback.append(result)

    finger_readback = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "diagnostic_supplier_cad_" in path and path.endswith("/mesh") and prim.HasAPI(UsdPhysics.CollisionAPI):
            finger_readback.append(
                {
                    "prim_path": path,
                    "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                }
            )

    traverse_proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    source_proxy_colliders = [
        str(prim.GetPath())
        for prim in source_stage.Traverse(traverse_proxies)
        if prim.IsInstanceProxy()
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and (
            str(prim.GetPath()).startswith("/World/follower_left/")
            or str(prim.GetPath()).startswith("/World/follower_right/")
        )
    ]
    deactivated_readback = []
    for path in deactivated_source_paths:
        prim = stage.GetPrimAtPath(path)
        deactivated_readback.append(
            {
                "prim_path": path,
                "is_valid": prim.IsValid(),
                "is_active": prim.IsActive() if prim.IsValid() else False,
            }
        )
    baseline_fallback = materialized_fallback

    semantics = json.loads(SEMANTICS_REPORT.read_text(encoding="utf-8"))
    virtual_paths = [
        item["usd_prim_path"] for item in semantics["links"] if item["classification"] == "VIRTUAL_FRAME_NO_COLLIDER"
    ]
    virtual_collider_count = sum(
        1
        for path in virtual_paths
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path))
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    blocked_link_paths = [
        f"/World/follower_{side}/vx300s_{side}/follower_{side}_{suffix}"
        for side in ("left", "right")
        for suffix in ("wrist_link", "gripper_prop_link")
    ]
    blocked_authored = [
        str(prim.GetPath())
        for path in blocked_link_paths
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path))
        if "cad_derived_" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    all_articulation_roots = sorted(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    articulation_roots = [path for path in all_articulation_roots if path.startswith("/World/follower_")]
    new_paths = [item["prim_path"] for item in readback]
    required_paths = [
        "/World",
        "/World/follower_left/vx300s_left",
        "/World/follower_right/vx300s_right",
        *new_paths,
        *(item["prim_path"] for item in materialized_fallback),
    ]
    diagnostic_joint_signature = _joint_signature(stage)
    diagnostic_rigid_signature = _api_signature(stage, UsdPhysics.RigidBodyAPI)
    diagnostic_mass_signature = _api_signature(stage, UsdPhysics.MassAPI)
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "status_reason": (
            "CAD-derived colliders are authored for six main links per follower; "
            "wrist B-Rep validity and moving gripper_prop identity remain blocked."
        ),
        "source_stage": {
            "absolute_path": str(SOURCE_STAGE.resolve()),
            "sha256_before": SOURCE_STAGE_SHA256,
            "sha256_after": _sha256(SOURCE_STAGE),
        },
        "source_layer_hashes_before": source_hashes_before,
        "source_layer_hashes_after": source_hashes_after,
        "source_hashes_unchanged": source_hashes_unchanged,
        "root_layer": {
            "absolute_path": str(ROOT_LAYER.resolve()),
            "sha256": _sha256(ROOT_LAYER),
            "default_prim": "/World",
            "sublayers": sublayers,
        },
        "geometry_layer": {
            "absolute_path": str(GEOMETRY_LAYER.resolve()),
            "sha256": _sha256(GEOMETRY_LAYER),
        },
        "physics_layer": {
            "absolute_path": str(PHYSICS_LAYER.resolve()),
            "sha256": _sha256(PHYSICS_LAYER),
        },
        "gripper_decomposition_diagnostic_variant": {
            "classification": "DIAGNOSTIC_ONLY_NOT_FINAL_COLLIDER",
            "changed_variable": ("ONLY_GRIPPER_LINK_CAD_PIECES_APPROXIMATION_CONVEX_HULL_TO_CONVEX_DECOMPOSITION"),
            "root_layer": {
                "absolute_path": str(GRIPPER_DECOMP_ROOT_LAYER.resolve()),
                "sha256": _sha256(GRIPPER_DECOMP_ROOT_LAYER),
                "sublayers": decomp_sublayers,
            },
            "physics_layer": {
                "absolute_path": str(GRIPPER_DECOMP_PHYSICS_LAYER.resolve()),
                "sha256": _sha256(GRIPPER_DECOMP_PHYSICS_LAYER),
            },
            "geometry_layer_sha256": _sha256(GEOMETRY_LAYER),
            "material_modified": False,
            "drive_modified": False,
            "timestep_modified": False,
        },
        "tabletop_zero_runtime_wrapper": {
            "classification": (
                "DIAGNOSTIC_ONLY_TABLETOP_ZERO_NOT_FINAL_COLLIDER"
            ),
            "absolute_path": str(TABLETOP_ZERO_ROOT_LAYER.resolve()),
            "sha256": _sha256(TABLETOP_ZERO_ROOT_LAYER),
            "default_prim": "/World",
            "sublayers": tabletop_zero_sublayers,
            "tabletop_configuration": {
                "absolute_path": str(TABLETOP_ZERO_LAYER.resolve()),
                "sha256": _sha256(TABLETOP_ZERO_LAYER),
            },
            "source_stage_modified": False,
            "final_or_default_asset_modified": False,
            "task8": "NOT_RUN",
        },
        "z_up_meters_runtime_wrapper": {
            "classification": (
                "DIAGNOSTIC_ONLY_EXPLICIT_Z_UP_METERS_NOT_FINAL_COLLIDER"
            ),
            "absolute_path": str(Z_UP_METERS_ROOT_LAYER.resolve()),
            "sha256": _sha256(Z_UP_METERS_ROOT_LAYER),
            "default_prim": "/World",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "sublayers": z_up_meters_sublayers,
            "composed_readback": z_up_meters_readback,
            "source_stage_modified": False,
            "final_or_default_asset_modified": False,
            "task8": "NOT_RUN",
        },
        "direct_nvidia_mcp_local_api_evidence": {
            "absolute_path": str(MCP_SUMMARY.resolve()),
            "sha256": _sha256(MCP_SUMMARY),
            "status": "PASS",
        },
        "new_collider_readback": readback,
        "source_instance_proxy_collider_count_before": len(source_proxy_colliders),
        "source_instance_proxy_colliders_before": source_proxy_colliders,
        "deactivated_source_collision_instances": {
            "count": len(deactivated_readback),
            "records": deactivated_readback,
            "all_inactive": all(not item["is_active"] for item in deactivated_readback),
        },
        "baseline_fallback_collider_readback": {
            "count": len(baseline_fallback),
            "records": baseline_fallback,
            "all_convex_hull": all(item["approximation"] == "convexHull" for item in baseline_fallback),
            "all_enabled": all(item["collision_enabled"] is True for item in baseline_fallback),
        },
        "materialized_fallback_collider_readback": {
            "count": len(materialized_fallback),
            "records": materialized_fallback,
            "all_convex_hull": all(item["approximation"] == "convexHull" for item in materialized_fallback),
            "all_enabled": all(item["collision_enabled"] is True for item in materialized_fallback),
            "reason": (
                "Isaac runtime enumeration did not expose the four required "
                "instance-proxy fallback shapes; exact authored mesh points "
                "were materialized into owning-link local space in this "
                "isolated diagnostic layer."
            ),
        },
        "compound_piece_count_by_link_suffix": {
            suffix: sum(item["link_suffix"] == suffix for item in readback if item["robot"] == "follower_left")
            for suffix in SUPPORTED_SUFFIXES
        },
        "existing_finger_collider_readback": {
            "count": len(finger_readback),
            "records": finger_readback,
            "all_convex_hull": all(item["approximation"] == "convexHull" for item in finger_readback),
        },
        "virtual_frame_collider_count": virtual_collider_count,
        "blocked_link_colliders_authored": blocked_authored,
        "blocked_physical_links": [Path(path).name for path in blocked_link_paths],
        "gripper_bar_fixed_group_coverage": {
            "count": 2,
            "owner_collider_paths": [
                item["prim_path"]
                for item in readback
                if item["link_suffix"] == "gripper_link" and item["piece_index"] == 0
            ],
            "rule": (
                "supplier Part__Feature006 covers the fixed gripper+bar group; "
                "the source gripper_bar collision instance is deactivated in "
                "the diagnostic layer to avoid duplicate coverage"
            ),
        },
        "articulation_count": len(articulation_roots),
        "articulation_root_paths": articulation_roots,
        "all_stage_articulation_root_paths": all_articulation_roots,
        "non_robot_articulation_root_paths": [
            path for path in all_articulation_roots if not path.startswith("/World/follower_")
        ],
        "joint_signature_unchanged": source_joint_signature == diagnostic_joint_signature,
        "rigid_body_signature_unchanged": source_rigid_signature == diagnostic_rigid_signature,
        "mass_signature_unchanged": source_mass_signature == diagnostic_mass_signature,
        "required_prims": required_paths,
        "required_prims_valid": all(stage.GetPrimAtPath(path).IsValid() for path in required_paths),
        "stage_open_probe": "PASS",
        "collision_authoring_scope": ("NEW_DIAGNOSTIC_MESH_PRIMS_PLUS_SOURCE_INSTANCE_DEACTIVATION"),
        "duplicate_source_candidate_collision_gate": (
            "PASS" if all(not item["is_active"] for item in deactivated_readback) else "FAIL"
        ),
        "drive_modified": False,
        "material_modified": False,
        "timestep_modified": False,
        "source_or_imported_asset_modified": False,
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "new_colliders": len(readback),
                "articulations": len(articulation_roots),
                "root_layer": str(ROOT_LAYER.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

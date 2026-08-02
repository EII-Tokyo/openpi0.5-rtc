#!/usr/bin/env python3
"""Author a geometry-only USD for the isolated supplier-CAD contact candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_compound_contact_candidate import compound_piece_prim_path
from tools.aloha1_mapping.cad_compound_contact_candidate import convex_triangle_topology


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_asset(path: Path, owner: Path) -> str:
    return os.path.relpath(path.resolve(), owner.resolve().parent)


def _author_geometry(path: Path, geometry: dict[str, Any]) -> None:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/CadFingerCompoundContactCandidate").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("aloha1:geometrySignature", geometry["deterministic_signature"])
    root.SetCustomDataByKey("aloha1:coordinateFrame", geometry["output_coordinate_frame"])
    root.SetCustomDataByKey("aloha1:assetDecision", "DIAGNOSTIC_ONLY_NOT_PROMOTED")
    for side, finger in sorted(geometry["fingers"].items()):
        container = UsdGeom.Xform.Define(stage, f"/CadFingerCompoundContactCandidate/{side}_finger").GetPrim()
        container.SetCustomDataByKey(
            "aloha1:cadGlobalToFingerLinkMatrixJson",
            json.dumps(finger["rigid_transform_matrix"], separators=(",", ":")),
        )
        container.SetCustomDataByKey("aloha1:transformDeterminant", finger["rigid_transform_determinant"])
        container.SetCustomDataByKey("aloha1:mirrorUsed", False)  # noqa: FBT003
        for index, piece in enumerate(finger["pieces"]):
            vertices = np.asarray(piece["vertices"], dtype=np.float64)
            topology = convex_triangle_topology(vertices)
            mesh = UsdGeom.Mesh.Define(stage, compound_piece_prim_path(side, index))
            points = [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in vertices]
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(topology["face_vertex_counts"])
            mesh.CreateFaceVertexIndicesAttr(topology["face_vertex_indices"])
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
            mesh.CreateDoubleSidedAttr().Set(False)  # noqa: FBT003
            mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(points))
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            prim.SetCustomDataByKey("aloha1:sourcePieceIndex", index)
            prim.SetCustomDataByKey("aloha1:construction", piece["construction"])
    stage.GetRootLayer().Save()


def _author_root(path: Path, geometry_path: Path, geometry: dict[str, Any]) -> None:
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/CadFingerCompoundContactCandidate").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(
        _relative_asset(geometry_path, path),
        "/CadFingerCompoundContactCandidate",
    )
    root.SetCustomDataByKey("aloha1:geometrySignature", geometry["deterministic_signature"])
    root.SetCustomDataByKey("aloha1:assetDecision", "DIAGNOSTIC_ONLY_NOT_PROMOTED")
    root.SetCustomDataByKey("aloha1:timelineValidated", False)  # noqa: FBT003
    root.SetCustomDataByKey("aloha1:articulationIntegrated", False)  # noqa: FBT003
    stage.GetRootLayer().Save()


def _readback(root_path: Path) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(root_path))
    if stage is None:
        raise RuntimeError(f"failed to open candidate USD: {root_path}")
    collision_records = []
    forbidden = []
    for prim in Usd.PrimRange.Stage(stage):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            collision_records.append(
                {
                    "path": path,
                    "type_name": prim.GetTypeName(),
                    "approximation": mesh_api.GetApproximationAttr().Get(),
                    "point_count": len(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                }
            )
        for schema in (
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI,
        ):
            if prim.HasAPI(schema):
                forbidden.append(  # noqa: PERF401
                    {"path": path, "schema": schema.__name__}
                )
    default_prim = stage.GetDefaultPrim()
    return {
        "default_prim": str(default_prim.GetPath()),
        "root_prim_valid": default_prim.IsValid(),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "collision_piece_count": len(collision_records),
        "collision_records": collision_records,
        "all_approximations_convex_hull": all(record["approximation"] == "convexHull" for record in collision_records),
        "forbidden_dynamic_schemas": forbidden,
        "sublayers": list(stage.GetRootLayer().subLayerPaths),
        "references": [
            {
                "asset_path": item.assetPath,
                "prim_path": str(item.primPath),
            }
            for item in default_prim.GetMetadata("references").GetAddedOrExplicitItems()
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Supplier-CAD compound contact geometry-only USD",
            "",
            f"- Status: **{report['status']}**",
            f"- Root USD: `{report['outputs']['root_usd']['absolute_path']}`",
            f"- Geometry USD: `{report['outputs']['geometry_usd']['absolute_path']}`",
            f"- Collision pieces: `{report['readback']['collision_piece_count']}`",
            f"- Coordinate frame: `{report['coordinate_frame']}`",
            f"- Determinism: **{report['determinism']['status']}**",
            "- Approximation readback: `convexHull` for every already-convex piece",
            "- RigidBody/Mass/Articulation schemas: absent",
            "- Timeline/contact dynamics: **NOT_RUN**",
            "- Asset promotion: **DIAGNOSTIC_ONLY_NOT_PROMOTED**",
            "- Final/default collider modified: `false`",
            "",
            "This asset is a geometry-only composition candidate. It is ready to "
            "be referenced under a finger-link collision branch in a later isolated "
            "integration layer, but it is not itself an articulation or a grasp test.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-input", type=Path, required=True)
    parser.add_argument("--runtime-certificate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--determinism-run", type=Path, action="append")
    args = parser.parse_args()
    geometry_path = args.geometry_input.resolve(strict=True)
    runtime_path = args.runtime_certificate.resolve(strict=True)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if geometry["output_coordinate_frame"] != "FINGER_LINK_LOCAL_METRES":
        raise RuntimeError("USD candidate requires finger-link-local geometry")
    if runtime["status"] != "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED":
        raise RuntimeError("runtime cooking certificate has not passed")
    if runtime["coordinate_frame"] != "FINGER_LINK_LOCAL_METRES":
        raise RuntimeError("runtime certificate has the wrong coordinate frame")

    output_root = args.output_root.resolve()
    geometry_usd = output_root / "geometry/supplier_cad_compound_contact_candidate.usda"
    root_usd = output_root / "aloha1_supplier_cad_compound_contact_candidate.usda"
    geometry_usd.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        _author_geometry(geometry_usd, geometry)
        _author_root(root_usd, geometry_usd, geometry)
        readback = _readback(root_usd)
    except Exception as error:
        print(f"USD authoring failed: {type(error).__name__}: {error}", flush=True)
        app.close()
        raise
    expected_pieces = sum(len(finger["pieces"]) for finger in geometry["fingers"].values())
    determinism: dict[str, Any] = {"status": "NOT_REQUESTED_THIS_BUILD"}
    if args.determinism_run:
        if len(args.determinism_run) != 2:
            raise ValueError("exactly two USD determinism reports are required")
        deterministic_reports = [
            json.loads(path.resolve(strict=True).read_text(encoding="utf-8")) for path in args.determinism_run
        ]
        root_hashes = [record["outputs"]["root_usd"]["sha256"] for record in deterministic_reports]
        geometry_hashes = [record["outputs"]["geometry_usd"]["sha256"] for record in deterministic_reports]
        determinism = {
            "status": (
                "PASS_TWO_FRESH_BUILDS_BYTE_IDENTICAL"
                if len(set(root_hashes)) == 1
                and len(set(geometry_hashes)) == 1
                and all(record["status"] == "PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD" for record in deterministic_reports)
                else "FAIL_TWO_FRESH_BUILDS_BYTE_IDENTICAL"
            ),
            "reports": [str(path.resolve()) for path in args.determinism_run],
            "root_usd_hashes": root_hashes,
            "geometry_usd_hashes": geometry_hashes,
        }
    passed = (
        readback["default_prim"] == "/CadFingerCompoundContactCandidate"
        and readback["collision_piece_count"] == expected_pieces
        and readback["all_approximations_convex_hull"]
        and not readback["forbidden_dynamic_schemas"]
        and determinism["status"]
        in {
            "NOT_REQUESTED_THIS_BUILD",
            "PASS_TWO_FRESH_BUILDS_BYTE_IDENTICAL",
        }
    )
    report = {
        "schema_version": 1,
        "status": "PASS_GEOMETRY_ONLY_DIAGNOSTIC_USD" if passed else "FAIL",
        "scope": "GEOMETRY_ONLY_NOT_ARTICULATION_NOT_CONTACT_DYNAMICS",
        "inputs": {
            "geometry": {
                "absolute_path": str(geometry_path),
                "sha256": _sha256(geometry_path),
                "deterministic_signature": geometry["deterministic_signature"],
            },
            "runtime_certificate": {
                "absolute_path": str(runtime_path),
                "sha256": _sha256(runtime_path),
                "status": runtime["status"],
            },
        },
        "outputs": {
            "root_usd": {
                "absolute_path": str(root_usd),
                "sha256": _sha256(root_usd),
            },
            "geometry_usd": {
                "absolute_path": str(geometry_usd),
                "sha256": _sha256(geometry_usd),
            },
        },
        "coordinate_frame": geometry["output_coordinate_frame"],
        "determinism": determinism,
        "readback": readback,
        "asset_decision": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "articulation_integration_status": "NOT_RUN",
        "timeline_status": "NOT_RUN",
        "contact_dynamics_status": "NOT_RUN",
        "final_or_default_collider_modified": False,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "root_usd": str(root_usd)}))
    app.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

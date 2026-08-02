#!/usr/bin/env python3
"""Cook exact supplier-CAD finger meshes in an isolated Isaac 5.1 process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from tools.aloha1_mapping.collider_surface_certificate import _load_obj
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_supplier_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _geometry_signature(cooked: dict[str, Any]) -> str:
    payload = {
        "source_sha256": cooked["source_sha256"],
        "approximation_readback": cooked["approximation_readback"],
        "pieces": [
            {
                "vertices": piece["vertices"],
                "indices": piece["indices"],
                "polygons": piece["polygons"],
            }
            for piece in cooked["pieces"]
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_probe_stage(root: Path, side: str, approximation: str):
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    contact = load_supplier_contact_surface(root, side)
    vertices, faces = _load_obj(Path(contact["source_path"]))
    stage = Usd.Stage.CreateInMemory()
    root_prim = UsdGeom.Xform.Define(stage, "/SupplierCadFingerProbe").GetPrim()
    stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/SupplierCadFingerProbe/physicsScene")
    mesh = UsdGeom.Mesh.Define(stage, "/SupplierCadFingerProbe/Collider")
    mesh.CreatePointsAttr(
        [Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])) for point in vertices]
    )
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    prim = mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(1.0)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_api.CreateApproximationAttr().Set(approximation)
    decomposition_parameters_authored = None
    if approximation == "convexHull":
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
    elif approximation == "convexDecomposition":
        api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        getters = (
            api.GetMaxConvexHullsAttr,
            api.GetVoxelResolutionAttr,
            api.GetErrorPercentageAttr,
            api.GetShrinkWrapAttr,
            api.GetMinThicknessAttr,
            api.GetHullVertexLimitAttr,
        )
        decomposition_parameters_authored = any(
            getter().HasAuthoredValueOpinion() for getter in getters
        )
        if decomposition_parameters_authored:
            raise RuntimeError("default decomposition parameters were unexpectedly authored")
    else:
        raise ValueError(f"unsupported approximation: {approximation}")
    return stage, str(prim.GetPath()), contact, mesh_api.GetApproximationAttr().Get(), decomposition_parameters_authored


def _cook_one(app: Any, root: Path, side: str, approximation: str) -> dict[str, Any]:
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    from tools.compare_aloha1_gripper_colliders import _convex_piece_metrics
    from tools.compare_aloha1_gripper_colliders import _request_cooked_representation

    stage, collider_path, contact, readback, parameters_authored = _build_probe_stage(
        root, side, approximation
    )
    result, convexes, runtime_s = _request_cooked_representation(
        app, stage=stage, collider_path=collider_path
    )
    if result != PhysxCollisionRepresentationResult.RESULT_VALID:
        raise RuntimeError(f"cooking failed for {side}/{approximation}: {result}")
    pieces = [_convex_piece_metrics(piece) for piece in convexes]
    contact_summary = summarize_contact_envelope(
        contact["samples"],
        np.asarray(contact["normal"], dtype=np.float64),
        pieces,
        tessellation_budget_m=contact["tessellation_error_budget_m"],
    )
    cooked = {
        "side": side,
        "source_path": contact["source_path"],
        "source_sha256": contact["source_sha256"],
        "source_face_count": contact["source_face_count"],
        "cad_face_index": contact["cad_face_index"],
        "cad_face_normal": contact["normal"],
        "contact_sample_count": contact["sample_count"],
        "approximation_readback": readback,
        "decomposition_parameters_authored": parameters_authored,
        "result": str(result),
        "piece_count": len(pieces),
        "pieces": pieces,
        "sum_piece_volume_m3": float(
            sum(piece["volume"] for piece in pieces if piece["volume"] is not None)
        ),
        "contact_envelope": contact_summary,
        "runtime_s": runtime_s,
    }
    cooked["geometry_signature"] = _geometry_signature(cooked)
    return cooked


def run(app: Any, root: Path) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface

    from tools.compare_aloha1_gripper_colliders import _cooking_statistics
    from tools.compare_aloha1_gripper_colliders import _local_api_probe
    from tools.compare_aloha1_gripper_colliders import _subtract_stats

    local_api = _local_api_probe()
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    for key, expected in expected_runtime.items():
        if local_api[key] != expected:
            raise RuntimeError(f"unexpected {key}: {local_api[key]} != {expected}")
    cooking = get_physx_cooking_interface()
    cooking.release_local_mesh_cache()
    start_stats = _cooking_statistics()
    profiles: dict[str, Any] = {}
    for approximation in ("convexHull", "convexDecomposition"):
        profiles[approximation] = {
            side: _cook_one(app, root, side, approximation)
            for side in ("left", "right")
        }
    end_stats = _cooking_statistics()
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "IN_MEMORY_SUPPLIER_CAD_FINGER_COOKING_NO_TIMELINE_NO_ASSET_AUTHORING",
        "process_id": os.getpid(),
        "completed_unix_time": time.time(),
        "runtime": expected_runtime,
        "local_api": local_api,
        "profiles": profiles,
        "cooking_statistics_delta": _subtract_stats(end_stats, start_stats),
        "stage_saved": False,
        "timeline_started": False,
        "final_or_default_asset_modified": False,
        "real_robot_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        report = run(app, root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

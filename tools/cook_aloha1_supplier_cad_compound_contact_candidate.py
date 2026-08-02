#!/usr/bin/env python3
"""Cook the isolated CAD-derived compound finger candidate in Isaac Sim 5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_compound_contact_candidate import canonical_runtime_cooking_signature
from tools.aloha1_mapping.cad_compound_contact_candidate import convex_triangle_topology
from tools.aloha1_mapping.cad_compound_contact_candidate import runtime_contact_region_status
from tools.aloha1_mapping.cad_compound_contact_candidate import tolerance_adjusted_contact_coverage
from tools.aloha1_mapping.finger_cooked_contact_certificate import derive_cooked_brep_numeric_tolerance
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_exact_brep_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rectangle_samples(vertices: np.ndarray, count_per_axis: int = 17) -> np.ndarray:
    samples = [
        (1.0 - first) * (1.0 - second) * vertices[0]
        + first * (1.0 - second) * vertices[1]
        + first * second * vertices[2]
        + (1.0 - first) * second * vertices[3]
        for first in np.linspace(0.0, 1.0, count_per_axis)
        for second in np.linspace(0.0, 1.0, count_per_axis)
    ]
    return np.asarray(samples, dtype=np.float64)


def _author_probe_stage(geometry: dict[str, Any]):
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/CompoundContactProbe").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/CompoundContactProbe/physicsScene")
    authored: dict[str, list[dict[str, Any]]] = {}
    for side, finger in sorted(geometry["fingers"].items()):
        UsdGeom.Scope.Define(stage, f"/CompoundContactProbe/{side}")
        authored[side] = []
        for index, source_piece in enumerate(finger["pieces"]):
            vertices = np.asarray(source_piece["vertices"], dtype=np.float64)
            topology = convex_triangle_topology(vertices)
            path = f"/CompoundContactProbe/{side}/piece_{index:03d}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr([Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])) for point in vertices])
            mesh.CreateFaceVertexCountsAttr(topology["face_vertex_counts"])
            mesh.CreateFaceVertexIndicesAttr(topology["face_vertex_indices"])
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            prim = mesh.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(1.0)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            authored[side].append(
                {
                    "source_piece_index": index,
                    "source_construction": source_piece["construction"],
                    "path": path,
                    "source_vertex_count": len(vertices),
                    "authored_face_count": topology["face_count"],
                    "authored_volume_m3": topology["volume_m3"],
                    "approximation_readback": mesh_api.GetApproximationAttr().Get(),
                }
            )
    return stage, authored


def run(
    app: Any,
    *,
    geometry_path: Path,
    brep_paths: list[Path],
) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    from tools.compare_aloha1_gripper_colliders import _convex_piece_metrics
    from tools.compare_aloha1_gripper_colliders import _cooking_statistics
    from tools.compare_aloha1_gripper_colliders import _local_api_probe
    from tools.compare_aloha1_gripper_colliders import _request_cooked_representation
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

    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry["final_or_default_collider_modified"] is not False:
        raise RuntimeError("compound input is not isolated from final/default collider")
    if len(brep_paths) != 2:
        raise ValueError("exactly two deterministic BRep reports are required")

    stage, authored = _author_probe_stage(geometry)
    cooking = get_physx_cooking_interface()
    cooking.release_local_mesh_cache()
    start_stats = _cooking_statistics()
    fingers: dict[str, Any] = {}
    for side, records in authored.items():
        cooked_union = []
        cooked_records = []
        for record in records:
            result, convexes, runtime_s = _request_cooked_representation(
                app,
                stage=stage,
                collider_path=record["path"],
            )
            if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                raise RuntimeError(f"cooking failed for {record['path']}: {result}")
            pieces = [_convex_piece_metrics(piece) for piece in convexes]
            cooked_union.extend(pieces)
            cooked_records.append(
                {
                    **record,
                    "result": str(result),
                    "runtime_s": runtime_s,
                    "cooked": {
                        "piece_count": len(pieces),
                        "pieces": pieces,
                    },
                }
            )

        exact = load_exact_brep_contact_surface(brep_paths, side)
        exact_samples = np.asarray(exact["samples_m"], dtype=np.float64)
        exact_normal = np.asarray(exact["normal"], dtype=np.float64)
        if geometry.get("output_coordinate_frame") == "FINGER_LINK_LOCAL_METRES":
            transform = np.asarray(
                geometry["fingers"][side]["rigid_transform_matrix"],
                dtype=np.float64,
            )
            exact_samples = exact_samples @ transform[:3, :3].T + transform[:3, 3]
            exact_normal = transform[:3, :3] @ exact_normal
            exact_normal /= np.linalg.norm(exact_normal)
        tolerance = derive_cooked_brep_numeric_tolerance(
            exact_samples,
            brep_membership_tolerance_m=exact["brep_membership_tolerance_m"],
        )
        rectangle = np.asarray(
            geometry["fingers"][side]["contact_rectangle_vertices_m"],
            dtype=np.float64,
        )
        contact_certificate = tolerance_adjusted_contact_coverage(
            summarize_contact_envelope(
                _rectangle_samples(rectangle),
                np.asarray(geometry["fingers"][side]["outward_normal"]),
                cooked_union,
                tessellation_budget_m=tolerance["numeric_tolerance_m"],
            ),
            numeric_tolerance_m=tolerance["numeric_tolerance_m"],
        )
        full_face_certificate = summarize_contact_envelope(
            exact_samples,
            exact_normal,
            cooked_union,
            tessellation_budget_m=tolerance["numeric_tolerance_m"],
        )
        fingers[side] = {
            "source_piece_count": len(records),
            "runtime_cooked_piece_count": len(cooked_union),
            "pieces": cooked_records,
            "numeric_tolerance": tolerance,
            "contact_region_certificate": contact_certificate,
            "contact_region_status": runtime_contact_region_status(
                contact_certificate, tolerance["numeric_tolerance_m"]
            ),
            "full_brep_face_certificate": full_face_certificate,
            "full_face_scope": "PARTIAL_CONTACT_REGION_ONLY",
            "coordinate_frame": geometry.get("output_coordinate_frame", "STEP_ASSEMBLY_GLOBAL_METRES"),
        }
    end_stats = _cooking_statistics()
    all_pass = all(finger["contact_region_status"] == "PASS" for finger in fingers.values())
    report = {
        "schema_version": 1,
        "status": (
            "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
            if all_pass
            else "FAIL_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
        ),
        "scope": ("IN_MEMORY_CAD_DERIVED_COMPOUND_COOKING_NO_TIMELINE_NO_ASSET_AUTHORING"),
        "process_id": os.getpid(),
        "completed_unix_time": time.time(),
        "runtime": expected_runtime,
        "local_api": local_api,
        "input_geometry": {
            "absolute_path": str(geometry_path),
            "sha256": _sha256(geometry_path),
            "deterministic_signature": geometry["deterministic_signature"],
        },
        "brep_inputs": [{"absolute_path": str(path), "sha256": _sha256(path)} for path in brep_paths],
        "authoring": {
            "mesh_schema": "UsdGeom.Mesh",
            "collision_api": "UsdPhysics.CollisionAPI",
            "mesh_collision_api": "UsdPhysics.MeshCollisionAPI",
            "approximation": "convexHull",
            "physx_schema": "PhysxSchema.PhysxConvexHullCollisionAPI",
        },
        "fingers": fingers,
        "cooking_statistics_delta": _subtract_stats(end_stats, start_stats),
        "stage_saved": False,
        "timeline_started": False,
        "diagnostic_usd_created": False,
        "final_or_default_asset_modified": False,
        "candidate_promoted": False,
        "real_robot_accessed": False,
    }
    report["deterministic_signature"] = canonical_runtime_cooking_signature(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--brep-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = args.geometry.resolve(strict=True)
    brep_paths = [path.resolve(strict=True) for path in args.brep_run]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        report = run(app, geometry_path=geometry, brep_paths=brep_paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "deterministic_signature": report["deterministic_signature"],
                    "output": str(args.output),
                }
            )
        )
        return 0 if report["status"].startswith("PASS_") else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build isolated correct-custom-finger diagnostics for Isaac Sim 5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np

from tools.aloha1_mapping.correct_finger_asset import (
    EXPECTED_RESTART_BOUNDARY,
)
from tools.aloha1_mapping.correct_finger_asset import load_correct_finger_profile
from tools.aloha1_mapping.correct_finger_asset import sha256_file
from tools.aloha1_mapping.correct_finger_asset import verify_correct_finger_sources


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stl_triangle_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = 84 + count * 50
    if len(payload) != expected_size:
        raise RuntimeError(f"binary STL size mismatch: {path}")
    values = np.empty((count * 3, 3), dtype=np.float32)
    for index in range(count):
        triangle = struct.unpack_from("<9f", payload, 84 + index * 50 + 12)
        values[index * 3 : index * 3 + 3] = np.asarray(
            triangle,
            dtype=np.float32,
        ).reshape(3, 3)
    return values


def _historical_mesh(
    stage: Any,
    *,
    side: str,
    branch: str,
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom

    link_path = f"/workcell/vx300s_left/vx300s_left_{side}_finger_link"
    link = stage.GetPrimAtPath(link_path)
    base = stage.GetPrimAtPath(f"{link_path}/{branch}")
    if not link.IsValid() or not base.IsValid() or not base.IsInstance():
        raise RuntimeError(f"historical {side}/{branch} branch is unavailable")
    prototype = base.GetPrototype()
    meshes = [
        prim for prim in Usd.PrimRange(prototype) if prim.IsA(UsdGeom.Mesh)
    ]
    if len(meshes) != 1:
        raise RuntimeError(
            f"expected one historical {side}/{branch} mesh, found {len(meshes)}"
        )
    prototype_mesh = meshes[0]
    relative = prototype_mesh.GetPath().MakeRelativePath(prototype.GetPath())
    proxy_path = base.GetPath().AppendPath(relative)
    proxy = stage.GetPrimAtPath(proxy_path)
    mesh = UsdGeom.Mesh(proxy)
    xform_cache = UsdGeom.XformCache()
    relative_transform, reset_stack = xform_cache.ComputeRelativeTransform(
        proxy,
        link,
    )
    points = mesh.GetPointsAttr().Get() or []
    face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    normals = mesh.GetNormalsAttr().Get() or []
    if len(points) != 4998 or len(face_counts) != 1666:
        raise RuntimeError(
            f"historical mesh topology mismatch for {side}/{branch}: "
            f"points={len(points)} faces={len(face_counts)}"
        )
    return {
        "proxy_path": str(proxy_path),
        "points": points,
        "face_counts": face_counts,
        "face_indices": face_indices,
        "normals": normals,
        "normals_interpolation": mesh.GetNormalsInterpolation(),
        "orientation": mesh.GetOrientationAttr().Get(),
        "double_sided": mesh.GetDoubleSidedAttr().Get(),
        "subdivision_scheme": mesh.GetSubdivisionSchemeAttr().Get(),
        "relative_transform": relative_transform,
        "reset_xform_stack": bool(reset_stack),
    }


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def _collider_inventory(stage: Any) -> list[dict[str, Any]]:
    from pxr import Usd
    from pxr import UsdPhysics

    result = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        token = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            token = UsdPhysics.MeshCollisionAPI(
                prim
            ).GetApproximationAttr().Get()
        result.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "approximation": token,
            }
        )
    return sorted(result, key=lambda item: item["path"])


def _articulation_roots(stage: Any) -> list[str]:
    from pxr import Usd
    from pxr import UsdPhysics

    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange.Stage(stage)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )


def _author_mesh(
    stage: Any,
    *,
    path: str,
    source: dict[str, Any],
    side: str,
) -> Any:
    from pxr import Gf
    from pxr import UsdGeom

    container = UsdGeom.Xform.Define(stage, path)
    container.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble,
    ).Set(source["relative_transform"])
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    mesh.CreatePointsAttr(source["points"])
    mesh.CreateFaceVertexCountsAttr(source["face_counts"])
    mesh.CreateFaceVertexIndicesAttr(source["face_indices"])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if source["normals"]:
        mesh.CreateNormalsAttr(source["normals"])
        mesh.SetNormalsInterpolation(source["normals_interpolation"])
    if source["orientation"] is not None:
        mesh.CreateOrientationAttr().Set(source["orientation"])
    if source["double_sided"] is not None:
        mesh.CreateDoubleSidedAttr().Set(source["double_sided"])
    points = source["points"]
    mesh.CreateExtentAttr(UsdGeom.Mesh.ComputeExtent(points))
    color = Gf.Vec3f(0.08, 0.24, 0.82) if side == "left" else Gf.Vec3f(
        0.92,
        0.28,
        0.04,
    )
    mesh.CreateDisplayColorAttr([color])
    return container


def _create_diagnostic_asset(
    *,
    project_root: Path,
    source_asset: Path,
    historical_stage: Any,
    robot: str,
    approximation: str,
    destination: Path,
    raw_meshes: dict[str, np.ndarray],
) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    source_stage = Usd.Stage.Open(str(source_asset))
    if source_stage is None:
        raise RuntimeError(f"failed to open protected source asset: {source_asset}")
    source_inventory = _collider_inventory(source_stage)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{robot}").GetPrim()
    root.GetReferences().AddReference(
        os.path.relpath(source_asset, destination.parent),
        Sdf.Path(f"/{robot}"),
    )
    root.SetInstanceable(False)  # noqa: FBT003 - USD API is positional-only
    stage.SetDefaultPrim(root)

    old_geometry_paths = []
    correct_meshes = []
    for side in ("left", "right"):
        expected_points = raw_meshes[side]
        for branch in ("visuals", "collisions"):
            branch_path = f"/{robot}/{robot}_{side}_finger_link/{branch}"
            branch_prim = stage.GetPrimAtPath(branch_path)
            if not branch_prim.IsValid():
                raise RuntimeError(f"finger branch is not composed: {branch_path}")
            branch_prim.SetInstanceable(
                False  # noqa: FBT003 - USD API is positional-only
            )
            old_path = f"{branch_path}/gripper_finger"
            old_prim = stage.GetPrimAtPath(old_path)
            if not old_prim.IsValid() or old_prim.IsInstanceProxy():
                raise RuntimeError(
                    f"generic finger geometry is not overrideable: {old_path}"
                )
            old_prim.SetActive(False)  # noqa: FBT003 - USD API is positional-only
            old_geometry_paths.append(old_path)

            historical = _historical_mesh(
                historical_stage,
                side=side,
                branch=branch,
            )
            historical_points = np.asarray(
                [[float(v[0]), float(v[1]), float(v[2])] for v in historical["points"]],
                dtype=np.float32,
            )
            maximum_source_point_error = float(
                np.max(np.abs(historical_points - expected_points))
            )
            if maximum_source_point_error > 1.0e-6:
                raise RuntimeError(
                    "historical USD points do not match fixed-commit STL: "
                    f"{side}/{branch} max_error={maximum_source_point_error}"
                )

            new_path = f"{branch_path}/correct_custom_finger_{side}"
            container = _author_mesh(
                stage,
                path=new_path,
                source=historical,
                side=side,
            )
            if branch == "collisions":
                UsdPhysics.CollisionAPI.Apply(container.GetPrim())
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(container.GetPrim())
                mesh_api.CreateApproximationAttr().Set(approximation)
                if approximation == "convexHull":
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
                        container.GetPrim()
                    )
                elif approximation == "convexDecomposition":
                    PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                        container.GetPrim()
                    )
                else:
                    raise ValueError(f"unsupported approximation: {approximation}")
            else:
                approximation_value = approximation
            correct_meshes.append(
                {
                    "side": side,
                    "branch": branch,
                    "container_path": new_path,
                    "mesh_path": f"{new_path}/mesh",
                    "historical_proxy_path": historical["proxy_path"],
                    "point_count": len(historical["points"]),
                    "face_count": len(historical["face_counts"]),
                    "relative_transform": _matrix_rows(
                        historical["relative_transform"]
                    ),
                    "historical_reset_xform_stack": historical[
                        "reset_xform_stack"
                    ],
                    "maximum_stl_point_error_source_units": (
                        maximum_source_point_error
                    ),
                    "approximation": approximation_value,
                }
            )
    stage.GetRootLayer().Save()

    composed = Usd.Stage.Open(str(destination))
    if composed is None:
        raise RuntimeError(f"failed to reopen diagnostic asset: {destination}")
    generic_active = any(
        composed.GetPrimAtPath(path).IsValid()
        and composed.GetPrimAtPath(path).IsActive()
        for path in old_geometry_paths
    )
    readback_inventory = _collider_inventory(composed)
    new_colliders = [
        item
        for item in readback_inventory
        if "/correct_custom_finger_" in item["path"]
        and item["approximation"] is not None
    ]
    nonfinger_source = [
        item
        for item in source_inventory
        if "_left_finger_link/" not in item["path"]
        and "_right_finger_link/" not in item["path"]
    ]
    nonfinger_composed = [
        item
        for item in readback_inventory
        if "_left_finger_link/" not in item["path"]
        and "_right_finger_link/" not in item["path"]
    ]
    articulation_roots = _articulation_roots(composed)
    gates = {
        "generic_finger_geometry_inactive": not generic_active,
        "two_new_finger_colliders": len(new_colliders) == 2,
        "approximation_readback": (
            {item["approximation"] for item in new_colliders}
            == {approximation}
        ),
        "one_articulation_root": len(articulation_roots) == 1,
        "nonfinger_colliders_unchanged": nonfinger_source == nonfinger_composed,
        "all_historical_points_match_fixed_stl": all(
            item["maximum_stl_point_error_source_units"] <= 1.0e-6
            for item in correct_meshes
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "robot": robot,
        "approximation": approximation,
        "absolute_path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "protected_source_asset": str(source_asset),
        "protected_source_sha256": sha256_file(source_asset),
        "gates": gates,
        "articulation_roots": articulation_roots,
        "articulation_root_count": len(articulation_roots),
        "new_finger_colliders": len(new_colliders),
        "new_finger_collider_readback": new_colliders,
        "generic_finger_geometry_active": generic_active,
        "deactivated_generic_geometry_paths": old_geometry_paths,
        "correct_finger_meshes": correct_meshes,
        "nonfinger_collider_count": len(nonfinger_composed),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build protected correct-finger diagnostic USD wrappers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/aloha1_gripper_correct_finger_profiles.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/gripper_correct_finger_preflight.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve(strict=True)
    profile_path = (
        args.profile
        if args.profile.is_absolute()
        else project_root / args.profile
    )
    report_path = (
        args.report if args.report.is_absolute() else project_root / args.report
    )
    profile = load_correct_finger_profile(profile_path, project_root)
    source_before = verify_correct_finger_sources(profile, project_root)
    if source_before["status"] != "PASS":
        raise RuntimeError("correct-finger source preflight failed")

    # Import Isaac only after all filesystem/source gates pass.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd

        historical_path = (
            project_root / profile["source"]["historical_usd"]["path"]
        ).resolve(strict=True)
        historical_stage = Usd.Stage.Open(str(historical_path))
        if historical_stage is None:
            raise RuntimeError(f"failed to open historical USD: {historical_path}")
        raw_meshes = {
            side: _stl_triangle_vertices(
                (
                    project_root
                    / profile["source"]["meshes"][side]["path"]
                ).resolve(strict=True)
            )
            for side in ("left", "right")
        }
        assets = []
        for robot in ("follower_left", "follower_right"):
            source_asset = (
                project_root / profile["frozen"]["drive_source"][robot]
            ).resolve(strict=True)
            for profile_name, collider in profile["profiles"].items():
                destination = (
                    project_root
                    / profile["diagnostic_directories"][profile_name]
                    / robot
                    / f"{robot}_{profile_name}.usd"
                )
                assets.append(
                    _create_diagnostic_asset(
                        project_root=project_root,
                        source_asset=source_asset,
                        historical_stage=historical_stage,
                        robot=robot,
                        approximation=collider["approximation"],
                        destination=destination,
                        raw_meshes=raw_meshes,
                    )
                )
        source_after = verify_correct_finger_sources(profile, project_root)
        report = {
            "schema_version": 1,
            "status": (
                "PASS"
                if source_before["status"] == "PASS"
                and source_after["status"] == "PASS"
                and all(asset["status"] == "PASS" for asset in assets)
                else "FAIL"
            ),
            "restart_boundary": EXPECTED_RESTART_BOUNDARY,
            "runtime": profile["runtime"],
            "source_evidence_before": source_before,
            "source_evidence_after": source_after,
            "diagnostic_assets": assets,
            "default_asset_modified": False,
            "historical_generic_finger_reports_modified": False,
            "next_gate": "ISAAC_RUNTIME_OPEN_CLOSED_SCREENSHOT_PREFLIGHT",
        }
        _write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(report_path),
                    "assets": [item["absolute_path"] for item in assets],
                },
                indent=2,
            ),
            file=sys.stdout,
            flush=True,
        )
    finally:
        app.close()

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

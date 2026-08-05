#!/usr/bin/env python3
"""Read back an exported calibration bundle with OpenUSD and fail closed.

Run this with the project's Isaac/OpenUSD Python environment. The normal
backend virtualenv intentionally does not depend on ``pxr``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd
from pxr import UsdGeom


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_matrix(gf_matrix: Any) -> np.ndarray:
    """Convert a Gf row-vector matrix to the JSON column-vector convention."""

    return np.asarray(gf_matrix, dtype=np.float64).T


def _finite_bbox(stage: Usd.Stage, prim: Usd.Prim) -> tuple[list[float], list[float]]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = [float(value) for value in aligned.GetMin()]
    maximum = [float(value) for value in aligned.GetMax()]
    if not all(math.isfinite(value) for value in minimum + maximum):
        raise RuntimeError(f"non-finite world bound for {prim.GetPath()}")
    if not all(high > low for low, high in zip(minimum, maximum, strict=True)):
        raise RuntimeError(f"empty world bound for {prim.GetPath()}: {minimum} .. {maximum}")
    return minimum, maximum


def audit(review_stage: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source_stage"]["path"])
    bottle = Path(manifest["bottle_asset"]["path"])
    if _sha256(source) != manifest["source_stage"]["sha256_before"]:
        raise RuntimeError("source Stage hash changed after bundle export")
    if _sha256(bottle) != manifest["bottle_asset"]["sha256"]:
        raise RuntimeError("Bottle500 asset hash changed after bundle export")

    stage = Usd.Stage.Open(str(review_stage))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {review_stage}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or default_prim.GetPath() != "/World":
        raise RuntimeError("composed review Stage must have defaultPrim /World")

    camera_prim = stage.GetPrimAtPath("/World/CameraHigh")
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError("composed review Stage is missing /World/CameraHigh Camera")
    actual_camera = _column_matrix(
        UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    optical = np.asarray(manifest["world_from_camera_high_optical"]["matrix"], dtype=np.float64)
    rx_pi = np.diag([1.0, -1.0, -1.0, 1.0])
    expected_camera = optical @ rx_pi
    if not np.allclose(actual_camera, expected_camera, atol=1e-9):
        raise RuntimeError(
            "CameraHigh readback differs from world_from_camera_high_optical @ Rx(pi)"
        )

    bottle_trials = manifest.get("bottle_trials")
    bottle_paths = (
        [f"/World/Calibration/Bottle_{item['id'].replace('-', '_')}" for item in bottle_trials["trials"]]
        if bottle_trials
        else ["/World/Calibration/BottleReferenceProbe"]
    )
    bottle_results: list[dict[str, Any]] = []
    for path in bottle_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"missing composed Bottle500 prim: {path}")
        meshes = [child for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"Bottle500 reference at {path} contains no Mesh")
        minimum, maximum = _finite_bbox(stage, prim)
        bottle_results.append(
            {
                "path": path,
                "mesh_count": len(meshes),
                "world_aabb_min_m": minimum,
                "world_aabb_max_m": maximum,
            }
        )

    return {
        "status": "CALIBRATION_BUNDLE_OPENUSD_AUDIT_PASS",
        "review_stage": str(review_stage.resolve()),
        "default_prim": str(default_prim.GetPath()),
        "camera_prim": str(camera_prim.GetPath()),
        "camera_world_from_usd_readback_column_vector": actual_camera.tolist(),
        "bottles": bottle_results,
        "source_stage_sha256": _sha256(source),
        "bottle_asset_sha256": _sha256(bottle),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_stage", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = audit(args.review_stage.resolve(), args.manifest.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

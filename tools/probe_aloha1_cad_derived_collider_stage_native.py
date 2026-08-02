#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Fresh-process Isaac Sim 5.1 Stage-open probe for CAD colliders."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_diagnostic.usda"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_cad_derived_collider_stage_native_probe.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import carb
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    import omni.usd
    from pxr import UsdPhysics

    stage_hash_before = _sha256(STAGE_PATH)
    if not open_stage(str(STAGE_PATH.resolve(strict=True))):
        raise RuntimeError(f"failed to open {STAGE_PATH}")
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac runtime returned no current Stage")

    extension_manager = app.get_extension_manager()
    physx_id = extension_manager.get_enabled_extension_id("omni.physx")
    physx_record = (
        extension_manager.get_extension_dict(physx_id) if physx_id else None
    )
    physx_full = (
        physx_record.get("package", {}).get("version")
        if physx_record
        else None
    )
    runtime = {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}"))
        .split("+", maxsplit=1)[0],
        "physx": str(physx_full).split("+", maxsplit=1)[0],
    }
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    colliders = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/cad_derived_collisions/" not in path:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        colliders.append(
            {
                "prim_path": path,
                "collision_enabled": UsdPhysics.CollisionAPI(prim)
                .GetCollisionEnabledAttr()
                .Get(),
                "approximation": UsdPhysics.MeshCollisionAPI(prim)
                .GetApproximationAttr()
                .Get(),
            }
        )
    stage_hash_after = _sha256(STAGE_PATH)
    all_convex = all(
        item["collision_enabled"] is True
        and item["approximation"] == "convexHull"
        for item in colliders
    )
    passed = (
        runtime == expected_runtime
        and len(colliders) == 34
        and all_convex
        and stage_hash_before == stage_hash_after
        and str(stage.GetDefaultPrim().GetPath()) == "/World"
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "runtime": runtime,
        "stage": {
            "absolute_path": str(STAGE_PATH.resolve()),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
        },
        "new_collider_count": len(colliders),
        "all_new_colliders_convex_hull": all_convex,
        "new_colliders": colliders,
        "timeline_started": False,
        "source_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "runtime": runtime,
                "new_collider_count": len(colliders),
                "output": str(OUTPUT.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())

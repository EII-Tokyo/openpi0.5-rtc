#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build reference-only Bottle500 and static-environment validator targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "Bottle500": (
        REPO / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd",
        "/Bottle500",
    ),
    "static_environment": (
        REPO
        / "local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose"
        / "aloha2_menagerie_scene.usd",
        "/scene/worldBody",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(output_root: Path) -> dict[str, Any]:
    from pxr import Gf
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    if output_root.exists():
        raise FileExistsError(f"scoped candidate directory already exists: {output_root}")
    records = {}
    for name, (source_path, source_prim) in SOURCES.items():
        source = source_path.resolve(strict=True)
        before = _sha256(source)
        folder = output_root / name
        folder.mkdir(parents=True)
        path = folder / f"{name}.usda"
        stage = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
        relative = Path(os.path.relpath(source, path.parent)).as_posix()
        if not root.GetReferences().AddReference(relative, Sdf.Path(source_prim)):
            raise RuntimeError(f"failed to reference {source_prim} from {source}")
        stage.SetDefaultPrim(root)
        stage.GetRootLayer().customLayerData = {
            "aloha1:scope": "TASK7_RULE_SCOPE_DIAGNOSTIC_ONLY",
            "aloha1:sourceSha256": before,
            "aloha1:task8": "NOT_RUN",
        }
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
        if reopened is None or not reopened.GetDefaultPrim().IsValid():
            raise RuntimeError(f"failed to compose scoped asset: {name}")
        records[name] = {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
            "default_prim": str(reopened.GetDefaultPrim().GetPath()),
            "source_absolute_path": str(source),
            "source_prim": source_prim,
            "source_sha256_before": before,
            "source_sha256_after": _sha256(source),
            "source_unchanged": _sha256(source) == before,
        }

    bottle_base = Path(records["Bottle500"]["absolute_path"])
    name = "Bottle500_principal_axes_candidate"
    folder = output_root / name
    folder.mkdir(parents=True)
    path = folder / f"{name}.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    relative = Path(os.path.relpath(bottle_base, path.parent)).as_posix()
    root.GetReferences().AddReference(relative, Sdf.Path("/Bottle500"))
    stage.SetDefaultPrim(root)
    before_axes = root.GetAttribute("physics:principalAxes").Get()
    root.GetAttribute("physics:principalAxes").Set(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    after_axes = readback.GetDefaultPrim().GetAttribute("physics:principalAxes").Get()
    records[name] = {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "default_prim": f"/{name}",
        "source_candidate": str(bottle_base),
        "change": "author normalized identity physics:principalAxes",
        "before": str(before_axes),
        "after": str(after_axes),
        "reason": (
            "The source diagonal inertia is already expressed on its authored axes, "
            "while the source principal-axes quaternion has zero length and is invalid."
        ),
        "promotion": "USER_REVIEW_REQUIRED",
    }

    environment_base = Path(records["static_environment"]["absolute_path"])
    name = "static_environment_candidate"
    folder = output_root / name
    folder.mkdir(parents=True)
    path = folder / f"{name}.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    relative = Path(os.path.relpath(environment_base, path.parent)).as_posix()
    root.GetReferences().AddReference(relative, Sdf.Path("/static_environment"))
    stage.SetDefaultPrim(root)
    removed = []
    rigid_prims = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    for prim in rigid_prims:
        child_name = prim.GetName()
        if not prim.RemoveAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"failed to remove RigidBodyAPI from static {child_name}")
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) and not prim.RemoveAPI(
            PhysxSchema.PhysxRigidBodyAPI
        ):
            raise RuntimeError(f"failed to remove PhysxRigidBodyAPI from static {child_name}")
        removed.append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if any(readback.GetPrimAtPath(item).HasAPI(UsdPhysics.RigidBodyAPI) for item in removed):
        raise RuntimeError("static-environment RigidBodyAPI removal failed readback")
    records[name] = {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "default_prim": f"/{name}",
        "source_candidate": str(environment_base),
        "change": "remove dynamic rigid-body APIs from every static environment prim",
        "changed_prims": removed,
        "reason": (
            "These prims are static environment/table geometry. Removing the dynamic "
            "body APIs leaves descendant CollisionAPI shapes as static colliders and "
            "does not fabricate mass, inertia, or collision geometry."
        ),
        "promotion": "USER_REVIEW_REQUIRED",
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "REFERENCE_ONLY_DIAGNOSTIC_TARGETS",
        "assets": records,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "task8": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = _build(args.output_root.resolve())
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "manifest": str(manifest)}, sort_keys=True))
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    try:
        return main()
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(run())

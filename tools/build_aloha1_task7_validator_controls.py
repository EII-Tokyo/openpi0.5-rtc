#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build isolated reversible negative controls for Task 7 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO_ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_task7_rule_candidates/1.0"
    / "Trossen/vx300s_left/1.0/vx300s_left.usda"
)
SOURCE_ROOT = "/vx300s_left"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(target: Path, owner: Path) -> str:
    return Path(os.path.relpath(target.resolve(), owner.resolve().parent)).as_posix()


def _create_wrapper(*, destination: Path, name: str) -> tuple[Any, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    destination.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(destination))
    root_path = f"/{name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    if not root.GetReferences().AddReference(
        _relative(SOURCE, destination),
        Sdf.Path(SOURCE_ROOT),
    ):
        raise RuntimeError(f"unable to reference source candidate: {destination}")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().customLayerData = {
        "aloha1:scope": "TASK7_NEGATIVE_CONTROL_ONLY",
        "aloha1:sourceSha256": _sha256(SOURCE),
        "aloha1:task8": "NOT_RUN",
    }
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(destination), Usd.Stage.LoadAll)
    if reopened is None:
        raise RuntimeError(f"unable to reopen negative control: {destination}")
    return reopened, reopened.GetDefaultPrim()


def _descendant_collisions(body: Any) -> list[Any]:
    from pxr import Usd
    from pxr import UsdPhysics

    return [
        prim
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _build_controls(output_root: Path) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdPhysics
    from usd.schema.isaac import robot_schema

    if output_root.exists():
        raise FileExistsError(f"negative-control directory already exists: {output_root}")
    source_hash = _sha256(SOURCE)
    records: dict[str, Any] = {}

    name = "negative_robot_api"
    path = output_root / name / f"{name}.usda"
    stage, root = _create_wrapper(destination=path, name=name)
    if not root.RemoveAPI(robot_schema.Classes.ROBOT_API.value):
        raise RuntimeError("failed to remove RobotAPI in isolated negative control")
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if readback.GetDefaultPrim().HasAPI(robot_schema.Classes.ROBOT_API.value):
        raise RuntimeError("RobotAPI negative control did not survive readback")
    records[name] = {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "removed": "RobotAPI",
        "target_prim": f"/{name}",
        "expected_rule": "RobotSchema",
    }

    name = "negative_mass_api"
    path = output_root / name / f"{name}.usda"
    stage, root = _create_wrapper(destination=path, name=name)
    candidates = sorted(
        (
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and prim.HasAPI(UsdPhysics.MassAPI)
            and _descendant_collisions(prim)
        ),
        key=lambda prim: str(prim.GetPath()),
    )
    if not candidates:
        raise RuntimeError("no physical body with MassAPI found for negative control")
    body = candidates[0]
    target = str(body.GetPath())
    if not body.RemoveAPI(UsdPhysics.MassAPI):
        raise RuntimeError(f"failed to remove MassAPI: {target}")
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if readback.GetPrimAtPath(target).HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError("MassAPI negative control did not survive readback")
    records[name] = {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "removed": "MassAPI",
        "target_prim": target,
        "expected_rule": "RigidBodyHasMassAPI",
    }

    name = "negative_collider"
    path = output_root / name / f"{name}.usda"
    stage, root = _create_wrapper(destination=path, name=name)
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        collisions = _descendant_collisions(prim)
        if len(collisions) == 1:
            bodies.append((str(prim.GetPath()), prim, collisions[0]))
    if not bodies:
        raise RuntimeError("no rigid body with exactly one collider found")
    target, body, collider = sorted(bodies, key=lambda item: item[0])[0]
    collider_path = str(collider.GetPath())
    if not collider.RemoveAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"failed to remove CollisionAPI: {collider_path}")
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if readback.GetPrimAtPath(collider_path).HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("CollisionAPI negative control did not survive readback")
    records[name] = {
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "removed": "CollisionAPI",
        "target_prim": target,
        "collider_prim": collider_path,
        "expected_rule": "RigidBodyHasCollider",
    }

    if _sha256(SOURCE) != source_hash:
        raise RuntimeError("source candidate changed while building controls")
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "REVERSIBLE_NEGATIVE_CONTROLS_ONLY",
        "source": {
            "absolute_path": str(SOURCE.resolve()),
            "sha256_before": source_hash,
            "sha256_after": _sha256(SOURCE),
            "unchanged": True,
        },
        "controls": records,
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
    report = _build_controls(args.output_root.resolve())
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

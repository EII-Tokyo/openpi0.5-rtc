#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build the isolated Task 7 topology plus joint-state diagnostic candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0/"
    "virtual_helper_topology_collapse"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0/"
    "combined_topology_joint_state"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_task7_physicsrules_combined_candidate.json"
)
FROZEN_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
JOINT_NAMES = ("elbow", "left_finger", "right_finger", "shoulder", "wrist_angle")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(target: Path, owner: Path) -> str:
    return Path(os.path.relpath(target.resolve(), owner.resolve().parent)).as_posix()


def _layer_for_path(stage: Any, path: Path) -> Any:
    resolved = path.resolve()
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.realPath and Path(layer.realPath).resolve() == resolved:
            return layer
    raise RuntimeError(f"layer not found: {resolved}")


def _build_one(follower: str) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root_name = "vx300s_left" if follower == "follower_left" else "vx300s_right"
    root_path = f"/{root_name}"
    source = (
        SOURCE_ROOT
        / follower
        / f"{follower}_virtual_helper_topology_collapse.usda"
    ).resolve(strict=True)
    source_before = _sha256(source)
    destination = OUTPUT_ROOT / follower
    destination.mkdir(parents=True)
    wrapper = destination / f"{follower}_combined_topology_joint_state.usda"
    override = destination / f"{follower}_combined_topology_joint_state_override.usda"
    layer = Sdf.Layer.CreateNew(str(override))
    if layer is None:
        raise RuntimeError(f"cannot create {override}")
    layer.Save()
    wrapper_stage = Usd.Stage.CreateNew(str(wrapper))
    root = UsdGeom.Xform.Define(wrapper_stage, root_path).GetPrim()
    root.GetReferences().AddReference(_relative(source, wrapper), root_path)
    wrapper_stage.GetRootLayer().subLayerPaths = [_relative(override, wrapper)]
    wrapper_stage.SetDefaultPrim(root)
    wrapper_stage.GetRootLayer().customLayerData = {
        "aloha1:scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "aloha1:profile": "combined_topology_joint_state",
        "aloha1:inputProfile": "virtual_helper_topology_collapse",
        "aloha1:frozenStageSha256": FROZEN_SHA256,
    }
    wrapper_stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open {wrapper}")
    stage.SetEditTarget(_layer_for_path(stage, override))
    states = []
    for name in JOINT_NAMES:
        path = f"{root_path}/joints/{name}"
        prim = stage.GetPrimAtPath(path)
        axis = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        api = PhysxSchema.JointStateAPI(prim, axis)
        before = float(api.GetPositionAttr().Get())
        api.GetPositionAttr().Set(0.0)
        states.append(
            {
                "joint_path": path,
                "axis": axis,
                "position_before": before,
                "position_after": float(api.GetPositionAttr().Get()),
            }
        )
    stage.GetEditTarget().GetLayer().Save()
    stage.GetRootLayer().Save()
    readback = Usd.Stage.Open(str(wrapper), Usd.Stage.LoadAll)
    if readback is None:
        raise RuntimeError(f"cannot reopen {wrapper}")
    source_after = _sha256(source)
    return {
        "follower": follower,
        "source": {"absolute_path": str(source), "sha256": source_before},
        "source_modified": source_before != source_after,
        "joint_state_position_count": len(states),
        "joint_states": states,
        "wrapper": {
            "absolute_path": str(wrapper.resolve()),
            "sha256": _sha256(wrapper),
            "default_prim": str(readback.GetDefaultPrim().GetPath()),
            "sublayers": list(readback.GetRootLayer().subLayerPaths),
        },
        "override_layer": {
            "absolute_path": str(override.resolve()),
            "sha256": _sha256(override),
        },
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"candidate output exists: {OUTPUT_ROOT}")
    candidates = [_build_one(follower) for follower in ("follower_left", "follower_right")]
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "DIAGNOSTIC_ONLY_NOT_FINAL",
        "input_profile": "virtual_helper_topology_collapse",
        "frozen_stage_sha256": FROZEN_SHA256,
        "candidates": candidates,
        "physics_equivalence": "PARTIAL_HELPER_MASS_NOT_CONSERVED",
        "mimic_modified": False,
        "friction_timestep_solver_modified": False,
        "final_or_default_asset_modified": False,
        "promotion_status": "BLOCKED_USER_REVIEW_AND_MASS_SEMANTICS",
        "task8": "NOT_RUN",
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "candidates": len(candidates)}))
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "create_new_stage": False})
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

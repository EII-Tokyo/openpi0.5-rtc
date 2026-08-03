#!/usr/bin/env python3
"""Validate isolated Task 8 collider pruning composition and PhysX cooking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.task8_collider_lod import compare_profile_inventories
from tools.audit_aloha1_task8_baseline import audit
from tools.audit_aloha1_task8_baseline import start_usd_runtime_if_needed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _cooked_signature(pieces: list[dict[str, Any]]) -> str:
    payload = [
        {
            "vertices": piece["vertices"],
            "indices": piece["indices"],
            "polygons": piece["polygons"],
        }
        for piece in pieces
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cook_upper_arm_colliders(app: Any, stage_path: Path) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Usd
    from pxr import UsdPhysics

    from tools.compare_aloha1_gripper_colliders import _convex_piece_metrics
    from tools.compare_aloha1_gripper_colliders import _cooking_statistics
    from tools.compare_aloha1_gripper_colliders import _request_cooked_representation
    from tools.compare_aloha1_gripper_colliders import _subtract_stats

    stage = Usd.Stage.Open(str(stage_path.resolve(strict=True)), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open Task 8 profile: {stage_path}")
    paths = []
    for prim in Usd.PrimRange.Stage(stage):
        path = str(prim.GetPath())
        if (
            "_upper_arm_link/cad_derived_collisions/cad_derived_upper_arm_link/"
            in path
            and path.endswith("/mesh")
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            paths.append(path)
    paths.sort()

    cooking = get_physx_cooking_interface()
    cooking.release_local_mesh_cache()
    before = _cooking_statistics()
    records = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        approximation = str(
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or ""
        )
        result, convexes, runtime_s = _request_cooked_representation(
            app, stage=stage, collider_path=path
        )
        if result != PhysxCollisionRepresentationResult.RESULT_VALID:
            raise RuntimeError(f"cooking failed for {path}: {result}")
        pieces = [_convex_piece_metrics(piece) for piece in convexes]
        records.append(
            {
                "path": path,
                "approximation": approximation,
                "result": str(result),
                "runtime_s": runtime_s,
                "piece_count": len(pieces),
                "vertex_count": sum(int(piece["vertex_count"]) for piece in pieces),
                "face_count": sum(int(piece["face_count"]) for piece in pieces),
                "cooked_geometry_signature": _cooked_signature(pieces),
                "pieces": pieces,
            }
        )
    after = _cooking_statistics()
    return {
        "authored_collider_count": len(paths),
        "cooked_convex_piece_count": sum(record["piece_count"] for record in records),
        "cooked_vertex_count": sum(record["vertex_count"] for record in records),
        "cooked_face_count": sum(record["face_count"] for record in records),
        "colliders": records,
        "cooking_statistics_delta": _subtract_stats(after, before),
    }


def validate(app: Any, candidate_report_path: Path) -> dict[str, Any]:
    from tools.compare_aloha1_gripper_colliders import _local_api_probe

    candidate_report_path = candidate_report_path.resolve(strict=True)
    candidate = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    runtime = _local_api_probe()
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    runtime_match = all(runtime.get(key) == value for key, value in expected_runtime.items())
    if not runtime_match:
        raise RuntimeError(f"Isaac runtime drift: {runtime}")

    fidelity_path = Path(candidate["layers"]["fidelity_profile"]["absolute_path"])
    throughput_path = Path(candidate["layers"]["throughput_profile"]["absolute_path"])
    expected_hashes = {
        "fidelity_profile": candidate["layers"]["fidelity_profile"]["sha256"],
        "throughput_profile": candidate["layers"]["throughput_profile"]["sha256"],
    }
    actual_hashes = {
        "fidelity_profile": _sha256(fidelity_path),
        "throughput_profile": _sha256(throughput_path),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"candidate profile hash drift: {actual_hashes}")

    finger_limit = ROOT / (
        "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "finger_limit_pair_collision_candidate/1.0/configuration/"
        "finger_source_limits.usda"
    )
    fidelity_inventory = audit(fidelity_path, finger_limit)
    throughput_inventory = audit(throughput_path, finger_limit)
    removed_paths = sorted(
        record["prim_path"]
        for record in candidate["collider_records"]
        if record["candidate_active"] is False
    )
    inventory_comparison = compare_profile_inventories(
        fidelity_inventory["protected_inventory"],
        throughput_inventory["protected_inventory"],
        removed_collider_paths=removed_paths,
    )

    cooking = {
        "fidelity_profile": _cook_upper_arm_colliders(app, fidelity_path),
        "throughput_profile": _cook_upper_arm_colliders(app, throughput_path),
    }
    fidelity_by_path = {
        record["path"]: record for record in cooking["fidelity_profile"]["colliders"]
    }
    throughput_by_path = {
        record["path"]: record
        for record in cooking["throughput_profile"]["colliders"]
    }
    retained_paths = sorted(throughput_by_path)
    retained_cooked_geometry_equal = {
        path: (
            path in fidelity_by_path
            and fidelity_by_path[path]["cooked_geometry_signature"]
            == throughput_by_path[path]["cooked_geometry_signature"]
        )
        for path in retained_paths
    }
    cooking_gate = (
        cooking["fidelity_profile"]["authored_collider_count"] == 8
        and cooking["throughput_profile"]["authored_collider_count"] == 2
        and all(retained_cooked_geometry_equal.values())
    )
    status = (
        "PASS"
        if runtime_match
        and inventory_comparison["status"] == "PASS"
        and cooking_gate
        and not fidelity_inventory["dependencies"]["unresolved"]
        and not throughput_inventory["dependencies"]["unresolved"]
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": "TASK8_COLLIDER_LOD_FRESH_PROCESS_VALIDATION",
        "runtime": runtime,
        "candidate_report": {
            "absolute_path": str(candidate_report_path),
            "sha256": _sha256(candidate_report_path),
        },
        "profiles": {
            "fidelity_profile": {
                "absolute_path": str(fidelity_path.resolve()),
                "sha256": actual_hashes["fidelity_profile"],
                "default_prim": fidelity_inventory["stage"]["default_prim"],
                "dependency_unresolved": fidelity_inventory["dependencies"]["unresolved"],
            },
            "throughput_profile": {
                "absolute_path": str(throughput_path.resolve()),
                "sha256": actual_hashes["throughput_profile"],
                "default_prim": throughput_inventory["stage"]["default_prim"],
                "dependency_unresolved": throughput_inventory["dependencies"]["unresolved"],
            },
        },
        "inventory_comparison": inventory_comparison,
        "cooking": cooking,
        "retained_cooked_geometry_equal": retained_cooked_geometry_equal,
        "cooking_gate": "PASS" if cooking_gate else "FAIL",
        "boundaries": {
            "candidate_promoted": False,
            "final_or_default_asset_modified": False,
            "physics_parameters_modified": False,
            "only_collider_complexity_changed": True,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = start_usd_runtime_if_needed()
    result = 1
    try:
        report = validate(app, args.candidate_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "inventory": report["inventory_comparison"]["status"],
                    "cooking": report["cooking_gate"],
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = 0 if report["status"] == "PASS" else 1
    except Exception:
        print("TASK8_COLLIDER_LOD_VALIDATION_EXCEPTION", flush=True)
        traceback.print_exc()
    finally:
        if app is not None:
            app.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())

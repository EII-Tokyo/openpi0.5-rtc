#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run dense five-pose swept collision checks on the isolated CAD Stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import traceback

import numpy as np

from tools.aloha1_mapping.cad_derived_collision_runtime import load_frozen_pose_manifest
from tools.aloha1_mapping.cad_derived_swept_collision import deterministic_sweep_signature
from tools.aloha1_mapping.cad_derived_swept_collision import summarize_swept_samples
from tools.validate_aloha1_cad_derived_collision_static import ARTICULATION_PATHS
from tools.validate_aloha1_cad_derived_collision_static import GRIPPER_DECOMP_STAGE
from tools.validate_aloha1_cad_derived_collision_static import STAGE_REPORT
from tools.validate_aloha1_cad_derived_collision_static import _active_mesh_colliders
from tools.validate_aloha1_cad_derived_collision_static import _adjacent_body_pairs
from tools.validate_aloha1_cad_derived_collision_static import _apply_full_state
from tools.validate_aloha1_cad_derived_collision_static import _documented_assembly_interface_pairs
from tools.validate_aloha1_cad_derived_collision_static import _numerical_overlaps
from tools.validate_aloha1_cad_derived_collision_static import _sha256

ROOT = Path(__file__).resolve().parents[1]
POSE_SOURCE = ROOT / "reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_cad_collision_replan_preflight.json"
OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision.json"
CURVES = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision_curves.csv"
TASK8_CANDIDATE_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=GRIPPER_DECOMP_STAGE)
    parser.add_argument("--pose-source", type=Path, default=POSE_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=("gripper_decomposition", "task8_fidelity", "task8_throughput"),
        default="gripper_decomposition",
    )
    return parser.parse_args()


def _sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> int:
    import carb
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    from omni.physx import get_physx_interface
    import omni.usd

    stage_report = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    variant = stage_report["gripper_decomposition_diagnostic_variant"]
    stage_path = args.stage.resolve(strict=True)
    if args.profile.startswith("task8_"):
        task8_candidate = json.loads(TASK8_CANDIDATE_REPORT.read_text(encoding="utf-8"))
        layer_key = (
            "throughput_profile"
            if args.profile == "task8_throughput"
            else "fidelity_profile"
        )
        expected_stage = Path(
            task8_candidate["layers"][layer_key]["absolute_path"]
        )
        expected_stage_hash = task8_candidate["layers"][layer_key]["sha256"]
    else:
        expected_stage = GRIPPER_DECOMP_STAGE
        expected_stage_hash = variant["root_layer"]["sha256"]
    if stage_path != expected_stage.resolve(strict=True):
        raise ValueError("Stage is not the frozen profile selected for this sweep")
    stage_hash_before = _sha256(stage_path)
    if stage_hash_before != expected_stage_hash:
        raise ValueError("diagnostic Stage root hash drift")
    geometry_path = Path(stage_report["geometry_layer"]["absolute_path"])
    physics_path = Path(variant["physics_layer"]["absolute_path"])
    sublayer_hashes_before = {
        str(geometry_path): _sha256(geometry_path),
        str(physics_path): _sha256(physics_path),
    }
    if sublayer_hashes_before[str(geometry_path)] != variant["geometry_layer_sha256"]:
        raise ValueError("diagnostic geometry hash drift")
    if sublayer_hashes_before[str(physics_path)] != variant["physics_layer"]["sha256"]:
        raise ValueError("diagnostic physics hash drift")

    pose_payload = json.loads(args.pose_source.resolve(strict=True).read_text(encoding="utf-8"))
    selected = pose_payload["selected_samples"]
    if [record["sample_id"] for record in selected] != [
        "sample_01",
        "sample_02",
        "sample_03",
        "sample_04",
        "sample_05",
    ]:
        raise ValueError("selected sample order drift")
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    manager = app.get_extension_manager()
    physx_id = manager.get_enabled_extension_id("omni.physx")
    physx_record = manager.get_extension_dict(physx_id) if physx_id else {}
    runtime = {
        "isaac_sim": version("isaacsim"),
        "kit": str(carb.tokens.get_tokens_interface().resolve("${kit_version}")).split("+", maxsplit=1)[0],
        "physx": str(physx_record.get("package", {}).get("version", "")).split("+", maxsplit=1)[0],
    }
    if runtime != {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }:
        raise RuntimeError(f"runtime drift: {runtime}")

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulations = {}
    for robot, path in ARTICULATION_PATHS.items():
        articulation = SingleArticulation(
            prim_path=path,
            name=f"cad_sweep_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()
    physx = get_physx_interface()
    adjacent = _adjacent_body_pairs(stage)
    assembly = _documented_assembly_interface_pairs()
    home_q = load_frozen_pose_manifest(
        ROOT / "reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results_downward_contact_gate_v5.json"
    )[0]["arm_q_rad"]
    rows = []
    sample_records = []
    for sample in selected:
        world.reset()
        _apply_full_state(
            articulations["follower_right"],
            home_q,
            (0.057, -0.057),
        )
        initial_q = [float(value) for value in sample["initial_arm_q_rad"]]
        trajectory = [
            {
                "phase": "initial_pose",
                "segment": 0,
                "joint_positions_rad": initial_q,
            },
            *sample["ik"]["waypoints"],
        ]
        waypoint_records = []
        unexpected_pairs = set()
        finite = True
        for waypoint_index, waypoint in enumerate(trajectory):
            q = [float(value) for value in waypoint["joint_positions_rad"]]
            finite = finite and bool(np.isfinite(q).all())
            _apply_full_state(
                articulations["follower_left"],
                q,
                (0.057, -0.057),
            )
            physx.update_transformations(True, True, False, False)  # noqa: FBT003
            colliders = _active_mesh_colliders(stage)
            overlaps = _numerical_overlaps(colliders, adjacent, assembly)
            unexpected = [item for item in overlaps if item.get("allowed") is False]
            for item in unexpected:
                unexpected_pairs.add(tuple(item["actor_pair"]))
            record = {
                "waypoint_index": waypoint_index,
                "phase": str(waypoint.get("phase", "unknown")),
                "segment": int(waypoint.get("segment", waypoint_index)),
                "joint_positions_rad": q,
                "unexpected_overlap_count": len(unexpected),
                "unexpected_overlaps": unexpected,
            }
            waypoint_records.append(record)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "waypoint_index": waypoint_index,
                    "phase": record["phase"],
                    "segment": record["segment"],
                    "unexpected_overlap_count": len(unexpected),
                }
            )
        blocked_count = sum(bool(record["unexpected_overlap_count"]) for record in waypoint_records)
        sample_record = {
            "sample_id": sample["sample_id"],
            "candidate_index": sample["candidate_index"],
            "finite": finite,
            "waypoint_count": len(waypoint_records),
            "unexpected_overlap_waypoint_count": blocked_count,
            "unexpected_pairs": [list(pair) for pair in sorted(unexpected_pairs)],
            "status": "PASS" if finite and blocked_count == 0 else "FAIL",
            "waypoints": waypoint_records,
        }
        sample_records.append(sample_record)

    summary = summarize_swept_samples(sample_records)
    signature = deterministic_sweep_signature(sample_records)
    stage_hash_after = _sha256(stage_path)
    sublayer_hashes_after = {
        str(geometry_path): _sha256(geometry_path),
        str(physics_path): _sha256(physics_path),
    }
    immutable = stage_hash_after == stage_hash_before and sublayer_hashes_after == sublayer_hashes_before
    if not immutable:
        raise RuntimeError("diagnostic Stage/layer hash drift during sweep")
    report = {
        "schema_version": 1,
        "status": summary["status"] if immutable else "FAIL",
        "scope": "DENSE_DISCRETE_WAYPOINT_SWEEP_NOT_CONTINUOUS_CCD",
        "runtime": runtime,
        "repeat_index": args.repeat_index,
        "profile": args.profile,
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "sublayer_hashes_before": sublayer_hashes_before,
            "sublayer_hashes_after": sublayer_hashes_after,
        },
        "pose_source": {
            "absolute_path": str(args.pose_source.resolve(strict=True)),
            "sha256": _sha256_local(args.pose_source),
        },
        "samples": sample_records,
        "summary": summary,
        "deterministic_signature": signature,
        "self_collision_policy_modified": False,
        "source_or_final_asset_modified": False,
        "task8": (
            "AUTHORIZED_IN_PROGRESS"
            if args.profile.startswith("task8_")
            else "NOT_RUN"
        ),
    }
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.curves.resolve().open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_count": len(sample_records),
                "waypoint_count": summary["total_waypoint_count"],
                "signature": signature,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    world.stop()
    return 0 if report["status"] == "PASS" else 2


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
        exit_code = main(_parse_args())
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())

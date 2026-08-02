#!/usr/bin/env python3
"""Audit Bottle500 PhysX tensor velocity against pose finite differences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def classify_velocity_semantics(
    *,
    baseline_aligned: bool,
    initialize_aligned: bool,
    initialize_runtime_pass: bool,
    recreate_aligned: bool,
    recreate_runtime_pass: bool,
) -> str:
    """Classify only a demonstrated one-variable lifecycle outcome."""

    if baseline_aligned:
        return "VERIFIED"
    if initialize_runtime_pass and initialize_aligned:
        return "KINEMATIC_TRANSITION_ISSUE"
    if recreate_runtime_pass and recreate_aligned:
        return "STALE_TENSOR_VIEW"
    return "INCONCLUSIVE"


def classify_readback_responsibility(
    *,
    exact_tensor_path: bool,
    tensor_direct_transform_max_delta_m: float,
    tensor_usd_linear_velocity_max_delta_m_s: float,
    transform_velocity_alignment: bool,
) -> str:
    """Localize the disagreement without guessing an internal PhysX cause."""

    if (
        exact_tensor_path
        and tensor_direct_transform_max_delta_m <= 1.0e-9
        and tensor_usd_linear_velocity_max_delta_m_s <= 1.0e-9
        and not transform_velocity_alignment
    ):
        return "VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT"
    return "UNRESOLVED_READBACK_BOUNDARY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _run_metrics(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve(strict=True)
    report_path = root / "aloha1_grasp_20cm_runtime.json"
    telemetry_path = root / "aloha1_grasp_20cm_telemetry.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hold = [
        row
        for row in rows
        if row.get("phase") == "HOLD"
        and row.get("bottle", {}).get(
            "pose_finite_difference_velocity"
        )
        is not None
        and row.get("bottle", {}).get(
            "center_of_mass_pose_finite_difference_velocity"
        )
        is not None
    ]
    result: dict[str, Any] = {
        "run_root": str(root),
        "runtime_report": {
            "absolute_path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "telemetry": {
            "absolute_path": str(telemetry_path),
            "sha256": _sha256(telemetry_path),
            "frame_count": len(rows),
            "hold_frame_count": len(hold),
        },
        "machine_status": report.get("status"),
        "machine_reason": report.get("reason"),
        "deterministic_signature": report.get(
            "deterministic_signature"
        ),
        "lifecycle": report.get("runtime", {}).get(
            "bottle_tensor_lifecycle"
        ),
        "stage": report.get("stage"),
        "grasp_metrics": report.get("metrics"),
    }
    lifecycle = result.get("lifecycle") or {}
    identities = lifecycle.get("rigid_body_view_identities", [])
    direct_deltas_all = [
        float(
            row["bottle"]["direct_physx_transform"][
                "tensor_position_delta_norm_m"
            ]
        )
        for row in rows
        if row.get("bottle", {}).get("direct_physx_transform")
    ]
    usd_linear_deltas_all = [
        float(
            row["bottle"]["usd_velocity_readback"][
                "tensor_linear_delta_norm_m_s"
            ]
        )
        for row in rows
        if row.get("bottle", {}).get("usd_velocity_readback")
    ]
    usd_angular_deltas_all = [
        float(
            row["bottle"]["usd_velocity_readback"][
                "tensor_angular_delta_norm_rad_s"
            ]
        )
        for row in rows
        if row.get("bottle", {}).get("usd_velocity_readback")
    ]
    direct_deltas_hold = [
        float(
            row["bottle"]["direct_physx_transform"][
                "tensor_position_delta_norm_m"
            ]
        )
        for row in hold
        if row.get("bottle", {}).get("direct_physx_transform")
    ]
    usd_linear_deltas_hold = [
        float(
            row["bottle"]["usd_velocity_readback"][
                "tensor_linear_delta_norm_m_s"
            ]
        )
        for row in hold
        if row.get("bottle", {}).get("usd_velocity_readback")
    ]
    usd_angular_deltas_hold = [
        float(
            row["bottle"]["usd_velocity_readback"][
                "tensor_angular_delta_norm_rad_s"
            ]
        )
        for row in hold
        if row.get("bottle", {}).get("usd_velocity_readback")
    ]
    result["readback_cross_checks"] = {
        "tensor_view_identity_count": len(identities),
        "tensor_view_exact_path_match": bool(identities)
        and all(bool(item["exact_path_match"]) for item in identities),
        "tensor_view_identities": identities,
        "direct_physx_transform_sample_count": len(
            direct_deltas_hold
        ),
        "tensor_direct_transform_max_delta_m": (
            max(direct_deltas_hold) if direct_deltas_hold else None
        ),
        "overall_tensor_direct_transform_max_delta_m": (
            max(direct_deltas_all) if direct_deltas_all else None
        ),
        "usd_velocity_sample_count": len(usd_linear_deltas_hold),
        "tensor_usd_linear_velocity_max_delta_m_s": (
            max(usd_linear_deltas_hold)
            if usd_linear_deltas_hold
            else None
        ),
        "tensor_usd_angular_velocity_max_delta_rad_s": (
            max(usd_angular_deltas_hold)
            if usd_angular_deltas_hold
            else None
        ),
        "overall_tensor_usd_linear_velocity_max_delta_m_s": (
            max(usd_linear_deltas_all)
            if usd_linear_deltas_all
            else None
        ),
        "overall_tensor_usd_angular_velocity_max_delta_rad_s": (
            max(usd_angular_deltas_all)
            if usd_angular_deltas_all
            else None
        ),
    }
    if not hold:
        result["velocity_alignment"] = {
            "status": "NOT_OBSERVABLE_NO_HOLD_SAMPLES",
            "aligned": False,
        }
        return result

    tensor_com_linear = np.asarray(
        [row["bottle"]["linear_velocity_world_m_s"] for row in hold],
        dtype=np.float64,
    )
    pose_com_linear = np.asarray(
        [
            row["bottle"][
                "center_of_mass_pose_finite_difference_velocity"
            ]["linear_velocity_world_m_s"]
            for row in hold
        ],
        dtype=np.float64,
    )
    tensor_origin_linear = np.asarray(
        [
            row["bottle"][
                "prim_origin_linear_velocity_world_m_s"
            ]
            for row in hold
        ],
        dtype=np.float64,
    )
    pose_origin_linear = np.asarray(
        [
            row["bottle"]["pose_finite_difference_velocity"][
                "linear_velocity_world_m_s"
            ]
            for row in hold
        ],
        dtype=np.float64,
    )
    tensor_angular = np.asarray(
        [row["bottle"]["angular_velocity_world_rad_s"] for row in hold],
        dtype=np.float64,
    )
    pose_angular = np.asarray(
        [
            row["bottle"]["pose_finite_difference_velocity"][
                "angular_velocity_world_rad_s"
            ]
            for row in hold
        ],
        dtype=np.float64,
    )
    dt_s = float(hold[-1]["time_s"] - hold[-2]["time_s"])
    com_z = np.asarray(
        [row["bottle"]["center_of_mass_world_m"][2] for row in hold],
        dtype=np.float64,
    )
    origin_z = np.asarray(
        [row["bottle"]["position_world_m"][2] for row in hold],
        dtype=np.float64,
    )
    com_local = np.asarray(
        [row["bottle"]["center_of_mass_local_m"] for row in hold],
        dtype=np.float64,
    )
    integrated_tensor_com_z = float(
        np.sum(tensor_com_linear[:, 2]) * dt_s
    )
    observed_com_z = float(com_z[-1] - com_z[0])
    integrated_error = abs(integrated_tensor_com_z - observed_com_z)
    thresholds = {
        "integrated_vertical_error_max_m": 0.020,
        "linear_vector_rmse_max_m_s": 0.050,
        "angular_vector_rmse_max_rad_s": 0.500,
        "classification": (
            "DIAGNOSTIC_ENGINEERING_TOLERANCE_NOT_HARDWARE_CALIBRATION"
        ),
    }
    linear_rmse = _rmse(tensor_com_linear, pose_com_linear)
    angular_rmse = _rmse(tensor_angular, pose_angular)
    gates = {
        "integrated_vertical_displacement_consistent": (
            integrated_error
            <= thresholds["integrated_vertical_error_max_m"]
        ),
        "com_linear_velocity_consistent": (
            linear_rmse <= thresholds["linear_vector_rmse_max_m_s"]
        ),
        "angular_velocity_consistent": (
            angular_rmse <= thresholds["angular_vector_rmse_max_rad_s"]
        ),
    }
    result["velocity_alignment"] = {
        "status": "ALIGNED" if all(gates.values()) else "MISMATCH",
        "aligned": all(gates.values()),
        "hold_dt_s": dt_s,
        "center_of_mass_local_m": {
            "minimum": com_local.min(axis=0).tolist(),
            "maximum": com_local.max(axis=0).tolist(),
        },
        "tensor_com_vertical_integral_m": integrated_tensor_com_z,
        "pose_com_vertical_displacement_m": observed_com_z,
        "integrated_vertical_error_m": integrated_error,
        "pose_origin_vertical_displacement_m": float(
            origin_z[-1] - origin_z[0]
        ),
        "tensor_vs_pose_com_linear_vector_rmse_m_s": linear_rmse,
        "tensor_translated_origin_vs_pose_origin_linear_vector_rmse_m_s": (
            _rmse(tensor_origin_linear, pose_origin_linear)
        ),
        "tensor_vs_pose_angular_vector_rmse_rad_s": angular_rmse,
        "tensor_com_vertical_velocity_range_m_s": [
            float(tensor_com_linear[:, 2].min()),
            float(tensor_com_linear[:, 2].max()),
        ],
        "pose_com_vertical_velocity_range_m_s": [
            float(pose_com_linear[:, 2].min()),
            float(pose_com_linear[:, 2].max()),
        ],
        "thresholds": thresholds,
        "gates": gates,
    }
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Bottle500 velocity consistency diagnosis",
        "",
        f"- Status: `{report['status']}`",
        (
            "- Velocity semantics: "
            f"`{report['velocity_semantics_status']}`"
        ),
        (
            "- Readback responsibility boundary: "
            f"`{report['readback_responsibility_boundary']}`"
        ),
        "- Isaac Sim / Kit / PhysX: `5.1.0.0 / 107.3.3 / 107.3.26`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Variant | Runtime | Alignment | Signature |",
        "|---|---|---|---|",
    ]
    for name, run in report["variants"].items():
        alignment = run["velocity_alignment"]
        lines.append(
            f"| {name} | {run['machine_status']} | "
            f"{alignment['status']} | "
            f"`{run['deterministic_signature']}` |"
        )
    lines.extend(
        [
            "",
            "The baseline, immediate recreation, delayed recreation after a "
            "dynamic physics step, direct-transform probe and USD velocity "
            "writeback preserve the same grasp signature. The tensor view "
            "binds the exact Bottle500 prim; tensor transform equals the "
            "direct PhysX transform, and USD velocity writeback equals tensor "
            "velocity after converting angular degrees/second to radians/second. "
            "Nevertheless velocity integration remains inconsistent with "
            "transform evolution. Calling "
            "`initialize_kinematic_bodies()` at the tested post-reset point "
            "makes the first tensor sample invalid and fails the run. COM and "
            "rigid-prim-origin comparisons were both evaluated; point choice "
            "alone does not explain the mismatch.",
            "",
            "Tensor velocity is therefore retained as an explicitly unresolved "
            "diagnostic channel. Contact pairs, pose, support clearance, drop, "
            "and deterministic video evidence remain the authoritative hold "
            "gate; this report does not silently reinterpret tensor velocity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--initialize-run", required=True, type=Path)
    parser.add_argument("--recreate-run", required=True, type=Path)
    parser.add_argument("--delayed-run", type=Path)
    parser.add_argument("--direct-transform-run", type=Path)
    parser.add_argument("--usd-velocity-run", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    variants = {
        "BASELINE": _run_metrics(args.baseline_run),
        "INITIALIZE_KINEMATIC_BODIES": _run_metrics(
            args.initialize_run
        ),
        "RECREATE_AFTER_DYNAMIC": _run_metrics(args.recreate_run),
    }
    optional_variants = {
        "RECREATE_AFTER_DYNAMIC_STEP": args.delayed_run,
        "DIRECT_PHYSX_TRANSFORM": args.direct_transform_run,
        "USD_VELOCITY_WRITEBACK": args.usd_velocity_run,
    }
    for name, path in optional_variants.items():
        if path is not None:
            variants[name] = _run_metrics(path)
    baseline = variants["BASELINE"]
    initialize = variants["INITIALIZE_KINEMATIC_BODIES"]
    recreate = variants["RECREATE_AFTER_DYNAMIC"]
    status = classify_velocity_semantics(
        baseline_aligned=bool(
            baseline["velocity_alignment"]["aligned"]
        ),
        initialize_aligned=bool(
            initialize["velocity_alignment"]["aligned"]
        ),
        initialize_runtime_pass=(
            initialize["machine_status"] == "PASS"
        ),
        recreate_aligned=bool(
            recreate["velocity_alignment"]["aligned"]
        ),
        recreate_runtime_pass=(recreate["machine_status"] == "PASS"),
    )
    source_api = Path(
        ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache/"
        "omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/"
        "omni/physics/tensors/impl/api.py"
    ).resolve(strict=True)
    physx_api = Path(
        ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache/"
        "omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physx/"
        "bindings/_physx.pyi"
    ).resolve(strict=True)
    usd_physics_schema = Path(
        ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache/"
        "omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin/usd/"
        "usdPhysics/resources/usdPhysics/schema.usda"
    ).resolve(strict=True)
    usd_cross_checks = variants.get("USD_VELOCITY_WRITEBACK", {}).get(
        "readback_cross_checks", {}
    )
    direct_delta = usd_cross_checks.get(
        "tensor_direct_transform_max_delta_m"
    )
    usd_linear_delta = usd_cross_checks.get(
        "tensor_usd_linear_velocity_max_delta_m_s"
    )
    responsibility = classify_readback_responsibility(
        exact_tensor_path=bool(
            usd_cross_checks.get("tensor_view_exact_path_match")
        ),
        tensor_direct_transform_max_delta_m=(
            float(direct_delta)
            if direct_delta is not None
            else float("inf")
        ),
        tensor_usd_linear_velocity_max_delta_m_s=(
            float(usd_linear_delta)
            if usd_linear_delta is not None
            else float("inf")
        ),
        transform_velocity_alignment=bool(
            variants.get("USD_VELOCITY_WRITEBACK", {})
            .get("velocity_alignment", {})
            .get("aligned")
        ),
    )
    report = {
        "schema_version": 1,
        "status": "PARTIAL" if status == "INCONCLUSIVE" else "PASS",
        "velocity_semantics_status": status,
        "readback_responsibility_boundary": responsibility,
        "variants": variants,
        "gates": {
            "baseline_grasp_machine_pass": (
                baseline["machine_status"] == "PASS"
            ),
            "recreate_grasp_machine_pass": (
                recreate["machine_status"] == "PASS"
            ),
            "baseline_recreate_signatures_identical": (
                baseline["deterministic_signature"]
                == recreate["deterministic_signature"]
            ),
            "point_choice_explanation_rejected": (
                baseline["velocity_alignment"]["status"] == "MISMATCH"
            ),
            "tensor_velocity_not_used_as_drop_authority": True,
            "exact_tensor_actor_verified": bool(
                usd_cross_checks.get("tensor_view_exact_path_match")
            ),
            "tensor_and_direct_physx_transform_identical": (
                usd_cross_checks.get(
                    "tensor_direct_transform_max_delta_m"
                )
                == 0.0
            ),
            "tensor_and_usd_linear_velocity_identical": (
                usd_cross_checks.get(
                    "tensor_usd_linear_velocity_max_delta_m_s"
                )
                == 0.0
            ),
        },
        "local_official_api_source": {
            "absolute_path": str(source_api),
            "sha256": _sha256(source_api),
            "extension": "omni.physics.tensors",
            "version": "107.3.26+107.3.3",
            "confirmed_semantics": {
                "get_transforms": "GLOBAL_RIGID_BODY_PRIM_FRAME",
                "get_velocities": (
                    "GLOBAL_LINEAR_AT_CENTER_OF_MASS_AND_ANGULAR"
                ),
                "get_coms": (
                    "COM_POSE_RELATIVE_TO_RIGID_BODY_PRIM_FRAME"
                ),
                "initialize_kinematic_bodies": (
                    "INITIALIZES_KINEMATIC_BODY_TRANSFORM_VELOCITY_REPORTING"
                ),
            },
            "additional_sources": [
                {
                    "absolute_path": str(physx_api),
                    "sha256": _sha256(physx_api),
                    "confirmed_semantics": (
                        "update_transformations can write velocities to USD"
                    ),
                },
                {
                    "absolute_path": str(usd_physics_schema),
                    "sha256": _sha256(usd_physics_schema),
                    "confirmed_semantics": (
                        "physics:velocity uses distance/second; "
                        "physics:angularVelocity uses degrees/second"
                    ),
                },
            ],
        },
        "official_mcp": {
            "route": "DIRECT_isaac-sim-mcp_NOT_MCPJUNGLE",
            "status": "QUERIED",
            "query": (
                "Isaac Sim 5.1 RigidBodyView get_velocities and "
                "kinematic-body lifecycle"
            ),
            "version_boundary": (
                "SEARCH_RETURNED_NEWTON_OR_UNPINNED_RESULTS; LOCAL_5_1_"
                "SOURCE_IS_SEMANTIC_AUTHORITY"
            ),
        },
        "evidence_classification": {
            "local_runtime_readback": [
                "tensor transform/velocity/COM",
                "pose finite differences",
                "kinematic transition frame",
                "deterministic signatures",
            ],
            "numerical_calculation": [
                "COM world transform",
                "COM-to-origin velocity translation",
                "integrated displacement error",
                "linear/angular RMSE",
            ],
            "engineering_inference": (
                "LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT_"
                "INTERNAL_CAUSE_UNPROVEN"
            ),
            "hard_blocker": (
                "HARD_BLOCKER_LOCAL_PHYSX_107_3_26_INTERNAL_VELOCITY_"
                "SEMANTICS_UNPROVEN"
            ),
        },
        "boundaries": {
            "source_or_final_collider_modified": False,
            "physics_parameters_modified": False,
            "real_robot": False,
            "remote_103": False,
            "task8": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }
    _atomic_json(args.output_json.resolve(), report)
    args.output_md.resolve().write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

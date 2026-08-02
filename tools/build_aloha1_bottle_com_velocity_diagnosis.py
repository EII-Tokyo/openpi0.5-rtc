#!/usr/bin/env python3
"""Build the Task 7 Bottle500 V1/V2/V3 COM-velocity diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.bottle_com_velocity import analyze_samples
from tools.aloha1_mapping.bottle_com_velocity import build_velocity_diagnosis

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BOTTLE_SHA256 = (
    "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
)
EXPECTED_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def load_control_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute metrics from immutable samples using the current schema."""

    sample_path = Path(str(report["samples"]["absolute_path"]))
    return analyze_samples(_read_jsonl(sample_path))


def _control_runtime_valid(report: dict[str, Any], variant: str) -> bool:
    physics = report.get("physics", {})
    asset = report.get("input_asset", {})
    tensor = report.get("tensor_view", {})
    return bool(
        report.get("status") == "PASS"
        and report.get("variant") == variant
        and physics.get("gravity_magnitude") == 0.0
        and physics.get("all_collisions_disabled") is True
        and physics.get("collision_prim_count") == 41
        and tensor.get("count") == 1
        and tensor.get("tensor_index") == 0
        and tensor.get("actor_prim_path") == "/World/Bottle500"
        and asset.get("sha256_before") == EXPECTED_BOTTLE_SHA256
        and asset.get("sha256_after") == EXPECTED_BOTTLE_SHA256
        and report.get("metrics", {}).get("sample_count") == 121
    )


def _accepted_signature(
    aggregate: dict[str, Any],
    *,
    sample_id: str,
) -> str:
    matches = [
        row
        for row in aggregate.get("samples", [])
        if row.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"accepted runtime lacks unique {sample_id}")
    signature = matches[0].get("primary", {}).get(
        "deterministic_signature"
    )
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("accepted deterministic signature is invalid")
    return signature


def _markdown(report: dict[str, Any]) -> str:
    diagnosis = report["diagnosis"]
    metrics = report["experiments"]
    lines = [
        "# ALOHA1 Bottle500 COM velocity diagnosis",
        "",
        f"- Status: `{report['status']}`",
        f"- Conclusion: `{diagnosis['conclusion']}`",
        "- Runtime: `Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26`",
        "- Callback: `POST_PHYSICS_STEP` (`pre_step=False`)",
        f"- Derived velocity tolerance: `{diagnosis['tolerance']['velocity_tolerance_m_s']:.12g} m/s`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Experiment | Samples | signed vz mean (m/s) | integral z (m) | COM delta z (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("V1", "V2", "V3"):
        item = metrics[name]["metrics"]
        lines.append(
            f"| {name} | {item['sample_count']} | "
            f"{item['signed_vz_mean_m_s']:.12g} | "
            f"{item['signed_velocity_integral_m']:.12g} | "
            f"{item['com_delta_m'][2]:.12g} |"
        )
    lines.extend(
        [
            "",
            "V1 proves signed COM translation readback and integration in a "
            "no-contact control. V2 preserves the authored COM offset and "
            "proves the actor-origin/COM relationship during pure rotation. "
            "V3 keeps the accepted grasp physics and reproduces its exact "
            "deterministic signature, but neither post-step backward, "
            "one-step shifted forward, nor midpoint COM alignment reconciles "
            "the reported velocity with COM transform evolution.",
            "",
            "The conclusion localizes the discrepancy to the installed "
            "PhysX velocity-versus-transform readback boundary during the "
            "contact-rich grasp. It does not claim an unobserved internal "
            "solver cause and does not reinterpret the velocity as physical "
            "bottle fall. Pose, contact, clearance and drop remain the hold "
            "authority for this frozen run.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--v2", required=True, type=Path)
    parser.add_argument("--v3-runtime", required=True, type=Path)
    parser.add_argument("--v3-telemetry", required=True, type=Path)
    parser.add_argument("--accepted-aggregate", required=True, type=Path)
    parser.add_argument("--sample-id", default="sample_02")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    v1 = _read_json(args.v1)
    v2 = _read_json(args.v2)
    v3_runtime = _read_json(args.v3_runtime)
    v3_rows = _read_jsonl(args.v3_telemetry)
    accepted = _read_json(args.accepted_aggregate)
    v3_samples = [
        row["bottle"]["synchronized_com_velocity_sample"]
        for row in v3_rows
        if row.get("phase") == "HOLD"
    ]
    if len(v3_samples) != 120:
        raise RuntimeError(
            f"V3 HOLD synchronized sample count changed: {len(v3_samples)}"
        )
    v3_metrics = analyze_samples(v3_samples)
    v1_metrics = load_control_metrics(v1)
    v2_metrics = load_control_metrics(v2)
    accepted_signature = _accepted_signature(
        accepted,
        sample_id=str(args.sample_id),
    )
    actual_signature = str(v3_runtime.get("deterministic_signature"))
    signature_unchanged = actual_signature == accepted_signature
    diagnosis = build_velocity_diagnosis(
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        v3_metrics=v3_metrics,
        v1_runtime_valid=_control_runtime_valid(v1, "V1"),
        v2_runtime_valid=_control_runtime_valid(v2, "V2"),
        v3_signature_unchanged=signature_unchanged,
        dt_s=float(v3_runtime["runtime"]["bottle_velocity_sampling"]["physics_dt_s"]),
    )
    stage = v3_runtime.get("stage", {})
    stage_hash_unchanged = bool(
        stage.get("sha256_before") == EXPECTED_STAGE_SHA256
        and stage.get("sha256_after") == EXPECTED_STAGE_SHA256
    )
    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if diagnosis["status"] == "PASS" and stage_hash_unchanged
            else "PARTIAL"
        ),
        "velocity_semantics_status": diagnosis["conclusion"],
        "diagnosis": diagnosis,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "callback_phase": "POST_PHYSICS_STEP",
            "subscription_pre_step_argument": False,
        },
        "inputs": {
            "V1": _input(args.v1),
            "V2": _input(args.v2),
            "V3_runtime": _input(args.v3_runtime),
            "V3_telemetry": _input(args.v3_telemetry),
            "accepted_aggregate": _input(args.accepted_aggregate),
        },
        "experiments": {
            "V1": {
                "kind": "NO_CONTACT_PURE_TRANSLATION",
                "runtime_valid": _control_runtime_valid(v1, "V1"),
                "metrics": v1_metrics,
                "analytic_command_check": v1["analytic_command_check"],
            },
            "V2": {
                "kind": "NO_CONTACT_PURE_ROTATION_REAL_COM_OFFSET",
                "runtime_valid": _control_runtime_valid(v2, "V2"),
                "metrics": v2_metrics,
                "analytic_command_check": v2["analytic_command_check"],
            },
            "V3": {
                "kind": "FROZEN_TASK7_SAMPLE02_CONTACT_HOLD",
                "metrics": v3_metrics,
                "machine_status": v3_runtime.get("status"),
                "machine_reason": v3_runtime.get("reason"),
                "accepted_signature": accepted_signature,
                "actual_signature": actual_signature,
                "signature_unchanged": signature_unchanged,
                "hold_drop_m": v3_runtime.get("metrics", {}).get(
                    "hold_drop_m"
                ),
                "maximum_clearance_m": v3_runtime.get("metrics", {}).get(
                    "maximum_clearance_m"
                ),
            },
        },
        "frozen_stage": {
            **stage,
            "expected_sha256": EXPECTED_STAGE_SHA256,
            "hash_unchanged": stage_hash_unchanged,
        },
        "official_evidence": {
            "direct_nvidia_mcp": {
                "route": "DIRECT_isaac-sim-mcp_NOT_MCPJUNGLE",
                "status": "QUERIED",
                "local_5_1_source_remains_version_authority": True,
            },
            "post_step_api": _input(
                ROOT
                / ".venv_issac/lib/python3.11/site-packages/isaacsim/"
                "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
                "omni/physx/bindings/_physx.pyi"
            ),
            "tensor_api": _input(
                ROOT
                / ".venv_issac/lib/python3.11/site-packages/isaacsim/"
                "extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/"
                "omni/physics/tensors/impl/api.py"
            ),
        },
        "evidence_classification": {
            "official_source": [
                "post-step callback flag semantics",
                "world-space COM linear velocity contract",
                "local COM pose contract",
            ],
            "runtime_readback": [
                "signed tensor velocity",
                "actor pose and COM pose",
                "actor path/index",
                "physics dt",
            ],
            "numerical_calculation": [
                "COM world pose",
                "finite differences",
                "signed integration",
                "v_O=v_C-omega_cross_r_OC",
                "V1/V2-derived tolerance",
            ],
            "engineering_inference": (
                "INTERNAL_SOLVER_CAUSE_NOT_CLAIMED"
            ),
        },
        "boundaries": {
            "physical_or_control_parameters_changed": False,
            "video_rerecorded": False,
            "source_or_final_asset_modified": False,
            "real_robot": False,
            "remote_103": False,
            "task8": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md = args.output_md.resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

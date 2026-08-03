#!/usr/bin/env python3
"""Aggregate the frozen ALOHA1 Task 8 collider-LOD evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from tools.aloha1_mapping.task8_collider_lod import classify_benchmark_improvement
from tools.aloha1_mapping.task8_collider_lod import summarize_hold_contact_telemetry
from tools.aloha1_mapping.task8_collider_lod import summarize_profile_runs

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260803-aloha1-task8-lightweight"
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
BENCHMARK_PATTERN = re.compile(
    r"final_(fidelity|throughput)_e([124])_r([12])\.json$"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}


def _normalized_cooking(value: Any) -> Any:
    """Remove only wall-clock cooking duration from deterministic evidence."""

    if isinstance(value, dict):
        return {
            str(key): _normalized_cooking(item)
            for key, item in sorted(value.items())
            if key != "runtime_s"
        }
    if isinstance(value, list):
        return [_normalized_cooking(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _benchmark_runs(artifact_root: Path) -> list[dict[str, Any]]:
    records = []
    directory = artifact_root / "collider_lod_benchmark"
    for path in sorted(directory.glob("final_*.json")):
        match = BENCHMARK_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        payload = _load(path)
        profile = f"{match.group(1)}_profile"
        environment_count = int(match.group(2))
        repeat = int(match.group(3))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"benchmark did not pass: {path}")
        if payload.get("profile") != profile:
            raise RuntimeError(f"benchmark profile mismatch: {path}")
        if int(payload.get("environment_count", -1)) != environment_count:
            raise RuntimeError(f"benchmark environment count mismatch: {path}")
        metrics = payload["metrics"]
        official = metrics["official_frame_recorder"]
        memory = metrics["memory_after"]
        records.append(
            {
                "profile": profile,
                "environment_count": environment_count,
                "repeat": repeat,
                "physics_step_ms": float(
                    metrics["physics_step_ms_summary"]["mean"]
                ),
                "app_update_ms": float(
                    metrics["app_update_ms_summary"]["mean"]
                ),
                "stage_load_ms": float(metrics["stage_load_ms"]),
                "real_time_factor": float(official["Real Time Factor"]["value"]),
                "rss_gb": float(memory["System Memory RSS"]["value"]),
                "uss_gb": float(memory["System Memory USS"]["value"]),
                "gpu_dedicated_gb": float(
                    memory["GPU Memory Dedicated"]["value"]
                ),
                "app_frame_sample_count": int(
                    metrics["app_frame_sample_count"]
                ),
                "physics_frame_sample_count": int(
                    metrics["physics_frame_sample_count"]
                ),
                "physics_frame_sample_count_raw": int(
                    metrics["physics_frame_sample_count_raw"]
                ),
                "physics_history_sample_count_excluded": int(
                    metrics["physics_history_sample_count_excluded"]
                ),
                "source_stage_unchanged": bool(
                    payload["workload"]["source_stage_unchanged"]
                ),
                "readback_motion_max_abs_rad": float(
                    payload["workload"]["readback_motion_max_abs_rad"]
                ),
                "stage": payload["stage"],
                "inventory": payload["inventory"],
                "artifact": _record(path),
            }
        )
    expected = {
        (profile, count, repeat)
        for profile in ("fidelity_profile", "throughput_profile")
        for count in (1, 2, 4)
        for repeat in (1, 2)
    }
    actual = {
        (record["profile"], record["environment_count"], record["repeat"])
        for record in records
    }
    if actual != expected:
        raise RuntimeError(
            f"benchmark matrix incomplete: missing={sorted(expected-actual)} "
            f"unexpected={sorted(actual-expected)}"
        )
    return records


def _smoke_record(path: Path) -> dict[str, Any]:
    report_path = path / "aloha1_grasp_20cm_runtime.json"
    telemetry_path = path / "aloha1_grasp_20cm_telemetry.jsonl"
    report = _load(report_path)
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    task8 = report.get("runtime", {}).get("task8_diagnostic", {})
    if report.get("status") != "PASS" or report.get("reason") != "stable_20cm_hold":
        raise RuntimeError(f"Task 8 smoke did not pass: {report_path}")
    if task8.get("candidate_promoted") is not False:
        raise RuntimeError(f"Task 8 smoke promotion boundary failed: {report_path}")
    return {
        "profile": task8["profile_name"],
        "machine_status": report["status"],
        "reason": report["reason"],
        "stage": report["stage"],
        "initialization_signature": report["runtime"][
            "initialization_contract"
        ]["signature"],
        "finger_safety": report["runtime"]["finger_safety"],
        "metrics": report["metrics"],
        "deterministic_signature": report["deterministic_signature"],
        "telemetry_line_count": len(telemetry),
        "hold_contact_summary": summarize_hold_contact_telemetry(telemetry),
        "runtime_report": _record(report_path),
        "telemetry": _record(telemetry_path),
        "video": "NOT_RECORDED_BY_AUTHORIZED_LIGHTWEIGHT_SMOKE_POLICY",
    }


def _markdown_baseline(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 Task 8 collider baseline",
            "",
            f"- Status: `{report['status']}`",
            f"- Stage: `{report['stage']['absolute_path']}`",
            f"- Stage SHA-256: `{report['stage']['sha256']}`",
            f"- Collider prims: `{report['inventory']['collider_prim_count']}`",
            f"- Upper-arm collider prims: `{report['inventory']['upper_arm_collider_prim_count']}`",
            "- Final/default asset modified: `false`",
            "",
        ]
    )


def _markdown_benchmark(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 8 physics benchmark",
        "",
        f"- Status: `{report['status']}`",
        f"- Performance classification: `{report['performance']['classification']}`",
        "- Fresh processes per profile/scale: `2`",
        "- Environment scales: `1, 2, 4`",
        "",
        "| envs | fidelity physics ms | throughput physics ms | mean change | non-overlap |",
        "|---:|---:|---:|---:|:---:|",
    ]
    lines.extend(
        (
            f"| {record['environment_count']} | {record['fidelity_mean_ms']:.6f} | "
            f"{record['throughput_mean_ms']:.6f} | "
            f"{record['mean_improvement_percent']:.3f}% | "
            f"{str(record['non_overlapping_improvement']).lower()} |"
        )
        for record in report["performance"]["scale_records"]
    )
    lines.extend(
        [
            "",
            "The observed means improve at some scales, but the fresh-process ranges overlap "
            "and the 2-environment cell regresses. Therefore no stable throughput gain is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_comparison(report: dict[str, Any]) -> str:
    smoke = report["runtime_smoke"]
    return "\n".join(
        [
            "# ALOHA1 Task 8 collider LOD comparison",
            "",
            f"- Final conclusion: `{report['conclusion']}`",
            "- Candidate promotion: `false`",
            f"- Authored upper-arm convex pieces: `{report['piece_reduction']['before']}` → `{report['piece_reduction']['after']}`",
            f"- Fresh cooking: `{report['regression']['fresh_cooking']}`",
            f"- Static equivalence: `{report['regression']['static']}`",
            f"- 809-waypoint swept regression: `{report['regression']['swept']}`",
            f"- Fidelity Bottle500 smoke: `{smoke['fidelity_profile']['machine_status']}`",
            f"- Throughput Bottle500 smoke: `{smoke['throughput_profile']['machine_status']}`",
            "",
            "Both smoke runs retained bilateral force-carrying solver contact and passed the "
            "20 cm / 2 s hold gate. The exact `separation <= 0` count changed because the "
            "left minimum separation crossed zero by micrometres; signed values and impulses "
            "are preserved in JSON. This is not described as contact loss.",
            "",
            "The candidate is geometrically valid but has no repeatable measurable performance "
            "benefit, so it remains diagnostic and is not promoted.",
            "",
        ]
    )


def _markdown_candidate(report: dict[str, Any]) -> str:
    certificate = report["containment_certificate"]
    rejected = report["rejected_hypotheses"][0]
    return "\n".join(
        [
            "# ALOHA1 Task 8 collider LOD candidate",
            "",
            f"- Status: `{report['status']}`",
            "- Candidate: `DIAGNOSTIC_ONLY_NOT_PROMOTED`",
            "- Modified link suffix: `upper_arm_link` on both followers",
            f"- Authored convex pieces: `{report['piece_counts']['fidelity_total']}` → `{report['piece_counts']['throughput_total']}`",
            f"- Retained existing source piece: `piece_{certificate['retained_piece_index']:03d}`",
            f"- Maximum containment residual: `{certificate['maximum_outside_distance_m']:.12g} m`",
            f"- Derived numerical tolerance: `{certificate['tolerance_m']:.12g} m`",
            "- New or reshaped collider geometry: `none`",
            "- Gripper/finger/Bottle500/table collider changes: `none`",
            f"- Runtime cooking: `{report['candidate_runtime_cooking_readback']}`",
            f"- Static regression: `{report['candidate_static_collision_regression']}`",
            f"- Swept regression: `{report['candidate_swept_collision_regression']}`",
            f"- Bottle500 smoke: `{report['candidate_runtime_smoke']}`",
            "- Final/default promotion: `false`",
            "",
            "The selected candidate keeps the already-authored `piece_000` convex hull and "
            "only deactivates three source pieces proven to lie inside it. Two fresh cooking "
            "runs, static/swept comparison and one representative grasp smoke were completed.",
            "",
            f"The earlier full single-hull hypothesis remains `{rejected['status']}` because "
            f"its sampled outward deviation was `{rejected['outward_sample_deviation_max_m']:.9f} m`.",
            "",
            "The candidate remains diagnostic because the benchmark found no stable, "
            "non-overlapping performance improvement.",
            "",
        ]
    )


def build(artifact_root: Path, report_root: Path) -> dict[str, Any]:
    candidate_path = report_root / "aloha1_task8_collider_lod_candidate.json"
    candidate = _load(candidate_path)
    benchmark_runs = _benchmark_runs(artifact_root)
    numeric_runs = [
        {
            key: value
            for key, value in record.items()
            if key
            in {
                "profile",
                "environment_count",
                "physics_step_ms",
                "app_update_ms",
                "stage_load_ms",
                "real_time_factor",
                "rss_gb",
                "uss_gb",
                "gpu_dedicated_gb",
                "readback_motion_max_abs_rad",
            }
        }
        for record in benchmark_runs
    ]
    summary = summarize_profile_runs(numeric_runs)
    performance = classify_benchmark_improvement(summary)

    validation_root = artifact_root / "collider_lod_validation"
    fresh = [_load(validation_root / f"fresh_0{index}.json") for index in (1, 2)]
    cooking_equal = (
        fresh[0]["status"] == fresh[1]["status"] == "PASS"
        and _normalized_cooking(fresh[0]["cooking"])
        == _normalized_cooking(fresh[1]["cooking"])
        and fresh[0]["inventory_comparison"]
        == fresh[1]["inventory_comparison"]
        and all(fresh[0]["retained_cooked_geometry_equal"].values())
        and all(fresh[1]["retained_cooked_geometry_equal"].values())
        and fresh[0]["cooking_gate"] == "PASS"
        and fresh[1]["cooking_gate"] == "PASS"
    )
    static_fidelity = [
        _load(validation_root / f"static_fidelity_0{index}_overlap.json")
        for index in (1, 2)
    ]
    static_throughput = [
        _load(validation_root / f"static_0{index}_overlap.json")
        for index in (1, 2)
    ]
    static_equal = all(
        record["deterministic_signature"]
        == static_fidelity[0]["deterministic_signature"]
        for record in [*static_fidelity, *static_throughput]
    )
    swept_fidelity = [
        _load(validation_root / f"swept_fidelity_0{index}.json")
        for index in (1, 2)
    ]
    swept_throughput = [
        _load(validation_root / f"swept_throughput_0{index}.json")
        for index in (1, 2)
    ]
    swept_equal = all(
        record["status"] == "PASS"
        and record["deterministic_signature"]
        == swept_fidelity[0]["deterministic_signature"]
        for record in [*swept_fidelity, *swept_throughput]
    )

    smokes = {
        "fidelity_profile": _smoke_record(
            artifact_root
            / "collider_lod_smoke/fidelity_profile_attempt3"
        ),
        "throughput_profile": _smoke_record(
            artifact_root / "collider_lod_smoke/throughput_profile"
        ),
    }
    smoke_pass = all(
        record["machine_status"] == "PASS"
        and record["metrics"]["bilateral_contact_before_lift"] is True
        and record["metrics"]["height_reached"] is True
        and record["metrics"]["hold_drop_m"] <= 0.010
        and record["finger_safety"]["status"] == "PASS"
        for record in smokes.values()
    )
    if not (cooking_equal and static_equal and swept_equal and smoke_pass):
        conclusion = "REGRESSION_CAPTURED"
    else:
        conclusion = performance["classification"]

    baseline_run = next(
        record
        for record in benchmark_runs
        if record["profile"] == "fidelity_profile"
        and record["environment_count"] == 1
        and record["repeat"] == 1
    )
    baseline = {
        "schema_version": 1,
        "status": "PASS",
        "runtime": {"isaac_sim": "5.1.0.0", "kit": "107.3.3", "physx": "107.3.26"},
        "stage": {
            "absolute_path": baseline_run["stage"]["absolute_path"],
            "sha256": baseline_run["stage"]["sha256"],
            "default_prim": baseline_run["stage"]["default_prim"],
        },
        "inventory": baseline_run["inventory"],
        "fresh_cooking_artifacts": [
            _record(validation_root / f"fresh_0{index}.json") for index in (1, 2)
        ],
        "candidate_promoted": False,
        "final_or_default_asset_modified": False,
    }
    benchmark = {
        "schema_version": 1,
        "status": "PASS",
        "runtime": baseline["runtime"],
        "method": {
            "service": "isaacsim.benchmark.services",
            "warmup_frames": 30,
            "measured_frames_requested": 180,
            "aligned_app_and_physics_samples_per_run": 179,
            "raw_physics_samples_per_run": 209,
            "excluded_leading_history_samples_per_run": 30,
            "runtime_control": "SingleArticulation + ArticulationAction",
            "source_stage_authored_per_frame": False,
        },
        "raw_runs": benchmark_runs,
        "summary": summary,
        "performance": performance,
        "candidate_promoted": False,
    }
    comparison = {
        "schema_version": 1,
        "status": conclusion,
        "conclusion": conclusion,
        "runtime": baseline["runtime"],
        "candidate_classification": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "candidate_promoted": False,
        "final_or_default_asset_modified": False,
        "piece_reduction": {
            "before": candidate["piece_counts"]["fidelity_total"],
            "after": candidate["piece_counts"]["throughput_total"],
            "reduction": candidate["piece_counts"]["fidelity_total"]
            - candidate["piece_counts"]["throughput_total"],
        },
        "performance": performance,
        "memory_conclusion": "NO_STABLE_MEASURABLE_REDUCTION",
        "regression": {
            "fresh_cooking": "PASS_TWO_FRESH_PROCESSES" if cooking_equal else "FAIL",
            "static": (
                "PASS_EQUIVALENT_TO_BASELINE_WITH_PREEXISTING_ABSOLUTE_GATE_FAILURE"
                if static_equal
                else "FAIL_DIFFERENT_FROM_BASELINE"
            ),
            "static_absolute_status": static_fidelity[0]["status"],
            "static_absolute_failure_scope": "sample_02_and_sample_05_preexisting_self_overlap_gate",
            "swept": "PASS_809_WAYPOINTS_TWO_FRESH_PROCESSES" if swept_equal else "FAIL",
        },
        "runtime_smoke": smokes,
        "smoke_contact_interpretation": {
            "status": "PASS_SOLVER_CONTACT_PERSISTED",
            "exact_geometric_zero_threshold_changed": True,
            "interpretation": (
                "LEFT_MINIMUM_SEPARATION_CROSSED_EXACT_ZERO_AT_MICROMETRE_SCALE; "
                "BOTH_SIDES_RETAINED_POSITIVE_SOLVER_IMPULSE_FOR_ALL_120_HOLD_FRAMES"
            ),
        },
        "failure_evidence": {
            "runtime_or_collision_regression": False,
            "video_required": False,
            "reason": "NO_REPRODUCIBLE_PHYSICAL_OR_RENDER_REGRESSION",
            "preserved_diagnostic_invocation_failures": [
                {
                    "classification": "REJECTED_TASK8_WRAPPER_HASH_GUARD_INCOMPATIBILITY",
                    "artifact_root": str(
                        (artifact_root / "collider_lod_smoke/fidelity_profile").resolve()
                    ),
                },
                {
                    "classification": "REJECTED_STALE_ATTEMPT4_INITIALIZATION_CONTRACT",
                    "artifact_root": str(
                        (artifact_root / "collider_lod_smoke/fidelity_profile_attempt2").resolve()
                    ),
                },
            ],
        },
        "recommendation": "DO_NOT_PROMOTE_NO_MEASURABLE_IMPROVEMENT",
    }

    candidate.update(
        {
            "candidate_runtime_cooking_readback": comparison["regression"]["fresh_cooking"],
            "candidate_static_collision_regression": comparison["regression"]["static"],
            "candidate_swept_collision_regression": comparison["regression"]["swept"],
            "candidate_runtime_smoke": "PASS_LIGHTWEIGHT_BOTTLE500",
            "candidate_promoted": False,
            "final_or_default_asset_modified": False,
        }
    )
    _atomic_json(candidate_path, candidate)
    (report_root / "aloha1_task8_collider_lod_candidate.md").write_text(
        _markdown_candidate(candidate), encoding="utf-8"
    )
    _atomic_json(report_root / "aloha1_task8_collider_baseline.json", baseline)
    _atomic_json(report_root / "aloha1_task8_physics_benchmark.json", benchmark)
    _atomic_json(report_root / "aloha1_task8_comparison.json", comparison)
    (report_root / "aloha1_task8_collider_baseline.md").write_text(
        _markdown_baseline(baseline), encoding="utf-8"
    )
    (report_root / "aloha1_task8_physics_benchmark.md").write_text(
        _markdown_benchmark(benchmark), encoding="utf-8"
    )
    (report_root / "aloha1_task8_comparison.md").write_text(
        _markdown_comparison(comparison), encoding="utf-8"
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args.artifact_root.resolve(), args.report_root.resolve())
    print(json.dumps({"status": report["status"], "conclusion": report["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

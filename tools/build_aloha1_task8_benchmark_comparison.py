#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task8_optimization import compare_lower_is_better

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260803-aloha1-task8-lightweight/benchmark"
OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task8_benchmark_comparison.json"
MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_task8_benchmark_comparison.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(run_id: str) -> dict[str, Any]:
    path = ARTIFACT_ROOT / f"{run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"run_id": run_id, "path": str(path.resolve()), "sha256": _sha256(path), **payload}


def _metric(run: dict[str, Any], name: str) -> float:
    metrics = run["metrics"]
    if name == "stage_load_ms":
        return float(metrics["stage_load_ms"])
    if name == "app_frame_ms":
        return float(metrics["official_frame_recorder"]["Mean App_Update Frametime"]["value"])
    if name == "physics_frame_ms":
        return float(metrics["official_frame_recorder"]["Mean Physics Frametime"]["value"])
    if name == "rss_gb":
        return float(metrics["memory_after"]["System Memory RSS"]["value"])
    if name == "gpu_dedicated_gb":
        return float(metrics["memory_after"]["GPU Memory Dedicated"]["value"])
    raise KeyError(name)


def build_report() -> dict[str, Any]:
    baseline = [_load(f"baseline_0{index}") for index in range(1, 4)]
    candidate = [_load(f"candidate_0{index}") for index in range(1, 4)]
    if any(run["status"] != "PASS" for run in baseline + candidate):
        raise RuntimeError("not all fresh-process benchmark cells passed")
    comparisons = {
        metric: compare_lower_is_better(
            [_metric(run, metric) for run in baseline],
            [_metric(run, metric) for run in candidate],
        )
        for metric in (
            "stage_load_ms",
            "app_frame_ms",
            "physics_frame_ms",
            "rss_gb",
            "gpu_dedicated_gb",
        )
    }
    improvements = [
        name
        for name, value in comparisons.items()
        if value["classification"] == "IMPROVES_NONOVERLAPPING_RANGE"
    ]
    regressions = [
        name
        for name, value in comparisons.items()
        if value["classification"] == "WORSENS_NONOVERLAPPING_RANGE"
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "NO_MEASURABLE_IMPROVEMENT",
        "classification": "TASK8_VISUAL_MATERIAL_CANDIDATE_NOT_PROMOTED",
        "method": {
            "fresh_process_runs_per_profile": 3,
            "warmup_frames": 30,
            "measured_frames": 180,
            "official_recorder": "isaacsim.benchmark.services 5.1",
            "decision_rule": "nonoverlapping fresh-process ranges; lower is better",
        },
        "inputs": {"baseline": baseline, "candidate": candidate},
        "comparisons": comparisons,
        "nonoverlapping_improvements": improvements,
        "nonoverlapping_regressions": regressions,
        "decision": {
            "candidate_promoted": False,
            "run_grasp_smoke": False,
            "reason": "No nonoverlapping performance improvement; physics frame time is reproducibly worse.",
            "final_or_default_asset_modified": False,
        },
        "visual_evidence": {
            "status": "NOT_APPLICABLE_PERFORMANCE_ONLY_NO_VISUAL_OR_PHYSICAL_FAILURE",
            "reason": "The rejected result is a sub-millisecond benchmark difference, not a visible render, collision, or grasp failure.",
        },
    }
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 8 benchmark comparison",
        "",
        f"Status: `{report['status']}`",
        "",
        "Three fresh Isaac Sim 5.1 processes per profile used the local official "
        "`isaacsim.benchmark.services` frame and memory recorders. Lower is better; "
        "only nonoverlapping run ranges are classified as directional evidence.",
        "",
        "| Metric | Baseline mean | Candidate mean | Delta | Classification |",
        "|---|---:|---:|---:|---|",
    ]
    for name, value in report["comparisons"].items():
        lines.append(
            f"| {name} | {value['baseline']['mean']:.6g} | "
            f"{value['candidate']['mean']:.6g} | "
            f"{value['candidate_minus_baseline_percent']:.3f}% | "
            f"`{value['classification']}` |"
        )
    lines.extend(
        [
            "",
            "The candidate is not promoted and no grasp smoke is run because it has no "
            "nonoverlapping improvement while physics frame time is reproducibly worse. "
            "No screenshot or video is applicable to this sub-millisecond performance-only "
            "negative result; no visible render, collision or grasp failure occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

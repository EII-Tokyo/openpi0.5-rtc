#!/usr/bin/env python3
"""Aggregate two fresh Isaac compound-cooking runs into a review certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.cad_compound_contact_candidate import classify_fresh_runtime_runs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for side, record in report["fingers"].items():
        rows.append(
            f"| {side} | {record['source_piece_count']} | "
            f"{record['runtime_cooked_piece_count']} | "
            f"{record['exact_ray_coverage_ratio']:.9f} | "
            f"{record['tolerance_adjusted_coverage_ratio']:.9f} | "
            f"{record['maximum_outward_crossing_m'] * 1.0e9:.6f} | "
            f"{(record['maximum_quantization_surface_distance_m'] or 0.0) * 1.0e9:.6f} | "
            f"{record['contact_region_status']} |"
        )
    return "\n".join(
        [
            "# ALOHA1 supplier-CAD compound runtime cooking certificate",
            "",
            f"- Status: **{report['status']}**",
            f"- Fresh-process determinism: **{report['fresh_process_determinism']['status']}**",
            f"- Asset decision: **{report['asset_decision']}**",
            "- Final/default collider modified: `false`",
            "- Timeline/video: `NOT_APPLICABLE_STATIC_COOKING_ONLY`",
            "",
            "| side | source pieces | cooked pieces | exact-ray coverage | tolerance-adjusted coverage | max outward crossing (nm) | max quantization distance (nm) | gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "The exact-ray coverage ratio is intentionally retained. PhysX stores "
            "the cooked vertices at float32 precision, so source points displaced "
            "by nanometres can lie just outside an exact half-space test. The "
            "adjusted result accepts only points whose nearest cooked surface "
            "distance and normal projection are both below the previously derived "
            "`MAX(OCCT membership tolerance, 8 float32 ULP)` floor.",
            "",
            "The first rejected report is preserved as "
            "`REJECTED_CERTIFICATE_EXACT_RAY_FALSE_NEGATIVE`; it did not prove a "
            "geometry failure. This certificate covers only the central, CAD-derived "
            "contact rectangle. Full-face coverage, articulation integration, "
            "contact dynamics and asset promotion remain outside this gate.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-run", type=Path, action="append", required=True)
    parser.add_argument("--rejected-attempt", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.runtime_run) != 2:
        raise ValueError("exactly two fresh runtime reports are required")
    loaded = [_load(path) for path in args.runtime_run]
    runs = [record for _, record in loaded]
    deterministic = classify_fresh_runtime_runs(runs)
    rejected_path, rejected = _load(args.rejected_attempt)
    fingers: dict[str, Any] = {}
    for side in ("left", "right"):
        first = runs[0]["fingers"][side]
        second = runs[1]["fingers"][side]
        if first["contact_region_certificate"] != second["contact_region_certificate"]:
            raise RuntimeError(f"non-deterministic contact certificate: {side}")
        certificate = first["contact_region_certificate"]
        fingers[side] = {
            "source_piece_count": first["source_piece_count"],
            "runtime_cooked_piece_count": first["runtime_cooked_piece_count"],
            "exact_ray_coverage_ratio": certificate["exact_ray_coverage_ratio"],
            "tolerance_adjusted_coverage_ratio": certificate["tolerance_adjusted_coverage_ratio"],
            "quantization_boundary_sample_count": certificate["quantization_boundary_sample_count"],
            "maximum_outward_crossing_m": certificate["positive_exit_distance_max_m"],
            "maximum_quantization_surface_distance_m": certificate["uncovered_nearest_surface_max_m"],
            "numeric_tolerance_m": first["numeric_tolerance"]["numeric_tolerance_m"],
            "contact_region_status": first["contact_region_status"],
            "full_face_scope": first["full_face_scope"],
            "full_brep_face_certificate": first["full_brep_face_certificate"],
            "coordinate_frame": first["coordinate_frame"],
        }
    overall_pass = deterministic["status"] == "PASS_DETERMINISTIC_FRESH_PROCESS_COOKING" and all(
        record["contact_region_status"] == "PASS" for record in fingers.values()
    )
    report = {
        "schema_version": 1,
        "status": (
            "PASS_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
            if overall_pass
            else "FAIL_RUNTIME_COOKED_CONTACT_REGION_GEOMETRY_NOT_PROMOTED"
        ),
        "scope": "CENTRAL_CAD_DERIVED_CONTACT_RECTANGLE_ONLY",
        "coordinate_frame": runs[0]["fingers"]["left"]["coordinate_frame"],
        "runtime": runs[0]["runtime"],
        "fresh_process_determinism": deterministic,
        "runtime_runs": [
            {
                "absolute_path": str(path),
                "sha256": _sha256(path),
                "process_id": record["process_id"],
                "deterministic_signature": record["deterministic_signature"],
            }
            for path, record in loaded
        ],
        "rejected_attempt": {
            "status": "REJECTED_CERTIFICATE_EXACT_RAY_FALSE_NEGATIVE",
            "absolute_path": str(rejected_path),
            "sha256": _sha256(rejected_path),
            "original_status": rejected["status"],
            "reason": (
                "exact-ray containment used near-zero floating tolerance despite "
                "the prederived OCCT/float32 comparison floor"
            ),
        },
        "fingers": fingers,
        "asset_decision": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "diagnostic_usd_status": "NOT_CREATED",
        "timeline_started": False,
        "video_status": "NOT_APPLICABLE_STATIC_COOKING_ONLY",
        "final_or_default_collider_modified": False,
        "task8_default_asset_optimization_status": "NOT_STARTED",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.json_output)}))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

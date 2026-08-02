#!/usr/bin/env python3
"""Compare Isaac 5.1 cooked finger convexes with exact supplier B-Rep faces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.finger_cooked_contact_certificate import classify_exact_brep_profiles
from tools.aloha1_mapping.finger_cooked_contact_certificate import derive_cooked_brep_numeric_tolerance
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_exact_brep_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_two(paths: list[Path], label: str) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError(f"exactly two fresh {label} reports are required")
    return [
        (path.resolve(strict=True), json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for side in ("left", "right"):
        for approximation in ("convexHull", "convexDecomposition"):
            profile = report["profiles_by_side"][side][approximation]
            rows.append(
                f"| {side} | `{approximation}` | {profile['piece_count']} | "
                f"{profile['exact_brep_sample_count']} | "
                f"{profile['source_point_coverage_ratio']:.6f} | "
                f"{profile['maximum_inward_crossing_m'] * 1000.0:.6f} | "
                f"{profile['maximum_undercoverage_m'] * 1000.0:.6f} | "
                f"{profile['exact_surface_status']} |"
            )
    return "\n".join(
        [
            "# ALOHA1 supplier-CAD B-Rep / cooked finger certificate",
            "",
            f"- Status: **{report['status']}**",
            f"- Exact surface: **{report['comparison']['exact_surface_status']}**",
            f"- Decomposition: **{report['comparison']['decomposition_comparison']}**",
            f"- Asset decision: **{report['comparison']['asset_decision']}**",
            "- Runtime grasp/hold claim: `NOT_MADE`",
            "- Final/default collider modified: `false`",
            "",
            "| Side | Approximation | Pieces | Exact B-Rep samples | Cooked coverage | Max inward crossing (mm) | Max undercoverage (mm) | Exact gate |",
            "|---|---|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "The contact points are evaluated directly on the audited, trimmed "
            "OCCT B-Rep faces in two fresh FreeCAD 1.1.1 / OCCT 7.8.1 "
            "processes. No OBJ tessellation supplies these points. The exact "
            "crossing gate uses only a derived numerical floor: the maximum of "
            "the OCCT membership tolerance and eight float32 ULPs at the largest "
            "sample coordinate.",
            "",
            "A failed exact-surface gate proves that the approximation is not an "
            "exact contact surface. It does not by itself define how much error is "
            "acceptable for the bottle task; that task-local approximation "
            "tolerance remains a HARD_BLOCKER and was not fitted from successful "
            "grasp videos.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooking-run", type=Path, action="append", required=True)
    parser.add_argument("--brep-run", type=Path, action="append", required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/"
            "aloha1_supplier_cad_finger_brep_cooked_certificate.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/"
            "aloha1_supplier_cad_finger_brep_cooked_certificate.md"
        ),
    )
    args = parser.parse_args()

    cooking_loaded = _load_two(args.cooking_run, "Isaac cooking")
    cooking_runs = [run for _, run in cooking_loaded]
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    if any(run["status"] != "PASS" for run in cooking_runs):
        raise RuntimeError("an Isaac cooking run did not pass")
    if any(run["runtime"] != expected_runtime for run in cooking_runs):
        raise RuntimeError("Isaac cooking runtime mismatch")
    if len({run["process_id"] for run in cooking_runs}) != 2:
        raise RuntimeError("Isaac cooking reports are not fresh processes")

    brep_paths = [path.resolve(strict=True) for path in args.brep_run]
    brep_by_side = {
        side: load_exact_brep_contact_surface(brep_paths, side)
        for side in ("left", "right")
    }
    numeric_by_side = {
        side: derive_cooked_brep_numeric_tolerance(
            contact["samples_m"],
            brep_membership_tolerance_m=contact[
                "brep_membership_tolerance_m"
            ],
        )
        for side, contact in brep_by_side.items()
    }
    comparison_tolerance_m = max(
        record["numeric_tolerance_m"] for record in numeric_by_side.values()
    )

    profiles_by_side: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        contact = brep_by_side[side]
        for approximation in ("convexHull", "convexDecomposition"):
            first = cooking_runs[0]["profiles"][approximation][side]
            second = cooking_runs[1]["profiles"][approximation][side]
            signatures = [first["geometry_signature"], second["geometry_signature"]]
            if len(set(signatures)) != 1:
                raise RuntimeError(
                    f"cooked geometry is not deterministic: {side}/{approximation}"
                )
            envelope = summarize_contact_envelope(
                contact["samples_m"],
                contact["normal"],
                first["pieces"],
                tessellation_budget_m=comparison_tolerance_m,
            )
            crossing = float(envelope["positive_exit_distance_max_m"] or 0.0)
            undercoverage = float(
                envelope["uncovered_nearest_surface_max_m"] or 0.0
            )
            profiles_by_side[side][approximation] = {
                "piece_count": first["piece_count"],
                "geometry_signature": signatures[0],
                "exact_brep_face_index_1_based": contact["face_index_1_based"],
                "exact_brep_sample_count": contact["sample_count"],
                "source_point_coverage_ratio": envelope[
                    "source_point_coverage_ratio"
                ],
                "maximum_inward_crossing_m": crossing,
                "maximum_inward_crossing_sample_index": envelope[
                    "maximum_inward_crossing_sample_index"
                ],
                "maximum_inward_crossing_source_point_m": envelope[
                    "maximum_inward_crossing_source_point_m"
                ],
                "maximum_inward_crossing_target_point_m": envelope[
                    "maximum_inward_crossing_target_point_m"
                ],
                "maximum_undercoverage_m": undercoverage,
                "maximum_combined_sample_deviation_m": envelope[
                    "maximum_contact_surface_deviation_m"
                ],
                "maximum_combined_deviation_kind": envelope[
                    "maximum_deviation_kind"
                ],
                "exact_surface_status": (
                    "FAIL_CROSSES_INWARD_CAD_SURFACE"
                    if crossing > comparison_tolerance_m
                    else "PASS_NO_SAMPLED_CROSSING_BEYOND_NUMERIC_TOLERANCE"
                ),
            }

    comparison = classify_exact_brep_profiles(
        profiles_by_side,
        numeric_tolerance_m=comparison_tolerance_m,
    )
    report = {
        "schema_version": 1,
        "status": "PASS_DETERMINISTIC_MEASUREMENT_FAIL_EXACT_SURFACE_GATE",
        "scope": "SUPPLIER_CAD_TRIMMED_BREP_CONTACT_FACE_VS_ISAAC_COOKED_CONVEX_UNION",
        "runtime": expected_runtime,
        "cooking_runs": [
            {
                "absolute_path": str(path),
                "sha256": _sha256(path),
                "process_id": run["process_id"],
            }
            for path, run in cooking_loaded
        ],
        "brep_runs": [
            {
                "absolute_path": str(path),
                "sha256": _sha256(path),
                "process_id": report_data["process_id"],
                "deterministic_signature": report_data[
                    "deterministic_signature"
                ],
            }
            for path, report_data in _load_two(args.brep_run, "FreeCAD B-Rep")
        ],
        "brep_sampling": {
            "source_geometry": "trimmed OCCT B-Rep face",
            "tessellation_used_for_contact_points": False,
            "fresh_process_count": 2,
            "deterministic_signature": brep_by_side["left"][
                "deterministic_signature"
            ],
        },
        "numeric_tolerance_by_side": numeric_by_side,
        "comparison_numeric_tolerance_m": comparison_tolerance_m,
        "profiles_by_side": profiles_by_side,
        "comparison": comparison,
        "runtime_hold_claim": "NOT_MADE",
        "timeline_started": False,
        "stage_saved": False,
        "task_local_approximation_tolerance": (
            "HARD_BLOCKER_NOT_DERIVED_OR_MEASURED"
        ),
        "final_or_default_collider_modified": False,
        "task8_optimization_candidate": "NOT_AUTHORED",
    }
    signature_payload = json.dumps(
        report, sort_keys=True, separators=(",", ":")
    ).encode()
    report["deterministic_signature"] = hashlib.sha256(
        signature_payload
    ).hexdigest()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "exact_surface_status": comparison["exact_surface_status"],
                "output": str(args.json_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

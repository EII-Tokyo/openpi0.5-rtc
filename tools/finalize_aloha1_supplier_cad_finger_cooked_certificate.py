#!/usr/bin/env python3
"""Finalize two fresh supplier-CAD finger cooking runs into one certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.finger_cooked_contact_certificate import classify_profile_comparison
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_supplier_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runs(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError("exactly two fresh-process cooking reports are required")
    return [
        (path.resolve(strict=True), json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for side in ("left", "right"):
        for approximation in ("convexHull", "convexDecomposition"):
            record = report["profiles_by_side"][side][approximation]
            envelope = record["contact_envelope"]
            rows.append(
                f"| {side} | `{approximation}` | {record['piece_count']} | "
                f"{envelope['source_point_coverage_ratio']:.6g} | "
                f"{envelope['maximum_contact_surface_deviation_m']:.9g} | "
                f"{envelope['status']} |"
            )
    return "\n".join(
        [
            "# ALOHA1 supplier-CAD finger cooked contact certificate",
            "",
            f"- Cooking status: **{report['status']}**",
            f"- Geometry classification: **{report['comparison']['classification']}**",
            f"- Asset decision: **{report['asset_decision']}**",
            "- Runtime hold claim: `NOT_MADE`",
            "- Final/default collider modified: `false`",
            "",
            "| Side | Approximation | Pieces | Exact source coverage | Maximum contact deviation (m) | Geometry gate |",
            "|---|---|---:|---:|---:|---|",
            *rows,
            "",
            "The maximum deviation combines outward normal-ray envelope for source "
            "points covered by the cooked union with nearest cooked-surface distance "
            "for uncovered source points. It is compared only with the pre-existing "
            "0.20 mm FreeCAD tessellation budget; no threshold was fitted from grasp "
            "success. This is a geometry certificate, not a grasp/hold promotion.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/"
            "aloha1_supplier_cad_finger_cooked_contact_certificate.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "reports/aloha1_mapping/"
            "aloha1_supplier_cad_finger_cooked_contact_certificate.md"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    loaded = _load_runs(args.run)
    runs = [run for _, run in loaded]
    expected_runtime = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    if any(run["status"] != "PASS" for run in runs):
        raise RuntimeError("at least one cooking run did not pass")
    if any(run["runtime"] != expected_runtime for run in runs):
        raise RuntimeError("cooking runtime version mismatch")
    if len({run["process_id"] for run in runs}) != 2:
        raise RuntimeError("cooking reports do not prove two distinct processes")

    profiles_by_side: dict[str, dict[str, Any]] = {
        "left": {},
        "right": {},
    }
    deterministic_records = []
    for side in ("left", "right"):
        contact = load_supplier_contact_surface(root, side)
        for approximation in ("convexHull", "convexDecomposition"):
            first = runs[0]["profiles"][approximation][side]
            second = runs[1]["profiles"][approximation][side]
            signatures = [first["geometry_signature"], second["geometry_signature"]]
            deterministic = len(set(signatures)) == 1
            if not deterministic:
                raise RuntimeError(f"cooked geometry is not deterministic: {side}/{approximation}")
            if first["source_sha256"] != contact["source_sha256"]:
                raise RuntimeError(f"supplier source mismatch: {side}/{approximation}")
            if first["approximation_readback"] != approximation:
                raise RuntimeError(f"approximation readback mismatch: {side}/{approximation}")
            if approximation == "convexDecomposition" and first[
                "decomposition_parameters_authored"
            ] is not False:
                raise RuntimeError("decomposition schema defaults were not preserved")
            envelope = summarize_contact_envelope(
                contact["samples"],
                np.asarray(contact["normal"], dtype=np.float64),
                first["pieces"],
                tessellation_budget_m=contact["tessellation_error_budget_m"],
            )
            profiles_by_side[side][approximation] = {
                "source_path": contact["source_path"],
                "source_sha256": contact["source_sha256"],
                "cad_face_index": contact["cad_face_index"],
                "cad_face_normal": contact["normal"],
                "piece_count": first["piece_count"],
                "sum_piece_volume_m3": first["sum_piece_volume_m3"],
                "geometry_signature": signatures[0],
                "contact_envelope": envelope,
            }
            deterministic_records.append(
                {
                    "side": side,
                    "approximation": approximation,
                    "run_geometry_signatures": signatures,
                    "deterministic": deterministic,
                }
            )

    comparison_input = {
        side: {
            approximation: profiles_by_side[side][approximation][
                "contact_envelope"
            ]
            for approximation in ("convexHull", "convexDecomposition")
        }
        for side in ("left", "right")
    }
    comparison = classify_profile_comparison(comparison_input)
    report = {
        "schema_version": 1,
        "status": "PASS_COOKING_DETERMINISTIC",
        "scope": "SUPPLIER_CAD_FINGER_LOCAL_CONTACT_GEOMETRY_ONLY",
        "runtime": expected_runtime,
        "source_identity_boundary": str(
            (
                root
                / "reports/aloha1_mapping/"
                "aloha1_finger_cooked_source_identity_boundary.json"
            ).resolve(strict=True)
        ),
        "raw_runs": [
            {
                "absolute_path": str(path),
                "sha256": _sha256(path),
                "process_id": run["process_id"],
            }
            for path, run in loaded
        ],
        "fresh_process_count": 2,
        "deterministic_records": deterministic_records,
        "local_api": runs[0]["local_api"],
        "profiles_by_side": profiles_by_side,
        "comparison": comparison,
        "asset_decision": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "runtime_hold_claim": "NOT_MADE",
        "timeline_started": False,
        "stage_saved": False,
        "final_or_default_collider_modified": False,
        "task8_runtime_optimization": "NOT_RUN",
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["deterministic_signature"] = hashlib.sha256(payload).hexdigest()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": comparison["classification"],
                "output": str(args.json_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

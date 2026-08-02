#!/usr/bin/env python3
"""Build an isolated contact-preserving finger compound geometry candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_compound_contact_candidate import build_contact_preserving_candidate
from tools.aloha1_mapping.cad_compound_contact_candidate import transform_contact_candidate
from tools.aloha1_mapping.cad_finger_installation import cad_global_to_finger_link_matrix
from tools.aloha1_mapping.finger_cooked_contact_certificate import derive_cooked_brep_numeric_tolerance
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_exact_brep_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import summarize_contact_envelope


def _load_two(paths: list[Path], label: str) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError(f"exactly two {label} reports are required")
    return [(path.resolve(strict=True), json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def _rectangle_samples(vertices: np.ndarray, count_per_axis: int = 17) -> np.ndarray:
    if vertices.shape != (4, 3):
        raise ValueError("rectangle requires four ordered vertices")
    return np.asarray(
        [
            (1.0 - first) * (1.0 - second) * vertices[0]
            + first * (1.0 - second) * vertices[1]
            + first * second * vertices[2]
            + (1.0 - first) * second * vertices[3]
            for first in np.linspace(0.0, 1.0, count_per_axis)
            for second in np.linspace(0.0, 1.0, count_per_axis)
        ],
        dtype=np.float64,
    )


def _geometry_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for side, record in report["fingers"].items():
        rectangle = record["contact_rectangle"]
        local = record["contact_region_certificate"]
        full = record["full_brep_face_certificate"]
        rows.append(
            f"| {side} | {rectangle['width_m'] * 1000.0:.6f} | "
            f"{rectangle['height_m'] * 1000.0:.6f} | "
            f"{rectangle['depth_m'] * 1000.0:.6f} | "
            f"{record['candidate_piece_count']} | "
            f"{local['source_point_coverage_ratio']:.6f} | "
            f"{(local['positive_exit_distance_max_m'] or 0.0) * 1000.0:.9f} | "
            f"{(full['uncovered_nearest_surface_max_m'] or 0.0) * 1000.0:.6f} |"
        )
    return "\n".join(
        [
            "# ALOHA1 supplier-CAD compound contact candidate",
            "",
            f"- Status: **{report['status']}**",
            f"- Contact-region gate: **{report['contact_region_gate']}**",
            f"- Full-face scope: **{report['full_face_scope']}**",
            f"- Asset decision: **{report['asset_decision']}**",
            "- Final/default collider modified: `false`",
            "",
            "| side | width (mm) | height (mm) | depth (mm) | pieces | contact coverage | max inward crossing (mm) | full-face max undercoverage (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The body pieces are the deterministic default Isaac decomposition "
            "clipped by the audited CAD contact plane. The added contact primitive "
            "is the maximum centered parameter rectangle obtained from exact OCCT "
            "face containment, extruded to the maximum uniform depth whose Boolean "
            "outside volume stays within the derived OCCT tolerance. No dimension "
            "was fitted from grasp success.",
            "",
            "This offline result certifies only the central CAD-derived contact "
            "rectangle. It does not certify the complete finger face, runtime "
            "cooking, contact stability, or final asset promotion.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooking-run", type=Path, action="append", required=True)
    parser.add_argument("--brep-run", type=Path, action="append", required=True)
    parser.add_argument("--prism-run", type=Path, action="append", required=True)
    parser.add_argument("--geometry-output", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/aloha1_mapping/aloha1_supplier_cad_compound_contact_candidate.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/aloha1_mapping/aloha1_supplier_cad_compound_contact_candidate.md"),
    )
    args = parser.parse_args()

    cooking_loaded = _load_two(args.cooking_run, "cooking")
    cooking = [record for _, record in cooking_loaded]
    if len({record["process_id"] for record in cooking}) != 2:
        raise RuntimeError("cooking reports do not prove fresh processes")
    prism_loaded = _load_two(args.prism_run, "FreeCAD prism")
    prism = [record for _, record in prism_loaded]
    if len({record["process_id"] for record in prism}) != 2:
        raise RuntimeError("prism reports do not prove fresh processes")
    if len({record["deterministic_signature"] for record in prism}) != 1:
        raise RuntimeError("FreeCAD contact primitive derivation is not deterministic")

    brep_paths = [path.resolve(strict=True) for path in args.brep_run]
    geometry_fingers = {}
    report_fingers = {}
    contact_gate_pass = True
    for side in ("left", "right"):
        first_cooked = cooking[0]["profiles"]["convexDecomposition"][side]
        second_cooked = cooking[1]["profiles"]["convexDecomposition"][side]
        if first_cooked["geometry_signature"] != second_cooked["geometry_signature"]:
            raise RuntimeError(f"non-deterministic cooked source: {side}")
        exact = load_exact_brep_contact_surface(brep_paths, side)
        tolerance = derive_cooked_brep_numeric_tolerance(
            exact["samples_m"],
            brep_membership_tolerance_m=exact["brep_membership_tolerance_m"],
        )
        prism_record = prism[0]["fingers"][side]["centered_contact_rectangle"]
        rectangle_vertices = np.asarray(prism_record["front_vertices_mm"], dtype=np.float64) * 0.001
        rectangle_triangles = np.asarray([rectangle_vertices[triangle] for triangle in prism_record["front_triangles"]])
        depth_m = prism_record["uniform_inward_depth"]["maximum_feasible_lower_bound_mm"] * 0.001
        candidate = build_contact_preserving_candidate(
            cooked_pieces=first_cooked["pieces"],
            contact_triangles=rectangle_triangles,
            plane_point=rectangle_vertices.mean(axis=0),
            outward_normal=exact["normal"],
            contact_prism_depth_m=depth_m,
            numeric_tolerance_m=tolerance["numeric_tolerance_m"],
        )
        contact_samples = _rectangle_samples(rectangle_vertices)
        contact_certificate = summarize_contact_envelope(
            contact_samples,
            exact["normal"],
            candidate["pieces"],
            tessellation_budget_m=tolerance["numeric_tolerance_m"],
        )
        full_face_certificate = summarize_contact_envelope(
            exact["samples_m"],
            exact["normal"],
            candidate["pieces"],
            tessellation_budget_m=tolerance["numeric_tolerance_m"],
        )
        contact_pass = (
            contact_certificate["source_point_coverage_ratio"] == 1.0
            and float(contact_certificate["positive_exit_distance_max_m"] or 0.0) <= tolerance["numeric_tolerance_m"]
        )
        contact_gate_pass &= contact_pass
        edge_lengths = [
            float(np.linalg.norm(rectangle_vertices[(index + 1) % 4] - vertex))
            for index, vertex in enumerate(rectangle_vertices)
        ]
        geometry_fingers[side] = transform_contact_candidate(
            {
                "pieces": candidate["pieces"],
                "outward_normal": exact["normal"].tolist(),
                "plane_point_m": rectangle_vertices.mean(axis=0).tolist(),
                "contact_rectangle_vertices_m": rectangle_vertices.tolist(),
                "contact_rectangle_triangles": prism_record["front_triangles"],
                "contact_prism_depth_m": depth_m,
            },
            np.asarray(cad_global_to_finger_link_matrix(side), dtype=np.float64),
        )
        report_fingers[side] = {
            "source_cooked_geometry_signature": first_cooked["geometry_signature"],
            "candidate_geometry_signature": _geometry_signature(geometry_fingers[side]),
            "output_coordinate_frame": "FINGER_LINK_LOCAL",
            "cad_global_to_finger_link_matrix": geometry_fingers[side]["rigid_transform_matrix"],
            "transform_determinant": geometry_fingers[side]["rigid_transform_determinant"],
            "mirror_used": geometry_fingers[side]["mirror_used"],
            "candidate_piece_count": candidate["piece_count"],
            "clipped_body_piece_count": candidate["clipped_body_piece_count"],
            "discarded_body_piece_count": candidate["discarded_body_piece_count"],
            "contact_prism_piece_count": candidate["contact_prism_piece_count"],
            "contact_rectangle": {
                "width_m": min(edge_lengths),
                "height_m": max(edge_lengths),
                "depth_m": depth_m,
                "area_m2": prism_record["area_mm2"] * 1.0e-6,
                "source_scale_lower_bound": prism_record["maximum_feasible_scale_lower_bound"],
                "seed": prism_record["seed"],
                "cad_boolean_depth_certificate": prism_record["uniform_inward_depth"],
            },
            "numeric_tolerance": tolerance,
            "contact_region_certificate": contact_certificate,
            "full_brep_face_certificate": full_face_certificate,
            "contact_region_status": "PASS" if contact_pass else "FAIL",
        }

    geometry_payload = {
        "schema_version": 1,
        "scope": ("DIAGNOSTIC_CAD_DERIVED_COMPOUND_CONTACT_GEOMETRY_FINGER_LINK_LOCAL_NOT_USD"),
        "source_coordinate_frame": "STEP_ASSEMBLY_GLOBAL_METRES",
        "output_coordinate_frame": "FINGER_LINK_LOCAL_METRES",
        "source_prism_signature": prism[0]["deterministic_signature"],
        "fingers": geometry_fingers,
        "final_or_default_collider_modified": False,
    }
    geometry_payload["deterministic_signature"] = _geometry_signature(geometry_payload)
    args.geometry_output.parent.mkdir(parents=True, exist_ok=True)
    args.geometry_output.write_text(
        json.dumps(geometry_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": (
            "PASS_OFFLINE_CONTACT_REGION_GEOMETRY" if contact_gate_pass else "FAIL_OFFLINE_CONTACT_REGION_GEOMETRY"
        ),
        "scope": "CAD_DERIVED_CENTRAL_CONTACT_RECTANGLE_ONLY",
        "contact_region_gate": "PASS" if contact_gate_pass else "FAIL",
        "full_face_scope": "PARTIAL_CONTACT_REGION_ONLY",
        "asset_decision": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "prism_fresh_process_count": 2,
        "prism_deterministic_signature": prism[0]["deterministic_signature"],
        "geometry_output_absolute_path": str(args.geometry_output.resolve()),
        "geometry_output_sha256": hashlib.sha256(args.geometry_output.read_bytes()).hexdigest(),
        "fingers": report_fingers,
        "isaac_runtime_cooking": "NOT_RUN",
        "runtime_grasp_hold": "NOT_RUN",
        "timeline_started": False,
        "final_or_default_collider_modified": False,
    }
    report["deterministic_signature"] = _geometry_signature(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "contact_region_gate": report["contact_region_gate"],
                "geometry": str(args.geometry_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

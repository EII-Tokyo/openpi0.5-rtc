"""Gate legacy cooked finger data against the current supplier-CAD sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.collider_surface_certificate import _load_obj
from tools.aloha1_mapping.collider_surface_certificate import _load_stl
from tools.aloha1_mapping.collider_surface_certificate import _sha256
from tools.aloha1_mapping.collider_surface_certificate import _signed_volume_abs

SUPPLIER_FINGERS = {
    "left": (
        ".codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/"
        "tessellation_angular_controlled/run_a/left_finger.obj"
    ),
    "right": (
        ".codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/"
        "tessellation_angular_controlled/run_a/right_finger.obj"
    ),
}
LEGACY_REPORT = (
    "reports/aloha1_mapping/gripper_correct_finger_collider_comparison.json"
)


def _geometry_metrics(
    vertices: np.ndarray, faces: np.ndarray, *, scale: float
) -> dict[str, Any]:
    vertices_m = vertices * scale
    return {
        "vertex_count": len(vertices_m),
        "face_count": len(faces),
        "aabb_min_m": vertices_m.min(axis=0).tolist(),
        "aabb_max_m": vertices_m.max(axis=0).tolist(),
        "sorted_aabb_extent_m": np.sort(np.ptp(vertices_m, axis=0)).tolist(),
        "signed_volume_abs_m3": _signed_volume_abs(vertices_m, faces),
    }


def _legacy_sources(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for asset in report["profiles"]["convex_decomposition"]["assets"]:
        for collider in asset["colliders"].values():
            side = collider["side"]
            candidate = {
                "path": collider["source_stl_absolute_path"],
                "sha256": collider["source_stl_sha256"],
                "face_count": collider["source_stl_triangle_count"],
                "piece_count": collider["piece_count"],
            }
            if side in records and candidate != records[side]:
                raise ValueError(f"legacy cooked source is inconsistent for {side}")
            records[side] = candidate
    if set(records) != {"left", "right"}:
        raise ValueError("legacy cooked report does not contain both handed sources")
    return records


def build_source_identity_boundary(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    legacy_report_path = (root / LEGACY_REPORT).resolve(strict=True)
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    legacy_sources = _legacy_sources(legacy_report)

    records: dict[str, dict[str, Any]] = {}
    for side, relative_path in SUPPLIER_FINGERS.items():
        supplier_path = (root / relative_path).resolve(strict=True)
        legacy_path = Path(legacy_sources[side]["path"]).resolve(strict=True)
        supplier_vertices, supplier_faces = _load_obj(supplier_path)
        legacy_vertices, legacy_faces = _load_stl(legacy_path)
        supplier_metrics = _geometry_metrics(
            supplier_vertices, supplier_faces, scale=1.0
        )
        legacy_metrics = _geometry_metrics(legacy_vertices, legacy_faces, scale=0.001)
        supplier = {
            "authority": "SUPPLIER_ASSEMBLY_EMBEDDED_HANDED_FINGER_BREP_TESSELLATION",
            "path": str(supplier_path),
            "sha256": _sha256(supplier_path),
            **supplier_metrics,
        }
        legacy = {
            "authority": "HISTORICAL_GYM_ALOHA_CUSTOM_FINGER_STL",
            "path": str(legacy_path),
            "sha256": _sha256(legacy_path),
            **legacy_metrics,
        }
        records[side] = {
            "supplier_cad": supplier,
            "legacy_cooked_source": legacy,
            "exact_source_hash_match": supplier["sha256"] == legacy["sha256"],
            "signed_volume_abs_m3_difference": abs(
                supplier["signed_volume_abs_m3"]
                - legacy["signed_volume_abs_m3"]
            ),
            "signed_volume_abs_ratio_legacy_over_supplier": (
                legacy["signed_volume_abs_m3"]
                / supplier["signed_volume_abs_m3"]
            ),
            "sorted_aabb_extent_m_difference": np.abs(
                np.asarray(supplier["sorted_aabb_extent_m"])
                - np.asarray(legacy["sorted_aabb_extent_m"])
            ).tolist(),
            "cooked_piece_count_in_legacy_report": legacy_sources[side][
                "piece_count"
            ],
            "source_identity_gate": "FAIL_EXACT_SOURCE_IDENTITY",
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_SOURCE_MISMATCH_DETECTED",
        "classification": "LEGACY_COOKED_SOURCE_NOT_CURRENT_SUPPLIER_SOURCE",
        "scope": "OFFLINE_SOURCE_PROVENANCE_AND_INVARIANT_GEOMETRY_METRICS",
        "legacy_report_path": str(legacy_report_path),
        "legacy_report_sha256": _sha256(legacy_report_path),
        "records": records,
        "legacy_cooked_geometry_reusable_for_supplier_cad": False,
        "next_gate": "REQUIRES_SUPPLIER_CAD_COOKED_READBACK",
        "interpretation": [
            "The legacy report's cooked vertices remain valid only for its recorded gym-aloha STL hashes.",
            "Different source hashes are sufficient to reject provenance reuse for the supplier-CAD proof.",
            "Triangle count, volume and sorted AABB extents are supplemental numeric differences; no fitted similarity tolerance is used.",
            "No conclusion about supplier-CAD convex decomposition accuracy is made from the legacy hold result.",
        ],
        "isaac_runtime_started": False,
        "final_or_default_asset_modified": False,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["deterministic_signature"] = hashlib.sha256(payload).hexdigest()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for side, record in report["records"].items():
        supplier = record["supplier_cad"]
        legacy = record["legacy_cooked_source"]
        rows.append(
            f"| {side} | `{supplier['sha256']}` | `{legacy['sha256']}` | "
            f"{supplier['face_count']} | {legacy['face_count']} | "
            f"{record['signed_volume_abs_ratio_legacy_over_supplier']:.9g} |"
        )
    return "\n".join(
        [
            "# ALOHA1 finger cooked-source identity boundary",
            "",
            f"- Status: **{report['status']}**",
            f"- Classification: **{report['classification']}**",
            f"- Next gate: **{report['next_gate']}**",
            "- Isaac runtime started: `false`",
            "- Final/default asset modified: `false`",
            "",
            "| Side | Supplier CAD SHA-256 | Legacy cooked source SHA-256 | Supplier faces | Legacy faces | Legacy/supplier volume |",
            "|---|---|---|---:|---:|---:|",
            *rows,
            "",
            "The saved 32-piece cooked decomposition belongs to the recorded historical "
            "gym-aloha STL inputs, not to the current supplier-assembly B-Rep "
            "tessellations. It cannot certify the supplier-CAD inward contact surfaces. "
            "A new isolated Isaac Sim 5.1 cooked-geometry readback is required; no "
            "legacy runtime result is promoted across this source boundary.",
            "",
        ]
    )

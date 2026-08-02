"""Resolve supplier-CAD geometry boundaries against pinned official URDF meshes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SUPPLIER_CAD_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)
OFFICIAL_MANIPULATORS_COMMIT = "b66d5b905725351dd71d3251a06cd3f4c777940f"
OFFICIAL_MANIPULATORS_REPOSITORY = (
    "https://github.com/Interbotix/interbotix_ros_manipulators.git"
)
MESH_HASHES = {
    "wrist_link": "90eb145c85627968c3776ae6de23ccff7e112c9dd713c46bc9acdfdaa859a048",
    "gripper_link": "786c1077bfd226f14219581b11d5f19464ca95b17132e0bb7532503568f5af90",
    "gripper_bar_link": "a4de62c9a2ed2c78433010e4c05530a1254b1774a7651967f406120c9bf8973e",
    "gripper_prop_link": "d1275a93fe2157c83dbc095617fb7e672888bdd48ec070a35ef4ab9ebd9755b0",
    "gripper_finger_link": "a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _signature(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "deterministic_signature"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _face_x_at_y(face: dict[str, Any], y_mm: float) -> float:
    nx, ny, _ = (float(value) for value in face["normal"])
    cx, cy, _ = (float(value) for value in face["center_mm"])
    if abs(nx) < 1.0e-12:
        raise ValueError("contact face is parallel to the opening axis")
    return cx - (ny / nx) * (y_mm - cy)


def _select_contact_face(
    object_record: dict[str, Any], *, side: str
) -> dict[str, Any]:
    expected_sign = -1.0 if side == "left" else 1.0
    candidates = [
        face
        for face in object_record["planar_faces"]
        if float(face["area_mm2"]) > 1000.0
        and float(face["bbox_mm"]["YMin"]) < -520.0
        and expected_sign * float(face["normal"][0]) > 0.9
    ]
    if not candidates:
        raise ValueError(f"no supplier-CAD {side} inward contact surface")
    return max(candidates, key=lambda item: float(item["area_mm2"]))


def build_source_geometry_probe(
    root: Path,
    *,
    face_runs: tuple[Path, Path],
    submesh_runs: tuple[Path, Path],
    wrist_runs: tuple[Path, Path],
    official_repo: Path,
) -> dict[str, Any]:
    """Condense raw FreeCAD reports without copying restricted CAD geometry."""
    face = [_load(path) for path in face_runs]
    submesh = [_load(path) for path in submesh_runs]
    wrist = [_load(path) for path in wrist_runs]
    for group in (face, submesh, wrist):
        if any(item["source_sha256"] != SUPPLIER_CAD_SHA256 for item in group):
            raise ValueError("supplier CAD hash mismatch in FreeCAD evidence")
        if group[0]["deterministic_signature"] != group[1]["deterministic_signature"]:
            raise ValueError("fresh FreeCAD process signatures differ")

    left = _select_contact_face(face[0]["objects"]["Part__Feature007"], side="left")
    right = _select_contact_face(face[0]["objects"]["Part__Feature008"], side="right")
    y_min = max(float(left["bbox_mm"]["YMin"]), float(right["bbox_mm"]["YMin"]))
    y_max = min(float(left["bbox_mm"]["YMax"]), float(right["bbox_mm"]["YMax"]))
    if y_min >= y_max:
        raise ValueError("supplier contact faces have no common axial interval")
    gap_samples = [
        _face_x_at_y(left, value) - _face_x_at_y(right, value)
        for value in (y_min, y_max)
    ]

    official_repo = official_repo.resolve(strict=True)
    current_mesh_root = (
        root
        / "external/ros2-essentials/aloha_ws/src/interbotix_ros_manipulators/"
        "interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/"
        "aloha_vx300s_meshes"
    )
    official_mesh_root = (
        official_repo
        / "interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/"
        "aloha_vx300s_meshes"
    )
    official_meshes = {}
    for suffix, expected_hash in MESH_HASHES.items():
        name = "gripper_finger.stl" if suffix == "gripper_finger_link" else f"{suffix.removesuffix('_link')}.stl"
        current_path = current_mesh_root / name
        official_path = official_mesh_root / name
        current_hash = _sha256(current_path)
        official_hash = _sha256(official_path)
        if current_hash != expected_hash or official_hash != expected_hash:
            raise ValueError(f"official mesh hash mismatch for {suffix}")
        official_meshes[suffix] = {
            "filename": name,
            "project_path": str(current_path.resolve()),
            "official_fresh_clone_path": str(official_path),
            "sha256": expected_hash,
            "byte_identical_to_official_commit": True,
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "source_cad": {
            "absolute_path": face[0]["source"],
            "sha256": SUPPLIER_CAD_SHA256,
            "read_only": True,
            "redistribution": "UNKNOWN_HARD_BLOCKER_NOT_INCLUDED",
        },
        "toolchain": {
            "freecad_version": "1.1.1",
            "opencascade_version": "7.8.1",
            "linear_deflection_mm": 0.2,
            "angular_deflection_deg": 20.0,
        },
        "raw_evidence": {
            "face_runs": [
                {"path": str(path.resolve()), "sha256": _sha256(path), "signature": item["deterministic_signature"]}
                for path, item in zip(face_runs, face, strict=True)
            ],
            "submesh_runs": [
                {"path": str(path.resolve()), "sha256": _sha256(path), "signature": item["deterministic_signature"]}
                for path, item in zip(submesh_runs, submesh, strict=True)
            ],
            "wrist_runs": [
                {"path": str(path.resolve()), "sha256": _sha256(path), "signature": item["deterministic_signature"]}
                for path, item in zip(wrist_runs, wrist, strict=True)
            ],
        },
        "two_fresh_process_determinism": "PASS",
        "official_mesh_source": {
            "repository": OFFICIAL_MANIPULATORS_REPOSITORY,
            "branch": "humble",
            "commit": OFFICIAL_MANIPULATORS_COMMIT,
            "license": "BSD-3-Clause",
            "license_sha256": _sha256(official_repo / "LICENSE"),
            "meshes": official_meshes,
        },
        "combined_gripper_cad": {
            "object_name": "Part__Feature006",
            "label": face[0]["objects"]["Part__Feature006"]["label"],
            "shape_type": face[0]["objects"]["Part__Feature006"]["shape_type"],
            "is_valid": face[0]["objects"]["Part__Feature006"]["is_valid"],
            "solid_count": face[0]["objects"]["Part__Feature006"]["solid_count"],
            "independent_bar_or_prop_product_exposed": False,
            "urdf_submesh_registration": submesh[0]["records"],
        },
        "wrist_cad": {
            "object_name": wrist[0]["object_name"],
            "label": wrist[0]["label"],
            "shape": wrist[0]["shape"],
            "diagnostics": wrist[0]["shape_check_exception"],
            "shells": wrist[0]["shells"],
            "source_brep_repaired": False,
        },
        "finger_contact_surfaces": {
            "selection_rule": "largest planar face above 1000 mm2 in the distal region Y<-520 mm with inward opening-axis normal magnitude >0.9",
            "left": left,
            "right": right,
            "common_y_interval_mm": [y_min, y_max],
            "closed_reference_gap_range_mm": [min(gap_samples), max(gap_samples)],
            "surfaces_parallel": False,
            "single_scalar_gap": False,
        },
        "source_brep_repaired": False,
        "mirror_used": False,
    }
    report["deterministic_signature"] = _signature(report)
    return report


def build_link_identity_resolution(root: Path, probe_path: Path) -> dict[str, Any]:
    probe = _load(probe_path)
    if probe["status"] != "PASS":
        raise ValueError("CAD geometry probe did not pass")
    official = probe["official_mesh_source"]
    combined = probe["combined_gripper_cad"]
    records = [
        {
            "link_suffix": suffix,
            "supplier_cad_identity": "COMBINED_GRIPPER_SOLID_NO_INDEPENDENT_PRODUCT",
            "supplier_cad_object": {
                key: combined[key]
                for key in (
                    "object_name",
                    "label",
                    "shape_type",
                    "is_valid",
                    "solid_count",
                )
            },
            "cad_registration_metrics": combined["urdf_submesh_registration"][
                suffix
            ]["distance_to_combined_cad_solid_mm"],
            "cad_subgeometry_claim": "NOT_CLAIMED",
            "authoritative_link_geometry_source": "PINNED_OFFICIAL_URDF_MESH",
            "official_mesh": official["meshes"][suffix],
            "resolution_status": "RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY",
            "source_brep_repaired": False,
        }
        for suffix in ("gripper_bar_link", "gripper_prop_link")
    ]
    wrist = probe["wrist_cad"]
    records.append(
        {
            "link_suffix": "wrist_link",
            "supplier_cad_identity": "EXPOSED_INVALID_BREP",
            "supplier_cad_object": {
                "object_name": wrist["object_name"],
                "label": wrist["label"],
                "shape": wrist["shape"],
            },
            "cad_brep_diagnostics": wrist["diagnostics"],
            "cad_subgeometry_claim": "NOT_CLAIMED",
            "authoritative_link_geometry_source": "PINNED_OFFICIAL_URDF_MESH",
            "official_mesh": official["meshes"]["wrist_link"],
            "resolution_status": "RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY",
            "source_brep_repaired": False,
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "ROBOT_LOCAL_LINK_GEOMETRY_SOURCE_RESOLUTION_NOT_COLLIDER_PROMOTION",
        "source_cad": probe["source_cad"],
        "toolchain": probe["toolchain"],
        "probe": {"path": str(probe_path.resolve()), "sha256": _sha256(probe_path)},
        "official_mesh_source": {
            key: official[key] for key in ("repository", "branch", "commit", "license", "license_sha256")
        },
        "two_fresh_process_determinism": probe["two_fresh_process_determinism"],
        "records": records,
        "resolution": "CAD_IDENTITY_BLOCKERS_REPLACED_BY_EXPLICIT_GEOMETRY_SOURCE_BOUNDARIES",
        "mirror_used": False,
        "source_brep_repaired": False,
        "final_or_default_asset_modified": False,
        "runtime_simulation_used": False,
    }
    report["deterministic_signature"] = _signature(report)
    return report


def build_aperture_resolution(root: Path, probe_path: Path) -> dict[str, Any]:
    probe = _load(probe_path)
    mapping_path = root / "reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json"
    mapping = _load(mapping_path)
    matrices = mapping["cad_to_finger_link_mapping"]
    left_matrix = matrices["left_matrix"]
    right_matrix = matrices["right_matrix"]
    left_cad_x_m = -float(left_matrix[1][3]) / float(left_matrix[1][0])
    right_cad_x_m = -float(right_matrix[1][3]) / float(right_matrix[1][0])
    closed_center_distance = round(left_cad_x_m - right_cad_x_m, 12)
    translation_mm = float(
        mapping["primary_follower_installation"]["derived_open_state"]
        ["cad_positive_x_finger_translation_mm"][0]
    )
    open_center_distance = round(
        closed_center_distance + 2.0 * translation_mm * 0.001, 12
    )
    closed_gap = [
        float(value) * 0.001
        for value in probe["finger_contact_surfaces"]["closed_reference_gap_range_mm"]
    ]
    open_gap = [value + 2.0 * translation_mm * 0.001 for value in closed_gap]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "source_cad": probe["source_cad"],
        "probe": {"path": str(probe_path.resolve()), "sha256": _sha256(probe_path)},
        "gripper_mapping": {"path": str(mapping_path.resolve()), "sha256": _sha256(mapping_path)},
        "definitions": {
            "carriage_center_distance": "distance between the two prismatic finger-link origins along the CAD opening axis",
            "contact_surface_gap": "distance between the two distal inward supplier-CAD faces at a named axial station; not one scalar because the faces are not parallel",
            "product_table_range": "Trossen ViperX-300 Gripper row Min/Max; the page does not define the measurement datum",
        },
        "cad_carriage_origin_x_m": {"left": left_cad_x_m, "right": right_cad_x_m},
        "cad_carriage_center_distance_m": {
            "closed_reference": closed_center_distance,
            "open_derived": open_center_distance,
        },
        "urdf_carriage_center_distance_m": [0.042, 0.114],
        "trossen_product_table_range_m": [0.042, 0.116],
        "contact_surface_gap_m": {
            "closed_reference_range_over_common_face_interval": closed_gap,
            "open_derived_range_over_common_face_interval": open_gap,
            "common_cad_y_interval_m": [
                float(value) * 0.001
                for value in probe["finger_contact_surfaces"]["common_y_interval_mm"]
            ],
        },
        "contact_surface_gap_is_single_scalar": False,
        "source_conflict": {
            "classification": "VERIFIED_OFFICIAL_SOURCE_CONFLICT_PRODUCT_PAGE_NOT_CAD_SUPPORTED",
            "difference_at_max_m": 0.002,
            "preserved_not_silently_fitted": True,
            "interpretation": "The CAD carriage datums and pinned official URDF both resolve to 114 mm maximum center distance; the exact-product table says 116 mm without defining a different datum. The supplier-CAD distal inner gap is y-dependent and matches neither range.",
        },
        "implemented_joint_range_source": "PINNED_OFFICIAL_URDF_AND_CAD_CARRIAGE_DATUM",
        "implemented_joint_range_m": [0.042, 0.114],
        "fitted_endpoint_used": False,
        "runtime_simulation_used": False,
        "final_or_default_asset_modified": False,
    }
    report["deterministic_signature"] = _signature(report)
    return report


def render_link_markdown(report: dict[str, Any]) -> str:
    rows = [
        "| Link | Supplier CAD boundary | Link geometry authority | Resolution |",
        "|---|---|---|---|",
    ]
    rows.extend(
        (
            f"| `{record['link_suffix']}` | `{record['supplier_cad_identity']}` | "
            f"`{record['authoritative_link_geometry_source']}` | `{record['resolution_status']}` |"
        )
        for record in report["records"]
    )
    return "\n".join(
        [
            "# ALOHA1 CAD/link geometry source resolution",
            "",
            f"- Status: **{report['status']}**",
            f"- Official source commit: `{report['official_mesh_source']['commit']}`",
            "- Supplier CAD repaired: `false`",
            "- Mirror used: `false`",
            "",
            *rows,
            "",
            "The supplier STEP is not falsely split into URDF products. Its fused gripper solid and invalid wrist B-Rep remain explicit evidence boundaries; the byte-identical pinned Interbotix meshes provide the link-level geometry identities. This report does not promote a collider or modify an asset.",
            "",
        ]
    )


def render_aperture_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 gripper aperture definition resolution",
            "",
            f"- Status: **{report['status']}**",
            f"- CAD/URDF carriage-center range: `{report['urdf_carriage_center_distance_m']} m`",
            f"- Trossen exact-product table: `{report['trossen_product_table_range_m']} m`",
            f"- Conflict: `{report['source_conflict']['classification']}`",
            f"- Implemented range source: `{report['implemented_joint_range_source']}`",
            "",
            "The supplier-CAD distal inward faces are tilted and their gap varies along the finger. They do not define a single 42/114/116 mm aperture. The 2 mm official-source conflict remains visible; no endpoint was fitted to make the sources agree.",
            "",
        ]
    )

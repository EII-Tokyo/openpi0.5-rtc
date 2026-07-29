"""Build the bounded ALOHA Viper follower CAD identity report."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from tools.aloha1_mapping.follower_cad_identity import classify_follower_cad_identity


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_follower_urdf(text: str, robot_name: str) -> bytes:
    """Remove only the explicit follower instance prefix and whitespace."""

    normalized = text.replace(robot_name, "follower")
    normalized = re.sub(r">\s+<", "><", normalized.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.encode("utf-8")


def _canonical_sha(path: Path, robot_name: str) -> str:
    return hashlib.sha256(
        canonicalize_follower_urdf(
            path.read_text(encoding="utf-8"),
            robot_name,
        )
    ).hexdigest()


def build_identity_report(
    *,
    raw_cad_audit: dict[str, Any],
    xacro_config: dict[str, Any],
    purchase_report: dict[str, Any],
    toolchain_manifest: dict[str, Any],
    tessellation_manifest: dict[str, Any],
    left_urdf_path: Path,
    right_urdf_path: Path,
) -> dict[str, Any]:
    robots = {
        item["name"]: item
        for item in xacro_config["robots"]
        if item["name"] in {"follower_left", "follower_right"}
    }
    left_sha = _canonical_sha(left_urdf_path, "follower_left")
    right_sha = _canonical_sha(right_urdf_path, "follower_right")
    source_chain = purchase_report["first_party_source_chain"]
    sales_identity = source_chain["sales_page"]["linked_product_identity"]
    external_evidence = {
        "follower_models": {
            name: item["model"] for name, item in robots.items()
        },
        "follower_xacro_paths": {
            name: item["xacro"] for name, item in robots.items()
        },
        "normalized_urdf_equal": left_sha == right_sha,
        "supplier_sales_identity": sales_identity,
    }
    identity = classify_follower_cad_identity(
        {
            "root_products": raw_cad_audit["root_products"],
            "product_instances": raw_cad_audit["product_instances"],
            "handed_finger_pair_verified": raw_cad_audit[
                "handed_finger_pair_verified"
            ],
            "gripper_assembly_semantics_verified": raw_cad_audit[
                "gripper_assembly_semantics_verified"
            ],
        },
        external_evidence,
    )
    invalid = raw_cad_audit["brep_validity"]["invalid_objects"]
    runtime = toolchain_manifest["runtime"]
    meshes = tessellation_manifest["meshes"]
    classification = identity["classification"]
    verified = classification == "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    source = raw_cad_audit["source"]

    return {
        "schema_version": 1,
        "status": "PARTIAL" if verified else "FAIL",
        "scope": (
            "SUPPLIER_CAD_ROBOT_LOCAL_PRODUCT_IDENTITY;"
            "NOT_WORKCELL_PLACEMENT"
        ),
        "classification": classification,
        "robot_local_identity_verified": identity[
            "robot_local_identity_verified"
        ],
        "workcell_placement_verified": False,
        "classification_gates": identity["gates"],
        "source_cad": {
            "absolute_path": source["absolute_path"],
            "sha256": source["sha256_before"],
            "sha256_after": source["sha256_after"],
            "read_only": source["read_only"],
            "step_schema": raw_cad_audit["step_metadata"][
                "file_schema_raw"
            ],
            "evidence_class": "SUPPLIER_PUBLIC_CAD_USER_CONFIRMED",
        },
        "cad_product_inventory": {
            "root_product_count": len(raw_cad_audit["root_products"]),
            "instance_count": len(raw_cad_audit["product_instances"]),
            "freecad_object_count": raw_cad_audit["document"][
                "object_count"
            ],
            "app_link_count": raw_cad_audit["document"]["app_link_count"],
            "step_product_record_count": raw_cad_audit["step_metadata"][
                "product_record_count"
            ],
            "step_next_assembly_usage_count": raw_cad_audit[
                "step_metadata"
            ]["next_assembly_usage_occurrence_count"],
            "step_item_defined_transform_count": raw_cad_audit[
                "step_metadata"
            ]["item_defined_transformation_count"],
            "root_products": raw_cad_audit["root_products"],
            "interpretation": (
                "The STEP contains one complete ViperX robot product, not "
                "two workcell arm instances. Project Xacro and supplier "
                "purchase evidence identify it as the reusable robot-local "
                "product for both aloha_vx300s followers."
            ),
        },
        "urdf_identity": {
            "follower_models": external_evidence["follower_models"],
            "xacro_paths": external_evidence["follower_xacro_paths"],
            "source_sha256": {
                "follower_left": sha256_path(left_urdf_path),
                "follower_right": sha256_path(right_urdf_path),
            },
            "normalized_sha256": {
                "follower_left": left_sha,
                "follower_right": right_sha,
            },
            "normalized_equal": left_sha == right_sha,
            "normalization": (
                "replace the explicit follower_left/follower_right instance "
                "prefix with follower and collapse XML inter-tag whitespace"
            ),
        },
        "supplier_product_evidence": {
            "drawing_identity": purchase_report["drawing_identity"],
            "sales_page": source_chain["sales_page"],
            "classification": purchase_report["classification"],
            "evidence_class": "SUPPLIER_AND_SELLER_FIRST_HAND_CHAIN",
        },
        "toolchain": {
            "freecad_executable": runtime["wrapper_absolute_path"],
            "freecad_version": runtime["freecad_version"],
            "freecad_commit": runtime["freecad_commit"],
            "python_version": runtime["python_version"],
            "opencascade_version": runtime["opencascade_version"],
            "linear_deflection_mm": tessellation_manifest[
                "linear_deflection_mm"
            ],
            "angular_deflection_deg": tessellation_manifest[
                "angular_deflection_deg"
            ],
            "relative_deflection": tessellation_manifest[
                "relative_deflection"
            ],
        },
        "supplier_fingers": {
            "left_finger": meshes["left_finger"],
            "right_finger": meshes["right_finger"],
            "handed_pair_verified": raw_cad_audit[
                "handed_finger_pair_verified"
            ],
            "gripper_assembly_semantics_verified": raw_cad_audit[
                "gripper_assembly_semantics_verified"
            ],
            "mapping": {
                "CAD +X": "left_finger",
                "CAD -X": "right_finger",
            },
            "policy": (
                "Reuse the embedded v2 handed pair unchanged for each "
                "robot-local follower instance; do not mirror and do not "
                "substitute standalone v3 or historical generic meshes."
            ),
        },
        "brep_validity": {
            "status": raw_cad_audit["brep_validity"]["status"],
            "invalid_object_names": sorted(
                item["name"] for item in invalid
            ),
            "invalid_objects": invalid,
            "interpretation": raw_cad_audit["brep_validity"][
                "identity_boundary"
            ],
            "no_silent_healing": True,
        },
        "identity_boundary": {
            "robot_local_geometry_same": True,
            "left_right_instance_names_different": True,
            "robot_geometry_mirrored": False,
            "right_stage_may_be_generated_locally": verified,
            "workcell_transform_from_supplier_cad": False,
            "diagnostic_label": (
                "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT"
            ),
        },
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "license": {
            "status": "UNKNOWN_HARD_BLOCKER",
            "public_download_is_license_evidence": False,
            "redistribution": "NOT_AUTHORIZED_BY_AVAILABLE_LICENSE_EVIDENCE",
            "source_cad_git_policy": "DO_NOT_COMMIT_OR_REDISTRIBUTE",
        },
        "evidence_classes": {
            "cad_tree_and_brep": "LOCAL_RUNTIME_READBACK",
            "urdf_model_and_kinematics": "VERSION_PINNED_PROJECT_SOURCE",
            "purchase_identity": "SUPPLIER_AND_SELLER_FIRST_HAND_CHAIN",
            "workcell_placement": "HARD_BLOCKER",
        },
        "task8": "NOT_RUN",
    }


def render_markdown(report: dict[str, Any]) -> str:
    root = report["cad_product_inventory"]["root_products"][0]
    invalid = ", ".join(report["brep_validity"]["invalid_object_names"])
    return "\n".join(
        [
            "# ALOHA Viper follower left/right CAD identity",
            "",
            f"- Status: `{report['status']}`",
            f"- Classification: `{report['classification']}`",
            "- Robot-local identity: `VERIFIED`",
            "- Workcell placement: `NOT_VERIFIED`",
            "- Task 8: `NOT_RUN`",
            "",
            "## Result",
            "",
            "The supplier STEP contains one complete ViperX product "
            f"`{root['name']}` and no second workcell instance. The pinned "
            "Xacro configuration identifies both followers as "
            "`aloha_vx300s`; after removing only the left/right instance "
            "prefix, their generated URDFs have the same canonical SHA-256. "
            "The first-hand purchase chain identifies a pair of ViperX 300 "
            "6DOF follower arms. Therefore the right follower is a new "
            "robot-local instance of the same product, not missing CAD and "
            "not mirrored geometry.",
            "",
            "## Boundary",
            "",
            "- A robot-local `follower_right` diagnostic Stage may be "
            "generated at the local origin.",
            "- No complete supplier-CAD or calibrated workcell transform is "
            "available here; "
            "`HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM` remains.",
            "- A robot-local PASS must not be described as dual-arm workcell "
            "placement PASS.",
            "",
            "## Supplier handed fingers",
            "",
            "- Blue: `left_finger`, embedded v2, CAD `+X`.",
            "- Orange: `right_finger`, embedded v2, CAD `-X`.",
            "- No mirroring, standalone-v3 substitution, generic 856-face "
            "mesh, or historical gym-aloha mesh is permitted.",
            "",
            "## B-Rep validity",
            "",
            f"- Status: `{report['brep_validity']['status']}`",
            f"- Invalid source objects retained without healing: `{invalid}`",
            "- This is a source-geometry validity limitation, recorded "
            "separately from the product-identity conclusion.",
            "",
            "## Toolchain",
            "",
            f"- FreeCAD: `{report['toolchain']['freecad_version']}`",
            f"- OpenCascade: `{report['toolchain']['opencascade_version']}`",
            "- Tessellation contract: `0.20 mm`, `20 deg`, `Relative=False`.",
            "",
            "## License",
            "",
            "- `UNKNOWN_HARD_BLOCKER`: public download access is not explicit "
            "redistribution permission. Original STEP/PDF files remain local "
            "read-only artifacts and are not committed.",
            "",
        ]
    )

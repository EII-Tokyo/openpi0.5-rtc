"""Identify the purchased ALOHA arm model from its engineering drawing."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

EXPECTED_PDF_SHA256 = (
    "b7cca2d069254e29b6b9081304ba405c4384580b47df5b2f82c012d5c7b7357e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_text(path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _source(audit: Mapping[str, Any], label: str) -> dict[str, Any]:
    return next(
        source
        for source in audit["sources"]
        if source["source_label"] == label
    )


def _object_by_label_prefix(
    source: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    matches = [
        obj for obj in source["objects"] if obj["label"].startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix!r} object, found {len(matches)}")
    return matches[0]


def _finger_objects(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in source["objects"]
        if obj["type_id"] == "Part::Feature"
        and obj["label"].startswith("Aloha VX Fingers 2024-4-21")
    ]


def _xyz_lengths(obj: Mapping[str, Any]) -> dict[str, float]:
    bbox = obj["shape"]["bound_box_mm"]
    return {
        axis.lower(): float(bbox[f"{axis}Length"]) for axis in "XYZ"
    }


def build_model_identification_report(
    drawing_pdf: Path,
    public_cad_audit_path: Path,
    widow_audit_path: Path,
) -> dict[str, Any]:
    drawing_pdf = drawing_pdf.resolve(strict=True)
    public_audit = json.loads(
        public_cad_audit_path.read_text(encoding="utf-8")
    )
    widow_audit = json.loads(widow_audit_path.read_text(encoding="utf-8"))
    text = _pdf_text(drawing_pdf)
    drawing_hash = _sha256(drawing_pdf)
    reference_root = drawing_pdf.parent
    sales_page_snapshot = (
        reference_root / "sales_page/aloha-viperx.html"
    ).resolve(strict=True)
    sales_catalog = (
        reference_root / "catalog/Aloha ViperX Sales Sheet.pdf"
    ).resolve(strict=True)
    sales_page_hash = _sha256(sales_page_snapshot)
    sales_catalog_hash = _sha256(sales_catalog)

    simple = _source(public_audit, "simple_viper")
    exact_finger = _source(public_audit, "exact_vx_finger")
    widow = _source(widow_audit, "widow_with_gripper")
    simple_base = _object_by_label_prefix(simple, "Simple VX Base")
    widow_base = _object_by_label_prefix(widow, "Simple WX Base")
    simple_root = next(
        obj for obj in simple["objects"] if obj["name"] in simple["root_objects"]
    )
    widow_root = next(
        obj for obj in widow["objects"] if obj["name"] in widow["root_objects"]
    )
    simple_group = _object_by_label_prefix(
        simple,
        "Aloha VX Fingers 2024-4-21 v002",
    )
    widow_group = _object_by_label_prefix(
        widow,
        "Aloha VX Fingers 2024-4-21 v002",
    )
    simple_fingers = sorted(
        _finger_objects(simple),
        key=lambda obj: obj["label"],
    )
    widow_fingers = sorted(
        _finger_objects(widow),
        key=lambda obj: obj["label"],
    )
    simple_finger_volumes = [
        float(obj["shape"]["volume_mm3"]) for obj in simple_fingers
    ]
    widow_finger_volumes = [
        float(obj["shape"]["volume_mm3"]) for obj in widow_fingers
    ]
    simple_pair_dimensions = _xyz_lengths(simple_group)
    widow_pair_dimensions = _xyz_lengths(widow_group)
    drawing_base = {"x": 204.0, "y": 299.46}
    simple_base_dimensions = _xyz_lengths(simple_base)
    widow_base_dimensions = _xyz_lengths(widow_base)
    simple_errors = {
        axis: abs(simple_base_dimensions[axis] - drawing_base[axis])
        for axis in drawing_base
    }
    widow_errors = {
        axis: abs(widow_base_dimensions[axis] - drawing_base[axis])
        for axis in drawing_base
    }
    finger_dimension = float(
        exact_finger["objects"][0]["shape"]["bound_box_mm"]["ZLength"]
    )
    shared_gripper = {
        "simple_viper_finger_labels": [
            obj["label"] for obj in simple_fingers
        ],
        "widow_finger_labels": [obj["label"] for obj in widow_fingers],
        "same_finger_labels": (
            [obj["label"] for obj in simple_fingers]
            == [obj["label"] for obj in widow_fingers]
        ),
        "same_finger_topology": (
            [obj["shape"]["topology_counts"] for obj in simple_fingers]
            == [obj["shape"]["topology_counts"] for obj in widow_fingers]
        ),
        "same_finger_volumes": (
            max(
                abs(left - right)
                for left, right in zip(
                    simple_finger_volumes,
                    widow_finger_volumes,
                    strict=True,
                )
            )
            < 1.0e-6
        ),
        "same_finger_pair_dimensions": (
            max(
                abs(
                    simple_pair_dimensions[axis]
                    - widow_pair_dimensions[axis]
                )
                for axis in "xyz"
            )
            < 1.0e-6
        ),
        "conclusion": (
            "gripper/finger visual similarity is expected and does not "
            "identify the arm model"
        ),
    }
    gates = {
        "drawing_hash_frozen": drawing_hash == EXPECTED_PDF_SHA256,
        "sales_page_snapshot_frozen": (
            sales_page_hash
            == "6b1e7c05b9fd1abab49b232002c08c54d05440f9fbebf764222bf53a7bf83a06"
        ),
        "sales_catalog_frozen": (
            sales_catalog_hash
            == "d06346070022f300b8fb73176fbeeaf4eb096300238a1142cde5a2399c3f3888"
        ),
        "drawing_names_viperx": "Aloha ViperX 6DOF" in text,
        "drawing_names_vx300s": "Aloha VX300S" in text,
        "drawing_names_follower": "Follower Robot Arm" in text,
        "viper_model_family_is_vx": simple_root["label"].startswith(
            "Dummy Aloha VX"
        ),
        "widow_model_family_is_wx": widow_root["label"].startswith(
            "Dummy Aloha WX"
        ),
        "viper_base_x_matches": simple_errors["x"] < 0.01,
        "viper_base_y_matches": simple_errors["y"] < 0.01,
        "widow_base_does_not_match": (
            widow_errors["x"] > 50.0 and widow_errors["y"] > 65.0
        ),
        "shared_gripper_explains_visual_similarity": all(
            shared_gripper[name]
            for name in (
                "same_finger_labels",
                "same_finger_topology",
                "same_finger_volumes",
                "same_finger_pair_dimensions",
            )
        ),
        "standalone_finger_matches_81_71_callout": (
            abs(finger_dimension - 81.71) < 0.0025
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "classification": "SIMPLE_ALOHA_VIPER_2024_5_13_STEP",
        "classification_confidence": "DIRECT_MODEL_AND_DIMENSION_MATCH",
        "drawing_identity": {
            "project": "Aloha ViperX 6DOF",
            "title": "Aloha VX300S Follower Robot Arm",
            "drawn_by": "AIDAN WEDDLE",
            "drawing_date": "2024-05-13",
            "revision": "A",
            "source_path": str(drawing_pdf),
            "source_sha256": drawing_hash,
            "google_drive_file_id": "11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU",
            "evidence_methods": [
                "PDF text-layer readback",
                "visual review of rendered drawing",
                "FreeCAD AP214 numeric comparison",
            ],
        },
        "first_party_source_chain": {
            "evidence_policy": (
                "The user identifies these linked seller/Trossen materials as "
                "official first-hand purchase documentation. The report keeps "
                "the publisher role explicit instead of silently treating the "
                "seller page as the CAD author."
            ),
            "sales_page": {
                "url": "https://idminer.com.tw/product/aloha-viperx/",
                "publisher": "採智科技股份有限公司 / IDMiner",
                "role": "PURCHASE_PRODUCT_PAGE_USER_CONFIRMED_FIRST_HAND",
                "accessed_date": "2026-07-29",
                "local_snapshot_path": str(sales_page_snapshot),
                "local_snapshot_sha256": sales_page_hash,
                "linked_product_identity": (
                    "Aloha ViperX follower set; pair of ViperX 300 6DOF arms"
                ),
                "linked_specs": {
                    "reach_mm": 750,
                    "span_mm": 1500,
                    "degrees_of_freedom": 6,
                    "payload_g": 750,
                    "servos": "6X XM540-W270-T | 2X XM430-W350-T",
                    "power": "12V, 10A",
                    "aloha_gripper": True,
                },
            },
            "sales_catalog": {
                "url": (
                    "https://drive.google.com/file/d/"
                    "11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh/view?usp=sharing"
                ),
                "google_drive_file_id": (
                    "11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh"
                ),
                "role": "TROSSEN_VIPERX_FOLLOWER_SALES_SHEET",
                "local_path": str(sales_catalog),
                "sha256": sales_catalog_hash,
            },
            "technical_drawing": {
                "url": (
                    "https://drive.google.com/file/d/"
                    "11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU/view?usp=sharing"
                ),
                "google_drive_file_id": (
                    "11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU"
                ),
                "role": "ALOHA_VX300S_FOLLOWER_TECHNICAL_DRAWING",
                "local_path": str(drawing_pdf),
                "sha256": drawing_hash,
            },
            "public_3d_cad": {
                "url": (
                    "https://drive.google.com/drive/folders/"
                    "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf"
                ),
                "google_drive_folder_id": (
                    "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf"
                ),
                "role": "ALOHA_KITS_PUBLIC_3D_CAD",
                "local_manifest": str(
                    (
                        public_cad_audit_path.parent
                        / "aloha_public_cad_source_manifest.json"
                    ).resolve()
                ),
            },
            "trossen_manual": {
                "url": "https://docs.trossenrobotics.com/aloha_docs/",
                "publisher": "Trossen Robotics",
                "role": "ALOHA_KITS_ONLINE_MANUAL",
            },
        },
        "drawing_dimensions_mm": {
            "base_x": drawing_base["x"],
            "base_y": drawing_base["y"],
            "finger_callout": 81.71,
            "other_visible_callouts": [
                128.47,
                78.7,
                190.0,
                96.27,
                85.24,
                59.55,
                197.0,
                103.0,
                73.25,
                73.29,
                228.1,
                300.0,
                305.85,
                126.75,
                67.96,
            ],
        },
        "candidate_comparison": {
            "simple_aloha_viper": {
                "source_path": simple["path"],
                "source_sha256": simple["sha256"],
                "root_label": simple_root["label"],
                "cad_model_family": "VX",
                "base_object": simple_base["name"],
                "base_label": simple_base["label"],
                "base_dimensions_mm": simple_base_dimensions,
                "drawing_base_absolute_error_mm": simple_errors,
                "result": "MATCH",
            },
            "aloha_widow_with_gripper": {
                "source_path": widow["path"],
                "source_sha256": widow["sha256"],
                "root_label": widow_root["label"],
                "cad_model_family": "WX",
                "base_object": widow_base["name"],
                "base_label": widow_base["label"],
                "base_dimensions_mm": widow_base_dimensions,
                "drawing_base_absolute_error_mm": widow_errors,
                "result": "NOT_THE_PURCHASED_FOLLOWER_ARM",
            },
        },
        "shared_gripper_explanation": shared_gripper,
        "finger_dimension_cross_check": {
            "drawing_callout_mm": 81.71,
            "standalone_vx_finger_step": exact_finger["path"],
            "standalone_vx_finger_sha256": exact_finger["sha256"],
            "standalone_vx_finger_bbox_axis": "Z",
            "standalone_vx_finger_bbox_dimension_mm": finger_dimension,
            "absolute_error_mm": abs(finger_dimension - 81.71),
            "role": (
                "supports finger-family identity but does not replace assembly "
                "installation transforms"
            ),
        },
        "asset_selection_policy": {
            "follower_arm_geometry": "Simple Aloha Viper 2024-5-13.step",
            "follower_model": "aloha_vx300s",
            "leader_arm_geometry": (
                "Aloha Widow with Gripper 2024-5-13.step is WX/Widow and "
                "must not replace the VX300S follower"
            ),
            "gripper_installation": (
                "use Simple Viper as the follower-primary assembly; use Widow "
                "only as a cross-check because both embed the same VX finger "
                "pair"
            ),
            "standalone_finger": (
                "3D-A1 VX Finger matches the 81.71 mm drawing callout but its "
                "revision must be mounting-feature aligned before replacing "
                "the embedded assembly geometry"
            ),
        },
        "gates": gates,
    }


def write_model_identification_reports(
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    viper = report["candidate_comparison"]["simple_aloha_viper"]
    widow = report["candidate_comparison"]["aloha_widow_with_gripper"]
    lines = [
        "# Purchased ALOHA Model Identification",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        f"- Confidence: `{report['classification_confidence']}`",
        "",
        (
            "The supplied engineering drawing directly names `Aloha ViperX "
            "6DOF` and `Aloha VX300S Follower Robot Arm`. Its 204 x 299.46 mm "
            "base matches the Simple Viper AP214 geometry."
        ),
        "",
        "## First-hand source chain",
        "",
        "- Sales/product page: `https://idminer.com.tw/product/aloha-viperx/`",
        (
            "- ViperX sales sheet: `https://drive.google.com/file/d/"
            "11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh/view?usp=sharing`"
        ),
        (
            "- VX300S technical drawing: `https://drive.google.com/file/d/"
            "11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU/view?usp=sharing`"
        ),
        (
            "- ALOHA 3D CAD folder: "
            "`https://drive.google.com/drive/folders/"
            "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf`"
        ),
        (
            "- Trossen ALOHA manual: "
            "`https://docs.trossenrobotics.com/aloha_docs/`"
        ),
        "",
        "| Candidate | CAD family | CAD base X x Y | Drawing error X x Y | Result |",
        "|---|---|---:|---:|---|",
        (
            "| Simple Aloha Viper | VX | "
            f"{viper['base_dimensions_mm']['x']:.6f} x "
            f"{viper['base_dimensions_mm']['y']:.6f} mm | "
            f"{viper['drawing_base_absolute_error_mm']['x']:.6f} x "
            f"{viper['drawing_base_absolute_error_mm']['y']:.6f} mm | MATCH |"
        ),
        (
            "| Aloha Widow with Gripper | WX | "
            f"{widow['base_dimensions_mm']['x']:.6f} x "
            f"{widow['base_dimensions_mm']['y']:.6f} mm | "
            f"{widow['drawing_base_absolute_error_mm']['x']:.6f} x "
            f"{widow['drawing_base_absolute_error_mm']['y']:.6f} mm | "
            "NOT_THE_PURCHASED_FOLLOWER_ARM |"
        ),
        "",
        (
            "The two STEP files look similar at the gripper because both embed "
            "the same `Aloha VX Fingers 2024-4-21` pair with equal labels, "
            "topology, volumes, and pair bounds. That shared end effector does "
            "not make the WX/Widow arm a VX300S follower."
        ),
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

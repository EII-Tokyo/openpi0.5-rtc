#!/usr/bin/env python3
"""Build the Isaac Sim 5.1 ALOHA Grasp Editor semantics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/"
    "frame_contract_correction"
)
CONTACT_RUN = (
    ARTIFACT_ROOT
    / "native_grasp_editor_variant_b_fully_closed_contact"
    / "grasp_editor_variant_b_gui_report.json"
)
NO_CONTACT_RUN = (
    ARTIFACT_ROOT
    / "native_grasp_editor_variant_b_fully_closed_no_contact"
    / "grasp_editor_variant_b_gui_report.json"
)
EXTERNAL_SKIP_SIM_RUN = (
    ARTIFACT_ROOT
    / "external_contact_skip_sim_run03_cross_axis"
    / "grasp_editor_variant_b_gui_report.json"
)
EXTERNAL_SCREENSHOT_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_editor_external_skip_sim_screenshot_review.json"
)
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "table_support_alignment/1.0/"
    "aloha1_table_support_aligned_workcell.usda"
)
EXPECTED_STAGE_SHA256 = (
    "2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c"
)
GRASP_TESTER_SOURCE = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
    "isaacsim.robot_setup.grasp_editor/isaacsim/robot_setup/"
    "grasp_editor/grasp_tester.py"
)
UI_BUILDER_SOURCE = GRASP_TESTER_SOURCE.with_name("ui_builder.py")
PHYSX_SCHEMA = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache/"
    "omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "plugins/PhysxSchema/resources/schema.usda"
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_editor_semantics_audit.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
V2_RUN = (
    ROOT
    / ".codex/artifacts/"
    "20260730-aloha1-official-gripper-unattended/stage4/"
    "grasp_editor_passing_coupling_run00/"
    "grasp_editor_variant_b_gui_report.json"
)
V2_COUPLING_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha1_gripper_coupling_ab.json"
)
V2_SCREENSHOT_REVIEW = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_editor_external_skip_sim_screenshot_review_v2.json"
)
V2_OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_editor_semantics_audit_v2.json"
)
V2_OUTPUT_MD = V2_OUTPUT_JSON.with_suffix(".md")
OFFICIAL_TUTORIAL = (
    "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/"
    "robot_simulation/grasp_editor.html"
)
MIMIC_TOLERANCE_M = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_declared_mimic_properties(schema_text: str) -> list[str]:
    match = re.search(
        r'class "PhysxMimicJointAPI".*?\n}\n\n//\n// vehicles',
        schema_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("PhysxMimicJointAPI block missing from local schema")
    block = match.group(0)
    names = re.findall(
        r"^\s*(?:rel|float|uniform token)\s+([A-Za-z][A-Za-z0-9]*)\b",
        block,
        flags=re.MULTILINE,
    )
    return sorted(set(names))


def _screenshot_records(
    report: dict[str, Any],
    *,
    review_status: str,
    review_note: str,
) -> list[dict[str, Any]]:
    records = []
    for item in report["result"]["screenshots"]:
        path = Path(item["path"]).resolve()
        records.append(
            {
                **item,
                "path": str(path),
                "sha256_readback": _sha256(path),
                "visual_model_review_status": review_status,
                "visual_model_review_note": review_note,
            }
        )
    return records


def _classify_mimic(
    contact_residual_m: float,
    no_contact_residual_m: float,
) -> dict[str, Any]:
    amplification = contact_residual_m / no_contact_residual_m
    if (
        contact_residual_m > MIMIC_TOLERANCE_M
        and no_contact_residual_m > MIMIC_TOLERANCE_M
        and amplification >= 2.0
    ):
        classification = (
            "OBJECT_CONTACT_AMPLIFIES_PERSISTENT_MIMIC_ERROR"
        )
        status = "FAIL"
    else:
        classification = "INCONCLUSIVE"
        status = "PARTIAL"
    return {
        "status": status,
        "classification": classification,
        "contact_residual_m": contact_residual_m,
        "no_contact_residual_m": no_contact_residual_m,
        "contact_minus_no_contact_m": (
            contact_residual_m - no_contact_residual_m
        ),
        "amplification_ratio": amplification,
        "tolerance_m": MIMIC_TOLERANCE_M,
    }


def build_report() -> dict[str, Any]:
    stage_sha256 = _sha256(STAGE)
    if stage_sha256 != EXPECTED_STAGE_SHA256:
        raise RuntimeError(
            f"approved Stage hash changed: {stage_sha256}"
        )
    contact = _load_json(CONTACT_RUN)
    no_contact = _load_json(NO_CONTACT_RUN)
    external = _load_json(EXTERNAL_SKIP_SIM_RUN)
    for label, run in (
        ("contact", contact),
        ("no_contact", no_contact),
        ("external_skip_sim", external),
    ):
        actual = run["inputs"]["stage"]["sha256"]
        if actual != EXPECTED_STAGE_SHA256:
            raise RuntimeError(f"{label} run used wrong Stage: {actual}")

    contact_result = contact["result"]
    no_contact_result = no_contact["result"]
    contact_residual = float(
        contact_result["joint_readback"]["mimic_error_abs_m"]
    )
    no_contact_residual = float(
        no_contact_result["joint_readback"]["mimic_error_abs_m"]
    )
    schema_text = PHYSX_SCHEMA.read_text(encoding="utf-8")
    declared_properties = _extract_declared_mimic_properties(schema_text)
    contact_summary = contact_result["contacts"]["summary"]
    no_contact_summary = no_contact_result["contacts"]["summary"]
    external_result = external["result"]
    external_gate = external_result["gate"]
    external_summary = external_result["contacts"]["summary"]
    external_screenshot_review = _load_json(EXTERNAL_SCREENSHOT_REVIEW)

    return {
        "schema_version": 1,
        "status": "PARTIAL",
        "classification": (
            "COORDINATE_CONTRACT_VERIFIED_"
            "NATIVE_SIMULATE_UNSUITABLE_"
            "EXTERNAL_SKIP_SIM_EXPORTABLE_MIMIC_BLOCKED"
        ),
        "scope": (
            "follower_left Grasp Editor authoring before IK; "
            "no dynamic task acceptance"
        ),
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "grasp_editor": "2.0.20",
        },
        "stage": {
            "absolute_path": str(STAGE.resolve()),
            "sha256": stage_sha256,
            "source_modified": False,
        },
        "coordinate_transform": {
            "status": "PASS",
            "stored_transform": "T_O_G",
            "application_formula": "T_W_G = T_W_O @ T_O_G",
            "inverse_placement_formula": "T_W_O = T_W_G @ inverse(T_O_G)",
            "object_frame": (
                "/World/ALOHA1GraspEditorSession/Bottle500"
            ),
            "object_frame_semantics": (
                "BOTTLE_BOTTOM_CENTER_LOCAL_PLUS_Z_TO_MOUTH"
            ),
            "gripper_frame": (
                "/World/follower_left/vx300s_left/"
                "follower_left_ee_gripper_link"
            ),
            "cad_contact_helper_is_gripper_frame": False,
            "contact_run_closure_max_abs": float(
                contact_result["placement"]["closure_max_abs"]
            ),
        },
        "joint_settings": {
            "status": "PASS",
            "active_joint": "left_finger",
            "mimic_observer_joint": "right_finger",
            "open_position_m": 0.057,
            "fully_closed_position_m": 0.021,
            "fully_closed_source": (
                "USD_AND_RUNTIME_LEFT_FINGER_LOWER_LIMIT_READBACK"
            ),
            "cad_contact_candidate_position_m": 0.048316874538855845,
            "cad_contact_candidate_source": (
                "SUPPLIER_CAD_CLEARANCE_GRASP_CANDIDATE"
            ),
            "previous_conflation_status": "CORRECTED",
            "previous_error": (
                "CAD contact candidate was incorrectly used as "
                "Position When Closed"
            ),
            "official_semantics": (
                "Position When Closed is the DOF position considered "
                "fully closed"
            ),
        },
        "native_simulate_suitability": {
            "status": "FAIL",
            "classification": (
                "NATIVE_SIMULATE_NOT_ACCEPTABLE_AS_SOLE_ALOHA_GRASP_GATE"
            ),
            "contact_case": {
                "native_success": bool(
                    contact_result["simulate"]["success"]
                ),
                "bilateral_physical_contact": bool(
                    contact_summary["bilateral_finger_contact"]
                ),
                "physical_contact_point_count": int(
                    contact_summary["physical_bottle_contact_count"]
                ),
                "mimic_residual_m": contact_residual,
            },
            "no_contact_control": {
                "native_success": bool(
                    no_contact_result["simulate"]["success"]
                ),
                "physical_contact_point_count": int(
                    no_contact_summary["physical_bottle_contact_count"]
                ),
                "false_positive_verified": (
                    bool(no_contact_result["simulate"]["success"])
                    and int(
                        no_contact_summary[
                            "physical_bottle_contact_count"
                        ]
                    )
                    == 0
                ),
                "mimic_residual_m": no_contact_residual,
                "placement_classification": (
                    "NO_OBJECT_CONTACT_CONTROL_NOT_TASK_PLACEMENT"
                ),
            },
            "local_tester_logic": {
                "stability_threshold_m": 1e-4,
                "failure_if_active_joint_reaches_fully_closed": True,
                "direct_contact_pair_requirement": False,
                "consequence": (
                    "A drive that stops short of the closed target can "
                    "produce native success without object contact."
                ),
            },
            "official_supported_fallback": (
                "EXTERNAL_PROGRAMMATIC_COUPLED_MOTION_THEN_SKIP_SIM"
            ),
        },
        "mimic_load_comparison": _classify_mimic(
            contact_residual,
            no_contact_residual,
        ),
        "external_programmatic_close_skip_sim": {
            "status": external_gate["status"],
            "execution_mode": external_gate["execution_mode"],
            "bilateral_contact": external_gate["bilateral_contact"],
            "mimic_accuracy": external_gate["mimic_accuracy"],
            "mimic_residual_m": float(
                external_result["joint_readback"]["mimic_error_abs_m"]
            ),
            "mimic_tolerance_m": MIMIC_TOLERANCE_M,
            "contact_point_count": int(
                external_summary["physical_bottle_contact_count"]
            ),
            "maximum_impulse_ns": float(
                external_summary["maximum_impulse_ns"]
            ),
            "minimum_separation_m": float(
                external_summary["minimum_separation_m"]
            ),
            "unexpected_robot_contact": bool(
                external_summary["unexpected_robot_contact"]
            ),
            "native_raw_export": external_result["native_export"],
            "derived_export": external_result["derived_export"],
            "derived_yaml_policy": (
                "RESTORE_ONLY_VERIFIED_OPEN_PREGRASP_0.057_M"
            ),
            "ik_promotion_allowed": False,
            "failure_reasons": list(external_gate["failure_reasons"]),
            "run_report": {
                "absolute_path": str(EXTERNAL_SKIP_SIM_RUN.resolve()),
                "sha256": _sha256(EXTERNAL_SKIP_SIM_RUN),
                "cleanup_errors": list(external["cleanup_errors"]),
            },
            "screenshot_review": {
                "status": external_screenshot_review["status"],
                "absolute_path": str(
                    EXTERNAL_SCREENSHOT_REVIEW.resolve()
                ),
                "sha256": _sha256(EXTERNAL_SCREENSHOT_REVIEW),
                "all_raw_and_annotated_visual_reviews": (
                    "PASS"
                    if all(
                        record["visual_model_review"] == "PASS"
                        for record in external_screenshot_review["records"]
                    )
                    else "FAIL"
                ),
                "numeric_failure_preserved": (
                    external_screenshot_review["numeric_gate"] == "FAIL"
                ),
            },
        },
        "local_physx_schema": {
            "version": "107.3.26",
            "absolute_path": str(PHYSX_SCHEMA.resolve()),
            "sha256": _sha256(PHYSX_SCHEMA),
            "declared_mimic_properties": declared_properties,
            "runtime_custom_properties_seen": [
                "naturalFrequency",
                "dampingRatio",
            ],
            "custom_property_schema_status": (
                "NOT_DECLARED_BY_LOCAL_PHYSX_MIMIC_SCHEMA"
            ),
            "custom_property_effect_status": "INCONCLUSIVE",
            "parameter_tuning_performed": False,
        },
        "local_source_evidence": {
            "grasp_tester": {
                "absolute_path": str(GRASP_TESTER_SOURCE.resolve()),
                "sha256": _sha256(GRASP_TESTER_SOURCE),
            },
            "ui_builder": {
                "absolute_path": str(UI_BUILDER_SOURCE.resolve()),
                "sha256": _sha256(UI_BUILDER_SOURCE),
            },
        },
        "official_source": {
            "url": OFFICIAL_TUTORIAL,
            "version_scope": "Isaac Sim 5.1.0",
            "confirmed_points": [
                "Grasp position/orientation is gripper relative to object.",
                "Position When Closed means fully closed.",
                "Only active gripper DOFs are exported.",
                (
                    "Complicated coupled grippers may be driven "
                    "programmatically and exported with Skip Sim."
                ),
            ],
        },
        "native_raw_yaml": {
            "status": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
            **contact_result["native_export"],
        },
        "screenshot_evidence": {
            "status": "PARTIAL",
            "contact_run": _screenshot_records(
                contact,
                review_status="PARTIAL",
                review_note=(
                    "Full arm and GUI are visible, but the fingertip "
                    "contact region is too small for final contact review."
                ),
            ),
            "no_contact_control": _screenshot_records(
                no_contact,
                review_status="REJECTED",
                review_note=(
                    "Control-only bottle relocation is not task geometry "
                    "and must not be used as grasp visual evidence."
                ),
            ),
        },
        "hard_blockers": [
            {
                "id": "HARD_BLOCKER_MIMIC_ACCURACY",
                "status": "OPEN",
                "reason": (
                    "The right-finger mimic relation exceeds the 1 mm "
                    "gate without object contact and is amplified under "
                    "bilateral bottle contact."
                ),
            },
            {
                "id": "HARD_BLOCKER_UNCALIBRATED_MIMIC_PARAMETERS",
                "status": "OPEN",
                "reason": (
                    "No measured or supplier-confirmed mimic stiffness/"
                    "damping data authorizes parameter tuning."
                ),
            },
        ],
        "next_gates": {
            "external_programmatic_grasp_then_skip_sim": (
                "FAIL_MIMIC_ACCURACY"
            ),
            "ik": "NOT_RUN",
            "five_random_bottle_videos": "NOT_RUN",
        },
        "task8": "NOT_RUN",
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
    }


def _yaml_summary(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    grasp = next(iter(document["grasps"].values()))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "format": document["format"],
        "format_version": document["format_version"],
        "object_frame": document["object_frame"],
        "gripper_frame": document["gripper_frame"],
        "active_joints": sorted(grasp["cspace_position"]),
        "cspace_position": grasp["cspace_position"],
        "pregrasp_cspace_position": grasp[
            "pregrasp_cspace_position"
        ],
        "position": grasp["position"],
        "orientation": grasp["orientation"],
    }


def build_v2_report() -> dict[str, Any]:
    stage_sha256 = _sha256(STAGE)
    if stage_sha256 != EXPECTED_STAGE_SHA256:
        raise RuntimeError(f"approved Stage hash changed: {stage_sha256}")
    run = _load_json(V2_RUN)
    coupling = _load_json(V2_COUPLING_REPORT)
    screenshots = _load_json(V2_SCREENSHOT_REVIEW)
    result = run["result"]
    raw = _yaml_summary(Path(result["native_export"]["path"]))
    derived = _yaml_summary(Path(result["derived_export"]["path"]))
    if raw["active_joints"] != ["left_finger"]:
        raise RuntimeError("native YAML must expose one hardware coordinate")
    if derived["active_joints"] != ["left_finger"]:
        raise RuntimeError("derived YAML invented another active coordinate")
    closure = float(result["placement"]["closure_max_abs"])
    gates = {
        "coupling": coupling["status"] == "PASS"
        and coupling["passing_path"] == "official_symmetric_adapter",
        "runtime": result["gate"]["status"] == "PASS",
        "contact": result["contacts"]["summary"]["status"] == "PASS",
        "raw_yaml": result["native_export"]["status"] == "PASS",
        "derived_yaml": result["derived_export"]["status"] == "PASS",
        "coordinate_closure": closure <= 1.0e-12,
        "screenshots": screenshots["status"] == "PASS",
        "stage_hash": run["inputs"]["stage"]["sha256"]
        == EXPECTED_STAGE_SHA256,
        "cleanup": not run["cleanup_errors"],
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema_version": 2,
        "status": status,
        "classification": (
            "GRASP_EDITOR_EXPORT_PASS_DIAGNOSTIC_COUPLING"
            if status == "PASS"
            else "GRASP_EDITOR_EXPORT_V2_FAILED"
        ),
        "stage": {
            "absolute_path": str(STAGE.resolve()),
            "sha256": stage_sha256,
        },
        "coupling": {
            "classification": coupling["classification"],
            "passing_path": coupling["passing_path"],
            "promotion_authorized": coupling["promotion_authorized"],
            "report_path": str(V2_COUPLING_REPORT.resolve()),
            "report_sha256": _sha256(V2_COUPLING_REPORT),
        },
        "runtime": {
            "isaac_sim": run["runtime"]["isaac_sim"],
            "kit": run["runtime"]["kit"],
            "physx": run["runtime"]["physx"],
            "grasp_editor": run["runtime"]["grasp_editor_version"],
            "mimic_residual_abs_m": result["joint_readback"][
                "mimic_error_abs_m"
            ],
            "mimic_tolerance_m": MIMIC_TOLERANCE_M,
            "bilateral_contact": result["contacts"]["summary"][
                "bilateral_finger_contact"
            ],
            "maximum_impulse_ns": result["contacts"]["summary"][
                "maximum_impulse_ns"
            ],
            "minimum_separation_m": result["contacts"]["summary"][
                "minimum_separation_m"
            ],
            "dof_order": result["dof_order"],
            "right_finger_policy": (
                "RUNTIME_OBSERVER_DERIVED_FROM_ONE_OFFICIAL_COORDINATE"
            ),
        },
        "coordinate_transform": {
            "formula": result["placement"]["formula"],
            "closure_max_abs": closure,
            "closure_status": (
                "PASS" if gates["coordinate_closure"] else "FAIL"
            ),
            "object_from_gripper": result["placement"][
                "object_from_gripper_input"
            ],
            "grasp_frame_status": result["frames"][
                "grasp_frame_readback"
            ]["status"],
            "ee_endpoint_is_grasp_center": result["frames"][
                "grasp_frame_readback"
            ]["ee_endpoint_is_grasp_center"],
        },
        "native_raw_yaml": raw,
        "derived_yaml": {
            **derived,
            "classification": (
                "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
            ),
        },
        "screenshot_review": {
            "status": screenshots["status"],
            "path": str(V2_SCREENSHOT_REVIEW.resolve()),
            "sha256": _sha256(V2_SCREENSHOT_REVIEW),
            "record_count": len(screenshots["records"]),
            "raw_and_annotated_paths_recorded": all(
                item.get("raw_absolute_path")
                and item.get("annotated_absolute_path")
                for item in screenshots["records"]
            ),
        },
        "gates": gates,
        "ik_diagnostic_allowed": status == "PASS",
        "final_asset_promotion_authorized": False,
        "next_gates": {
            "aloha_specific_fk_ik": (
                "READY_DIAGNOSTIC" if status == "PASS" else "BLOCKED"
            ),
            "dynamic_horizontal_bottle_grasp": "NOT_RUN",
            "five_random_bottle_videos": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }


def _render_v2_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA 1 Grasp Editor semantics audit V2",
            "",
            f"- Status: `{report['status']}`",
            f"- Classification: `{report['classification']}`",
            (
                "- Passing coupling path: "
                f"`{report['coupling']['passing_path']}`"
            ),
            (
                "- Runtime residual: "
                f"`{report['runtime']['mimic_residual_abs_m']} m`"
            ),
            (
                "- Bilateral physical contact: "
                f"`{report['runtime']['bilateral_contact']}`"
            ),
            (
                "- Transform closure: "
                f"`{report['coordinate_transform']['closure_status']}`"
            ),
            f"- Screenshot review: `{report['screenshot_review']['status']}`",
            f"- Diagnostic IK allowed: `{report['ik_diagnostic_allowed']}`",
            "- Final asset promotion authorized: `False`",
            "- Task 8: `NOT_RUN`",
            "",
            "The raw and derived YAML expose only `left_finger`, matching the "
            "one physical gripper actuation coordinate. The right finger "
            "remains a source-backed runtime observer derived as `-q`.",
            "",
            "The vertical bottle screenshots are robot-local Grasp Editor "
            "authoring evidence, not horizontal task-placement evidence.",
            "",
        ]
    )


def _render_markdown(report: dict[str, Any]) -> str:
    comparison = report["mimic_load_comparison"]
    raw_yaml = report["native_raw_yaml"]
    external = report["external_programmatic_close_skip_sim"]
    return "\n".join(
        [
            "# ALOHA 1 Grasp Editor semantics audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Classification: `{report['classification']}`",
            (
                "- Frozen Stage: "
                f"`{report['stage']['absolute_path']}` "
                f"(`{report['stage']['sha256']}`)"
            ),
            "- Task 8: `NOT_RUN`",
            "",
            "## Confirmed coordinate contract",
            "",
            "- Stored grasp transform: `T_O_G`.",
            "- Application: `T_W_G = T_W_O @ T_O_G`.",
            "- Inverse authoring placement: "
            "`T_W_O = T_W_G @ inverse(T_O_G)`.",
            "- Bottle frame: bottom center, local `+Z` toward the mouth.",
            "- Canonical gripper/IK frame: "
            "`follower_left_ee_gripper_link`.",
            "",
            "## Corrected Grasp Editor setting",
            "",
            "- `Position When Open`: `0.057 m`.",
            "- `Position When Closed`: `0.021 m` "
            "(verified legal fully-closed lower limit).",
            "- `0.048316874538855845 m` is a CAD contact candidate, "
            "not the fully-closed setting.",
            "",
            "## Native SIMULATE finding",
            "",
            "- Native SIMULATE is `FAIL` as a sole ALOHA acceptance gate.",
            "- With bilateral bottle contact, mimic residual: "
            f"`{comparison['contact_residual_m']:.9f} m`.",
            "- With zero bottle contact, mimic residual: "
            f"`{comparison['no_contact_residual_m']:.9f} m`.",
            "- Amplification ratio: "
            f"`{comparison['amplification_ratio']:.3f}`.",
            "- The no-contact control still returned native success, "
            "proving a false positive when contact reports are omitted.",
            "",
            "The Isaac Sim 5.1 tutorial explicitly supports an external "
            "programmatic closing trajectory followed by **Skip Sim** for "
            "heavily coupled grippers.",
            "",
            "## External close + Skip Sim result",
            "",
            "- Bilateral runtime contact: "
            f"`{external['bilateral_contact']}` "
            f"({external['contact_point_count']} contact points).",
            "- Native raw Skip Sim export: "
            f"`{external['native_raw_export']['status']}`.",
            "- Derived pregrasp export: "
            f"`{external['derived_export']['status']}` "
            "(`DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`).",
            "- Runtime mimic residual: "
            f"`{external['mimic_residual_m']:.9f} m` "
            f"(gate `{external['mimic_tolerance_m']:.6f} m`).",
            "- Overall external path: `FAIL_MIMIC_ACCURACY`; "
            "IK promotion remains forbidden.",
            "",
            "## Evidence",
            "",
            f"- Native raw YAML: `{raw_yaml['path']}`",
            f"- Native raw YAML SHA-256: `{raw_yaml['sha256']}`",
            f"- Official 5.1 tutorial: {OFFICIAL_TUTORIAL}",
            (
                "- Full JSON report: "
                f"`{OUTPUT_JSON.resolve()}`"
            ),
            "",
            "## Open blockers",
            "",
            "- `HARD_BLOCKER_MIMIC_ACCURACY`",
            "- `HARD_BLOCKER_UNCALIBRATED_MIMIC_PARAMETERS`",
            "- IK: `NOT_RUN`",
            "- Five random-bottle videos: `NOT_RUN`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()
    report = build_v2_report() if args.v2 else build_report()
    output_json = V2_OUTPUT_JSON if args.v2 else OUTPUT_JSON
    output_md = V2_OUTPUT_MD if args.v2 else OUTPUT_MD
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(
        (
            _render_v2_markdown(report)
            if args.v2
            else _render_markdown(report)
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(output_json.resolve()),
                "markdown": str(output_md.resolve()),
                "classification": report["classification"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the exact-model ALOHA 1 ViperX gripper hardware audit.

This tool validates a frozen, source-backed configuration. It does not query or
control a real robot and it does not translate hardware register values into
PhysX gains or force limits.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/aloha1_vx300s_gripper_hardware_model.yaml"
DEFAULT_JSON = PROJECT_ROOT / "reports/aloha1_mapping/aloha_vx300s_gripper_hardware_parameter_audit.json"
DEFAULT_MD = PROJECT_ROOT / "reports/aloha1_mapping/aloha_vx300s_gripper_hardware_parameter_audit.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motor_angle_to_linear_position(theta: float, radius: float, arm: float) -> float:
    """Replicate the Interbotix X-Series driver linkage equation."""
    first = radius * math.sin(theta)
    intermediate_squared = max(0.0, radius * radius - first * first)
    second_squared = max(0.0, arm * arm - intermediate_squared)
    return first + math.sqrt(second_squared)


def _inverse_linear_position(position: float, radius: float, arm: float) -> float:
    low = -math.pi / 2.0
    high = math.pi / 2.0
    low_value = motor_angle_to_linear_position(low, radius, arm)
    high_value = motor_angle_to_linear_position(high, radius, arm)
    if not low_value <= position <= high_value:
        raise ValueError(f"position {position} is outside linkage range [{low_value}, {high_value}]")
    for _ in range(80):
        middle = (low + high) / 2.0
        if motor_angle_to_linear_position(middle, radius, arm) < position:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _validate_formula(document: dict[str, Any]) -> dict[str, Any]:
    linkage = document["hardware"]["gripper_linkage"]
    limits = document["simulation_description"]["urdf_finger_limits"]
    radius = float(linkage["horn_radius_m"])
    arm = float(linkage["arm_length_m"])
    lower_position = float(limits["left_lower_m"])
    upper_position = float(limits["left_upper_m"])
    lower_angle = _inverse_linear_position(lower_position, radius, arm)
    upper_angle = _inverse_linear_position(upper_position, radius, arm)

    samples = [
        motor_angle_to_linear_position(
            lower_angle + (upper_angle - lower_angle) * index / 1000.0,
            radius,
            arm,
        )
        for index in range(1001)
    ]
    monotonic = all(current > previous for previous, current in pairwise(samples))
    endpoint_error = max(
        abs(samples[0] - lower_position),
        abs(samples[-1] - upper_position),
    )
    return {
        "status": "PASS" if monotonic and endpoint_error < 1.0e-12 else "FAIL",
        "source_function": linkage["formula_implementation_source"],
        "formula": linkage["motor_angle_to_linear_formula"],
        "horn_radius_m": radius,
        "arm_length_m": arm,
        "lower_linear_position_m": lower_position,
        "upper_linear_position_m": upper_position,
        "lower_motor_angle_rad": lower_angle,
        "upper_motor_angle_rad": upper_angle,
        "endpoint_max_abs_error_m": endpoint_error,
        "sample_count": len(samples),
        "monotonic_over_urdf_range": monotonic,
        "published_sign_relation": linkage["published_finger_positions"],
        "right_is_negative_left": (
            linkage["published_finger_positions"] == {"left_finger": "+x", "right_finger": "-x"}
        ),
    }


def _verify_sources(
    document: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatches: list[str] = []
    provenance_missing: list[str] = []
    required_provenance = (
        "repository",
        "branch_or_tag",
        "commit",
        "license",
        "local_path",
        "sha256",
    )

    for name, source in document["frozen_sources"].items():
        absent_fields = [field for field in required_provenance if not source.get(field)]
        if absent_fields:
            provenance_missing.append(f"{name}:{','.join(absent_fields)}")

        path = project_root / source["local_path"]
        exists = path.is_file()
        actual_hash = _sha256(path) if exists else None
        matches = actual_hash == source["sha256"] if exists else False
        if not exists:
            missing.append(name)
        elif not matches:
            mismatches.append(name)
        records.append(
            {
                "name": name,
                "absolute_path": str(path.resolve()),
                "exists": exists,
                "expected_sha256": source["sha256"],
                "actual_sha256": actual_hash,
                "sha256_matches": matches,
                "repository": source["repository"],
                "branch_or_tag": source["branch_or_tag"],
                "commit": source["commit"],
                "license": source["license"],
            }
        )

    return {
        "records": records,
        "missing_sources": missing,
        "hash_mismatches": mismatches,
        "missing_provenance_fields": provenance_missing,
        "all_local_hashes_match": not missing and not mismatches,
        "all_required_provenance_present": not provenance_missing,
    }


def build_report(
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = (
        config_path.resolve()
        if config_path is not None
        else project_root / "configs/aloha1_vx300s_gripper_hardware_model.yaml"
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_verification = _verify_sources(document, project_root)
    formula_validation = _validate_formula(document)

    actuator = document["hardware"]["gripper_actuator"]
    scope = document["scope"]
    identity_pass = (
        scope["generation"] == "STATIONARY_ALOHA_1"
        and scope["follower_product"] == "ViperX-300 6DOF"
        and scope["project_model"] == "aloha_vx300s"
        and actuator["model"] == "XM430-W350"
        and actuator["dynamixel_id"] == 9
        and actuator["physical_actuator_count"] == 1
        and actuator["operating_mode"] == "pwm"
        and actuator["right_finger_independently_sensed"] is False
    )
    source_pass = (
        source_verification["all_local_hashes_match"] and source_verification["all_required_provenance_present"]
    )
    status = "PASS" if identity_pass and source_pass and formula_validation["status"] == "PASS" else "FAIL"

    return {
        "schema_version": 1,
        "status": status,
        "official_hardware_model_status": status,
        "scope": scope,
        "identity_check": {
            "status": "PASS" if identity_pass else "FAIL",
            "product": scope["follower_product"],
            "project_model": scope["project_model"],
            "actuator": f"{actuator['manufacturer']} {actuator['model']}",
            "dynamixel_id": actuator["dynamixel_id"],
            "physical_actuator_count": actuator["physical_actuator_count"],
            "right_finger_state": document["hardware"]["gripper_linkage"]["right_finger_state_kind"],
        },
        "official_source_chain": document["official_source_chain"],
        "hardware": document["hardware"],
        "simulation_description": document["simulation_description"],
        "source_verification": source_verification,
        "formula_validation": formula_validation,
        "provenance_classes": document["provenance_classes"],
        "unconfirmed_physical_quantities": document["unconfirmed_physical_quantities"],
        "important_boundaries": {
            "official_aperture_selection": document["hardware"]["maximum_aperture_conflict"]["selection_status"],
            "supplier_cad_license": document["frozen_sources"]["supplier_cad"]["license"],
            "physx_max_force_mapping": actuator["current_limit"]["physx_max_force_mapping"],
            "real_robot_accessed": False,
        },
        "config": {
            "absolute_path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "task8": document["task8"],
    }


def _markdown(report: dict[str, Any]) -> str:
    identity = report["identity_check"]
    conflict = report["hardware"]["maximum_aperture_conflict"]
    source = report["source_verification"]
    formula = report["formula_validation"]
    lines = [
        "# ALOHA ViperX-300 Gripper Hardware Parameter Audit",
        "",
        f"- `OFFICIAL_HARDWARE_MODEL_STATUS={report['official_hardware_model_status']}`",
        f"- Scope: `{report['scope']['generation']}`",
        f"- Follower: `{identity['product']}` / `{identity['project_model']}`",
        f"- Actuator: `{identity['actuator']}`, DYNAMIXEL ID `{identity['dynamixel_id']}`",
        f"- Physical gripper actuators: `{identity['physical_actuator_count']}`",
        f"- Right finger state: `{identity['right_finger_state']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "## Verified linkage",
        "",
        f"- Horn radius: `{formula['horn_radius_m']} m`",
        f"- Arm length: `{formula['arm_length_m']} m`",
        f"- Formula: `{formula['formula']}`",
        f"- URDF-range monotonicity: `{formula['monotonic_over_urdf_range']}`",
        f"- Published sign relation: `{formula['published_sign_relation']}`",
        "",
        "## Source integrity",
        "",
        f"- Frozen local hashes all match: `{source['all_local_hashes_match']}`",
        f"- Missing sources: `{source['missing_sources']}`",
        f"- Hash mismatches: `{source['hash_mismatches']}`",
        f"- Missing provenance fields: `{source['missing_provenance_fields']}`",
        "",
        "## Fail-closed boundaries",
        "",
        ("- Official maximum-aperture claims: " + ", ".join(f"`{claim['value_m']} m`" for claim in conflict["claims"])),
        f"- Aperture selection: `{conflict['selection_status']}`",
        "- `Current_Limit=200` is a pinned motor-config register value; using the "
        "ROBOTIS current unit gives `0.538 A`, but it is not a calibrated "
        "fingertip-force or PhysX max-force value.",
        "- The supplier STEP license remains `UNKNOWN_HARD_BLOCKER`; the STEP "
        "is retained only in `.codex/artifacts` and is not redistributable.",
        "",
        "## Evidence classes",
        "",
    ]
    for key, values in report["provenance_classes"].items():
        lines.append(f"- `{key}`: {', '.join(values) if values else 'none'}")
    lines.extend(["", "## Unconfirmed physical quantities", ""])
    lines.extend(f"- {item}" for item in report["unconfirmed_physical_quantities"])
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    *,
    json_path: Path = DEFAULT_JSON,
    md_path: Path = DEFAULT_MD,
) -> dict[str, str]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    report = build_report(
        project_root=args.project_root,
        config_path=args.config,
    )
    written = write_reports(report, json_path=args.json, md_path=args.markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "official_hardware_model_status": report["official_hardware_model_status"],
                "written": written,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

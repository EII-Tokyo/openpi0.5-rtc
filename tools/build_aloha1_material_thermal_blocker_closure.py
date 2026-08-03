#!/usr/bin/env python3
"""Build the material and continuous-duty evidence closure for ALOHA1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.material_thermal_blocker_closure import classify_material_thermal_gate

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/aloha1_mapping"
DEFAULT_MATERIAL_AUDIT = REPORT_DIR / "gripper_material_audit.json"
DEFAULT_DYNAMICS = REPORT_DIR / "aloha1_dynamics_contract.json"
DEFAULT_RESEARCH = REPORT_DIR / "aloha1_model_blocker_deep_research.json"
DEFAULT_BOTTLE_CONFIG = ROOT / "configs/aloha1_bottle_asset.yaml"
DEFAULT_JSON = REPORT_DIR / "aloha1_material_thermal_blocker_closure.json"
DEFAULT_MARKDOWN = REPORT_DIR / "aloha1_material_thermal_blocker_closure.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def build_report(
    *,
    material_audit_path: Path,
    dynamics_path: Path,
    research_path: Path,
    bottle_config_path: Path,
) -> dict[str, Any]:
    material = json.loads(material_audit_path.read_text(encoding="utf-8"))
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    research = json.loads(research_path.read_text(encoding="utf-8"))
    gate = classify_material_thermal_gate(
        material_binding_status=str(material["status"]),
        temporary_material_status=str(material["temporary_material_status"]),
        finger_material_identity=None,
        bottle_material_identity=None,
        pair_friction_measurement=None,
        measured_continuous_thermal_curve=None,
    )
    contact_research = research["contact_material_research"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": gate["status"],
        "inputs": {
            "runtime_material_audit": _input(material_audit_path),
            "dynamics_contract": _input(dynamics_path),
            "model_blocker_research": _input(research_path),
            "bottle_asset_config": _input(bottle_config_path),
        },
        "gate": gate,
        "runtime_binding": {
            "status": material["status"],
            "temporary_material_status": material[
                "temporary_material_status"
            ],
            "friction_status": material["FRICTION_STATUS"],
            "effective_authored_static_friction": 0.7,
            "effective_authored_dynamic_friction": 0.7,
            "classification": "RUNTIME_READBACK_TEMPORARY_VALUE",
        },
        "physical_material_identity": {
            "finger_pad": None,
            "bottle_body": None,
            "pair_measurement": None,
            "missing_exact_identity": contact_research[
                "missing_exact_physical_identity"
            ],
            "invalid_substitutes": contact_research["not_valid_substitutes"],
        },
        "continuous_duty": {
            "manufacturer_tables_status": dynamics["actuator_contract"][
                "manufacturer_tables_status"
            ],
            "official_estimated_continuous_torque_Nm": dynamics[
                "actuator_contract"
            ]["official_estimated_continuous_torque_Nm"],
            "estimate_is_measured_thermal_curve": dynamics[
                "actuator_contract"
            ]["continuous_estimate_is_measured_thermal_curve"],
            "stall_torque_used_as_continuous": dynamics["actuator_contract"][
                "stall_torque_used_as_continuous"
            ],
            "measured_curve": None,
        },
        "required_measurements": {
            "friction": contact_research[
                "minimum_measurement_if_official_pair_data_remains_unavailable"
            ],
            "continuous_duty": (
                "Exact actuator and loaded gripper linkage: torque/force, speed, "
                "current, voltage, winding/case temperature, ambient temperature, "
                "duty cycle, duration, and thermal shutdown/derating boundary."
            ),
        },
        "diagnostic_parameter_scan_status": "NOT_RUN_NO_CALIBRATED_PAIR_PROPERTY",
        "real_robot_or_material_test_run": False,
        "final_or_default_asset_modified": False,
        "task8_status": "AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE",
        "interpretation": (
            "The existing Isaac runtime audit verifies that the temporary physics "
            "materials are bound and combine as authored. It does not identify the "
            "real finger-pad/bottle materials or calibrate their friction. The exact "
            "manufacturer motor tables likewise do not provide the measured loaded "
            "continuous gripper force/thermal envelope needed for a final maxForce."
        ),
    }
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _markdown(report: dict[str, Any]) -> str:
    missing = "\n".join(f"- `{item}`" for item in report["gate"]["missing_inputs"])
    return f"""# ALOHA1 material and continuous-duty closure

- Status: **{report['status']}**
- Runtime material binding verified: `{report['gate']['runtime_binding_verified']}`
- Physical friction calibrated: `{report['gate']['physical_friction_calibrated']}`
- Continuous force envelope verified: `{report['gate']['continuous_force_envelope_verified']}`
- Diagnostic friction scan: **{report['diagnostic_parameter_scan_status']}**
- Final/default asset modified: `{report['final_or_default_asset_modified']}`

## Result

{report['interpretation']}

## Missing exact evidence

{missing}

The authored value `0.7` remains `TEMPORARY_UNCALIBRATED`. A successful bottle
hold or a generic plastic table is not accepted as calibration. The published
stall and 20%-of-stall estimates are not treated as a measured loaded thermal
envelope.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-audit", type=Path, default=DEFAULT_MATERIAL_AUDIT)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--bottle-config", type=Path, default=DEFAULT_BOTTLE_CONFIG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(
        material_audit_path=args.material_audit,
        dynamics_path=args.dynamics,
        research_path=args.research,
        bottle_config_path=args.bottle_config,
    )
    args.json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "missing": report["gate"]["missing_inputs"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

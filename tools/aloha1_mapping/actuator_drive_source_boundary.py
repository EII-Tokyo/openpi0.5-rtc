"""Build the exact-model actuator and controller-to-PhysX evidence boundary."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

import yaml

from tools.aloha1_mapping.official_parameter_sources import load_source_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    without_tags = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _require_number(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"official source is missing {label}")
    return float(match.group(1))


def _source_path(root: Path, source: dict[str, Any]) -> Path:
    path = (root / source["local_path"]).resolve(strict=True)
    if _sha256(path) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {source['id']}")
    return path


def _product_record(
    *,
    root: Path,
    source: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    path = _source_path(root, source)
    text = _plain_text(path)
    disclosure = (
        "This is an estimated value for continuous torque, calculated at 20% "
        "of stall torque."
    )
    if disclosure not in text:
        raise ValueError(f"{model} page lacks the continuous-torque disclosure")
    continuous = _require_number(
        text,
        r"Estimated Rated Torque:\s*([0-9.]+)\s*Nm",
        f"{model} estimated rated torque",
    )
    stall = _require_number(
        text, r"Stall Torque:\s*([0-9.]+)\s*Nm", f"{model} stall torque"
    )
    voltage = _require_number(
        text, r"Input Voltage:\s*([0-9.]+)\s*v", f"{model} input voltage"
    )
    no_load_rpm = _require_number(
        text, r"No Load RPM:\s*([0-9.]+)\s*RPM", f"{model} no-load speed"
    )
    stall_current = _require_number(
        text, r"Stall Current:\s*([0-9.]+)\s*A", f"{model} stall current"
    )
    fraction = continuous / stall
    if abs(fraction - 0.2) > 1.0e-12:
        raise ValueError(f"{model} continuous estimate is not 20% of stall")
    return {
        "model": model,
        "source_id": source["id"],
        "source_url": source["url"],
        "local_path": str(path),
        "sha256": source["sha256"],
        "reference_voltage_V": voltage,
        "stall_torque_Nm": stall,
        "stall_current_A": stall_current,
        "no_load_speed_rpm": no_load_rpm,
        "estimated_continuous_torque_Nm": continuous,
        "continuous_estimate_fraction_of_stall": fraction,
        "continuous_value_is_manufacturer_estimate": True,
        "continuous_value_is_measured_thermal_curve": False,
        "continuous_torque_speed_current_curve": "NOT_PUBLISHED_ON_EXACT_MODEL_PRODUCT_PAGE",
        "stall_torque_used_as_continuous": False,
    }


def build_report(root: Path, source_manifest_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = load_source_manifest(source_manifest_path)
    sources = {record["id"]: record for record in manifest["sources"]}
    models = {
        "XM540-W270": _product_record(
            root=root,
            source=sources["robotis_xm540_w270_product"],
            model="XM540-W270",
        ),
        "XM430-W350": _product_record(
            root=root,
            source=sources["robotis_xm430_w350_product"],
            model="XM430-W350",
        ),
    }

    modes_path = _source_path(root, sources["interbotix_xsarm_default_modes"])
    modes = yaml.safe_load(modes_path.read_text(encoding="utf-8"))
    motor_path = _source_path(root, sources["interbotix_aloha_vx300s_motor_config"])
    motor_config = yaml.safe_load(motor_path.read_text(encoding="utf-8"))
    current_ticks = int(motor_config["motors"]["gripper"]["Current_Limit"])
    current_per_tick_a = 0.00269

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "EXACT_ALOHA_VX300S_ACTUATOR_AND_CONTROLLER_SOURCE_BOUNDARY",
        "product": manifest["product"],
        "inputs": {
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": _sha256(source_manifest_path),
            },
            "modes": {
                "path": str(modes_path),
                "sha256": _sha256(modes_path),
                "source_id": "interbotix_xsarm_default_modes",
            },
            "motor_config": {
                "path": str(motor_path),
                "sha256": _sha256(motor_path),
                "source_id": "interbotix_aloha_vx300s_motor_config",
            },
        },
        "actuator_models": models,
        "control_modes": {
            "arm": modes["groups"]["arm"]["operating_mode"],
            "gripper": modes["singles"]["gripper"]["operating_mode"],
            "source": "interbotix_xsarm_default_modes",
        },
        "joint_actuator_identity": {
            name: {
                "id": int(record["ID"]),
                "model": (
                    "XM430-W350" if int(record["ID"]) >= 8 else "XM540-W270"
                ),
                "secondary_id": record.get("Secondary_ID"),
            }
            for name, record in motor_config["motors"].items()
        },
        "gripper_control_boundary": {
            "operating_mode": "pwm",
            "current_limit_ticks": current_ticks,
            "current_unit_A_per_tick": current_per_tick_a,
            "current_limit_A": round(current_ticks * current_per_tick_a, 12),
            "current_limit_is_physx_max_force": False,
            "pwm_mode_is_closed_loop_position_control": False,
            "pwm_command_to_output_torque_mapping": "NOT_DEFINED_BY_OFFICIAL_SOURCES",
            "linkage_conversion_required": True,
        },
        "physx_drive_mapping": {
            "status": "HARD_BLOCKER",
            "hardware_integer_gain_direct_mapping": "PROHIBITED",
            "stiffness_Nm_per_rad": None,
            "damping_Nm_s_per_rad": None,
            "max_force_Nm": None,
            "reason": (
                "ROBOTIS register gains are dimensionless firmware-controller values, "
                "while PhysX drive stiffness and damping are physical joint-space "
                "coefficients. The official sources do not publish an equivalent closed-loop "
                "transfer model for this assembled arm and PWM gripper."
            ),
        },
        "continuous_envelope": {
            "status": "PARTIAL",
            "official_12V_estimated_torque_available": True,
            "measured_continuous_torque_speed_current_thermal_curve_available": False,
            "formal_use": "CONSERVATIVE_TORQUE_REFERENCE_NOT_A_COMPLETE_DYNAMIC_ENVELOPE",
        },
        "hard_blockers": [
            {
                "id": "HARD_BLOCKER_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE",
                "blocks": ["FORMAL_DYNAMIC_OUTPUT_ENVELOPE"],
                "does_not_block": ["SOURCE_AUDIT", "12V_ESTIMATED_CONTINUOUS_TORQUE_REFERENCE"],
            },
            {
                "id": "HARD_BLOCKER_PHYSX_DRIVE_PHYSICAL_DERIVATION",
                "blocks": ["FORMAL_PHYSX_DRIVE_LAYER"],
                "does_not_block": ["SOURCE_AUDIT", "KINEMATICS", "COLLIDER_GEOMETRY"],
            },
        ],
        "runtime_simulation_used": False,
        "real_robot_accessed": False,
        "engineering_inference_inserted": False,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["deterministic_signature"] = hashlib.sha256(payload).hexdigest()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        (
            f"| `{model['model']}` | {model['reference_voltage_V']:.0f} V | "
            f"{model['stall_torque_Nm']:.2f} N·m | "
            f"{model['estimated_continuous_torque_Nm']:.2f} N·m | "
            "manufacturer estimate = 20% stall |"
        )
        for model in report["actuator_models"].values()
    ]
    return "\n".join(
        [
            "# ALOHA1 exact-model actuator and drive source boundary",
            "",
            f"- Overall: **{report['status']}**",
            "- Arm mode: `position`",
            "- Gripper mode: `pwm`",
            "- Direct DYNAMIXEL integer-gain → PhysX gain mapping: `PROHIBITED`",
            "",
            "| Model | Reference | Stall torque | Estimated continuous torque | Evidence class |",
            "|---|---:|---:|---:|---|",
            *rows,
            "",
            "ROBOTIS explicitly labels the continuous values as estimates calculated at 20% "
            "of stall torque. They are retained as conservative official references, not "
            "misrepresented as measured thermal torque-speed-current curves.",
            "",
            "The pinned Interbotix configuration uses position control for the arm and PWM "
            "control for the gripper. The 200-tick gripper Current_Limit converts to 0.538 A, "
            "but this is not a PhysX maxForce and does not define PWM-command torque. No "
            "physical stiffness, damping, or maxForce was guessed.",
            "",
        ]
    )

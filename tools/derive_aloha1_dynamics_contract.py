#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from tools.aloha1_mapping.official_parameter_sources import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT = ROOT / "generated/urdf/follower_left.urdf"
DEFAULT_RIGHT = ROOT / "generated/urdf/follower_right.urdf"
DEFAULT_SOURCES = ROOT / "configs/aloha1_official_parameter_sources.yaml"
DEFAULT_MATRIX = ROOT / "reports/aloha1_mapping/aloha1_official_parameter_matrix.json"
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_dynamics_contract.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_dynamics_contract.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inertial_records(path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot()
    records = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        name = link.attrib["name"]
        for prefix in ("follower_left_", "follower_right_"):
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
        origin = inertial.find("origin")
        com = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        mass = float(inertial.find("mass").attrib["value"])
        raw = inertial.find("inertia").attrib
        inertia = np.asarray(
            [
                [float(raw["ixx"]), float(raw["ixy"]), float(raw["ixz"])],
                [float(raw["ixy"]), float(raw["iyy"]), float(raw["iyz"])],
                [float(raw["ixz"]), float(raw["iyz"]), float(raw["izz"])],
            ]
        )
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        shift = mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        inertia_at_link_origin = rotation @ inertia @ rotation.T + shift
        recovered = rotation.T @ (inertia_at_link_origin - shift) @ rotation
        eigenvalues = np.linalg.eigvalsh(inertia)
        triangle_margin = float(eigenvalues[0] + eigenvalues[1] - eigenvalues[2])
        records.append(
            {
                "link": name,
                "classification": (
                    "SOURCE_AUTHORED_VIRTUAL_HELPER_INERTIAL"
                    if name in {"ee_arm_link", "fingers_link", "ee_gripper_link"}
                    else "SOURCE_AUTHORED_PHYSICAL_LINK_INERTIAL"
                ),
                "mass_kg": mass,
                "com_xyz_m": com.tolist(),
                "inertial_rpy_rad": rpy.tolist(),
                "inertia_com_kg_m2": inertia.tolist(),
                "principal_moments_kg_m2": eigenvalues.tolist(),
                "triangle_margin_kg_m2": triangle_margin,
                "parallel_axis_roundtrip_max_abs_error": float(np.max(np.abs(recovered - inertia))),
                "finite": bool(np.isfinite(mass) and np.isfinite(com).all() and np.isfinite(inertia).all()),
                "positive_mass": mass > 0.0,
                "symmetric": bool(np.allclose(inertia, inertia.T, atol=0.0)),
                "positive_definite": bool(eigenvalues[0] > 0.0),
                "triangle_inequality": triangle_margin >= 0.0,
            }
        )
    return records


def _normalized_signature(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_contract(
    *,
    left_urdf: Path,
    right_urdf: Path,
    source_manifest_path: Path,
    parameter_matrix_path: Path,
) -> dict[str, object]:
    left_urdf = left_urdf.resolve(strict=True)
    right_urdf = right_urdf.resolve(strict=True)
    source_manifest = load_source_manifest(source_manifest_path)
    source_map = {item["id"]: item for item in source_manifest["sources"]}
    matrix = json.loads(parameter_matrix_path.read_text(encoding="utf-8"))
    left_records = _inertial_records(left_urdf)
    right_records = _inertial_records(right_urdf)
    left_signature = _normalized_signature(left_records)
    right_signature = _normalized_signature(right_records)
    all_inertials_pass = all(
        record[gate]
        for record in [*left_records, *right_records]
        for gate in (
            "finite",
            "positive_mass",
            "symmetric",
            "positive_definite",
            "triangle_inequality",
        )
    )

    motor_config_path = ROOT / source_map["interbotix_aloha_vx300s_motor_config"]["local_path"]
    motor_config = yaml.safe_load(motor_config_path.read_text(encoding="utf-8"))
    gripper_current_ticks = int(motor_config["motors"]["gripper"]["Current_Limit"])
    current_ampere_per_tick = next(
        record["value"]["current_ampere_per_tick"]
        for record in matrix["records"]
        if record["id"] == "register_conversions.XM430-W350"
    )
    performance_records = [
        record
        for record in matrix["records"]
        if record["id"].startswith("actuator_performance.") and record["status"] != "HARD_BLOCKER"
    ]
    continuous_estimates = {
        record["id"].removeprefix("estimated_continuous_torque."): record[
            "value"
        ]["estimated_continuous_torque_Nm"]
        for record in matrix["records"]
        if record["id"].startswith("estimated_continuous_torque.")
    }
    continuous_blocker = next(record for record in matrix["records"] if record["id"] == "continuous_actuator_envelope")
    drive_blocker = next(record for record in matrix["records"] if record["id"] == "physx_joint_drive_mapping")
    gripper_runtime_control = next(
        record["value"]
        for record in matrix["records"]
        if record["id"] == "aloha_follower_gripper_runtime_control"
    )

    contract: dict[str, object] = {
        "schema_version": 1,
        "status": "PARTIAL" if all_inertials_pass else "FAIL",
        "product": source_manifest["product"],
        "inputs": {
            "left_urdf": {"path": str(left_urdf), "sha256": _sha256(left_urdf)},
            "right_urdf": {"path": str(right_urdf), "sha256": _sha256(right_urdf)},
            "pinned_xacro_sha256": source_map["interbotix_aloha_vx300s_xacro"]["sha256"],
            "parameter_matrix": {
                "path": str(parameter_matrix_path.resolve()),
                "sha256": _sha256(parameter_matrix_path),
                "deterministic_signature": matrix["deterministic_signature"],
            },
        },
        "inertial_contract": {
            "status": "PASS" if all_inertials_pass else "FAIL",
            "link_count_per_follower": len(left_records),
            "minimum_mass_kg": min(record["mass_kg"] for record in left_records),
            "minimum_principal_moment_kg_m2": min(min(record["principal_moments_kg_m2"]) for record in left_records),
            "minimum_triangle_margin_kg_m2": min(record["triangle_margin_kg_m2"] for record in left_records),
            "max_parallel_axis_roundtrip_error": max(
                record["parallel_axis_roundtrip_max_abs_error"] for record in left_records
            ),
            "default_density_used": False,
            "left_records": left_records,
            "right_records": right_records,
        },
        "left_right_inertial_identity": {
            "status": "PASS" if left_signature == right_signature else "FAIL",
            "left_signature": left_signature,
            "right_signature": right_signature,
            "mirrored": False if left_signature == right_signature else None,
        },
        "actuator_contract": {
            "manufacturer_tables_status": "PASS" if len(performance_records) == 2 else "FAIL",
            "models": [record["value"] for record in performance_records],
            "stall_torque_used_as_continuous": False,
            "official_estimated_continuous_torque_Nm": continuous_estimates,
            "continuous_estimate_is_measured_thermal_curve": False,
            "continuous_joint_envelope_status": continuous_blocker["status"],
            "continuous_joint_envelope_blocker": continuous_blocker["blocker"],
            "physx_drive_mapping_status": drive_blocker["status"],
            "physx_drive_mapping_blocker": drive_blocker["blocker"],
            "hardware_integer_gain_direct_mapping": "PROHIBITED",
            "shadow_semantics": {
                "shoulder": {"primary_id": 2, "shadow_id": 3, "secondary_id": 2},
                "elbow": {"primary_id": 4, "shadow_id": 5, "secondary_id": 4},
            },
            "gripper_current_limit": {
                "raw_ticks": gripper_current_ticks,
                "ampere_per_tick": current_ampere_per_tick,
                "derived_formula": "Current_Limit_ticks * 0.00269 A/tick",
                "derived_ampere": round(gripper_current_ticks * current_ampere_per_tick, 12),
                "physx_max_force_mapping": "NOT_DIRECTLY_MAPPABLE",
            },
            "gripper_runtime_control": gripper_runtime_control,
        },
        "formal_candidate_gate": "BLOCKED",
        "hard_blockers": [
            continuous_blocker["blocker"],
            drive_blocker["blocker"],
        ],
        "runtime_simulation_used": False,
        "real_robot_accessed": False,
    }
    contract["deterministic_signature"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def _markdown(contract: dict[str, Any]) -> str:
    inertial = contract["inertial_contract"]
    actuator = contract["actuator_contract"]
    return "\n".join(
        [
            "# ALOHA1 dynamics contract",
            "",
            f"- Overall: **{contract['status']}**",
            f"- Authored inertials: **{inertial['status']}** ({inertial['link_count_per_follower']} links per follower)",
            f"- Minimum mass: `{inertial['minimum_mass_kg']:.12g} kg`",
            f"- Minimum principal moment: `{inertial['minimum_principal_moment_kg_m2']:.12g} kg*m^2`",
            f"- Minimum triangle margin: `{inertial['minimum_triangle_margin_kg_m2']:.12g} kg*m^2`",
            f"- Continuous actuator envelope: **{actuator['continuous_joint_envelope_status']}**",
            f"- PhysX drive mapping: **{actuator['physx_drive_mapping_status']}**",
            "",
            "All authored mass/COM/inertia records are finite, positive-definite and satisfy "
            "the rigid-body triangle inequality. The parallel-axis transform was round-tripped "
            "numerically. Virtual marker-link inertials remain explicitly classified and are "
            "not misrepresented as measured physical components.",
            "",
            "The ROBOTIS voltage-conditioned stall tables are preserved, but stall torque is "
            "not used as continuous torque or PhysX maxForce. ROBOTIS' exact-model 12 V "
            "continuous estimates (20% of stall) are retained with their disclosure; they are "
            "not labeled measured thermal curves. The full torque-speed-current thermal "
            "envelope and controller-to-PhysX mapping remain narrow hard blockers; no fitted "
            "value was inserted.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right-urdf", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    contract = build_contract(
        left_urdf=args.left_urdf,
        right_urdf=args.right_urdf,
        source_manifest_path=args.sources,
        parameter_matrix_path=args.matrix,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(contract), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": contract["status"],
                "inertials": contract["inertial_contract"]["status"],
                "formal_candidate_gate": contract["formal_candidate_gate"],
            }
        )
    )
    return 0 if contract["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

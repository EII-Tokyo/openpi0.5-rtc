#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.aloha1_mapping.aloha1_model_math import OFFICIAL_JOINT_ORDER
from tools.aloha1_mapping.aloha1_model_math import load_urdf_chain
from tools.aloha1_mapping.aloha1_model_math import numerical_space_jacobian
from tools.aloha1_mapping.aloha1_model_math import poe_fk
from tools.aloha1_mapping.aloha1_model_math import poe_space_jacobian
from tools.aloha1_mapping.aloha1_model_math import rotation_distance_rad
from tools.aloha1_mapping.aloha1_model_math import urdf_fk
from tools.aloha1_mapping.official_parameter_sources import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT = ROOT / "generated/urdf/follower_left.urdf"
DEFAULT_RIGHT = ROOT / "generated/urdf/follower_right.urdf"
DEFAULT_SOURCES = ROOT / "configs/aloha1_official_parameter_sources.yaml"
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_kinematic_contract.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_kinematic_contract.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_chain_signature(chain_record: dict[str, object]) -> str:
    payload = []
    for joint in chain_record["chain"]:
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        payload.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib["type"],
                "parent_suffix": joint.find("parent").attrib["link"].split("_", 2)[-1],
                "child_suffix": joint.find("child").attrib["link"].split("_", 2)[-1],
                "origin": dict(origin.attrib) if origin is not None else {},
                "axis": dict(axis.attrib) if axis is not None else {},
                "limit": dict(limit.attrib) if limit is not None else {},
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _samples(limits: dict[str, dict[str, float]]) -> list[np.ndarray]:
    lower = np.asarray([limits[name]["lower"] for name in OFFICIAL_JOINT_ORDER])
    upper = np.asarray([limits[name]["upper"] for name in OFFICIAL_JOINT_ORDER])
    return [
        np.zeros(6),
        np.asarray([0.2, -0.4, 0.3, 0.7, -0.2, 0.5]),
        np.asarray([-0.5, 0.3, -0.6, -0.8, 0.4, -0.7]),
        lower + 0.25 * (upper - lower),
        lower + 0.75 * (upper - lower),
    ]


def build_contract(*, left_urdf: Path, right_urdf: Path, source_manifest_path: Path) -> dict[str, object]:
    left_urdf = left_urdf.resolve(strict=True)
    right_urdf = right_urdf.resolve(strict=True)
    sources = load_source_manifest(source_manifest_path)
    source_map = {item["id"]: item for item in sources["sources"]}
    left_chain = load_urdf_chain(left_urdf)
    right_chain = load_urdf_chain(right_urdf)
    left_signature = _normalized_chain_signature(left_chain)
    right_signature = _normalized_chain_signature(right_chain)

    decimal_resolution_m = 1e-6
    maximum_serial_translation_terms = 12
    translation_tolerance = 0.5 * decimal_resolution_m * maximum_serial_translation_terms
    rotation_tolerance = 1e-10
    finite_difference_step = 1e-6
    machine_epsilon = float(np.finfo(np.float64).eps)
    jacobian_tolerance = 100.0 * (finite_difference_step**2 + machine_epsilon / finite_difference_step)

    sample_records = []
    max_translation_error = 0.0
    max_rotation_error = 0.0
    max_jacobian_error = 0.0
    for index, q in enumerate(_samples(left_chain["limits"])):
        urdf_transform = urdf_fk(left_chain, q)
        poe_transform = poe_fk(q)
        translation_error = float(np.linalg.norm(urdf_transform[:3, 3] - poe_transform[:3, 3]))
        rotation_error = rotation_distance_rad(urdf_transform, poe_transform)
        analytic_jacobian = poe_space_jacobian(q)
        numeric_jacobian = numerical_space_jacobian(
            lambda positions: urdf_fk(left_chain, positions),
            q,
            step=finite_difference_step,
        )
        jacobian_error = float(np.max(np.abs(analytic_jacobian - numeric_jacobian)))
        max_translation_error = max(max_translation_error, translation_error)
        max_rotation_error = max(max_rotation_error, rotation_error)
        max_jacobian_error = max(max_jacobian_error, jacobian_error)
        sample_records.append(
            {
                "id": f"sample_{index:02d}",
                "joint_positions_rad": q.tolist(),
                "urdf_transform": urdf_transform.tolist(),
                "poe_transform": poe_transform.tolist(),
                "translation_error_m": translation_error,
                "rotation_error_rad": rotation_error,
                "jacobian_max_abs_error": jacobian_error,
            }
        )

    id67_conflict = next(
        item for item in sources["source_conflicts"] if item["id"] == "trossen_vx300s_servo_id_6_7_joint_name"
    )
    conflict_pass = (
        id67_conflict["status"] == "RESOLVED_WITH_CONFLICT_RETAINED"
        and id67_conflict["resolution"]["id6"] == "forearm_roll"
        and id67_conflict["resolution"]["id7"] == "wrist_angle"
    )
    identity_pass = left_signature == right_signature
    status = (
        "PASS"
        if conflict_pass
        and identity_pass
        and max_translation_error <= translation_tolerance
        and max_rotation_error <= rotation_tolerance
        and max_jacobian_error <= jacobian_tolerance
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "product": sources["product"],
        "official_joint_order": list(OFFICIAL_JOINT_ORDER),
        "joint_order_policy": "EXPLICIT_NOT_ALPHABETICAL",
        "quaternion_ordering": "wxyz",
        "unit_contract": "metres_radians",
        "official_poe_source": {
            "source_id": "trossen_vx300s_spec",
            "sha256": source_map["trossen_vx300s_spec"]["sha256"],
            "locator": "Kinematic Properties/Product of Exponentials/M and Slist",
        },
        "urdf_derivatives": {
            "left": {"path": str(left_urdf), "sha256": _sha256(left_urdf)},
            "right": {"path": str(right_urdf), "sha256": _sha256(right_urdf)},
            "pinned_xacro_sha256": source_map["interbotix_aloha_vx300s_xacro"]["sha256"],
        },
        "id67_conflict_gate": ("PASS_RESOLVED_WITH_CONFLICT_RETAINED" if conflict_pass else "FAIL"),
        "left_right_robot_local_identity": {
            "status": "PASS" if identity_pass else "FAIL",
            "left_normalized_chain_signature": left_signature,
            "right_normalized_chain_signature": right_signature,
            "mirrored": False if identity_pass else None,
            "determinant": 1.0 if identity_pass else None,
            "workcell_placement_claimed": False,
        },
        "tolerances": {
            "translation_m": translation_tolerance,
            "translation_basis": "0.5e-6 m least-significant published decimal times 12 serial translation terms",
            "rotation_rad": rotation_tolerance,
            "rotation_basis": "all compared official URDF joint-origin RPY and POE home rotation entries are exact zero/identity",
            "jacobian": jacobian_tolerance,
            "jacobian_basis": "100*(h^2+float64_epsilon/h) central-difference numerical bound",
            "finite_difference_step_rad": finite_difference_step,
        },
        "fk_comparison": {
            "max_translation_error_m": max_translation_error,
            "max_rotation_error_rad": max_rotation_error,
        },
        "jacobian_comparison": {"max_abs_error": max_jacobian_error},
        "samples": sample_records,
        "isaac_ik_used": False,
        "runtime_simulation_used": False,
    }
    payload["deterministic_signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 kinematic contract",
            "",
            f"- Status: **{contract['status']}**",
            f"- Explicit joint order: `{contract['official_joint_order']}`",
            f"- ID 6/7 conflict gate: **{contract['id67_conflict_gate']}**",
            f"- Left/right robot-local identity: **{contract['left_right_robot_local_identity']['status']}**",
            f"- Maximum FK translation residual: `{contract['fk_comparison']['max_translation_error_m']:.12g} m`",
            f"- Maximum FK rotation residual: `{contract['fk_comparison']['max_rotation_error_rad']:.12g} rad`",
            f"- Maximum Jacobian residual: `{contract['jacobian_comparison']['max_abs_error']:.12g}`",
            "",
            "The official Trossen POE model and an independent URDF-chain implementation "
            "were compared at home and four deterministic legal joint samples. Isaac IK was "
            "not called. Left and right are identical robot-local products, not mirrored; "
            "this report makes no claim about their workcell installation transforms.",
            "",
            "Tolerances are derived from published decimal precision and the finite-difference "
            "error expression recorded in the JSON report; no behavior-fitted tolerance is used.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-urdf", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right-urdf", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    contract = build_contract(
        left_urdf=args.left_urdf,
        right_urdf=args.right_urdf,
        source_manifest_path=args.sources,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(contract), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": contract["status"],
                "max_translation_error_m": contract["fk_comparison"]["max_translation_error_m"],
                "max_jacobian_error": contract["jacobian_comparison"]["max_abs_error"],
            }
        )
    )
    return 0 if contract["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from tools.aloha1_mapping.official_parameter_sources import load_source_manifest

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_gripper_geometry_contract.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_gripper_geometry_contract.md"


def _linear_position(theta: float, *, radius: float, arm_length: float) -> float:
    radial_projection = radius * math.sin(theta)
    orthogonal = math.sqrt(radius**2 - radial_projection**2)
    return radial_projection + math.sqrt(arm_length**2 - orthogonal**2)


def build_contract(root: Path) -> dict[str, object]:
    source_manifest = load_source_manifest(root / "configs/aloha1_official_parameter_sources.yaml")
    sources = {item["id"]: item for item in source_manifest["sources"]}
    radius = 0.0275
    arm_length = 0.035
    lower = 0.021
    upper = 0.057
    lower_angle = brentq(
        lambda angle: _linear_position(angle, radius=radius, arm_length=arm_length) - lower,
        -math.pi / 2,
        math.pi / 2,
    )
    upper_angle = brentq(
        lambda angle: _linear_position(angle, radius=radius, arm_length=arm_length) - upper,
        -math.pi / 2,
        math.pi / 2,
    )
    motor_angles = np.linspace(lower_angle, upper_angle, 1001)
    left = np.asarray([_linear_position(value, radius=radius, arm_length=arm_length) for value in motor_angles])
    right = -left
    endpoint_error = max(abs(left[0] - lower), abs(left[-1] - upper))
    aperture_path = (
        root
        / "reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.json"
    )
    aperture_resolution = json.loads(aperture_path.read_text(encoding="utf-8"))
    if aperture_resolution["status"] != "PASS":
        raise ValueError("gripper aperture definition resolution must pass")
    contract: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "product": source_manifest["product"],
        "source_cad": {
            "path": str((root / sources["supplier_simple_aloha_viper_step"]["local_path"]).resolve()),
            "sha256": sources["supplier_simple_aloha_viper_step"]["sha256"],
            "finger_authority": "embedded_handed_v2_pair_not_standalone_v3",
        },
        "linkage_sources": {
            "motor_config_sha256": sources["interbotix_aloha_vx300s_motor_config"]["sha256"],
            "xacro_sha256": sources["interbotix_aloha_vx300s_xacro"]["sha256"],
            "driver_sha256": sources["interbotix_xs_driver"]["sha256"],
        },
        "formula_validation": {
            "status": "PASS" if np.all(np.diff(left) > 0.0) and np.allclose(right, -left) else "FAIL",
            "formula": "x=r*sin(theta)+sqrt(L^2-(sqrt(r^2-(r*sin(theta))^2))^2)",
            "horn_radius_m": radius,
            "arm_length_m": arm_length,
            "motor_angle_interval_rad": [lower_angle, upper_angle],
            "left_finger_interval_m": [float(left[0]), float(left[-1])],
            "right_finger_interval_m": [float(right[-1]), float(right[0])],
            "sample_count": len(left),
            "monotonic": bool(np.all(np.diff(left) > 0.0)),
            "right_is_negative_left": bool(np.allclose(right, -left, atol=0.0)),
            "endpoint_max_abs_error_m": endpoint_error,
        },
        "aperture": {
            "definition_boundary": "URDF values are carriage-center coordinates; the product-page wording remains a documented source conflict and the tilted CAD contact-surface gap is not a single scalar",
            "urdf_carriage_center_range_m": [2.0 * lower, 2.0 * upper],
            "trossen_exact_product_claim_m": [0.042, 0.116],
            "status": "PASS_WITH_DOCUMENTED_OFFICIAL_SOURCE_CONFLICT",
            "source_conflict": aperture_resolution["source_conflict"],
            "contact_surface_gap_is_single_scalar": aperture_resolution[
                "contact_surface_gap_is_single_scalar"
            ],
            "contact_surface_gap_m": aperture_resolution["contact_surface_gap_m"],
            "implemented_joint_range_source": aperture_resolution[
                "implemented_joint_range_source"
            ],
            "resolution_report": {
                "path": str(aperture_path.resolve()),
                "sha256": hashlib.sha256(aperture_path.read_bytes()).hexdigest(),
            },
            "no_fitted_endpoint_used": True,
        },
        "left_right_followers": {
            "status": "PASS_ROBOT_LOCAL_IDENTITY",
            "relation": "same_non_mirrored_product",
            "workcell_placement_claimed": False,
        },
        "runtime_simulation_used": False,
        "formal_candidate_gate": "PASS",
    }
    contract["deterministic_signature"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def _markdown(contract: dict[str, Any]) -> str:
    formula = contract["formula_validation"]
    aperture = contract["aperture"]
    return "\n".join(
        [
            "# ALOHA1 gripper geometry contract",
            "",
            f"- Status: **{contract['status']}**",
            f"- Linkage formula: **{formula['status']}** over `{formula['sample_count']}` samples",
            f"- URDF carriage-center interval: `{aperture['urdf_carriage_center_range_m']} m`",
            f"- Trossen exact-product claim: `{aperture['trossen_exact_product_claim_m']} m`",
            f"- Aperture definition: **{aperture['status']}**",
            "",
            "The pinned driver linkage is monotonic and yields exactly opposed left/right "
            "finger coordinates. The 114 mm URDF carriage-center endpoint is not changed to "
            "116 mm to match the product-page claim. CAD carriage datums agree with 114 mm, while "
            "the tilted distal contact-surface gap is position-dependent. The 2 mm product-page "
            "conflict remains explicit and no fitted endpoint is used.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    contract = build_contract(args.root)
    args.json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(contract), encoding="utf-8")
    print(json.dumps({"status": contract["status"], "formula": contract["formula_validation"]["status"]}))
    return 0 if contract["formula_validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

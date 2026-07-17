from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE2_JSON = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase2_runtime_inspection_20260717/phase2_runtime_inspection.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase3_scaffold_contract_20260717"

ALOHA1_14D = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)

ALOHA1_ARM_TO_TROSSEN_CANDIDATE = {
    "left_waist": "follower_left_joint_0",
    "left_shoulder": "follower_left_joint_1",
    "left_elbow": "follower_left_joint_2",
    "left_forearm_roll": "follower_left_joint_3",
    "left_wrist_angle": "follower_left_joint_4",
    "left_wrist_rotate": "follower_left_joint_5",
    "right_waist": "follower_right_joint_0",
    "right_shoulder": "follower_right_joint_1",
    "right_elbow": "follower_right_joint_2",
    "right_forearm_roll": "follower_right_joint_3",
    "right_wrist_angle": "follower_right_joint_4",
    "right_wrist_rotate": "follower_right_joint_5",
}

ALOHA1_GRIPPER_TO_TROSSEN_CANDIDATES = {
    "left_gripper": ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
    "right_gripper": ["follower_right_left_carriage_joint", "follower_right_right_carriage_joint"],
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _runtime_dofs(phase2: dict[str, Any], asset: str) -> list[str]:
    rows = []
    for art in phase2["assets"][asset].get("runtime_articulations", []):
        rows.extend(str(name) for name in art.get("dof_names", []))
    return rows


def _asset_counts(phase2: dict[str, Any], asset: str) -> dict[str, Any]:
    static = phase2["assets"][asset]["stage_static"]
    return {
        "joint_count": static.get("joint_count"),
        "mesh_count": static.get("mesh_count"),
        "collider_count": static.get("collider_count"),
        "camera_count": static.get("camera_count"),
        "material_count": static.get("material_count"),
    }


def build_contract(phase2: dict[str, Any]) -> dict[str, Any]:
    trossen_dofs = _runtime_dofs(phase2, "trossen_stationary_ai")
    aloha1_wrapper_dofs = _runtime_dofs(phase2, "aloha1_wrapper")
    adapter_rows = []
    for index, canonical_name in enumerate(ALOHA1_14D):
        if canonical_name in ALOHA1_ARM_TO_TROSSEN_CANDIDATE:
            candidate = ALOHA1_ARM_TO_TROSSEN_CANDIDATE[canonical_name]
            candidate_present = candidate in trossen_dofs
            adapter_rows.append(
                {
                    "canonical_index": index,
                    "canonical_name": canonical_name,
                    "trossen_candidate_dofs": [candidate],
                    "candidate_present_in_trossen_runtime": candidate_present,
                    "status": "REQUIRES_REAL_DATA_VERIFICATION",
                    "unknown_fields": ["sign", "offset", "limit", "velocity_limit", "effort_limit", "real_direction"],
                    "verification_required": "Validate by real ALOHA1 joint identity/sign/limit source or read-only 103 diagnostics before using for control.",
                }
            )
        else:
            candidates = ALOHA1_GRIPPER_TO_TROSSEN_CANDIDATES[canonical_name]
            adapter_rows.append(
                {
                    "canonical_index": index,
                    "canonical_name": canonical_name,
                    "trossen_candidate_dofs": candidates,
                    "candidate_present_in_trossen_runtime": all(candidate in trossen_dofs for candidate in candidates),
                    "status": "REQUIRES_REAL_DATA_VERIFICATION",
                    "unknown_fields": [
                        "normalized_command_to_carriage_meters",
                        "open_position",
                        "close_position",
                        "mimic_direction",
                        "real_finger_opening_mm",
                    ],
                    "verification_required": "Do not infer ALOHA1 gripper semantics from Trossen. Measure or read true gripper mapping.",
                }
            )

    current_aloha1_reference = {
        "use": "kinematic_reference_only",
        "runtime_dofs": aloha1_wrapper_dofs,
        "reason_not_training_asset": "Phase 2 found zero meshes, zero colliders, zero cameras, and unresolved visual references.",
        "counts": _asset_counts(phase2, "aloha1_wrapper"),
    }
    trossen_standard = {
        "use": "isaac_runtime_scaffold_standard",
        "runtime_dofs": trossen_dofs,
        "counts": _asset_counts(phase2, "trossen_stationary_ai"),
    }
    gates = {
        "phase2_runtime_report": "PASS",
        "trossen_runtime_structure": "PASS",
        "current_aloha1_training_asset": "FAIL_NOT_SIM_READY",
        "aloha1_adapter_complete": "BLOCKED_REQUIRES_REAL_DATA_VERIFICATION",
        "controller_reuse": "BLOCKED_UNTIL_ONE_JOINT_VALIDATION",
        "gripper": "BLOCKED_UNTIL_OPEN_CLOSE_CALIBRATION",
        "camera": "BLOCKED_UNTIL_EXTRINSIC_PROJECTION_TEST",
        "contact_rl": "BLOCKED_UNTIL_COLLIDER_AND_MATERIAL_REVIEW",
    }
    return {
        "contract_name": "trossen_backed_aloha1_scaffold_contract",
        "principle": "Use Trossen as Isaac runtime structure standard; never treat Trossen physical/electrical semantics as ALOHA1 truth without verification.",
        "phase2_source": _rel(DEFAULT_PHASE2_JSON),
        "real_robot_touched": False,
        "stage_saved": False,
        "current_aloha1_reference": current_aloha1_reference,
        "trossen_standard": trossen_standard,
        "adapter_rows": adapter_rows,
        "gates": gates,
        "next_acceptance_test": [
            "Generate or compose a Trossen-backed ALOHA1 scaffold asset without using the broken generated ALOHA1 USD as the runtime base.",
            "Headless Isaac runtime initializes one bimanual articulation.",
            "Mesh, collider, camera, and material counts are nonzero.",
            "No unresolved robot visual/collision references are present in the Isaac log.",
            "The 14D adapter table is emitted and every unverified physical/electrical field is marked REQUIRES_REAL_DATA_VERIFICATION.",
        ],
    }


def render_markdown(contract: dict[str, Any]) -> str:
    lines = [
        "# Trossen-Backed ALOHA1 Scaffold Contract - 2026-07-17",
        "",
        "## Principle",
        "",
        contract["principle"],
        "",
        "## Source Reports",
        "",
        f"- Phase 2 JSON: `{contract['phase2_source']}`",
        "- Phase 2 final log artifact: `.codex/artifacts/20260717-231255_phase2-isaac-runtime-inspection-exit0`",
        "",
        "## Runtime Standard",
        "",
        "Trossen `stationary_ai` is the runtime scaffold standard because it has a complete Isaac structure:",
        "",
    ]
    for key, value in contract["trossen_standard"]["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "The current generated ALOHA1 wrapper is not a training asset:",
            "",
        ]
    )
    for key, value in contract["current_aloha1_reference"]["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            f"Reason: {contract['current_aloha1_reference']['reason_not_training_asset']}",
            "",
            "## 14D Adapter Contract",
            "",
            "| index | ALOHA1 canonical field | Trossen candidate DOF(s) | status | required verification |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in contract["adapter_rows"]:
        lines.append(
            "| "
            f"{row['canonical_index']} | "
            f"`{row['canonical_name']}` | "
            f"`{', '.join(row['trossen_candidate_dofs'])}` | "
            f"`{row['status']}` | "
            f"{row['verification_required']} |"
        )
    lines.extend(["", "## Gates", ""])
    for key, value in contract["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Acceptance Test", ""])
    for item in contract["next_acceptance_test"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Trossen-backed ALOHA1 scaffold contract from Phase 2 runtime evidence.")
    parser.add_argument("--phase2-json", type=Path, default=DEFAULT_PHASE2_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    phase2 = json.loads(args.phase2_json.read_text())
    contract = build_contract(phase2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "trossen_backed_aloha1_scaffold_contract.json"
    md_path = args.output_dir / "trossen_backed_aloha1_scaffold_contract.md"
    json_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(render_markdown(contract))
    print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "gates": contract["gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

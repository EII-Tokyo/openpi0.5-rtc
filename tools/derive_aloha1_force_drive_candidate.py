#!/usr/bin/env python3
"""Close the ALOHA1 force-drive candidate gate without inventing inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.force_drive_candidate import evaluate_force_drive_candidate_gate

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/aloha1_mapping"
DEFAULT_CONVERGENCE = REPORT_DIR / "aloha1_physics_numerical_convergence.json"
DEFAULT_DYNAMICS = REPORT_DIR / "aloha1_dynamics_contract.json"
DEFAULT_GEOMETRY = REPORT_DIR / "aloha1_gripper_geometry_contract.json"
DEFAULT_RESEARCH = REPORT_DIR / "aloha1_model_blocker_deep_research.json"
DEFAULT_JSON = REPORT_DIR / "aloha1_force_drive_candidate.json"
DEFAULT_MARKDOWN = REPORT_DIR / "aloha1_force_drive_candidate.md"
GAIN_TUNER_ROOT = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts"
    / "isaacsim.robot_setup.gain_tuner"
)
GAIN_TUNER_MANIFEST = GAIN_TUNER_ROOT / "config/extension.toml"
GAIN_TUNER_SOURCE = (
    GAIN_TUNER_ROOT
    / "isaacsim/robot_setup/gain_tuner/gains_tuner.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def build_report(
    *,
    convergence_path: Path,
    dynamics_path: Path,
    geometry_path: Path,
    research_path: Path,
) -> dict[str, Any]:
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    research = json.loads(research_path.read_text(encoding="utf-8"))
    manifest_text = GAIN_TUNER_MANIFEST.read_text(encoding="utf-8")
    if 'version = "3.0.6"' not in manifest_text:
        raise RuntimeError("unexpected local Gain Tuner version")

    gate = evaluate_force_drive_candidate_gate(
        convergence_status=str(convergence.get("status", "NOT_RUN")),
        effective_inertia_si=None,
        natural_frequency_hz=None,
        damping_ratio=None,
        continuous_force_limit=None,
        linkage_efficiency=None,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": gate["status"],
        "inputs": {
            "numerical_convergence": _input_record(convergence_path),
            "dynamics_contract": _input_record(dynamics_path),
            "gripper_geometry_contract": _input_record(geometry_path),
            "model_blocker_research": _input_record(research_path),
            "local_gain_tuner_manifest": _input_record(GAIN_TUNER_MANIFEST),
            "local_gain_tuner_source": _input_record(GAIN_TUNER_SOURCE),
        },
        "software_scope": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "gain_tuner_extension": "isaacsim.robot_setup.gain_tuner 3.0.6",
        },
        "numerical_gate": {
            "input_status": convergence.get("status", "NOT_RUN"),
            "selected_frequency_hz": convergence.get("selected_frequency_hz"),
            "selected_solver_position_iterations": convergence.get(
                "selected_solver_position_iterations"
            ),
            "selected_solver_velocity_iterations": convergence.get(
                "selected_solver_velocity_iterations"
            ),
        },
        "physical_inputs": {
            "effective_gripper_mass_at_declared_configuration": None,
            "closed_loop_natural_frequency_hz": None,
            "closed_loop_damping_ratio": None,
            "continuous_force_limit_n": None,
            "loaded_linkage_efficiency": None,
        },
        "gate": gate,
        "official_equation_boundary": {
            "source_classification": research["isaac_drive_mapping"][
                "classification"
            ],
            "gain_tuner_relations": research["isaac_drive_mapping"][
                "gain_tuner_relations"
            ],
            "units": research["isaac_drive_mapping"]["units"],
            "hardware_integer_gain_direct_mapping": "PROHIBITED",
        },
        "dynamics_evidence": {
            "contract_status": dynamics["status"],
            "runtime_control": dynamics["actuator_contract"][
                "gripper_runtime_control"
            ],
            "continuous_estimate_is_measured_thermal_curve": dynamics[
                "actuator_contract"
            ]["continuous_estimate_is_measured_thermal_curve"],
            "stall_torque_used_as_continuous": dynamics["actuator_contract"][
                "stall_torque_used_as_continuous"
            ],
            "hard_blockers": dynamics["hard_blockers"],
        },
        "geometry_evidence": {
            "contract_status": geometry["status"],
            "formal_candidate_gate": geometry["formal_candidate_gate"],
            "effective_dynamic_mass_provided": False,
        },
        "candidate_authored": False,
        "runtime_scan_status": "NOT_RUN_PREREQUISITES_UNSATISFIED",
        "final_or_default_asset_modified": False,
        "promotion_allowed": False,
        "interpretation": (
            "The local Isaac 5.1 Gain Tuner relations define how sourced SI "
            "mass/inertia and response targets map to PD gains, but they do not "
            "supply those physical inputs. Numerical convergence is not established, "
            "and the exact gripper effective mass, closed-loop response targets, "
            "continuous output-force envelope, and loaded linkage efficiency are "
            "not present in the audited sources. Therefore no force-drive USD or "
            "runtime parameter scan is permitted."
        ),
    }
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _markdown(report: dict[str, Any]) -> str:
    missing = "\n".join(
        f"- `{item}`" for item in report["gate"]["missing_inputs"]
    )
    return f"""# ALOHA1 force-drive candidate gate

- Status: **{report['status']}**
- Candidate authored: `{report['candidate_authored']}`
- Runtime scan: **{report['runtime_scan_status']}**
- Final/default asset modified: `{report['final_or_default_asset_modified']}`
- Promotion allowed: `{report['promotion_allowed']}`

## Result

{report['interpretation']}

## Missing evidence

{missing}

The official Gain Tuner equations are retained as the derivation method, not as
a source of missing robot parameters. DYNAMIXEL integer gains and stall torque
are not copied into PhysX SI drive parameters.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence", type=Path, default=DEFAULT_CONVERGENCE)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(
        convergence_path=args.convergence,
        dynamics_path=args.dynamics,
        geometry_path=args.geometry,
        research_path=args.research,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "runtime_scan_status": report["runtime_scan_status"],
                "missing_inputs": report["gate"]["missing_inputs"],
            }
        )
    )
    return 0 if report["status"] != "HARD_BLOCKER" else 2


if __name__ == "__main__":
    raise SystemExit(main())

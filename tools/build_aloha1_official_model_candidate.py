#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.task8_optimization import build_model_first_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_official_model_candidate.json"
DEFAULT_ASSET_DIRECTORY = ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/official_model_contract/1.0"
REPORTS = {
    "source_audit": "aloha1_official_parameter_source_audit.json",
    "parameter_matrix": "aloha1_official_parameter_matrix.json",
    "kinematic_contract": "aloha1_kinematic_contract.json",
    "dynamics_contract": "aloha1_dynamics_contract.json",
    "gripper_geometry_contract": "aloha1_gripper_geometry_contract.json",
    "collider_geometry_contract": "aloha1_collider_geometry_contract.json",
    "runtime_contract": "aloha1_official_model_runtime.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path, asset_directory: Path) -> dict[str, object]:
    gates: dict[str, dict[str, Any]] = {}
    inputs = []
    for gate_id, filename in REPORTS.items():
        path = root / "reports/aloha1_mapping" / filename
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        candidate_status = None
        if gate_id == "parameter_matrix":
            candidate_status = report["formal_parameter_candidate_gate"]["status"]
        elif "formal_candidate_gate" in report:
            candidate_status = report["formal_candidate_gate"]
        gates[gate_id] = {"status": report["status"], "candidate_gate": candidate_status}
        inputs.append(
            {
                "id": gate_id,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "deterministic_signature": report.get("deterministic_signature"),
            }
        )
    gate = build_model_first_gate(gates)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "NOT_BUILT_BLOCKED" if gate["status"] != "PASS" else "READY_TO_AUTHOR",
        "model_first_gate": gate,
        "inputs": inputs,
        "asset_directory": str(asset_directory.resolve()),
        "asset_directory_created_by_this_run": False,
        "final_or_default_asset_modified": False,
        "isaac_runtime_started": False,
        "reason": (
            "Formal candidate authoring is prohibited until every exact-model mathematical and runtime gate passes."
            if gate["status"] != "PASS"
            else "All gates pass; explicit authoring invocation is still required."
        ),
    }
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-directory", type=Path, default=DEFAULT_ASSET_DIRECTORY)
    args = parser.parse_args()
    report = evaluate(ROOT, args.asset_directory)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "asset_created": False}))
    return 0 if report["status"] in {"NOT_BUILT_BLOCKED", "READY_TO_AUTHOR"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

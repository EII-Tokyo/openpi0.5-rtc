#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_aloha1_official_model_candidate import DEFAULT_ASSET_DIRECTORY
from tools.build_aloha1_official_model_candidate import DEFAULT_OUTPUT as CANDIDATE_OUTPUT
from tools.build_aloha1_official_model_candidate import evaluate

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_official_model_runtime.json"
GATE_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task8_model_first_gate.json"


def _signature(record: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    runtime: dict[str, object] = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "reason": "No source-complete isolated candidate exists; Isaac runtime verification is forbidden by the model-first gate.",
        "isaac_sim_started": False,
        "nvidia_mcp_required_at_this_gate": False,
        "stage_loaded": None,
        "final_or_default_asset_modified": False,
    }
    runtime["deterministic_signature"] = _signature(runtime)
    RUNTIME_OUTPUT.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    candidate = evaluate(ROOT, DEFAULT_ASSET_DIRECTORY)
    CANDIDATE_OUTPUT.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate: dict[str, object] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "task8_status": "AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE",
        "candidate_status": candidate["status"],
        "model_first_gate": candidate["model_first_gate"],
        "baseline_inventory_status": "PASS_READ_ONLY_EVIDENCE",
        "candidate_asset_created": False,
        "isaac_runtime_started": False,
        "final_or_default_asset_modified": False,
        "next_actions": [
            "derive a task-local collider acceptance error budget; the all-link numerical certificate is complete and the correct-finger single hull fails its 0.20 mm contact-surface budget",
            "retain the resolved 42-114 mm CAD/URDF carriage-center range and the documented 116 mm product-page source conflict without endpoint fitting",
            "complete the torque-speed-current thermal envelope beyond ROBOTIS' official 12 V 20%-of-stall continuous estimates",
            "derive controller-to-PhysX stiffness, damping and maxForce without copying DYNAMIXEL integer gains",
            "obtain exact finger-bottle-table material properties or physical measurements",
            "derive a numerical timestep/solver error budget",
        ],
    }
    gate["deterministic_signature"] = _signature(gate)
    GATE_OUTPUT.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "task8": gate["task8_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

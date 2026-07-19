from __future__ import annotations

import dataclasses
from typing import Any


DRIVE_GATE = "drive_target_tracking"
RESET_GATE = "reset_distribution"
CAUSALITY_GATE = "closed_loop_causality"
CONTACT_REWARD_GATE = "contact_reward_semantics"
OBSERVATION_GATE = "observation_schema"


@dataclasses.dataclass(frozen=True)
class ReadinessGate:
    name: str
    status: str
    evidence: str
    next_action: str


def build_rl_readiness_report(
    *,
    drive_gate_pass: bool,
    drive_gate_evidence: str,
    reset_gate_pass: bool = False,
    causality_gate_pass: bool = False,
    contact_reward_gate_pass: bool = False,
    observation_gate_pass: bool = False,
) -> dict[str, Any]:
    """Return a conservative RL-readiness report.

    Replay or drive-target tracking can only satisfy the first gate.  The whole
    environment is trainable only after reset, causality, reward/contact, and
    observation gates are also proven.
    """

    gates = [
        ReadinessGate(
            name=DRIVE_GATE,
            status="PASS" if drive_gate_pass else "FAIL",
            evidence=drive_gate_evidence,
            next_action="add_reset_action_observation_reward_api" if drive_gate_pass else "fix_drive_tracking",
        ),
        ReadinessGate(
            name=RESET_GATE,
            status="PASS" if reset_gate_pass else "NOT_EVALUATED",
            evidence="not evaluated by drive-target replay smoke",
            next_action="prove deterministic reset plus randomized task reset distribution",
        ),
        ReadinessGate(
            name=CAUSALITY_GATE,
            status="PASS" if causality_gate_pass else "NOT_EVALUATED",
            evidence="not evaluated by replaying a fixed HDF5 sequence",
            next_action="prove different actions from same reset produce different next states",
        ),
        ReadinessGate(
            name=CONTACT_REWARD_GATE,
            status="PASS" if contact_reward_gate_pass else "NOT_EVALUATED",
            evidence="not evaluated by drive tracking alone",
            next_action="separate target pipe/bottle contact reward from non-target collision penalties",
        ),
        ReadinessGate(
            name=OBSERVATION_GATE,
            status="PASS" if observation_gate_pass else "NOT_EVALUATED",
            evidence="not evaluated by drive tracking alone",
            next_action="freeze observation schema, units, finite checks, and no future-label leakage",
        ),
    ]
    overall_ready = all(gate.status == "PASS" for gate in gates)
    return {
        "overall_rl_training_ready": bool(overall_ready),
        "status": "READY_FOR_RL_TRAINING" if overall_ready else "NOT_READY_FOR_RL_TRAINING",
        "gates": [dataclasses.asdict(gate) for gate in gates],
    }


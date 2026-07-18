# Phase 110 Workcell Contact Policy Negative Gate

## Question

Phase108 passed when `workcell_or_environment` was globally allowed.

Phase109 then showed the BottleUSD object was not touching a calibrated tabletop. It was touching this imported workcell collider:

```text
/scene/worldBody/__22/collisions/__22/__22/extrusion_1220
```

Its parent `/scene/worldBody/__22` is a long, thin frame or rail member.

Phase110 asks:

```text
Can the replay gate explicitly reject this contact class?
```

## Implementation

Added a conservative path-prefix policy:

```text
examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml
```

The policy denies:

```text
/scene/worldBody/__22/** -> denied_frame_rail_collision
/scene/worldBody/table/** -> candidate_table_prim_not_full_measured_table
```

The validator now accepts:

```bash
--workcell-contact-policy examples/aloha_isaac/config/phase110_workcell_contact_policy.yaml
```

and writes a `workcell_contact_policy_gate` block into the metrics JSON.

## Command

```bash
codex-evidence --name aloha-phase110-workcell-contact-policy-negative-gate -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase110_workcell_contact_policy_negative_gate.py
```

## Expected Result

This phase is a negative control.

The correct result is not a PASS. The correct result is:

```text
status = FAILED_GATE
contact_trace_status = FAIL_WORKCELL_CONTACT_POLICY
failure_reasons includes workcell_contact_policy_gate_failed
denied_semantic_classes includes denied_frame_rail_collision
```

That proves the gate no longer accepts every `workcell_or_environment` contact as if it were tabletop support.

## Verified Result

Artifact:

```text
.codex/artifacts/20260719-010930_aloha-phase110-workcell-contact-policy-negative-gate
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase110_workcell_contact_policy_negative_gate_20260719/gripper_passive_contact_metrics.json
```

The run produced the expected negative-control result:

| Field | Value |
| --- | --- |
| `status` | `FAILED_GATE` |
| `overall_pass` | `False` |
| `contact_trace_status` | `FAIL_WORKCELL_CONTACT_POLICY` |
| `failure_reasons` | `contact_trace_gate_failed`, `workcell_contact_policy_gate_failed` |
| `controller_tracking_gate.pass` | `True` |
| `target_limit_gate_ok` | `True` |
| denied semantic class | `denied_frame_rail_collision` |
| denied path | `/scene/worldBody/__22/collisions/__22/__22/extrusion_1220` |

This is the intended outcome: the controller replay still tracks correctly, but the contact semantic is rejected.

## Interpretation

This does not mean the HDF5 replay or BottleUSD asset is useless.

It means the current replay is not yet a final table or pipe validation. Before a future PASS can be trusted, the scene needs an explicit calibrated support/contact semantic:

```text
allowed_tabletop_support
allowed_pipe_contact
allowed_pipe_edge_contact
denied_frame_rail_collision
denied_robot_body_collision
unknown_workcell_collision
```

Phase110 therefore closes the broad-category loophole that existed in Phase108.

# Phase101 Active-Grasp Negative Control

## Purpose

Phase97 is a controller/reference pass, not an active grasp pass. It starts from an already-contacting or contact-candidate HDF5 replay state.

Phase101 fixes that boundary with a negative control:

- reuse the exact Phase97 replay, stage, mapping, gains, object, and strict non-target gate;
- remove `--already-in-contact-setup`;
- add `--require-active-target-contact`;
- require the run to fail because the first target contact was found during `settle`, not during `close`.

If this negative control unexpectedly passes, the validator is too weak and could mislabel an already-contacting replay as a real active grasp.

## Command

```bash
codex-evidence --name aloha-phase101-active-grasp-negative-control -- \
  .venv/bin/python aloha_isaac_replay/scripts/run_phase101_active_grasp_negative_control.py
```

## Expected Result

The wrapper should return success only when the child validator fails in the expected way.

Expected wrapper status:

```text
PASS_EXPECTED_NEGATIVE_CONTROL
```

Expected child validator fields:

| Field | Expected value |
| --- | --- |
| child return code | `3` |
| validator status | `FAILED_GATE` |
| failure reason | `active_target_contact_gate_failed` |
| contact trace status | `FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE` |
| first target contact found phase | `settle` |

## Verified Run

Artifact:

`.codex/artifacts/20260719-001836_aloha-phase101-active-grasp-negative-control`

Report:

`reports/aloha1_isaac_adaptation/phase101_phase97_active_grasp_negative_control_20260719/gripper_passive_contact_metrics.json`

Observed wrapper result:

```text
PASS_EXPECTED_NEGATIVE_CONTROL
```

Observed child validator result:

| Field | Observed value |
| --- | --- |
| child return code | `3` |
| validator status | `FAILED_GATE` |
| contact trace status | `FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE` |
| failure reasons | `contact_trace_gate_failed`, `active_target_contact_gate_failed` |
| first target contact found phase | `settle` |
| observed target contact found phases | `close`, `settle` |

The important detail is the first found target-contact phase. The replay later still has target contact in `close`, but the first target contact is already present in `settle`. Therefore this run must not be accepted as active grasp.

## Why This Matters

The active-grasp claim is stricter than the Phase97 controller claim.

Phase97 says:

```text
The arm and gripper drives can follow this HDF5 replay with bounded error, and the object/finger contact candidate remains physically traceable.
```

An active-grasp gate would say:

```text
The object was not already in target contact before closing; target contact first appeared because the gripper actively closed.
```

These are different claims. Phase101 prevents them from being collapsed into one result.

## Next Positive Gate

The next real milestone is not another negative control. It is a positive active-grasp setup where:

1. the object starts out of target finger contact;
2. arm tracking remains within the same Phase97 controller threshold;
3. the first target contact found event appears during `close`;
4. non-target object contact remains absent or explicitly allowed by category;
5. the table/base and object pose are calibrated enough that the test is not only a local gripper fixture.

Until that positive gate exists, the current validated status is:

```text
ALOHA1 Isaac drive-target replay reference: PASS
ALOHA1 active bottle grasp: not yet proven
```

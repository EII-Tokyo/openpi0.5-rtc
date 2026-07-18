# Phase 126: Pipe Contact Policy Boundary

## Question

The strict workcell policy defaults to `deny` for non-finger contacts. That is correct for unknown workcell collisions, but the bottle-in-pipe task eventually requires contact with the measured pipe placeholder.

Phase126 asks:

```text
Can we allow intended pipe contact without accidentally allowing stale table, frame, rail, or floor collisions?
```

## Change

The workcell contact policy now explicitly allows:

```text
/World/PipePlaceholder
```

with semantic class:

```text
candidate_measured_pipe_contact
```

It still denies the old stale workcell leaves:

```text
/scene/worldBody/__22
/scene/worldBody/table
```

It also explicitly denies the measured table overlay:

```text
/World/Table
```

with semantic class:

```text
measured_table_contact_not_yet_task_valid
```

The policy still uses:

```text
default_decision: deny
```

## Why Table Is Not Allowed Yet

`/World/Table` is not globally allowed in this phase. It is explicitly denied, not just left to the default unknown collision rule.

Reason:

- table contact can mean normal support before grasp;
- but it can also mean the bottle fell or dragged on the table;
- the current Phase115 gate starts with the bottle already in the gripper contact window, so table support is not required for that gate.

Allowing table contact now would make future failures look valid. It should remain denied or unclassified until there is a calibrated table/base/contact phase design.

## Verification

Unit test:

```bash
.venv/bin/pytest -q aloha_isaac_replay/tests/test_workcell_contact_policy.py
```

Result:

```text
4 passed
```

Strict measured-workcell regression:

```text
.codex/artifacts/20260719-023515_aloha-phase115-strict-gate-after-pipe-policy
```

Result:

```text
status: PASS
overall_pass: True
failure_reasons: []
workcell_contact_policy_gate: PASS_WORKCELL_CONTACT_POLICY
```

The Phase115 replay had no workcell contact rows, so this regression only proves that the new allow rule did not break the existing strict pass.

The direct pipe-contact behavior is covered by the unit test.

## Engineering Meaning

This is a semantic policy improvement, not a proof of successful insertion.

It ensures that a future physically meaningful pipe contact will not be rejected as a generic environment collision.

It does not prove:

- the pipe pose is calibrated;
- the contact occurred at the bottle mouth;
- insertion depth is correct;
- the bottle did not collide with the wrong pipe region;
- the trajectory was stable.

Those must be validated by later contact and trajectory gates.

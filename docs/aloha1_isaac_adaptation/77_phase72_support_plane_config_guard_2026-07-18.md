# Phase 72 Support-Plane Config Guard

## Question

Phase 67 required final replay/contact validation to use:

```text
--require-calibrated-table-frame
```

QA review found one remaining confusion path:

1. The user passes a calibrated `--support-plane-config`.
2. The script audits that config and the audit passes.
3. The user also passes command-line support-plane geometry overrides.
4. Isaac replay uses the overridden geometry, not the audited geometry.

That would mean the script audits one table but simulates another table.

## Implementation

Updated script:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

Two guards now run before Isaac startup:

1. Any use of `--support-plane-config` without `--require-calibrated-table-frame` must explicitly opt into diagnostic mode:

```text
--allow-diagnostic-support-plane-config
```

2. If `--require-calibrated-table-frame` is enabled, calibrated replay rejects support-plane geometry overrides:

```text
--support-plane-center
--support-plane-size
--support-plane-size-x
--support-plane-size-y
--support-plane-thickness
```

This keeps diagnostic replay reproducible, but prevents diagnostic or overridden table geometry from being silently promoted into final validation.

## Correct Modes

Diagnostic replay with the Phase63 candidate:

```text
.venv/bin/python aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py \
  --support-plane-config examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml \
  --allow-diagnostic-support-plane-config
```

Final calibrated replay:

```text
.venv/bin/python aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py \
  --support-plane-config local_eval_assets/aloha1_calibration/table_to_base_calibration.yaml \
  --stage-units-in-meters 1.0 \
  --require-calibrated-table-frame
```

Final calibrated replay must not also pass support-plane CLI geometry overrides.

## Validation

Validated locally:

```text
.venv/bin/python -m pytest -q \
  aloha_isaac_replay/tests/test_passive_contact_csv_writer.py \
  aloha_isaac_replay/tests/test_table_frame_candidate_audit.py \
  aloha_isaac_replay/tests/test_table_calibration_readiness.py \
  aloha_isaac_replay/tests/test_calibrated_table_base_overlay.py
```

Result:

```text
23 passed
```

No real robot, `192.168.1.103`, or Isaac runtime action was used.

## Interpretation

This does not solve table/base measurement. It only prevents a class of false-positive validation:

```text
audited table config != simulated support plane
```

The real blocker remains the same:

```text
T_world_table
T_table_left_base
T_table_right_base
```

must come from physical measurement or a trusted explicit geometry source.

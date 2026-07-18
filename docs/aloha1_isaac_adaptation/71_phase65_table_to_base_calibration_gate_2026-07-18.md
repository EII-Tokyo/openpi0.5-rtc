# Phase 65 Table-To-Base Calibration Gate

## Question

Phase 64 proved that the Phase 63 table candidate is explicit but still diagnostic.

Phase 65 asks:

Can the project provide a measured-calibration entry point that refuses to run final replay validation until the table and both ALOHA base transforms are real measured or trusted values?

## Implementation

New calibration template:

```text
examples/aloha_isaac/config/phase65_table_to_base_calibration_template.yaml
```

Updated audit script:

```text
aloha_isaac_replay/scripts/audit_table_frame_candidate.py
```

The audit now checks:

1. `T_world_table`;
2. `T_table_left_base`;
3. `T_table_right_base`;
4. transform status;
5. transform source;
6. 3D translation shape;
7. Isaac/USD quaternion order `[qw, qx, qy, qz]`;
8. quaternion norm close to `1.0`;
9. `T_world_table.translation` matches the support plane top center;
10. derived `T_world_left_base` and `T_world_right_base` when calibration is complete.

The accepted calibrated statuses are:

```text
measured
calibrated
read_from_103
read_from_usd
```

The blocking statuses are:

```text
unknown
not_calibrated
diagnostic_candidate
```

## Command

```text
.venv/bin/python aloha_isaac_replay/scripts/audit_table_frame_candidate.py \
  --config examples/aloha_isaac/config/phase65_table_to_base_calibration_template.yaml \
  --output-dir reports/aloha1_isaac_adaptation/phase65_table_to_base_calibration_gate_20260718
```

Structured outputs:

```text
reports/aloha1_isaac_adaptation/phase65_table_to_base_calibration_gate_20260718/table_frame_static_audit.json
reports/aloha1_isaac_adaptation/phase65_table_to_base_calibration_gate_20260718/table_frame_static_audit.md
```

## Result

The template returns:

```text
BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM
```

This is the expected safe result.

The template contains placeholder transforms, so it must not pass final replay validation.

## What Will Pass

A real calibration file may pass only when all three transforms are measured or read from a trusted source:

```text
T_world_table
T_table_left_base
T_table_right_base
```

Each transform must contain:

```text
source
translation
rotation_quat_wxyz
convention
status
```

The `rotation_quat_wxyz` field must use Isaac/USD order:

```text
[qw, qx, qy, qz]
```

The quaternion norm must be approximately:

```text
1.0
```

## Why This Matters

The project now has three distinct states:

| State | Meaning | Allowed Use |
| --- | --- | --- |
| diagnostic candidate | Useful replay support geometry, but not real workcell truth | local debugging only |
| calibration template | Correct schema, but placeholder values | cannot validate replay |
| measured calibration | Real or trusted table-to-base transforms | can enter replay contact validation |

This prevents a common failure mode:

```text
temporary no-collision table pose
```

being silently promoted into:

```text
real ALOHA1 workcell geometry
```

## Next Gate

Phase 66 should create or ingest a real measured calibration file.

Until then, the correct status remains:

```text
BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM
```

No real robot or `192.168.1.103` action was used in this phase.

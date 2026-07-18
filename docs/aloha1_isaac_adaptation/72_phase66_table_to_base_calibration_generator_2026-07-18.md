# Phase 66 Table-To-Base Calibration Generator

## Question

Phase 65 created a strict calibration gate, but hand-writing the YAML is error-prone.

Phase 66 asks:

Can we provide a small generator that converts measured table/base values into a Phase65-compatible calibration file and immediately verifies it?

## Implementation

New script:

```text
aloha_isaac_replay/scripts/create_table_to_base_calibration.py
```

The script accepts:

1. table top center in Isaac world coordinates;
2. table size;
3. table yaw;
4. left ALOHA base origin in table coordinates;
5. left ALOHA base yaw;
6. right ALOHA base origin in table coordinates;
7. right ALOHA base yaw;
8. source and status.

It writes a Phase65-compatible YAML file and immediately calls:

```text
aloha_isaac_replay/scripts/audit_table_frame_candidate.py
```

If the generated file does not pass the audit, the command fails.

## Example Command

The following command uses example numbers only:

```text
.venv/bin/python aloha_isaac_replay/scripts/create_table_to_base_calibration.py \
  --output /tmp/aloha_phase66_example/calibration.yaml \
  --table-top-center 1.0,2.0,0.5 \
  --left-base=-0.3,0.1,0.0 \
  --right-base=0.3,0.1,0.0 \
  --right-yaw-deg 180.0
```

Smoke-test result:

```text
PASS_TABLE_TO_BASE_CALIBRATION_READY
```

The generated file is not committed because the numbers above are not real measurements.

## Important Semantics

The table top center is expressed in Isaac world coordinates:

```text
T_world_table.translation
```

The left and right base origins are expressed in the table frame:

```text
T_table_left_base.translation
T_table_right_base.translation
```

Yaw angles are converted into Isaac/USD quaternion order:

```text
[qw, qx, qy, qz]
```

The support plane center is derived from the table top center and thickness:

```text
support_plane.center = table_top_center - table_up_axis * table_thickness / 2
```

For the normal case where table yaw is around +Z, this means:

```text
support_plane.center.z = table_top_center.z - table_thickness / 2
```

## What Still Blocks Final Workcell Calibration

The generator does not solve measurement.

It only prevents transcription mistakes after the real values are known.

The missing real-world inputs remain:

```text
T_world_table
T_table_left_base
T_table_right_base
```

These must come from physical measurement, trusted USD, or read-only robot-side diagnostics. They must not be guessed from Phase62/63 diagnostic support geometry.

## Validation

Tests added:

```text
aloha_isaac_replay/tests/test_create_table_to_base_calibration.py
```

Validated cases:

1. generated measured config passes audit;
2. support plane center is derived from table top center;
3. table yaw rotates base translations correctly;
4. generated world-base transforms match expected values.

No real robot or `192.168.1.103` action was used in this phase.

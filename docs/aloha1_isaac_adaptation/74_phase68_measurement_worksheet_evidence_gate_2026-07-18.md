# Phase 68 Measurement Worksheet Evidence Gate

## Question

Phase65 to Phase67 prevented diagnostic table poses from entering final replay, but there was still a loophole:

```text
manual CLI numbers -> status measured -> calibrated config
```

Phase68 closes that loophole.

## Implementation

New measurement worksheet:

```text
examples/aloha_isaac/config/phase68_table_to_base_measurement_worksheet.yaml
```

New converter:

```text
aloha_isaac_replay/scripts/create_table_to_base_calibration_from_worksheet.py
```

Updated evidence-aware files:

```text
aloha_isaac_replay/scripts/audit_table_frame_candidate.py
aloha_isaac_replay/scripts/create_table_to_base_calibration.py
```

## Evidence Rule

A calibrated table-to-base config must include:

```text
calibration_evidence
```

with:

```text
type
path
sha256
real_robot_touched
remote_103_touched
```

The audit verifies:

1. evidence exists;
2. evidence file exists;
3. evidence sha256 matches the file;
4. `real_robot_touched` is `false`;
5. `remote_103_touched` is `false` or `readonly`.

Without evidence, even a config marked as `measured` is blocked.

## Worksheet Required Fields

The worksheet must include measurement metadata:

```text
measurement.source
measurement.status
measurement.measured_at
measurement.measured_by
measurement.units
measurement.coordinate_frame
measurement.tool
measurement.uncertainty_m
measurement.real_robot_touched
measurement.remote_103_touched
```

It must also include geometry:

```text
table.top_center_world_m
table.size_m
table.yaw_deg
left_base.translation_table_m
left_base.yaw_deg
right_base.translation_table_m
right_base.yaw_deg
output.calibration_path
```

For `source: read_from_103`, the worksheet must explicitly set:

```text
measurement.remote_103_touched: readonly
```

## Default Template Result

The default worksheet is intentionally incomplete.

Command:

```text
.venv/bin/python aloha_isaac_replay/scripts/create_table_to_base_calibration_from_worksheet.py \
  --worksheet examples/aloha_isaac/config/phase68_table_to_base_measurement_worksheet.yaml \
  --output-dir reports/aloha1_isaac_adaptation/phase68_table_to_base_measurement_worksheet_20260718
```

Result:

```text
BLOCKED_REQUIRES_MEASUREMENT_FIELDS
```

This is the correct current state.

## Why This Matters

The project now distinguishes:

| Input | Result |
| --- | --- |
| diagnostic Phase63 config | blocked by calibrated replay gate |
| Phase65 placeholder template | blocked by missing calibration |
| measured numbers without evidence | blocked by evidence gate |
| complete worksheet with evidence hash | can generate calibrated config |

This protects later Isaac replay results from being based on undocumented manual numbers.

## Validation

Tests added or updated:

```text
aloha_isaac_replay/tests/test_table_to_base_measurement_worksheet.py
aloha_isaac_replay/tests/test_create_table_to_base_calibration.py
aloha_isaac_replay/tests/test_table_frame_candidate_audit.py
aloha_isaac_replay/tests/test_passive_contact_csv_writer.py
```

Validated:

```text
python py_compile
pytest
default worksheet blocking report
git diff --check
```

No real robot or `192.168.1.103` action was used in this phase.

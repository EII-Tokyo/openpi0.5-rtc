# Phase 64 Table Frame Static Audit

## Question

Phase 63 made the fixed table candidate explicit in YAML, but the candidate still contains diagnostic values.

Phase 64 asks:

Can we add a gate that reads the table-frame candidate without running Isaac and refuses to treat it as complete when the real table-to-robot base transforms are still missing?

## Implementation

New script:

```text
aloha_isaac_replay/scripts/audit_table_frame_candidate.py
```

Default config:

```text
examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml
```

The script computes:

1. table top center in Isaac world coordinates;
2. table top four corners;
3. status of `T_world_table`;
4. status of `T_table_left_base`;
5. status of `T_table_right_base`.

It does not start Isaac Sim and does not touch the real robot.

## Command

```text
.venv/bin/python aloha_isaac_replay/scripts/audit_table_frame_candidate.py \
  --config examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml \
  --output-dir reports/aloha1_isaac_adaptation/phase64_table_frame_static_audit_20260718
```

Structured outputs:

```text
reports/aloha1_isaac_adaptation/phase64_table_frame_static_audit_20260718/table_frame_static_audit.json
reports/aloha1_isaac_adaptation/phase64_table_frame_static_audit_20260718/table_frame_static_audit.md
```

## Result

The audit returns:

```text
BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM
```

This is the expected safe result.

Frame statuses:

| Transform | Status |
| --- | --- |
| `T_world_table` | `diagnostic_candidate` |
| `T_table_left_base` | `not_calibrated` |
| `T_table_right_base` | `not_calibrated` |

Computed table geometry:

| Item | Value |
| --- | --- |
| table top center | `[0.593227851197621, 0.7853100288947757, -0.2971450733686908]` |
| `xmin_ymin` | `[-0.016772148802378983, 0.4728100288947757, -0.2971450733686908]` |
| `xmax_ymin` | `[1.203227851197621, 0.4728100288947757, -0.2971450733686908]` |
| `xmax_ymax` | `[1.203227851197621, 1.0978100288947758, -0.2971450733686908]` |
| `xmin_ymax` | `[-0.016772148802378983, 1.0978100288947758, -0.2971450733686908]` |

## Interpretation

Phase 64 passes as a safety gate because it blocks the unsafe promotion of diagnostic values.

The table candidate is now explicit and replayable, but the real workcell calibration is not complete.

This is the correct current state:

```text
explicit diagnostic candidate exists
real table-to-base transform missing
final workcell pose not complete
```

## Decision

Keep the Phase 63 config as a diagnostic candidate only.

Do not use it as the final workcell truth until both transforms are measured:

```text
T_table_left_base
T_table_right_base
```

## Next Gate

Phase 65 requires physical measurement or a 103 read-only diagnostic source for the table-to-base transforms.

Until that information exists, the implementation should remain blocked at:

```text
BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM
```

This is preferable to continuing with hidden assumptions.

# Phase 63 Configured Table-Frame Candidate

## Question

Phase 62 found a non-interfering fixed table candidate, but the command still passed the table center directly:

```text
--support-plane-center 0.593227851197621 0.7853100288947757 -0.3171450733686908
```

Phase 63 asks:

Can the replay validator consume an explicit configuration artifact for the fixed table candidate, record its provenance, and reproduce the Phase 62 minus-6-cm result without command-line magic numbers?

## Implementation

New diagnostic config:

```text
examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml
```

The config contains:

```text
support_plane.center
support_plane.size
support_plane.provenance
table_frame.T_world_table
table_frame.T_table_left_base
table_frame.T_table_right_base
```

The validator now supports:

```text
--support-plane-config examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml
```

The config intentionally marks unresolved transforms as not calibrated:

```text
T_table_left_base.status = not_calibrated
T_table_right_base.status = not_calibrated
```

This prevents Phase 62 diagnostic values from being silently promoted to real table-to-robot calibration.

## Command Artifact

The first attempt failed before Isaac startup because an unquoted YAML date was parsed as a Python `date` object and could not be serialized into JSON:

```text
.codex/artifacts/20260718-191022_phase63-configured-fixed-table-replay
```

The config was fixed by quoting the date string.

Successful run:

```text
.codex/artifacts/20260718-191058_phase63-configured-fixed-table-replay-rerun
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase63_configured_fixed_table_replay_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| support config | `examples/aloha_isaac/config/phase63_fixed_table_candidate.yaml` |
| support center | `[0.593227851197621, 0.7853100288947757, -0.3171450733686908]` |
| support size | `[1.22, 0.625, 0.04]` |
| object displacement | `0.0788512` |
| max object displacement | `0.1301710` |
| left-arm max error | `0.0202949` |
| gripper max error | `0.0338054` |
| table-object rows | `655` |
| table-finger rows | `0` |
| other table rows | `0` |

The configured run reproduced the Phase 62 minus-6-cm candidate metrics.

## Interpretation

Phase 63 passes as a configuration/provenance gate.

The important improvement is not a new physical table pose. The improvement is that the fixed table candidate is no longer only an ad hoc command-line value. It is now an explicit artifact with source labels:

| Field | Status |
| --- | --- |
| table footprint | user measured |
| table center XY | borrowed from Phase 60 diagnostic placement |
| table center Z | selected from Phase 62 height scan |
| table-to-left-base transform | not calibrated |
| table-to-right-base transform | not calibrated |

This gives later phases a safer input boundary: every replay can now record whether it used a diagnostic table candidate or a calibrated physical transform.

## Decision

Keep `phase63_fixed_table_candidate.yaml` as the current diagnostic fixed-table candidate.

Do not rename it to a real workcell calibration file until the physical transforms are measured:

```text
T_world_table
T_table_left_base
T_table_right_base
```

## Next Gate

Phase 64 should stop treating the diagnostic candidate as enough for workcell truth.

The next minimum gate should be a static frame audit that reads a table calibration YAML and reports:

1. table top center and four corners in Isaac world coordinates;
2. left and right base positions in table coordinates;
3. initial fingertip proxy boxes relative to the table top;
4. whether the table box overlaps any fingertip proxy before simulation;
5. whether every transform is `measured`, `read_from_103`, `diagnostic`, or `unknown`.

If `T_table_left_base` or `T_table_right_base` remains `not_calibrated`, the gate should report:

```text
BLOCKED_REQUIRES_MEASURED_TABLE_TO_BASE_TRANSFORM
```

instead of silently falling back to Phase 62 borrowed XY.

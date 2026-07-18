# Phase 71 Table Calibration Readiness Check

## Purpose

Phase 71 adds a repeatable readiness check for the ALOHA table-to-base calibration chain.

The goal is to avoid relying on memory or conversation state when deciding whether the Isaac workcell can move from measurement to calibrated overlay and replay validation.

## Entry point

```bash
.venv/bin/python aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py
```

Default inputs:

```text
worksheet: examples/aloha_isaac/config/phase68_table_to_base_measurement_worksheet.yaml
calibration: local_eval_assets/aloha1_calibration/table_to_base_calibration.yaml
```

Default outputs:

```text
reports/aloha1_isaac_adaptation/phase71_table_calibration_readiness_20260718/table_calibration_readiness.json
reports/aloha1_isaac_adaptation/phase71_table_calibration_readiness_20260718/table_calibration_readiness.md
```

## Default behavior is read-only

By default, the readiness script:

1. checks whether the canonical calibration YAML exists;
2. statically checks the worksheet for missing fields;
3. reports the next required action;
4. does not start Isaac Sim;
5. does not touch `192.168.1.103`;
6. does not touch the real robot;
7. does not generate calibration or overlay files unless explicitly requested.

This keeps the check safe to run during unattended progress loops.

## Explicit generation mode

Only when the worksheet is complete should calibration be generated:

```bash
.venv/bin/python aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py \
  --try-generate-calibration
```

Overlay generation is also explicit:

```bash
.venv/bin/python aloha_isaac_replay/scripts/summarize_table_calibration_readiness.py \
  --try-generate-calibration \
  --try-generate-overlay
```

## Current expected status

Until the real table/base measurement worksheet is filled, the expected status is:

```text
BLOCKED_REQUIRES_TABLE_BASE_MEASUREMENT
```

That is not a software failure. It means the code is correctly refusing to proceed without the physical table-to-base transforms.

## Why this matters

Earlier phases showed that diagnostic support planes and root-level collider shortcuts can make Isaac tests look active while using the wrong physical assumptions.

This readiness check makes the final gating explicit:

```text
real measurement worksheet
-> audited calibration YAML
-> calibrated overlay manifest
-> visual Isaac review
-> contact/replay validation
```

No later phase should bypass this sequence for the real workcell.

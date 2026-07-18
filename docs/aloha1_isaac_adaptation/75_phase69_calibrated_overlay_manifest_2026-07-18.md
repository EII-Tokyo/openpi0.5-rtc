# Phase 69 Calibrated Overlay Manifest

## Purpose

Phase 69 turns a Phase 68 audited table-to-base calibration YAML into a simulation-only USD overlay and a replay command manifest.

It does not measure anything. It does not touch `192.168.1.103`. It does not start Isaac Sim. It only prepares a reviewable stage layer after the calibration audit has already passed.

## Hard gate

The entry point is:

```bash
.venv/bin/python aloha_isaac_replay/scripts/create_calibrated_table_base_overlay.py \
  --calibration <calibration.yaml> \
  --output-dir local_eval_assets/aloha1_calibrated_table_base_overlay
```

The script first runs `audit_table_frame()` on the calibration YAML.

If the audit status is not:

```text
PASS_TABLE_TO_BASE_CALIBRATION_READY
```

then the script returns:

```text
BLOCKED_CALIBRATION_AUDIT_NOT_READY
```

and does not write an overlay USD.

## What the overlay uses

The overlay uses only the audited output:

```text
world_base_transforms.T_world_left_base
world_base_transforms.T_world_right_base
```

Those transforms are authored onto the configured target prims, defaulting to the user-confirmed GUI stage scopes:

```text
/scene/left_base_link
/scene/right_base_link
```

This avoids copying raw worksheet values or ad-hoc command-line numbers into an Isaac stage.

## Outputs

When the gate passes, the output directory contains:

```text
aloha1_calibrated_table_base_overlay.usda
replay_command_manifest.json
calibrated_overlay_summary.json
calibrated_overlay_summary.md
```

The manifest records:

- base USD;
- calibration YAML;
- target prims;
- audited world-base transforms;
- calibration evidence hash;
- open command;
- simulation-only safety flags.

The command is intentionally recorded but not executed automatically.

## Why this is separate from replay

Phase 69 is a review boundary. The expected workflow is:

1. generate a calibrated YAML from a completed measurement worksheet;
2. audit it;
3. generate this overlay and manifest;
4. open Isaac for visual inspection;
5. only then run replay/contact validation.

This prevents a failed or incomplete measurement worksheet from silently changing the ALOHA scene used for replay.

# ALOHA1 Native Physics Wrapper Candidate

This directory contains thin USD wrapper stages around the generated ALOHA1 physics layers.

The source importer output under `../original_stationary_aloha/` remains unchanged. These wrappers are a candidate asset entry point for validation, replay, and future controller work.

## Files

- `aloha1_left.usda`: left ALOHA1 physics-layer wrapper.
- `aloha1_right.usda`: right ALOHA1 physics-layer wrapper.
- `manifest.json`: generated provenance and validation summary.

## Known Limits

- This is not yet a final production robot asset.
- It intentionally preserves the generated physics layers instead of flattening or rewriting them.
- Some visual reference warnings may still appear in Isaac logs; runtime articulation validation is the current acceptance gate.
- The next required gates are DOF limits/drives, pose replay, collision/contact behavior, and controller integration.

## Current Authoring Gate

| Asset | Source exists | Wrapper written | Relative sublayer | Gate |
| --- | --- | --- | --- | --- |
| left | True | True | True | PASS |
| right | True | True | True | PASS |

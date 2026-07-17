# Phase 1 Asset Comparison Result - 2026-07-17

## Report

- JSON: `reports/aloha1_isaac_adaptation/phase1_asset_comparison_20260717/phase1_asset_comparison.json`
- Markdown: `reports/aloha1_isaac_adaptation/phase1_asset_comparison_20260717/phase1_asset_comparison.md`
- Script: `scripts/compare_aloha1_isaac_assets_phase1.py`

## Result

Phase 1 status:

```text
PASS_WITH_BLOCKED_RUNTIME_FIELDS
```

This means the read-only static comparison ran and produced useful evidence, but it did not prove runtime USD/articulation facts.

## Confirmed By This Phase

- The ALOHA1-style generated asset, Trossen `stationary_ai.usd`, and Menagerie ALOHA MJCF are distinct sources.
- Trossen / Menagerie remain references, not ALOHA1 truth.
- ALOHA1 import-report joint names differ from Trossen demo index assumptions.
- Trossen Stationary AI demo uses interleaved arm DOF indices and gripper values in meters.
- Menagerie provides ALOHA2/MJCF joint, camera, mesh, and geom information.
- The current ALOHA1 import report shows `collision_count=0` and `mesh_count=0` for side reports, so contact/RL must stay blocked until runtime collider inspection is complete.

## Still Blocked

These fields require USD/PXR/Isaac runtime inspection:

- composed USD prim tree for binary crate USD;
- runtime articulation DOF order for Trossen `stationary_ai`;
- collider shapes and contact material for both USD assets;
- camera prim world transforms.

## Next Safe Step

Run a read-only Isaac runtime/articulation inspection for the selected USD assets.

Before modifying any Isaac code, USD stage, scene-generation script, physics setup, GUI control, or runtime behavior, use the NVIDIA official Isaac MCP as required by `docs/agents/isaac_mcp_toolchain.md`.


# Phase 6 Reference FK Smoke Test - 2026-07-17

## Result

The repository already contains a usable trusted ALOHA1 FK source for offline
comparison:

```text
Pinocchio FK from archived 103 robot_description URDF
```

This source is independent of the broken current ALOHA1 Isaac scene.

The smoke test compared Pinocchio FK against Isaac FK for the generated
original VX300S side assets on one real HDF5 episode and 20 frames.

Result:

```text
status = PASS
left_position_max_m = 3.001506e-7
right_position_max_m = 2.581759e-7
left_orientation_max_deg = 3.448465e-5
right_orientation_max_deg = 2.263513e-5
```

This is strong evidence that the archived URDF and generated VX300S USD agree
with each other for FK.

## Evidence

- Script: `aloha_isaac_replay/scripts/compare_aloha_fk.py`
- Output directory: `reports/aloha1_isaac_adaptation/phase6_reference_fk_smoke_20260717/`
- Metrics: `reports/aloha1_isaac_adaptation/phase6_reference_fk_smoke_20260717/fk_metrics.json`
- Bounded log artifact: `.codex/artifacts/20260717-234506_phase6-reference-fk-smoke`
- Supporting tests:
  - `aloha_isaac_replay/tests/test_reference_fk_independence.py`
  - `aloha_isaac_replay/tests/test_fk_frame_alignment.py`
  - `aloha_isaac_replay/tests/test_original_vx300s_asset_identity.py`
  - `aloha_isaac_replay/tests/test_arm_only_mapping_coverage.py`

The smoke test was offline/headless Isaac only:

```text
real_robot_touched = false
stage_saved = false
controller/replay not started
```

## What This Proves

It proves:

```text
archived ALOHA1 robot_description URDF -> Pinocchio FK is usable
generated original VX300S Isaac side assets agree with that URDF FK
HDF5 observations/qpos can drive the FK smoke test
```

It does **not** prove:

```text
Trossen-backed scaffold FK matches ALOHA1
ALOHA1-to-Trossen sign/offset is solved
gripper carriage mapping is solved
controller replay is safe
contact/material dynamics are correct
```

## Important Warning

The Isaac smoke log still contains unresolved visual reference warnings for the
generated original VX300S USD.

These warnings did not break the FK body transform comparison, but they confirm
that this generated USD should be treated as an FK reference/smoke asset, not as
the final Isaac runtime scaffold.

The final runtime scaffold remains:

```text
Trossen-backed stationary_ai scaffold
```

## Decision

Use the archived 103 URDF + Pinocchio FK as the ALOHA1 FK truth source for the
next offline comparison.

Do not use the broken current ALOHA1 Isaac scene as truth.

## Next Gate

The next gate should compare:

```text
ALOHA1 Pinocchio FK from HDF5 qpos
vs
Trossen-backed scaffold FK after candidate q_isaac = sign * q_aloha + offset
```

This should output candidate ranking, not final controller mapping.

The gate should remain blocked if:

```text
left_forearm_roll is unexplained
any arm DOF has only limit-fit evidence
Trossen link frame cannot be matched to ALOHA1 ee_gripper_link
```

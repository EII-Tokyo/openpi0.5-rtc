# Phase 19: ALOHA1 Native Physics Wrapper Candidate

## Goal

Promote the Phase 17/18 diagnostic ALOHA1 physics-layer wrapper into a tracked project asset candidate, without overwriting the original importer output and without treating the Trossen ALOHA2 framework as ALOHA1 joint truth.

## NVIDIA Isaac Guidance Used

Before modifying the Isaac asset generation path, the official NVIDIA Isaac MCP was queried for:

- asset structure;
- Omniverse/USD composition;
- Isaac Sim units and coordinate conventions.

The relevant constraints are:

- source imported assets should remain unchanged;
- final or candidate assets can be lightweight USD compositions over source layers;
- Isaac Sim uses meters and Z-up;
- a visible mesh alone is not enough; physics and articulation validation remain required.

## New Tracked Candidate Asset

The new candidate directory is:

```text
assets/isaac/aloha1_native_physics_wrapper/
```

It contains:

```text
aloha1_left.usda
aloha1_right.usda
manifest.json
README.md
```

The wrapper files are intentionally small. They only set a default prim and sublayer the useful generated ALOHA1 physics layer:

```text
assets/isaac/original_stationary_aloha/generated/configuration/vx300s_left_physics.usd
assets/isaac/original_stationary_aloha/generated/configuration/vx300s_right_physics.usd
```

This keeps the imported source asset read-only and makes the candidate entry point explicit.

## Why No Dual Wrapper In The Candidate Directory

Phase 17 already proved a dual diagnostic composition can expose both arms statically. During Phase 19, the dual composition was not promoted as a formal candidate because the next runtime validation path references left and right arms separately into `/World/left` and `/World/right`.

The candidate asset therefore promotes only the two units that are actually validated by the current runtime articulation script.

## Authoring Gate

The promotion script verifies:

| Gate | Result |
| --- | --- |
| left physics source exists | PASS |
| right physics source exists | PASS |
| left wrapper written | PASS |
| right wrapper written | PASS |
| sublayers are relative paths | PASS |
| source importer output overwritten | NO |

Command:

```bash
python3 aloha_isaac_replay/scripts/promote_aloha1_native_physics_wrapper_asset.py
```

Result:

```text
overall_authoring_gate = true
```

## Runtime Articulation Gate

The candidate wrappers were then passed to the Phase 18 runtime articulation validator:

```bash
.venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_physics_wrapper_runtime.py \
  --left-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda \
  --right-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda \
  --output-dir reports/aloha1_isaac_adaptation/phase19_native_wrapper_candidate_20260718
```

The validator wrote:

```text
status = PASS
overall_pass = true
```

Runtime result:

| Side | Articulation root | DOFs | Bodies | Core DOFs | EE candidate |
| --- | --- | ---: | ---: | --- | --- |
| left | `/World/left/root_joint` | 9 | 14 | PASS | PASS |
| right | `/World/right/root_joint` | 9 | 14 | PASS | PASS |

The DOF names are:

```text
waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper, left_finger, right_finger
```

## Important Runtime Note

The final `codex-evidence` wrapper process was manually terminated after the validator had already written `overall_pass = true`. The termination was only to stop Isaac shutdown from hanging. It did not affect the generated JSON report.

Evidence:

```text
reports/aloha1_isaac_adaptation/phase19_native_wrapper_candidate_20260718/physics_wrapper_runtime_articulation.json
```

## Known Warning

Isaac still logs unresolved visual reference warnings for several `/visuals/...` prim references. This warning existed in Phase 17/18 as well.

Current interpretation:

- the candidate is usable for runtime articulation investigation;
- the candidate is not yet a final clean visual asset;
- the next cleanup phase should repair or replace the generated visual-reference composition.

Do not hide this warning. It is still a real asset-quality issue.

## Next Gates

The next ALOHA1-native Isaac milestones are:

1. Validate joint limits, drive stiffness, damping, and effort/velocity values.
2. Replay a real ALOHA1 qpos trajectory into the candidate asset.
3. Verify gripper opening semantics and finger motion.
4. Repair visual reference warnings or generate a clean visual layer.
5. Add stable workcell/table/pipe/bottle assets around this candidate robot.

## Files Added

```text
aloha_isaac_replay/scripts/promote_aloha1_native_physics_wrapper_asset.py
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda
assets/isaac/aloha1_native_physics_wrapper/manifest.json
assets/isaac/aloha1_native_physics_wrapper/README.md
```

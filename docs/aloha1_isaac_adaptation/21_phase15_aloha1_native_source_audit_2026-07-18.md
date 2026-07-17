# Phase 15 ALOHA1 Native Asset Source Audit

## Question

After Phase 14, the Trossen-backed scalar sign/offset mapping is no longer a valid route for controlling an ALOHA1 robot in Isaac Sim. The next question is:

Can the existing project-local ALOHA1 URDF/USD assets become the source of truth for an ALOHA1-native rebuild?

## Method

I added a static source audit script:

```bash
python3 aloha_isaac_replay/scripts/audit_aloha1_native_asset_sources.py
```

The script reads existing project-local assets only. It does not start Isaac Sim and does not modify USD files.

It audits:

- `assets/isaac/original_stationary_aloha`
- `assets/isaac/original_stationary_aloha_dynamic`
- `assets/isaac/original_stationary_aloha_arm_only`

For each variant, it checks:

- import report metrics;
- generated USD file presence and size;
- URDF link and joint counts;
- URDF visual and collision mesh references;
- whether referenced mesh files resolve locally;
- whether the generated USD has Mesh prims, CollisionAPI prims, rigid bodies, joints, and articulation roots.

## Evidence

Generated report:

- JSON: `reports/aloha1_isaac_adaptation/phase15_aloha1_native_source_audit_20260718/aloha1_native_asset_source_audit.json`
- Markdown: `reports/aloha1_isaac_adaptation/phase15_aloha1_native_source_audit_20260718/aloha1_native_asset_source_audit.md`

Verification artifact:

- `.codex/artifacts/20260718-002122_phase15-aloha1-native-asset-source-audit`

## Result

| Variant | Mesh prims | Collision prims | Rigid bodies | Joints | Articulation roots | Controller ready |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `assets/isaac/original_stationary_aloha` | 0 | 0 | 28 | 28 | 2 | BLOCKED |
| `assets/isaac/original_stationary_aloha_dynamic` | 0 | 0 | 28 | 28 | 2 | BLOCKED |
| `assets/isaac/original_stationary_aloha_arm_only` | 0 | 0 | 14 | 14 | 2 | BLOCKED |

The resolved URDFs are not empty:

| URDF type | Links | Joints | Mesh refs | Missing mesh refs |
| --- | ---: | ---: | ---: | ---: |
| full resolved left/right | 14 | 13 | 22 each | 0 |
| arm-only resolved left/right | 7 | 6 | 14 each | 0 |

This means the local ALOHA1 URDF source path contains useful robot semantics and local mesh references. The failure is downstream: the current generated USDs preserve an articulation-like skeleton, but they lose all visual Mesh prims and all CollisionAPI prims.

## Interpretation

The current ALOHA1 generated USDs are not controller-ready Isaac robot assets.

They have enough structure to show that the URDF importer created rigid bodies and joints, but they do not have the visual/collision bodies required for:

- visual validation in the viewport;
- contact with bottle, pipe, table, or gripper;
- Grasp Editor validation;
- physically meaningful replay;
- Isaac asset validation for robot/physics readiness.

This is different from saying "ALOHA1 source is unusable." The better conclusion is:

1. ALOHA1 URDF joint semantics should become the kinematic source of truth.
2. The local Interbotix mesh package should become the visual/collision source of truth.
3. The current generated USD output is not acceptable and must be regenerated or repaired.
4. Trossen should be reused for engineering patterns, not as joint-chain truth.

Phase 16 refines this conclusion: generated `configuration/*_base.usd` and `configuration/*_physics.usd` layers do contain Mesh prims, and the physics layer contains CollisionAPI prims. The blocker is that the composed wrapper USD does not compose those visual/collision prims correctly. Therefore the next repair target is USD layer/reference composition, not the original STL mesh package.

## Decision

Do not continue controller work on the current generated ALOHA1 USDs.

The next implementation phase should be ALOHA1-native import repair:

1. Repair the URDF-to-USD import path so generated USD has nonzero Mesh prims.
2. Ensure collision geometry is imported or generated from visual geometry.
3. Validate Robot and Physics asset rules in Isaac Sim.
4. Verify DOF names, order, limits, and signs against real ALOHA1 qpos.
5. Only then run controller replay and grasp tests.

## Why This Changes The Strategy

Previous phases tried to use the Trossen `stationary_ai` runtime asset as a strong Isaac Sim base and map ALOHA1 qpos into it. That route looked attractive because the Trossen asset has a working Isaac structure.

Phase 13 and Phase 14 showed the weakness:

- similar-looking arms do not guarantee matching joint frames;
- scalar sign/offset fitting cannot fix mismatched joint axes and terminal orientation;
- the best orientation-aware candidate still failed orientation holdout and violated ALOHA1 joint-limit semantics.

Phase 15 identifies the replacement route:

- keep Trossen as the reference for USD organization, drive tuning, validation style, and Isaac Lab integration;
- rebuild the robot asset around ALOHA1's own URDF and mesh package.

## Next Acceptance Gates

The next phase should not be considered successful unless all of these are true:

- generated ALOHA1 USD has nonzero Mesh prims;
- generated ALOHA1 USD has nonzero CollisionAPI prims;
- both arms initialize as articulations;
- active DOF names match ALOHA1 control semantics;
- home/sleep qpos values are inside USD joint limits;
- a short real qpos replay produces smooth motion without forced Trossen remapping;
- bottle/table/pipe contact tests can run with collision enabled.

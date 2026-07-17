# Phase 17 Physics Layer Wrapper

## Question

Phase 16 showed that the generated ALOHA1 `*_base.usd` and `*_physics.usd` layers contain useful Mesh, CollisionAPI, RigidBodyAPI, and joint data, but the original importer wrapper composes into a stage with no Mesh or CollisionAPI prims.

The question for this phase:

Can we build a minimal non-destructive wrapper that composes the useful physics layers directly?

## Method

I added:

```bash
python3 aloha_isaac_replay/scripts/build_aloha1_physics_layer_wrapper.py
```

The script does not overwrite any original generated asset.

It creates diagnostic wrappers under:

```text
reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718/
```

It directly sublayers:

```text
assets/isaac/original_stationary_aloha/generated/configuration/vx300s_left_physics.usd
assets/isaac/original_stationary_aloha/generated/configuration/vx300s_right_physics.usd
```

## Evidence

Generated report:

- JSON: `reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718/physics_layer_wrapper_report.json`
- Markdown: `reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718/physics_layer_wrapper_report.md`

Verification artifact:

- `.codex/artifacts/20260718-002637_phase17-aloha1-physics-layer-wrapper`

## Result

| Asset | Mesh prims | Collision prims | Rigid bodies | Joints | Articulation roots | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| left wrapper | 32 | 11 | 14 | 14 | 1 | PASS |
| right wrapper | 32 | 11 | 14 | 14 | 1 | PASS |
| combined wrapper | 54 | 22 | 28 | 28 | 2 | PASS |

This confirms the current ALOHA1 import chain is not hopeless. The generated physics layers contain a usable robot body structure. The broken part is the default importer wrapper/layer composition.

## Important Caveat

The wrapper still emits unresolved-reference warnings from visual scopes. That means this is not yet a final clean production asset.

However, it is good enough as a diagnostic proof:

- Mesh prims are visible in composed stage traversal.
- CollisionAPI prims are visible.
- RigidBodyAPI prims are visible.
- Joint prims are visible.
- Articulation roots are visible.

The next phase should clean layer composition rather than continue Trossen joint remapping.

## Decision

The ALOHA1-native rebuild should proceed from the generated physics layer, not from the original top-level generated wrapper.

Trossen remains useful for:

- USD organization patterns;
- drive and damping strategy;
- Isaac Lab task structure;
- validation workflow.

Trossen should not be used as the ALOHA1 joint-chain source of truth.

## Next Gates

The next implementation phase should:

1. Create a clean production wrapper under a proper project asset directory, not `reports/`.
2. Remove or resolve remaining unresolved-reference warnings.
3. Set a deterministic default prim for the combined dual-arm stage.
4. Inspect DOF names and limits from the repaired wrapper.
5. Compare repaired wrapper DOF names/order against real ALOHA1 qpos.
6. Run a small qpos replay smoke test.
7. Only after replay passes, connect workcell table, pipe, bottle, and grasp tests.

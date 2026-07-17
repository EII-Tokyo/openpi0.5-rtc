# Phase 27: Runtime Collision Composition

## Question

Phase 25 had a collision-disable probe that returned:

```text
disabled_collision_prims = 0
```

That was ambiguous. It could mean:

1. there were no runtime collision prims under `/World/left` and `/World/right`; or
2. the helper scanned the wrong paths and missed composed collision prims.

This phase directly counts `UsdPhysics.CollisionAPI` prims after runtime `add_reference_to_stage`.

## Probe

The probe loads three entry points into an empty Isaac stage:

| Case | left asset | right asset |
| --- | --- | --- |
| wrapper | `assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda` | `assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda` |
| interface | `assets/isaac/original_stationary_aloha/generated/vx300s_left.usd` | `assets/isaac/original_stationary_aloha/generated/vx300s_right.usd` |
| physics layer direct | `assets/isaac/original_stationary_aloha/generated/configuration/vx300s_left_physics.usd` | `assets/isaac/original_stationary_aloha/generated/configuration/vx300s_right_physics.usd` |

It then traverses the stage and counts prims with `UsdPhysics.CollisionAPI`.

## Result

```text
wrapper collision_count = 0
interface collision_count = 0
physics_layer_direct collision_count = 0
```

Report:

```text
reports/aloha1_isaac_adaptation/phase27_runtime_collision_composition_20260718/runtime_collision_composition.json
```

Artifact:

```text
.codex/artifacts/20260718-012821_phase27-runtime-collision-composition-count-file
```

## Interpretation

The current runtime ALOHA1 articulation has no active composed collision prims under the loaded `/World` reference paths.

Therefore:

- the earlier `disabled_collision_prims = 0` was not simply a bad scanner;
- collision contact is not the cause of the Phase 25/26 joint drift;
- the current asset is also not ready for bottle grasp/contact, because collision geometry is not present in the runtime stage where the articulation is loaded.

This is separate from static source-layer inspection, where `/colliders` exists as a root-level scope in the generated physics/base layers. Those root-level collider scopes are not composing into the runtime referenced robot in the current entry points.

## Important USD Warning

The runtime logs still show unresolved visual reference warnings such as:

```text
Unresolved reference prim path ... </visuals/puppet_left_fingers_link>
```

This means the generated ALOHA1 USD package is not a clean runtime composition. It can expose an articulation, but visual/collider side references are not composing correctly through the current reference entry points.

## Decision

The dynamic drift investigation should now deprioritize collision contact and focus on:

1. mass and inertia authored on the ALOHA1 links;
2. drive gains, max force, damping, and velocity limits;
3. joint axes and local frame consistency;
4. unresolved USD reference composition;
5. comparison against a known-working Trossen Stationary AI Isaac asset.

Do not proceed to grasp/contact simulation until collision composition is fixed and verified.

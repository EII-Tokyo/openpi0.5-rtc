# Phase 29: DefaultPrim Composition Trap

## Question

Phase 28 showed:

```text
ALOHA1 native source stage:
  static collision prims = 11
  static mass APIs = 14
```

But Phase 27 showed:

```text
runtime /World reference path:
  collision prims = 0
```

This phase explains the apparent contradiction.

## Evidence

The current ALOHA1 wrapper is:

```text
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
```

Its content is:

```text
#usda 1.0
(
    defaultPrim = "puppet_left_vx300s"
    subLayers = [
        @../original_stationary_aloha/generated/configuration/vx300s_left_physics.usd@
    ]
)
```

Phase 28 inspected the source stage directly and found static collider paths such as:

```text
/colliders/puppet_left_base_link/base/node_STL_BINARY_
/colliders/puppet_left_shoulder_link/shoulder/node_STL_BINARY_
...
```

But the default prim is:

```text
/puppet_left_vx300s
```

Therefore `/colliders` and `/visuals` are root-level siblings of the default prim, not children of the default prim.

## Why Runtime Loses The Colliders

When Isaac code references:

```text
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
```

under:

```text
/World/left
```

USD normally composes the referenced file's default prim under `/World/left`.

That means this comes in:

```text
/puppet_left_vx300s
```

as:

```text
/World/left/...
```

But these root-level siblings do not come in:

```text
/visuals
/colliders
/materials or Looks-like scopes
```

This matches the runtime warnings:

```text
Unresolved reference prim path ... </visuals/puppet_left_fingers_link>
```

The default prim can expose the articulation, so `SingleArticulation` can initialize. But the visual/collider data that lives outside the default prim is not part of the referenced subtree.

## Practical Meaning

The current ALOHA1 asset is in a dangerous middle state:

```text
source file contains some physics data
        ↓
defaultPrim reference only brings in the robot prim subtree
        ↓
root-level visual/collider scopes are left behind
        ↓
runtime articulation exists, but composition is incomplete
```

This is why "the robot initializes" and "the robot is physically ready" are not the same thing.

## What Not To Do

Do not simply copy Trossen gains into this asset yet.

Reasons:

1. The runtime stage still lacks composed collision geometry.
2. The source asset uses `meters_per_unit = 0.01`.
3. The source asset uses `up_axis = Y`.
4. The active drive damping is `0`.
5. The reference composition has unresolved visual paths.

Changing gains before fixing composition would mix multiple failure modes and make the next failure harder to interpret.

## Candidate Repair Directions

### Option A: Rebuild ALOHA1 Into Isaac Asset Structure

Follow the Isaac/Trossen asset pattern:

```text
aloha1_left.usd
configuration/
  aloha1_left_base.usd
  aloha1_left_physics.usd
  aloha1_left_robot.usd
  aloha1_left_sensor.usd
```

The final interface file should have a default prim whose subtree contains everything needed at runtime:

```text
/aloha1_left
  links
  joints
  visuals
  colliders
  materials
```

This is the cleanest long-term path.

### Option B: Diagnostic Composition Wrapper

Build a temporary diagnostic wrapper that references or copies the root-level `/visuals` and `/colliders` scopes into the runtime subtree.

This is useful only as a gate:

```text
can runtime /World/left compose colliders and visuals?
```

It should not be treated as the final production asset unless it also passes joint hold and replay gates.

### Option C: Sublayer The Whole Asset Into The Runtime Stage

Instead of referencing only the default prim, add the whole source USD as a sublayer in a diagnostic stage.

This can prove the composition hypothesis, but it is not a clean reusable robot asset because root-level prims from left/right can collide in names or scope organization.

## Next Gate

The next gate should be a diagnostic composition wrapper:

1. load the current ALOHA1 source stage;
2. compose visuals/colliders into the same runtime subtree as the robot default prim;
3. verify runtime collision count under `/World/left` and `/World/right` is nonzero;
4. verify unresolved `/visuals/...` warnings disappear or are reduced;
5. only then rerun the zero-hold and single-joint response gates.

## Current Decision

```text
root cause class = USD defaultPrim / reference composition
contact simulation = blocked
gain tuning = blocked
next implementation = diagnostic runtime-complete wrapper
```


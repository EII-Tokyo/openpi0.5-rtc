# Phase 28: Physics Property Comparison

## Question

Phase 26 showed that the ALOHA1-native Isaac asset cannot reliably hold several joints, even though a minimal one-joint Isaac articulation can hold its target exactly.

This phase asks whether the current ALOHA1-native USD physics properties differ from the known-working Trossen Stationary AI Isaac asset in ways that could explain the instability.

## Official Isaac Basis

The NVIDIA Isaac MCP robot setup, physics, and USD guidance was checked before this diagnostic.

The relevant rule is:

- a robot that initializes as an articulation is not automatically controller-ready;
- a controller-ready robot needs coherent USD composition, active joint drives, meaningful effort/damping/limits, rigid bodies, colliders, mass/inertia, and a clear articulation root;
- physics properties live in USD layers and must compose into the actual runtime stage path used by the simulation.

## Tool

Added:

```text
aloha_isaac_replay/scripts/compare_aloha1_trossen_physics_properties.py
```

It is read-only:

- starts Isaac Sim headless only to make `pxr` and USD schemas available;
- opens USD stages;
- traverses joints, drives, rigid bodies, colliders, and mass APIs;
- writes JSON/Markdown reports;
- does not save USD stages;
- does not touch the real robot.

## Command

```bash
codex-evidence --name phase28-physics-property-comparison-v2 -- \
  .venv_issac/bin/python \
  aloha_isaac_replay/scripts/compare_aloha1_trossen_physics_properties.py \
  --output-dir reports/aloha1_isaac_adaptation/phase28_physics_property_comparison_20260718
```

Artifact:

```text
.codex/artifacts/20260718-013507_phase28-physics-property-comparison-v2
```

Report:

```text
reports/aloha1_isaac_adaptation/phase28_physics_property_comparison_20260718/physics_property_comparison.json
reports/aloha1_isaac_adaptation/phase28_physics_property_comparison_20260718/physics_property_comparison.md
```

## Result

| asset | units | up axis | roots | joints | driven | mimic | rigid bodies | colliders | mass APIs | max force values | stiffness values | damping values |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ALOHA1 left native | 0.01 | Y | 1 | 14 | 8 | 1 | 14 | 11 | 14 | `[1, 2, 5, 10, 15, 20]` | `[625]` | `[0]` |
| ALOHA1 right native | 0.01 | Y | 1 | 14 | 8 | 1 | 14 | 11 | 14 | `[1, 2, 5, 10, 15, 20]` | `[625]` | `[0]` |
| Trossen Stationary AI | 1.0 | Z | 1 | 32 | 14 | 2 | 34 | 64 | 34 | `[7, 27, 400]` | multiple tuned values | multiple positive values |

## Key Interpretation

The current ALOHA1-native source USD is not empty. It has joints, drives, rigid bodies, collision APIs, and mass APIs.

That means the failure is more specific than "there is no physics data".

The stronger evidence is:

1. ALOHA1 source layers use `meters_per_unit = 0.01`, while Trossen uses `1.0`.
2. ALOHA1 source layers use `up_axis = Y`, while Trossen uses `Z`.
3. ALOHA1 authored drive damping is `0`, while Trossen uses positive damping values.
4. ALOHA1 static source layers contain collision prims, but Phase 27 showed the current runtime `/World` reference paths compose zero collision prims.
5. ALOHA1 still emits unresolved reference warnings for several visual/collider-like subpaths when loaded through the current wrapper.

So the likely problem is not one single missing switch. It is a USD composition and physics-parameter consistency problem:

```text
source layer has some physics data
        ↓
runtime reference path does not compose all required physics/visual/collider data cleanly
        ↓
drive damping / unit / axis conventions differ from the known-good Trossen pattern
        ↓
articulation initializes but cannot stably hold targets
```

## Consequence For The Plan

Do not copy Trossen gains directly into ALOHA1 yet.

Because the unit system and up-axis differ, copying numbers directly can be physically wrong. First the ALOHA1 asset must be normalized or explicitly converted into the Isaac/Trossen-style runtime convention.

The next repair plan should be:

1. fix ALOHA1 runtime composition so visual/collider references resolve under the actual loaded `/World` paths;
2. decide whether to normalize ALOHA1 to meter/Z-up at USD generation time or add a clearly documented conversion layer;
3. inspect and compare per-link mass/inertia after unit conversion;
4. add positive damping and physically meaningful effort values only after the unit/frame contract is clear;
5. rerun the single-joint hold gate before any trajectory replay or grasp simulation.

## Current Gate

```text
physics_property_comparison = PASS
aloha1_ready_for_contact_or_grasp = NO
aloha1_ready_for_gain_copy_from_trossen = NO
next_required_gate = runtime composition + unit/up-axis normalization plan
```


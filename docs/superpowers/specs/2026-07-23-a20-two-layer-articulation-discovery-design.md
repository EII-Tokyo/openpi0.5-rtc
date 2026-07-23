# A20 Two-Layer Articulation Discovery Design

## Goal

Determine whether the A19 clean ALOHA candidate is both structurally coherent in USD and discoverable by Isaac Sim 5.1 as exactly one 16-DOF articulation, without stepping physics, applying actions, changing the stage, or claiming collision, replay, control, or training readiness.

## Context

The A19 static audit and Isaac open-stage smoke pass. The current A20 Asset Validator result remains blocked by `JointStateChecker`. Running the same selected validator rules against Trossen `stationary_ai.usd` reaches `JointStateChecker: "/stationary_ai/root_joint" has no articulations` and does not return from `ValidationEngine.validate`. This baseline observation does not waive A19's separately recorded joint-state/XForm incoherence failure.

The next gate therefore separates authored USD evidence from runtime PhysX discovery evidence. A failure in either layer remains a hard failure for advancement to target/readback, hold, collision, contact, replay, or RL gates.

## Design

### Layer 1: Pure USD Metadata Gate

Open `aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda` with `Usd.Stage.Open(..., load=Usd.Stage.LoadAll)` and do not initialize PhysX.

Collect and validate:

- default prim is `/aloha`;
- exactly one `PhysicsArticulationRootAPI`, at `/aloha/root_joint`;
- exactly 16 DOF joints under `/aloha/joints`;
- every DOF is `PhysicsRevoluteJoint` or `PhysicsPrismaticJoint`;
- joint path, name, type, axis, lower limit, upper limit, body0, and body1;
- all limits are finite and `lower < upper`;
- observed clean joint paths and order match `proposed_canonical_dof_order` in `a17_clean_articulation_mapping_plan.json` exactly;
- the hash-bound A17 artifact yields a valid versioned 14D policy/16-DOF canonical contract, including the two-finger expansion for OpenPI indices 6 and 13;
- no missing, duplicated, or unexpected DOF paths exist.

Revolute limits remain in authored USD degrees in the Layer 1 report. Prismatic limits remain in stage meters. No unit conversion is needed to compare A17 and A19 authored metadata.

### Layer 2: Runtime Articulation Discovery Gate

Start Isaac Sim 5.1 headless, open the same A19 stage, and initialize only the minimum runtime state required to obtain a valid articulation handle. Do not advance the timeline or call any physics-step API.

Collect and validate:

- exactly one runtime articulation is discovered at `/aloha/root_joint`;
- runtime DOF count is exactly 16;
- the raw runtime 16-DOF order is preserved exactly as returned by PhysX;
- raw runtime order is identical across three fresh processes, but is not required to equal the Layer 1 canonical array order;
- every runtime DOF resolves exactly once to a Layer 1/A17 DOF by unique clean joint path;
- path-aligned runtime names, types, axes, limits, and bodies match Layer 1;
- a versioned adapter provides complete canonical-to-runtime, runtime-to-canonical, and 14D-policy-to-runtime index mappings;
- pure 14D -> 16D -> 14D round trips pass for both paired-finger grippers;
- runtime DOF types and limits are finite and consistent with Layer 1 after applying documented Isaac runtime units;
- repeated discovery facts and provenance are deterministic across three fresh process runs.

This amendment follows the approved [A20 policy-to-runtime order adapter design](2026-07-23-a20-policy-runtime-order-adapter-design.md). The observed deterministic PhysX order alternates the twelve arm joints left/right, then groups the two left fingers followed by the two right fingers. This noncanonical raw order is valid evidence; changing USD authoring merely to imitate that traversal order is out of scope.

The implementation must record whether a physics scene or handle initialization was required. It must always report `physics_stepped=false`, `actions_applied=false`, `targets_written=false`, and `stage_saved=false`.

If Isaac Sim cannot create a valid handle without a simulation reset or one initialization update, the script must stop and report `BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION`; it must not silently step to make the test pass.

## Inputs And Outputs

Inputs:

- `aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda`
- `aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json`
- `aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml`

Outputs:

- `aloha_isaac_rebuild/artifacts/validation/a20_usd_dof_metadata_gate.json`
- `aloha_isaac_rebuild/artifacts/validation/a20_runtime_articulation_discovery_gate.json`
- `aloha_isaac_rebuild/reports/a20_two_layer_articulation_discovery.md`

Generated outputs must include the absolute input paths, input SHA-256 hashes, Isaac Sim version for Layer 2, timestamps, exact observed records, expected records, mismatch lists, and explicit safety flags.

## Status Semantics

- `PASS_A20_USD_DOF_METADATA`: Layer 1 passes all checks.
- `FAIL_A20_USD_DOF_METADATA`: Layer 1 contains any mismatch or invalid value. Layer 2 must not run.
- `PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP`: Layer 2 discovers one deterministic 16-DOF articulation with complete path-aligned semantic and policy mappings, a passing round trip, and no prohibited action. Raw positional equality with canonical order is informational only.
- `FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY`: runtime count, DOF metadata, determinism, or handle validity fails.
- `BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION`: the requested evidence cannot be obtained within the no-step boundary.

A full two-layer pass does not change the existing Asset Validator result into a clean pass. The final report must present the validator blocker and the two-layer result as separate evidence.

## Safety Boundary

The gate must not:

- call timeline Play, `World.step`, `SimulationContext.step`, or an equivalent stepping API;
- initialize or invoke an articulation controller;
- write position, velocity, effort, drive target, or drive gain;
- apply Asset Validator fixes or modify transforms;
- save, flatten, or overwrite a USD layer;
- add collision, a PhysicsScene, gravity, contact, or an object;
- read or control the real robot;
- run HDF5 replay, policy inference, reward code, or training.

## Testing Strategy

Pure helper functions will be test-driven with synthetic expected/observed records, covering deterministic interleaving, one-run order changes, duplicate/missing DOFs, invalid limits, wrong units/types, adapter corruption, 14D/16D round trips, and safety-flag rejection. Layer 1 will then run against the real A17/A19 artifacts. Layer 2 will run through `codex-evidence`; full Isaac logs will stay in `.codex/artifacts/`, while the checked output remains bounded JSON and Markdown.

Existing dirty A19/config/audit changes are treated as user-owned inputs. Implementation must not revert or commit them incidentally.

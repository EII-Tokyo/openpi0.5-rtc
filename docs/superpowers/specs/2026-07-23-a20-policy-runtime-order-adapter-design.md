# A20 ALOHA Policy-to-Runtime Order Adapter Design

## Goal

Allow the A19 dual-arm ALOHA articulation to keep Isaac/PhysX's deterministic
left/right-interleaved 16-DOF runtime order while preserving the established
ALOHA/OpenPI 14D policy and dataset order: six left-arm joints, one left
gripper scalar, six right-arm joints, and one right gripper scalar.

This design replaces A20's invalid requirement that raw PhysX order must equal
the A17 canonical semantic order. It does not weaken joint identity, metadata,
determinism, provenance, or safety checks.

## Decision And Rejected Alternatives

The selected design is an explicit, versioned policy-to-runtime adapter.

- Rejected: reshape the USD hierarchy merely to force PhysX to expose all left
  DOFs before all right DOFs. This couples control semantics to an internal
  traversal order and risks distorting the Stationary AI-style articulation.
- Selected: preserve raw runtime order, resolve every DOF by unique joint path
  and canonical name, and precompute index mappings once per runtime handle.
- Rejected: migrate ALOHA datasets, OpenPI checkpoints, real-robot interfaces,
  and gripper indices to an interleaved convention. That would change the
  established 14D contract without a simulation requirement.

## Official Isaac Basis

The official NVIDIA Isaac Sim documentation MCP was accessed through the
MCPJungle `codex-research` Gateway group. Its code examples show that
`ArticulationActions` accepts explicit `joint_indices` or `joint_names`, and
that target arrays are assigned at those resolved indices. The adapter uses
this supported interface boundary instead of assuming that a full action array
already matches raw PhysX order.

The Gateway evidence is:

- MCP endpoint: `http://127.0.0.1:18080/v0/groups/codex-research/mcp`;
- official server: `nvidia-isaac-docs`;
- queried tool: `nvidia-isaac-docs__search_isaac_sim_code_examples`;
- relevant examples: `Articulation.apply_action`,
  `ArticulationSubset.get_applied_action`, and
  `OgnIsaacArticulationControllerInternalState.apply_action`.

No direct NVIDIA MCP server entry is introduced.

## Three Explicit Order Contracts

### 1. Runtime Order

`runtime_order` is the exact 16-DOF order returned by the initialized PhysX
articulation handle. It is evidence, not a policy convention. A20 must preserve
it unchanged, including each raw runtime index.

For the current A19 candidate, the observed pattern is left/right interleaved
by corresponding arm depth. A20 does not hard-code that pattern; it records the
actual order and requires three fresh processes to return the same order and
metadata.

### 2. Canonical Joint Semantic Order

`canonical_joint_order` is the 16-record A17 order used to identify all real
joint DOFs, including both fingers for each gripper. It remains left-arm first
and right-arm second. It is not required to equal `runtime_order` by index.

Each canonical record must resolve to exactly one runtime record by unique
clean joint path. The reverse mapping must also be one-to-one. Joint type,
axis, limits, bodies, and canonical name are compared after this explicit
path-based join.

### 3. ALOHA Policy Order

`policy_order` remains the standard 14D ALOHA/OpenPI convention:

1. left arm indices 0-5;
2. left gripper scalar at index 6;
3. right arm indices 7-12;
4. right gripper scalar at index 13.

The A17 records already bind every runtime DOF to an `openpi_index` and
`dataset_index`. Arm policy indices map to one runtime DOF. Gripper policy
indices 6 and 13 each map to two finger DOFs using their existing affine
`sign`, `offset`, and `scale` metadata.

## Components

### Pure Mapping Module

Add a focused pure-Python module responsible for:

- validating the 14D policy schema and 16D canonical/runtime inventories;
- building `canonical_to_runtime_indices` by unique clean joint path;
- grouping canonical DOFs into `policy_to_runtime` entries by `openpi_index`;
- validating arm cardinality 1 and gripper cardinality 2;
- expanding a normalized 14D policy vector to a raw-order 16D runtime vector;
- collapsing a raw-order 16D state vector to 14D policy order;
- rejecting a gripper readback when its two independently inverted finger
  scalars disagree beyond a documented tolerance;
- providing a round-trip self-check with finite synthetic samples.

The module performs no Isaac import, handle initialization, action application,
timeline operation, or USD write.

### Layer 2 Aggregation

Layer 2 continues to collect the untouched raw records from three fresh
processes. Aggregation changes as follows:

- require exact equality of raw runtime facts across the three runs;
- require the runtime joint path set to equal the A17 canonical path set;
- compare joint metadata through the explicit path join;
- require a complete, deterministic policy-to-runtime mapping;
- require the pure adapter round-trip self-check to pass;
- retain all existing handle, process, cleanup, provenance, no-step, no-action,
  no-target-write, and no-save checks.

A raw-order difference from A17 becomes informational evidence, not a failure.
An order difference among the three runtime runs remains a hard failure.

### Generated Evidence

The runtime JSON gains a versioned `order_adapter` object containing:

- `schema_version`;
- `policy_order`;
- `canonical_joint_order`;
- `runtime_order`;
- `canonical_to_runtime_indices`;
- `runtime_to_canonical_indices`;
- `policy_to_runtime` records, including gripper expansion metadata;
- `mapping_complete`;
- `mapping_bijective_at_dof_level`;
- `round_trip_check`;
- hashes of all trusted mapping inputs.

The Markdown report shows separate lines for:

- three-run raw runtime determinism;
- runtime joint inventory and metadata match;
- policy-to-runtime mapping completeness;
- adapter round-trip check;
- whether raw order happens to equal canonical order, marked informational;
- overall readiness.

## Failure Semantics

Layer 2 fails closed for any of the following:

- duplicate, missing, or unexpected runtime joint paths;
- nondeterministic raw order or metadata across fresh processes;
- path-aligned joint name, type, limit, axis, body, or unit mismatch;
- missing or invalid OpenPI/dataset index;
- an arm policy index resolving to other than one runtime DOF;
- a gripper policy index resolving to other than two configured finger DOFs;
- duplicate runtime indices in the adapter;
- non-finite affine mapping values;
- inconsistent two-finger inverse readback;
- failed synthetic round trip;
- stale or mismatched provenance;
- any existing prohibited operation or process-integrity failure.

No failure is converted to PASS by sorting records only for presentation.

## Safety And Scope

This change does not:

- modify the A19 USD hierarchy, joints, transforms, mass, collision, or physics;
- apply an articulation action or initialize a controller;
- play or step physics;
- write a drive target, state, gain, effort, velocity, or position;
- change OpenPI, HDF5, LeRobot, or real-robot 14D conventions;
- control or inspect the real robot;
- claim collision, control, replay, contact, or training readiness;
- waive the independent Asset Validator blocker.

Existing dirty A19/config/audit/report files remain user-owned inputs and must
not be included in adapter commits.

## Testing

Implementation follows test-driven development.

Pure unit tests cover:

- the current deterministic interleaved 16-DOF runtime order;
- a different but internally deterministic raw order;
- complete 16D canonical/runtime bijection;
- all 14 policy indices and both two-finger gripper expansions;
- policy-to-runtime-to-policy round trips at normalized values 0, 0.5, and 1;
- duplicate, missing, unexpected, and nondeterministic runtime records;
- invalid cardinality, indices, affine values, and gripper disagreement;
- report semantics where deterministic interleaving is PASS and raw-order
  equality is informational;
- unchanged rejection of unsafe flags and stale provenance.

The existing A20 focused suite and Ruff must pass before any real Layer 2
regeneration. A fresh Layer 1 and three-process Layer 2 are run only from a
clean reviewed commit through `codex-evidence`.

## Acceptance Criteria

The design is complete when:

1. the raw 16-DOF PhysX order is preserved exactly in evidence;
2. all three fresh runs return identical raw order and runtime facts;
3. every canonical DOF maps exactly once to a runtime index by path;
4. every ALOHA/OpenPI policy index maps to the expected one or two runtime DOFs;
5. pure policy/runtime round-trip checks pass;
6. a deterministic interleaved runtime order can pass Layer 2 without changing
   the 14D ALOHA convention;
7. all existing safety, process-integrity, and provenance gates remain strict;
8. the independent Asset Validator result remains visible and capable of
   keeping overall status `NOT_READY`.

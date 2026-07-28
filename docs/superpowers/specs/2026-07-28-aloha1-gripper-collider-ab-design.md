# ALOHA 1 Follower Gripper Collider A/B Design

Date: 2026-07-28

## Scope

This experiment answers one question only: with the current follower gripper
trajectory, mimic behavior, drives, material, bottle proxy, and physics step
held constant, does replacing only the two finger collision approximations
from `convexHull` to `convexDecomposition` materially improve the bottle hold
gate?

It does not modify the original URDF, imported USD, configuration layers, or
the default follower asset. It does not extend the workcell, cameras, ROS,
insertion task, or Task 8 optimization.

## Evidence authority

Runtime behavior and authored API names are derived from the local Isaac Sim
5.1.0 / Kit 107.3.3 installation:

- `isaacsim.asset.importer.urdf` 2.4.30;
- `omni.physx` 107.3.26+107.3.3;
- `omni.usd.schema.physx` 107.3.26+107.3.3;
- the local `PhysxSchema` Python bindings and `schema.usda`.

The NVIDIA Isaac MCP is used as the mandatory official-source review gate.
Any MCP result from a different importer version is recorded as non-authoritative
for this experiment. Local 5.1 source, schema, and runtime readback win.

## Baseline protection

Before creating diagnostic assets, the implementation records SHA-256 hashes
of:

- generated follower URDFs;
- imported follower USDs;
- existing debug configuration layers;
- current gripper baseline report, contact-event reports, and curves;
- the shared `gripper_finger.stl`.

The comparison and validation commands verify those hashes again before and
after the experiment. New outputs use dedicated diagnostic and report paths;
existing baseline reports are never overwritten.

## Diagnostic assets

The preferred implementation creates two independent wrapper layers per
follower under:

`assets/Trossen/ALOHA1/1.0/diagnostics/gripper_collision/`

Each wrapper references the current debug-acceleration configuration asset.
The hull wrapper provides no collision-behavior change and records the
baseline token readback. The decomposition wrapper authors opinions only on
the exact left and right finger collision mesh prims:

- `UsdPhysics.MeshCollisionAPI.approximation = convexDecomposition`;
- `PhysxSchema.PhysxConvexDecompositionCollisionAPI` with all attributes left
  at the local schema defaults.

The wrapper is accepted only when reopening the composed USD proves that the
two finger colliders have the requested token and no non-finger collider token
changed. If local composition cannot isolate those opinions, the fallback is
a separate full diagnostic re-import with `ImportConfig.convex_decomp = True`.
That fallback must enumerate every affected collider and cannot be described
as finger-only.

## Geometry audit

The audit combines three evidence classes:

1. USD readback: collider prim paths, authored/composed approximation tokens,
   applied schema APIs, transforms, and source asset provenance.
2. Source-mesh metrics: SHA-256, triangle/vertex counts, scaled AABB, volume
   where the mesh is watertight, single-hull AABB/volume, and sampled
   source-to-hull distances.
3. PhysX runtime evidence: collision-cooking log excerpts, available cooked
   representation metadata, CPU/GPU fallback warnings, and deterministic
   collider visualizations.

If the local runtime does not expose cooked convex vertices/pieces through a
supported 5.1 API, the report marks those fields `UNAVAILABLE_IN_SUPPORTED_API`
instead of substituting an offline decomposition as if it were PhysX cooking.
Offline source/hull sampling may be reported only as supplemental geometry
evidence.

## Frozen A/B variables

The profile file is the single machine-readable manifest. Both variants must
read back:

- friction `0.7`;
- restitution `0`;
- bottle mass `0.020 kg`;
- bottle diameter `0.065 m`;
- physics frequency `60 Hz`;
- `solve_articulation_contact_last = true`;
- the same initial joint state and close trajectory;
- identical stiffness, damping, max force, mimic relation, and hold interval;
- hold interval `2 s`;
- drop gate `0.010 m`;
- self-collision disabled and bottle collision enabled;
- no surface gripper and no bottle fixed constraint after release.

The first round changes exactly one field: finger collision approximation.
Each follower/profile pair runs at least 20 trials. Every trial constructs a
new stage and `World`, performs a fresh `world.reset()`, starts with a fixed
bottle for bilateral-contact establishment, and releases that same bottle for
the hold interval. It never resumes from an earlier trial.

## Trial evidence and gates

Every trial stores:

- first left/right contact frame and time;
- collider and material paths;
- contact position, normal, impulse, separation;
- estimated normal force (`normal impulse / dt`);
- left/right contact duration and loss time;
- bottle position, linear/angular velocity, drop;
- persistent penetration and unexpected internal collision flags;
- wall-clock runtime;
- a deterministic signature over canonical machine results.

The pass gate is unchanged from the baseline. A run cannot pass without
bilateral contact before release, finite impulses/state, no persistent
penetration, no unexpected finger/bar/internal contact, and bottle drop no
greater than `0.010 m` over `2 s`. Missing measured material and bottle inertia
remain calibration blockers and therefore cap physics validity at `PARTIAL`.

## Second-round root-cause experiment

After round one is complete, the same harness runs:

1. Hull + current mimic;
2. Decomposition + current mimic;
3. Hull + explicit symmetric finger targets;
4. Decomposition + explicit symmetric finger targets.

The explicit mode sends both targets with
`right_finger_target = -left_finger_target` and is labeled
`DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`. No other variable changes.

Classification is computed from the hold success-rate and gate evidence:

- `collider_primary`;
- `mimic_primary`;
- `collider_and_mimic`;
- `neither_resolved`;
- `inconclusive`.

The final decomposition status is one of:

- `IMPROVES_HOLD`;
- `NO_MEANINGFUL_EFFECT`;
- `WORSENS_CONTACT`;
- `INCONCLUSIVE`.

No decomposition parameter is tuned in the initial experiment. Any later
parameter study requires a documented evidence trigger and changes one schema
attribute at a time.

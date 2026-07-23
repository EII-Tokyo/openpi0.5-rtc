# A20 Joint-State/XForm Coherence Repair Design

## Goal

Make the A19 clean ALOHA articulation pass Isaac Sim 5.1
`JointStateChecker` without changing the confirmed baked home pose, the A17
joint inventory, the ALOHA/OpenPI 14D convention, the PhysX 16-DOF runtime
order, limits, axes, drive targets, or joint states.

## Confirmed Failure

The independent Asset Validator reports one articulation-level failure:

```text
Joint State for "/aloha/root_joint" is not coherent with transforms of rigid
bodies belonging to the articulation
```

This does not mean that `/aloha/root_joint` itself is displaced. Read-only
joint-frame diagnostics show:

- the root fixed joint has zero positional residual;
- all 21 joints and all 16 movable DOFs exist;
- the articulation graph is single-rooted and fully connected;
- the four finger prismatic displacements agree with their authored
  `0.058 m` states and joint axes;
- several revolute joints differ from their authored state by roughly
  `0.00001-0.006 degrees`;
- the left tabletop-to-base reparent joint has roughly `0.0865 mm` positional
  residual.

The generator composes body XForms from the A16 runtime-baked home pose, then
copies joint local frames and state attributes from a freshly opened source
USD. Those two representations describe nearly the same pose but are not one
exact constraint solution after runtime evaluation, reparenting, float
conversion, and USDA serialization.

## Selected Approach

Treat the following as immutable inputs:

- baked A16 body XForms;
- authored joint types and body relationships;
- joint axes and limits;
- revolute/prismatic `state:*:physics:position`;
- drive target positions;
- the A17 14D/16D mapping contract.

For every mapped A19 joint, preserve its body0-side local frame and solve a new
body1-side local frame that makes the current joint state exactly consistent
with the baked body transforms.

NVIDIA's row-vector transform convention is:

```text
joint_world = joint_local * body_world
```

Let:

- `B0`, `B1` be the current body world transforms;
- `L0` be the preserved body0-side joint-local transform;
- `M(q)` be the joint motion transform for the authored state;
- `W1_desired = M(q) * L0 * B0`.

The repaired body1 local frame is:

```text
L1_new = W1_desired * inverse(B1)
```

`M(q)` is:

- identity for a fixed joint;
- rotation by the authored angular state around the authored X/Y/Z axis for a
  revolute joint;
- translation by the authored linear state along the authored X/Y/Z axis for
  a prismatic joint.

Only `physics:localPos1` and `physics:localRot1` may change. The repair must not
change body XForms, `localPos0/localRot0`, state, drive target, limit, axis,
body relationship, joint path, or joint type.

## Rejected Alternatives

### Rewrite joint state from the baked XForms

This is simpler, but it silently changes the confirmed home-pose values and
could drift policy initialization from the source ALOHA contract.

### Rebuild all body XForms with forward kinematics

This can also produce a coherent result, but it expands the change into the
visual/collider reconstruction and risks invalidating A15/A16 spatial
evidence.

### Delete joint-state attributes

This may suppress the checker or force runtime inference, but it removes
explicit initial-state evidence and is not fail-closed.

## Components

### Pure coherence module

Create:

```text
aloha_isaac_rebuild/scripts/a19_joint_state_coherence.py
```

Responsibilities:

- build a joint-local matrix from `localPos/localRot`;
- build `M(q)` for fixed, revolute, and prismatic joints;
- compute desired and observed body1 world joint frames;
- report translation and orientation residuals;
- solve and author a new body1 local frame;
- reject missing bodies, missing/non-finite state, unsupported axes or joint
  types, non-invertible transforms, non-rigid decompositions, and non-finite
  results.

The module must not open stages, save layers, start Isaac, step physics, apply
actions, or touch the real robot.

### A19 generator integration

After a mapped joint has copied its reviewed source attributes and established
its clean body relationships, call the coherence solver. Record before/after
residuals in the A19 audit payload.

The generator must fail before saving if any joint cannot be solved or if the
post-repair residual exceeds:

```text
translation: 1e-6 m
orientation: 1e-4 degrees
```

### A19 static audit integration

Extend the existing A19 audit to independently recompute joint-state
coherence. A structural PASS now also requires every mapped joint to meet the
same residual thresholds. Emit bounded per-joint residual evidence and maxima.

This closes the current gap where topology can pass while the initial pose is
not constraint-coherent.

## Error Handling

The repair is all-or-nothing:

- no partial stage save;
- no silent default state for a movable joint;
- no axis inference from joint names;
- no tolerance widening after a failure;
- no Asset Validator auto-fix;
- no mutation of A17 mapping or A16 body XForms.

Every failure identifies the joint path and violated invariant.

## Testing

Test-driven coverage must include:

1. a synthetic fixed joint with mismatched body1 pose;
2. a synthetic revolute joint with non-zero authored state;
3. a synthetic prismatic joint with non-zero authored state;
4. X/Y/Z axes;
5. failure on missing/non-finite state, unsupported axis, missing body, and
   singular body transform;
6. preservation of body XForms, body0 local frame, state, target, limits,
   relationships, type, and axis;
7. current A19 pre-repair evidence fails the new coherence audit;
8. regenerated A19 passes the coherence audit.

Integration verification order:

1. focused unit tests and Ruff;
2. regenerate A19 and run its static audit;
3. run the independent Asset Validator;
4. regenerate A20 Layer 1;
5. run three fresh no-step A20 Layer 2 processes;
6. regenerate the bounded report.

The result is complete only if Asset Validator is clean, Layer 1 passes, Layer
2 passes, the 14D/16D adapter and round trip remain unchanged, all safety flags
remain false, and no real-robot action occurs.

## Scope Boundary

This work does not:

- add collision or contact;
- create or step a PhysicsScene;
- apply joint targets or actions;
- change the home pose;
- change runtime DOF order;
- change the policy mapping;
- run replay, policy inference, reward learning, or training;
- control the physical ALOHA robot.

# ALOHA supplier-CAD Task 7 robot-scope design

## Goal

Close the supplier-CAD `follower_left` Task 7 validation gap without
modifying the approved source Stage, the Task 5 bottle Stage, the default
configuration, or the final collider.

## Evidence boundary

- Keep the Task 5 bottle Stage as the authoritative runtime grasp/hold
  evidence.
- Create a robot-scoped physical diagnostic for `IsaacSim.PhysicsRules` and a
  schema-only wrapper of the same hierarchy for `IsaacSim.RobotRules`.
- Preserve the approved source and Task 5 Stage hashes as immutable gates.
- Keep `follower_right`, calibrated friction/dynamics, and redistribution
  license issues as explicit blockers.

## Composition

The robot-scoped asset references the existing Task 5 diagnostic workcell but
deactivates table, midair, placeholder pipe, bottle, and other environment
children. Its default prim is a robot asset root with:

- `IsaacRobotAPI`;
- ordered `isaac:physics:robotLinks`;
- ordered `isaac:physics:robotJoints`;
- `IsaacLinkAPI` on the selected robot link prims;
- `IsaacJointAPI` on the selected articulation joints.

The physical diagnostic's independent configuration layer applies Isaac Sim 5.1
`PhysxSchema.JointStateAPI` with `angular` or `linear` instance names and
initial positions equal to the existing drive targets.

The schema-only RobotRules wrapper composes the same robot hierarchy and the
Robot Schema opinions but excludes the diagnostic physics sublayer. This
prevents Task 5 mass, inertia, drive, and collider diagnostics from being
misclassified as prohibited Robot-Schema source-layer overrides.

## Mass/inertia policy

Do not author guessed mass or inertia. Environment helpers are excluded by the
robot-scoped composition. Any remaining zero-mass rigid body is classified by
its actual articulation/joint participation:

- a nonphysical helper with no joint-body participation may have the
  accidental rigid-body API removed only in the diagnostic layer;
- a participating robot body with missing dynamics remains `HARD_BLOCKER`.

## Validation

Run the three official Isaac Sim 5.1 categories against their correct targets:

- PhysicsRules: physical robot-scoped diagnostic asset;
- RobotRules: schema-only robot wrapper;
- SimReadyAssetRules: physical robot-scoped diagnostic asset.

Open each Stage fresh twice and require deterministic machine-readable
signatures. Task 5's existing 20/20 hold result is referenced, not replaced.
Task 8 remains `NOT_RUN`.

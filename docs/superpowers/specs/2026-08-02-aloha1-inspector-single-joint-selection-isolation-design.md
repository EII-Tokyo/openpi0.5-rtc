# ALOHA1 Physics Inspector Single-Joint Selection Isolation Design

## Status and Scope

This design corrects the user-facing Physics Inspector launch after the native
tabletop collision gate was made to associate the complete follower-left
physics set and the confirmed table with one Inspector panel. It changes only
the transient Inspector selection lifecycle and its validation. It does not
change the approved USD, joint limits, gains, masses, collider geometry,
collision filtering, table geometry, or real robot.

The approved Stage remains:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`

Its frozen SHA-256 is
`165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2`.
The user-facing application remains Isaac Sim Full on GNOME workspace index 2
(workspace 3), in Perspective view, with the main timeline stopped.

## Confirmed Root Cause

The current launcher uses the global USD Stage selection to associate 50
physics prims with one Inspector model. That set contains the follower-left
articulation anchor, the confirmed table, all 13 left joints, all left rigid
bodies, and all required left colliders. The association is correct for
collision, but the launcher leaves those 50 prims selected after calling the
Inspector toolbar's `Use Selection` action.

NVIDIA's installed `InspectorSelectionHandler` mirrors the global Stage
selection into the Inspector tree selection. Its joint value callback then
iterates over every selected joint and calls `set_joint_value()` with the
value from the slider being changed. Runtime evidence showed all six visible
arm Drive Target Position values changing to `10.8` together. This is a real
multi-target edit, not ordinary downstream-link motion in a serial robot.

Evidence is retained at:

`.codex/artifacts/20260802-native-left-inspector-final-handoff/user_joint_coupling_report.png`

The Inspector model's associated prim set and its current GUI selection are
therefore distinct states and must have separate lifecycles.

## Selected Architecture

### 1. Association phase

The launcher opens the frozen Stage and temporarily selects the same explicit
50-prim physics set already proven by the native collision gate. It invokes
the native Inspector `Use Selection` action exactly once. The associated set
must retain both anchor paths:

- `/World/follower_left/vx300s_left/root_joint`;
- `/World/environment/worldBody/user_confirmed_table`.

The launcher then configures Joint Drive target control, quasistatic mode on,
fixed articulation base on, and gravity off. Before selection is cleared, it
verifies the association label, all 13 joint rows, and authoring state.

### 2. Interaction phase

After association succeeds, the launcher replaces the 50-prim global Stage
selection with the two non-joint anchors and allows Inspector selection
synchronization to settle. This selection change must not call `Use Selection`
again and must not replace or reparse the Inspector model. The Inspector
association label remains the original 50-prim association, while both of
these transient interaction selections contain only the articulation and
table anchors:

- `omni.usd` global Stage selection;
- `InspectorSelectionHandler`'s current selection.

Neither anchor is a joint row. NVIDIA's value callback therefore falls back to
the row that emitted the change. Clicking or dragging one slider writes only
that joint's Drive Target Position. Subsequent explicit Ctrl/Shift
multi-selection remains available to the user and intentionally restores
batch editing; the launcher itself must start with zero selected joint rows.

Runtime evidence showed that an entirely empty selection changes native
Inspector authoring behavior, while retaining these two anchors preserves the
articulation/table simulation set. The anchors are consequently part of the
interaction contract, not an incidental visual selection.

### 3. Disposable single-joint runtime validation

A temporary Isaac Sim Full process reopens the frozen Stage and performs the
same association and joint-selection-isolation sequence. It records all non-mimic left
joint Drive Target Position values, invokes the same Inspector row-value
callback path for `shoulder`, and checks the result before any handoff:

- the shoulder target changes to the requested value;
- every other joint target remains bitwise or numerically unchanged within
  `1e-9` in the joint's authored units;
- the Stage and Inspector transient selections contain only the two non-joint
  anchors and contain zero joint paths;
- all 13 joint rows remain present.

The validation restores the probe target before executing the native
pressure-contact trial with the confirmed table still associated. It must
retain the three-trial collision gates: exact table/finger contact, 180/180
supported hold steps (a reported contact, or after a confirmed contact a
settled finger/table position within the same `0.5 mm` solver tolerance), at
most `0.5 mm` table-top penetration, at most `0.1 mm`
visual/collision disagreement, valid limits and CCD, no disallowed support
contact, and unchanged Stage hash.

This settled-support rule is required because PhysX stops emitting repeated
contact reports when the correctly isolated single-joint system sleeps. The
older 180/180 raw-contact result was produced by the bug itself: the shoulder
target was copied to other selected joints, keeping the articulation moving.
The replacement rule still requires a real allowed contact before settled
support can count and fails closed on penetration or separation.

Each validation process is disposable, never saves the Stage, never commits
Inspector authoring results, and never touches the real robot.

### 4. Clean Full-GUI handoff

Only after the single-joint isolation check and the three native collision
trials pass does the workflow terminate the current dirty Full process by its
verified PID. It does not save or commit the current state in which multiple
targets equal `10.8`.

A fresh Full process then loads the frozen Stage from disk, performs the
association phase, isolates transient joint selection, and verifies:

- exact Stage URL and frozen hash;
- Z-up, meter scale, Perspective view, and stopped main timeline;
- Inspector state `AUTHORING`;
- 13 joint rows;
- one 50-prim Inspector association containing both required anchors;
- exactly the two non-joint anchors selected and zero selected Inspector joint
  rows;
- Joint Drive control, quasistatic on, fixed base on, and gravity off;
- Full Kit command line and workspace index 2.

The final screenshot must show distinct original joint target values rather
than one duplicated value across all rows. The user can then operate a single
joint without first clearing a launcher-created multi-selection.

## Error Handling

The workflow fails closed and does not hand off a new GUI when:

- the approved Stage path or hash differs;
- the Inspector association loses either the articulation or table anchor;
- isolating the transient joint selection removes the 13 joint rows or disables
  authoring;
- the Stage or Inspector selection remains multi-selected;
- changing shoulder changes any other joint target;
- the native collision gate regresses;
- a PhysX, USD, rendering, or out-of-memory error invalidates validation;
- the disposable process cannot prove that it did not save the Stage.

On failure, the current dirty process is not saved. Diagnostic JSON, bounded
logs, and screenshots remain under `.codex/artifacts/`.

## Rejected Alternatives

- **Require the user to click a joint name before every drag:** this is fragile
  and leaves the same launcher-created multi-selection active by default.
- **Use two Inspector panels:** runtime evidence already showed that the table
  did not constrain the left arm through this path.
- **Switch to Joint State Position:** it avoids Drive dynamics but can
  kinematically place geometry through the table, so it is not a substitute
  for the requested collision test.
- **Build a custom joint-control panel:** it would no longer validate native
  Physics Inspector interaction and is unnecessary once selection state is
  isolated.

## Test-Driven Implementation

Implementation follows red-green-refactor:

1. Add a failing source/runtime contract requiring the launcher to isolate
   joint selection only after `Use Selection` and association verification.
2. Add a pure helper test for deciding whether one target changed while all
   other targets remained unchanged.
3. Add a failing native contract requiring the exact Inspector row callback,
   anchor-only transient selection, and per-joint before/after target evidence.
4. Implement the minimal joint-selection-isolation helper and fail-closed assertions.
5. Run the disposable single-joint test, then rerun all three native collision
   trials.
6. Close the dirty process without saving and launch a clean Full handoff on
   workspace 3.

## Completion Evidence

Completion requires fresh evidence for all of the following:

- focused unit and startup-contract tests pass after a demonstrated red phase;
- the native UI-equivalent edit changes shoulder and no other joint target;
- the Inspector retains 13 rows after transient joint selection is isolated;
- all three native tabletop collision trials pass their existing gates;
- the USD hash remains unchanged and no Stage is saved;
- the current dirty `10.8` posture is discarded by process replacement;
- the final process is `isaacsim.exp.full.kit`, on workspace index 2, with the
  timeline stopped;
- the final screenshot shows distinct clean joint targets and a ready
  single-joint Inspector interaction state;
- unrelated working-tree files and runtime artifacts are not committed.

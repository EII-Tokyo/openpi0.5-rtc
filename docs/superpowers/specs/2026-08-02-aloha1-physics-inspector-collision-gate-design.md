# ALOHA1 Physics Inspector Collision Gate Design

## Goal

Provide a repeatable, evidence-backed Physics Inspector configuration for the
approved ALOHA1 tabletop-zero diagnostic Stage. Before handing the GUI to the
user, an automated runtime gate must prove that the follower-left wrist,
gripper, and finger colliders contact the confirmed tabletop without tunneling
through it.

## Frozen Runtime Identities

- Full Isaac Sim experience: `isaacsim.exp.full.kit`
- Approved Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`
- Approved Stage pre-change SHA-256:
  `5c9d1379da92cfcc858ab10ced587b31c117e797f4e5a943ed815f4d735168a7`
- Stage convention: `metersPerUnit = 1`, `upAxis = "Z"`
- Left articulation root:
  `/World/follower_left/vx300s_left/root_joint`
- Confirmed table collider:
  `/World/environment/worldBody/user_confirmed_table`
- User-facing GNOME workspace: index `2` (workspace 3)

Before editing, switching, or reopening the Stage, implementation must
revalidate the absolute path, pre-change hash, default prim, sublayers, and
both required prim paths. After the reviewed physics sublayer is added, the
intentional new root-layer hash becomes the frozen post-change identity and
must be used for all automated and manual-test launches.

## Current Evidence and Failure Mode

The Inspector association and control mode are currently correct: the left
articulation is exposed through `Joint Drives Target Position`, and the exact
confirmed table is loaded in a second Inspector panel. The table has an enabled
static collider, the left wrist/gripper/finger links have enabled colliders,
the finger visual and collision bounds agree, and no collision groups or
filtered-pair properties exclude the pair.

The composed Stage has no persistent `UsdPhysics.PhysicsScene`. The relevant
left rigid bodies have no authored `physxRigidBody:enableCCD` value. The table
is 15 mm thick, while Physics Inspector authoring simulation advances at a
fixed 1/60 second step. These facts make discrete-step tunneling the documented
hypothesis to test. This is not yet treated as a passing diagnosis: the runtime
gate must reproduce contact and prove non-crossing after the fix.

## Considered Approaches

### 1. Dedicated diagnostic physics override layer — selected

Add a small, traceable USD physics override layer as a new sublayer of the
approved diagnostic root layer. It will define the scene-level stepping and
CCD policy and author CCD on the follower-left dynamic links without changing
source CAD geometry, joint gains, table dimensions, or collision filtering.
The root-layer hash change is intentional, reviewed, and recorded; the base
geometry, collider, and tabletop source layers remain unchanged.

This approach is persistent, statically auditable, reproducible across GUI
restarts, and isolated from the lower-level geometry and collider layers.

### 2. Runtime-only Session Layer — rejected

Applying PhysicsScene and CCD values only after application startup would avoid
a disk USD edit but would be harder to audit and reproduce. Inspector reparsing
or a later clean restart could lose the settings.

### 3. Thicker table collider — rejected

Increasing the table collider thickness could hide tunneling but would make the
simulation geometry disagree with the confirmed 15 mm tabletop. It is not an
acceptable physics fix or validation method.

## Physics Configuration

The diagnostic override will provide one explicit PhysicsScene and the minimum
policy needed for this test:

- physics simulation rate: 240 Hz;
- scene CCD enabled;
- swept CCD enabled on the follower-left articulation's dynamic links,
  including the wrist, gripper, gripper mechanism, and both finger links;
- existing table collider retained as a static 15 mm cube;
- existing drive stiffness, damping, type, force limits, joint limits, masses,
  materials, and collision filtering retained unchanged;
- no gravity change unless the existing Inspector authoring mode requires its
  established gravity-off setting;
- no real-robot connection or command.

Before authoring these fields, implementation must query the NVIDIA official
Isaac MCP for the exact Isaac Sim 5.1 USD/PhysX semantics for PhysicsScene
stepping, scene and rigid-body CCD, articulation-link CCD, contact reporting,
and Inspector drive authoring. The project Isaac/physics and robotics-example
expert gates must produce concrete acceptance and regression criteria. If the
mandatory NVIDIA MCP cannot answer the required API point, implementation
stops before modifying Isaac files.

## Automated Runtime Collision Gate

### Preflight

The verifier must fail closed unless all of these are true:

1. The exact approved Stage identity and composition match the frozen values.
2. The Stage is meter-scaled and Z-up.
3. The left articulation and confirmed table prims exist.
4. The table has enabled static collision and retains its 15 mm thickness.
5. The selected follower-left wrist/gripper/finger colliders exist and are
   enabled.
6. One PhysicsScene provides the expected simulation rate and scene CCD.
7. The required dynamic follower-left links have CCD enabled.
8. No collision group or filtered-pair rule suppresses the tested pair.

### Contact-Seeking Motion

Each trial starts from a clean Stage reset and a deterministic, collision-free
pose. It uses normal articulation Drive Target semantics, not direct transform
teleportation. A bounded, small-increment joint sweep approaches the tabletop
from above. The sweep stays inside authored joint limits and stops on a hard
timeout, a non-finite state, an unexpected collision, or successful table
contact.

After the first valid table contact, the verifier commands a bounded target
that is kinematically beyond the tabletop. The target may be below the surface;
the actual simulated articulation must remain blocked by contact. Blind gain,
offset, thickness, or friction sweeps are forbidden.

### Trial Acceptance

A trial passes only when all conditions hold:

- PhysX reports at least one contact between the exact confirmed table and a
  collider in the allowed follower-left wrist/gripper/finger set;
- no tested follower-left collider crosses the table bottom plane beyond the
  documented solver tolerance;
- the realized joint state does not converge to the infeasible below-table
  target;
- contact remains numerically stable for the required hold interval;
- all joint positions, velocities, transforms, and contact values remain
  finite;
- the articulation stays within authored joint limits;
- no disallowed robot/environment collision is used as a substitute support;
- no PhysX error invalidates the trial.

The exact solver tolerance and hold interval must come from official semantics
or a justified scale-aware derivation and be recorded in the runtime report;
they may not be widened after observing a failure.

### Gate Acceptance

The full gate runs three independent trials with a clean reset between them.
All three must pass. One failure makes the gate fail. The verifier writes:

- a bounded human-readable log;
- a machine-readable JSON report containing configuration, tested paths,
  target and realized states, contact pairs, minimum clearance, tolerances, and
  per-trial reasons;
- at least one screenshot of the verified contact state;
- an explicit overall `PASS` or `FAIL` marker.

The test posture is never saved into the approved Stage.

## Physics Inspector GUI Handoff

Only after the runtime gate reports three passing trials may the implementation
prepare the manual test session:

1. Discard the automated test posture and close its test runtime.
2. Resolve the existing Isaac process by exact PID and command line before
   stopping it; never use a broad process kill.
3. Start Isaac Sim Full with the reviewed application-native launcher.
4. Load the exact verified Stage and keep the main timeline stopped.
5. Select the Perspective viewport camera.
6. Open and bind the first Inspector panel to
   `/World/follower_left/vx300s_left/root_joint`.
7. Open and bind the second Inspector panel to
   `/World/environment/worldBody/user_confirmed_table`.
8. Configure the left panel for `Joint Drives Target Position`,
   `Fix Articulation Base`, and `Use QuasiStatic mode`, with gravity disabled.
9. Require populated left joint rows, non-disabled authoring state, and the
   exact selected paths.
10. Move the Isaac window to GNOME workspace index `2` without switching the
    user's current workspace.

The handoff Stage must be at its clean deterministic initial posture. The
launcher must not save joint state, drive targets, or Inspector authoring
results.

## Testing Strategy

Implementation follows test-driven development:

1. Add static contract tests that fail before the new physics layer exists.
2. Add pure tests for path classification, tolerance evaluation, three-trial
   aggregation, and fail-closed report logic; verify each test fails for the
   intended missing behavior.
3. Implement the minimum physics layer and verifier behavior to pass the tests.
4. Run Isaac-bundled OpenUSD composition checks.
5. Run the real Isaac runtime collision gate and require three passing trials.
6. Restart Full and verify process command, Stage URL, Inspector paths and rows,
   stopped timeline, workspace, and final screenshot.
7. Recheck the approved Stage post-change hash and unchanged hashes for the
   base geometry, collider, and tabletop layers; inspect the final git diff so
   no test posture or unrelated file is included.

Unit or static tests alone cannot claim the collision bug fixed. A GUI
screenshot alone also cannot claim the runtime gate passed.

## Error Handling and Stop Conditions

The workflow stops without GUI handoff if any of the following occurs:

- mandatory NVIDIA official Isaac MCP or required expert evidence is
  unavailable;
- Stage identity, composition, or required prim validation fails;
- CCD or PhysicsScene configuration is not effective in the Isaac runtime;
- no exact table/gripper contact is reported;
- any tested collider crosses the bottom-plane threshold;
- one of the three trials fails or times out;
- Inspector enters a repeated disabled/structural-change state;
- the final application is not the Full experience;
- the current dirty Inspector session cannot be discarded without risking an
  unintended Stage save.

Failures remain visible in bounded artifacts. The implementation must not
relax contact policy, table geometry, or tolerance merely to obtain `PASS`.

## Safety and Repository Boundaries

- Never connect to or command the real ALOHA hardware.
- Do not alter CAD source geometry or confirmed table dimensions.
- Do not change drive gains as part of this work.
- Do not save automated or manual Inspector poses.
- Preserve unrelated dirty and untracked worktree files.
- Keep large runtime logs under `.codex/artifacts/` and report only bounded
  summaries.
- Commit only reviewed source, tests, and documentation; do not commit runtime
  artifacts unless explicitly requested.

## Completion Evidence

Completion requires fresh evidence for every item below:

- official MCP/expert rationale recorded;
- static Stage and physics-layer tests pass;
- focused unit tests pass with prior RED evidence;
- Isaac runtime report shows exactly three passing trials and no failure
  reasons;
- contact paths include the confirmed table and an allowed follower-left
  collider;
- bottom-plane non-crossing and infeasible-target blocking checks pass;
- Full Isaac process, exact Stage URL, Perspective view, two bound Inspector
  panels, populated joint rows, stopped timeline, and workspace index `2` are
  verified;
- final screenshot shows the clean manual-test configuration;
- approved Stage post-change identity is revalidated, its intentional new hash
  is recorded, and no test posture was saved;
- repository diff contains no unrelated files.

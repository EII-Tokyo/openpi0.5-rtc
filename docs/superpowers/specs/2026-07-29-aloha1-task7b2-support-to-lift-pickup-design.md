# ALOHA1 Task 7B.2 Support-to-Lift Pickup Design

## Goal

Prove, with Isaac Sim 5.1 runtime evidence, whether the current
supplier-CAD follower-left gripper can lift the project-authored Bottle500
from the user-confirmed support surface and hold it for two seconds.

This is a digital, isolated Task 7B.2 diagnostic. It is not calibrated
sim-to-real dynamics, bottle insertion, final asset promotion, or Task 8.

## Approved Inputs

- Isaac Sim `5.1.0.0`, Kit `107.3.3`, PhysX `107.3.26`.
- Frozen signal-correspondence Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`,
  SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
- follower-left articulation:
  `/World/follower_left/vx300s_left`.
- User-confirmed support collider:
  `/World/environment/worldBody/user_confirmed_table`.
- Project Bottle500 USD:
  `assets/bottle_500ml/isaac/bottle_500ml_sim.usd`, SHA-256
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`.
- Referenced Bottle500 product prim: `/Bottle500`, with 41 collision prims.
- Bottle mass remains a session-only `0.020 kg` diagnostic override. The
  source-authored `0.025 kg` value remains unchanged and uncalibrated.
- The current supplier-CAD V2 handed finger colliders, drive, explicit
  symmetric finger targets, friction `0.7`, restitution `0`, `60 Hz`,
  solver settings, mimic disposition, and self-collision setting remain
  unchanged.
- Approach signal: replay the first 98 frames of the already validated Task 7A
  `follower_left:shoulder:positive` sweep (`180` sweep frames, target
  `1.1945033764839172 rad`). Starting from the frozen home target
  `-0.96 rad`, this produces an approach target of
  `0.2605069595575333 rad`.
- Lift signal: from that approach target, change only follower-left shoulder
  by the already validated `-0.08 rad` small-up direction, ending at
  `0.18050695955753326 rad`.

## Approaches Considered

### A. Frozen Task 7A Stage plus session-only Bottle500 — selected

This Stage contains the current follower-left supplier-CAD fingers, the
validated small-up signal correspondence, and the exact
`user_confirmed_table` support collider. The source Stage remains immutable;
Bottle500, materials, contact reporting, and diagnostic state are authored
only in the session layer.

### B. Extend the prior Task 5 isolated Stage — rejected

The prior static-hold Stage contains `/workcell/table/table`, but that path is
not the currently explicit `user_confirmed_table` boundary. Using it would
make the support identity less traceable and would require a second
cross-Stage transform argument.

### C. Add a synthetic support under the existing suspended bottle — rejected

This would make pickup easy by changing the environment to fit the grasp. It
would not prove that the current workcell support-to-grasp relation works.

## Dependency Boundary

The project imports `scipy.spatial` directly. Add `scipy==1.15.3` as an
explicit uv-managed project dependency, retaining Python `3.11.13` and the
resolved NumPy `2.4.0`. Do not install or modify system Python packages.

## Evidence-Derived Initial Placement

No bottle height or table height is manually guessed.

1. Load the frozen Stage and verify its path, hash, default prim, sublayers,
   references, follower-left articulation, handed finger collider prims, and
   `user_confirmed_table`.
2. Add an explicit session-only reference to `/Bottle500`.
3. Read the table collider world AABB and use its maximum Z as the support
   top.
4. Read Bottle500's composed world AABB and calculate the root translation
   that puts its minimum Z on the support top.
5. With no bottle present, replay the frozen Task 7A shoulder trajectory to
   frame 98 and read the runtime midpoint between the open left/right finger
   colliders. Use that approach-pose midpoint for bottle X/Y. The corrected
   fresh-process probe read back shoulder `0.2680850625 rad`, found the
   lowest finger point `0.0049044243 m` above the table, and reproduced the
   source command within `2.15e-5 rad`.
6. Return the robot to home/open before composing the bottle. This no-bottle
   geometry probe contributes placement evidence only and cannot contribute
   to pickup PASS.
7. This is a single evidence-derived diagnostic placement, not a measured
   workcell grasp pose. Do not search alternative arm poses.
8. Make the bottle dynamic before the support-settle phase. Kinematic setup
   is allowed only to author the initial transform and cannot contribute to
   pickup PASS.
9. Let the bottle settle under gravity. Require table contact, finite state,
   and a stable pose before closing the fingers.

If this single evidence-derived placement cannot establish bilateral finger
contact without changing robot home, geometry, friction, drive, mass,
timestep, solver, or support transform, classify the result as
`HARD_BLOCKER_SUPPORT_TO_GRASP_POSE`. Do not search arbitrary poses.

## Runtime Sequence

Each trial uses a fresh Isaac process/world reset:

1. Load and verify the frozen Stage.
2. Initialize follower-left at the approved home state with open fingers,
   replay the Task 7A approach trajectory without a bottle, and derive the
   approach-pose aperture midpoint.
3. Return follower-left to home/open, compose Bottle500, and place it on
   `user_confirmed_table` by the AABB
   procedure above.
4. Switch Bottle500 to dynamic and settle it on the support.
5. Record `support_settle`.
6. With fingers open, replay the exact 98-frame approach trajectory.
7. Close both fingers with the existing explicit symmetric targets.
8. Require physical left and right finger contact before lift and record
   `bilateral_contact_on_support`.
9. Keep the closed finger targets unchanged.
10. Ramp only the shoulder target from `0.2605069595575333` to
    `0.18050695955753326 rad`.
11. Record `lift_start`, the first verified support-clear frame, and
    `lift_end`.
12. Hold the arm and gripper targets for 120 steps (`2 s`) and record
    `hold_end`.

No SurfaceGripper, fixed joint, parent attachment, bottle teleport during
dynamic execution, or support movement is permitted.

## Machine Acceptance

A trial is `PASS` only when all checks pass:

- Bottle500 starts dynamic on `user_confirmed_table`.
- Support contact exists during the pre-grasp settle interval.
- Both supplier-CAD fingers establish physical bottle contact before lift.
- The approach trajectory is the frozen 98-frame Task 7A prefix; the lift
  shoulder delta is exactly `-0.08 rad`; all other arm targets remain at
  home.
- The bottle loses support contact after lift begins.
- Bottle bottom rises above the table top by more than the effective contact
  envelope and at least `0.005 m`.
- The bottle remains clear of the support at lift end and hold end.
- Bilateral finger contact persists through the accepted lift/hold interval.
- Full-interval drop after lift end is at most `0.010 m`.
- Contact impulses, poses, linear/angular velocities, joint targets, and
  readbacks are finite.
- No forbidden cross-follower, non-adjacent self, gripper-internal, or
  non-whitelisted environment contact is introduced.
- No persistent excessive penetration or numerical ejection occurs.
- No fixed constraint, SurfaceGripper, or parent attachment exists.

Group `PASS` requires 20/20 fresh-reset trials and one deterministic
signature. A smoke run is always `PARTIAL`, even if its physical trial passes.

Allowed user-confirmed finger/table contact before support clearance is
reported as workcell behavior, not an automatic failure. Bottle/table contact
before lift is required. Bottle/table contact after the support-clear gate is
a pickup failure.

## Failure Classification

Each failed trial is classified as exactly one primary result:

- `support_settle_failed`
- `bilateral_contact_not_established`
- `support_to_grasp_pose_blocked`
- `bottle_never_left_support`
- `contact_lost_during_lift`
- `continuous_slip_during_hold`
- `rotation_induced_escape`
- `support_recontact_after_lift`
- `normal_force_decay`
- `numerical_penetration_or_ejection`
- `forbidden_contact`
- `inconclusive`

The aggregate report uses only `PASS`, `FAIL`, `PARTIAL`, or `NOT_RUN`.

## Screenshot Evidence

The first trial captures raw and annotated images for:

- `support_settle`
- `bilateral_contact_on_support`
- `lift_start`
- `support_clear`
- `lift_end`
- `hold_end`

Each image records Isaac/Kit/PhysX versions, absolute Stage path and hash,
follower-left, Bottle500 source hash, frame/time, camera pose, shoulder
target/readback, finger target/readback, bottle Z, bottle-bottom/table-top
separation, contact state, and PASS/FAIL/PARTIAL.

Every raw and annotated image must be inspected individually with the vision
model. Retake images that hide either finger, the bottle/support interface,
the grasp region, or the vertical displacement. Screenshots are auxiliary;
runtime contact, pose, velocity, drop, and joint data remain authoritative.

## Outputs

- `configs/aloha1_task7b2_support_to_lift.yaml`
- `tools/aloha1_mapping/task7b2_support_to_lift.py`
- `tools/validate_aloha1_task7b2_support_to_lift.py`
- `tools/annotate_aloha1_task7b2_support_to_lift.py`
- `tests/aloha1_mapping/test_task7b2_support_to_lift.py`
- `reports/aloha1_mapping/aloha1_task7b2_support_to_lift.json`
- `reports/aloha1_mapping/aloha1_task7b2_support_to_lift_trials.jsonl`
- `reports/aloha1_mapping/aloha1_task7b2_support_to_lift_screenshot_review.json`
- `reports/aloha1_mapping/aloha1_task7b2_support_to_lift.md`

High-output logs and screenshots go under:

`.codex/artifacts/20260729-aloha1-task7b2-support-to-lift/`

## Protection And Promotion Boundaries

- Source USD, source CAD, imported assets, Task 7A Stage, current Task 7B
  reports, default/final colliders, and final configuration remain immutable.
- Task 7A status remains unchanged.
- Task 7B static-hold geometry A/B remains `PASS`.
- Task 7B.2 reports its own result and cannot silently promote assets.
- Asset-promotion readiness remains `PARTIAL`.
- Task 8 remains `NOT_RUN`.
- No real robot is connected, and `192.168.1.103` is not accessed.

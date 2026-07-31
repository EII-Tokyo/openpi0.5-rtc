# ALOHA1 Five-Pose Randomized IK Grasp Design

## Purpose

Replace the superseded five-position Bottle500 experiment with a stronger,
machine-verifiable test of the `follower_left` grasp pipeline. Each of five
frozen samples must combine:

1. a distinct horizontal Bottle500 tabletop position;
2. a distinct Bottle500 principal-axis direction;
3. a distinct collision-free left-arm initial pose whose end-effector
   position is visibly separated from the other four starts; and
4. a complete motion from that start to pregrasp, vertical descent, bilateral
   grasp, 0.200 m support clearance, and a 2 s hold.

Five successful samples establish evidence for those five sampled
start/object conditions. They do not prove global IK completeness over the
entire robot workspace.

## Confirmed Problem In The Previous Experiment

The previous report and all five raw-video contact sheets were reviewed.
They establish that:

- only Bottle500 world `XY` translation changed;
- every run recorded `rotation_unchanged: true`;
- all five preflight candidates used the same
  `ik_start_state.joint_readback_arm_rad`;
- all five target-orientation quaternions were identical; and
- the videos show essentially the same left-arm initial pose.

That experiment remains valid evidence for its original translation-only
scope, including the literal 4/5 physical result. It must not be described as
evidence for randomized bottle direction or randomized arm start pose.

## Chosen Design

The accepted design is a fixed-seed, preflight-filtered four-variable sample:

```text
(bottle geometric-center x,
 bottle geometric-center y,
 bottle principal-line yaw modulo 180 degrees,
 follower_left initial arm q / FK-derived EE pose)
```

The five formal samples are generated and frozen before any physical grasp
outcome is observed. Preflight may reject a candidate only for an objective
geometric, kinematic, limit, or collision failure. A runtime grasp failure
must remain a failed formal sample; it cannot be replaced by another random
draw.

Two alternatives were rejected:

- unconstrained independent random draws, because they can cluster into
  visually indistinguishable starts and directions; and
- hand-picked boundary cases, because they do not meet the user's request for
  reproducible random samples.

The selected fixed-seed method preserves randomness while enforcing explicit
diversity and legality gates.

## Coordinate And Bottle-Placement Semantics

The world origin is the user-approved center of the tabletop. World `+Z` is
the table normal. The table's vertical centerline in the accepted true-top
diagram is world `x=0`.

The user-approved spatial structure is:

- sample 1: Bottle500 geometric center lies on `x=0`, at a positive world
  `y`;
- sample 4: Bottle500 geometric center lies on `x=0`, at a different,
  negative world `y`;
- samples 2, 3, and 5: Bottle500 geometric centers are reproducible random
  points in the left-arm-reachable tabletop region with `x<0`; and
- all five bottles remain horizontal, dynamically settle on the table, and
  have distinct principal-line yaw values.

`x=0` applies to the Bottle500 **geometric center derived from CAD/USD
geometry**, not blindly to the Bottle USD prim translation. The implementation
must transform the local CAD-derived center into world coordinates and solve
the prim translation that places that center on the requested line.

The Bottle500 roll/support pose is preserved. A sampled yaw is applied about
world `+Z`:

```text
R_WO(sample) = Rz(delta_yaw) @ R_WO(nominal)
```

with the translation solved from the desired world geometric center. The
directed world `A` and `B`, axis vector, angle to `+Z`, lowest-point/table gap,
and rotated collision bounds must be recomputed from the resulting transform.

The approved Grasp Editor relation remains frozen:

```text
T_WG(sample) = T_WO(sample) @ T_OG
```

where `T_OG` is the already validated object-to-gripper transform. Pregrasp
and lift targets are derived from that sample-specific grasp pose; the
gripper orientation must not remain fixed when the bottle yaw changes.

## Bottle-Direction Diversity

Bottle principal-line orientation is compared modulo 180 degrees because the
visual requirement concerns the main axis line. The directed `A→B` vector is
still stored without discarding its sign.

The five frozen yaw values must:

- be generated reproducibly from the configured seed;
- span the half-circle rather than one narrow cluster;
- have a minimum pairwise circular separation of 25 degrees modulo
  180 degrees; and
- be visibly distinguishable in a true-top evidence frame.

The accepted visual design used representative directions near
15°, 47°, 82°, 121°, and 158°. These are design targets, not authoritative
runtime values. Exact values become authoritative only after the fixed-seed
sampler, rotated-bounds test, Lula preflight, and manifest freeze pass.

## Left-Arm Initial-Pose Diversity

Initial arm poses are sampled reproducibly in joint space from the explicit
`follower_left` DOF order and verified joint limits. The resulting
end-effector poses are computed with local Isaac Sim 5.1 Lula FK.

Each candidate start must satisfy:

- finite six-arm-DOF values in the explicit, unsorted joint order;
- runtime readback within the configured start tolerance;
- no initial self, table, bottle, frame, or non-target collision that blocks
  the task;
- no first-frame jump;
- a stable dynamic setup hold before motion starts;
- end-effector height above the table;
- a complete IK path from the sampled start to sample-specific pregrasp;
- FK position and orientation residuals within the existing validated IK
  gates;
- per-waypoint joint and velocity limits; and
- no path collision in the preflight representation or runtime collision
  evidence.

The five initial EE world positions must have a minimum pairwise Euclidean
distance of 0.050 m. This is an engineering evidence-diversity gate, not a
hardware calibration value. Video frame 1 and the initial hold segment must
also make all five starts visually distinguishable.

The sampled initial arm state is applied only during session setup before the
formal timeline. It is not written to the source Stage. The video begins
before the arm leaves that initial state.

## Preflight

Preflight runs in a fresh Isaac Sim 5.1 process against the frozen,
user-approved diagnostic Stage:

```text
/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda
```

Before opening it, the implementation must re-read and freeze:

- absolute path;
- SHA-256;
- default/root prim;
- sublayers and references;
- follower-left articulation prim;
- table prim; and
- Bottle500 geometry/configuration inputs.

The preflight candidate pool is deterministic. Selection proceeds in generated
order and may select a candidate only after all of these pass:

1. rotated Bottle500 geometry is fully inside the legal tabletop/free-surface
   region;
2. samples 1 and 4 satisfy geometric-center `x=0` within a numerical
   tolerance recorded in the config;
3. bottle yaw diversity against previously selected samples passes;
4. initial EE spatial diversity against previously selected samples passes;
5. initial arm state and initial collision gates pass;
6. start-to-pregrasp, vertical-descent, and vertical-lift IK paths pass;
7. FK residual, joint-limit, velocity-limit, and finite-value gates pass; and
8. the original Stage hash is unchanged after preflight.

Preflight never uses grasp success or hold outcome as a candidate-selection
criterion.

## Runtime Sequence

Each of the five frozen samples runs in a fresh Isaac Sim process:

1. apply the sampled arm start state and kinematic bottle setup in a
   session-only layer;
2. record and hold the distinct initial arm pose;
3. release Bottle500 to dynamic simulation and settle it on the table;
4. verify runtime Bottle500 center, `A/B`, yaw, horizontal support, and
   table gap;
5. move the arm from the sampled start to the sample-specific open pregrasp;
6. descend primarily along world `-Z`;
7. establish real left/right finger contact;
8. apply the unchanged closing preload;
9. lift primarily along world `+Z`;
10. reach at least 0.200 m support clearance; and
11. maintain the existing 2 s hold and 0.010 m drop gate.

No SurfaceGripper, fixed joint, parent attachment, runtime bottle teleport,
abnormal friction, or post-outcome sample replacement is allowed.

The following remain frozen across all samples:

- source Stage and composed default layers;
- supplier-CAD finger geometry and current diagnostic collider;
- friction and material binding;
- drive type, stiffness, damping, and max force;
- mimic/coupling mapping;
- bottle mass and collider;
- physics timestep and solver iterations;
- Grasp Editor `T_OG`;
- contact/hold acceptance gates; and
- Task 8 status.

Only Bottle500 `XY` center, Bottle500 yaw, and the left-arm initial state are
formal randomized variables.

## IK Evidence

For every sample and motion phase, save:

- initial arm q and runtime readback;
- initial EE position/orientation from Lula FK and USD/runtime readback;
- bottle center, `A`, `B`, directed axis, and yaw;
- sample-specific pregrasp, grasp, and lift transforms;
- every IK waypoint and warm start;
- solver status;
- FK position error and orientation error;
- joint-limit and velocity-limit results;
- non-target DOF drift;
- initial, waypoint, and runtime collision classifications; and
- a deterministic signature covering all frozen inputs and numerical
  results.

The final conclusion must say either:

```text
FIVE_SAMPLED_START_AND_BOTTLE_POSE_IK_GRASP_GATE = PASS
```

or:

```text
FIVE_SAMPLED_START_AND_BOTTLE_POSE_IK_GRASP_GATE = FAIL
```

It must not claim that five samples prove universal IK correctness.

## Video And Visual Review

Generate exactly one complete primary raw video and one complete annotated
video for each formal sample. Machine-only repeat runs may be added for
determinism, but they do not replace the five primary videos.

Every video must include:

- the entire left articulation from base through both fingers;
- an initial hold long enough to compare the five arm starts;
- Bottle500 and the table;
- move-to-pregrasp, vertical descent, bilateral contact, lift, and hold end;
- a synchronized gripper/Bottle500 close-up inset; and
- no camera crop that hides the arm base, bottle, or gripper.

Annotated evidence must show:

- sample ID and frozen seed;
- initial arm q and EE world position;
- Bottle500 geometric-center world coordinates;
- `A/B`, directed axis, and yaw;
- sample 1/4 `x=0` residual;
- pairwise EE/yaw diversity margins;
- target/readback, frame/time, contact state, bottle clearance, drop, and
  machine `PASS/FAIL`.

For each sample, create a full-video contact sheet and annotated keyframe
montage. The vision review must inspect the complete video evidence, not only
file existence or hashes. It must explicitly compare the five initial arm
poses and five Bottle500 directions. If any pair is visually
indistinguishable, the evidence gate fails; the formal sample is not silently
replaced.

## Acceptance

Overall `PASS` requires all five formal samples to pass:

- frozen-input and Stage-integrity gates;
- distinct Bottle500 yaw gate;
- distinct initial EE spatial-position gate;
- runtime initial-state readback;
- IK/FK residual, limit, velocity, and collision gates;
- dynamic horizontal Bottle500 settle;
- bilateral physical contact before lift;
- 0.200 m support clearance;
- 2 s hold with no more than 0.010 m drop;
- finite contact/velocity/pose values;
- raw and annotated video generation;
- visual-model evidence review; and
- user video confirmation.

Any one failed sample makes the aggregate result `FAIL`. Runtime failure does
not authorize resampling.

## Files And Isolation

The implementation uses new versioned files so the previous experiment remains
reproducible:

- new config:
  `configs/aloha1_grasp_20cm_five_pose_ik.yaml`;
- new sampler/preflight:
  `tools/plan_aloha1_grasp_20cm_five_pose_ik.py`;
- new execution harness:
  `tools/run_aloha1_grasp_20cm_five_pose_ik.py`;
- new visual finalizer:
  `tools/finalize_aloha1_grasp_20cm_five_pose_video_review.py`;
- focused tests:
  `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`;
- reports:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_preflight.json`,
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results.json`,
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.json`,
  and
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.md`;
- high-output logs, frames, screenshots, and videos:
  `.codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/`.

Existing reports, videos, configuration, source/default/final USD layers, and
colliders remain unchanged.

## Version And Safety Boundary

- Isaac Sim: `5.1.0.0`
- Kit: `107.3.3`
- PhysX: `107.3.26`
- IK/FK: local Isaac Sim 5.1 Lula implementation
- NVIDIA official Isaac MCP: mandatory before Isaac implementation changes
- no Isaac Sim latest or 6.0 APIs
- no real robot or `192.168.1.103`
- no leader, camera expansion, ROS, pipe insertion, workcell expansion, or
  Task 8
- no source/final asset promotion without separate user authorization

The implementation must stop before Isaac changes if direct NVIDIA official
Isaac MCP is unavailable. Independent static design and pure-Python tests may
continue.

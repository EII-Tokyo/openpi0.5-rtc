# ALOHA1 Bottle Collision Runtime Gate Design

## Scope

This diagnostic determines whether the project Bottle500 visual, collision
geometry, rigid body, collision filtering, and runtime response agree before
Grasp Editor or IK is allowed to continue. It is limited to Isaac Sim
5.1.0.0, Kit 107.3.3, and PhysX 107.3.26. It does not modify the Bottle500
source USD, follower source/configuration layers, final collider, friction,
drive, mimic, timestep, solver settings, or Task 8.

## Frozen inputs

- Review Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`
- Bottle product:
  `assets/bottle_500ml/isaac/bottle_500ml_sim.usd`, explicit product prim
  `/Bottle500`
- Bottle source SHA-256:
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`
- Existing supplier-CAD follower finger visual and collider prims from the
  frozen review Stage.

The experiment must recompute all hashes before and after every runtime run.

## Diagnostic architecture

The experiment creates session-only prims below
`/World/BottleCollisionDiagnosticSession`. The source bottle is referenced
with an explicit `/Bottle500` prim path. Two independent probes run in fresh
Isaac processes:

1. **Standard pusher probe**: a kinematic cube moves slowly into the horizontal
   dynamic bottle after table settle. This isolates Bottle500 rigid-body,
   collider, filtering, and response from the ALOHA controller.
2. **Follower finger probe**: one verified supplier-CAD follower finger moves
   slowly into the same bottle. This tests the composed finger
   visual/collider/body chain without attempting a grasp or lift.

Both probes hold mass, friction, restitution, timestep, solver iterations, and
collider authoring fixed. No SurfaceGripper, fixed joint, parent attachment,
or runtime bottle teleport is permitted after dynamic release.

## Machine evidence

Every frame records:

- bottle visual and collision world transforms and AABBs;
- pusher/finger visual and collision world transforms and AABBs;
- rigid-body enabled, kinematic, gravity, and mass readback;
- collision-enabled and approximation readback for every involved shape;
- filtered-pair and collision-group relationships;
- contact event, actor/collider paths, point, normal, impulse, and separation;
- bottle pose, linear/angular velocity, and displacement;
- deterministic signature.

The standard pusher gate passes only if a finite physical contact is reported,
the bottle moves in the expected push direction, no forbidden constraint is
present, and the visual/collider registration stays within the fixed geometry
tolerance. The finger probe is reported independently and cannot make the
standard pusher result green.

## Visual evidence

At pre-contact, first contact, maximum compression, and post-contact, save
paired images with identical camera pose and physics frame:

- normal visual image;
- Physics Collider Debug Visualization overlay image.

Required views are true top, side, and full-arm oblique for the finger probe.
The overlay must expose the Bottle500 collision envelope, table collider,
pusher or left/right finger collider, and their relative placement. Annotated
images include prim paths, frame/time, AABBs, contact point/normal, bottle
displacement, and PASS/FAIL/PARTIAL. A visual-model review is required for
every raw and annotated image. Occluded, cropped, unreadable, or
non-differentiated images are rejected and retaken.

## Classification

The root-cause result is exactly one of:

- `BOTTLE_COLLISION_MISSING_OR_DISABLED`
- `BOTTLE_RIGID_BODY_CONFIGURATION`
- `COLLISION_FILTERING_OR_MASK`
- `BOTTLE_VISUAL_COLLIDER_MISREGISTRATION`
- `FINGER_VISUAL_COLLIDER_MISREGISTRATION`
- `VIDEO_PHYSICS_FRAME_MISMATCH`
- `SOLVER_OR_TUNNELING_SUSPECTED`
- `COLLISION_PIPELINE_VERIFIED`
- `INCONCLUSIVE`

The previous smoke grasp remains non-acceptance evidence until this gate
passes. Grasp Editor, IK, and five-random-position grasp trials stay paused.

## Deliverables

- `tools/aloha1_mapping/bottle_collision_runtime_audit.py`
- `tools/audit_aloha1_bottle_collision_runtime.py`
- `tests/aloha1_mapping/test_bottle_collision_runtime_audit.py`
- `configs/aloha1_bottle_collision_runtime_audit.yaml`
- `reports/aloha1_mapping/aloha1_bottle_collision_runtime_audit.json`
- `reports/aloha1_mapping/aloha1_bottle_collision_runtime_audit.md`
- `.codex/artifacts/20260730-aloha1-bottle-collision-runtime-gate/`

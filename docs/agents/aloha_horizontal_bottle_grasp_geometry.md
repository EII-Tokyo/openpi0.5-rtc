# ALOHA Horizontal Bottle Grasp Geometry

Unless the user explicitly specifies another pose, ALOHA digital bottle
grasp tests must use the following default geometry and evidence contract.

## Initial Bottle Geometry

- Place the bottle horizontally on the table with physical support. Do not
  default to an upright bottle, a suspended bottle, or a bottle pre-positioned
  inside the gripper.
- Define the bottle longitudinal principal axis as the directed line `AB`.
  `AB` must be parallel to the table world-coordinate `XY` plane.
- Derive `AB` from the Bottle CAD/USD axis definition and the runtime world
  transform. Do not infer it from appearance.
- Record the world coordinates of `A` and `B`, the normalized axis vector, its
  angle to table normal `+Z`, and the gap between the bottle's lowest point
  and the table top.
- Gravity must be enabled and the bottle must physically settle on the table
  collider. The setup phase may temporarily use a kinematic body to establish
  the initial pose, but settle, contact, grasp, lift, and hold must all be
  dynamic. Setup evidence cannot contribute to grasp `PASS`.
- Do not guess the bottle roll about `AB`, the grasp location along `AB`, the
  bottle/table relation, or robot target joint values. When CAD, confirmed
  data, or physical measurement cannot determine one of these values, record
  a `HARD_BLOCKER` and continue work that does not depend on it.

## Gripper Geometry And Motion

- Move the open gripper above the bottle first, then approach primarily along
  world `-Z`. A lateral trajectory that enters the bottle from the side is not
  a substitute.
- The “gripper line” is the line joining the centers of the effective inner
  contact regions of the left and right fingers.
- The projection of the gripper line onto the table `XY` plane must be
  perpendicular to the bottle axis `AB`. The fingers must lie on opposite
  sides of the bottle body and close along its diameter.
- Both inward finger contact surfaces must face each other and the bottle
  body. Do not mirror a finger, exchange handedness, add an arbitrary
  180-degree rotation, or replace the finger embedded in the supplier
  assembly to create the desired appearance.
- The default grasp location is on a graspable cylindrical body region. Do not
  move it to the cap, neck, base, or a visibly tapered region without CAD
  evidence, user confirmation, or confirmed real data.
- The standard action order is:

  1. dynamic settle on the table;
  2. open gripper above the bottle;
  3. vertical descent;
  4. physical bilateral contact;
  5. closing preload;
  6. world `+Z` lift;
  7. support clearance;
  8. hold.

- Do not use a surface-grasp helper, fixed joint, parent attachment, bottle
  teleport during runtime, or abnormally high friction to fabricate a grasp.
- Finger/table contact is not inherently `FAIL`. Classify it from contact
  location, impulse, penetration depth, duration, and whether it prevents the
  bottle pickup. Do not impose an absolute no-contact rule detached from the
  real workcell.

## Real-Data Calibration Boundary

The current confirmed calibration boundary is:

```text
/home/eii/project/bottles_data/episode_18.hdf5
frames 208-244 inclusive
```

Use this window to verify open gripper, vertical approach, bottle clamping,
and lift onset. Do not substitute episode 19 for this calibration task.

## Machine Evidence And Screenshots

Acceptance requires runtime evidence for:

- bottle pose and the directed `AB` vector;
- gripper descent direction;
- the angle between the gripper line projection and `AB`;
- left/right contact pairs, normals, impulses, and separations;
- bottle clearance from the table;
- linear and angular velocity;
- drop;
- deterministic signature.

Capture at least a true top view and a side view. Annotate:

- `A` and `B`;
- the bottle axis;
- left and right fingers;
- the gripper line;
- descent direction;
- contact points;
- table;
- key angles.

Every screenshot must be reviewed individually with the vision model.
Screenshots remain supporting evidence and cannot replace machine data.

## Invalidated Legacy Geometry

Any old Task 7B.2 placement, approach, or lift result based on a bottle
standing upright on the table is not applicable to this default task. Recompute
and validate the support pose, vertical approach, bilateral grasp, and lift
from the horizontal-bottle geometry.

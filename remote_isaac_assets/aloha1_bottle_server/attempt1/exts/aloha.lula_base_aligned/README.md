# ALOHA Lula Base Aligned

Project-local Isaac Sim 5.1 diagnostic extension for the stationary ALOHA
follower-left arm. It fixes the missing world-base update in the stock Lula
Test Widget workflow without modifying NVIDIA's installed extension.

The extension starts inert. It will only command the six arm joints after:

1. the articulation and Lula files load with the expected six-joint order;
2. the Lula base pose is synchronized from the USD base-link Prim;
3. Lula FK and the USD end-effector pose pass the 1 mm / 0.5 degree gates;
4. a target is created at the current end-effector pose; and
5. the operator explicitly enables IK Follow.

The panel is organized as an operator-guided workflow rather than a collection
of independent controls:

1. prepare the paused robot;
2. synchronize and validate the Lula/USD frame contract;
3. load the Bottle `isaac_grasp` YAML and apply the audited object-local
   `(-5.5, -1.5, -10.0) mm` diagnostic correction; and
4. arm IK at the current end-effector pose before selecting one guarded target
   waypoint.

The `0. Repeatable random Bottle test` section also provides
`RESET BOTTLE TO INITIAL POSE`. It pauses the Timeline and atomically restores
Bottle500, BottleCap, and BottleThreadSlider to the canonical center
`(0, 0, 0.034) m` and yaw `0 deg`; clears all three rigid-body velocities;
restores Dynamic mode and the locked `THREADED` state; and leaves the arm and
gripper targets unchanged. The reset is authored only in the session layer and
never saves the Stage.

The automatic grasp button uses four explicit phases only:
`Sleep -> Plan Hover`, `Plan Hover -> Near (+10 mm)`, continuous physical
gripper closure, and `Lift -> Plan Hover`.  The two arm-motion phases before
contact each execute one continuous 50 Hz joint-reference route.  The former
automatic stops at PREGRASP and `+80/+40/+20/+0 mm` are not used.  The first
leg uses the operator's configured joint-step limit; the Near leg remains
capped at `0.010 rad / 20 ms`.  The button accepts either a verified random
pose or the verified canonical pose produced by RESET.

`GRASP VERTICAL: LIFT + ROTATE CAP TO +Z` shares the validated
Sleep/Hover/Near/continuous-close prefix, then replaces the last translation
with a conservative loaded route.  The Bottle centre first rises `150 mm`
while the Bottle remains horizontal.  Only after reaching that safe height
does rotation begin.  During rotation the end effector compensates for the
off-centre grasp so the Bottle centre stays at approximately the same height,
instead of sweeping back toward the table.  Because joint-continuous motion
and the off-centre grasp can produce a small height arc, preflight and runtime
both require the Bottle centre to remain at least `80 mm` above its grasp-time
height throughout the rotation stage; the final centre-height target remains
`150 mm`.
The long loaded lift uses a route-specific `12.5 mm` Mimic residual gate only
while bilateral Bottle contact is continuously monitored.  Gripper closure and
ordinary gripper operations retain the existing `10 mm` dynamic gate; loaded
motion still aborts on sustained unilateral contact or any non-finger contact.
The route preserves the measured post-close EE-to-Bottle transform, maps the
Bottle local cap axis to world `+Z`, limits loaded joint-reference increments
to `0.0075 rad` per 50 Hz period for both the lift and rotation phases, and rejects the route
before motion if any loaded end-effector step exceeds `4 mm` or the predicted Bottle
bottom drops below `20 mm`.  Runtime execution also rejects sustained loss of
bilateral finger contact, non-finger collision, excessive Mimic residual, or
an actual Bottle-bottom height below `15 mm`.  PASS additionally requires the
physical cap axis to finish within `10 deg` of world `+Z`.
The operational loaded route uses the repeatably accepted `0.0075 rad` step.
The experimental `0.010 rad` route was withdrawn after intermittent unilateral
contact loss; HOVER and NEAR retain their separate faster limits.
Every cap-to-`+Z` workflow, including near-centre Bottle poses, uses three
loaded legs: lift the horizontal Bottle by
`170 mm` (including approximately `20 mm` rotation-drop compensation),
translate it horizontally to the centre Hover, then rotate the cap
axis to world `+Z` without requesting another lift.  Every leg retains the
bilateral-contact, Mimic-residual, and non-finger-collision abort gates.  All
three loaded legs use the same route-specific `12.5 mm` Mimic structural gate;
this does not relax the bilateral-contact or non-finger-collision checks, and
ordinary gripper operations continue to use the `10 mm` gate.  All three loaded
legs are sampled at `0.005 rad` per 50 Hz period.  HOVER and NEAR retain their
faster configured limits; only motion after physical closure uses the lower
step so lift-off, transfer, and wrist reorientation do not tear down the
contact patch.
The route report records all six arm joints in order, including
`forearm_roll`, `wrist_angle`, and `wrist_rotate`, with their start, goal, and
delta values so wrist participation can be audited directly.
Automatic closure now requires `15` consecutive bilateral-contact report
samples before lift-off, rather than accepting the first short `5`-sample
contact transient.

Every automatic grasp now begins with a simulation-only recovery transaction.
The complete kinematic Bottle/Cap/thread assembly is temporarily parked at
world `z=0.75 m`, stale contact pairs are cleared, the gripper opens, and the
arm returns to Sleep.  The verified requested Bottle pose is then restored and
the contact tracker is cleared again before HOVER.  This prevents a failed
loaded trial from contaminating the next test.  A clean session with no prior
random placement uses the canonical Bottle pose automatically.
During this recovery-only open operation, residual arm motion is recorded but
does not block the transaction because the Bottle has already been parked and
the very next operation resets all six arm joints to Sleep.  The standalone
OPEN LEFT GRIPPER control retains its strict arm-stationary gate.

The orange `/World/ALOHAAlignedIKTarget` is a grasp-tool frame, not the native
Lula/URDF end-effector frame. Its `+Z` axis is constrained to world `+Z` for
the horizontal-Bottle top-down grasp. In the ALOHA URDF, native EE `+X` runs
from the wrist toward the fingertips, so the fixed frame contract is
`R_W_EE = R_W_Target * RotY(+90 deg)`. Consequently Target `+Z == world +Z`
and native EE `+X == world -Z`; the fingers point vertically down while the
orange Target retains the conventional upright axis display. The two frames
share an origin.

`PLAN VERTICAL HOVER (Bottle bottom L/3)` also checks
`dot(Target +Z, world +Z) == 1`, solves the final IK once while paused, samples
a continuous joint-space route at no more than `0.003 rad` per 20 ms control
period, and validates FK continuity, minimum end-effector height, and the final
pose before any motion.  Pressing Play then executes this prevalidated 50 Hz
reference directly.  It does not wait for every physics substep to settle and
does not repeat moving-frame Lula/USD alignment checks that could stall a valid
route.  Static alignment is still checked at the endpoints.

The old `HOVER +80 mm` definition is invalid for this corrected top-down
orientation: visual inspection showed the fingers already around/contacting the
Bottle. With the Bottle mouth persistently oriented along world `+X`, the
validated grasp station is exactly one third of the `0.206 m` body length from
the bottom (`68.667 mm`) and the selected vertical clearance is `160 mm`.
`SAFE PREGRASP +120 mm` remains available as a manual diagnostic waypoint.
Like HOVER, guided waypoints now solve their endpoint while Paused, select the
closest valid warm-start solution, sample a continuous joint route, and reject
excessive lateral deviation before Play. The automatic workflow now selects
`NEAR +10 mm` directly as its second and final pre-contact arm goal; successful
physical closure and lift still require runtime contact validation.
`ABORT` pauses the Timeline,
disables IK Follow, and returns the extension target to the current
end-effector pose.

Arm waypoint controls never command gripper DOFs. The separate gripper
preparation control commands only the active `left_finger` to the validated
open target `0.057 m`; `right_finger` remains exclusively Mimic-driven and the
fixed `gripper` DOF is not commanded. Grasp YAML contact values remain metadata
until incremental closure is implemented and validated. Generic online IK
target motion remains limited to 5 mm and 2 degrees per control step, with a
0.02 rad joint-command limit. The planned HOVER route instead uses its already
validated `0.01 rad / 20 ms` samples. This remains an IK diagnostic, not a
collision-aware motion planner. Do not save the Stage with the diagnostic
target present.

`Reset Left Arm to Sleep (Paused)` is a complete preparation transaction. It
pauses the Timeline, disables IK Follow, removes a stale extension Target and
HOVER plan, restores the six arm joints and Drive targets to the configured
sleep vector, zeroes arm velocity, preserves all gripper DOFs, publishes the
visible link transforms for 30 physics updates, and returns to Paused. An
active unsaved joint log remains the only intentional reset interlock.

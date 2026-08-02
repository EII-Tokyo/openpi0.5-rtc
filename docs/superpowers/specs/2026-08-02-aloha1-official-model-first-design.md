# ALOHA1 Official-Model-First Design

## Method correction

The model must be built from exact-model official evidence and deterministic
mathematics. Parameter sweeps, successful grasp outcomes and visual appearance
must not be used to infer missing hardware or physics values.

The only permitted evidence order is:

1. exact Stationary ALOHA / ViperX-300 6DOF official product source;
2. exact component-manufacturer manual;
3. pinned official Interbotix description, configuration and driver source;
4. supplier CAD and deterministic CAD-derived calculation;
5. NVIDIA Isaac Sim 5.1 definition and runtime readback;
6. physical measurement, only for quantities not published or derivable.

An experiment can reject an implementation. It cannot promote an unknown
number into an official parameter.

## Exact hardware identity

- robot product: Trossen Interbotix ViperX-300 6DOF / `aloha_vx300s`;
- follower_left and follower_right: independent instances of the same local
  robot product, not mirrored geometry;
- servo IDs 1-7: ROBOTIS XM540-W270 according to the Trossen product table;
- servo IDs 8-9: ROBOTIS XM430-W350;
- gripper actuator: ID 9 XM430-W350;
- geometry authority for the handed fingers: embedded v2 pair in
  `Simple Aloha Viper 2024-5-13.step`.

The Trossen page currently contains an explicit source conflict: its joint
limit table assigns wrist-angle/forearm-roll to IDs 6/7 in one order, while its
servo configuration table lists the names in the opposite order. This is not
silently resolved. The pinned official `vx300s.yaml`, URDF/Xacro, driver joint
order and published product-of-exponentials model must agree before the
mapping is accepted.

## Parameter contract

Every parameter record contains:

- exact product, component and role;
- source URL or repository/branch/tag/commit;
- local frozen path and SHA-256;
- license and redistribution boundary;
- source table/register/symbol/drawing dimension;
- value, units, coordinate frame, sign and applicability conditions;
- evidence class: `OFFICIAL_DIRECT`, `OFFICIAL_PINNED_SOURCE`,
  `CAD_DERIVED`, `MATHEMATICALLY_DERIVED`, `ISAAC_5_1_READBACK`,
  `PHYSICAL_MEASUREMENT`, or `HARD_BLOCKER`;
- derivation formula and input record IDs for a derived value;
- conflicts and selection state.

Formal model generation rejects `TEMPORARY_UNCALIBRATED`, inferred, fitted or
unproven values. Diagnostic values can remain in historical reports but cannot
enter the new candidate.

## Mathematical contracts

### Geometry and coordinates

For every CAD point:

\[
p_W = T_{W,R} T_{R,L} T_{L,C} p_C
\]

All transforms must be finite, use metres, have an orthonormal rotation, and
have determinant `+1` unless an explicitly proven handed B-Rep is involved.
No visual mirror or corrective 180-degree rotation is permitted.

### Kinematics

Forward kinematics is checked independently through both the pinned URDF chain
and Trossen's published product-of-exponentials model:

\[
T(\theta)=e^{[S_1]\theta_1}\cdots e^{[S_6]\theta_6}M
\]

Home pose, sampled legal poses and analytic/numerical Jacobians must agree
within tolerances derived from source precision and deterministic CAD
tessellation error, not from observed simulation success.

### Dynamics

The model contract is:

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)+\tau_f
=\tau_m+J(q)^T\lambda
\]

Link mass, COM and inertia come from the pinned official robot description or
a source-backed CAD/material derivation. Inertia tensors must be symmetric
positive definite and consistent with the parallel-axis theorem. Default
density is prohibited.

Motor torque/speed/current limits are derived from the exact XM540-W270 and
XM430-W350 official curves/register units plus the official joint/transmission
mapping. Raw DYNAMIXEL PID integers are not copied into PhysX stiffness or
damping. If no official one-to-one controller mapping exists, the PhysX drive
is classified separately and cannot be called manufacturer calibrated.

### Gripper

The CAD linkage and pinned Interbotix gripper code establish the mapping from
actuator coordinate to left/right finger displacement. The mapping must prove:

- official aperture range consistency;
- opposed direction and symmetry;
- legal endpoint clearance;
- no finger/finger or finger/internal-gripper intersection over the legal
  interval;
- the same result for both follower instances.

### Collision

The source B-Rep is the geometric truth. Each collider receives an explicit
approximation class and numerical error certificate:

- one-sided and symmetric sampled surface distance;
- maximum penetration and over-coverage;
- AABB and volume difference;
- topology/piece count;
- contact-region preservation for the finger inner surfaces;
- swept minimum clearance over the legal joint range.

No collider is accepted because a bottle grasp happens to pass.

## Runtime boundary

Isaac Sim is used only after the source and mathematical contracts pass. The
minimum runtime suite checks that Isaac Sim 5.1 readback equals the contract:

1. no-motion initialization;
2. one joint at a time;
3. one gripper open/close cycle;
4. one horizontal Bottle500 grasp/lift/hold smoke;
5. one fresh deterministic repeat when composition changes.

No parameter scan is part of acceptance. A mismatch produces a failure report,
three raw/annotated screenshot phases and a full-arm collision-enabled video.

## Task 8 boundary

The completed baseline inventory is retained as read-only evidence. It is not
permission to build the previously proposed visual-instancing candidate.
Performance optimization remains paused until the official parameter matrix,
mathematical model, collider certificate and static USD equivalence gates pass.

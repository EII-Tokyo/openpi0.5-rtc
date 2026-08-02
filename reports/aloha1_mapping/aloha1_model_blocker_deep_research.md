# ALOHA1 official-model blocker deep research

Date: 2026-08-03

Scope: Stationary ALOHA 1 follower (`aloha_vx300s`) with Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26

Research status: **PASS_RESEARCH_BOUNDARIES_ESTABLISHED**

Asset acceptance status: **unchanged**

The machine-readable companion is
`reports/aloha1_mapping/aloha1_model_blocker_deep_research.json`. The complete
source freeze is
`reports/aloha1_mapping/aloha1_model_blocker_research_source_manifest.json`.

## Executive conclusion

The five existing blockers were too coarse. They combined three fundamentally
different types of unknown:

1. quantities exactly derivable from pinned CAD, URDF, driver code, and Isaac
   5.1 schema;
2. numerical tolerances that must be derived by an offline convergence study;
3. physical material and thermal quantities that CAD cannot contain.

The research narrows the real blockers to two physical-data families:

- continuous motor/linkage thermal and efficiency behavior;
- friction/restitution of the exact ALOHA finger-pad and Bottle500 surface pair.

Collider acceptance and solver accuracy are not vendor parameters. They can and
should be closed with signed geometry certificates and numerical convergence,
without asking the real robot to reveal them through repeated grasp trials.

## Exact product and source chain

Trossen identifies the follower as the ViperX-300 6DOF ALOHA variant. The exact
ViperX page identifies XM540-W270 arm actuators, XM430-W350 wrist/gripper
actuators, gripper ID 9, and a published 42–116 mm gripper range:

- <https://docs.trossenrobotics.com/aloha_docs/1.0/specifications.html>
- <https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html>

The gripper's exact actuator manual is:

- <https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/>

The arm actuator manual is:

- <https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/>

The supplier CAD remains the SHA-256-pinned `Simple Aloha Viper
2024-5-13.step`. Its public download location is recorded, but no redistribution
license was found; this remains `UNKNOWN_HARD_BLOCKER`, so the STEP remains in
`.codex/artifacts` and is not made a Git deliverable.

## Important control-contract correction

The current project hardware YAML records the gripper's operating mode as
`pwm`, derived from static `modes.yaml`. That is not the complete ALOHA runtime
contract.

At official ALOHA commit
`4fa6b2c4428f5334441a7bee5ab2b2e8071cff93`,
`aloha/robot_utils.py::setup_follower_bot` reboots the follower gripper and
switches it to `current_based_position` before torque-on. ROBOTIS explicitly
describes this mode as controlling position and current/torque and as suitable
for grippers. The project's production YAML was not changed in this research;
the conflict is now formally recorded for a reviewed follow-up.

The current limit is pipeline-scoped:

| Source path | Raw limit | Derived current | Meaning |
|---|---:|---:|---|
| pinned `aloha_vx300s.yaml` | 200 ticks | 0.538 A | configuration value if no later override applies |
| official ALOHA acquisition/teleoperation scripts | 300 ticks | 0.807 A | explicit runtime override in those pipelines |

The conversion uses the ROBOTIS unit 2.69 mA/tick. Therefore “the ALOHA
gripper current is 200” and “the ALOHA gripper current is 300” are both
incomplete statements unless the exact runtime pipeline is named.

## Gripper force is mathematically derivable up to physical losses

Pinned Interbotix geometry gives horn radius (r=0.0275\,m), linkage arm
length (L=0.035\,m), and the driver mapping

\[
x(\theta)=r\sin\theta+\sqrt{L^2-r^2\cos^2\theta}.
\]

Its exact local Jacobian is

\[
J(\theta)=\frac{dx}{d\theta}
=r\cos\theta+
\frac{r^2\sin\theta\cos\theta}
{\sqrt{L^2-r^2\cos^2\theta}}.
\]

For symmetric fingers and an ideal lossless linkage, virtual work gives

\[
N_{each,ideal}=\frac{\tau}{2|J(\theta)|}.
\]

This is not an empirical fit. It follows from the pinned linkage geometry and
the two symmetric finger displacements.

At the three sampled carriage positions:

| Left finger x | Motor angle | J | Ideal N per motor Nm, each finger |
|---:|---:|---:|---:|
| 0.021 m | -0.024028 rad | 0.0266535 m/rad | 18.759 N/Nm |
| 0.039 m | 0.512732 rad | 0.0366366 m/rad | 13.648 N/Nm |
| 0.057 m | 1.090465 rad | 0.0222097 m/rad | 22.513 N/Nm |

Using the XM430 12 V stall ratio (4.1/2.3=1.783\,Nm/A) only as a
**momentary linearized upper estimate**:

| Current limit | Estimated motor torque | Ideal each-finger force range over samples |
|---:|---:|---:|
| 200 ticks / 0.538 A | 0.959 Nm | 13.09–21.59 N |
| 300 ticks / 0.807 A | 1.439 Nm | 19.63–32.39 N |

These numbers are not continuous grip-force claims. They omit linkage
efficiency, internal friction, contact-normal alignment, current-controller
dynamics, voltage variation, and thermal derating. They are useful because
they prove that the kinematic transmission is not an unknowable tuning
parameter. The remaining uncertainty is physical loss and thermal behavior.

For a 20 g bottle, ideal symmetric static balance is

\[
\mu_{min}=\frac{mg}{2N_{each}}.
\]

Substitution into the momentary ideal upper estimate gives very small
μ thresholds (roughly 0.003–0.0075), but these values must **not** be used to
claim that the current material is adequate: actual delivered normal force has
not been established by this ideal estimate.

## PhysX drive mapping: what is exact and what is not

The local Isaac 5.1 USD schema defines a force drive as an implicit
spring-damper effort, clamped by `maxForce`:

\[
\tau=K_p(q_{target}-q)+K_d(\dot q_{target}-\dot q).
\]

The locally installed Gain Tuner 3.0.6 implements

\[
f_n=\frac{1}{2\pi}\sqrt{\frac{K_p}{I_{eq}}},\qquad
K_p=I_{eq}(2\pi f_n)^2,\qquad
K_d=2\zeta\sqrt{I_{eq}K_p}.
\]

The versioned NVIDIA references are:

- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/joint_tuning.html>

ROBOTIS table gains are not SI `N·m/rad` or `N·m·s/rad`. The manual defines
internal coefficient conversions such as table P divided by 128 and table D
divided by 16, inside the servo controller. Copying those integers into PhysX
is dimensionally invalid.

A defensible mapping order is:

1. freeze the exact ALOHA runtime pipeline and current limit;
2. derive a bounded motor torque envelope from the exact ROBOTIS model;
3. map torque through the exact linkage Jacobian;
4. separately account for linkage efficiency and thermal limits;
5. obtain desired or identified closed-loop natural frequency and damping;
6. compute effective inertia at a declared robot configuration;
7. use the local Gain Tuner equations for SI stiffness and damping;
8. validate tracking, overshoot, saturation, effort, and contact force without
   changing unrelated variables.

## Collider acceptance must be a signed CAD contract

Existing machine evidence already proves the following:

- single-hull inward contact-surface recession: maximum 0.7978 mm, mean
  0.4646 mm;
- convex-hull inward crossing: about 0.6812 mm on both sides;
- convex-decomposition inward crossing: 0.5481 mm left and 1.3497 mm right;
- B-Rep/float numeric comparison allowance: 0.0004768 mm;
- FreeCAD linear tessellation deflection: 0.20 mm;
- the 68-piece CAD-derived contact-region compound cooks deterministically,
  but is not promoted;
- the five-pose swept collision report found zero unexpected overlap
  waypoints under its declared scope.

The right-side decomposition result is worse than the single hull. Therefore
`convexDecomposition` is not inherently “more correct”; approximation must be
accepted by signed surface measurements over the actual task contact band.

The required collider certificate should cover the swept Bottle500 contact
band over the legal finger trajectory and include:

- no crossing to the wrong side of the authoritative inward CAD face;
- signed normal offset, not only unsigned nearest distance;
- contact-normal angular error;
- left/right symmetry using their distinct handed B-Reps and placements;
- clearance to the gripper bar, carriage, shell, and opposite finger;
- Bottle500 swept-volume overlap/tunneling margin.

The numerical bound must combine independent terms: the 0.20 mm tessellation
deflection, measured transform residual, measured cooking deviation,
float/B-Rep allowance, bottle geometry uncertainty, and actual
`contactOffset/restOffset` readback. It must be declared before grasp results
are inspected. This turns the former collider `HARD_BLOCKER` into actionable
offline work rather than a missing manufacturer specification.

## Material friction cannot be recovered from CAD

NVIDIA 5.1 defines how Physics Materials bind and combine, but not the
coefficient for an unknown real surface pair:

- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html>

The local PhysX schema confirms combine-mode priority
`average < min < multiply < max`. A material may be bound at a collider or
overridden by a stronger parent binding. Thus “both values are 0.7” is not a
complete effective-friction statement unless binding strength and combine mode
are also read back.

No official source found in this search identifies both the ALOHA pad material
and finish and the project's Bottle500 material and finish. Generic
plastic/plastic tables, visual color, NVIDIA's Robotiq tutorial values, or a
coefficient selected because a grasp passes are not valid substitutes.

This absence was also checked locally rather than inferred from the web search:
the FreeCAD STEP probe exposes `ShapeMaterial`/`Material` property names for
the embedded v2 fingers but no populated material value, while
`configs/aloha1_bottle_asset.yaml` explicitly records `material: null` and
`PRIMARY_BOTTLE_MATERIAL_AND_WALL_PROPERTIES_NOT_CONFIRMED`. The existing 0.7
coefficient is already labeled `TEMPORARY_UNCALIBRATED`.

If supplier/manufacturing records cannot identify the exact pair, the minimum
resolution is a controlled incline, pull, or tribometer measurement with
normal load, direction, sliding speed, temperature, and surface condition
recorded. Until then, any simulated coefficient is
`TEMPORARY_UNCALIBRATED`.

## Continuous torque remains a real physical blocker

ROBOTIS supplies voltage-conditioned stall/no-load points, N–T performance
graphs, current units, temperature limits, and over-temperature shutdown. It
does not thereby supply a continuous thermal torque curve. Stall torque is a
momentary endpoint; an 80 °C protection threshold is not an allowable
continuous operating condition.

The remaining continuous claim needs either:

- an exact-model ROBOTIS continuous-duty/thermal application curve; or
- an authorized thermal test recording current, voltage, speed, applied load,
  duty cycle, ambient temperature, and actuator temperature.

The public ViperX dynamic-identification paper is useful supporting research:

- <https://doi.org/10.1016/j.mechatronics.2025.103419>

It is experimental arm identification, not a supplier calibration of the
ALOHA finger, its material pair, or this exact runtime controller. The linked
author repository has no license file and is retained only as research
evidence.

## Timestep and solver accuracy are offline numerical work

The numerical error budget does not require manufacturer or real-robot data.
It requires a predeclared convergence experiment with the entire physical
model frozen.

Recommended order:

1. fix high solver iterations and compare 60, 120, 240, 480, then if necessary
   960 Hz;
2. freeze the selected time step and sweep position iterations;
3. freeze position iterations and sweep velocity iterations.

Compare joint pose/velocity, contact onset, signed impulse integrals,
penetration, Bottle500 pose/velocity/drop, drive error and saturation,
energy/work, PhysX residuals, and fresh-process deterministic signatures.
Use analytic free-motion baselines and successive fine-grid differences to
establish the integration/noise floor, then combine it with the independently
derived geometry budget. Do not select tolerances after seeing which setting
makes the grasp pass.

## Blocker disposition

| Original blocker | Research disposition |
|---|---|
| continuous torque/speed/current/thermal curve | **remains HARD_BLOCKER, narrowed to continuous thermal/duty and loaded linkage loss** |
| PhysX drive physical derivation | **partially derived; calibrated response/efficiency/continuous saturation remain** |
| collider acceptance error budget | **reclassified to actionable offline signed-geometry derivation** |
| exact contact material properties | **remains HARD_BLOCKER** |
| numerical error budget | **reclassified to actionable offline convergence study** |

## Recommended next execution order

1. Correct the hardware-model source semantics in an isolated reviewed change:
   runtime `current_based_position`, with pipeline-scoped current limits.
2. Implement the Bottle500 swept-contact-band signed collider certificate and
   evaluate the existing CAD-derived compound candidate without promotion.
3. Run the predeclared timestep/solver convergence matrix.
4. Derive diagnostic force-drive candidates from verified inertia and a
   declared/identified closed-loop response.
5. Obtain exact material identity or perform an approved pair-friction test.
6. Obtain ROBOTIS continuous-duty data or authorization for a controlled
   thermal test.

No USD, collider, material, drive, controller, or final/default asset was
modified by this research.

# ALOHA Official Hardware Parameter Sources

Read this before answering or implementing any ALOHA hardware-parameter
question or mapping physical ALOHA hardware into URDF, USD, PhysX, controllers,
datasets, or validation gates.

## Current Confirmed Hardware Identity

- The current Stationary ALOHA follower product is the Trossen Robotics
  Interbotix ViperX-300 6DOF, represented by the project model name
  `aloha_vx300s`.
- `follower_left` and `follower_right` are two instances of that same robot
  product. Robot-local geometry is not mirrored; their workcell installation
  transforms are separate evidence and must not be guessed.
- The exact supplier assembly used to confirm the follower and gripper is
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step`,
  SHA-256
  `337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571`.
- The confirmed ViperX-300 6DOF gripper actuator is DYNAMIXEL ID 9,
  ROBOTIS `XM430-W350`. The supplier assembly's embedded left/right handed
  finger pair remains the geometry authority; do not replace it with a
  generic, legacy, independent, mirrored, or different-generation finger.
- Leader hardware is outside the current follower scope. If leader parameters
  become necessary, establish the exact leader product and component identities
  through the same source chain instead of inheriting follower values.

## Mandatory Official Source Chain

For every hardware question, start from the exact ALOHA product and traverse
the source chain. Do not begin with a guessed component or a configuration
snapshot:

The mandatory operating assumption is that the hardware parameter has an
official exact-model source. Failure to find it on the first ALOHA or Trossen
page means the source search is incomplete; continue through the product BOM,
the identified component manufacturer, its exact-model manual, drawings, and
official driver/configuration sources before considering a blocker.

1. Trossen Robotics ALOHA official documentation for the Stationary ALOHA
   product and follower/leader role.
2. Trossen Robotics Interbotix documentation for the exact robot model,
   including specifications, drawings, CAD, default joint limits, and exact
   actuator models.
3. The exact hardware manufacturer's official manual for each identified
   component. For DYNAMIXEL hardware this is the ROBOTIS e-Manual page for the
   exact model, not a related X-series motor.
4. The official Interbotix robot description, motor configuration, operating
   mode configuration, SDK, and driver implementation for the exact model.
5. The frozen local source copy used by this project, with repository,
   branch/tag, commit, license, local path, and file SHA-256 recorded.
6. NVIDIA Isaac Sim 5.1.0 / Kit 107.3 documentation and local runtime readback
   for the simulation-side meaning of any USD, PhysX, drive, mimic, contact, or
   solver parameter.

Primary entry points currently include:

- <https://docs.trossenrobotics.com/aloha_docs/2.0/>
- <https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html>
- <https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/>
- <https://github.com/Interbotix/interbotix_ros_manipulators>
- <https://github.com/Interbotix/interbotix_ros_core>
- `reports/aloha1_mapping/aloha_vx300s_official_reference_manifest.json`

## Required Evidence For Every Parameter

Before using a number or token, record:

- exact ALOHA product, robot model, component model, and hardware role;
- official page or repository URL;
- document/repository version, branch/tag, commit, and access date;
- license and redistribution boundary;
- local frozen path and SHA-256 when a local source is used;
- exact table, register, source symbol, Xacro/YAML key, or drawing dimension;
- units, coordinate frame, sign convention, operating mode, voltage, firmware,
  and other applicability conditions;
- whether the value is directly stated, runtime-read, mathematically derived,
  measured, or still blocked;
- the complete formula and input provenance for a derived value;
- conflicts between official sources, without silently selecting the value that
  makes a test pass.

An official value from a related model is not evidence for the current model.
Marketing copy, photographs, search-result snippets, forum posts, historical
simulation settings, and generic library defaults are not parameter authority.

## Hardware And Simulation Parameters Must Not Be Conflated

- A URDF mimic multiplier/offset, a real mechanical linkage, a DYNAMIXEL
  operating mode/PID/current/PWM register, and a PhysX mimic or drive parameter
  are different quantities.
- A hardware register may be mapped into simulation only through an explicit,
  unit-consistent physical/controller derivation supported by official
  documentation. For example, do not copy a DYNAMIXEL integer PID gain directly
  into a PhysX stiffness value.
- For simulation-only parameters without a one-to-one hardware register, use
  the NVIDIA Isaac Sim 5.1 definition together with exact-model manufacturer
  data and an auditable identification or validation procedure. Do not claim
  that a manufacturer directly specified a PhysX parameter.
- Temporary diagnostic values must remain explicitly marked
  `TEMPORARY_UNCALIBRATED` or `DIAGNOSTIC_ONLY_NOT_FINAL`; they cannot satisfy a
  calibrated or sim-to-real gate.

## Prohibition On Guessing From Machine 103

- Do not use files, launch-time values, register snapshots, or historical notes
  from `192.168.1.103` to guess product identity or manufacturer specifications.
- Access to `192.168.1.103` and the real robot still requires explicit user
  authorization and the remote/hardware safety documents.
- If the user later authorizes a read-only runtime snapshot, classify it as
  runtime configuration evidence only. Cross-check it against the exact-model
  official manuals; it does not replace them.
- Never control hardware merely to discover a parameter that should first be
  obtained from official documentation.

## Missing Or Conflicting Evidence

- Exhaust the mandatory official source chain before declaring a parameter
  unavailable.
- If exact-model official sources conflict, record every value, source,
  applicability condition, and the unresolved selection gate.
- If an exact value or defensible mapping still cannot be established, record a
  narrowly scoped `HARD_BLOCKER`, do not invent the value, and continue all
  independent work.
- Do not relax validation thresholds or copy a parameter from another robot to
  make a report pass.

# Physics Inspector CCD Collision Gate

Date: 2026-08-02

## Scope and frozen identities

This note freezes the evidence and acceptance contract for the ALOHA1
follower-left/table Physics Inspector test. It does not authorize or command
the real robot.

- Approved Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`
- Pre-change root-layer SHA-256: `5c9d1379da92cfcc858ab10ced587b31c117e797f4e5a943ed815f4d735168a7`
- Stage convention: `metersPerUnit = 1`, `upAxis = "Z"`
- Left articulation: `/World/follower_left/vx300s_left/root_joint`
- Confirmed table collider: `/World/environment/worldBody/user_confirmed_table`
- Confirmed tabletop vertical bounds: `[-0.015, 0.0] m`

## NVIDIA official evidence

The mandatory NVIDIA Isaac MCP review covered the Isaac Sim physics,
robot-simulation, and USD instructions before any Stage edit. The official
Isaac Sim 5.1 Physics Simulation Fundamentals documentation states that the
PhysicsScene controls physics steps per second and that a Stage without a
PhysicsScene uses 60 steps per second. NVIDIA's rigid-body documentation
identifies thin-object tunnelling as a consequence of discrete simulation and
requires sweep-based CCD at both the PhysicsScene and selected rigid bodies.

NVIDIA's Isaac API documents an additional hard constraint: traditional CCD is
not supported by GPU dynamics; a CCD request is ignored when GPU dynamics is
enabled. Therefore this diagnostic uses CPU PhysX explicitly:

```text
physxScene:broadphaseType = SAP
physxScene:enableGPUDynamics = false
physxScene:enableCCD = true
physxRigidBody:enableCCD = true  # every follower-left rigid body
```

The persistent PhysicsScene uses `physxScene:timeStepsPerSecond = 240` for normal
timeline operation. The collision verifier separately advances each Inspector
stress step at exactly `1/60 s`, matching the Inspector authoring simulation's
fixed step and exercising the coarse-step tunnelling case rather than hiding it
with 240 Hz stepping.

For contact identity, NVIDIA's `PhysxContactReportAPI` documentation defines a
force threshold and the official contact-sensor example applies the API to the
rigid body with threshold zero. The verifier therefore applies
`PhysxContactReportAPI` in an anonymous Session Layer, calls
`CreateThresholdAttr().Set(0)`, reads `get_contact_report()` after each fixed
step, and decodes both collider ids to exact USD paths. A generic contact count
cannot satisfy this gate.

Official references:

- [Isaac Sim 5.1 Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html)
- [Omni Physics rigid bodies and CCD](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)
- [Isaac API PhysicsContext CCD/GPU behavior](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/deprecated/isaacsim.core.api/docs/index.html)
- [PhysxContactReportAPI schema reference](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/106.1/class_physx_schema_physx_contact_report_a_p_i.html)

## Existing project expert evidence

The Phase 48 contact trace established two relevant regressions to avoid:

1. Contact counts can be false positives from unrelated stage pairs. Passing
   requires the exact confirmed-table/allowed-tip descendant path pair.
2. Disabling USD/Fabric updates makes live BBox readback stale and can hide
   ejection or penetration. USD updates remain enabled throughout the gate.

The verifier also separates Drive Target semantics from direct Joint State
editing. It ramps only the shoulder Drive Target; it does not teleport joint
state, change gains, change collision offsets, thicken the table, or save a
test pose.

Project reference:
`docs/aloha1_isaac_adaptation/54_phase48_first_contact_pair_trace_2026-07-18.md`.

## Frozen trial contract

```text
trial_count = 3
stress_step = 1/60 second
table_bottom_z_m = -0.015
bottom_crossing_tolerance_m = 0.0015
bottom_crossing_fail_plane_m = -0.0165
minimum_target_error = 2 degrees
hold_steps = 30
minimum_persistent_contact_steps = 10
contact_report_threshold = 0
```

The 1.5 mm bottom tolerance is 10% of the confirmed 15 mm tabletop thickness.
It is a fixed scale-aware solver tolerance, not a value to be widened after a
failed run. An allowed test collider below `-0.0165 m` fails immediately.

A trial passes only if all of these are true:

- at least one exact confirmed-table/allowed-tip contact pair exists;
- no tested allowed-tip collider crosses the frozen bottom fail plane;
- the commanded infeasible shoulder target remains blocked by at least
  2 degrees;
- exact contact persists for at least 10 of the 30 hold steps;
- all state is finite and every joint stays within its authored limits;
- CCD is active on the CPU pipeline;
- there are no allowed-tip contacts with a different environment collider;
- there are no PhysX errors invalidating the run.

Exactly three independently reset trials must pass. Any missing measurement,
unexpected path, PhysX error, or single failed trial makes the aggregate gate
fail closed.

## Persistent Stage policy

The reviewed override layer may author only the PhysicsScene and follower-left
CCD overrides. It must not change transforms, table geometry, materials,
collision filters, joint state, drive gains, limits, masses, or offsets. The
PhysicsScene uses zero gravity because the established Inspector authoring
handoff explicitly disables gravity and fixes the articulation base; this keeps
the automated test and manual control policy aligned without editing robot
state.

After the layer is composed, this report records the new root hash and verifies
that all referenced tabletop, collider, and geometry source-layer hashes are
unchanged. Runtime artifacts remain uncommitted under `.codex/artifacts/`.

## Hash ledger

The reviewed layer was composed with Isaac Sim 5.1 on 2026-08-02. The runtime
check resolved one schema-name defect in the written plan: Isaac's
`PhysxSceneAPI.GetTimeStepsPerSecondAttr()` reads
`physxScene:timeStepsPerSecond`, not a custom
`physics:timeStepsPerSecond` attribute. After correcting that authored name,
the composed values were `240`, `CCD=true`, `GPU=false`, and `SAP`; all 14
follower-left rigid bodies had CCD enabled; and the confirmed table world Z
bounds remained `(-0.015, 0.0)`.

```text
post_change_root_sha256 = 165093c3e7bf359b2ef5dbb595feb4ed976b194844830e70f387d6b882c1d6f2
physics_override_sha256 = 7f631cd99086f79ca3d88c4d189209f80b0acb71f68e01b3eecfb823909305c6
tabletop_alignment_sha256 = c9f8c656ed3a1b2722bfe0fb18dd2e5056da0320782e5dd3a7d4df2d7e845afc
diagnostic_collider_root_sha256 = 7c76a8970e6b0dbca562304c64e9bfcd2e1ca3e790d25dfa52baf2e510b82c68
base_collider_physics_sha256 = 352503bd2ccc8d4253fdc0561da67d2ec8bc92afba45d3112e677e4b5bb7ac9e
base_collider_geometry_sha256 = 06295762f1f513dae63a1188e360c209ac2bbbabef730db8a103eb3ea4b1df81
unchanged_source_layer_git_diff = clean
```

# ALOHA1 Official Historical Sleep Selection Plan

**Goal:** Replace the out-of-limit ROS 2 Humble Sleep command used by the
digital Home/Sleep correspondence experiment with the user-selected official
historical ALOHA variant Sleep command, then regenerate only the affected
digital and offline-preflight evidence.

**Selected command authority:** Interbotix official PR #189, ROS 2 historical
commit `dbc6aefb53e956181fe97f60474f1ad292491f0c`, file
`interbotix_ros_xsarms/interbotix_xsarm_control/config/aloha_vx300s.yaml`,
Sleep `[0, -1.80, 1.55, 0, -1.57, 0]` rad for the six arm joints. The current
Humble commit remains a comparison source and the local third-party aggregate
workspace is never the command authority.

## Safety and version boundary

- Keep the current Humble URDF limits, Isaac Sim 5.1 Stage, physics, drives,
  colliders, and final/default assets unchanged.
- Do not access `192.168.1.103`, ROS transport, serial devices, or motors.
- Mark this as an explicit cross-version command selection, not an untouched
  Humble default.
- Preserve the prior out-of-range run and its video as immutable historical
  failure evidence.

## Implementation sequence

1. Add RED tests for the exact selected vector, official historical git blob,
   source classification, and a fully accepted Interbotix group-limit gate.
2. Add git-blob verification to the manifest builder, switch the command
   constant/config, and retain the current Humble source as a comparison.
3. Generate the manifest twice and verify identical command signatures,
   1850 samples, all samples within limits, and three complete cycles.
4. Run two new independent Isaac Sim 5.1 headless processes with the frozen
   Stage and no physical/configuration changes other than the command vector.
5. Capture new full-arm normal and collision-overlay evidence for the selected
   trajectory, retain raw/annotated key frames, and visually review it.
6. Aggregate the digital gate, generate an offline-only real dry-run boundary,
   update README/TASK_STATE, run fresh regression checks, and commit in logical
   batches without pushing.

## Acceptance

- The exact old Sleep endpoint is legal and reached in all three cycles.
- All 1850 samples pass the modeled official Interbotix whole-group gate.
- Two fresh Isaac processes agree on the normalized numeric signature.
- The complete arm and distinct Home/Sleep states are visible in retained
  evidence; telemetry remains authoritative.
- Real execution remains `NOT_RUN_AUTHORIZATION_REQUIRED` and final/default
  assets remain unchanged.

# Current Project Verified ALOHA/Isaac Facts - 2026-07-17

## Scope

This note records what this repository has already verified locally, and what remains blocked.

Primary sources:

- `reports/aloha_isaac_replay/right_shoulder_audit/summary.json`
- `reports/aloha_isaac_replay/right_shoulder_audit/unattended_gate.json`
- `reports/aloha_isaac_replay/arm_only/.../replay_metrics.json`
- `reports/aloha_isaac_replay/controller_system_id/action_replay_arm_only_separated/action_replay_metrics.json`
- `assets/bottle_500ml/grasp/reports/summary.json`
- `assets/bottle_500ml/grasp/reports/reachability_results.json`
- `assets/bottle_500ml/grasp/reports/dynamics_results.json`
- `aloha_isaac_replay/adapters/standard_aloha.py`
- `aloha_isaac_replay/adapters/gripper_mapping.py`
- `examples/aloha_real/real_env.py`
- `examples/aloha_real/constants.py`
- `examples/aloha_real/robot_utils.py`

## Verified Passes

### Runtime DOF Identity

The right-shoulder audit passed these gates:

- runtime DOF identity;
- target readback index consistency;
- full 16-DOF target construction;
- right shoulder runtime limit;
- gravity-off hold;
- gravity-on hold;
- right-shoulder step response;
- left/right shoulder symmetry;
- readback physical consistency.

The report marks the system ready for controller parameter fitting, not ready for full grasp/contact/RL.

### Arm-Only Qpos Replay

Arm-only qpos replay passed on the tested sequence:

- 115 frames;
- max readback error: `0.0` radians;
- mean readback error: `0.0` radians.

This validates arm target replay for that path. It does not validate gripper contact, bottle dynamics, or pipe insertion.

### Corrected Arm Action Replay

The corrected arm action replay analysis passed the configured arm replay checks:

- mode: corrected arm absolute position targets;
- uses controller: true;
- uses action: true;
- gripper action not used;
- arm RMSE around `0.0155` radians in the referenced report.

This supports arm-level action replay, but it still excludes real gripper dynamics.

### Bottle Grasp Transform Math

The bottle grasp transform validation passed:

- `grasp_lower`: near-zero position error, zero rotation error;
- `grasp_mid`: near-zero position error, zero rotation error;
- `grasp_upper`: near-zero position error, zero rotation error;
- selected grasp: `grasp_mid`.

This proves the written grasp transforms are internally consistent. It does not prove the robot can execute the grasp dynamically.

## Verified Blocks

### IK / Reachability

Reachability is still blocked with:

`BLOCKED_NO_PROJECT_IK_CONFIG_FOUND`

This means the project has not yet proven a complete Isaac Sim 5.1 ALOHA left-arm IK/controller configuration that can drive the menagerie-style ALOHA asset to the grasp pose.

### Dynamic Grasp

Dynamic lift, flip, and external-force grasp validation are blocked. The reports do not mark these as passed.

### Collision and Contact

Collision/contact/reward validation is not complete. A visible mesh or static transform does not prove that:

- colliders are present;
- contact materials are correct;
- bottle/pipe collisions are stable;
- gripper finger collisions grasp the bottle reliably;
- friction and mass are plausible.

### RL Environment Readiness

The bottle grasp reports mark the dynamic grasp path as not ready for RL.

## Current Project ALOHA1 Semantics

The local canonical ALOHA convention is a 14D order:

```text
left_waist
left_shoulder
left_elbow
left_forearm_roll
left_wrist_angle
left_wrist_rotate
left_gripper
right_waist
right_shoulder
right_elbow
right_forearm_roll
right_wrist_angle
right_wrist_rotate
right_gripper
```

The real environment uses Interbotix robots:

- follower arms: `vx300s`;
- leader arms: `wx250s`.

The ALOHA1 home/sleep arm poses in the real environment are not the same as the Trossen AI examples.

## Practical Meaning

The project has a useful foundation:

- arm DOF replay is much better understood than before;
- selected bottle grasp transforms are mathematically valid;
- ALOHA1 canonical order is explicit.

But the missing pieces are exactly the pieces required for physical insertion simulation:

- executable IK;
- gripper closing dynamics;
- collision/contact;
- pipe contact;
- camera projection;
- reward generation.


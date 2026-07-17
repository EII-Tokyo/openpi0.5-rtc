# Phase 26: Minimal Drive and Native Failure Isolation

## Question

Phase 25 showed that the native ALOHA1 Isaac asset cannot reliably hold selected joints at a zero target. This phase narrows the failure:

1. Is the failure caused by the thin native wrapper entry point?
2. Is the failure caused by the test script or Isaac `SingleArticulation` target API?
3. Is the fixed base actually fixed?
4. Is the failure just a gripper/finger constraint side effect?

## Official Isaac Basis

The official NVIDIA Isaac MCP robot setup, physics, and USD guidance was consulted before adding the new gate.

Relevant guidance:

- a robot asset should have a clear root/default prim and physics layer structure;
- references must compose the intended default prim and referenced child prims;
- active joints need drives and physically meaningful limits/effort/damping;
- a robot that initializes as an articulation is not automatically controller-ready;
- simple dynamic gates should precede task-level grasp or RL simulation.

## New Tool

Added:

```text
aloha_isaac_replay/scripts/validate_minimal_revolute_drive.py
```

This script creates a temporary one-joint Isaac stage:

- one fixed base;
- one hinged link;
- fixed joint from world to base with `ArticulationRootAPI`;
- one revolute joint with the same nominal drive values used by the imported ALOHA1 arm joints:
  - stiffness: `625.0` authored, runtime reports about `35809.86`;
  - damping: `0.0`;
  - max force: `10.0`.

It then uses the same helper functions used by the ALOHA1 diagnostics:

```text
_set_full_state
_set_full_target
SingleArticulation.get_joint_positions
SingleArticulation.get_joint_velocities
```

## Result 1: Wrapper vs Interface Entry Point

Command:

```bash
codex-evidence --name phase26-interface-single-joint-zero-hold-four-joints -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --left-usd assets/isaac/original_stationary_aloha/generated/vx300s_left.usd \
  --right-usd assets/isaac/original_stationary_aloha/generated/vx300s_right.usd \
  --left-prim-path /World/left/root_joint/root_joint \
  --right-prim-path /World/right/root_joint/root_joint \
  --output-dir reports/aloha1_isaac_adaptation/phase26_interface_single_joint_zero_hold_four_joints_20260718 \
  --joint left:waist \
  --joint right:waist \
  --joint left:shoulder \
  --joint right:shoulder \
  --phase-offset 0 \
  --phase-steps 20 \
  --settle-steps 0
```

Result:

| Entry point | left waist final | right waist final | left shoulder final | right shoulder final |
| --- | ---: | ---: | ---: | ---: |
| native wrapper | -2.0092 | -1.1593 | 0.0569 | -0.2846 |
| generated interface | -2.0092 | -1.1593 | 0.0569 | -0.2846 |

The values are identical. Therefore, the thin wrapper is not the root cause of this dynamic drift.

## Result 2: Minimal Revolute Gate

Command:

```bash
codex-evidence --name phase26-minimal-revolute-drive-gate -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_minimal_revolute_drive.py \
  --output-dir reports/aloha1_isaac_adaptation/phase26_minimal_revolute_drive_20260718 \
  --steps 20 \
  --timeout-seconds 45
```

Result:

```text
status = PASS
dof_names = ["hinge"]
final_abs_error = 0.0
max_abs_qvel = 0.0
```

This proves the diagnostic control path is basically valid. The same Isaac runtime, the same `SingleArticulation` API, and the same target helpers can hold a simple revolute joint exactly still.

Therefore, the ALOHA1 failure should not be explained as a generic script bug.

## Result 3: Settle From Zero

Command:

```bash
codex-evidence --name phase26-native-zero-hold-after-settle -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --output-dir reports/aloha1_isaac_adaptation/phase26_native_zero_hold_after_settle_20260718 \
  --joint left:waist \
  --joint right:waist \
  --joint left:shoulder \
  --joint right:shoulder \
  --phase-offset 0 \
  --phase-steps 20 \
  --settle-steps 200
```

Result:

| Joint | qpos after settle | final qpos | interpretation |
| --- | ---: | ---: | --- |
| left waist | -2.0459 | -2.0472 | zero target is not the dynamic resting state |
| right waist | -0.3007 | -0.3111 | zero target is not held |
| left shoulder | 1.2569 | 1.2570 | moves near upper limit |
| right shoulder | 0.0438 | 0.0101 | this one is comparatively stable |

This shows that, in the current ALOHA1 asset, a nominal all-zero runtime state is not dynamically neutral.

## Result 4: Episode First Qpos

The real episode first frame is not out of runtime arm limits.

Sample targets:

| Joint | target | lower | upper | gate |
| --- | ---: | ---: | ---: | --- |
| left waist | 0.6642 | -3.1416 | 3.1416 | OK |
| left shoulder | -0.0982 | -1.8500 | 1.2566 | OK |
| right waist | -0.8253 | -3.1416 | 3.1416 | OK |
| right shoulder | 0.0261 | -1.8500 | 1.2566 | OK |

But short hold from that pose still fails.

Without gripper locking:

```text
gravity_off_hold = FAIL
max_abs_position_error = 7.3029
limit_violations = 1
```

With gripper/finger limit locking:

```text
gravity_off_hold = FAIL
max_abs_position_error = 4.7259
limit_violations = 1
```

Gripper locking improves the error but does not fix the failure. So gripper/finger constraints may contribute, but they are not the only cause.

## Result 5: Fixed Base Check

A base-motion probe recorded world poses for:

```text
/World/left/root_joint/puppet_left_base_link
/World/left/root_joint/puppet_left_shoulder_link
/World/right/root_joint/puppet_right_base_link
/World/right/root_joint/puppet_right_shoulder_link
```

The base links stayed fixed to numerical precision:

```text
left base translation:
  first: approximately [0, 0, 0]
  last:  approximately [0, 0, 0]

right base translation:
  first: approximately [0, 0, 0]
  last:  approximately [0, 0, 0]
```

The shoulder links rotated substantially while the base links stayed fixed.

Therefore, the failure is not explained by the fixed base floating away.

## Current Interpretation

The remaining failure is inside the ALOHA1 imported articulation chain.

Evidence already rules out:

| Hypothesis | Status | Evidence |
| --- | --- | --- |
| thin wrapper entry point is the cause | unlikely | wrapper and interface produce identical drift |
| generic `SingleArticulation` target API is broken | unlikely | minimal revolute gate passes exactly |
| arm targets are outside runtime limits | no | episode first-frame arm targets are in limits |
| fixed base is not fixed | no | base world pose stays fixed |
| gripper/finger is the only cause | no | locking improves but does not pass |

The most likely remaining causes are:

1. imported ALOHA1 joint inertias/masses/axes are not dynamically consistent enough for PhysX;
2. drive gains and max force are not tuned for the imported chain;
3. gripper/finger mimic/linkage constraints inject forces into the chain but are not the only issue;
4. unresolved visual/collider references indicate the asset composition is still not clean, even if it is not the only dynamic cause;
5. self-collision/collider settings still need a corrected all-stage collision-disable test.

## Decision

Do not proceed to bottle grasp, contact, reward, or RL in Isaac yet.

The next implementation gate should be:

1. create a corrected all-stage collision-disable probe that actually touches composed collision prims;
2. add a native ALOHA1 hold diagnostic that records base, shoulder, elbow, wrist world transforms and per-DOF qpos/qvel over time;
3. compare imported ALOHA1 mass/inertia/drive values against the Trossen Stationary AI asset that is known to run in Isaac;
4. tune or rebuild the ALOHA1 physics layer only after the above evidence identifies which physical property is inconsistent.

## Evidence

- `reports/aloha1_isaac_adaptation/phase26_interface_single_joint_zero_hold_four_joints_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase26_minimal_revolute_drive_20260718/minimal_revolute_drive_metrics.json`
- `reports/aloha1_isaac_adaptation/phase26_native_zero_hold_after_settle_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase26_episode_first_qpos_short_hold_no_lock_20260718/summary.json`
- `reports/aloha1_isaac_adaptation/phase26_episode_first_qpos_short_hold_lock_gripper_20260718/summary.json`
- `reports/aloha1_isaac_adaptation/phase26_native_base_motion_probe_20260718/base_motion_probe.json`
- `.codex/artifacts/20260718-011616_phase26-native-entrypoint-composition-audit`
- `.codex/artifacts/20260718-011658_phase26-interface-single-joint-zero-hold-four-joints`
- `.codex/artifacts/20260718-012137_phase26-minimal-revolute-drive-gate`
- `.codex/artifacts/20260718-012227_phase26-native-zero-hold-after-settle`
- `.codex/artifacts/20260718-012307_phase26-episode-first-qpos-short-hold-no-lock`
- `.codex/artifacts/20260718-012341_phase26-episode-first-qpos-short-hold-lock-gripper`

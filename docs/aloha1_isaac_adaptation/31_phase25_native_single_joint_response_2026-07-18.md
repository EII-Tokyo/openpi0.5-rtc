# Phase 25: Native Wrapper Single-Joint Dynamic Response

## Question

Phase 24 showed that full dynamic replay fails even when the controller target is changed from HDF5 `action[t]` to `observations/qpos[t + 1]`. This phase asks a narrower question:

Can the native ALOHA1 Isaac wrapper hold and step a single joint target?

If this fails, then full arm replay, grasping, bottle contact, and RL task simulation are not yet valid targets.

## New Tool

Added:

```text
aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py
```

The script:

1. opens the native ALOHA1 left and right wrapper USDs;
2. resolves the native articulation roots;
3. commands one selected runtime joint at a time;
4. records target, qpos, qvel, error, limits, and limit violations;
5. writes JSON, CSV, and Markdown reports.

Default joints:

```text
left:waist
right:shoulder
```

These were chosen because:

- `left:waist` produced the Phase 24 qpos-next instability;
- `right:shoulder` was sensitive to gain changes in Phase 23.

## Smoke Test Result

Command:

```bash
codex-evidence --name phase25-native-single-joint-response-smoke -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --output-dir reports/aloha1_isaac_adaptation/phase25_native_single_joint_response_smoke_20260718 \
  --phase-steps 80
```

Result:

| side | joint | status | max abs error | max final abs error | limit violations | direction ok |
| --- | --- | --- | ---: | ---: | ---: | --- |
| left | waist | FAIL | 2.2058 rad | 2.0970 rad | 0 | false |
| right | shoulder | FAIL | 1.6846 rad | 0.5894 rad | 15 | false |

## Micro Test: left waist hold target 0

Command:

```bash
codex-evidence --name phase25-left-waist-zero-hold-micro-v2 -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --output-dir reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_micro_v2_20260718 \
  --joint left:waist \
  --phase-offset 0 \
  --phase-steps 20 \
  --settle-steps 0
```

Result:

| metric | value |
| --- | ---: |
| target | 0.0 rad |
| start qpos | 0.0 rad |
| final qpos after 20 steps | -2.0092 rad |
| max absolute error | 2.0092 rad |
| max qvel | 25.4301 rad/s |
| limit violations | 0 |

The first rows show the drift starts immediately:

```text
step 0: qpos -0.0869 rad, qvel -13.8489 rad/s
step 1: qpos -0.3861 rad, qvel -11.1560 rad/s
step 2: qpos -0.4743 rad, qvel  +9.0551 rad/s
step 3: qpos -1.4662 rad, qvel -21.8282 rad/s
```

## Micro Test: left waist hold target 0 with damping override

Command:

```bash
codex-evidence --name phase25-left-waist-zero-hold-kd100 -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --output-dir reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_kd100_20260718 \
  --joint left:waist \
  --phase-offset 0 \
  --phase-steps 20 \
  --settle-steps 0 \
  --arm-kd 100
```

Result:

| metric | value |
| --- | ---: |
| target | 0.0 rad |
| final qpos after 20 steps | -2.3054 rad |
| max absolute error | 2.6216 rad |
| max qvel | 45.8912 rad/s |
| runtime damping | 100.0 |

Adding damping did not fix the zero-target hold; it made the immediate instability worse in this probe.

## Micro Test: left waist hold target 0 with collision-disable request

Command:

```bash
codex-evidence --name phase25-left-waist-zero-hold-collisions-off -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_native_single_joint_response.py \
  --output-dir reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_collisions_off_20260718 \
  --joint left:waist \
  --phase-offset 0 \
  --phase-steps 20 \
  --settle-steps 0 \
  --disable-robot-collisions
```

Result:

| metric | value |
| --- | ---: |
| disabled collision prims | 0 |
| target | 0.0 rad |
| final qpos after 20 steps | -2.0092 rad |
| max absolute error | 2.0092 rad |

This test did not actually disable any collision prims. It therefore does not prove that collisions are irrelevant. It proves that the current collision-disabling scan does not cover the collision prims composed by this wrapper.

## Root-joint structure check

The wrapper stages are thin sublayer wrappers:

```text
aloha1_left.usda  -> ../original_stationary_aloha/generated/configuration/vx300s_left_physics.usd
aloha1_right.usda -> ../original_stationary_aloha/generated/configuration/vx300s_right_physics.usd
```

Runtime/static inspection showed:

```text
left root_joint:
  type: PhysicsFixedJoint
  schemas: PhysicsArticulationRootAPI, PhysxArticulationAPI
  body0: []
  body1: /puppet_left_vx300s/puppet_left_base_link

left waist:
  type: PhysicsRevoluteJoint
  body0: /puppet_left_vx300s/puppet_left_base_link
  body1: /puppet_left_vx300s/puppet_left_shoulder_link
  localPos0: (0, 0, 0.079)
  localPos1: (0, 0, 0)
```

The same structure exists on the right side. This suggests the intended root joint is a fixed joint to the world/base, not a free-floating base. It still does not explain why the first dynamic step moves `left:waist` away from a zero target.

The inspection also found 11 collision prims in the composed source stage, but the earlier collision-disable helper did not touch them because they are not under the path pattern it scans. This needs a separate collision-composition audit before drawing a collision conclusion.

## Interpretation

This is a lower-level failure than action replay semantics.

The native wrapper currently cannot reliably hold `left:waist` at a zero target under dynamic stepping. Since the target is zero, gravity is disabled, and no full arm trajectory is involved, the failure should be investigated before any grasp or task-level simulation.

The evidence points to one or more of:

- root articulation / fixed-base setup;
- joint constraint configuration;
- drive target application semantics;
- initial state or velocity reset semantics;
- collision or internal constraint interaction inside the wrapper;
- PhysX drive gain/force configuration.

The evidence does not yet prove which one is the cause.

## Decision

Do not proceed to full arm dynamic tracking or grasp simulation yet.

Next gate:

1. inspect the left native wrapper articulation root and fixed-base chain;
2. compare left and right wrapper root/joint schemas;
3. repair the collision-disable audit so it covers composed `/colliders` prims and then retest;
4. test the same zero-hold on a minimal one-joint mock articulation to confirm the script logic is not the source;
5. only after single-joint hold passes should multi-joint dynamic replay be tuned.

## Evidence

- `reports/aloha1_isaac_adaptation/phase25_native_single_joint_response_smoke_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase25_native_single_joint_response_smoke_20260718/single_joint_response_timeseries.csv`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_micro_v2_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_micro_v2_20260718/single_joint_response_timeseries.csv`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_kd100_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_kd100_20260718/single_joint_response_timeseries.csv`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_collisions_off_20260718/single_joint_response_metrics.json`
- `reports/aloha1_isaac_adaptation/phase25_left_waist_zero_hold_collisions_off_20260718/single_joint_response_timeseries.csv`
- `.codex/artifacts/20260718-010346_phase25-native-single-joint-response-smoke`
- `.codex/artifacts/20260718-010504_phase25-left-waist-zero-hold-micro-v2`
- `.codex/artifacts/20260718-010536_phase25-left-waist-zero-hold-kd100`
- `.codex/artifacts/20260718-010825_phase25-left-waist-zero-hold-collisions-off`
- `.codex/artifacts/20260718-011059_phase25-root-joint-body-relations-flush`

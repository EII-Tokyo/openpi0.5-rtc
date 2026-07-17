# Phase 24: Native Wrapper qpos-next Target Tracking

## Question

Phase 23 proved that the native ALOHA1 wrapper can run dynamic action replay without exploding, but the arm tracking error was still too high. One possible explanation was that the HDF5 `action` field might not be the right dynamic target for Isaac.

This phase tests that hypothesis by replaying the same episode while using `observations/qpos[t + 1]` as the controller target instead of `action[t]`.

## Test Setup

- Episode:
  `local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5`
- Mapping:
  `configs/aloha/original_stationary_aloha_mapping.yaml`
- Left native wrapper:
  `assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda`
- Right native wrapper:
  `assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda`
- Articulation roots:
  - `/World/left/root_joint`
  - `/World/right/root_joint`
- Replayed steps: 40
- Physics dt: 0.02 s
- Steps per action: 1
- Gripper action: disabled

## Result

| Target source | Arm RMSE | Arm MAE | Max abs error | Best lag | Limit violations | Explosion check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `action[t]` | 0.1555 rad | 0.0813 rad | 1.1652 rad | 15 | 0 | PASS |
| `qpos[t + 1]` | 0.4818 rad | 0.1608 rad | 3.6207 rad | 15 | 27 | FAIL |

The first `qpos[t + 1]` limit violation was:

```text
step: 21
joint: left_waist
value: 4.2803 rad
lower: -3.1416 rad
upper: 3.1416 rad
```

The raw target range for `left_waist` was small:

```text
raw qpos-next left_waist range: 0.6581 to 0.6657 rad
known Isaac left_waist limit: -3.1416 to 3.1416 rad
```

So the failure is not because the requested left-waist target itself was outside the Isaac joint limit. The dynamic simulation drifted into an invalid state after stepping.

## Interpretation

Using `qpos[t + 1]` as the drive target does not solve the tracking problem. It makes the smoke test worse:

- tracking error increases;
- left waist becomes unstable;
- the simulation violates limits;
- the run no longer passes the no-explosion gate.

This means the remaining problem is not simply “HDF5 `action` is the wrong semantic field.” The evidence points to the native wrapper's dynamic drive behavior, joint dynamics, or per-joint response, especially around `left_waist`.

## Decision

Do not proceed directly to full grasp, bottle contact, or RL task simulation from the current dynamic replay result.

The next validation gate should be a single-joint dynamic response test:

1. command one joint at a time;
2. compare requested target, actual qpos, overshoot, settling time, and limit behavior;
3. start with `left_waist` because it caused the qpos-next failure;
4. include `right_shoulder` because Phase 23 showed it is sensitive to gain changes;
5. only after single-joint response is stable should multi-joint trajectory tracking be tuned.

## Evidence

- `reports/aloha1_isaac_adaptation/phase23_native_dynamic_tracking_smoke_20260718/action_replay_metrics.json`
- `reports/aloha1_isaac_adaptation/phase24_native_qpos_target_tracking_smoke_20260718/action_replay_metrics.json`
- `.codex/artifacts/20260718-005745_phase24-native-qpos-target-tracking-smoke`

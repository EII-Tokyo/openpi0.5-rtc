# Phase 23: Native Wrapper Dynamic Tracking Smoke

## Goal

Phase 21 and Phase 22 used direct `set_joint_positions()` followed by immediate readback. That proves DOF names and qpos mapping, but it does not prove physical tracking.

This phase runs a first dynamic smoke test:

```text
HDF5 action target
-> Isaac ArticulationAction
-> world.step()
-> compare simulated qpos with recorded real qpos after lag scan
```

This is closer to controller replay, but it is still not a full bottle grasp or insertion simulation.

## Script Change

Script:

```text
aloha_isaac_replay/scripts/replay_aloha_action.py
```

The script now accepts:

```text
--left-prim-path
--right-prim-path
```

This keeps old default behavior for older assets while allowing the native wrapper roots:

```text
/World/left/root_joint
/World/right/root_joint
```

## Input

Episode:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5
```

Native wrapper:

```text
assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda
assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda
```

Smoke settings:

```text
max_steps = 40
physics_dt = 0.02
steps_per_action = 1
base_separation = 0
gripper excluded
```

## Commands

Default native drive settings:

```bash
codex-evidence --name phase23-native-dynamic-action-tracking-smoke -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/replay_aloha_action.py \
  --episode local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5 \
  --mapping configs/aloha/original_stationary_aloha_mapping.yaml \
  --left-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda \
  --right-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda \
  --left-prim-path /World/left/root_joint \
  --right-prim-path /World/right/root_joint \
  --output-dir reports/aloha1_isaac_adaptation/phase23_native_dynamic_tracking_smoke_20260718 \
  --max-steps 40 \
  --base-separation 0 \
  --steps-per-action 1
```

High gain probe:

```bash
codex-evidence --name phase23-native-dynamic-action-tracking-kp1000kd100 -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/replay_aloha_action.py \
  --episode local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_0e2f4956f0a64d55acb3fa7363b2fdc4/episode.hdf5 \
  --mapping configs/aloha/original_stationary_aloha_mapping.yaml \
  --left-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_left.usda \
  --right-usd assets/isaac/aloha1_native_physics_wrapper/aloha1_right.usda \
  --left-prim-path /World/left/root_joint \
  --right-prim-path /World/right/root_joint \
  --output-dir reports/aloha1_isaac_adaptation/phase23_native_dynamic_tracking_smoke_kp1000kd100_20260718 \
  --max-steps 40 \
  --base-separation 0 \
  --steps-per-action 1 \
  --arm-kp 1000 \
  --arm-kd 100
```

Evidence artifacts:

```text
.codex/artifacts/20260718-005500_phase23-native-dynamic-action-tracking-smoke
.codex/artifacts/20260718-005535_phase23-native-dynamic-action-tracking-kp1000kd100
```

## Result

The dynamic smoke ran without joint-limit violation or simulation explosion, but it did not pass a meaningful tracking-quality gate.

| Probe | arm RMSE | arm max abs | best lag | joint limit violations | no explosion |
| --- | ---: | ---: | ---: | ---: | --- |
| native drive defaults | 0.1555 rad | 1.1652 rad | 15 steps | 0 | true |
| kp=1000, kd=100 | 0.3258 rad | 1.9855 rad | 15 steps | 0 | true |

Worst default-drive joints:

| Joint | RMSE | max abs | bias |
| --- | ---: | ---: | ---: |
| left_waist | 0.3893 | 1.1652 | -0.1473 |
| right_shoulder | 0.2966 | 0.5079 | 0.2659 |

High gain did not fix the issue. It improved `right_shoulder` but made `left_waist` much worse.

## Interpretation

This phase is a useful negative result.

It proves:

1. the native wrapper can execute dynamic stepping with `ArticulationAction`;
2. the simulation does not immediately explode for this short arm-only action replay;
3. arm target commands can be sent through the native wrapper root paths;
4. blindly increasing gains is not the correct next fix.

It does not prove:

1. controller fidelity;
2. action-space correctness;
3. base placement correctness;
4. collision/contact realism;
5. gripper correctness.

## Most Likely Causes

The poor tracking could come from several places:

1. `action` may not be the correct target to compare directly against later `qpos` for this native wrapper test.
2. The smoke used `base_separation = 0`, which is acceptable for arm DOF plumbing but not a realistic dual-arm physical scene.
3. Drive gains from the imported native wrapper may be unsuitable for replay tracking.
4. The lag scan hits the maximum allowed lag, which means the simulated response is not aligned well enough within the tested range.
5. Visual/collider reference warnings remain unresolved and may affect future contact tests, although they do not explain pure qpos readback.

## Next Gates

Do not proceed directly to bottle grasp simulation from this result.

Recommended next steps:

1. Add a qpos-target dynamic tracking script that drives recorded `qpos[t+1]` rather than HDF5 `action[t]`.
2. Compare action-target tracking and qpos-target tracking.
3. If qpos-target tracking passes but action-target tracking fails, the issue is action semantics or delay.
4. If qpos-target tracking also fails, tune drives and base/collision setup before grasp work.
5. Keep gripper disabled until the gripper semantics phase passes.


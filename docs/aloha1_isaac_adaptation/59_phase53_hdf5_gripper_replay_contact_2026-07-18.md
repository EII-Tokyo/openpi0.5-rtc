# Phase 53 HDF5 Gripper-Replay Contact Gate

## Question

Phase 52 proved that Bottle500 USD collision geometry can pass a clean local left-gripper contact gate when the finger target is hand-authored.

Phase 53 asks a stricter question:

Can a real recorded ALOHA1 HDF5 left-gripper qpos sequence drive the same local Bottle500 contact gate?

This is important because a hand-authored finger close command can hide replay-format mistakes. The next gate must consume the same kind of recorded signal that appears in real robot data.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

New optional inputs:

```text
--hdf5-gripper-episode <episode.hdf5>
--hdf5-gripper-start-frame <int>
--hdf5-gripper-end-frame <int>
--hdf5-gripper-max-frames <int>
```

When `--hdf5-gripper-episode` is provided, the validator:

1. reads only `observations/qpos`;
2. uses `qpos[:, 6]` for the left gripper and `qpos[:, 13]` for the right gripper;
3. maps the normalized ALOHA gripper qpos through `standard_gripper_qpos_to_isaac_fingers`;
4. uses the first frame as the open placement state;
5. replays later frames as the closing trajectory;
6. keeps the gate local: only gripper finger DOFs are driven, not the whole arm.

This is not full-arm replay yet. It is a controlled bridge from synthetic gripper closure to recorded gripper closure.

## Candidate Selection

The local HDF5 pool contains 248 key-region `episode.hdf5` files under:

```text
local_rlt_data/raw_from_103/rollouts/key_regions
```

The selected candidate was:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_15c193959d7d449783517a9c9d257529/episode.hdf5
```

Selection reason:

- task path is `twist_off_the_bottle_cap`, not `unknown_task`;
- 207 frames;
- left gripper qpos has a clear closing movement;
- raw qpos starts near `0.6552` and ends near `0.2610`;
- net change is about `-0.3942`.

Given the current gripper convention, lower normalized qpos means more closed.

## Command Artifact

```text
.codex/artifacts/20260718-141729_phase53-left-only-bottle-usd-hdf5-gripper-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase53_left_only_bottle_usd_hdf5_gripper_replay_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| object shape | `bottle_usd` |
| object USD | `assets/bottle_500ml/isaac/bottle_500ml_sim.usd` |
| control source | `observations/qpos` |
| HDF5 samples | 207 |
| replayed close steps | 180 |
| raw left gripper qpos start | `0.6551978` |
| raw left gripper qpos end | `0.2609603` |
| raw left gripper qpos range | `0.3956556` |
| both expected fingers contacted object | true |
| both expected fingers had `CONTACT_FOUND` events | true |
| wrong contact pairs | 0 |
| object close displacement | `0.003415 m` |
| max object displacement | `0.004407 m` |
| no explosion | true |

The first and last Isaac finger targets derived from HDF5 qpos were:

| Target | First | Last |
| --- | ---: | ---: |
| left finger | `0.0445871` | `0.0303946` |
| right finger | `-0.0445871` | `-0.0303946` |

## Interpretation

This is stronger than Phase 52.

It proves:

1. real HDF5 `observations/qpos` can drive the Isaac left gripper finger DOFs;
2. the dataset gripper-qpos convention maps to Isaac finger positions without changing action gripper fields;
3. Bottle500 USD collision remains stable under a recorded gripper-close trajectory;
4. both expected left fingertip proxies contact the object;
5. no non-target contact pairs appear in the left-only stage;
6. object motion remains small and bounded.

It still does not prove:

1. full left-arm qpos replay with bottle contact;
2. grasp lift;
3. bottle-pipe insertion;
4. correct dual-arm base transforms;
5. calibrated friction, bottle mass, bottle deformation, or pipe contact.

## Decision

Treat this as the current clean real-data local gripper replay gate:

```text
reports/aloha1_isaac_adaptation/phase53_left_only_bottle_usd_hdf5_gripper_replay_20260718/gripper_passive_contact_metrics.json
```

The important boundary is:

- Phase 52: hand-authored gripper close with real Bottle500 USD;
- Phase 53: real HDF5 gripper qpos close with real Bottle500 USD.

Phase 53 should be the preferred gate before moving to full-arm replay.

## Next Gate

The next gate should use a short real left-arm plus left-gripper HDF5 segment.

Minimum requirements:

1. drive left arm qpos and left gripper qpos together;
2. keep right arm disabled or absent;
3. keep the test left-only;
4. keep Bottle500 as `bottle_usd`;
5. verify qpos readback error;
6. verify both fingertip contacts;
7. verify wrong contact pairs remain zero;
8. verify object motion is finite and bounded.

Only after this passes should full dual-arm scene replay be reintroduced.

# Phase 54 HDF5 Left-Arm + Gripper Replay Contact Gate

## Question

Phase 53 replayed only the recorded left gripper signal from a real ALOHA1 HDF5 episode.

Phase 54 asks the next stricter question:

Can the same local Bottle500 USD contact gate consume a real recorded HDF5 left-arm plus left-gripper qpos sequence?

This matters because a gripper-only gate can pass even if arm-qpos mapping is wrong. A useful ALOHA1 Isaac bridge must eventually replay the same left-arm joint signal that appears in real robot data.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

The validator now supports:

```text
--hdf5-replay-mode gripper_only
--hdf5-replay-mode left_arm_and_gripper
--mapping configs/aloha/original_stationary_aloha_mapping.yaml
```

In `left_arm_and_gripper` mode, the validator:

1. reads `observations/qpos` from the selected real HDF5 episode;
2. maps the left arm qpos through the canonical ALOHA mapping file;
3. maps left gripper qpos through `standard_gripper_qpos_to_isaac_fingers`;
4. uses the first recorded frame as the placement state;
5. replays each subsequent recorded frame as a full left-arm plus gripper target;
6. keeps the gate local: only the left ALOHA1 stage is loaded, not the full workcell.

## Candidate

The same real HDF5 candidate from Phase 53 was used:

```text
local_rlt_data/raw_from_103/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-17/rl/key_region_15c193959d7d449783517a9c9d257529/episode.hdf5
```

It contains 207 qpos frames. The left gripper closes clearly:

| Signal | Value |
| --- | ---: |
| raw left gripper qpos start | `0.6551978` |
| raw left gripper qpos end | `0.2609603` |
| raw left gripper qpos range | `0.3956556` |
| raw left gripper qpos net | `-0.3942375` |

The left arm also moves during the replay:

| Signal | Value |
| --- | ---: |
| max absolute frame-to-frame left-arm qpos delta | `0.0107379` |
| max absolute net left-arm qpos delta | `0.6120584` |

## Command Artifact

```text
.codex/artifacts/20260718-142114_phase54-left-arm-gripper-bottle-usd-hdf5-full-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase54_left_arm_gripper_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.json
```

Human-readable report:

```text
reports/aloha1_isaac_adaptation/phase54_left_arm_gripper_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.md
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| overall pass | true |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| object shape | `bottle_usd` |
| object USD | `assets/bottle_500ml/isaac/bottle_500ml_sim.usd` |
| control mode | `hdf5_left_arm_and_gripper_qpos_replay` |
| replay source | `observations/qpos` |
| HDF5 samples | 207 |
| replayed steps | 206 |
| both expected fingers contacted object | true |
| both expected fingers had `CONTACT_FOUND` events | true |
| wrong contact pairs | 0 |
| object close displacement | `0.0903985` stage units |
| max object displacement | `0.0903985` stage units |
| no explosion | true |

The first and last Isaac finger targets derived from HDF5 qpos were:

| Target | First | Last |
| --- | ---: | ---: |
| left finger | `0.0445871` | `0.0303946` |
| right finger | `-0.0445871` | `-0.0303946` |

## Why The Displacement Limit Was Relaxed

Phase 53 gripper-only replay used a small displacement bound because the arm was fixed and only the fingers closed.

Phase 54 intentionally drives the left arm. The object can move more because it is following a changing recorded arm pose, not just being squeezed in place. Therefore the run used:

```text
--max-object-displacement 1.0
```

This is not a task-success threshold. It is only a numerical stability guard: the object must remain finite, bounded, and free of wrong contact pairs.

## Interpretation

This is stronger than Phase 53.

It proves:

1. a real HDF5 left-arm plus gripper qpos sequence can be mapped into the current ALOHA1 Isaac articulation;
2. the left arm can move while Bottle500 USD collision remains stable;
3. both left fingertip proxies contact the Bottle500 collision asset;
4. no non-target contact pairs appear in the left-only stage;
5. the replay does not numerically explode.

It still does not prove:

1. full dual-arm replay;
2. qpos readback tracking accuracy under force/contact;
3. grasp lift;
4. bottle-pipe insertion;
5. calibrated table, pipe, friction, bottle mass, bottle deformation, or camera geometry;
6. controller-level policy execution.

## Decision

Treat Phase 54 as the current real-data local left-arm replay contact gate:

```text
reports/aloha1_isaac_adaptation/phase54_left_arm_gripper_bottle_usd_hdf5_full_replay_20260718/gripper_passive_contact_metrics.json
```

The correct gate ordering is now:

1. Phase 52: hand-authored gripper close with real Bottle500 USD.
2. Phase 53: real HDF5 gripper qpos close with real Bottle500 USD.
3. Phase 54: real HDF5 left-arm plus gripper qpos replay with real Bottle500 USD.

## Next Gate

The next gate should add tracking and workcell realism without jumping directly to full RL:

1. report qpos target-vs-readback error for all left-arm DOFs;
2. run a short real HDF5 segment with gravity enabled;
3. add table collision while keeping the test left-only;
4. add pipe geometry as a fixed collision object;
5. only then reintroduce the right arm and full workcell replay.


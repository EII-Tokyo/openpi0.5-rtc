# Current109 + 37 Actor Lower-Right 4-Layer Strict-Policy 1000-Step Probe

Date: 2026-07-02

## Purpose

This probe checks whether the previous low critic AUC was mainly caused by the earlier data conversion issue.

The previous conversion regenerated `z_rl` / `next_z_rl` but kept old `proprio` / `next_proprio`. The strict-policy dataset fixes that by recomputing all four fields from the same lower+right 4-layer RLToken policy and the same replay row frame indices.

## Data

- Train: 117 shards, 4253 transitions
- Train labels: 36 success shards, 81 failure shards
- Holdout: 29 shards, 972 transitions
- Holdout labels: 7 success shards, 22 failure shards
- Feature shape: `z_dim=2048`, `proprio_dim=32`, `action_horizon=10`, `action_dim=14`
- Strict train root: `local_rlt_reencoded/current109_37_actor6000_20260630_lower_right_z2048_4layer_strict_policy/train`
- Strict holdout root: `local_rlt_reencoded/current109_37_actor6000_20260630_lower_right_z2048_4layer_strict_policy/holdout`

## 1000-Step Result

| Dataset | Eval split | Step | AUC | Q gap | Success Q mean | Failure Q mean | Floor violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict-policy | train | 1000 | 0.4628 | -0.0089 | -0.2570 | -0.2481 | 0.9871 |
| strict-policy | holdout | 1000 | 0.3444 | -0.0511 | -0.2826 | -0.2315 | 0.9979 |
| fixed_segments old conversion | holdout | 1000 | 0.3082 | -0.0680 | -0.3259 | -0.2579 | 1.0000 |

## Interpretation

The strict-policy data improves holdout AUC only slightly at step 1000: `0.3082 -> 0.3444`. It does not flip the core ordering problem. Failure transitions still receive higher Q than success transitions, so `q_gap` remains negative.

The train split also remains weak at step 1000: AUC is only `0.4628`, and `q_gap` is still negative. That means this is not just holdout noise. At 1000 steps, the critic has not learned a reliable success/failure ordering even on the data distribution it trained on.

## Conclusion

This 1000-step probe does **not** prove that the previous low AUC was mainly caused by the old conversion bug.

What is proven:

- The old conversion was indeed wrong because `z_rl` and `proprio` came from different encoders/sources.
- The strict-policy dataset fixes that source mismatch.

What is not proven:

- The AUC problem is not fixed by that correction at 1000 steps.
- The remaining failure is likely also affected by reward/return propagation, sparse terminal rewards, key-region slicing, or the critic training target.

## Next Step

Continue critic-only training to 10000 as a diagnostic, then evaluate all saved checkpoints. Do not start actor training unless a later critic checkpoint reaches positive `q_gap` and materially better AUC.

## Follow-Up: Target Critic Freeze Found

After the probe, code review found a separate critic-only training bug.

`training_stage=critic_only` disables actor updates. The old `train_step` only called target-network soft update inside the actor-update branch. Therefore, during critic-only training, the target critic stayed frozen at initialization. Non-terminal transitions used this frozen target critic for bootstrap:

`target = discounted reward_seq + gamma^horizon * target_critic(next_state, target_actor(next_state))`

This explains why terminal reward did not propagate backward through non-terminal transitions.

Fix:

- `src/openpi/models/rlt.py`: split target updates into `soft_update_target_actor()` and `soft_update_target_critic()`.
- `src/openpi/training/rlt_training.py`: update target critic after every critic update; update target actor only after actor updates.
- `src/openpi/training/rlt_training_test.py`: added a regression test proving target critic moves even when actor is not updated.

Verification:

- Local: `PYTHONPATH=. .venv/bin/python -m pytest -q src/openpi/training/rlt_training_test.py src/openpi/models/rlt_test.py` -> `19 passed`.
- 103: `PYTHONPATH=. .venv_eval/bin/python -m pytest -q src/openpi/training/rlt_training_test.py::test_rlt_train_step_updates_target_critic_without_actor_update src/openpi/models/rlt_test.py` -> `7 passed`.

1000-step target-fix probe:

| Dataset | Eval split | Step | AUC | Q gap | Success Q mean | Failure Q mean | Floor violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict-policy + target fix | train | 1000 | 0.6301 | 0.0987 | -0.0353 | -0.1341 | 0.8596 |
| strict-policy + target fix | holdout | 1000 | 0.3420 | -0.0489 | -0.1308 | -0.0819 | 0.8508 |

Interpretation:

- Train split now shows positive reward propagation: AUC improved from `0.4628` to `0.6301`, and q_gap flipped from `-0.0089` to `+0.0987`.
- Holdout is still poor: AUC stays around `0.34`, and q_gap remains negative.
- Therefore, the conversion mismatch was not the only issue. The target critic freeze was a real training bug, and after fixing it the critic learns the train split, but holdout generalization still needs diagnosis before actor training.

## Follow-Up: Target-Fix 1000-9000 Curve

Power loss interrupted the target-fix 10000-step run after the 9000 checkpoint. The available checkpoints are enough to show the trend.

Run root:

`local_rlt_runs/current109_37_lower_right4_strict_policy_critic10000_targetfix_20260702/critic_only_10000`

Train split:

| Step | AUC | Q gap | Success Q mean | Failure Q mean | Floor violation |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.6301 | 0.0987 | -0.0353 | -0.1339 | 0.8594 |
| 2000 | 0.7836 | 0.2635 | 0.2377 | -0.0257 | 0.5620 |
| 3000 | 0.9081 | 0.3678 | 0.3626 | -0.0052 | 0.4641 |
| 4000 | 0.9553 | 0.3853 | 0.3912 | 0.0059 | 0.3409 |
| 5000 | 0.9631 | 0.4079 | 0.3875 | -0.0204 | 0.5460 |
| 6000 | 0.9889 | 0.4155 | 0.3940 | -0.0215 | 0.5709 |
| 7000 | 0.9950 | 0.4066 | 0.3996 | -0.0070 | 0.4439 |
| 8000 | 0.9974 | 0.4273 | 0.4211 | -0.0062 | 0.4378 |
| 9000 | 0.9978 | 0.4538 | 0.4368 | -0.0169 | 0.5509 |

Holdout split:

| Step | AUC | Q gap | Success Q mean | Failure Q mean | Floor violation |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.3423 | -0.0488 | -0.1306 | -0.0817 | 0.8508 |
| 2000 | 0.4197 | -0.0273 | 0.0306 | 0.0578 | 0.2994 |
| 3000 | 0.5736 | 0.0254 | 0.1099 | 0.0845 | 0.1276 |
| 4000 | 0.5959 | 0.0266 | 0.1310 | 0.1044 | 0.0586 |
| 5000 | 0.5805 | 0.0193 | 0.1066 | 0.0872 | 0.1245 |
| 6000 | 0.5707 | 0.0174 | 0.1100 | 0.0926 | 0.0874 |
| 7000 | 0.5201 | 0.0091 | 0.1064 | 0.0973 | 0.0648 |
| 8000 | 0.5897 | 0.0247 | 0.1204 | 0.0957 | 0.0730 |
| 9000 | 0.5732 | 0.0209 | 0.1074 | 0.0864 | 0.1060 |

Interpretation:

- The critic now learns the train split strongly.
- Holdout improves only to AUC `0.5959` at step 4000, still below a reliable actor-training threshold.
- Later steps overfit train while holdout does not improve.
- Do not start actor training from this critic without addressing holdout split/data quality.

## Follow-Up: Holdout Failure Mode

A read-only holdout audit found no clear label inversion or train/holdout leakage:

- Train/holdout `key_region_id` overlap: `0`.
- Success shards have exactly one terminal positive reward at `(last_transition, 9)`.
- Failure shards have no positive reward.
- Each shard has one `done=True` at the last transition.

The stronger evidence is small holdout size plus distribution shift:

- Holdout: 29 shards, only 7 success shards.
- `z_rl` L2 direction flips: train success norm is higher than train failure, but holdout success norm is lower than holdout failure.
- `action-reference_action` delta is also shifted: holdout success looks closer to train failure than train success.

At best holdout checkpoint step 4000, the highest-Q failure rows concentrate in a few failure shards:

- `key_region_1638c0372ab64c049875339230d830ad...`, label 0, length 81, phase `rl`, late rows reach Q about `0.35`, often with `action_ref_delta_norm=0`.
- `key_region_82434b36b38d4d72862017088638937c...`, label 0, length 37, phase `rl`, late rows reach Q about `0.32`.
- `key_region_e922d16c51d540bfaaea10ef1535ef2e...`, label 0, phase `warmup`, rows reach Q about `0.31`.

The lowest-Q success rows concentrate in one long success shard:

- `key_region_151039164a6142e4a28d1ac15ff4dec8...`, label 1, length 79, phase `rl`, early rows have Q from about `-0.05` to near `0.03`, with large early `action_ref_delta_norm` around `0.36-0.53`.

This points to holdout instability driven by a few shards and by distribution shift, not an obvious conversion bug.

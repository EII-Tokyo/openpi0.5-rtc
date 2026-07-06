# Paper Anchor Today142 RLT Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 2026-07-06 online rollout batch into formal `paper_subsampled_anchor` replay, retrain a critic from scratch with the original clean VLA-token replay, continue the previous clean actor for 6000 steps, and publish an Obsidian report with Q/TD/rewarding curves.

**Architecture:** The conversion preserves verified executed action, reference action, reward, and done arrays from the online shard, but rebuilds `z_rl/proprio/next_z_rl/next_proprio` from the raw rollout frame anchors. Critic training consumes only explicit manifests whose shards declare `replay_state_grain=paper_subsampled_anchor`; actor training initializes from the previous clean actor and the selected new critic.

**Tech Stack:** Python, JAX/OpenPI policy checkpoints, NumPy NPZ replay shards, `scripts/train_rlt_offline.py`, `src/openpi/training/rlt_eval.py`, Matplotlib/HTML report assets, Obsidian markdown.

---

### Task 1: Build Paper Anchor Converter

**Files:**
- Create: `scripts/rebuild_online_rollout_paper_anchor_replay.py`
- Test: `scripts/rebuild_online_rollout_paper_anchor_replay_test.py`

- [ ] Add a converter that discovers `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions/twist_off_the_bottle_cap/2026-07-06/shards/*.npz`.
- [ ] Pair each shard with `/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/twist_off_the_bottle_cap/2026-07-06/rl/key_region_<id>`.
- [ ] Verify HDF5 action windows exactly match shard `action`.
- [ ] Extract VLA same-forward low/right token blocks at each unique current and next frame anchor.
- [ ] Encode low/right token blocks through the lower+right 4-layer RLToken autoencoder.
- [ ] Write rebuilt shards with `replay_state_grain=paper_subsampled_anchor`.
- [ ] Write an audit JSON/CSV including repeat fractions for `z_rl`, `proprio`, and `x=(z_rl, proprio)`.

### Task 2: Convert Today142 And Build Manifests

**Files:**
- Create: `local_rlt_manifests/paper_anchor_today142_plus_original_20260706/today142_paper_anchor_manifest.jsonl`
- Create: `local_rlt_manifests/paper_anchor_today142_plus_original_20260706/train_original116_plus_today142.jsonl`
- Use: `local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/train_manifest.jsonl`
- Use: `local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/holdout_manifest.jsonl`

- [ ] Run converter extract phase.
- [ ] Run converter encode phase.
- [ ] Validate every rebuilt shard has `z_rl_dim=2048`, `replay_state_grain=paper_subsampled_anchor`, and low exact adjacent repeat rate.
- [ ] Combine original clean 116 train shards with the rebuilt 142 shards.

### Task 3: Train And Select Critic

**Files:**
- Create: `local_rlt_runs/paper_anchor_today142_plus_original_critic10000_20260706/`

- [ ] Train critic from scratch for 10000 steps with `training_stage=critic_only`.
- [ ] Evaluate checkpoints against the original clean holdout manifest and the rebuilt today142 train data.
- [ ] Select one critic checkpoint even if metrics are imperfect, recording the selection rule.

### Task 4: Continue Actor And Select Actor

**Files:**
- Use: `local_rlt_runs/strict_td3_z_ablation_20260704/actor_from_vla_token_critic6000_actor5000_td3_20260706/inference_actor/00005000`
- Create: `local_rlt_runs/paper_anchor_today142_plus_original_actor6000_from_clean_actor_20260706/`

- [ ] Train actor for 6000 steps with `training_stage=actor_only`.
- [ ] Initialize actor from the previous clean actor checkpoint.
- [ ] Initialize critic from the selected 10000-step critic.
- [ ] Evaluate actor checkpoints and select the best available actor.

### Task 5: Curves, Web View, And Obsidian Report

**Files:**
- Create: `local_eval_assets/paper_anchor_today142_plus_original_20260706/`
- Create: `/home/eii/Documents/Notes/openpi0.5-rtc-reward-learning/70_Experiments/2026-07-06_today142_paper_anchor_RLT训练.md`

- [ ] Plot selected successful and failed trajectories with Q(data action), Q(actor action), TD target, reference/rewarding value, and actor delta.
- [ ] Include an audit plot proving `x=(z_rl, proprio)` no longer has cache-block repeats.
- [ ] Serve the report assets through a local HTTP server.
- [ ] Write the final Obsidian note with data counts, selected checkpoints, metrics, audit evidence, and the web link.

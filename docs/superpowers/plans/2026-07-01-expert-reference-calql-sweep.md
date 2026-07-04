# Expert Reference CalQL Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a four-way fixed-actor critic experiment comparing TD3 and CalQL with and without expert-demo reference replay.

**Architecture:** Add one focused sweep launcher that prepares current-only and current-plus-expert train/holdout manifests, then runs `train_rlt_online.py`, `evaluate_rlt_holdout.py`, and `compare_rlt_holdout_runs.py` for four fixed-actor critic conditions. The launcher does not change replay contents; expert demo replay remains expert `action == reference_action` and CalQL consumes the existing `reference_value` floor.

**Tech Stack:** Python dataclasses, tyro CLI, existing RLT online trainer/evaluator/comparison scripts, pytest, ruff.

---

### Task 1: Four-Way Sweep Launcher

**Files:**
- Create: `scripts/run_expert_reference_calql_sweep.py`
- Test: `scripts/run_expert_reference_calql_sweep_test.py`

- [ ] Write tests that assert the sweep builds exactly four specs: `TD3-current`, `TD3-current+expert`, `CalQL-current`, `CalQL-current+expert`.
- [ ] Verify the tests fail before the launcher exists.
- [ ] Implement manifest preparation helpers that combine current and expert manifests only for `+expert` runs.
- [ ] Implement train/eval/compare command builders that keep actor updates disabled and train critic for `10000` steps.
- [ ] Run the launcher unit tests and ruff.

### Task 2: Remote Data Readiness

**Files:**
- Source data: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_cam4_provenance_20260629`
- Remote target: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_cam4_provenance_20260629`

- [ ] Verify remote cwd is `/home/eii/openpi0.5-rtc-reward-learning`.
- [ ] Sync the 59 expert replay shards to 103 without touching code outside the project path.
- [ ] Verify all 59 expert manifest paths exist on 103.

### Task 3: Remote Experiment

**Files:**
- Remote script: `/home/eii/openpi0.5-rtc-reward-learning/scripts/run_expert_reference_calql_sweep.py`
- Remote output: `/app/rlt_online/expert_reference_calql_sweep_20260701_10k`

- [ ] Sync the launcher and test to 103.
- [ ] Compile the launcher inside the `rlt_online_trainer` container.
- [ ] Run only the train profile container with no robot services.
- [ ] Confirm each run reaches `10000` steps and writes holdout metrics.

### Task 4: Report Copyback

**Files:**
- Local report target: `rlt_online_reports/expert_reference_calql_sweep_20260701_10k`

- [ ] Copy only comparison and holdout evaluation reports back to the local project.
- [ ] Read `comparison_summary.md` and the per-run metrics.
- [ ] Report the best run and whether CalQL with expert reference improved AUC, q_gap, floor violation, and actor-advantage direction.

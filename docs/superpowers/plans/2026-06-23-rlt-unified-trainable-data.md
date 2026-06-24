# RLT Unified Trainable Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local clean key-region training data flow through one frozen manifest so UI counts, training, and evaluation use the same shard set.

**Architecture:** Add a small manifest module under `src/openpi/training` that normalizes container paths, validates replay shards, exports JSONL manifests, and provides summary counts. `RLTReplayStore`, offline training, and critic evaluation will accept a manifest path. The web backend will use the same path normalization/counting logic for Key Regions trainable counts.

**Tech Stack:** Python stdlib `sqlite3`, JSONL, NumPy replay shards, existing pytest suite, existing RLT training/evaluation scripts.

---

### Task 1: Unified Manifest Core

**Files:**
- Create: `src/openpi/training/rlt_trainable_manifest.py`
- Test: `src/openpi/training/rlt_trainable_manifest_test.py`

- [ ] Write failing tests for `/app/replay/rlt_key_regions_clean/...` path mapping, committed segment filtering, required NPZ key validation, and summary counts.
- [ ] Implement path mapping from container clean roots to local clean roots.
- [ ] Implement `build_manifest_from_segment_db`, `read_manifest_paths`, and `summarize_manifest`.
- [ ] Verify tests pass.

### Task 2: Training and Evaluation Entry Points

**Files:**
- Modify: `src/openpi/training/rlt_replay_store.py`
- Modify: `scripts/train_rlt_offline.py`
- Modify: `scripts/eval_rlt_critic_curves.py`
- Test: `src/openpi/training/rlt_replay_store_test.py`
- Test: `scripts/train_rlt_online_test.py`

- [ ] Write failing tests showing `RLTReplayStore` can load exactly manifest-listed paths and ignores unrelated directory files.
- [ ] Add `manifest_path` to `RLTReplayStore` and script args.
- [ ] Persist `manifest_path` and loaded manifest summary in training/eval metadata.
- [ ] Verify replay store and training argument tests pass.

### Task 3: Key Regions Count Fix

**Files:**
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/rollout_tree_test.py`

- [ ] Write failing backend test for trainable clean count when segment DB contains `/app/replay/.../manual/...` paths.
- [ ] Update backend count/stat code to use the unified manifest path mapping.
- [ ] Verify backend tests pass.

### Task 4: Generate Manifest, Train, Evaluate, Report

**Files:**
- Output: `local_rlt_manifests/*.jsonl`
- Output: `local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/`
- Output: `local_rlt_runs/rlt_unified_468_eval_report_zh.md`

- [ ] Generate a frozen manifest from `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3`.
- [ ] Verify manifest has 468 trainable shards, 145 success, 323 failure, 15817 transitions.
- [ ] Train TD3 with 5000 critic burn-in and actor through 15000 total steps.
- [ ] Evaluate against previous two-batch TD3, previous three-batch TD3, and new unified run.
- [ ] Write Chinese report with data definition, counts, metrics, and deployment recommendation.

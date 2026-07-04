# Canonical RLT 2048 Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one canonical lower+right 4-layer RLToken replay dataset root and machine-readable manifests without modifying legacy 512-dim replay shards.

**Architecture:** Add a small standalone manifest/copy builder that validates each `.npz` shard has `z_rl` and `next_z_rl` with dimension 2048, links or copies it into a canonical directory, and emits JSON/JSONL reports. Existing training code remains unchanged; future training should consume the generated manifest instead of scanning mixed legacy directories.

**Tech Stack:** Python stdlib, NumPy, pytest, existing RLT replay `.npz` manifest conventions.

---

### Task 1: Canonical Builder Test

**Files:**
- Create: `scripts/build_canonical_rlt_2048_dataset_test.py`

- [ ] **Step 1: Write tests covering valid shards, split manifests, duplicate IDs, and 512-dim rejection.**

Run: `uv run pytest scripts/build_canonical_rlt_2048_dataset_test.py -q`

Expected before implementation: import failure because `scripts.build_canonical_rlt_2048_dataset` does not exist.

### Task 2: Canonical Builder

**Files:**
- Create: `scripts/build_canonical_rlt_2048_dataset.py`

- [ ] **Step 1: Implement source parsing.**

Source spec format:

```text
kind|split|machine|batch|/absolute/source/root
```

Allowed `kind`: `rlt_raw`, `rlt_clean`, `expert`, `bootstrap`.

Allowed `split`: `unsplit`, `train`, `holdout`.

- [ ] **Step 2: Implement shard validation.**

For every `.npz`, require `z_rl.shape[-1] == 2048`, `next_z_rl.shape[-1] == 2048`, finite values, and a parseable `manifest`.

- [ ] **Step 3: Implement canonical placement.**

Write shards under:

```text
<canonical_root>/<kind>/<batch>/<split-or-all>/shards/<filename>
```

Use hardlink by default when possible; fall back to copy if cross-device linking fails.

- [ ] **Step 4: Implement manifests.**

Write:

```text
<manifest_root>/canonical_2048_all.jsonl
<manifest_root>/canonical_2048_train.jsonl
<manifest_root>/canonical_2048_holdout.jsonl
<manifest_root>/inventory.json
```

Each row includes `key_region_id`, `canonical_path`, `source_path`, `kind`, `split`, `machine`, `batch`, `reward`, `rows`, `z_dim`, `rl_token_config`, and `rl_token_checkpoint_path`.

- [ ] **Step 5: Run tests.**

Run: `uv run pytest scripts/build_canonical_rlt_2048_dataset_test.py -q`

Expected: all tests pass.

### Task 3: Local and 103 Execution

**Files:**
- Use: `scripts/build_canonical_rlt_2048_dataset.py`

- [ ] **Step 1: Run the builder locally for available lower+right 2048 sources.**

- [ ] **Step 2: Sync the builder to `192.168.1.103:/home/eii/openpi0.5-rtc-reward-learning`.**

- [ ] **Step 3: Run the builder on 103 for `/data/openpi0.5-rtc-reward-learning` plus the project-local bootstrap 146 source.**

- [ ] **Step 4: Copy 103 reports/manifests back to local.**

### Task 4: Final Verification

- [ ] **Step 1: Verify no canonical row has 512-dim `z_rl`.**
- [ ] **Step 2: Verify train and holdout key-region IDs do not overlap.**
- [ ] **Step 3: Report local and 103 canonical counts by source kind, split, and reward.**

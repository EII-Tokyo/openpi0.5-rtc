# Key Region Rescore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow operators to change an existing key region reward and make the changed reward affect future Q training.

**Architecture:** Add a backend file-level rescore helper next to the crop helper so manifest and NPZ updates stay consistent. Expose a small API endpoint that updates the replay shard, rollout manifest, and SQLite ledger, then add per-card UI actions on the Key Regions page.

**Tech Stack:** FastAPI backend, SQLite segment ledger, NumPy NPZ replay shards, React/TypeScript frontend.

---

### Task 1: Backend File Rescore

**Files:**
- Modify: `voice_assistant_web/backend/app/rlt_key_region_crop_test.py`
- Modify: `voice_assistant_web/backend/app/rlt_key_region_crop.py`

- [ ] Write a failing test that verifies rescoring rewrites `manifest.json`, NPZ `manifest`, and terminal `reward_seq`.
- [ ] Run the targeted pytest and confirm it fails because `rescore_key_region_files` is missing.
- [ ] Implement `rescore_key_region_files(rollout_dir, shard_path, reward)`.
- [ ] Run the targeted pytest and confirm it passes.

### Task 2: Backend API and Ledger

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/rlt_segment_ledger.py`
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Modify: `voice_assistant_web/backend/app/main.py`

- [ ] Add a request model for key region rescore using `reward`, `source`, and `reason`.
- [ ] Add a ledger method that updates reward for an existing segment without changing its committed shard path.
- [ ] Add a control-store method that records the event, recomputes stats, and publishes a review event.
- [ ] Add `POST /api/rlt/key-region/{key_region_id}/rescore` that locates the review record, rewrites files, updates ledger, and returns current RLT state.

### Task 3: Frontend Controls

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/components/RolloutBrowser.tsx`
- Modify: `voice_assistant_web/frontend/src/styles.css`

- [ ] Add `rescoreKeyRegion` API helper.
- [ ] Add card-level `Score 1` and `Score 0` controls near the existing reward readout.
- [ ] Disable the selected score while the request is pending and refresh review data after success.
- [ ] Keep styles scoped to the Key Regions card controls.

### Task 4: Verification

**Files:**

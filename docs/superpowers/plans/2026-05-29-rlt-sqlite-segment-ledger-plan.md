# RLT SQLite Segment Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SQLite-backed RLT key-region segment ledger so incorrectly marked start/end/reward data can be discarded before training or voided after writing.

**Architecture:** Use Python stdlib `sqlite3` in the backend so no new database service is needed. Backend becomes the source of truth for segment lifecycle (`started`, `ended`, `accepted`, `discarded`, `committed`, `rejected`, `voided`); recorder emits committed/rejected acks; replay store filters `.npz` shards by embedded `train_eligible` and `voided` manifest flags, and front-end exposes discard/confirm/void controls.

**Tech Stack:** FastAPI, Pydantic, SQLite stdlib, Redis pub/sub, React/TypeScript, pytest.

---

### Task 1: SQLite Segment Ledger

**Files:**
- Create: `voice_assistant_web/backend/app/rlt_segment_ledger.py`
- Modify: `voice_assistant_web/backend/app/config.py`
- Test: `voice_assistant_web/backend/app/rlt_segment_ledger_test.py`

- [ ] Write failing tests for creating a segment, transitioning states, recording committed shard metadata, voiding a segment, and recomputing warmup/online counters from committed non-voided rows.
- [ ] Implement `RLTSegmentLedger` with tables `segments` and `segment_events`.
- [ ] Add `rlt_segment_db_path` config defaulting to `/app/rollouts/rlt_segments.sqlite3`.
- [ ] Verify with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest voice_assistant_web/backend/app/rlt_segment_ledger_test.py -q -p no:cacheprovider`.

### Task 2: Backend State Machine

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/rlt_control_test.py`

- [ ] Write failing tests that `discard` during `key_region` or `await_score` does not publish `score`, that `score` moves to `pending_replay`, that `confirm` publishes `score` and records `accepted`, and that `void` prevents committed segments from contributing to counts.
- [ ] Add request models for discard and void.
- [ ] Add store methods `discard_key_region()`, `confirm_key_region()`, `void_segment()`.
- [ ] Process `rlt_replay_segment_committed/rejected/voided` acks idempotently through SQLite.
- [ ] Add routes `/api/rlt/key-region/discard`, `/api/rlt/key-region/confirm`, `/api/rlt/key-region/{id}/void`.
- [ ] Verify backend tests.

### Task 3: Recorder And Replay Store Filtering

**Files:**
- Modify: `examples/aloha_real/rlt_key_region_recorder.py`
- Modify: `packages/openpi-client/src/openpi_client/runtime/runtime.py`
- Modify: `src/openpi/training/rlt_replay_store.py`
- Test: `examples/aloha_real/rlt_key_region_recorder_test.py`
- Test: `src/openpi/training/rlt_replay_store_test.py`

- [ ] Write failing tests that discard events clear active/pending recorder state without writing replay, committed acks include `train_eligible=true`, and replay store skips shards whose manifest has `train_eligible=false` or `voided=true`.
- [ ] Add subscriber hook `on_key_region_discard` and runtime event mapping.
- [ ] Change recorder ack type from generic `written` to `rlt_replay_segment_committed` or `rlt_replay_segment_rejected`.
- [ ] Add `train_eligible`, `voided`, and `segment_status` fields to manifest.
- [ ] Verify recorder/replay store tests.

### Task 4: Frontend Controls

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/components/RLTControlPanel.tsx`
- Optionally modify: `voice_assistant_web/frontend/src/components/RolloutBrowser.tsx`

- [ ] Add API methods `discardKeyRegion`, `confirmKeyRegion`, `voidKeyRegion`.
- [ ] Add visible discard controls during `key_region`, `await_score`, and `pending_replay`.
- [ ] Change score buttons to draft reward followed by confirm, or implement a minimal flow where score enters `pending_replay` and confirm is the actual write event.
- [ ] Display `pending_replay` and recent segment state in the stats panel.
- [ ] Run frontend build if dependencies are usable.

### Task 5: Final Verification

**Files:**
- Existing tests above.

- [ ] Run all affected backend/recorder/replay tests.
- [ ] Run `py_compile` with `PYTHONPYCACHEPREFIX=/tmp/openpi_pycache_check`.
- [ ] Run `docker compose --profile rlt config --services`.
- [ ] Leave unrelated untracked `command.txt` untouched.

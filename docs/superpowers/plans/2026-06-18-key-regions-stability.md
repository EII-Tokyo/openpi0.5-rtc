# Key Regions Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Key Regions review page stable with hundreds of four-camera records by enforcing bounded DOM, bounded video players, and paginated data loading.

**Architecture:** Split Key Regions out of the overloaded `RolloutBrowser` mode path. The backend provides a paginated review response plus a detail endpoint; the frontend renders a fixed-size card window and owns all video elements in a dedicated preview component with explicit media teardown.

**Tech Stack:** FastAPI/Pydantic backend, React 18 + TypeScript + Vite frontend, pytest backend tests, manual Chrome DevTools resource verification.

---

### Task 1: Backend Paginated Key-Region API

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/main.py`
- Modify: `voice_assistant_web/backend/app/rollout_tree_test.py`

- [ ] **Step 1: Write failing backend tests**

Add tests that call `rlt_key_region_review(limit=2, offset=1)` and `rlt_key_region_detail("id")`. Expected failure before implementation: unexpected keyword argument or missing function/route.

- [ ] **Step 2: Implement response schema**

Add `RLTKeyRegionReviewSummary` and `RLTKeyRegionReviewPage` so the API can return `items`, `total`, `next_offset`, and aggregate counts.

- [ ] **Step 3: Implement endpoint behavior**

Keep `/api/rlt/key-regions/review` backward-compatible at the record field level but change the response to a page object. Support `limit`, `offset`, `status`, and `reward`; add `/api/rlt/key-region/{key_region_id}` for detail.

- [ ] **Step 4: Run backend tests**

Run: `.venv/bin/python -m pytest voice_assistant_web/backend/app/rollout_tree_test.py -q`

### Task 2: Frontend API Types

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`

- [ ] **Step 1: Add page response types**

Add `RLTKeyRegionReviewSummary` and `RLTKeyRegionReviewPage`.

- [ ] **Step 2: Replace full-list fetch helper**

Change `fetchRLTKeyRegionReview()` to accept `{limit, offset, status, reward}` and return a page. Add `fetchRLTKeyRegionDetail(id)`.

### Task 3: Key Regions Page Split

**Files:**
- Create: `voice_assistant_web/frontend/src/components/KeyRegionsPage.tsx`
- Modify: `voice_assistant_web/frontend/src/App.tsx`
- Modify: `voice_assistant_web/frontend/src/components/RolloutBrowser.tsx`

- [ ] **Step 1: Move Key Regions mode into a dedicated page**

Create `KeyRegionsPage` with toolbar, summary strip, fixed card window, detail state, crop/rescore/delete actions, and no rollout tree fetch.

- [ ] **Step 2: Update App**

Render `KeyRegionsPage` for the Key Regions tab instead of `RolloutBrowser enableKeyRegionActions`.

- [ ] **Step 3: Remove Key Regions mode from RolloutBrowser imports/props**

Leave ordinary rollout browsing intact.

### Task 4: Bounded Rendering and Video Lifecycle

**Files:**
- Modify: `voice_assistant_web/frontend/src/components/KeyRegionsPage.tsx`
- Modify: `voice_assistant_web/frontend/src/styles.css`

- [ ] **Step 1: Fixed card window**

Render only one page/window of cards at a time. Use Next/Previous page controls instead of Load More so mounted card count stays constant.

- [ ] **Step 2: Dedicated active preview**

Mount videos only for the selected card. On unmount or selected-card change, pause each video, clear `src`, and call `load()`.

- [ ] **Step 3: Throttle playback state**

Update playhead state at about 10 FPS instead of every animation frame.

### Task 5: Verification

**Files:**
- No production file changes expected.

- [ ] **Step 1: Run backend tests**

Run: `.venv/bin/python -m pytest voice_assistant_web/backend/app/rollout_tree_test.py -q`

- [ ] **Step 2: Run frontend build**

Run: `npm run build` in `voice_assistant_web/frontend`.

- [ ] **Step 3: Browser resource check**

Open `http://127.0.0.1:5173/`, click Key Regions, page through several pages, select cards, and verify:
- card DOM count stays near page size
- `<video>` count stays at 4 or less
- no WebMediaPlayer limit errors
- JS heap does not grow linearly with total dataset size

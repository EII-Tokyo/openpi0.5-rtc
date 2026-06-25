# Key Region Frame Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace key region cropping playback with frame-based previews so local annotation no longer depends on multiple browser video decoders.

**Architecture:** The backend exposes key-region media metadata and cached JPEG frame endpoints. The frontend renders static frame previews for the active key region, stores crop handles as frame indices, and converts frames back to seconds when calling the existing crop endpoint.

**Tech Stack:** FastAPI, ffmpeg/ffprobe, React, TypeScript, Vite, pytest.

---

### Task 1: Backend Frame Media API

**Files:**
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/rlt_key_region_media_test.py`

- [ ] Add tests for `GET /api/rlt/key-region/{id}/media-metadata` returning camera names, fps, frame count, duration, and frame URLs.
- [ ] Add tests for `GET /api/rlt/key-region/{id}/frame?camera=cam_low&frame=0` returning a JPEG response.
- [ ] Implement metadata lookup from the existing review-detail record and mp4 files.
- [ ] Implement JPEG frame extraction with a deterministic cache path under `/tmp/eii_key_region_frame_cache`.
- [ ] Reject unknown cameras and out-of-range frame indexes with `404` or `400`.

### Task 2: Frontend API Types

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`

- [ ] Add `RLTKeyRegionMediaMetadata` and `RLTKeyRegionCameraMedia` types.
- [ ] Add `fetchRLTKeyRegionMediaMetadata(keyRegionId)`.
- [ ] Add `keyRegionFrameUrl(keyRegionId, camera, frame)`.

### Task 3: KeyRegionsPage Frame Preview

**Files:**
- Modify: `voice_assistant_web/frontend/src/components/KeyRegionsPage.tsx`

- [ ] Remove active-card `<video>` creation from key-region review cards.
- [ ] Add active media metadata state keyed by key-region id.
- [ ] Track crop frame ranges, deriving defaults from existing crop seconds and fps.
- [ ] Render active camera previews as `<img>` elements using frame URLs.
- [ ] Update crop handles and playback controls to seek frames instead of setting video currentTime.
- [ ] Convert frame ranges to seconds when saving with the existing crop API.

### Task 4: Styling and Verification

**Files:**
- Modify: `voice_assistant_web/frontend/src/styles.css`

- [ ] Style frame preview tiles so loading/missing states do not shift layout.
- [ ] Run focused backend tests.
- [ ] Run frontend TypeScript build.
- [ ] Verify `http://127.0.0.1:3011/` shows key region frames and no longer emits repeated mp4 Range requests while dragging crop handles.

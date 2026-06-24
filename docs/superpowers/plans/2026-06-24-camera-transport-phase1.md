# Camera Transport Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move live camera pixels out of `/ws/realtime` by default and render four live cameras through a transport abstraction that supports MJPEG now and WebRTC later.

**Architecture:** Backend exposes camera capabilities and controls whether realtime payloads include legacy `camera_jpeg_b64`. Frontend `CameraGrid` chooses `mjpeg` or legacy `jpeg_ws` transport without changing RLT controls. RLT websocket uses a fast state snapshot that avoids scanning replay shards on every tick.

**Tech Stack:** FastAPI, Pydantic, React/TypeScript, Vite, pytest, existing MJPEG endpoint.

---

### Task 1: Backend Camera Capabilities

**Files:**
- Modify: `voice_assistant_web/backend/app/config.py`
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/camera_capabilities_test.py`

- [ ] Add settings for camera transport:
  - `EII_CAMERA_TRANSPORT`, default `mjpeg`
  - `EII_REALTIME_INCLUDE_CAMERA_FRAMES`, default `false`
  - `EII_CAMERA_WEBRTC_ENABLED`, default `false`

- [ ] Add response schema with `preferred_transport`, `transports`, `cameras`, `include_realtime_frames`, and `webrtc`.

- [ ] Add `GET /api/cameras/capabilities`.

- [ ] Test that default capabilities prefer `mjpeg`, include `jpeg_ws`, and report WebRTC disabled.

### Task 2: Realtime Payload Without Camera Frames

**Files:**
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/camera_capabilities_test.py`

- [ ] Extract a helper that builds camera frame payload only when `settings.realtime_include_camera_frames` is true.

- [ ] Use the helper inside `/ws/realtime`.

- [ ] Test that default payload helper returns `{}` and enabled helper returns camera frame data from a fake bridge.

### Task 3: Fast RLT Snapshot

**Files:**
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Test: `voice_assistant_web/backend/app/rlt_control_test.py`

- [ ] Add `snapshot_fast()` that updates runtime metrics, applies score timeout, refreshes derived fields, and returns a copy without `_apply_ledger_stats_locked()`.

- [ ] Use `snapshot_fast()` in `/ws/realtime`.

- [ ] Keep `/api/rlt/status` using full `snapshot()` for detailed status.

- [ ] Test that `snapshot_fast()` does not call ledger stats scanning while `snapshot()` still does.

### Task 4: Frontend Camera Transport

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/App.tsx`
- Modify: `voice_assistant_web/frontend/src/components/CameraGrid.tsx`
- Modify: `voice_assistant_web/frontend/src/styles.css`

- [ ] Add camera capabilities types and `fetchCameraCapabilities()`.

- [ ] Load capabilities once in `App` and pass selected transport to `CameraGrid`.

- [ ] In `CameraGrid`, render MJPEG `<img>` tiles for `mjpeg` and keep existing canvas decoding for `jpeg_ws`.

- [ ] Keep camera overlays and focus/quad layout unchanged.

### Task 5: Verification And Deployment

**Files:**
- No production code changes unless verification finds a bug.

- [ ] Run backend tests for camera capabilities and RLT control.

- [ ] Run frontend `npm run build`.

- [ ] Commit Phase 1.

- [ ] Push to GitHub.

- [ ] Pull on `192.168.1.103` only in `/home/eii/openpi0.5-rtc-reward-learning`.

- [ ] Rebuild/recreate `eii_pilot_backend` and `eii_pilot_frontend`.

- [ ] Verify `/api/cameras/capabilities`, `/ws/realtime` payload size, frontend page, and container health.

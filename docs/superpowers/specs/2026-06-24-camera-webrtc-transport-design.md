# Long-Term Camera WebRTC Transport Design

## Goal

Provide a long-term, low-latency four-camera display path for the robot UI without letting video transport block RLT controls, robot task commands, replay annotation, or policy inference.

## Design Decision

Use a staged transport architecture:

1. Keep `/ws/realtime` for robot/RLT/control metadata.
2. Move live camera pixels out of `/ws/realtime`.
3. Add a camera transport abstraction with `webrtc`, `mjpeg`, and legacy `jpeg_ws`.
4. Use MJPEG as the first production-safe transport because the backend already exposes `/api/cameras/{camera_name}/stream.mjpg`.
5. Add an independent `eii_pilot_webrtc` media service later for GStreamer/WebRTC. Do not put GStreamer pipelines inside the existing FastAPI control backend.

## Non-Goals

- Do not modify robot inference image inputs.
- Do not modify `aloha_ros_nodes` or the RealSense publisher.
- Do not make WebRTC the default until MJPEG fallback and transport health are proven.
- Do not remove legacy JPEG websocket support in the first phase.

## Architecture

### Phase 1: Transport Split

- Backend exposes `GET /api/cameras/capabilities`.
- Frontend selects a camera transport from server capabilities and environment defaults.
- `CameraGrid` renders MJPEG with `<img src="/api/cameras/{name}/stream.mjpg">`.
- `/ws/realtime` no longer includes `camera_jpeg_b64` by default.
- Legacy `jpeg_ws` remains available through a backend flag.
- RLT realtime uses a fast snapshot that does not scan all replay `.npz` files.

### Phase 2: WebRTC Signaling Skeleton

- Add session/capability APIs for WebRTC while keeping `webrtc.enabled=false` by default.
- Frontend can select WebRTC, but falls back to MJPEG if session creation or playback fails.
- Add session cleanup and connection-state reporting tests.

### Phase 3: GStreamer WebRTC Media Service

- Add a separate `eii_pilot_webrtc` container.
- It subscribes to `/cam_high`, `/cam_low`, `/cam_left_wrist`, `/cam_right_wrist`.
- It consumes `aloha.msg.RGBGrayscaleImage`, extracts `images[0]`, converts only private display copies, and never changes the policy image path.
- Start with `videotestsrc`, then one real camera, then four real cameras.

### Phase 4: Encoder Hardening

- Add x264 software encoding first.
- Add NVENC only after checking `gst-inspect-1.0` and container GPU video capability on `192.168.1.103`.
- Keep CPU fallback.

### Phase 5: Default WebRTC

- `EII_CAMERA_TRANSPORT=auto` prefers WebRTC.
- MJPEG remains the first fallback.
- Legacy `jpeg_ws` remains a diagnostic fallback.

## Runtime Safety

The WebRTC service is a display-only sidecar. It must not:

- Publish robot commands.
- Alter ROS camera messages.
- Reinitialize or replace the existing ROS node used by runtime inference.
- Block `/ws/realtime`, `/api/robot/task`, or RLT key region endpoints.

## Transport Fallback

Fallback order:

1. WebRTC, when enabled and healthy.
2. MJPEG stream.
3. Legacy JPEG websocket.
4. Offline tile with last health state.

Fallback is per camera when possible. One failed camera must not hide the other three.

## Testing Requirements

Phase 1:

- Backend tests for camera capabilities and realtime payload without camera frames.
- Frontend build verifies `CameraGrid` supports MJPEG and legacy canvas paths.
- Robot UI still shows camera status and timestamps.
- `/ws/realtime` payload size drops substantially when image frames are disabled.
- RLT key region end/score/confirm remain functional.

Phase 2:

- WebRTC disabled path is identical to Phase 1.
- Session creation rejects unknown cameras and honors max sessions.
- Disconnect cleanup is tested.

Phase 3:

- `videotestsrc` renders in browser.
- One real camera renders with correct color.
- Four real cameras render without blocking RLT controls.

Phase 4:

- x264 and NVENC are separately smoke-tested.
- GPU/video encoder use does not degrade policy inference timing.

Phase 5:

- WebRTC default mode has MJPEG fallback.
- Long-run test shows no session, pipeline, memory, or CPU leak.

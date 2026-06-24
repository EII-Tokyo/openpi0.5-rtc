# Camera Smoothness Diagnostics And WebRTC Report

## Summary

The current live camera UI is smoother than the old base64 WebSocket path, but it is still not expected to feel like real video. The robot publishes each camera topic at about 60 Hz, while the current MJPEG UI path emits about 9.3-9.5 fps because `/api/cameras/{camera}/stream.mjpg` sleeps for 0.1 seconds per loop.

This means the camera source is not the primary bottleneck. The current bottlenecks are the display transport, JPEG encoding work, and the lack of video-style buffering/synchronization.

## Measurements On 192.168.1.103

Measured on 2026-06-24 with the robot containers running under `/home/eii/openpi0.5-rtc-reward-learning`.

### MJPEG Output FPS

Sampling `/api/cameras/{name}/stream.mjpg` for about 3 seconds:

| Camera | MJPEG fps | Notes |
|---|---:|---|
| `cam_high` | 9.45 | Stable around the current 10 fps cap |
| `cam_low` | 9.34 | Stable around the current 10 fps cap |
| `cam_left_wrist` | 9.30 | Stable around the current 10 fps cap |
| `cam_right_wrist` | 9.37 | Earlier low-fps observation did not reproduce in this run |

### ROS Topic Source FPS

`rostopic hz` inside `openpi_reward_learning_eii-aloha_ros_nodes-1`:

| Topic | Type | Observed rate |
|---|---|---:|
| `/cam_high` | `aloha/RGBGrayscaleImage` | about 59.5-59.8 Hz |
| `/cam_low` | `aloha/RGBGrayscaleImage` | about 59.8-60.3 Hz |
| `/cam_left_wrist` | `aloha/RGBGrayscaleImage` | about 59.2-60.0 Hz |
| `/cam_right_wrist` | `aloha/RGBGrayscaleImage` | about 59.3-60.1 Hz |

### Container CPU During Sampling

| Container | CPU |
|---|---:|
| `eii_pilot_backend` | about 74.5% |
| `aloha_ros_nodes` | about 82.3% |
| `rlt_warmup_runtime` | about 26.0% |
| `openpi_server` | about 0.0% at that instant |

The backend and ROS camera node are doing substantial image work even though the browser only receives about 10 fps.

### Diagnostics API After Deployment

After deploying this change and restarting `eii_pilot_backend`, `GET /api/cameras/diagnostics` reported:

| Camera | Source fps | Encoded fps | Frame age | JPEG bytes | Mean encode time |
|---|---:|---:|---:|---:|---:|
| `cam_high` | 60.0 | 59.9 | about 39 ms | 41,936 | 1.02 ms |
| `cam_low` | 59.9 | 59.9 | about 37 ms | 40,476 | 0.92 ms |
| `cam_left_wrist` | 59.9 | 59.9 | about 37 ms | 45,893 | 1.01 ms |
| `cam_right_wrist` | 59.9 | 59.9 | about 37 ms | 49,570 | 0.94 ms |

All four cameras had `dropped_frames_total=0` and `error_count=0` in that sample.

This confirms the backend is currently encoding close to the full 60 Hz camera source, while the browser MJPEG endpoint only emits about 10 fps. The most direct smoothness limit is therefore the MJPEG stream loop and browser transport choice, not camera capture failure.

## Root Cause Analysis

### 1. The UI is capped by the MJPEG loop

`voice_assistant_web/backend/app/main.py::stream_camera()` currently yields one frame and then sleeps for 0.1 seconds. That caps each stream around 10 fps.

The browser cannot display a 60 Hz source smoothly if the server only emits about 10 frames per second.

### 2. JPEG work still happens before MJPEG delivery

`voice_assistant_web/backend/app/camera_bridge.py` subscribes to each `aloha/RGBGrayscaleImage` topic and encodes frames with OpenCV JPEG. The previous Phase 1 change removed base64 camera bytes from `/ws/realtime` by default, but it did not remove JPEG encoding cost.

That is why MJPEG improves control responsiveness but does not make video feel like H264/WebRTC.

### 3. MJPEG is not a real-time video transport

MJPEG is a sequence of independent JPEG images. It has no video codec, no inter-frame compression, no jitter buffer, no media clock, and no per-track connection state like WebRTC. Four camera tiles are four independent image streams, so their apparent timing can differ.

### 4. WebSocket is now lighter, but video is still not optimized

Phase 1 successfully moved camera pixels out of `/ws/realtime` by default. This is correct for control responsiveness. It does not solve:

- 10 fps MJPEG output
- JPEG encoding CPU
- four independent streams
- lack of video synchronization
- lack of hardware video decode/encode path

## Code Added In This Step

### Camera diagnostics API

Added:

```text
GET /api/cameras/diagnostics
```

It reports, per camera:

- whether a latest frame exists
- latest frame age
- recent ROS source fps
- recent encoded fps
- total raw frames seen
- total encoded frames
- dropped/error counts
- last encoding, width, height
- latest JPEG bytes
- recent mean/max encode time in milliseconds

This is intentionally read-only and does not affect robot runtime commands or policy inference.

### Files touched

- `voice_assistant_web/backend/app/camera_bridge.py`
- `voice_assistant_web/backend/app/main.py`
- `voice_assistant_web/backend/app/schemas.py`
- `voice_assistant_web/backend/app/camera_capabilities_test.py`

## How To Use The Diagnostics

After deploying/restarting the backend:

```bash
curl -s http://192.168.1.103:8011/api/cameras/diagnostics | jq
```

Important fields:

- `source_fps_recent`: whether the ROS source is publishing frames.
- `encoded_fps_recent`: whether the backend is keeping up with JPEG encoding.
- `encode_ms_mean_recent`: CPU cost per encoded frame.
- `frame_age_seconds`: whether the displayed frame is stale.
- `dropped_frames_total` and `last_error`: decoding/encoding failures.

Interpretation:

- Source fps high, encoded fps low: backend image conversion/encoding is bottlenecked or intentionally throttled.
- Source fps high, MJPEG fps low: stream endpoint or browser transport is the bottleneck.
- Source fps low: camera publisher, USB, RealSense, ROS node, or upstream hardware path needs investigation.
- Frame age high: latest frame cache is stale or callback stopped.

## Long-Term WebRTC/GStreamer Plan

Official references support this direction:

- GStreamer `webrtcbin` supports WebRTC offer/answer and SDP flow: https://gstreamer.freedesktop.org/documentation/webrtc/
- GStreamer `webrtcsink` is intended to serve media streams to multiple WebRTC consumers: https://gstreamer.freedesktop.org/documentation/rswebrtc/webrtcsink.html
- Browser WebRTC is driven by `RTCPeerConnection`: https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection

### Phase A: Observability

Status: started.

- Keep MJPEG as current default.
- Add `/api/cameras/diagnostics`.
- Measure source fps, encoded fps, encode time, frame age, and CPU.
- Do not change robot control, policy inference, or ROS publisher.

Exit criteria:

- We can explain any observed camera stutter by checking one endpoint and container stats.

### Phase B: MJPEG Tuning

Goal: make the current fallback path less wasteful.

Recommended changes:

- Add `fps`, `quality`, and optional `scale` query parameters to `/stream.mjpg`.
- Only yield when a camera timestamp changes, instead of blindly re-sending the same JPEG every loop.
- Use lower fps for quad view and higher fps for focus view.
- Consider backend JPEG encode throttling if `encode_ms` or CPU is high.

Boundary condition:

- Do not reduce the camera frames used by policy inference.
- Any throttling must apply only to UI display encoding.

### Phase C: WebRTC Signaling Skeleton

Goal: introduce the WebRTC API surface without turning it on by default.

Recommended APIs:

```text
GET  /api/cameras/capabilities
POST /api/cameras/webrtc/sessions
WS   /ws/cameras/webrtc/{session_id}
DELETE /api/cameras/webrtc/sessions/{session_id}
```

Frontend:

- Add a `CameraStreamProvider`.
- Use WebRTC if enabled and healthy.
- Fall back per camera to MJPEG.
- Keep legacy `jpeg_ws` as a diagnostic fallback.

Exit criteria:

- WebRTC disabled path is identical to current MJPEG behavior.
- Failed WebRTC session creation does not affect RLT controls or robot task buttons.

### Phase D: Independent GStreamer Media Service

Goal: isolate video transport from the FastAPI control backend.

Add a separate service, for example:

```text
eii_pilot_webrtc
```

Responsibilities:

- Subscribe to `/cam_high`, `/cam_low`, `/cam_left_wrist`, `/cam_right_wrist`.
- Consume `aloha/RGBGrayscaleImage`.
- Extract `images[0]`.
- Convert private display copies to encoder format.
- Publish WebRTC video tracks.

Hard rule:

- Do not modify the policy image path.
- Do not modify `aloha_ros_nodes`.
- Do not read RealSense devices directly in the media service at first, to avoid fighting the existing ROS publisher.

### Phase E: GStreamer Smoke Tests

Sequence:

1. `videotestsrc` to browser.
2. One synthetic camera track to browser.
3. One real ROS camera to browser.
4. Four real ROS cameras to browser.

Each step must check:

- video nonblank
- color not swapped
- latency
- browser reconnection
- server cleanup
- CPU/memory stability
- no effect on RLT control latency

### Phase F: Encoder Hardening

Start with software H264/x264 or VP8 for correctness. Add NVENC only after:

- `gst-inspect-1.0 webrtcbin` succeeds.
- chosen encoder plugin exists.
- container has `NVIDIA_DRIVER_CAPABILITIES=video,utility`.
- policy inference timing is not degraded.

The media service should report its active encoder and fall back to CPU encoding if hardware encoding is unavailable.

## Why Not Put GStreamer Inside The Existing Backend?

The current FastAPI backend owns robot/RLT control APIs, key region editing APIs, MJPEG fallback, and WebSocket status. GStreamer/WebRTC pipelines have very different lifecycle and failure modes:

- codec negotiation
- UDP/ICE state
- per-browser sessions
- pipeline cleanup
- encoder hardware failures
- long-running media buffers

Putting those into the control backend risks making robot control less reliable. A sidecar service gives a clearer boundary: if video fails, robot control can still work.

## Immediate Next Recommendations

1. Keep `/api/cameras/diagnostics` visible during robot tests and check source fps, encoded fps, frame age, encode time, and drops.
2. If `encode_ms_mean_recent` is high or backend CPU exceeds one core under browser load, tune JPEG quality/fps and proceed to WebRTC sidecar.
3. Begin Phase C/D with `videotestsrc`, not real robot cameras.

## Phase B Deployment Result

After Phase B, `/api/cameras/{name}/stream.mjpg` accepts `?fps=` and clamps it with:

- `EII_CAMERA_MJPEG_DEFAULT_FPS`, default `20`
- `EII_CAMERA_MJPEG_MAX_FPS`, default `30`

The frontend now requests:

- Quad view: `15fps` per camera.
- Focus main camera: `30fps`.
- Focus secondary cameras: `10fps`.

Remote measurements on 103 after deployment:

| Request | Observed fps |
|---|---:|
| `cam_high?fps=15` | 14.10 |
| `cam_high?fps=30` | 28.16 |
| `cam_low?fps=15` | 14.26 |
| `cam_left_wrist?fps=15` | 14.29 |
| `cam_right_wrist?fps=15` | 14.24 |

The diagnostics API still showed all four cameras around 59.8-59.9 source/encoded fps with zero dropped frames. This means the MJPEG fallback now supports visibly smoother focus mode while keeping quad mode below the maximum load of four 30fps streams.

## Phase C Signaling Skeleton Result

Phase C adds the WebRTC control-plane skeleton while keeping WebRTC disabled by default.

New settings:

- `EII_CAMERA_WEBRTC_ENABLED`, default `false`
- `EII_CAMERA_WEBRTC_SESSION_TTL_SECONDS`, default `30`
- `EII_CAMERA_WEBRTC_MAX_SESSIONS`, default `4`

New endpoints:

```text
POST   /api/cameras/webrtc/sessions
DELETE /api/cameras/webrtc/sessions/{session_id}
WS     /ws/cameras/webrtc/{session_id}
```

Current behavior:

- When WebRTC is disabled, session creation returns `503` with a clear message.
- When enabled, session creation validates requested cameras and creates an in-memory signaling session.
- The signaling WebSocket sends an initial `state` message.
- Since the media service is not attached yet, offer/ICE messages receive `media_service_not_available` and tell the frontend to use MJPEG fallback.
- Session delete closes the in-memory session.

This is intentionally not a video implementation yet. It prepares the browser/backend contract for the next phase without touching robot control, ROS image publishing, or policy inference.

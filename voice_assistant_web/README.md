# EII Pilot Web

Web UI for RLT stage-2 data collection and control.

## Services

- Backend: FastAPI on `http://localhost:8011`
- Frontend: Vite/nginx UI on `http://localhost:3011`

## Current Scope

- Four live camera panels from ROS image topics.
- Realtime robot state websocket.
- RLT critical-region controls: `S`, `E`, `1`, `0`.
- Configurable `warmup_target`, `beta`, `intervention_scale`, and `max_delta`.
- Rollout video browser.

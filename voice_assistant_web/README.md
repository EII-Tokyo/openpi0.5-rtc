# Voice Assistant Web

New web UI for Aloha voice control, camera monitoring, and robot 3D visualization.

## Services

- Backend: FastAPI on `http://localhost:8011`
- Frontend: Vite dev server on `http://localhost:3011`

## Run

```bash
docker compose -f voice_assistant_web/compose.yml up --build
```

## Current scope

- 4 live camera panels from ROS image topics
- Realtime robot state websocket
- 3D Aloha viewer driven by latest action/qpos
- Voice/text command panel that publishes tasks through Redis

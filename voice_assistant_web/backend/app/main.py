from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .camera_bridge import CameraBridge
from .config import settings
from .redis_commands import TASK_MAPPING
from .redis_commands import create_redis_client
from .redis_commands import publish_task
from .robot_state_bridge import RobotStateBridge
from .schemas import HealthResponse, RealtimePayload, RuntimeStatePayload, VoiceRequest, VoiceResponse
from .voice_session import VoiceAssistantEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Aloha Voice Assistant Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins if settings.allow_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_bridge = CameraBridge()
robot_state_bridge = RobotStateBridge()
redis_client = create_redis_client()
voice_engine = VoiceAssistantEngine(redis_client)


@app.on_event("startup")
def on_startup() -> None:
    try:
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("voice_assistant_web_backend", anonymous=True)
    except Exception:
        logging.exception("ROS node initialization failed")
    camera_bridge.start()
    robot_state_bridge.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    camera_bridge.stop()
    robot_state_bridge.stop()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/cameras/{camera_name}/latest.jpg")
def latest_camera_image(camera_name: str) -> Response:
    jpeg = camera_bridge.get_latest_jpeg(camera_name)
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"No frame available for {camera_name}")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/cameras/{camera_name}/stream.mjpg")
def stream_camera(camera_name: str) -> StreamingResponse:
    if camera_name not in camera_bridge.camera_names:
        raise HTTPException(status_code=404, detail=f"Unknown camera {camera_name}")

    async def frame_generator():
        while True:
            jpeg = camera_bridge.get_latest_jpeg(camera_name)
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.1)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/realtime")
async def realtime_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    interval = 1.0 / settings.realtime_hz if settings.realtime_hz > 0 else 0.1
    last_camera_push = 0.0
    try:
        while True:
            now = time.time()
            camera_jpeg_b64: dict[str, str] = {}
            if last_camera_push == 0.0 or now - last_camera_push >= 0.1:
                camera_jpeg_b64 = camera_bridge.snapshot_jpeg_b64_all()
                last_camera_push = now
            payload = RealtimePayload(
                robot=RuntimeStatePayload(**robot_state_bridge.snapshot()),
                camera_status=camera_bridge.get_camera_status(),
                camera_timestamps=camera_bridge.get_camera_timestamps(),
                camera_jpeg_b64=camera_jpeg_b64,
            )
            await websocket.send_json(payload.model_dump())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


@app.post("/api/voice/text", response_model=VoiceResponse)
async def voice_text(request: VoiceRequest) -> VoiceResponse:
    return await voice_engine.process_text(request.text, language=request.language)


@app.post("/api/tasks/{task_number}")
def dispatch_task(task_number: str) -> dict[str, str]:
    if task_number not in TASK_MAPPING:
        raise HTTPException(status_code=404, detail=f"Unknown task {task_number}")
    publish_task(redis_client, task_number)
    return {
        "status": "ok",
        "task_number": task_number,
        "task_name": TASK_MAPPING[task_number],
    }


@app.post("/api/voice/audio", response_model=VoiceResponse)
async def voice_audio(file: UploadFile = File(...), language: str = Form("en")) -> VoiceResponse:
    return await voice_engine.process_audio(file, language=language)

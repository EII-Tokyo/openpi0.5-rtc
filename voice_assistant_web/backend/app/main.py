from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .camera_bridge import CameraBridge
from .config import settings
from .redis_commands import TASK_MAPPING
from .redis_commands import create_redis_client
from .redis_commands import publish_task
from .redis_commands import INFERENCE_SETTINGS_KEY, read_inference_settings
from .robot_state_bridge import RobotStateBridge
from .schemas import HealthResponse, RealtimePayload, RuntimeStatePayload, VoiceRequest, VoiceResponse
from .schemas import InferenceSettings, OPERATORS
from .schemas import RecordingCommand
from .voice_session import VoiceAssistantEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from .recording_folders import router as recording_folders_router

app = FastAPI(title="Aloha Voice Assistant Web")
app.include_router(recording_folders_router)
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


@app.get("/api/inference-settings")
def inference_settings() -> dict:
    return {**read_inference_settings(redis_client).model_dump(), "operators": OPERATORS}


@app.put("/api/inference-settings")
def update_inference_settings(request: InferenceSettings) -> dict:
    # Store and notify atomically; this never dispatches a start/home command.
    transaction = redis_client.pipeline()
    transaction.set(INFERENCE_SETTINGS_KEY, request.model_dump_json())
    transaction.publish("aloha_inference_settings_updates", json.dumps({"type": "inference_settings", **request.model_dump()}))
    transaction.execute()
    return {**request.model_dump(), "operators": OPERATORS}


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


@app.get("/api/recording")
def recording_status() -> dict:
    """Report runtime acknowledgements and disk completion, including stale/offline state."""
    raw = redis_client.get("aloha_recording_state")
    state = json.loads(raw) if raw else {"status": "idle", "frames": 0, "paused": False}
    state["online"] = time.time() - state.get("timestamp", 0) < 3
    return state


@app.post("/api/recording")
def control_recording(request: RecordingCommand) -> dict:
    """Publish intent only; the UI waits for the control-thread acknowledgement."""
    if request.action == "start" and not request.batch_name:
        raise HTTPException(422, "Set a batch name before recording")
    state = recording_status()
    if not state["online"]:
        raise HTTPException(503, "Robot runtime is offline; no recording command sent")
    if request.action == "start" and (state.get("mode") != "policy" or state.get("status") in {"recording", "saving"}):
        raise HTTPException(409, "Recording requires an active policy task and no pending save")
    if request.action == "resume" and (not state.get("paused") or state.get("status") != "saved"):
        raise HTTPException(409, "Resume requires a paused robot and a successfully saved recording")
    if request.action == "stop" and state.get("status") not in {"recording", "saving"}:
        raise HTTPException(409, "There is no active recording to finish")
    command_id = str(uuid.uuid4())
    listeners = redis_client.publish("aloha_voice_commands", json.dumps({
        "type": "recording_control", "command_id": command_id, **request.model_dump()
    }))
    if not listeners:
        raise HTTPException(503, "Robot runtime did not receive the recording command")
    return {"status": "queued", "command_id": command_id}


@app.post("/api/voice/audio", response_model=VoiceResponse)
async def voice_audio(file: UploadFile = File(...), language: str = Form("en")) -> VoiceResponse:
    return await voice_engine.process_audio(file, language=language)

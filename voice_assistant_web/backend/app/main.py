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
from .camera_exposure import router as camera_exposure_router
from .nozzle_calibration import NozzleCalibration
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
app.include_router(camera_exposure_router)
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
    try:
        while True:
            payload = RealtimePayload(
                robot=RuntimeStatePayload(**robot_state_bridge.snapshot()),
                camera_status=camera_bridge.get_camera_status(),
                camera_timestamps=camera_bridge.get_camera_timestamps(),
                camera_jpeg_b64={},
            )
            await websocket.send_json(payload.model_dump())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


@app.get('/api/cameras/preview-metrics')
def preview_metrics():
    return camera_bridge.get_preview_metrics()


@app.get('/api/cameras/exposure/actual')
def actual_camera_exposure():
    return camera_bridge.frame_exposure.snapshot()


@app.get('/api/cameras/{camera_name}/latest.png')
def latest_camera_png(camera_name: str):
    image = camera_bridge.get_latest_png(camera_name)
    if image is None:
        raise HTTPException(503, 'Lossless preview is not ready')
    return Response(content=image, media_type='image/png', headers={'Cache-Control': 'no-store'})


@app.websocket('/ws/cameras')
async def lossless_camera_socket(websocket: WebSocket):
    await websocket.accept()
    previous = None
    try:
        while True:
            sequence, packet = camera_bridge.get_preview()
            if packet is None or sequence == previous:
                await asyncio.sleep(.005)
                continue
            await websocket.send_bytes(packet)
            # At most one undrawn packet per viewer. Slow clients skip to latest.
            acknowledgment = await asyncio.wait_for(websocket.receive_json(), timeout=5.)
            if acknowledgment != {'type': 'ack', 'sequence': sequence}:
                await websocket.close(code=1008)
                return
            previous = sequence
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return


@app.post("/api/voice/text", response_model=VoiceResponse)
async def voice_text(request: VoiceRequest) -> VoiceResponse:
    return await voice_engine.process_text(request.text, language=request.language)


@app.get("/api/high-level/status")
def high_level_status() -> dict:
    raw = redis_client.get("aloha:sol_high_level:status")
    if not raw:
        return {"enabled": False, "status": "not_configured"}
    from openpi.shared import sol_run_grant
    state = json.loads(raw)
    calibration = json.loads(redis_client.get(sol_run_grant.CALIBRATION_KEY) or "{}")
    state["nozzle_tips"] = calibration.get("tips") if sol_run_grant.active(redis_client, calibration.get("grant")) else None
    robot = json.loads(redis_client.get("aloha_recording_state") or "{}")
    if time.time() - robot.get("timestamp", 0) <= 3:
        if robot.get("mode") == "waiting_high_level":
            state["status"] = "waiting_result"
        elif robot.get("mode") in ("waiting", "paused") and not state["nozzle_tips"]:
            if state.get("status") == "running":
                state["status"] = "idle"
    return state


@app.get("/api/high-level/prompts")
def high_level_prompts() -> dict:
    from openpi.serving.sol_rinse_protocol import rinse_prompt

    return {language: rinse_prompt(language) for language in ("zh", "en")}


@app.post("/api/high-level/confirm-dwell/{token}")
def confirm_rinse_dwell(token: str) -> dict:
    status = high_level_status()
    if (not token or token != status.get("dwell_token") or not status.get("updated_unix")
            or time.time() - status["updated_unix"] > 8 or status.get("status") != "running"):
        raise HTTPException(409, "There is no current insertion to confirm")
    redis_client.set("aloha:rinse:dwell:" + token, "1", ex=30)
    return {"confirmed": True}


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


@app.get("/api/nozzle-calibration")
def read_nozzle_calibration():
    from .nozzle_calibration import calibration_status

    return calibration_status(redis_client)


@app.put("/api/nozzle-calibration")
def write_nozzle_calibration(request: NozzleCalibration):
    from .nozzle_calibration import save_calibration

    raw = redis_client.get("aloha_recording_state")
    state = json.loads(raw) if raw else {}
    if state.get("mode") not in ("waiting", "paused") or time.time() - state.get("timestamp", 0) > 3:
        raise HTTPException(409, "Calibrate while the robot is waiting")
    return save_calibration(redis_client, camera_bridge, request.model_dump())


@app.post("/api/sol-rinse/start")
def start_sol_rinse() -> dict:
    from openpi.shared import sol_run_grant

    raw = redis_client.get("aloha_recording_state")
    state = json.loads(raw) if raw else {}
    if state.get("mode") not in ("waiting", "paused") or time.time() - state.get("timestamp", 0) > 3:
        raise HTTPException(409, "Robot must be waiting before starting Sol")
    if not redis_client.set(sol_run_grant.KEY, "starting", ex=2, nx=True):
        raise HTTPException(409, "Sol is already starting or running")
    message = publish_task(redis_client, "2", authorize_sol=True)
    return {"status": "ok", "command_id": message["command_id"]}


@app.post("/api/tasks/{task_number}")
def dispatch_task(task_number: str) -> dict[str, str]:
    if task_number not in TASK_MAPPING:
        raise HTTPException(status_code=404, detail=f"Unknown task {task_number}")
    if task_number in {"1", "2"} and high_level_status().get("enabled"):
        raise HTTPException(409, "Use the Start rinse button to enable paid Sol requests")
    if task_number in {"3", "4", "5", "6"}:
        from openpi.shared import sol_run_grant
        sol_run_grant.revoke(redis_client)
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
    if request.action == "stop":
        from openpi.shared import sol_run_grant
        sol_run_grant.revoke(redis_client)
    listeners = redis_client.publish("aloha_voice_commands", json.dumps({
        "type": "recording_control", "command_id": command_id, **request.model_dump()
    }))
    if not listeners:
        raise HTTPException(503, "Robot runtime did not receive the recording command")
    return {"status": "queued", "command_id": command_id}


@app.post("/api/voice/audio", response_model=VoiceResponse)
async def voice_audio(file: UploadFile = File(...), language: str = Form("en")) -> VoiceResponse:
    return await voice_engine.process_audio(file, language=language)

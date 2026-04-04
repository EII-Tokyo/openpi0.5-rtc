from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .camera_bridge import CameraBridge
from .config import settings
from .config_store import RuntimeConfigStore
from .redis_commands import create_redis_client, publish_runtime_config, publish_task
from .robot_state_bridge import RobotStateBridge
from .schemas import (
    AnnouncementAudioRequest,
    AnnouncementAudioResponse,
    HealthResponse,
    RealtimePayload,
    RuntimeConfigPayload,
    RuntimeConfigRequest,
    RuntimeStatePayload,
    TranslateRequest,
    TranslateResponse,
    VoiceRequest,
    VoiceResponse,
)
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
runtime_config_store = RuntimeConfigStore()


def _merge_runtime_config_request(request: RuntimeConfigRequest) -> RuntimeConfigPayload:
    current = runtime_config_store.load()
    return RuntimeConfigPayload(
        dataset_dir=request.dataset_dir if request.dataset_dir is not None else current.dataset_dir,
        manual_dataset_dir=request.manual_dataset_dir if request.manual_dataset_dir is not None else current.manual_dataset_dir,
        include_bottle_description=(
            request.include_bottle_description
            if request.include_bottle_description is not None
            else current.include_bottle_description
        ),
        include_bottle_position=(
            request.include_bottle_position
            if request.include_bottle_position is not None
            else current.include_bottle_position
        ),
        include_bottle_state=request.include_bottle_state if request.include_bottle_state is not None else current.include_bottle_state,
        include_subtask=request.include_subtask if request.include_subtask is not None else current.include_subtask,
        forced_low_level_subtask=(
            request.forced_low_level_subtask
            if request.forced_low_level_subtask is not None
            else current.forced_low_level_subtask
        ),
        video_memory_num_frames=(
            request.video_memory_num_frames
            if request.video_memory_num_frames in (1, 4)
            else current.video_memory_num_frames
        ),
    )


@app.on_event("startup")
def on_startup() -> None:
    camera_bridge.start()
    robot_state_bridge.start()
    stored = runtime_config_store.load()
    publish_runtime_config(
        redis_client,
        dataset_dir=stored.dataset_dir,
        manual_dataset_dir=stored.manual_dataset_dir,
        include_bottle_description=stored.include_bottle_description,
        include_bottle_position=stored.include_bottle_position,
        include_bottle_state=stored.include_bottle_state,
        include_subtask=stored.include_subtask,
        forced_low_level_subtask=stored.forced_low_level_subtask,
        video_memory_num_frames=stored.video_memory_num_frames,
    )


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
            await asyncio.sleep(0.15)

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
            )
            await websocket.send_json(payload.model_dump())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


@app.post("/api/voice/text", response_model=VoiceResponse)
async def voice_text(request: VoiceRequest) -> VoiceResponse:
    direct_task = request.text.strip()
    if direct_task in {"1", "2", "3", "4"}:
        message = publish_task(
            redis_client,
            direct_task,
            dataset_dir=request.dataset_dir,
            manual_dataset_dir=request.manual_dataset_dir,
            include_bottle_description=request.include_bottle_description,
            include_bottle_position=request.include_bottle_position,
            include_bottle_state=request.include_bottle_state,
            include_subtask=request.include_subtask,
            forced_low_level_subtask=request.forced_low_level_subtask,
            video_memory_num_frames=request.video_memory_num_frames,
        )
        return VoiceResponse(
            transcript=request.text,
            reply_text=f"Sent task {direct_task}.",
            task_number=direct_task,
            task_name=message["task_name"],
            debug={"direct_task": True},
        )
    return await voice_engine.process_text(
        request.text,
        language=request.language,
        dataset_dir=request.dataset_dir,
        manual_dataset_dir=request.manual_dataset_dir,
        include_bottle_description=request.include_bottle_description,
        include_bottle_position=request.include_bottle_position,
        include_bottle_state=request.include_bottle_state,
        include_subtask=request.include_subtask,
        forced_low_level_subtask=request.forced_low_level_subtask,
        video_memory_num_frames=request.video_memory_num_frames,
    )


@app.post("/api/voice/audio", response_model=VoiceResponse)
async def voice_audio(
    file: UploadFile = File(...),
    language: str = Form("en"),
    dataset_dir: str | None = Form(None),
    manual_dataset_dir: str | None = Form(None),
    include_bottle_description: bool = Form(True),
    include_bottle_position: bool = Form(False),
    include_bottle_state: bool = Form(True),
    include_subtask: bool = Form(True),
    forced_low_level_subtask: str | None = Form(None),
    video_memory_num_frames: int = Form(1),
) -> VoiceResponse:
    return await voice_engine.process_audio(
        file,
        language=language,
        dataset_dir=dataset_dir,
        manual_dataset_dir=manual_dataset_dir,
        include_bottle_description=include_bottle_description,
        include_bottle_position=include_bottle_position,
        include_bottle_state=include_bottle_state,
        include_subtask=include_subtask,
        forced_low_level_subtask=forced_low_level_subtask,
        video_memory_num_frames=video_memory_num_frames,
    )


@app.get("/api/runtime/config", response_model=RuntimeConfigPayload)
def get_runtime_config() -> RuntimeConfigPayload:
    return runtime_config_store.load()


@app.post("/api/runtime/config", response_model=RuntimeConfigPayload)
def runtime_config(request: RuntimeConfigRequest) -> RuntimeConfigPayload:
    merged = _merge_runtime_config_request(request)
    runtime_config_store.save(merged)
    publish_runtime_config(
        redis_client,
        dataset_dir=merged.dataset_dir,
        manual_dataset_dir=merged.manual_dataset_dir,
        include_bottle_description=merged.include_bottle_description,
        include_bottle_position=merged.include_bottle_position,
        include_bottle_state=merged.include_bottle_state,
        include_subtask=merged.include_subtask,
        forced_low_level_subtask=merged.forced_low_level_subtask,
        video_memory_num_frames=merged.video_memory_num_frames,
    )
    return merged


@app.post("/api/runtime/translate", response_model=TranslateResponse)
def runtime_translate(request: TranslateRequest) -> TranslateResponse:
    translated_text = voice_engine.translate_text(
        request.text,
        target_language=request.target_language,
    )
    return TranslateResponse(translated_text=translated_text)


@app.post("/api/runtime/announcement-audio", response_model=AnnouncementAudioResponse)
def runtime_announcement_audio(request: AnnouncementAudioRequest) -> AnnouncementAudioResponse:
    translated_text, audio_base64, audio_mime_type = voice_engine.synthesize_announcement(
        request.text,
        target_language=request.target_language,
    )
    return AnnouncementAudioResponse(
        translated_text=translated_text,
        audio_base64=audio_base64,
        audio_mime_type=audio_mime_type,
    )

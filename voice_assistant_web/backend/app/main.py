from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from .camera_bridge import CameraBridge
from .config import settings
from .redis_commands import TASK_MAPPING
from .redis_commands import create_redis_client
from .redis_commands import publish_task
from .robot_state_bridge import RobotStateBridge
from .schemas import (
    HealthResponse,
    RealtimePayload,
    RLTLabelRequest,
    RLTReplayStatus,
    RLTTrajectoryListResponse,
    RLTTrajectoryRecord,
    RLTTrajectoryTrimRequest,
    RuntimeStatePayload,
    VoiceRequest,
    VoiceResponse,
)
from .voice_session import VoiceAssistantEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app_logger = logging.getLogger("uvicorn.error")

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


def _latest_rlt_episode() -> Path | None:
    replay_dir = Path(settings.rlt_replay_dir)
    if not replay_dir.exists():
        return None
    files = sorted(replay_dir.glob("episode_*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files:
        if _try_read_rlt_metadata(path) is not None:
            return path
    return None


def _newest_rlt_episode_path() -> Path | None:
    replay_dir = Path(settings.rlt_replay_dir)
    if not replay_dir.exists():
        return None
    return max(replay_dir.glob("episode_*.npz"), key=lambda path: path.stat().st_mtime, default=None)


def _read_rlt_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            metadata = {}
        else:
            metadata = json.loads(str(data["metadata_json"]))
    label_metadata = _read_rlt_label_metadata(path)
    if label_metadata:
        metadata.update(label_metadata)
    return metadata


def _rlt_label_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".label.json")


def _read_rlt_label_metadata(path: Path) -> dict:
    label_path = _rlt_label_path(path)
    if not label_path.exists():
        return {}
    with label_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rlt_trim_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".trim.json")


def _read_rlt_trim_metadata(path: Path) -> dict:
    trim_path = _rlt_trim_path(path)
    if not trim_path.exists():
        return {}
    with trim_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_rlt_trim_metadata(path: Path, metadata: dict) -> None:
    trim_path = _rlt_trim_path(path)
    tmp_path = trim_path.with_name(f".{trim_path.name}.tmp")
    with tmp_path.open("wb") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        f.write(b"\n")
    tmp_path.replace(trim_path)


def _rlt_replay_root() -> Path:
    return Path(settings.rlt_replay_dir).expanduser().resolve()


def _safe_rlt_episode_path(path_value: str) -> Path:
    root = _rlt_replay_root()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.expanduser().resolve()
    if root not in (candidate, *candidate.parents):
        raise HTTPException(status_code=400, detail="episode path is outside RLT replay dir")
    if not candidate.exists() or candidate.suffix != ".npz":
        raise HTTPException(status_code=404, detail="episode npz not found")
    return candidate


def _relative_rlt_path(path: Path) -> str:
    root = _rlt_replay_root()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _rlt_image_keys(data: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(key.removeprefix("image_") for key in data.files if key.startswith("image_") and not key.startswith("image_mask_"))


def _rlt_trajectory_record(path: Path) -> RLTTrajectoryRecord | None:
    metadata = _try_read_rlt_metadata(path)
    if metadata is None:
        return None
    trim_metadata = _read_rlt_trim_metadata(path)
    metadata.update(trim_metadata)
    with np.load(path, allow_pickle=False) as data:
        if "raw_state" in data:
            num_steps = int(data["raw_state"].shape[0])
        elif "executed_action" in data:
            num_steps = int(data["executed_action"].shape[0])
        else:
            return None
        camera_names = _rlt_image_keys(data)
    fps = float(metadata.get("fps", 50.0) or 50.0)
    trim_start = int(metadata.get("trim_start_step", 0) or 0)
    trim_end = int(metadata.get("trim_end_step", num_steps) or num_steps)
    trim_start = max(0, min(trim_start, max(num_steps - 1, 0)))
    trim_end = max(trim_start + 1, min(trim_end, num_steps))
    return RLTTrajectoryRecord(
        path=_relative_rlt_path(path),
        name=path.name,
        terminal_label=metadata.get("terminal_label"),
        terminal_success=metadata.get("terminal_success"),
        num_steps=num_steps,
        num_chunks=metadata.get("num_chunks"),
        duration_s=float(num_steps / fps) if fps > 0 else None,
        fps=fps,
        camera_names=camera_names,
        trim_start_step=trim_start,
        trim_end_step=trim_end,
        mtime=path.stat().st_mtime,
    )


def _rlt_video_cache_path(path: Path, camera: str) -> Path:
    stat = path.stat()
    digest = hashlib.sha1(f"{path}:{camera}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()[:16]
    cache_dir = Path("/tmp/rlt_trajectory_video_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path.stem}_{camera}_{digest}.mp4"


def _write_rlt_video_cache(path: Path, camera: str) -> Path:
    output_path = _rlt_video_cache_path(path, camera)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    with np.load(path, allow_pickle=False) as data:
        key = f"image_{camera}"
        if key not in data:
            raise HTTPException(status_code=404, detail=f"camera {camera} not found in episode")
        frames = np.asarray(data[key])
        metadata = _read_rlt_metadata(path)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise HTTPException(status_code=422, detail=f"camera {camera} has no video frames")
    frames = frames.astype(np.uint8, copy=False)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    channels = int(frames.shape[3])
    if channels == 4:
        frames = frames[..., :3]
    elif channels == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif channels != 3:
        raise HTTPException(status_code=422, detail=f"camera {camera} frames must have 1, 3, or 4 channels")
    fps = float(metadata.get("fps", 50.0) or 50.0)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise HTTPException(status_code=500, detail=f"failed to build video with ffmpeg: {exc}") from exc
    return output_path


def _try_read_rlt_metadata(path: Path) -> dict | None:
    try:
        return _read_rlt_metadata(path)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        logging.warning("Skipping incomplete or invalid RLT replay npz: %s", path, exc_info=True)
        return None


def _write_rlt_label_metadata(path: Path, metadata: dict) -> None:
    start = time.perf_counter()
    label_path = _rlt_label_path(path)
    tmp_path = label_path.with_name(f".{label_path.name}.tmp")
    app_logger.info("RLT label sidecar write started: %s", label_path)
    with tmp_path.open("wb") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        f.write(b"\n")
    tmp_path.replace(label_path)
    app_logger.info("RLT label sidecar write finished in %.3fs: %s", time.perf_counter() - start, label_path)


def _rlt_status_from_episode(path: Path | None) -> RLTReplayStatus:
    if path is None:
        return RLTReplayStatus(replay_dir=settings.rlt_replay_dir)
    metadata = _try_read_rlt_metadata(path)
    if metadata is None:
        return RLTReplayStatus(replay_dir=settings.rlt_replay_dir)
    return RLTReplayStatus(
        replay_dir=settings.rlt_replay_dir,
        latest_episode=str(path),
        terminal_label=metadata.get("terminal_label"),
        terminal_success=metadata.get("terminal_success"),
        num_steps=metadata.get("num_steps"),
        num_chunks=metadata.get("num_chunks"),
    )


@app.get("/api/rlt/status", response_model=RLTReplayStatus)
def rlt_status() -> RLTReplayStatus:
    return _rlt_status_from_episode(_latest_rlt_episode())


@app.get("/api/rlt/trajectories", response_model=RLTTrajectoryListResponse)
def rlt_trajectories() -> RLTTrajectoryListResponse:
    replay_dir = _rlt_replay_root()
    records: list[RLTTrajectoryRecord] = []
    if replay_dir.exists():
        for path in sorted(replay_dir.glob("episode_*.npz"), key=lambda item: item.stat().st_mtime, reverse=True):
            if path.name.startswith("."):
                continue
            record = _rlt_trajectory_record(path)
            if record is not None:
                records.append(record)
    return RLTTrajectoryListResponse(replay_dir=str(replay_dir), trajectories=records)


@app.get("/api/rlt/trajectories/video")
def rlt_trajectory_video(path: str, camera: str = "cam_high") -> FileResponse:
    episode_path = _safe_rlt_episode_path(path)
    video_path = _write_rlt_video_cache(episode_path, camera)
    return FileResponse(video_path, media_type="video/mp4")


@app.post("/api/rlt/trajectories/trim", response_model=RLTTrajectoryRecord)
def rlt_trajectory_trim(request: RLTTrajectoryTrimRequest) -> RLTTrajectoryRecord:
    episode_path = _safe_rlt_episode_path(request.path)
    with np.load(episode_path, allow_pickle=False) as data:
        if "raw_state" in data:
            num_steps = int(data["raw_state"].shape[0])
        elif "executed_action" in data:
            num_steps = int(data["executed_action"].shape[0])
        else:
            raise HTTPException(status_code=422, detail="episode has no raw_state or executed_action")
    trim_start = max(0, min(int(request.trim_start_step), max(num_steps - 1, 0)))
    trim_end = max(trim_start + 1, min(int(request.trim_end_step), num_steps))
    metadata = {
        "trim_start_step": trim_start,
        "trim_end_step": trim_end,
        "updated_at": time.time(),
    }
    if request.terminal_label is not None:
        label = request.terminal_label.lower()
        if label not in {"success", "failure", "unlabeled"}:
            raise HTTPException(status_code=400, detail="terminal_label must be success, failure, or unlabeled")
        metadata["terminal_label"] = label
        metadata["terminal_success"] = 1 if label == "success" else 0 if label == "failure" else -1
    _write_rlt_trim_metadata(episode_path, metadata)
    record = _rlt_trajectory_record(episode_path)
    if record is None:
        raise HTTPException(status_code=422, detail="failed to read saved trajectory metadata")
    return record


@app.post("/api/rlt/start")
def rlt_start() -> dict[str, str]:
    publish_task(redis_client, "2")
    return {"status": "ok", "task_number": "2", "task_name": TASK_MAPPING["2"]}


@app.post("/api/rlt/record")
def rlt_record() -> dict[str, str]:
    publish_task(redis_client, "rlt_begin_recording", task_name="Start RLT replay recording")
    return {"status": "ok", "task_number": "rlt_begin_recording", "task_name": "Start RLT replay recording"}


@app.post("/api/rlt/actor/enable")
def rlt_actor_enable() -> dict[str, str]:
    publish_task(redis_client, "rlt_actor_enable", task_name="Enable RLT actor sampling")
    return {"status": "ok", "task_number": "rlt_actor_enable", "task_name": "Enable RLT actor sampling"}


@app.post("/api/rlt/actor/disable")
def rlt_actor_disable() -> dict[str, str]:
    publish_task(redis_client, "rlt_actor_disable", task_name="Disable RLT actor sampling")
    return {"status": "ok", "task_number": "rlt_actor_disable", "task_name": "Disable RLT actor sampling"}


@app.post("/api/rlt/end")
def rlt_end() -> dict[str, str]:
    publish_task(redis_client, "4", defer_episode_save=True)
    return {"status": "ok", "task_number": "4", "task_name": TASK_MAPPING["4"]}


@app.post("/api/rlt/label", response_model=RLTReplayStatus)
async def rlt_label(request: RLTLabelRequest) -> RLTReplayStatus:
    request_start = time.perf_counter()
    label = request.label.lower()
    app_logger.info("RLT label request started: label=%s", label)
    if label not in {"success", "failure"}:
        raise HTTPException(status_code=400, detail="label must be success or failure")

    previous_episode = _latest_rlt_episode()
    publish_task(
        redis_client,
        "rlt_save_label",
        task_name="Save RLT replay with label",
        rlt_terminal_label=label,
    )
    app_logger.info(
        "RLT label save command published after %.3fs: label=%s previous=%s",
        time.perf_counter() - request_start,
        label,
        previous_episode,
    )

    episode_path: Path | None = None
    metadata: dict | None = None
    deadline = time.time() + 180.0
    while time.time() < deadline:
        episode_path = _latest_rlt_episode()
        if episode_path is not None:
            metadata = _try_read_rlt_metadata(episode_path)
        if episode_path is not None and episode_path != previous_episode and metadata is not None:
            break
        await asyncio.sleep(0.25)
    if episode_path is None or episode_path == previous_episode or metadata is None:
        raise HTTPException(status_code=409, detail="RLT replay has not been saved yet. Try again.")
    app_logger.info("RLT labeled replay ready after %.3fs: %s", time.perf_counter() - request_start, episode_path)
    app_logger.info("RLT label request finished in %.3fs: %s", time.perf_counter() - request_start, episode_path)
    return _rlt_status_from_episode(episode_path)


@app.post("/api/voice/audio", response_model=VoiceResponse)
async def voice_audio(file: UploadFile = File(...), language: str = Form("en")) -> VoiceResponse:
    return await voice_engine.process_audio(file, language=language)

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import time

from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from .camera_bridge import CameraBridge
from .config import settings
from .redis_commands import create_redis_client
from .rlt_control import RLTControlStore
from .robot_state_bridge import RobotStateBridge
from .schemas import HealthResponse
from .schemas import RealtimePayload
from .schemas import RLTConfigRequest
from .schemas import RLTControlRequest
from .schemas import RLTControlState
from .schemas import RLTScoreRequest
from .schemas import RobotTaskRequest
from .schemas import RuntimeStatePayload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="EII Pilot")
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
rlt_control = RLTControlStore(redis_client)
ROLLOUTS_ROOT = Path(settings.rollouts_root).expanduser().resolve()
VIDEO_CHUNK_SIZE = 1024 * 1024
VIDEO_CACHE_ROOT = Path(os.getenv("ROLLOUTS_VIDEO_CACHE", "/tmp/eii_rollout_video_cache"))
ROBOT_TASK_LABELS = {
    "1": "twist bottle",
    "4": "home",
    "5": "sleep",
}


@app.on_event("startup")
def on_startup() -> None:
    try:
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("eii_pilot_backend", anonymous=True)
    except Exception:
        logging.exception("ROS node initialization failed")
    camera_bridge.start()
    robot_state_bridge.start()
    rlt_control.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    camera_bridge.stop()
    robot_state_bridge.stop()
    rlt_control.stop()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/robot/task")
def robot_task(request: RobotTaskRequest) -> dict[str, str]:
    payload = {
        "type": "robot_task",
        "task_num": request.task_num,
        "task_name": ROBOT_TASK_LABELS[request.task_num],
        "source": request.source,
        "timestamp": time.time(),
    }
    redis_client.publish(settings.rlt_control_channel, json.dumps(payload))
    return {"status": "ok", "task_num": request.task_num, "task_name": payload["task_name"]}


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


def _safe_rollout_path(relative_path: str) -> Path:
    candidate = (ROLLOUTS_ROOT / relative_path).resolve()
    if candidate != ROLLOUTS_ROOT and ROLLOUTS_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid rollout path")
    return candidate


def _scan_rollout_tree(path: Path, relative_path: str = "") -> dict:
    try:
        entries = list(path.iterdir())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Rollouts root not found: {ROLLOUTS_ROOT}") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {relative_path or '.'}") from None

    children = []
    for entry in entries:
        rel = f"{relative_path}/{entry.name}" if relative_path else entry.name
        if entry.is_dir():
            children.append(_scan_rollout_tree(entry, rel))
            continue
        if entry.suffix.lower() not in {".mp4", ".hdf5"}:
            continue
        stat = entry.stat()
        children.append(
            {
                "name": entry.name,
                "path": rel,
                "type": "file",
                "extension": entry.suffix.lower(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )

    def natural_key(value: str) -> list[int | str]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    children.sort(key=lambda item: (item["type"] == "file", natural_key(item["name"])))
    stat = path.stat()
    return {
        "name": path.name or "rollouts",
        "path": relative_path,
        "type": "directory",
        "modified": stat.st_mtime,
        "children": children,
    }


@app.get("/api/rollouts/tree")
def rollout_tree() -> dict:
    return _scan_rollout_tree(ROLLOUTS_ROOT)


def _parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int, int]:
    if not range_header:
        return 0, file_size - 1, 200
    unit, _, range_spec = range_header.partition("=")
    if unit.strip().lower() != "bytes" or not range_spec:
        raise HTTPException(status_code=416, detail="Invalid range header")

    start_text, _, end_text = range_spec.partition("-")
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_length = int(end_text)
            start = max(file_size - suffix_length, 0)
            end = file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid range header") from None

    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable")
    return start, min(end, file_size - 1), 206


def _file_iterator(path: Path, start: int, end: int):
    with path.open("rb") as file:
        file.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = file.read(min(VIDEO_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _video_codec(path: Path) -> str:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.warning("ffprobe failed for %s: %s", path, exc.stderr)
        return ""
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""


def _cache_path_for_video(path: Path) -> Path:
    stat = path.stat()
    digest = hashlib.sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    return VIDEO_CACHE_ROOT / f"{digest}.h264.mp4"


def _wait_for_cache(cache_path: Path, lock_path: Path) -> None:
    deadline = time.time() + 300
    while lock_path.exists() and not cache_path.exists():
        if time.time() > deadline:
            raise HTTPException(status_code=503, detail="Timed out waiting for video transcode")
        time.sleep(0.25)


def _browser_playable_video(path: Path) -> Path:
    codec = _video_codec(path)
    if codec == "h264":
        return path

    cache_path = _cache_path_for_video(path)
    if cache_path.exists():
        return cache_path

    VIDEO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _wait_for_cache(cache_path, lock_path)
        if cache_path.exists():
            return cache_path
        raise HTTPException(status_code=503, detail="Video transcode did not complete") from None

    tmp_path = cache_path.with_suffix(".tmp.mp4")
    try:
        os.close(lock_fd)
        logging.info("Transcoding rollout video for browser playback: %s -> %s", path, cache_path)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path),
            ],
            check=True,
        )
        tmp_path.replace(cache_path)
    except subprocess.CalledProcessError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        logging.exception("Video transcode failed for %s", path)
        raise HTTPException(status_code=500, detail="Video transcode failed") from exc
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return cache_path


@app.get("/api/rollouts/video")
def rollout_video(path: str, range_header: str | None = Header(default=None, alias="Range")) -> StreamingResponse:
    video_path = _safe_rollout_path(path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    if video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Only mp4 files can be streamed")

    playable_path = _browser_playable_video(video_path)
    file_size = playable_path.stat().st_size
    start, end, status_code = _parse_range_header(range_header, file_size)
    media_type = mimetypes.guess_type(playable_path.name)[0] or "video/mp4"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "no-store",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(
        _file_iterator(playable_path, start, end),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


@app.head("/api/rollouts/video")
def rollout_video_head(path: str) -> Response:
    video_path = _safe_rollout_path(path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    if video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Only mp4 files can be streamed")

    playable_path = _browser_playable_video(video_path)
    return Response(
        status_code=200,
        media_type=mimetypes.guess_type(playable_path.name)[0] or "video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(playable_path.stat().st_size),
            "Cache-Control": "no-store",
        },
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
                rlt=rlt_control.snapshot(),
            )
            await websocket.send_json(payload.model_dump())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


@app.get("/api/rlt/status", response_model=RLTControlState)
def rlt_status() -> RLTControlState:
    return rlt_control.snapshot()


@app.post("/api/rlt/key-region/start", response_model=RLTControlState)
def rlt_key_region_start(request: RLTControlRequest | None = None) -> RLTControlState:
    try:
        return rlt_control.start_key_region(request or RLTControlRequest())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/rlt/key-region/end", response_model=RLTControlState)
def rlt_key_region_end(request: RLTControlRequest | None = None) -> RLTControlState:
    try:
        return rlt_control.end_key_region(request or RLTControlRequest())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/rlt/key-region/score", response_model=RLTControlState)
def rlt_key_region_score(request: RLTScoreRequest) -> RLTControlState:
    try:
        return rlt_control.score_key_region(request.reward, source=request.source)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/rlt/config", response_model=RLTControlState)
def rlt_config(request: RLTConfigRequest) -> RLTControlState:
    return rlt_control.update_config(request)

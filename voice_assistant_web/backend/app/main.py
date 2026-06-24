from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sqlite3
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
from .rlt_key_region_crop import crop_key_region_files
from .rlt_key_region_crop import rescore_key_region_files
from .robot_state_bridge import RobotStateBridge
from .schemas import HealthResponse
from .schemas import CameraCapabilitiesResponse
from .schemas import RealtimePayload
from .schemas import RLTBatchSegmentRequest
from .schemas import RLTConfigRequest
from .schemas import RLTControlRequest
from .schemas import RLTControlState
from .schemas import RLTDiscardRequest
from .schemas import RLTKeyRegionCropRequest
from .schemas import RLTKeyRegionCropResponse
from .schemas import RLTPreferencePairResponse
from .schemas import RLTPreferenceRecord
from .schemas import RLTPreferenceRequest
from .schemas import RLTPreferenceStats
from .schemas import RLTKeyRegionReviewPage
from .schemas import RLTKeyRegionReviewRecord
from .schemas import RLTKeyRegionReviewSummary
from .schemas import RLTKeyRegionRescoreRequest
from .schemas import RLTScoreRequest
from .schemas import RLTSegmentRecord
from .schemas import RLTVoidRequest
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
REPLAY_ROOT = Path(settings.replay_root).expanduser().resolve()
VIDEO_CHUNK_SIZE = 1024 * 1024
VIDEO_CACHE_ROOT = Path(os.getenv("ROLLOUTS_VIDEO_CACHE", "/tmp/eii_rollout_video_cache"))
DEFAULT_RLT_PRE_ROLL_SECONDS = float(os.getenv("RLT_DEFAULT_PRE_ROLL_SECONDS", "2.0"))
PREFERENCE_ROUND_BUDGET = int(os.getenv("RLT_PREFERENCE_ROUND_BUDGET", "800"))
PREFERENCE_SAMPLE_ROUND = int(os.getenv("RLT_PREFERENCE_SAMPLE_ROUND", "1"))
ROBOT_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS = float(os.getenv("ROBOT_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS", "3.0"))
PREFERENCE_PAIR_TYPE_QUOTAS = {
    "success_success": 0.40,
    "success_failure": 0.30,
    "failure_failure": 0.20,
}
ROBOT_TASK_LABELS = {
    "1": "twist bottle",
    "4": "home",
    "5": "sleep",
    "9": "shutdown",
}


@app.on_event("startup")
def on_startup() -> None:
    if settings.enable_ros:
        try:
            import rospy

            if not rospy.core.is_initialized():
                rospy.init_node("eii_pilot_backend", anonymous=True, disable_signals=True)
        except Exception:
            logging.exception("ROS node initialization failed")
        camera_bridge.start()
        robot_state_bridge.start()
    else:
        logging.info("ROS bridges disabled by EII_PILOT_ENABLE_ROS=false")
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
    runtime_timestamp = robot_state_bridge.snapshot().get("runtime_timestamp")
    if runtime_timestamp is None or time.time() - float(runtime_timestamp) > ROBOT_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS:
        raise HTTPException(status_code=409, detail="runtime is not listening; restart robot runtime before sending tasks")
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


@app.get("/api/cameras/capabilities", response_model=CameraCapabilitiesResponse)
def camera_capabilities() -> CameraCapabilitiesResponse:
    transports = ["mjpeg", "jpeg_ws"]
    if settings.camera_webrtc_enabled:
        transports.insert(0, "webrtc")
    preferred_transport = settings.camera_transport if settings.camera_transport in transports else transports[0]
    return CameraCapabilitiesResponse(
        preferred_transport=preferred_transport,
        transports=transports,
        cameras=list(camera_bridge.camera_names),
        include_realtime_frames=settings.realtime_include_camera_frames,
        webrtc={
            "enabled": settings.camera_webrtc_enabled,
            "codec": "h264",
            "ice_servers": [],
        },
    )


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


def _realtime_camera_frames() -> dict[str, str]:
    if not settings.realtime_include_camera_frames:
        return {}
    return camera_bridge.snapshot_jpeg_b64_all()


def _safe_rollout_path(relative_path: str) -> Path:
    candidate = (ROLLOUTS_ROOT / relative_path).resolve()
    if candidate != ROLLOUTS_ROOT and ROLLOUTS_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid rollout path")
    return candidate


def _float_or_none(value: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return None


def _manifest_summary(path: Path) -> dict | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        logging.warning("Could not read rollout manifest: %s", manifest_path)
        return None

    summary_keys = {
        "key_region_id",
        "task",
        "phase",
        "reward",
        "score_timeout",
        "start_time",
        "end_time",
        "score_time",
        "duration_seconds",
        "key_region_duration_seconds",
        "key_region_start_sec",
        "key_region_end_sec",
        "pre_roll_seconds",
        "post_roll_seconds",
        "num_frames",
        "num_replay_transitions",
        "fps",
        "crop_start_sec",
        "crop_end_sec",
        "crop_start_sample",
        "crop_end_sample",
        "crop_original_num_replay_transitions",
        "segment_status",
        "train_eligible",
        "replay_status",
        "missing_rlt_metadata",
        "voided",
        "shard_path",
    }
    summary = {key: manifest[key] for key in summary_keys if key in manifest}

    key_region_duration = _float_or_none(summary.get("key_region_duration_seconds"))
    start_time = _float_or_none(summary.get("start_time"))
    end_time = _float_or_none(summary.get("end_time"))
    if key_region_duration is None and start_time is not None and end_time is not None:
        key_region_duration = max(end_time - start_time, 0.0)
    if key_region_duration is not None:
        summary["key_region_duration_seconds"] = key_region_duration

    video_duration = None
    with contextlib.suppress(TypeError, ValueError, ZeroDivisionError):
        num_frames = int(summary.get("num_frames", 0))
        fps = float(summary.get("fps", 0.0))
        if num_frames > 0 and fps > 0:
            video_duration = num_frames / fps
    if video_duration is None:
        video_duration = _float_or_none(summary.get("duration_seconds"))
    if video_duration is None and key_region_duration is not None:
        video_duration = key_region_duration
    if video_duration is not None:
        summary["duration_seconds"] = max(video_duration, 0.0)

    if video_duration is not None and key_region_duration is not None:
        key_region_start = _float_or_none(summary.get("key_region_start_sec"))
        if key_region_start is None:
            extra_context = max(video_duration - key_region_duration, 0.0)
            key_region_start = min(DEFAULT_RLT_PRE_ROLL_SECONDS, extra_context)
        key_region_end = _float_or_none(summary.get("key_region_end_sec"))
        if key_region_end is None:
            key_region_end = key_region_start + key_region_duration
        summary["key_region_start_sec"] = min(max(key_region_start, 0.0), max(video_duration, 0.0))
        summary["key_region_end_sec"] = min(
            max(key_region_end, summary["key_region_start_sec"]),
            max(video_duration, 0.0),
        )
    return summary


def _scan_rollout_tree(path: Path, relative_path: str = "") -> dict:
    try:
        entries = list(path.iterdir())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Rollout path not found: {relative_path or '.'}") from None
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

    def rollout_sort_key(item: dict) -> tuple:
        if item["type"] == "file":
            return (2, natural_key(item["name"]))
        if item.get("manifest_summary", {}).get("key_region_id"):
            return (0, -(item.get("modified") or 0), natural_key(item["name"]))
        return (1, natural_key(item["name"]))

    children.sort(key=rollout_sort_key)
    stat = path.stat()
    result = {
        "name": path.name or "rollouts",
        "path": relative_path,
        "type": "directory",
        "modified": stat.st_mtime,
        "children": children,
    }
    manifest = _manifest_summary(path)
    if manifest:
        result["manifest_summary"] = manifest
    return result


@app.get("/api/rollouts/tree")
def rollout_tree(path: str = "") -> dict:
    rollout_path = _safe_rollout_path(path)
    if not rollout_path.exists() or not rollout_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Rollout path not found: {path or '.'}")
    relative_path = "" if rollout_path == ROLLOUTS_ROOT else str(rollout_path.relative_to(ROLLOUTS_ROOT))
    return _scan_rollout_tree(rollout_path, relative_path)


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
                camera_jpeg_b64 = _realtime_camera_frames()
                last_camera_push = now
            payload = RealtimePayload(
                robot=RuntimeStatePayload(**robot_state_bridge.snapshot()),
                camera_status=camera_bridge.get_camera_status(),
                camera_timestamps=camera_bridge.get_camera_timestamps(),
                camera_jpeg_b64=camera_jpeg_b64,
                rlt=rlt_control.snapshot_fast(),
            )
            await websocket.send_json(payload.model_dump())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


@app.get("/api/rlt/status", response_model=RLTControlState)
def rlt_status() -> RLTControlState:
    return rlt_control.snapshot()


@app.get("/api/rlt/segments", response_model=list[RLTSegmentRecord])
def rlt_segments(limit: int = 500) -> list[RLTSegmentRecord]:
    return [RLTSegmentRecord(**segment) for segment in rlt_control.list_segments(limit=limit)]


def _host_path_for_container_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() and candidate.parts[:3] == ("/", "app", "replay"):
        try:
            return (REPLAY_ROOT / candidate.relative_to("/app/replay")).resolve()
        except ValueError:
            return candidate
    return candidate


def _batch_from_rollout_path(path: Path) -> str | None:
    with contextlib.suppress(ValueError):
        parts = path.resolve().relative_to(ROLLOUTS_ROOT).parts
        if len(parts) >= 3 and parts[0] == "key_regions":
            return parts[2]
    return None


def _batch_from_replay_path(path: Path) -> str | None:
    with contextlib.suppress(ValueError):
        parts = path.resolve().relative_to(REPLAY_ROOT).parts
        if "manual" in parts:
            return "manual"
        for part in parts:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
                return part
        if len(parts) >= 3 and parts[0] == "rlt_key_regions":
            return parts[2]
    return None


def _key_region_video_paths(rollout_dir: Path) -> list[str]:
    if not rollout_dir.exists() or not rollout_dir.is_dir():
        return []
    videos = sorted(rollout_dir.glob("*.mp4"), key=lambda item: (item.name != "cam_right_wrist.mp4", item.name))
    result = []
    for video in videos:
        with contextlib.suppress(ValueError):
            result.append(str(video.resolve().relative_to(ROLLOUTS_ROOT)))
    return result


def _record_has_trainable_files(record: dict) -> bool:
    return (
        bool(record.get("npz_exists"))
        and bool(record.get("shard_path"))
        and record.get("train_eligible") is not False
        and not bool(record.get("voided"))
        and str(record.get("segment_status") or "committed") == "committed"
        and int(record.get("num_replay_transitions") or 0) > 0
    )


def _record_can_auto_commit(record: dict) -> bool:
    status = str(record.get("status") or "untracked")
    return status in {"accepted", "untracked", "orphan_npz", "ended"} and _record_has_trainable_files(record)


def _record_needs_crop(record: dict) -> bool:
    if not bool(record.get("video_exists")):
        return False
    if not bool(record.get("npz_exists")) or not bool(record.get("shard_path")):
        return False
    if bool(record.get("voided")):
        return False
    return (
        record.get("crop_start_sec") is None
        or record.get("crop_end_sec") is None
        or record.get("train_eligible") is not True
        or str(record.get("segment_status") or "") != "committed"
    )


def _reconcile_key_region_record(record: dict) -> None:
    if not _record_can_auto_commit(record):
        return
    rlt_control.commit_key_region_from_files(
        key_region_id=str(record["key_region_id"]),
        phase=str(record.get("phase") or "warmup"),
        reward=int(record.get("reward") or 0),
        shard_path=str(record["shard_path"]),
        num_replay_transitions=int(record.get("num_replay_transitions") or 0),
    )
    record["status"] = "committed"


def _key_region_review_batches_from_files() -> list[str]:
    batches: set[str] = set()
    rollout_root = (ROLLOUTS_ROOT / "key_regions").resolve()
    if rollout_root.exists():
        for manifest_path in rollout_root.glob("**/key_region_*/manifest.json"):
            batch = _batch_from_rollout_path(manifest_path.parent)
            if batch:
                batches.add(batch)
    replay_root = (REPLAY_ROOT / "rlt_key_regions").resolve()
    if replay_root.exists():
        for shard_path in replay_root.glob("**/shards/key_region_*.npz"):
            batch = _batch_from_replay_path(shard_path)
            if batch:
                batches.add(batch)
    return sorted(batches, reverse=True)


def _segment_batch(segment: dict) -> str | None:
    shard_path = _host_path_for_container_path(segment.get("shard_path"))
    if shard_path is not None:
        batch = _batch_from_replay_path(shard_path)
        if batch:
            return batch
    updated_at = segment.get("updated_at")
    with contextlib.suppress(TypeError, ValueError, OSError):
        return time.strftime("%Y-%m-%d", time.localtime(float(updated_at)))
    return None


def _key_region_review_records(*, batch: str = "all") -> list[dict]:
    batch_filter = None if not batch or batch == "all" else batch
    by_id: dict[str, dict] = {}
    segments = rlt_control.list_segments(limit=100000)
    for segment in segments:
        key_region_id = str(segment.get("key_region_id") or "")
        if not key_region_id:
            continue
        if str(segment.get("status") or "") == "deleted":
            continue
        segment_batch = _segment_batch(segment)
        if batch_filter and segment_batch and segment_batch != batch_filter:
            continue
        by_id[key_region_id] = {
            "key_region_id": key_region_id,
            "status": str(segment.get("status") or "untracked"),
            "phase": segment.get("phase"),
            "reward": segment.get("reward"),
            "shard_path": segment.get("shard_path"),
            "num_replay_transitions": int(segment.get("num_replay_transitions") or 0),
            "updated_at": segment.get("updated_at"),
            "batch": segment_batch,
        }

    rollout_root = (ROLLOUTS_ROOT / "key_regions").resolve()
    if rollout_root.exists():
        for manifest_path in rollout_root.glob("**/key_region_*/manifest.json"):
            rollout_dir = manifest_path.parent
            rollout_batch = _batch_from_rollout_path(rollout_dir)
            if batch_filter and rollout_batch != batch_filter:
                continue
            key_region_id = rollout_dir.name.removeprefix("key_region_")
            record = by_id.setdefault(key_region_id, {"key_region_id": key_region_id, "status": "untracked"})
            manifest = _manifest_summary(rollout_dir) or {}
            record.update(
                {
                    "batch": record.get("batch") or rollout_batch,
                    "manifest_exists": True,
                    "video_exists": bool(_key_region_video_paths(rollout_dir)),
                    "rollout_path": str(rollout_dir.resolve().relative_to(ROLLOUTS_ROOT)),
                    "local_rollout_path": str(rollout_dir.resolve()),
                    "video_paths": _key_region_video_paths(rollout_dir),
                    "task": manifest.get("task"),
                    "start_time": manifest.get("start_time"),
                    "end_time": manifest.get("end_time"),
                    "score_time": manifest.get("score_time"),
                    "duration_seconds": manifest.get("duration_seconds"),
                    "key_region_duration_seconds": manifest.get("key_region_duration_seconds"),
                    "key_region_start_sec": manifest.get("key_region_start_sec"),
                    "key_region_end_sec": manifest.get("key_region_end_sec"),
                    "fps": manifest.get("fps"),
                    "num_frames": manifest.get("num_frames"),
                    "crop_start_sec": manifest.get("crop_start_sec"),
                    "crop_end_sec": manifest.get("crop_end_sec"),
                    "crop_start_sample": manifest.get("crop_start_sample"),
                    "crop_end_sample": manifest.get("crop_end_sample"),
                    "crop_original_num_replay_transitions": manifest.get("crop_original_num_replay_transitions"),
                    "phase": record.get("phase") or manifest.get("phase"),
                    "segment_status": manifest.get("segment_status"),
                    "train_eligible": manifest.get("train_eligible"),
                    "replay_status": manifest.get("replay_status"),
                    "missing_rlt_metadata": manifest.get("missing_rlt_metadata") or [],
                    "voided": manifest.get("voided"),
                    "shard_path": record.get("shard_path") or manifest.get("shard_path"),
                    "reward": record.get("reward") if record.get("reward") is not None else manifest.get("reward"),
                    "num_replay_transitions": record.get("num_replay_transitions") or manifest.get("num_replay_transitions") or 0,
                }
            )
            if record["video_paths"]:
                record["default_video_path"] = record["video_paths"][0]

    replay_root = (REPLAY_ROOT / "rlt_key_regions").resolve()
    if replay_root.exists():
        for shard_path in replay_root.glob("**/shards/key_region_*.npz"):
            shard_batch = _batch_from_replay_path(shard_path)
            if batch_filter and shard_batch != batch_filter:
                continue
            key_region_id = shard_path.stem.removeprefix("key_region_")
            record = by_id.setdefault(key_region_id, {"key_region_id": key_region_id, "status": "orphan_npz"})
            record["npz_exists"] = True
            record["batch"] = record.get("batch") or shard_batch
            if not record.get("shard_path"):
                record["shard_path"] = str(shard_path)

    for record in by_id.values():
        shard_path = _host_path_for_container_path(record.get("shard_path"))
        record["npz_exists"] = False
        if shard_path and shard_path.exists():
            record["npz_exists"] = True
            record["local_shard_path"] = str(shard_path.resolve())
            record["batch"] = record.get("batch") or _batch_from_replay_path(shard_path)
        record["manifest_exists"] = bool(record.get("manifest_exists"))
        record["video_exists"] = bool(record.get("video_exists"))
        _reconcile_key_region_record(record)
        record["trainable"] = record.get("status") == "committed" and _record_has_trainable_files(record)
        record["needs_crop"] = _record_needs_crop(record)
        if not record["trainable"]:
            if record.get("status") != "committed":
                reason = f"not_committed:{record.get('status') or 'untracked'}"
            elif not record.get("shard_path") or not record.get("npz_exists"):
                reason = "missing_npz"
            elif not record.get("manifest_exists"):
                reason = "missing_manifest"
            elif not record.get("video_exists"):
                reason = "missing_video"
            elif record.get("train_eligible") is not True or record.get("segment_status") != "committed":
                reason = "not_train_eligible"
            elif record.get("voided"):
                reason = "voided_manifest"
            else:
                reason = "not_trainable"
            record["incomplete_reason"] = reason
    return sorted(by_id.values(), key=lambda item: item.get("score_time") or item.get("updated_at") or 0, reverse=True)


def _key_region_review_summary(records: list[dict]) -> RLTKeyRegionReviewSummary:
    return RLTKeyRegionReviewSummary(
        total=len(records),
        trainable=sum(1 for record in records if record.get("trainable")),
        needs_crop=sum(1 for record in records if record.get("needs_crop")),
        success=sum(1 for record in records if record.get("reward") == 1),
        failure=sum(1 for record in records if record.get("reward") == 0),
        replay_samples=sum(int(record.get("num_replay_transitions") or 0) for record in records),
    )


def _filter_key_region_review_records(
    records: list[dict],
    *,
    status: str = "all",
    reward: str = "all",
    batch: str = "all",
) -> list[dict]:
    filtered = records
    if batch and batch != "all":
        filtered = [record for record in filtered if str(record.get("batch") or "") == batch]

    if status == "trainable":
        filtered = [record for record in filtered if record.get("trainable")]
    elif status in {"needsCrop", "needs_crop"}:
        filtered = [record for record in filtered if record.get("needs_crop")]
    elif status != "all":
        filtered = [record for record in filtered if str(record.get("status") or "") == status]

    if reward in {"success", "1"}:
        filtered = [record for record in filtered if record.get("reward") == 1]
    elif reward in {"failure", "0"}:
        filtered = [record for record in filtered if record.get("reward") == 0]
    return filtered


def _key_region_review_batches(records: list[dict]) -> list[str]:
    return sorted({str(record["batch"]) for record in records if record.get("batch")}, reverse=True)


def _preference_pair_key(left_key_region_id: str, right_key_region_id: str) -> str:
    first, second = sorted((left_key_region_id, right_key_region_id))
    return f"{first}::{second}"


def _connect_preference_db():
    db_path = Path(settings.rlt_segment_db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preference_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            left_key_region_id TEXT NOT NULL,
            right_key_region_id TEXT NOT NULL,
            pair_key TEXT NOT NULL,
            preference TEXT NOT NULL,
            reason_tags TEXT NOT NULL DEFAULT '[]',
            notes TEXT,
            source TEXT NOT NULL DEFAULT 'ui',
            created_at REAL NOT NULL
        )
        """
    )
    rows = conn.execute("PRAGMA table_info(preference_pairs)").fetchall()
    columns = {str(row["name"]) for row in rows}
    for name, definition in (
        ("pair_type", "TEXT"),
        ("strategy", "TEXT NOT NULL DEFAULT 'budgeted'"),
        ("sample_round", "INTEGER NOT NULL DEFAULT 1"),
        ("left_reward", "INTEGER"),
        ("right_reward", "INTEGER"),
        ("left_batch", "TEXT"),
        ("right_batch", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE preference_pairs ADD COLUMN {name} {definition}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_preference_pairs_pair_key ON preference_pairs(pair_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_preference_pairs_round_type ON preference_pairs(sample_round, pair_type)")
    return conn


def _preference_stats(conn) -> RLTPreferenceStats:
    rows = conn.execute("SELECT preference, COUNT(*) AS count FROM preference_pairs GROUP BY preference").fetchall()
    counts = {str(row["preference"]): int(row["count"]) for row in rows}
    return RLTPreferenceStats(
        total_preferences=sum(counts.values()),
        left_wins=counts.get("left", 0),
        right_wins=counts.get("right", 0),
        ties=counts.get("tie", 0),
        both_bad=counts.get("both_bad", 0),
        skipped=counts.get("skip", 0),
    )


def _seen_preference_pairs(conn) -> set[str]:
    rows = conn.execute("SELECT DISTINCT pair_key FROM preference_pairs WHERE preference != 'skip'").fetchall()
    return {str(row["pair_key"]) for row in rows}


def _preference_pair_type(left: dict, right: dict) -> str:
    left_reward = int(left.get("reward") or 0)
    right_reward = int(right.get("reward") or 0)
    if left_reward == 1 and right_reward == 1:
        return "success_success"
    if left_reward == 0 and right_reward == 0:
        return "failure_failure"
    return "success_failure"


def _record_is_clean_preference_candidate(record: dict) -> bool:
    return (
        bool(record.get("trainable"))
        and bool(record.get("video_paths"))
        and record.get("crop_start_sec") is not None
        and record.get("crop_end_sec") is not None
        and "rlt_key_regions_clean" in str(record.get("shard_path") or "")
    )


def _preference_candidate_records(*, batch: str, reward: str) -> list[dict]:
    records = _filter_key_region_review_records(_key_region_review_records(), status="trainable", reward=reward, batch=batch)
    return [record for record in records if _record_is_clean_preference_candidate(record)]


def _preference_round_counts(conn) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT pair_type, COUNT(*) AS count
        FROM preference_pairs
        WHERE sample_round = ? AND preference != 'skip'
        GROUP BY pair_type
        """,
        (PREFERENCE_SAMPLE_ROUND,),
    ).fetchall()
    return {str(row["pair_type"] or ""): int(row["count"]) for row in rows}


def _preference_degree(conn) -> dict[str, int]:
    degree: dict[str, int] = {}
    rows = conn.execute(
        """
        SELECT left_key_region_id, right_key_region_id
        FROM preference_pairs
        WHERE sample_round = ? AND preference != 'skip'
        """,
        (PREFERENCE_SAMPLE_ROUND,),
    ).fetchall()
    for row in rows:
        for key in (str(row["left_key_region_id"]), str(row["right_key_region_id"])):
            degree[key] = degree.get(key, 0) + 1
    return degree


def _target_pair_type(round_counts: dict[str, int], requested: str) -> str:
    if requested in PREFERENCE_PAIR_TYPE_QUOTAS:
        return requested
    labeled = sum(round_counts.values())
    if labeled >= PREFERENCE_ROUND_BUDGET:
        return "success_success"
    best_type = "success_success"
    best_gap = float("-inf")
    for pair_type, quota in PREFERENCE_PAIR_TYPE_QUOTAS.items():
        target = PREFERENCE_ROUND_BUDGET * quota
        gap = target - round_counts.get(pair_type, 0)
        if gap > best_gap:
            best_type = pair_type
            best_gap = gap
    return best_type


def _select_preference_pair(
    records: list[dict],
    seen_pairs: set[str],
    *,
    pair_type: str,
    degree: dict[str, int],
) -> tuple[dict | None, dict | None, int]:
    candidates: list[tuple[tuple[int, int, int, str], dict, dict]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if _preference_pair_type(left, right) != pair_type:
                continue
            pair_key = _preference_pair_key(str(left["key_region_id"]), str(right["key_region_id"]))
            if pair_key in seen_pairs:
                continue
            left_degree = degree.get(str(left["key_region_id"]), 0)
            right_degree = degree.get(str(right["key_region_id"]), 0)
            same_batch_penalty = 0 if str(left.get("batch") or "") == str(right.get("batch") or "") else 1
            stable_hash = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()
            candidates.append(((left_degree + right_degree, max(left_degree, right_degree), same_batch_penalty, stable_hash), left, right))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return None, None, 0
    _, left, right = candidates[0]
    return left, right, len(candidates)


@app.get("/api/rlt/key-regions/review", response_model=RLTKeyRegionReviewPage)
def rlt_key_region_review(
    limit: int = 20,
    offset: int = 0,
    status: str = "all",
    reward: str = "all",
    batch: str = "all",
    focus_key_region_id: str | None = None,
) -> RLTKeyRegionReviewPage:
    if focus_key_region_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", focus_key_region_id):
        raise HTTPException(status_code=400, detail=f"Invalid key_region_id: {focus_key_region_id}")
    batches = _key_region_review_batches_from_files()
    resolved_batch = batches[0] if batch == "latest" and batches else batch
    all_records = _key_region_review_records(batch=resolved_batch)
    filtered = _filter_key_region_review_records(all_records, status=status, reward=reward, batch=resolved_batch)
    safe_limit = min(max(limit, 1), 100)
    safe_offset = min(max(offset, 0), len(filtered))
    if focus_key_region_id:
        focus_index = next(
            (index for index, record in enumerate(filtered) if record.get("key_region_id") == focus_key_region_id),
            None,
        )
        if focus_index is not None:
            safe_offset = focus_index
    page_records = filtered[safe_offset : safe_offset + safe_limit]
    next_offset = safe_offset + safe_limit if safe_offset + safe_limit < len(filtered) else None
    return RLTKeyRegionReviewPage(
        items=[RLTKeyRegionReviewRecord(**record) for record in page_records],
        total=len(filtered),
        limit=safe_limit,
        offset=safe_offset,
        next_offset=next_offset,
        summary=_key_region_review_summary(all_records),
        batches=batches or _key_region_review_batches(all_records),
    )


@app.get("/api/rlt/key-region/{key_region_id}", response_model=RLTKeyRegionReviewRecord)
def rlt_key_region_detail(key_region_id: str) -> RLTKeyRegionReviewRecord:
    record = next(
        (item for item in _key_region_review_records() if item.get("key_region_id") == key_region_id),
        None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown key_region_id: {key_region_id}")
    return RLTKeyRegionReviewRecord(**record)


@app.get("/api/rlt/preferences/next-pair", response_model=RLTPreferencePairResponse)
def rlt_preference_next_pair(
    batch: str = "all",
    reward: str = "all",
    pair_type: str = "auto",
) -> RLTPreferencePairResponse:
    with _connect_preference_db() as conn:
        stats = _preference_stats(conn)
        records = _preference_candidate_records(batch=batch, reward=reward)
        round_counts = _preference_round_counts(conn)
        selected_pair_type = _target_pair_type(round_counts, pair_type)
        left, right, remaining_unseen = _select_preference_pair(
            records,
            _seen_preference_pairs(conn),
            pair_type=selected_pair_type,
            degree=_preference_degree(conn),
        )
        if left is None and pair_type == "auto":
            for fallback_type in PREFERENCE_PAIR_TYPE_QUOTAS:
                if fallback_type == selected_pair_type:
                    continue
                left, right, remaining_unseen = _select_preference_pair(
                    records,
                    _seen_preference_pairs(conn),
                    pair_type=fallback_type,
                    degree=_preference_degree(conn),
                )
                if left is not None:
                    selected_pair_type = fallback_type
                    break
        round_labeled = sum(round_counts.values())
    return RLTPreferencePairResponse(
        left=None if left is None else RLTKeyRegionReviewRecord(**left),
        right=None if right is None else RLTKeyRegionReviewRecord(**right),
        stats=stats,
        remaining_unseen_pairs=remaining_unseen,
        pair_type=None if left is None else selected_pair_type,
        strategy="budgeted",
        round_budget=PREFERENCE_ROUND_BUDGET,
        round_labeled=round_labeled,
        round_remaining=max(0, PREFERENCE_ROUND_BUDGET - round_labeled),
    )


@app.post("/api/rlt/preferences", response_model=RLTPreferenceRecord)
def rlt_preference_record(request: RLTPreferenceRequest) -> RLTPreferenceRecord:
    if request.left_key_region_id == request.right_key_region_id:
        raise HTTPException(status_code=400, detail="Preference pair must use two different key regions")
    for key_region_id in (request.left_key_region_id, request.right_key_region_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key_region_id):
            raise HTTPException(status_code=400, detail=f"Invalid key_region_id: {key_region_id}")
    pair_key = _preference_pair_key(request.left_key_region_id, request.right_key_region_id)
    created_at = time.time()
    reason_tags = json.dumps(request.reason_tags, ensure_ascii=False)
    records_by_id = {str(record.get("key_region_id")): record for record in _key_region_review_records()}
    left_record = records_by_id.get(request.left_key_region_id)
    right_record = records_by_id.get(request.right_key_region_id)
    pair_type = _preference_pair_type(left_record or {"reward": 0}, right_record or {"reward": 0})
    with _connect_preference_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO preference_pairs (
                left_key_region_id, right_key_region_id, pair_key, preference,
                reason_tags, notes, source, created_at, pair_type, strategy, sample_round,
                left_reward, right_reward, left_batch, right_batch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.left_key_region_id,
                request.right_key_region_id,
                pair_key,
                request.preference,
                reason_tags,
                request.notes,
                request.source,
                created_at,
                pair_type,
                "budgeted",
                PREFERENCE_SAMPLE_ROUND,
                None if left_record is None else left_record.get("reward"),
                None if right_record is None else right_record.get("reward"),
                None if left_record is None else left_record.get("batch"),
                None if right_record is None else right_record.get("batch"),
            ),
        )
        row_id = int(cursor.lastrowid)
    return RLTPreferenceRecord(
        id=row_id,
        left_key_region_id=request.left_key_region_id,
        right_key_region_id=request.right_key_region_id,
        pair_key=pair_key,
        preference=request.preference,
        pair_type=pair_type,
        strategy="budgeted",
        sample_round=PREFERENCE_SAMPLE_ROUND,
        reason_tags=request.reason_tags,
        notes=request.notes,
        source=request.source,
        created_at=created_at,
    )


def _crop_output_shard_path(source_shard_path: Path, key_region_id: str) -> Path:
    source = source_shard_path.resolve()
    raw_root = (REPLAY_ROOT / "rlt_key_regions").resolve()
    clean_root = (REPLAY_ROOT / "rlt_key_regions_clean").resolve()
    try:
        relative = source.relative_to(raw_root)
    except ValueError:
        try:
            relative = source.relative_to(clean_root)
        except ValueError:
            relative = Path("manual") / f"key_region_{key_region_id}.npz"
    timestamp_ms = int(time.time() * 1000)
    return clean_root / relative.parent / f"{source.stem}.crop_{timestamp_ms}.npz"


@app.post("/api/rlt/key-region/{key_region_id}/rescore", response_model=RLTControlState)
def rlt_key_region_rescore(key_region_id: str, request: RLTKeyRegionRescoreRequest) -> RLTControlState:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key_region_id):
        raise HTTPException(status_code=400, detail=f"Invalid key_region_id: {key_region_id}")

    records = _key_region_review_records()
    record = next((item for item in records if item.get("key_region_id") == key_region_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown key_region_id: {key_region_id}")
    rollout_path = record.get("rollout_path")
    if not rollout_path:
        raise HTTPException(status_code=409, detail="Key region rollout manifest is missing")
    rollout_dir = (ROLLOUTS_ROOT / str(rollout_path)).resolve()
    if not rollout_dir.is_relative_to(ROLLOUTS_ROOT):
        raise HTTPException(status_code=400, detail="Invalid rollout path")
    shard_path = _host_path_for_container_path(record.get("shard_path"))
    if shard_path is None or not shard_path.exists():
        raise HTTPException(status_code=409, detail="Key region replay shard is missing")

    try:
        manifest = rescore_key_region_files(rollout_dir, shard_path, reward=request.reward)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return rlt_control.rescore_key_region_from_files(
        key_region_id=key_region_id,
        phase=str(manifest.get("phase") or record.get("phase") or "warmup"),
        reward=int(manifest.get("reward") or 0),
        shard_path=str(manifest.get("shard_path") or shard_path),
        num_replay_transitions=int(manifest.get("num_replay_transitions") or record.get("num_replay_transitions") or 0),
        source=request.source,
        reason=request.reason,
    )


@app.post("/api/rlt/key-region/{key_region_id}/crop", response_model=RLTKeyRegionCropResponse)
def rlt_key_region_crop(key_region_id: str, request: RLTKeyRegionCropRequest) -> RLTKeyRegionCropResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key_region_id):
        raise HTTPException(status_code=400, detail=f"Invalid key_region_id: {key_region_id}")
    if request.end_sec <= request.start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")

    records = _key_region_review_records()
    record = next((item for item in records if item.get("key_region_id") == key_region_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown key_region_id: {key_region_id}")
    rollout_path = record.get("rollout_path")
    if not rollout_path:
        raise HTTPException(status_code=409, detail="Key region rollout manifest is missing")
    rollout_dir = (ROLLOUTS_ROOT / str(rollout_path)).resolve()
    if not rollout_dir.is_relative_to(ROLLOUTS_ROOT):
        raise HTTPException(status_code=400, detail="Invalid rollout path")
    shard_path = _host_path_for_container_path(record.get("shard_path"))
    if shard_path is None or not shard_path.exists():
        missing_metadata = record.get("missing_rlt_metadata") or []
        replay_status = record.get("replay_status") or "unknown"
        detail = (
            f"Key region replay shard is missing; replay_status={replay_status}; "
            f"missing_rlt_metadata={missing_metadata}"
        )
        raise HTTPException(status_code=409, detail=detail)

    output_shard_path = _crop_output_shard_path(shard_path, key_region_id)
    try:
        manifest = crop_key_region_files(
            rollout_dir,
            shard_path,
            output_shard_path,
            start_sec=request.start_sec,
            end_sec=request.end_sec,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rlt_control.crop_key_region_from_files(
        key_region_id=key_region_id,
        phase=str(manifest.get("phase") or record.get("phase") or "warmup"),
        reward=int(manifest.get("reward") or record.get("reward") or 0),
        shard_path=str(manifest["shard_path"]),
        num_replay_transitions=int(manifest.get("num_replay_transitions") or 0),
        source=request.source,
        reason=request.reason,
    )
    return RLTKeyRegionCropResponse(
        key_region_id=key_region_id,
        status="committed",
        trainable=True,
        shard_path=str(manifest["shard_path"]),
        source_shard_path=str(manifest.get("source_shard_path") or shard_path),
        crop_start_sec=float(manifest["crop_start_sec"]),
        crop_end_sec=float(manifest["crop_end_sec"]),
        crop_start_sample=int(manifest["crop_start_sample"]),
        crop_end_sample=int(manifest["crop_end_sample"]),
        num_replay_transitions=int(manifest["num_replay_transitions"]),
        manifest_path=str((rollout_dir / "manifest.json").resolve()),
    )


def _delete_key_region_files(key_region_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key_region_id):
        raise HTTPException(status_code=400, detail=f"Invalid key_region_id: {key_region_id}")
    region_name = f"key_region_{key_region_id}"
    rollout_root = (ROLLOUTS_ROOT / "key_regions").resolve()
    if rollout_root.exists():
        for path in rollout_root.glob(f"**/{region_name}"):
            resolved = path.resolve()
            if resolved.is_dir() and resolved.is_relative_to(rollout_root):
                shutil.rmtree(resolved)
    replay_root = (REPLAY_ROOT / "rlt_key_regions").resolve()
    if replay_root.exists():
        for path in replay_root.glob(f"**/shards/{region_name}.npz*"):
            resolved = path.resolve()
            if resolved.is_file() and resolved.is_relative_to(replay_root):
                resolved.unlink()
    clean_replay_root = (REPLAY_ROOT / "rlt_key_regions_clean").resolve()
    if clean_replay_root.exists():
        for path in clean_replay_root.glob(f"**/shards/{region_name}*.npz*"):
            resolved = path.resolve()
            if resolved.is_file() and resolved.is_relative_to(clean_replay_root):
                resolved.unlink()


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


@app.post("/api/rlt/key-region/confirm", response_model=RLTControlState)
def rlt_key_region_confirm(request: RLTControlRequest | None = None) -> RLTControlState:
    request = request or RLTControlRequest()
    try:
        return rlt_control.confirm_key_region(source=request.source)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/rlt/key-region/discard", response_model=RLTControlState)
def rlt_key_region_discard(request: RLTDiscardRequest | None = None) -> RLTControlState:
    request = request or RLTDiscardRequest()
    try:
        return rlt_control.discard_key_region(source=request.source, reason=request.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/rlt/key-region/{key_region_id}/void", response_model=RLTControlState)
def rlt_key_region_void(key_region_id: str, request: RLTVoidRequest | None = None) -> RLTControlState:
    request = request or RLTVoidRequest()
    return rlt_control.void_segment(key_region_id, source=request.source, reason=request.reason)


@app.post("/api/rlt/key-regions/void", response_model=RLTControlState)
def rlt_key_regions_void(request: RLTBatchSegmentRequest) -> RLTControlState:
    return rlt_control.void_segments(request.key_region_ids, source=request.source, reason=request.reason)


@app.post("/api/rlt/key-regions/restore", response_model=RLTControlState)
def rlt_key_regions_restore(request: RLTBatchSegmentRequest) -> RLTControlState:
    return rlt_control.restore_segments(request.key_region_ids, source=request.source, reason=request.reason)


@app.post("/api/rlt/key-regions/delete", response_model=RLTControlState)
def rlt_key_regions_delete(request: RLTBatchSegmentRequest) -> RLTControlState:
    for key_region_id in request.key_region_ids:
        _delete_key_region_files(key_region_id)
    return rlt_control.delete_segments(request.key_region_ids, source=request.source, reason=request.reason)


@app.post("/api/rlt/config", response_model=RLTControlState)
def rlt_config(request: RLTConfigRequest) -> RLTControlState:
    return rlt_control.update_config(request)

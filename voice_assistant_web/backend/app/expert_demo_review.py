from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from functools import lru_cache

import pyarrow.parquet as pq

from .schemas import RLTExpertDemoCropResponse
from .schemas import RLTExpertDemoPage
from .schemas import RLTExpertDemoRecord

DEFAULT_EXCLUDED_DATASETS = {
    "2026-01-20-twist-one-bottle",
    "2026-05-04_direction-lerobot-with-rinse",
    "2026-05-01_turn_over-lerobot-with-rinse",
}
VIDEO_PREFIX = "/api/rlt/expert-demos/video"
EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return {}


def _episode_index_from_video(path: Path) -> int | None:
    match = re.fullmatch(r"file-(\d+)\.mp4", path.name)
    if not match:
        return None
    return int(match.group(1))


def _video_url(dataset_id: str, camera: str, file_index: int) -> str:
    return f"{VIDEO_PREFIX}/{dataset_id}?camera={camera}&file_index={file_index}"


@lru_cache(maxsize=2048)
def _video_frame_count(path: str) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    if result.returncode == 0 and text and text != "N/A":
        return int(float(text))
    return 0


def _dataset_episode_bounds(dataset_dir: Path) -> dict[int, dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path, columns=["episode_index", "index"])
        episodes = table["episode_index"].to_pylist()
        indices = table["index"].to_pylist()
        for episode_index, frame_index in zip(episodes, indices, strict=True):
            episode = int(episode_index)
            frame = int(frame_index)
            row = rows.setdefault(episode, {"start_frame": frame, "end_frame": frame, "frame_count": 0})
            row["start_frame"] = min(row["start_frame"], frame)
            row["end_frame"] = max(row["end_frame"], frame)
            row["frame_count"] += 1
    return rows


def _camera_video_files(dataset_dir: Path, camera: str) -> list[tuple[int, Path, int, int]]:
    videos_root = dataset_dir / "videos" / camera
    result = []
    cursor = 0
    for video_path in sorted(videos_root.glob("chunk-*/file-*.mp4")):
        file_index = _episode_index_from_video(video_path)
        if file_index is None:
            continue
        frame_count = _video_frame_count(str(video_path.resolve()))
        if frame_count <= 0:
            continue
        result.append((file_index, video_path, cursor, cursor + frame_count))
        cursor += frame_count
    return result


def _video_for_global_frame(dataset_dir: Path, camera: str, global_frame: int) -> tuple[int, Path, int] | None:
    for file_index, video_path, start_frame, end_frame in _camera_video_files(dataset_dir, camera):
        if start_frame <= global_frame < end_frame:
            return file_index, video_path, global_frame - start_frame
    return None


def _dataset_video_records(dataset_dir: Path) -> list[RLTExpertDemoRecord]:
    info = _load_info(dataset_dir)
    fps = info.get("fps")
    fps_float = None if fps is None else float(fps)
    episode_bounds = _dataset_episode_bounds(dataset_dir)
    records: list[RLTExpertDemoRecord] = []
    for episode_index, bounds in sorted(episode_bounds.items()):
        start_frame = int(bounds["start_frame"])
        frame_count = int(bounds["frame_count"])
        videos: list[tuple[str, int, Path, float]] = []
        missing_cameras: list[str] = []
        for camera in EXPECTED_CAMERAS:
            located = _video_for_global_frame(dataset_dir, camera, start_frame)
            if located is None:
                missing_cameras.append(camera.replace("observation.images.", ""))
                continue
            file_index, video_path, frame_offset = located
            start_sec = frame_offset / fps_float if fps_float and fps_float > 0 else 0.0
            videos.append((camera, file_index, video_path, start_sec))
        records.append(
            RLTExpertDemoRecord(
                episode_key=f"{dataset_dir.name}::{episode_index}",
                dataset_id=dataset_dir.name,
                episode_index=episode_index,
                fps=fps_float,
                num_frames=frame_count,
                duration_seconds=frame_count / fps_float if fps_float and fps_float > 0 else None,
                video_paths=[_video_url(dataset_dir.name, camera, file_index) for camera, file_index, _, _ in videos],
                local_video_paths=[str(path.resolve()) for _, _, path, _ in videos],
                video_start_secs=[start_sec for _, _, _, start_sec in videos],
                camera_count=len(videos),
                missing_cameras=missing_cameras,
                camera_complete=len(missing_cameras) == 0,
                source_dataset_path=str(dataset_dir.resolve()),
            )
        )
    return records


def list_expert_demos(
    dataset_root: str | Path,
    *,
    dataset: str = "all",
    search: str = "",
    camera_status: str = "complete",
    limit: int = 20,
    offset: int = 0,
) -> RLTExpertDemoPage:
    root = Path(dataset_root).expanduser().resolve()
    dataset_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name not in DEFAULT_EXCLUDED_DATASETS and (path / "videos").exists()
    ] if root.exists() else []
    dataset_ids = [path.name for path in dataset_dirs]
    if dataset and dataset != "all":
        dataset_dirs = [path for path in dataset_dirs if path.name == dataset]

    records: list[RLTExpertDemoRecord] = []
    for dataset_dir in dataset_dirs:
        records.extend(_dataset_video_records(dataset_dir))

    if camera_status == "complete":
        records = [record for record in records if record.camera_complete]
    elif camera_status == "incomplete":
        records = [record for record in records if not record.camera_complete]

    query = search.strip().lower()
    if query:
        records = [
            record
            for record in records
            if query in record.dataset_id.lower()
            or query in f"episode {record.episode_index}".lower()
            or query in str(record.episode_index)
        ]

    total = len(records)
    safe_offset = max(0, offset)
    safe_limit = max(1, min(200, limit))
    page_records = records[safe_offset : safe_offset + safe_limit]
    next_offset = safe_offset + safe_limit if safe_offset + safe_limit < total else None
    return RLTExpertDemoPage(
        items=page_records,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
        next_offset=next_offset,
        datasets=dataset_ids,
    )


def find_expert_demo_video(dataset_root: str | Path, dataset_id: str, camera: str, file_index: int) -> Path | None:
    dataset_dir = Path(dataset_root).expanduser().resolve() / dataset_id
    videos_root = dataset_dir / "videos" / camera
    matches = sorted(videos_root.glob(f"chunk-*/file-{file_index:03d}.mp4"))
    return matches[0].resolve() if matches else None


def crop_expert_demo(
    dataset_root: str | Path,
    crop_root: str | Path,
    *,
    dataset_id: str,
    episode_index: int,
    start_sec: float,
    end_sec: float,
    reward: int = 1,
) -> RLTExpertDemoCropResponse:
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    if reward not in (0, 1):
        raise ValueError("reward must be 0 or 1")
    page = list_expert_demos(dataset_root, dataset=dataset_id, search=f"episode {episode_index}", camera_status="any", limit=1000)
    record = next((item for item in page.items if item.episode_index == episode_index), None)
    if record is None:
        raise FileNotFoundError(f"Expert demo {dataset_id} episode {episode_index} was not found")

    output_dir = Path(crop_root).expanduser().resolve() / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_index = len(sorted(output_dir.glob(f"episode_{episode_index:06d}_crop_*.json")))
    metadata_path = output_dir / f"episode_{episode_index:06d}_crop_{crop_index:06d}.json"
    metadata = {
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "reward": int(reward),
        "label": "expert",
        "source_dataset_path": record.source_dataset_path,
        "local_video_paths": record.local_video_paths,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return RLTExpertDemoCropResponse(
        dataset_id=dataset_id,
        episode_index=episode_index,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        reward=int(reward),
        metadata_path=str(metadata_path),
    )

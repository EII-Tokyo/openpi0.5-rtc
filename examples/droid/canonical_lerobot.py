from __future__ import annotations

from dataclasses import dataclass
import copy
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

CANONICAL_FPS = 15
CANONICAL_ROBOT_TYPE = "Franka"
UNKNOWN_LABEL = "unknown"
SUBTASKS_PATH = "meta/subtasks.parquet"
CONVERSION_REPORT_PATH = "meta/conversion_report.json"
MIGRATION_REPORT_PATH = "meta/episode_migration.parquet"
CAMERA_VIDEO_KEYS = (
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
)
VIDEO_SYNC_TOLERANCE_S = 1.0 / CANONICAL_FPS

_MONTH_TO_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

_CANONICAL_FEATURES = {
    "is_first": {"dtype": "bool", "shape": (1,), "names": None},
    "is_last": {"dtype": "bool", "shape": (1,), "names": None},
    "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
    "language_instruction": {"dtype": "string", "shape": (1,), "names": None},
    "language_instruction_2": {"dtype": "string", "shape": (1,), "names": None},
    "language_instruction_3": {"dtype": "string", "shape": (1,), "names": None},
    "observation.state.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"axes": ["gripper"]},
    },
    "observation.state.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "observation.state.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)]},
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)] + ["gripper"]},
    },
    "observation.images.wrist_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.exterior_1_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.exterior_2_left": {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": ["height", "width", "channels"],
    },
    "action.gripper_position": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"axes": ["gripper"]},
    },
    "action.gripper_velocity": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"axes": ["gripper"]},
    },
    "action.cartesian_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "action.cartesian_velocity": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "action.joint_position": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)]},
    },
    "action.joint_velocity": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)]},
    },
    "action.original": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
    "action.source_joint_velocity_gripper": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)] + ["gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"axes": [f"joint_{idx}" for idx in range(7)] + ["gripper"]},
    },
    "discount": {"dtype": "float32", "shape": (1,), "names": None},
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "task_category": {"dtype": "string", "shape": (1,), "names": None},
    "building": {"dtype": "string", "shape": (1,), "names": None},
    "collector_id": {"dtype": "string", "shape": (1,), "names": None},
    "datetime": {"dtype": "string", "shape": (1,), "names": None},
    "camera_extrinsics.wrist_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "camera_extrinsics.exterior_1_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "camera_extrinsics.exterior_2_left": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "is_episode_successful": {"dtype": "bool", "shape": (1,), "names": None},
    "environment.conveyor_speed": {"dtype": "float32", "shape": (1,), "names": None},
    "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def build_canonical_features() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(_CANONICAL_FEATURES)


def ensure_vector(
    value: Any,
    size: int,
    *,
    dtype: np.dtype = np.float32,
    fill_value: float = 0.0,
) -> np.ndarray:
    if value is None:
        return np.full((size,), fill_value, dtype=dtype)
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size == size:
        return array.astype(dtype, copy=False)
    result = np.full((size,), fill_value, dtype=dtype)
    limit = min(size, array.size)
    if limit:
        result[:limit] = array[:limit].astype(dtype, copy=False)
    return result


def ensure_scalar_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.shape == (1,) and value.dtype == np.dtype(dtype):
        return value
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else 0
    return np.asarray([value], dtype=dtype)


def _normalize_bool_array(value: Any) -> np.ndarray:
    return ensure_scalar_array(bool(value), dtype=np.bool_)


def _normalize_float_array(value: Any) -> np.ndarray:
    return ensure_scalar_array(value, dtype=np.float32)


def _canonicalize_datetime_tokens(year: int, month: int, day: int, hour: int, minute: int, second: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}h-{minute:02d}m-{second:02d}s"


def normalize_datetime(value: str | None) -> str:
    if not value:
        return "unknown-datetime"

    text = value.strip()
    canonical = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})h-(\d{2})m-(\d{2})s", text)
    if canonical:
        return text

    dash = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[-T ](\d{2}):(\d{2}):(\d{2})", text)
    if dash:
        return _canonicalize_datetime_tokens(*(int(part) for part in dash.groups()))

    underscore = re.fullmatch(r"(\d{4})_(\d{2})_(\d{2})_(\d{2}):(\d{2}):(\d{2})", text)
    if underscore:
        return _canonicalize_datetime_tokens(*(int(part) for part in underscore.groups()))

    weekday = re.fullmatch(r"[A-Za-z]{3}_([A-Za-z]{3})_(\d{1,2})_(\d{2}):(\d{2}):(\d{2})_(\d{4})", text)
    if weekday:
        month_name, day, hour, minute, second, year = weekday.groups()
        month = _MONTH_TO_NUM[month_name]
        return _canonicalize_datetime_tokens(int(year), month, int(day), int(hour), int(minute), int(second))

    return text.replace(" ", "_")


def parse_uuid(uuid: str | None) -> tuple[str | None, str | None, str | None]:
    if not uuid:
        return None, None, None
    parts = uuid.split("+")
    if len(parts) != 3:
        return None, None, None
    building, collector_id, dt = parts
    return building, collector_id, normalize_datetime(dt)


def build_episode_uuid(building: str, collector_id: str, datetime_value: str) -> str:
    return f"{building}+{collector_id}+{datetime_value}"


def normalize_episode_identity(
    *,
    building: str | None = None,
    collector_id: str | None = None,
    datetime_value: str | None = None,
    date_value: str | None = None,
    uuid: str | None = None,
    fallback_prefix: str = "legacy",
) -> tuple[str, str, str]:
    uuid_building, uuid_collector, uuid_datetime = parse_uuid(uuid)
    final_building = building or uuid_building or "unknown-building"
    final_collector = collector_id or uuid_collector or "unknown-collector"
    final_datetime = normalize_datetime(datetime_value or date_value or uuid_datetime)
    if final_datetime == "unknown-datetime":
        final_datetime = f"{fallback_prefix}-unknown-datetime"
    return final_building, final_collector, final_datetime


def build_frame_meta(
    *,
    building: str,
    collector_id: str,
    datetime_value: str,
    is_episode_successful: bool,
    task_category: str | None = None,
    camera_extrinsics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    camera_extrinsics = camera_extrinsics or {}
    return {
        "task_category": task_category or building,
        "building": building,
        "collector_id": collector_id,
        "datetime": datetime_value,
        "camera_extrinsics.wrist_left": ensure_vector(
            camera_extrinsics.get("camera_extrinsics.wrist_left"), 6, dtype=np.float32
        ),
        "camera_extrinsics.exterior_1_left": ensure_vector(
            camera_extrinsics.get("camera_extrinsics.exterior_1_left"), 6, dtype=np.float32
        ),
        "camera_extrinsics.exterior_2_left": ensure_vector(
            camera_extrinsics.get("camera_extrinsics.exterior_2_left"), 6, dtype=np.float32
        ),
        "is_episode_successful": _normalize_bool_array(is_episode_successful),
    }


@dataclass
class EpisodeAnnotation:
    uuid: str | None
    task: str | None
    subtasks: Any = None
    conveyor_speed: float | None = None
    source_repo_id: str | None = None
    source_episode_index: int | None = None

    @property
    def identity(self) -> tuple[str | None, str | None, str | None]:
        return parse_uuid(self.uuid)


class AnnotationIndex:
    def __init__(self, records: list[EpisodeAnnotation]):
        self.records = records
        self.by_uuid = {record.uuid: record for record in records if record.uuid}
        self.by_source_episode = {
            (record.source_repo_id, record.source_episode_index): record
            for record in records
            if record.source_repo_id is not None and record.source_episode_index is not None
        }
        self._ordered_records = records
        self._cursor = 0

    @classmethod
    def load(cls, annotations_path: str | Path | None) -> "AnnotationIndex | None":
        if annotations_path is None:
            return None
        path = Path(annotations_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        records: list[EpisodeAnnotation] = []
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                for uuid, value in payload.items():
                    if isinstance(value, dict):
                        record_payload = {"uuid": uuid, **value}
                    else:
                        record_payload = {"uuid": uuid, "task": value}
                    records.append(_annotation_from_payload(record_payload))
            elif isinstance(payload, list):
                records.extend(_annotation_from_payload(item) for item in payload)
        else:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                records.append(_annotation_from_payload(json.loads(line)))

        return cls(records)

    def set_cursor(self, offset: int) -> None:
        self._cursor = max(offset, 0)

    def match_raw_episode(self, episode_dir_name: str, task: str | None = None) -> EpisodeAnnotation | None:
        normalized_dt = normalize_datetime(episode_dir_name)
        matches = [record for record in self.records if record.identity[2] == normalized_dt]
        if not matches:
            return None
        if task is None:
            return matches[0]
        for record in matches:
            if record.task == task:
                return record
        return matches[0]

    def match_legacy_episode(
        self,
        *,
        source_repo_id: str,
        source_episode_index: int,
        task: str | None = None,
    ) -> EpisodeAnnotation | None:
        direct = self.by_source_episode.get((source_repo_id, source_episode_index))
        if direct is not None:
            return direct

        if self._cursor >= len(self._ordered_records):
            return None

        candidate = self._ordered_records[self._cursor]
        if task is not None and candidate.task is not None and candidate.task != task:
            logging.warning(
                "Skipping ordered annotation candidate at cursor %s due to task mismatch: source='%s' annotation='%s'",
                self._cursor,
                task,
                candidate.task,
            )
            return None

        self._cursor += 1
        return candidate


def _annotation_from_payload(payload: dict[str, Any]) -> EpisodeAnnotation:
    prompts = payload.get("prompts") or {}
    task = (
        payload.get("task")
        or payload.get("language_instruction1")
        or payload.get("language_instruction")
        or prompts.get("default")
        or prompts.get("task")
    )
    source_episode_index = payload.get("source_episode_index")
    if source_episode_index is not None:
        source_episode_index = int(source_episode_index)
    return EpisodeAnnotation(
        uuid=payload.get("uuid"),
        task=task,
        subtasks=payload.get("subtasks") or payload.get("subtask"),
        conveyor_speed=_coerce_conveyor_speed(
            payload.get("conveyor_speed")
            or payload.get("speed")
            or prompts.get("speed")
        ),
        source_repo_id=payload.get("source_repo_id"),
        source_episode_index=source_episode_index,
    )


def _coerce_conveyor_speed(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def resolve_subtask_for_frame(
    annotation: EpisodeAnnotation | None,
    *,
    frame_index: int,
    episode_length: int,
) -> str:
    if annotation is None or annotation.subtasks is None:
        return UNKNOWN_LABEL

    spec = annotation.subtasks
    if isinstance(spec, str):
        return spec or UNKNOWN_LABEL

    if isinstance(spec, dict):
        if "default" in spec and isinstance(spec["default"], str):
            return spec["default"] or UNKNOWN_LABEL
        if "segments" in spec:
            return _resolve_segmented_label(spec["segments"], frame_index, episode_length)

    if isinstance(spec, list):
        if not spec:
            return UNKNOWN_LABEL
        if all(isinstance(item, str) for item in spec):
            if len(spec) == episode_length:
                return spec[frame_index] or UNKNOWN_LABEL
            return spec[min(frame_index, len(spec) - 1)] or UNKNOWN_LABEL
        if all(isinstance(item, dict) for item in spec):
            return _resolve_segmented_label(spec, frame_index, episode_length)

    return UNKNOWN_LABEL


def _resolve_segmented_label(segments: list[dict[str, Any]], frame_index: int, episode_length: int) -> str:
    for segment in segments:
        label = segment.get("subtask") or segment.get("label") or segment.get("task")
        if not label:
            continue
        start_frame = segment.get("start_frame")
        end_frame = segment.get("end_frame")
        if start_frame is None and "start" in segment:
            start_frame = int(float(segment["start"]) * episode_length)
        if end_frame is None and "end" in segment:
            end_frame = int(float(segment["end"]) * episode_length)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = episode_length
        if int(start_frame) <= frame_index < int(end_frame):
            return str(label)
    return UNKNOWN_LABEL


def compute_env_stats_for_frame(
    frame: dict[str, Any],
    *,
    frame_index: int,
    episode_length: int,
    annotation: EpisodeAnnotation | None = None,
) -> dict[str, np.ndarray]:
    del frame, frame_index, episode_length
    conveyor_speed = np.nan if annotation is None or annotation.conveyor_speed is None else annotation.conveyor_speed
    return {
        "environment.conveyor_speed": _normalize_float_array(conveyor_speed),
    }


def finalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for key, value in frame.items():
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            finalized[key] = value.astype(np.float32)
        else:
            finalized[key] = value
    return finalized


def validate_camera_frame(frame: dict[str, Any], camera_key: str) -> None:
    image = frame.get(camera_key)
    if image is None:
        raise ValueError(f"Missing camera frame for '{camera_key}'.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Camera frame '{camera_key}' must be a numpy array, got {type(image)}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Camera frame '{camera_key}' must have shape (H, W, 3), got {image.shape}.")
    if image.size == 0:
        raise ValueError(f"Camera frame '{camera_key}' is empty.")


def validate_episode_frames_sync(frames: list[dict[str, Any]]) -> None:
    if not frames:
        raise ValueError("Attempted to write an empty episode.")
    for frame_index, frame in enumerate(frames):
        for camera_key in CAMERA_VIDEO_KEYS:
            try:
                validate_camera_frame(frame, camera_key)
            except ValueError as exc:
                raise ValueError(f"Episode camera validation failed at frame {frame_index}: {exc}") from exc


def validate_episode_video_segments(dataset: LeRobotDataset, episode_index: int, expected_length: int) -> None:
    expected_duration = expected_length / dataset.fps
    episode_meta = None
    latest_episode = getattr(dataset.meta, "latest_episode", None)
    if latest_episode is not None and latest_episode.get("episode_index", [None])[0] == episode_index:
        episode_meta = {
            key: value[0] if isinstance(value, list) else value
            for key, value in latest_episode.items()
        }
    else:
        if dataset.meta.episodes is None:
            dataset.meta.load_metadata()
        episode_meta = dataset.meta.episodes[episode_index]

    for camera_key in CAMERA_VIDEO_KEYS:
        from_timestamp = episode_meta[f"videos/{camera_key}/from_timestamp"]
        to_timestamp = episode_meta[f"videos/{camera_key}/to_timestamp"]
        duration = to_timestamp - from_timestamp
        if abs(duration - expected_duration) > VIDEO_SYNC_TOLERANCE_S:
            raise ValueError(
                f"Video segment duration mismatch for '{camera_key}' in episode {episode_index}: "
                f"expected {expected_duration:.6f}s, got {duration:.6f}s."
            )


def validate_raw_camera_timestamps(
    timestamp_dict: dict[str, Any] | None,
    camera_ids: list[str],
    *,
    tolerance_s: float = VIDEO_SYNC_TOLERANCE_S,
) -> None:
    if not timestamp_dict:
        return

    timestamps: list[float] = []
    for camera_id in camera_ids:
        for key in (f"{camera_id}_frame_received", camera_id):
            value = timestamp_dict.get(key)
            if value is not None:
                timestamps.append(float(value))
                break

    if len(timestamps) < 2:
        return

    if max(timestamps) - min(timestamps) > tolerance_s:
        raise ValueError(
            f"Raw camera timestamps are out of sync: {timestamps}. Allowed spread is {tolerance_s:.6f}s."
        )


@contextlib.contextmanager
def suppress_stderr(enabled: bool = True):
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class SubtaskRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / SUBTASKS_PATH
        self.index_to_label: list[str] = [UNKNOWN_LABEL]
        self.label_to_index = {UNKNOWN_LABEL: 0}
        if self.path.exists():
            df = pd.read_parquet(self.path)
            labels = list(df.index.astype(str))
            indices = list(df["subtask_index"].astype(int))
            ordered = [label for _, label in sorted(zip(indices, labels, strict=True))]
            self.index_to_label = ordered
            self.label_to_index = {label: idx for idx, label in enumerate(ordered)}

    def get_or_add(self, label: str | None) -> int:
        normalized = (label or UNKNOWN_LABEL).strip() or UNKNOWN_LABEL
        if normalized not in self.label_to_index:
            self.label_to_index[normalized] = len(self.index_to_label)
            self.index_to_label.append(normalized)
        return self.label_to_index[normalized]

    def write(self) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"subtask_index": range(len(self.index_to_label))}, index=self.index_to_label)
        df.index.name = "subtask"
        df.to_parquet(path)


def load_subtasks(root: str | Path) -> pd.DataFrame | None:
    path = Path(root) / SUBTASKS_PATH
    if not path.exists():
        return None
    return pd.read_parquet(path)


def create_canonical_dataset(
    *,
    repo_id: str,
    root: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    image_writer_processes: int = 0,
    image_writer_threads: int = 8,
    batch_encoding_size: int = 1,
) -> LeRobotDataset:
    dataset_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    if resume and (dataset_root / "meta/info.json").exists():
        return LeRobotDataset(
            repo_id,
            root=dataset_root,
            download_videos=False,
            batch_encoding_size=batch_encoding_size,
        )

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type=CANONICAL_ROBOT_TYPE,
        fps=CANONICAL_FPS,
        features=build_canonical_features(),
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
    )


class CanonicalDatasetWriter:
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path | None = None,
        resume: bool = False,
        overwrite: bool = False,
        image_writer_processes: int = 0,
        image_writer_threads: int = 8,
        batch_encoding_size: int = 1,
        suppress_encoder_output: bool = True,
    ):
        self.dataset = create_canonical_dataset(
            repo_id=repo_id,
            root=root,
            resume=resume,
            overwrite=overwrite,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
        )
        self.root = self.dataset.root
        self.subtasks = SubtaskRegistry(self.root)
        self.suppress_encoder_output = suppress_encoder_output
        self.report = {
            "total_episodes_converted": 0,
            "total_frames_converted": 0,
            "frames_unknown_subtask": 0,
            "frames_missing_env_stats": 0,
        }
        self.migration_rows: list[dict[str, Any]] = []

    def get_subtask_index(self, label: str | None) -> np.ndarray:
        normalized = (label or UNKNOWN_LABEL).strip() or UNKNOWN_LABEL
        if normalized == UNKNOWN_LABEL:
            self.report["frames_unknown_subtask"] += 1
        return ensure_scalar_array(self.subtasks.get_or_add(normalized), dtype=np.int64)

    def add_episode(
        self,
        frames: list[dict[str, Any]],
        *,
        source_repo_id: str | None = None,
        source_episode_index: int | None = None,
        source_path: str | None = None,
    ) -> int:
        validate_episode_frames_sync(frames)
        destination_episode_index = self.dataset.meta.total_episodes
        for frame in frames:
            self.dataset.add_frame(finalize_frame(frame))
        with suppress_stderr(self.suppress_encoder_output):
            self.dataset.save_episode()
        validate_episode_video_segments(self.dataset, destination_episode_index, len(frames))

        self.report["total_episodes_converted"] += 1
        self.report["total_frames_converted"] += len(frames)

        if frames:
            first = frames[0]
            self.migration_rows.append(
                {
                    "source_repo_id": source_repo_id,
                    "source_episode_index": source_episode_index,
                    "source_path": source_path,
                    "destination_episode_index": destination_episode_index,
                    "building": first.get("building"),
                    "collector_id": first.get("collector_id"),
                    "datetime": first.get("datetime"),
                }
            )

        return destination_episode_index

    def finalize(self) -> None:
        self.dataset.finalize()
        self.subtasks.write()
        self.report["task_vocab_size"] = 0 if self.dataset.meta.tasks is None else len(self.dataset.meta.tasks)
        self.report["subtask_vocab_size"] = len(self.subtasks.index_to_label)
        report_path = self.root / CONVERSION_REPORT_PATH
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.report, indent=2, sort_keys=True))
        if self.migration_rows:
            pd.DataFrame(self.migration_rows).to_parquet(self.root / MIGRATION_REPORT_PATH, index=False)


class CanonicalLeRobotDatasetView:
    def __init__(self, *args, **kwargs):
        self.dataset = LeRobotDataset(*args, **kwargs)
        subtasks = load_subtasks(self.dataset.root)
        self.subtasks = None if subtasks is None else subtasks

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        if self.subtasks is not None and "subtask_index" in item:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.subtasks.iloc[subtask_idx].name
        return item


def validate_canonical_dataset(root: str | Path, repo_id: str) -> None:
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    features = dataset.meta.features
    if "datetime" not in features:
        raise ValueError("Canonical dataset is missing the 'datetime' feature.")
    if "date" in features:
        raise ValueError("Canonical dataset should not contain the legacy 'date' feature.")
    expected_video_keys = {
        "observation.images.wrist_left",
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
    }
    if set(dataset.meta.video_keys) != expected_video_keys:
        raise ValueError(f"Unexpected video keys: {dataset.meta.video_keys}")
    if (Path(root) / SUBTASKS_PATH).exists():
        subtasks = load_subtasks(root)
        if subtasks is None or "subtask_index" not in subtasks.columns:
            raise ValueError("Invalid subtask metadata file.")

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any
import warnings

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tyro

from examples.droid.canonical_lerobot import AnnotationIndex
from examples.droid.canonical_lerobot import CanonicalDatasetWriter
from examples.droid.canonical_lerobot import build_frame_meta
from examples.droid.canonical_lerobot import compute_env_stats_for_frame
from examples.droid.canonical_lerobot import ensure_scalar_array
from examples.droid.canonical_lerobot import ensure_vector
from examples.droid.canonical_lerobot import format_duration
from examples.droid.canonical_lerobot import normalize_episode_identity
from examples.droid.canonical_lerobot import resolve_subtask_for_frame
from examples.droid.canonical_lerobot import validate_canonical_dataset
from examples.droid.droid_metadata import resolve_legacy_episode_metadata
from examples.droid.droid_mongo import ReadOnlyDroidMongo


def _torch_image_to_hwc_uint8(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Encountered a missing legacy camera frame.")
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.dtype == object:
        raise ValueError("Encountered an invalid legacy camera frame payload.")
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return array


def _identity_from_source_frame(
    source_frame: dict[str, Any],
    *,
    annotation,
    fallback_prefix: str,
) -> tuple[str, str, str]:
    return normalize_episode_identity(
        building=source_frame.get("building"),
        collector_id=source_frame.get("collector_id"),
        datetime_value=source_frame.get("datetime"),
        date_value=source_frame.get("date"),
        uuid=annotation.uuid if annotation else None,
        fallback_prefix=fallback_prefix,
    )


def _source_task_text(source_frame: dict[str, Any]) -> str:
    task = source_frame.get("task")
    if isinstance(task, str) and task:
        return task
    language_instruction = source_frame.get("language_instruction")
    if isinstance(language_instruction, str) and language_instruction:
        return language_instruction
    return "unknown task"


def _normalize_raw_episode_datetime(episode_dir: str) -> str | None:
    parts = episode_dir.split("_")
    if len(parts) < 5:
        return None
    month = parts[1][:3]
    day = next((part for part in parts[2:] if part.isdigit()), None)
    time_part = parts[-2]
    year = parts[-1]
    month_map = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    if month not in month_map or day is None:
        return None
    return f"{year}-{month_map[month]}-{int(day):02d}-{time_part.replace(':', 'h-', 1).replace(':', 'm-', 1)}s"


def _build_raw_order_annotations(
    raw_root: Path | None,
    annotations: AnnotationIndex | None,
) -> list | None:
    if raw_root is None or annotations is None or not raw_root.exists():
        return None

    annotation_by_datetime = {record.identity[2]: record for record in annotations.records if record.identity[2] is not None}
    raw_paths = list(raw_root.glob("**/trajectory.h5"))
    ordered = []
    for path in raw_paths:
        dt = _normalize_raw_episode_datetime(path.parent.name)
        ordered.append(annotation_by_datetime.get(dt))
    return ordered


def _map_legacy_frame(
    source_frame: dict[str, Any],
    *,
    frame_index: int,
    episode_length: int,
    task: str,
    building: str,
    collector_id: str,
    datetime_value: str,
    is_episode_successful: bool,
    annotation,
) -> tuple[dict[str, Any], str]:
    wrist = _torch_image_to_hwc_uint8(
        source_frame.get("observation.images.wrist_left", source_frame.get("wrist_image_left"))
    )
    exterior_1 = _torch_image_to_hwc_uint8(
        source_frame.get("observation.images.exterior_1_left", source_frame.get("exterior_image_1_left"))
    )
    exterior_2 = _torch_image_to_hwc_uint8(
        source_frame.get("observation.images.exterior_2_left", source_frame.get("exterior_image_2_left"))
    )

    joint_position = ensure_vector(
        source_frame.get("observation.state.joint_position", source_frame.get("joint_position")),
        7,
    )
    gripper_position = ensure_vector(
        source_frame.get("observation.state.gripper_position", source_frame.get("gripper_position")),
        1,
    )
    cartesian_position = ensure_vector(source_frame.get("observation.state.cartesian_position"), 6)

    source_action = ensure_vector(source_frame.get("action.source_joint_velocity_gripper", source_frame.get("actions")), 8)
    action_joint_velocity = ensure_vector(source_frame.get("action.joint_velocity"), 7)
    if not np.any(action_joint_velocity):
        action_joint_velocity = source_action[:7]
    action_gripper_position = ensure_vector(source_frame.get("action.gripper_position"), 1)
    if not np.any(action_gripper_position):
        action_gripper_position = source_action[7:8]
    action_joint_position = ensure_vector(source_frame.get("action.joint_position"), 7)
    if not np.any(action_joint_position):
        action_joint_position = joint_position
    action_cartesian_position = ensure_vector(source_frame.get("action.cartesian_position"), 6)
    action_cartesian_velocity = ensure_vector(source_frame.get("action.cartesian_velocity"), 6)
    action_gripper_velocity = ensure_vector(source_frame.get("action.gripper_velocity"), 1)
    action_original = ensure_vector(source_frame.get("action.original"), 7)

    camera_extrinsics = {
        "camera_extrinsics.wrist_left": source_frame.get("camera_extrinsics.wrist_left"),
        "camera_extrinsics.exterior_1_left": source_frame.get("camera_extrinsics.exterior_1_left"),
        "camera_extrinsics.exterior_2_left": source_frame.get("camera_extrinsics.exterior_2_left"),
    }
    frame = {
        "is_first": ensure_scalar_array(bool(source_frame.get("is_first", frame_index == 0)), dtype=np.bool_),
        "is_last": ensure_scalar_array(bool(source_frame.get("is_last", frame_index == episode_length - 1)), dtype=np.bool_),
        "is_terminal": ensure_scalar_array(
            bool(source_frame.get("is_terminal", frame_index == episode_length - 1)), dtype=np.bool_
        ),
        "language_instruction": source_frame.get("language_instruction", task),
        "language_instruction_2": source_frame.get("language_instruction_2", ""),
        "language_instruction_3": source_frame.get("language_instruction_3", ""),
        "subtask_index": ensure_scalar_array(0, dtype=np.int64),
        "observation.state.gripper_position": gripper_position.astype(np.float32),
        "observation.state.cartesian_position": cartesian_position.astype(np.float32),
        "observation.state.joint_position": joint_position.astype(np.float32),
        "observation.state": np.concatenate([joint_position, gripper_position]).astype(np.float32),
        "observation.images.wrist_left": wrist,
        "observation.images.exterior_1_left": exterior_1,
        "observation.images.exterior_2_left": exterior_2,
        "action.gripper_position": action_gripper_position.astype(np.float32),
        "action.gripper_velocity": action_gripper_velocity.astype(np.float32),
        "action.cartesian_position": action_cartesian_position.astype(np.float32),
        "action.cartesian_velocity": action_cartesian_velocity.astype(np.float32),
        "action.joint_position": action_joint_position.astype(np.float32),
        "action.joint_velocity": action_joint_velocity.astype(np.float32),
        "action.original": action_original.astype(np.float32),
        "action.source_joint_velocity_gripper": source_action.astype(np.float32),
        "action": np.concatenate([action_joint_position, action_gripper_position]).astype(np.float32),
        "discount": ensure_scalar_array(source_frame.get("discount", 0.0), dtype=np.float32),
        "reward": ensure_scalar_array(source_frame.get("reward", 0.0), dtype=np.float32),
        "task": task,
    }
    frame.update(
        build_frame_meta(
            building=building,
            collector_id=collector_id,
            datetime_value=datetime_value,
            is_episode_successful=is_episode_successful,
            task_category=source_frame.get("task_category", building),
            camera_extrinsics=camera_extrinsics,
        )
    )
    subtask_label = resolve_subtask_for_frame(annotation, frame_index=frame_index, episode_length=episode_length)
    frame.update(compute_env_stats_for_frame(frame, frame_index=frame_index, episode_length=episode_length, annotation=annotation))
    return frame, subtask_label


@dataclass
class LegacyConversionConfig:
    source_repo_ids: list[str]
    destination_repo_id: str
    destination_root: Path | None = None
    source_roots: list[Path] | None = None
    raw_data_roots: list[Path] | None = None
    annotations_path: Path | None = None
    annotation_offset: int = 0
    overwrite: bool = False
    resume: bool = False
    push_to_hub: bool = False
    start_episode: int = 0
    end_episode: int | None = None
    batch_encoding_size: int = 1
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    suppress_encoder_output: bool = True
    mongo_url: str | None = None
    mongo_db_name: str = "eii_data_system"
    mongo_project_path_filters: list[str] | None = None
    mongo_project_date_filters: list[str] | None = None
    allow_task_length_fallback: bool = False


def _resolve_source_root(repo_id: str, source_roots: list[Path] | None, index: int) -> Path | None:
    if source_roots is None:
        return HF_LEROBOT_HOME / repo_id
    if index >= len(source_roots):
        return HF_LEROBOT_HOME / repo_id
    return source_roots[index]


def _resolve_raw_root(raw_data_roots: list[Path] | None, index: int) -> Path | None:
    if raw_data_roots is None or index >= len(raw_data_roots):
        return None
    return raw_data_roots[index]


def main(config: LegacyConversionConfig) -> None:
    warnings.filterwarnings(
        "ignore",
        message="Converting input from bool to <class 'numpy.uint8'> for compatibility.",
        category=RuntimeWarning,
    )
    annotations = AnnotationIndex.load(config.annotations_path)
    if annotations is not None:
        annotations.set_cursor(config.annotation_offset)
    metadata_index = None
    if config.mongo_url is not None:
        mongo = ReadOnlyDroidMongo.connect(config.mongo_url, db_name=config.mongo_db_name)
        metadata_index = mongo.build_export_index(
            path_substrings=config.mongo_project_path_filters,
            date_substrings=config.mongo_project_date_filters,
        )

    writer = CanonicalDatasetWriter(
        repo_id=config.destination_repo_id,
        root=config.destination_root,
        overwrite=config.overwrite,
        resume=config.resume,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        batch_encoding_size=config.batch_encoding_size,
        suppress_encoder_output=config.suppress_encoder_output,
    )
    writer.report.setdefault("episodes_unmatched_metadata", 0)
    writer.report.setdefault("episodes_ambiguous_metadata", 0)
    writer.report.setdefault("episodes_missing_speed", 0)

    for repo_list_index, source_repo_id in enumerate(config.source_repo_ids):
        source_root = _resolve_source_root(source_repo_id, config.source_roots, repo_list_index)
        raw_root = _resolve_raw_root(config.raw_data_roots, repo_list_index)
        raw_order_annotations = _build_raw_order_annotations(raw_root, annotations)
        source_dataset = LeRobotDataset(source_repo_id, root=source_root, download_videos=False)
        total_episodes = source_dataset.meta.total_episodes
        end_episode = total_episodes if config.end_episode is None else min(config.end_episode, total_episodes)
        episodes_to_convert = max(end_episode - config.start_episode, 0)
        logging.info("Migrating %s episodes from %s", episodes_to_convert, source_repo_id)
        if raw_order_annotations is not None and len(raw_order_annotations) != total_episodes:
            logging.warning(
                "Raw-order annotation count (%s) does not match legacy episode count (%s) for %s; "
                "falling back to legacy matching for out-of-range episodes.",
                len(raw_order_annotations),
                total_episodes,
                source_repo_id,
            )

        started_at = time.perf_counter()
        visited_count = 0
        converted_count = 0
        skipped_count = 0
        for episode_index in range(config.start_episode, end_episode):
            visited_count += 1
            episode_meta = source_dataset.meta.episodes[episode_index]
            start = episode_meta["dataset_from_index"]
            end = episode_meta["dataset_to_index"]
            source_frames = [source_dataset[idx] for idx in range(start, end)]
            if not source_frames:
                skipped_count += 1
                continue

            annotation = None
            if raw_order_annotations is not None and episode_index < len(raw_order_annotations):
                annotation = raw_order_annotations[episode_index]
            elif annotations is not None:
                annotation = annotations.match_legacy_episode(
                    source_repo_id=source_repo_id,
                    source_episode_index=episode_index,
                    task=_source_task_text(source_frames[0]),
                )
            source_task = _source_task_text(source_frames[0])
            if annotation is not None and annotation.task and annotation.task != source_task:
                logging.warning(
                    "Prompt mismatch for %s episode %s: source='%s' annotation='%s'",
                    source_repo_id,
                    episode_index,
                    source_task,
                    annotation.task,
                )
            resolved_metadata = resolve_legacy_episode_metadata(
                index=metadata_index,
                source_frame=source_frames[0],
                source_repo_id=source_repo_id,
                source_episode_index=episode_index,
                source_task=source_task,
                source_length=len(source_frames),
                annotation=annotation,
                allow_task_length_fallback=config.allow_task_length_fallback,
            )
            if resolved_metadata.match_status.startswith("ambiguous"):
                skipped_count += 1
                writer.report["episodes_ambiguous_metadata"] += 1
                writer.migration_rows.append(
                    {
                        "source_repo_id": source_repo_id,
                        "source_episode_index": episode_index,
                        "source_path": str(source_root),
                        "destination_episode_index": None,
                        "building": resolved_metadata.building,
                        "collector_id": resolved_metadata.collector_id,
                        "datetime": resolved_metadata.datetime_value,
                        "metadata_match_status": resolved_metadata.match_status,
                        "mongo_episode_id": None,
                        "match_candidates": list(resolved_metadata.match_candidates),
                    }
                )
                continue
            if resolved_metadata.match_status == "unmatched":
                writer.report["episodes_unmatched_metadata"] += 1
            if resolved_metadata.conveyor_speed is None:
                writer.report["episodes_missing_speed"] += 1
            task = resolved_metadata.task or source_task
            building, collector_id, datetime_value = (
                resolved_metadata.building,
                resolved_metadata.collector_id,
                resolved_metadata.datetime_value,
            )
            is_episode_successful = bool(
                source_frames[0].get("is_episode_successful", torch.tensor([False])).item()
                if isinstance(source_frames[0].get("is_episode_successful"), torch.Tensor)
                else source_frames[0].get("is_episode_successful", False)
            )

            frames: list[dict[str, Any]] = []
            for frame_index, source_frame in enumerate(source_frames):
                frame, subtask_label = _map_legacy_frame(
                    source_frame,
                    frame_index=frame_index,
                    episode_length=len(source_frames),
                    task=task,
                    building=building,
                    collector_id=collector_id,
                    datetime_value=datetime_value,
                    is_episode_successful=is_episode_successful,
                    annotation=resolved_metadata,
                )
                frame["subtask_index"] = writer.get_subtask_index(subtask_label)
                if np.isnan(frame["environment.conveyor_speed"][0]):
                    writer.report["frames_missing_env_stats"] += 1
                frames.append(frame)

            writer.add_episode(
                frames,
                source_repo_id=source_repo_id,
                source_episode_index=episode_index,
                source_path=str(source_root),
            )
            converted_count += 1
            writer.migration_rows[-1].update(
                {
                    "metadata_match_status": resolved_metadata.match_status,
                    "mongo_episode_id": resolved_metadata.mongo_episode_id,
                    "mongo_project_id": resolved_metadata.mongo_project_id,
                    "mongo_source_folder_name": resolved_metadata.source_folder_name,
                }
            )
            elapsed = time.perf_counter() - started_at
            eta_seconds = (elapsed / visited_count) * (episodes_to_convert - visited_count) if visited_count else 0.0
            logging.info(
                "[%s] visited %s/%s episodes (%.1f%%), converted=%s, skipped=%s, ETA %s",
                source_repo_id,
                visited_count,
                episodes_to_convert,
                (visited_count / episodes_to_convert * 100.0) if episodes_to_convert else 100.0,
                converted_count,
                skipped_count,
                format_duration(eta_seconds),
            )

    writer.finalize()
    validate_canonical_dataset(writer.root, config.destination_repo_id)

    if config.push_to_hub:
        writer.dataset.push_to_hub(tags=["droid", "canonical"], private=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(tyro.cli(LegacyConversionConfig))

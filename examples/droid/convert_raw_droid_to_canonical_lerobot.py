from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image
import tyro

from examples.droid.canonical_lerobot import AnnotationIndex
from examples.droid.canonical_lerobot import CanonicalDatasetWriter
from examples.droid.canonical_lerobot import build_frame_meta
from examples.droid.canonical_lerobot import compute_env_stats_for_frame
from examples.droid.canonical_lerobot import ensure_scalar_array
from examples.droid.canonical_lerobot import ensure_vector
from examples.droid.canonical_lerobot import format_duration
from examples.droid.canonical_lerobot import resolve_subtask_for_frame
from examples.droid.canonical_lerobot import validate_canonical_dataset
from examples.droid.canonical_lerobot import validate_raw_camera_timestamps
from examples.droid.droid_metadata import resolve_raw_episode_metadata
from examples.droid.droid_mongo import ReadOnlyDroidMongo


def resize_rgb_frame(image: np.ndarray, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize(size, resample=Image.BICUBIC))


def _extract_camera_images(step: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_type_dict = step["observation"]["camera_type"]
    wrist_ids = [key for key, value in camera_type_dict.items() if value == 0]
    exterior_ids = [key for key, value in camera_type_dict.items() if value != 0]
    if len(wrist_ids) != 1 or len(exterior_ids) < 2:
        raise ValueError(f"Unexpected camera layout: wrist={wrist_ids}, exterior={exterior_ids}")
    validate_raw_camera_timestamps(
        step.get("observation", {}).get("timestamp", {}).get("cameras"),
        [wrist_ids[0], exterior_ids[0], exterior_ids[1]],
    )

    image_map = step["observation"]["image"]
    wrist = resize_rgb_frame(image_map[wrist_ids[0]][..., ::-1])
    exterior_1 = resize_rgb_frame(image_map[exterior_ids[0]][..., ::-1])
    exterior_2 = resize_rgb_frame(image_map[exterior_ids[1]][..., ::-1])
    return wrist, exterior_1, exterior_2


def _find_camera_extrinsics(step: dict) -> dict[str, np.ndarray]:
    observation = step.get("observation", {})
    metadata = step.get("metadata", {})
    candidates = [
        observation.get("camera_extrinsics"),
        observation.get("extrinsics"),
        metadata.get("camera_extrinsics"),
        metadata.get("extrinsics"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return {
                "camera_extrinsics.wrist_left": ensure_vector(
                    candidate.get("wrist_left") or candidate.get("hand_camera"), 6
                ),
                "camera_extrinsics.exterior_1_left": ensure_vector(
                    candidate.get("exterior_1_left") or candidate.get("varied_camera_1"), 6
                ),
                "camera_extrinsics.exterior_2_left": ensure_vector(
                    candidate.get("exterior_2_left") or candidate.get("varied_camera_2"), 6
                ),
            }
    return {}


def _build_raw_frame(
    step: dict,
    *,
    frame_index: int,
    episode_length: int,
    task: str,
    building: str,
    collector_id: str,
    datetime_value: str,
    is_episode_successful: bool,
    annotation,
) -> dict:
    wrist_img, exterior_1_img, exterior_2_img = _extract_camera_images(step)
    robot_state = step.get("observation", {}).get("robot_state", {})
    action = step.get("action", {})

    joint_position = ensure_vector(robot_state.get("joint_positions"), 7)
    gripper_position = ensure_vector(robot_state.get("gripper_position"), 1)
    cartesian_source = robot_state.get("cartesian_position")
    if cartesian_source is None:
        cartesian_source = robot_state.get("cartesian_pose")
    cartesian_position = ensure_vector(cartesian_source, 6)

    action_joint_position = ensure_vector(action.get("joint_position"), 7, fill_value=joint_position[0] if joint_position.size else 0.0)
    if np.allclose(action_joint_position, action_joint_position[0]):
        action_joint_position = joint_position.copy()
    action_joint_velocity = ensure_vector(action.get("joint_velocity"), 7)
    action_gripper_position = ensure_vector(action.get("gripper_position"), 1)
    action_gripper_velocity = ensure_vector(action.get("gripper_velocity"), 1)
    action_cartesian_position = ensure_vector(action.get("cartesian_position"), 6)
    action_cartesian_velocity = ensure_vector(action.get("cartesian_velocity"), 6)

    frame = {
        "is_first": ensure_scalar_array(frame_index == 0, dtype=np.bool_),
        "is_last": ensure_scalar_array(frame_index == episode_length - 1, dtype=np.bool_),
        "is_terminal": ensure_scalar_array(frame_index == episode_length - 1, dtype=np.bool_),
        "language_instruction": task,
        "language_instruction_2": "",
        "language_instruction_3": "",
        "subtask_index": ensure_scalar_array(0, dtype=np.int64),
        "observation.state.gripper_position": gripper_position.astype(np.float32),
        "observation.state.cartesian_position": cartesian_position.astype(np.float32),
        "observation.state.joint_position": joint_position.astype(np.float32),
        "observation.state": np.concatenate([joint_position, gripper_position]).astype(np.float32),
        "observation.images.wrist_left": wrist_img,
        "observation.images.exterior_1_left": exterior_1_img,
        "observation.images.exterior_2_left": exterior_2_img,
        "action.gripper_position": action_gripper_position.astype(np.float32),
        "action.gripper_velocity": action_gripper_velocity.astype(np.float32),
        "action.cartesian_position": action_cartesian_position.astype(np.float32),
        "action.cartesian_velocity": action_cartesian_velocity.astype(np.float32),
        "action.joint_position": action_joint_position.astype(np.float32),
        "action.joint_velocity": action_joint_velocity.astype(np.float32),
        "action.original": np.concatenate([action_cartesian_velocity, action_gripper_position]).astype(np.float32),
        "action.source_joint_velocity_gripper": np.concatenate(
            [action_joint_velocity, action_gripper_position]
        ).astype(np.float32),
        "action": np.concatenate([action_joint_position, action_gripper_position]).astype(np.float32),
        "discount": ensure_scalar_array(0.0, dtype=np.float32),
        "reward": ensure_scalar_array(float(is_episode_successful and frame_index == episode_length - 1), dtype=np.float32),
        "task": task,
    }

    frame.update(
        build_frame_meta(
            building=building,
            collector_id=collector_id,
            datetime_value=datetime_value,
            is_episode_successful=is_episode_successful,
            task_category=building,
            camera_extrinsics=_find_camera_extrinsics(step),
        )
    )

    subtask_label = resolve_subtask_for_frame(annotation, frame_index=frame_index, episode_length=episode_length)
    frame["subtask_index"] = ensure_scalar_array(0, dtype=np.int64)
    frame.update(compute_env_stats_for_frame(frame, frame_index=frame_index, episode_length=episode_length, annotation=annotation))
    return frame, subtask_label


@dataclass
class RawConversionConfig:
    data_dir: Path
    repo_id: str
    output_root: Path | None = None
    annotations_path: Path | None = None
    push_to_hub: bool = False
    overwrite: bool = False
    resume: bool = False
    batch_encoding_size: int = 1
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    start_episode: int = 0
    end_episode: int | None = None
    suppress_encoder_output: bool = True
    mongo_url: str | None = None
    mongo_db_name: str = "eii_data_system"
    mongo_project_path_filters: list[str] | None = None
    mongo_project_date_filters: list[str] | None = None


def _load_raw_droid_trajectory(trajectory_path: str, *, recording_folderpath: str):
    module = importlib.import_module("examples.droid.convert_droid_data_to_lerobot")
    return module.load_trajectory(trajectory_path, recording_folderpath=recording_folderpath)


def main(config: RawConversionConfig) -> None:
    warnings.filterwarnings(
        "ignore",
        message="Converting input from bool to <class 'numpy.uint8'> for compatibility.",
        category=RuntimeWarning,
    )
    annotations = AnnotationIndex.load(config.annotations_path)
    metadata_index = None
    if config.mongo_url is not None:
        mongo = ReadOnlyDroidMongo.connect(config.mongo_url, db_name=config.mongo_db_name)
        metadata_index = mongo.build_export_index(
            path_substrings=config.mongo_project_path_filters,
            date_substrings=config.mongo_project_date_filters,
        )
    writer = CanonicalDatasetWriter(
        repo_id=config.repo_id,
        root=config.output_root,
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

    episode_paths = sorted(config.data_dir.glob("**/trajectory.h5"))
    end_episode = len(episode_paths) if config.end_episode is None else min(config.end_episode, len(episode_paths))
    selected_paths = episode_paths[config.start_episode:end_episode]
    logging.info("Converting %s raw episodes from %s", len(selected_paths), config.data_dir)

    started_at = time.perf_counter()
    visited_count = 0
    converted_count = 0
    skipped_count = 0
    for source_episode_index, episode_path in enumerate(selected_paths, start=config.start_episode):
        visited_count += 1
        recording_folderpath = episode_path.parent / "recordings" / "MP4"
        trajectory = _load_raw_droid_trajectory(str(episode_path), recording_folderpath=str(recording_folderpath))
        if len(trajectory) == 0:
            logging.warning("Skipping empty trajectory: %s", episode_path)
            skipped_count += 1
            continue

        annotation = None if annotations is None else annotations.match_raw_episode(episode_path.parent.name)
        resolved_metadata = resolve_raw_episode_metadata(
            index=metadata_index,
            folder_name=episode_path.parent.name,
            annotation=annotation,
            fallback_task=annotation.task if annotation and annotation.task else "Do something",
        )
        if resolved_metadata.match_status == "ambiguous":
            skipped_count += 1
            writer.report["episodes_ambiguous_metadata"] += 1
            writer.migration_rows.append(
                {
                    "source_repo_id": config.repo_id,
                    "source_episode_index": source_episode_index,
                    "source_path": str(episode_path),
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
        task = resolved_metadata.task or "Do something"
        building, collector_id, datetime_value = (
            resolved_metadata.building,
            resolved_metadata.collector_id,
            resolved_metadata.datetime_value,
        )
        is_episode_successful = "/success/" in str(episode_path)

        frames: list[dict] = []
        for frame_index, step in enumerate(trajectory):
            frame, subtask_label = _build_raw_frame(
                step,
                frame_index=frame_index,
                episode_length=len(trajectory),
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
            source_repo_id=config.repo_id,
            source_episode_index=source_episode_index,
            source_path=str(episode_path),
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
        eta_seconds = (elapsed / visited_count) * (len(selected_paths) - visited_count) if visited_count else 0.0
        logging.info(
            "[%s] visited %s/%s episodes (%.1f%%), converted=%s, skipped=%s, ETA %s",
            config.repo_id,
            visited_count,
            len(selected_paths),
            (visited_count / len(selected_paths) * 100.0) if selected_paths else 100.0,
            converted_count,
            skipped_count,
            format_duration(eta_seconds),
        )

    writer.finalize()
    validate_canonical_dataset(writer.root, config.repo_id)

    if config.push_to_hub:
        writer.dataset.push_to_hub(tags=["droid", "canonical"], private=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(tyro.cli(RawConversionConfig))

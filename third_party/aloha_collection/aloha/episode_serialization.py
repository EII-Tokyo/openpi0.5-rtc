"""Serialize an accepted in-memory episode and publish it atomically."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np

from aloha.episode_validation import validate_episode_outputs
from aloha.video_encoding import encode_cameras_parallel
from aloha.episode_image_spool import EpisodeImageSpool


CAMERA_SAVE_NAMES = {
    "camera_high": "cam_high",
    "camera_low": "cam_low",
    "camera_wrist_right": "cam_right_wrist",
    "camera_wrist_left": "cam_left_wrist",
}


def build_camera_map(camera_names: list[str]) -> dict[str, str]:
    """Map ROS camera names to the established dataset video names."""
    return {
        camera_name: CAMERA_SAVE_NAMES.get(camera_name, camera_name)
        for camera_name in camera_names
    }


@dataclass(frozen=True)
class EpisodeSavePayload:
    """Immutable ownership bundle transferred to the background saver."""

    staged: Any
    dataset_name: str
    timesteps: tuple[Any, ...]
    actions: tuple[Any, ...]
    camera_map: Mapping[str, str]
    video_fps: float
    total_joint_size: int
    is_mobile: bool
    continuous_roll_joints: bool
    allow_existing: bool
    video_backend: str = "auto"
    ffmpeg_bin: str = "ffmpeg"
    artifact: Any | None = None
    image_spool: EpisodeImageSpool | None = None


def _write_hdf5(
    payload: EpisodeSavePayload,
    dataset_path: Path,
    video_paths: Mapping[str, Path],
) -> None:
    timestep_count = len(payload.actions)
    qpos = np.asarray(
        [
            timestep.observation["qpos"]
            for timestep in payload.timesteps
        ]
    )
    qvel = np.asarray(
        [
            timestep.observation["qvel"]
            for timestep in payload.timesteps
        ]
    )
    effort = np.asarray(
        [
            timestep.observation["effort"]
            for timestep in payload.timesteps
        ]
    )
    actions = np.asarray(payload.actions)
    expected_shape = (timestep_count, payload.total_joint_size)
    for dataset_name, array in (
        ("qpos", qpos),
        ("qvel", qvel),
        ("effort", effort),
        ("action", actions),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"{dataset_name} shape {array.shape} does not match "
                f"{expected_shape}"
            )

    with h5py.File(dataset_path, "w", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        root.attrs["image_storage"] = "mp4"
        root.attrs["video_fps"] = float(payload.video_fps)
        root.attrs["video_frame_count"] = timestep_count
        root.attrs["continuous_roll_joints"] = bool(
            payload.continuous_roll_joints
        )
        if payload.continuous_roll_joints:
            root.attrs["continuous_joint_indices"] = np.asarray(
                [3, 5],
                dtype=np.int32,
            )
            root.attrs["continuous_joint_names"] = json.dumps(
                ["forearm_roll", "wrist_rotate"]
            )

        observations = root.create_group("observations")
        if video_paths:
            videos = observations.create_group("videos")
            string_type = h5py.string_dtype(encoding="utf-8")
            for camera_name, path in video_paths.items():
                videos.create_dataset(
                    camera_name,
                    data=path.name,
                    dtype=string_type,
                )
        observations.create_dataset("qpos", data=qpos)
        observations.create_dataset("qvel", data=qvel)
        observations.create_dataset("effort", data=effort)
        root.create_dataset("action", data=actions)
        if payload.is_mobile:
            base_action = np.asarray(
                [
                    timestep.observation.get("base_vel", [0.0, 0.0])
                    for timestep in payload.timesteps
                ]
            )
            if base_action.shape != (timestep_count, 2):
                raise ValueError(
                    f"base_action shape {base_action.shape} does not match "
                    f"{(timestep_count, 2)}"
                )
            root.create_dataset("base_action", data=base_action)


def save_episode(
    payload: EpisodeSavePayload,
    *,
    encode_videos: Callable[..., Mapping[str, Path]] = (
        encode_cameras_parallel
    ),
    validate_outputs: Callable[..., None] = validate_episode_outputs,
    logger: Callable[[str], None] = print,
) -> Path:
    """Encode, serialize, validate and publish one worker-owned episode."""
    timestep_count = len(payload.actions)
    if timestep_count < 1:
        raise ValueError("cannot save an episode with no actions")
    if len(payload.timesteps) != timestep_count:
        raise ValueError(
            "timesteps and actions must have identical lengths before saving"
        )

    staging_path = Path(payload.staged.staging_path)
    dataset_path = staging_path / "episode.hdf5"
    start_time = time.monotonic()
    try:
        encode_start = time.monotonic()
        encode_kwargs = {
            "fps": payload.video_fps,
            "backend": payload.video_backend,
            "ffmpeg_bin": payload.ffmpeg_bin,
        }
        if payload.image_spool is not None:
            if (
                payload.image_spool.selected_frame_count
                != timestep_count
            ):
                raise ValueError(
                    "image spool selected frame count does not match actions"
                )
            if set(payload.image_spool.camera_names) != set(
                payload.camera_map
            ):
                raise ValueError(
                    "image spool cameras do not match episode camera map"
                )
            encode_kwargs["frame_source"] = payload.image_spool.frames
        video_paths = encode_videos(
            payload.timesteps,
            payload.camera_map,
            staging_path,
            **encode_kwargs,
        )
        if payload.image_spool is not None:
            payload.image_spool.discard()
        logger(
            f"[保存:{payload.dataset_name}] 视频编码完成: "
            f"{time.monotonic() - encode_start:.2f}s"
        )

        hdf5_start = time.monotonic()
        _write_hdf5(payload, dataset_path, video_paths)
        logger(
            f"[保存:{payload.dataset_name}] HDF5 写入完成: "
            f"{time.monotonic() - hdf5_start:.2f}s"
        )

        if payload.artifact is not None:
            payload.artifact.commit_into_existing(
                staging_path,
                allow_existing_destination=False,
            )

        validate_start = time.monotonic()
        validate_outputs(
            staging_path,
            expected_timesteps=timestep_count,
            camera_file_names=payload.camera_map.values(),
        )
        logger(
            f"[保存:{payload.dataset_name}] 完整性校验完成: "
            f"{time.monotonic() - validate_start:.2f}s"
        )
        final_path = payload.staged.publish(
            allow_existing_destination=payload.allow_existing,
        )
        logger(
            f"[保存:{payload.dataset_name}] 已原子发布到 {final_path}; "
            f"总耗时 {time.monotonic() - start_time:.2f}s"
        )
        return final_path
    except BaseException:
        payload.staged.discard()
        if payload.artifact is not None:
            try:
                payload.artifact.discard()
            except (RuntimeError, FileNotFoundError):
                pass
        raise

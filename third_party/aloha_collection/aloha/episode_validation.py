"""Behavioral validation for staged HDF5 and MP4 episode outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import h5py


class EpisodeValidationError(RuntimeError):
    """Raised when a staged episode cannot be read back as usable data."""


REQUIRED_HDF5_DATASETS = (
    "/action",
    "/observations/qpos",
    "/observations/qvel",
    "/observations/effort",
)


def validate_episode_outputs(
    staging_dir: str | os.PathLike[str],
    *,
    expected_timesteps: int,
    camera_file_names: Iterable[str],
) -> None:
    """Reopen and validate every output required before episode publication."""
    if expected_timesteps < 0:
        raise ValueError("expected_timesteps must be non-negative")
    staging_path = Path(staging_dir)
    _validate_hdf5(
        staging_path / "episode.hdf5",
        expected_timesteps=expected_timesteps,
    )
    for camera_name in camera_file_names:
        file_name = (
            camera_name
            if str(camera_name).endswith(".mp4")
            else f"{camera_name}.mp4"
        )
        _validate_mp4(staging_path / file_name)


def _validate_hdf5(path: Path, *, expected_timesteps: int) -> None:
    if not path.is_file():
        raise EpisodeValidationError(f"missing HDF5 output: {path}")
    try:
        with h5py.File(path, "r") as root:
            reference_tail_shape = None
            for dataset_name in REQUIRED_HDF5_DATASETS:
                if dataset_name not in root:
                    raise EpisodeValidationError(
                        f"missing required HDF5 dataset: {dataset_name}"
                    )
                dataset = root[dataset_name]
                if not hasattr(dataset, "shape") or len(dataset.shape) != 2:
                    raise EpisodeValidationError(
                        f"HDF5 dataset has invalid shape "
                        f"{getattr(dataset, 'shape', None)}: {dataset_name}"
                    )
                actual_timesteps = int(dataset.shape[0])
                if actual_timesteps != expected_timesteps:
                    raise EpisodeValidationError(
                        f"HDF5 dataset {dataset_name} expected "
                        f"{expected_timesteps} timesteps, found "
                        f"{actual_timesteps}"
                    )
                if reference_tail_shape is None:
                    reference_tail_shape = dataset.shape[1:]
                elif dataset.shape[1:] != reference_tail_shape:
                    raise EpisodeValidationError(
                        f"HDF5 dataset shape {dataset.shape} does not match "
                        f"required tail shape {reference_tail_shape}: "
                        f"{dataset_name}"
                    )
    except EpisodeValidationError:
        raise
    except BaseException as exc:
        raise EpisodeValidationError(
            f"unreadable HDF5 output {path}: {exc}"
        ) from exc


def _validate_mp4(path: Path) -> None:
    if not path.is_file():
        raise EpisodeValidationError(f"missing MP4 output: {path.name}")
    if path.stat().st_size <= 0:
        raise EpisodeValidationError(f"empty MP4 output: {path.name}")
    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        readable, _frame = capture.read() if opened else (False, None)
    finally:
        capture.release()
    if not opened or not readable:
        raise EpisodeValidationError(f"unreadable MP4 output: {path.name}")

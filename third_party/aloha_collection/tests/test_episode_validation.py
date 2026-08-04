from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from aloha.episode_validation import (
    EpisodeValidationError,
    validate_episode_outputs,
)


REQUIRED_DATASETS = (
    "/action",
    "/observations/qpos",
    "/observations/qvel",
    "/observations/effort",
)


def write_hdf5(path: Path, *, timesteps: int = 3, omit: str | None = None):
    with h5py.File(path, "w") as root:
        for dataset_name in REQUIRED_DATASETS:
            if dataset_name == omit:
                continue
            root.create_dataset(dataset_name, data=np.zeros((timesteps, 14)))


def write_mp4(path: Path, *, frames: int = 2):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 16),
    )
    assert writer.isOpened()
    for value in range(frames):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()


def test_valid_episode_outputs_are_accepted(tmp_path):
    write_hdf5(tmp_path / "episode.hdf5")
    write_mp4(tmp_path / "cam_high.mp4")
    write_mp4(tmp_path / "cam_low.mp4")

    validate_episode_outputs(
        tmp_path,
        expected_timesteps=3,
        camera_file_names=["cam_high", "cam_low"],
    )


def test_missing_required_hdf5_dataset_is_rejected(tmp_path):
    write_hdf5(tmp_path / "episode.hdf5", omit="/observations/qvel")

    with pytest.raises(EpisodeValidationError, match="observations/qvel"):
        validate_episode_outputs(
            tmp_path,
            expected_timesteps=3,
            camera_file_names=[],
        )


def test_inconsistent_hdf5_length_is_rejected(tmp_path):
    write_hdf5(tmp_path / "episode.hdf5")
    with h5py.File(tmp_path / "episode.hdf5", "a") as root:
        del root["/observations/qpos"]
        root.create_dataset("/observations/qpos", data=np.zeros((2, 14)))

    with pytest.raises(EpisodeValidationError, match="expected 3.*found 2"):
        validate_episode_outputs(
            tmp_path,
            expected_timesteps=3,
            camera_file_names=[],
        )


@pytest.mark.parametrize(
    "replacement",
    [
        np.zeros((3,)),
        np.zeros((3, 13)),
    ],
)
def test_required_hdf5_datasets_need_matching_two_dimensional_shapes(
    tmp_path,
    replacement,
):
    write_hdf5(tmp_path / "episode.hdf5")
    with h5py.File(tmp_path / "episode.hdf5", "a") as root:
        del root["/observations/effort"]
        root.create_dataset("/observations/effort", data=replacement)

    with pytest.raises(EpisodeValidationError, match="shape"):
        validate_episode_outputs(
            tmp_path,
            expected_timesteps=3,
            camera_file_names=[],
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"", "empty"),
        (b"not an mp4", "unreadable"),
    ],
)
def test_invalid_mp4_is_rejected(tmp_path, contents, message):
    write_hdf5(tmp_path / "episode.hdf5")
    (tmp_path / "cam_high.mp4").write_bytes(contents)

    with pytest.raises(EpisodeValidationError, match=message):
        validate_episode_outputs(
            tmp_path,
            expected_timesteps=3,
            camera_file_names=["cam_high"],
        )


def test_missing_mp4_is_rejected(tmp_path):
    write_hdf5(tmp_path / "episode.hdf5")

    with pytest.raises(EpisodeValidationError, match="missing.*cam_high"):
        validate_episode_outputs(
            tmp_path,
            expected_timesteps=3,
            camera_file_names=["cam_high"],
        )

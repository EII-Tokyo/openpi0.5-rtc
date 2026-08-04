from types import SimpleNamespace

import numpy as np
import pytest

from aloha.episode_image_spool import (
    EpisodeImageSpoolWriter,
    strip_and_spool_timestep,
)


def _images(value, *, shape=(3, 4, 3)):
    return {
        "camera_high": np.full(shape, value, dtype=np.uint8),
        "camera_wrist": np.full(shape, value + 10, dtype=np.uint8),
    }


def test_spool_round_trips_selected_prefix_without_retaining_images(tmp_path):
    writer = EpisodeImageSpoolWriter(
        tmp_path,
        ("camera_high", "camera_wrist"),
    )
    original = []
    for value in range(4):
        images = _images(value)
        original.append(images)
        writer.append(images)

    spool = writer.seal(selected_frame_count=3)

    assert spool.available_frame_count == 4
    assert spool.selected_frame_count == 3
    assert tuple(spool.camera_names) == ("camera_high", "camera_wrist")
    for camera_name in spool.camera_names:
        frames = list(spool.frames(camera_name))
        assert len(frames) == 3
        for frame_idx, frame in enumerate(frames):
            np.testing.assert_array_equal(
                frame,
                original[frame_idx][camera_name],
            )


def test_spool_rejects_camera_shape_and_dtype_changes_before_writing(tmp_path):
    writer = EpisodeImageSpoolWriter(
        tmp_path,
        ("camera_high", "camera_wrist"),
    )
    writer.append(_images(0))

    changed_shape = _images(1)
    changed_shape["camera_high"] = np.zeros((5, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="camera_high.*shape"):
        writer.append(changed_shape)

    changed_dtype = _images(1)
    changed_dtype["camera_wrist"] = np.zeros(
        (3, 4, 3),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="camera_wrist.*uint8"):
        writer.append(changed_dtype)

    assert writer.frame_count == 1


def test_spool_discard_removes_only_its_owned_directory(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    writer = EpisodeImageSpoolWriter(tmp_path, ("camera_high",))
    writer.append({"camera_high": np.zeros((2, 2, 3), dtype=np.uint8)})
    spool_dir = writer.spool_dir

    writer.discard()
    writer.discard()

    assert not spool_dir.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_strip_and_spool_timestep_keeps_numeric_observation_only(tmp_path):
    writer = EpisodeImageSpoolWriter(tmp_path, ("camera_high",))
    timestep = SimpleNamespace(
        observation={
            "qpos": np.arange(3),
            "images": {
                "camera_high": np.full((2, 2, 3), 7, dtype=np.uint8)
            },
        }
    )

    stripped = strip_and_spool_timestep(writer, timestep)
    spool = writer.seal(selected_frame_count=1)

    assert stripped is not timestep
    assert stripped.observation["images"] == {}
    np.testing.assert_array_equal(stripped.observation["qpos"], np.arange(3))
    np.testing.assert_array_equal(
        next(spool.frames("camera_high")),
        np.full((2, 2, 3), 7, dtype=np.uint8),
    )

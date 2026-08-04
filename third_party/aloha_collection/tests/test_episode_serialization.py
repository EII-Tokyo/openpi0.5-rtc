from types import SimpleNamespace
import pickle
import shutil

import h5py
import numpy as np
import pytest

from aloha.episode_serialization import (
    EpisodeSavePayload,
    build_camera_map,
    save_episode,
)
from aloha.episode_image_spool import EpisodeImageSpoolWriter
from aloha.episode_storage import StagedEpisode
from aloha.episode_attempt import AttemptArtifact


def _payload(tmp_path, *, index=2, frame_count=3):
    staged = StagedEpisode.create(tmp_path, index)
    camera_map = {
        "camera_high": "cam_high",
        "camera_wrist_left": "cam_left_wrist",
    }
    timesteps = []
    actions = []
    for frame_idx in range(frame_count):
        timesteps.append(
            SimpleNamespace(
                observation={
                    "qpos": np.full(14, frame_idx, dtype=np.float64),
                    "qvel": np.full(14, frame_idx + 1, dtype=np.float64),
                    "effort": np.full(14, frame_idx + 2, dtype=np.float64),
                    "images": {
                        name: np.full(
                            (6, 8, 3),
                            frame_idx,
                            dtype=np.uint8,
                        )
                        for name in camera_map
                    },
                }
            )
        )
        actions.append(np.full(14, frame_idx + 3, dtype=np.float64))
    return EpisodeSavePayload(
        staged=staged,
        dataset_name=f"episode_{index}",
        timesteps=tuple(timesteps),
        actions=tuple(actions),
        camera_map=camera_map,
        video_fps=50,
        total_joint_size=14,
        is_mobile=False,
        continuous_roll_joints=True,
        allow_existing=False,
        video_backend="cpu",
    )


def _fake_encode(timesteps, camera_map, output_dir, **_kwargs):
    outputs = {}
    for save_name in camera_map.values():
        path = output_dir / f"{save_name}.mp4"
        path.write_bytes(b"fake-mp4")
        outputs[save_name] = path
    return outputs


def test_build_camera_map_preserves_dataset_naming_contract():
    assert build_camera_map(
        [
            "camera_high",
            "camera_low",
            "camera_wrist_right",
            "camera_wrist_left",
            "custom",
        ]
    ) == {
        "camera_high": "cam_high",
        "camera_low": "cam_low",
        "camera_wrist_right": "cam_right_wrist",
        "camera_wrist_left": "cam_left_wrist",
        "custom": "custom",
    }


def test_save_episode_writes_hdf5_references_then_atomically_publishes(tmp_path):
    payload = _payload(tmp_path)
    validated = []

    final_path = save_episode(
        payload,
        encode_videos=_fake_encode,
        validate_outputs=lambda path, **kwargs: validated.append(
            (path, kwargs)
        ),
    )

    assert final_path == tmp_path / "episode_2"
    assert final_path.is_dir()
    assert not payload.staged.staging_path.exists()
    assert validated[0][0] == payload.staged.staging_path
    assert validated[0][1]["expected_timesteps"] == 3
    with h5py.File(final_path / "episode.hdf5", "r") as root:
        assert root.attrs["image_storage"] == "mp4"
        assert root.attrs["video_fps"] == 50
        assert root.attrs["video_frame_count"] == 3
        assert root.attrs["continuous_roll_joints"]
        assert root["/action"].shape == (3, 14)
        assert root["/observations/qpos"].shape == (3, 14)
        assert root["/observations/videos/cam_high"][()].decode() == (
            "cam_high.mp4"
        )
        assert root["/observations/videos/cam_left_wrist"][()].decode() == (
            "cam_left_wrist.mp4"
        )


def test_validation_failure_discards_staging_and_releases_claim(tmp_path):
    payload = _payload(tmp_path, index=7)

    with pytest.raises(RuntimeError, match="injected validation failure"):
        save_episode(
            payload,
            encode_videos=_fake_encode,
            validate_outputs=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected validation failure")
            ),
        )

    assert not (tmp_path / "episode_7").exists()
    assert not payload.staged.staging_path.exists()
    assert not (tmp_path / ".episode_7.claim").exists()


def test_save_episode_reads_images_from_spool_and_removes_transport_files(
    tmp_path,
):
    payload = _payload(tmp_path, index=9, frame_count=3)
    writer = EpisodeImageSpoolWriter(
        payload.staged.staging_path,
        tuple(payload.camera_map),
    )
    stripped_timesteps = []
    for timestep in payload.timesteps:
        writer.append(timestep.observation["images"])
        observation = dict(timestep.observation)
        observation["images"] = {}
        stripped_timesteps.append(
            SimpleNamespace(observation=observation)
        )
    spool = writer.seal(selected_frame_count=len(payload.actions))
    payload = EpisodeSavePayload(
        **{
            **payload.__dict__,
            "timesteps": tuple(stripped_timesteps),
            "image_spool": spool,
        }
    )
    seen = {}

    def fake_spool_encode(
        timesteps,
        camera_map,
        output_dir,
        *,
        frame_source=None,
        **_kwargs,
    ):
        seen["timesteps"] = timesteps
        seen["frames"] = {
            camera_name: [frame.copy() for frame in frame_source(camera_name)]
            for camera_name in camera_map
        }
        return _fake_encode(timesteps, camera_map, output_dir)

    final_path = save_episode(
        payload,
        encode_videos=fake_spool_encode,
        validate_outputs=lambda *_args, **_kwargs: None,
    )

    assert all(
        timestep.observation["images"] == {}
        for timestep in seen["timesteps"]
    )
    assert [int(frame[0, 0, 0]) for frame in seen["frames"]["camera_high"]] == [
        0,
        1,
        2,
    ]
    assert not any(path.name.startswith(".image-spool-") for path in final_path.iterdir())


def test_spooled_payload_is_picklable_for_spawn_transport(tmp_path):
    payload = _payload(tmp_path, index=10, frame_count=2)
    artifact = AttemptArtifact.create(tmp_path, "episode_10")
    writer = EpisodeImageSpoolWriter(
        payload.staged.staging_path,
        tuple(payload.camera_map),
    )
    for timestep in payload.timesteps:
        writer.append(timestep.observation["images"])
    payload = EpisodeSavePayload(
        **{
            **payload.__dict__,
            "artifact": artifact,
            "image_spool": writer.seal(selected_frame_count=2),
        }
    )

    restored = pickle.loads(pickle.dumps(payload))

    assert restored.dataset_name == "episode_10"
    assert restored.image_spool.selected_frame_count == 2
    assert restored.staged.staging_path == payload.staged.staging_path
    payload.staged.discard()
    artifact.discard()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg unavailable")
def test_real_cpu_episode_save_is_readable_and_published(tmp_path):
    payload = _payload(tmp_path, index=8, frame_count=4)

    final_path = save_episode(payload, logger=lambda _message: None)

    assert (final_path / "episode.hdf5").is_file()
    assert (final_path / "cam_high.mp4").is_file()
    assert (final_path / "cam_left_wrist.mp4").is_file()

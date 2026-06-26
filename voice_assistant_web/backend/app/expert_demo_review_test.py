import json

import pyarrow as pa
import pyarrow.parquet as pq

from voice_assistant_web.backend.app import expert_demo_review
from voice_assistant_web.backend.app.expert_demo_review import crop_expert_demo
from voice_assistant_web.backend.app.expert_demo_review import list_expert_demos


def _write_demo_dataset(root, dataset_id="demo-rinse", *, episodes=2):
    dataset = root / dataset_id
    (dataset / "meta").mkdir(parents=True)
    data_dir = dataset / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({"fps": 50, "total_episodes": episodes}))
    episode_indices = []
    frame_indices = []
    for episode_index in range(episodes):
        for offset in range(10):
            episode_indices.append(episode_index)
            frame_indices.append(episode_index * 10 + offset)
    pq.write_table(
        pa.table({"episode_index": episode_indices, "index": frame_indices}),
        data_dir / "file-000.parquet",
    )
    for camera in (
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        video_dir = dataset / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"mp4")
    return dataset


def test_list_expert_demos_indexes_lerobot_videos(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_demo_review, "_video_frame_count", lambda _: 1000)
    _write_demo_dataset(tmp_path, episodes=2)
    _write_demo_dataset(tmp_path, "2026-01-20-twist-one-bottle", episodes=1)
    _write_demo_dataset(tmp_path, "2026-05-04_direction-lerobot-with-rinse", episodes=1)
    _write_demo_dataset(tmp_path, "2026-05-01_turn_over-lerobot-with-rinse", episodes=1)

    page = list_expert_demos(tmp_path, limit=10, offset=0)

    assert page.total == 2
    assert [record.episode_key for record in page.items] == ["demo-rinse::0", "demo-rinse::1"]
    assert page.items[0].dataset_id == "demo-rinse"
    assert page.items[0].episode_index == 0
    assert page.items[0].fps == 50
    assert len(page.items[0].video_paths) == 4
    assert page.items[0].camera_complete is True
    assert page.items[0].camera_count == 4
    assert page.items[1].video_start_secs == [0.2, 0.2, 0.2, 0.2]
    assert page.datasets == ["demo-rinse"]


def test_list_expert_demos_filters_dataset_and_search(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_demo_review, "_video_frame_count", lambda _: 1000)
    _write_demo_dataset(tmp_path, "alpha-rinse", episodes=1)
    _write_demo_dataset(tmp_path, "beta-rinse", episodes=1)

    page = list_expert_demos(tmp_path, dataset="beta-rinse", search="episode 0")

    assert page.total == 1
    assert page.items[0].dataset_id == "beta-rinse"


def test_crop_expert_demo_writes_discriminator_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_demo_review, "_video_frame_count", lambda _: 1000)
    dataset_root = tmp_path / "hf"
    crop_root = tmp_path / "crops"
    _write_demo_dataset(dataset_root, "demo-rinse", episodes=1)

    result = crop_expert_demo(
        dataset_root,
        crop_root,
        dataset_id="demo-rinse",
        episode_index=0,
        start_sec=0.5,
        end_sec=1.5,
    )

    assert result.label == "expert"
    assert result.metadata_path is not None
    metadata = json.loads((crop_root / "demo-rinse" / "episode_000000_crop_000000.json").read_text())
    assert metadata["dataset_id"] == "demo-rinse"
    assert metadata["episode_index"] == 0
    assert metadata["start_sec"] == 0.5
    assert metadata["end_sec"] == 1.5
    assert metadata["label"] == "expert"

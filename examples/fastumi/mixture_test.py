import dataclasses
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.fastumi import prepare_mixture
from openpi import transforms
from openpi.training import config
from openpi.training import data_loader
from scripts import compute_norm_stats


def test_mixture_cli_and_shapes_at_dataset_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "pi05_fastumi_pick_fan_relative_mix",
            "--exp-name=umi_pi05_run3_relative_pose",
            "--fsdp-devices=2",
        ],
    )
    cfg = config.cli()
    assert cfg.wandb_enabled
    assert cfg.num_train_steps == 50000
    assert cfg.data.repo_id is None
    assert len(cfg.data.repo_ids) == 3
    assert cfg.data.assets.asset_id == "fastumi_pick_fan_run3"
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    dc = cfg.data.create(tmp_path, cfg.model)
    calls = []

    class FakeDataset:
        def __init__(self, repo_id, **kwargs):
            self.number = cfg.data.repo_ids.index(repo_id)
            calls.append(kwargs["delta_timestamps"])

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "observation.images.front": np.zeros((3, 8, 10), np.float32),
                "observation.state": np.array([self.number, idx, 0, 0, 0, 0, 0.5]),
                "action": np.tile([self.number, idx + 1, 0, 0, 0, 0, 0.5], (16, 1)),
                "task": "fan",
            }

        def get_sample_context(self, idx):
            return {"repo_id": cfg.data.repo_ids[self.number], "index": idx}

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: SimpleNamespace(fps=20))
    monkeypatch.setattr(data_loader, "SchemaRemappingLeRobotDataset", FakeDataset)
    dataset = data_loader.create_torch_dataset(dc, 16, cfg.model)
    assert len(dataset) == 6
    transform = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
    for idx in (1, 2, 3, 4, 5):
        raw = dataset[idx]
        assert raw["observation.state"].shape == (7,)
        assert raw["observation.images.front"].shape == (3, 8, 10)
        out = transform(raw)
        assert out["state"].shape == (1,)
        assert out["actions"].shape == (16, 10)
        np.testing.assert_array_equal(out["actions"][:, :3], np.tile([0, 1, 0], (16, 1)))
    for times in calls:
        assert set(times) == {"action"}
        np.testing.assert_allclose(times["action"], np.arange(1, 17) / 20)
    context = data_loader._extract_lerobot_sample_context(dataset, 2)  # noqa: SLF001
    assert context == {"repo_id": cfg.data.repo_ids[1], "index": 0}
    # Statistics accept a mixture with no single repo_id and see every dataset.
    loader, batches = compute_norm_stats.create_torch_dataloader(dc, 16, 2, cfg.model, 0)
    assert batches == 3
    assert next(iter(loader))["actions"].shape == (2, 16, 10)
    with pytest.raises(ValueError, match="distinct"):
        dataclasses.replace(cfg.data, repo_ids=["duplicate", "duplicate"]).create(tmp_path, cfg.model)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: SimpleNamespace(fps=30))
    with pytest.raises(ValueError, match="20 Hz"):
        data_loader.create_torch_dataset(dc, 16, cfg.model)


def test_unfinished_manifest_stops_before_network(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"datasets": [{"repo_id": "user/uploading", "revision": None}]}))
    monkeypatch.setattr(prepare_mixture, "HfApi", lambda: pytest.fail("Must fail before contacting Hub"))
    with pytest.raises(ValueError, match="uploads to finish"):
        prepare_mixture.main(path)


def test_source_requires_all_episode_files():
    info = {
        "codebase_version": "v2.1",
        "fps": 20,
        "total_episodes": 2,
        "total_frames": 4,
        "chunks_size": 1000,
        "data_path": "data/{episode_index}.parquet",
        "video_path": "videos/{episode_index}.mp4",
        "features": {
            key: {"shape": [7], "names": prepare_mixture.POSE_NAMES} for key in ("observation.state", "action")
        },
    }
    info["features"]["observation.images.front"] = {"dtype": "video", "shape": [720, 1280, 3]}
    names = {
        "data/0.parquet",
        "data/1.parquet",
        "videos/0.mp4",
        "videos/1.mp4",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
    }
    prepare_mixture.validate_source(info, names)
    with pytest.raises(ValueError, match="Upload incomplete"):
        prepare_mixture.validate_source(info, names - {"videos/1.mp4"})
    info["features"]["action"]["names"] = ["wrong"] * 7
    with pytest.raises(ValueError, match="schema"):
        prepare_mixture.validate_source(info, names)


@pytest.mark.parametrize("config_name", ["pi05_fastumi_pick_fan_relative", "pi05_fastumi_pick_fan_relative_mix"])
def test_server_warmup_uses_relative_policy_contract(config_name, monkeypatch):
    from scripts import serve_policy  # noqa: PLC0415

    calls = []
    fake = SimpleNamespace(
        metadata=config.get_config(config_name).policy_metadata, infer=lambda obs, **kwargs: calls.append((obs, kwargs))
    )
    monkeypatch.setattr(serve_policy, "create_policy", lambda _: fake)
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda _: "127.0.0.1")
    monkeypatch.setattr(
        serve_policy.websocket_policy_server,
        "WebsocketPolicyServer",
        lambda **_: SimpleNamespace(serve_forever=lambda: None),
    )
    serve_policy.main(serve_policy.Args(policy=serve_policy.Checkpoint(config=config_name, dir="unused")))
    assert len(calls) == 1
    assert calls[0][1] == {"use_rtc": False}
    assert set(calls[0][0]) == {"observation/wrist_image", "observation/gripper_position"}


def test_norm_stats_main_saves_under_mixture_asset_id(tmp_path, monkeypatch):
    from openpi.shared import normalize  # noqa: PLC0415

    cfg = dataclasses.replace(config.get_config("pi05_fastumi_pick_fan_relative_mix"), assets_base_dir=str(tmp_path))
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    monkeypatch.setattr(compute_norm_stats._config, "get_config", lambda _: cfg)  # noqa: SLF001
    batch = {"state": np.array([[0.2], [0.8]]), "actions": np.arange(320).reshape(2, 16, 10).astype(np.float32)}
    monkeypatch.setattr(compute_norm_stats, "create_torch_dataloader", lambda *_: ([batch], 1))
    compute_norm_stats.main(cfg.name, num_workers=2)
    stats = normalize.load(cfg.assets_dirs / "fastumi_pick_fan_run3")
    assert stats["state"].mean.shape == (1,)
    assert stats["actions"].mean.shape == (10,)
    np.testing.assert_allclose(stats["state"].mean, [0.5])

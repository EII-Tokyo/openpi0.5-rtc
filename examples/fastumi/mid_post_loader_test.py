import hashlib
import json

import numpy as np
import pyarrow as pa
import pytest

from examples.fastumi.prepare_mid_post_data import audit_arrays
from openpi.policies.fastumi_policy import FastUMIRelativeInputs
from openpi.training.fastumi_two_phase import FastUMIEpisodes
from openpi.training.fastumi_two_phase import GroupSampler


def test_balanced_batches_and_weighted_midtrain():
    for fraction in (0.1, 0.5):
        sampler = GroupSampler(5, 100, 32, fraction, seed=42, batches=100)
        indices = np.array(list(sampler)).reshape(100, 32)
        assert (indices >= 0).all()
        assert (indices < 105).all()
        counts = (indices < 5).sum(axis=1)
        assert counts.sum() == round(3200 * fraction)
        assert np.max(abs(counts.cumsum() - np.arange(1, 101) * 32 * fraction)) <= 0.5
        if fraction == 0.5:
            assert (counts == 16).all()
        assert np.array_equal(indices.ravel(), list(GroupSampler(5, 100, 32, fraction, seed=42, batches=100)))
        assert not np.array_equal(indices.ravel(), list(sampler))


@pytest.mark.parametrize("compact", [False, True])
def test_action_alignment_and_cross_phase_filtering(tmp_path, compact):
    episodes = []
    for number, (group, phase, offset) in enumerate(
        (("fan", "both", 1), ("public", "midtrain", 0), ("public", "posttrain", 0))
    ):
        state = np.zeros((4, 7), np.float32)
        state[:, 0] = np.arange(4) * 0.01 + number
        state[:, 6] = 0.5
        action = state.copy() if offset else np.concatenate([state[1:], state[-1:]])
        table = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": np.arange(4) / 20,
                "task_index": [0] * 4,
            }
        )
        audit_arrays(table, 4, offset)
        with pytest.raises(ValueError, match="timing"):
            audit_arrays(table, 4, 1 - offset)
        numeric = f"{number}.npz"
        np.savez(tmp_path / numeric, state=state, action=action, timestamp=np.arange(4) / 20, task_index=np.zeros(4))
        episodes.append(
            {
                "group": group,
                "phase": phase,
                "source": str(number),
                "episode": 0,
                "length": 4,
                "numeric": numeric,
                "action_offset": offset,
                "tasks": {"0": f"task {number}"},
            }
        )
    if compact:
        for episode in episodes:
            directory = tmp_path / (episode["source"] + "_arrays")
            directory.mkdir()
            with np.load(tmp_path / episode["numeric"]) as source:
                for name in source.files:
                    values = source[name]
                    padded = np.concatenate([np.zeros_like(values[:2]), values])
                    np.save(directory / f"{name}.npy", padded)
            episode["numeric"] = directory.name
            episode["array_start"] = 2
    index = tmp_path / "episodes.json"
    index.write_text(json.dumps(episodes))
    (tmp_path / "ready.json").write_text(
        json.dumps({"ready": True, "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()})
    )
    transform = FastUMIRelativeInputs(prompt=None)
    for phase, public_id in (("midtrain", "1"), ("posttrain", "2")):
        ds = FastUMIEpisodes(tmp_path, phase, horizon=3, numeric_only=True)
        assert len(ds) == 8
        assert ds.fan_frames == ds.public_frames == 4
        assert ds.get_sample_context(4)["repo_id"] == public_id
        for frame in (0, 4):
            sample = transform(ds[frame])
            np.testing.assert_allclose(sample["actions"][:, 0], [0.01, 0.02, 0.03], atol=1e-6)
            assert sample["state"].shape == (1,)
        assert transform(ds[4])["prompt"] == f"task {public_id}"
        from openpi import transforms  # noqa: PLC0415
        from openpi.training.config import DataConfig  # noqa: PLC0415
        from openpi.training.data_loader import transform_dataset  # noqa: PLC0415

        data_config = DataConfig(
            repo_id="fastumi/two_phase", fastumi_phase=phase, data_transforms=transforms.Group(inputs=[transform])
        )
        processed = transform_dataset(ds, data_config, skip_norm_stats=True)[4]
        assert processed["prompt"] == f"task {public_id}"
        assert processed["actions"].shape == (3, 10)

        np.testing.assert_allclose(transform(ds[7])["actions"][:, :3], 0, atol=1e-6)
    index.write_text("[]")
    with pytest.raises(ValueError, match="audited"):
        FastUMIEpisodes(tmp_path, "midtrain")


def test_phase_configs_and_schedules():
    from openpi.training.config import get_config  # noqa: PLC0415

    mid = get_config("pi05_fastumi_midtrain")
    post = get_config("pi05_fastumi_posttrain")
    assert mid.num_train_steps == 200000
    assert post.num_train_steps == 100000
    assert mid.data.fan_fraction == 0.1
    assert post.data.fan_fraction == 0.5
    assert mid.data.assets == post.data.assets
    assert mid.weight_loader.params_path.endswith("pi05_base/params")
    assert post.weight_loader.params_path.endswith("umi_pi05_run4_midtrain/199999/params")
    for cfg, peak, end in ((mid, 2.5e-5, 2.5e-6), (post, 1e-5, 1e-6)):
        lr = cfg.lr_schedule.create()
        assert float(lr(1000)) == pytest.approx(peak)
        assert float(lr(cfg.num_train_steps)) == pytest.approx(end)


def test_norm_does_not_read_posttraining_public_samples(tmp_path):
    from examples.fastumi.compute_mid_post_norm import main as compute_norm  # noqa: PLC0415

    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(4)
    episodes = []
    for i, (group, phase, offset) in enumerate(
        (("fan", "both", 1), ("public", "midtrain", 0), ("public", "posttrain", 0))
    ):
        state = rng.normal(0, 0.15, (32, 7)).astype(np.float32)
        state[:, 6] = rng.uniform(0.2, 0.8, 32)
        action = state.copy() if offset else np.concatenate([state[1:], state[-1:]])
        path = f"{i}.npz"
        np.savez(root / path, state=state, action=action, timestamp=np.arange(32) / 20, task_index=np.zeros(32))
        episodes.append(
            {
                "group": group,
                "phase": phase,
                "source": str(i),
                "episode": 0,
                "length": 32,
                "numeric": path,
                "action_offset": offset,
                "tasks": {"0": "test"},
            }
        )
    index = root / "episodes.json"
    index.write_text(json.dumps(episodes))
    (root / "ready.json").write_text(
        json.dumps({"ready": True, "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()})
    )
    compute_norm(root, samples=1024, output=tmp_path / "first")
    (root / "2.npz").unlink()  # Midtraining stats must not open reserved public data at all.
    compute_norm(root, samples=1024, output=tmp_path / "second")
    assert (tmp_path / "first/norm_stats.json").read_bytes() == (tmp_path / "second/norm_stats.json").read_bytes()


@pytest.mark.parametrize("phase", ["midtrain", "posttrain"])
def test_actual_phase_launch_cli_requires_no_unused_dataset_ids(monkeypatch, phase):
    import sys  # noqa: PLC0415
    from openpi.training import config  # noqa: PLC0415

    monkeypatch.setattr(
        sys, "argv", ["train.py", f"pi05_fastumi_{phase}", f"--exp-name=umi_pi05_run4_{phase}", "--fsdp-devices=2"]
    )
    cfg = config.cli()
    assert cfg.data.repo_ids is None
    assert cfg.data.phase == phase
    assert cfg.fsdp_devices == 2

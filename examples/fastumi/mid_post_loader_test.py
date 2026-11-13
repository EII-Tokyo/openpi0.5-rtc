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


def test_action_alignment_and_cross_phase_filtering(tmp_path):
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

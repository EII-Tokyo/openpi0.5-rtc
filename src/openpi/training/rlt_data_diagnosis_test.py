import json

import numpy as np
import pytest

from openpi.training import rlt_data_diagnosis as diagnosis


def _write_shard(
    path,
    *,
    reward: int,
    n: int,
    z_value: float,
    action_value: float,
    horizon: int = 2,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_seq = np.zeros((n, horizon), dtype=np.float32)
    reward_seq[-1, -1] = float(reward)
    done = np.zeros((n,), dtype=np.bool_)
    done[-1] = True
    z_rl = np.full((n, 4), z_value, dtype=np.float32)
    proprio = np.full((n, 3), z_value * 0.1, dtype=np.float32)
    action = np.full((n, horizon, 2), action_value, dtype=np.float32)
    np.savez(
        path,
        z_rl=z_rl,
        proprio=proprio,
        action=action,
        reference_action=np.zeros_like(action),
        reward_seq=reward_seq,
        next_z_rl=z_rl + 0.01,
        next_proprio=proprio + 0.01,
        next_reference_action=np.zeros_like(action),
        done=done,
        manifest=json.dumps({"key_region_id": path.stem, "reward": reward}),
    )


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_episode_summaries_and_flags_short_slices(tmp_path):
    success = tmp_path / "2026-06-19" / "shards" / "success.npz"
    failure = tmp_path / "2026-06-19" / "shards" / "failure.npz"
    _write_shard(success, reward=1, n=8, z_value=1.0, action_value=0.2)
    _write_shard(failure, reward=0, n=2, z_value=1.1, action_value=0.25)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"shard_path": str(success), "batch": "2026-06-19", "reward": 1},
            {"shard_path": str(failure), "batch": "2026-06-19", "reward": 0},
        ],
    )

    episodes = diagnosis.load_episode_summaries(manifest, sources={"2026-06-19"}, min_transitions=3)
    slice_rows = diagnosis.slice_audit_rows(episodes, min_transitions=3)

    assert len(episodes) == 2
    short_row = next(row for row in slice_rows if row["episode_id"] == "failure")
    assert short_row["suspected_issue"] == "too_few_transitions"
    assert short_row["num_transitions"] == 2


def test_similarity_finds_nearest_success_and_hard_negative(tmp_path):
    success_a = tmp_path / "2026-06-22" / "shards" / "success_a.npz"
    success_b = tmp_path / "2026-06-22" / "shards" / "success_b.npz"
    hard_failure = tmp_path / "2026-06-22" / "shards" / "hard_failure.npz"
    easy_failure = tmp_path / "2026-06-22" / "shards" / "easy_failure.npz"
    _write_shard(success_a, reward=1, n=5, z_value=1.0, action_value=0.2)
    _write_shard(success_b, reward=1, n=5, z_value=3.0, action_value=0.5)
    _write_shard(hard_failure, reward=0, n=5, z_value=1.02, action_value=0.22)
    _write_shard(easy_failure, reward=0, n=5, z_value=9.0, action_value=1.0)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"shard_path": str(success_a), "batch": "2026-06-22"},
            {"shard_path": str(success_b), "batch": "2026-06-22"},
            {"shard_path": str(hard_failure), "batch": "2026-06-22"},
            {"shard_path": str(easy_failure), "batch": "2026-06-22"},
        ],
    )

    episodes = diagnosis.load_episode_summaries(manifest, sources={"2026-06-22"}, min_transitions=3)
    similarity = diagnosis.nearest_success_rows(episodes)
    candidates = diagnosis.hard_negative_rows(similarity, max_distance=0.2)

    hard_row = next(row for row in similarity if row["failure_episode_id"] == "hard_failure")
    easy_row = next(row for row in similarity if row["failure_episode_id"] == "easy_failure")
    assert hard_row["nearest_success_episode_id"] == "success_a"
    assert hard_row["combined_distance"] < easy_row["combined_distance"]
    assert [row["failure_episode_id"] for row in candidates] == ["hard_failure"]
    assert candidates[0]["recommended_use"] == "hard_negative"


def test_similarity_prefers_failure_with_matching_trajectory(tmp_path):
    success = tmp_path / "2026-06-22" / "shards" / "success.npz"
    terminal_only_failure = tmp_path / "2026-06-22" / "shards" / "terminal_only_failure.npz"
    trajectory_failure = tmp_path / "2026-06-22" / "shards" / "trajectory_failure.npz"
    _write_shard(success, reward=1, n=6, z_value=1.0, action_value=0.2)
    _write_shard(terminal_only_failure, reward=0, n=6, z_value=9.0, action_value=0.2)
    _write_shard(trajectory_failure, reward=0, n=6, z_value=1.05, action_value=0.22)
    with np.load(terminal_only_failure) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["z_rl"] = np.linspace(9.0, 1.0, 6, dtype=np.float32)[:, None].repeat(4, axis=1)
    arrays["proprio"] = np.linspace(0.9, 0.1, 6, dtype=np.float32)[:, None].repeat(3, axis=1)
    np.savez(terminal_only_failure, **arrays)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"shard_path": str(success), "batch": "2026-06-22"},
            {"shard_path": str(terminal_only_failure), "batch": "2026-06-22"},
            {"shard_path": str(trajectory_failure), "batch": "2026-06-22"},
        ],
    )

    episodes = diagnosis.load_episode_summaries(manifest, sources={"2026-06-22"}, min_transitions=3)
    similarity = diagnosis.nearest_success_rows(episodes)

    assert [row["failure_episode_id"] for row in similarity] == ["trajectory_failure", "terminal_only_failure"]
    trajectory_row = similarity[0]
    terminal_only_row = similarity[1]
    assert trajectory_row["trajectory_distance"] < terminal_only_row["trajectory_distance"]


def test_hard_negative_rows_include_datetime_fields():
    rows = [
        {
            "source": "2026-06-22",
            "failure_episode_id": "key_region_failure.crop_1782100029243",
            "nearest_success_episode_id": "key_region_success.crop_1782099362718",
            "combined_distance": 0.1,
        }
    ]

    candidates = diagnosis.hard_negative_rows(rows, max_distance=0.2)

    assert candidates[0]["failure_datetime"] == "2026-06-22 12:47:09"
    assert candidates[0]["nearest_success_datetime"] == "2026-06-22 12:36:02"


def test_source_stats_exposes_success_failure_balance(tmp_path):
    rows = []
    for index, reward in enumerate([1, 1, 0]):
        shard = tmp_path / "2026-06-19" / "shards" / f"ep_{index}.npz"
        _write_shard(shard, reward=reward, n=5, z_value=float(index), action_value=0.1)
        rows.append({"shard_path": str(shard), "batch": "2026-06-19"})
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)

    episodes = diagnosis.load_episode_summaries(manifest, sources={"2026-06-19"}, min_transitions=3)
    stats = diagnosis.source_stats(episodes)

    assert stats["2026-06-19"]["success_episodes"] == 2
    assert stats["2026-06-19"]["failure_episodes"] == 1
    assert stats["2026-06-19"]["success_rate"] == pytest.approx(2 / 3)

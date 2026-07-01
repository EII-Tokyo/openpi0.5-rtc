import json

import numpy as np

from openpi.training import rlt_eval


def test_auc_rank_handles_ties():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.5, 0.5, 0.1])

    assert rlt_eval.auc_rank(labels, scores) == 0.875


def test_split_shards_is_reproducible_and_disjoint(tmp_path):
    paths = []
    for index in range(10):
        path = tmp_path / f"shard_{index}.npz"
        path.write_bytes(b"")
        paths.append(path)

    first = rlt_eval.split_shards(paths, holdout_ratio=0.2, seed=7)
    second = rlt_eval.split_shards(paths, holdout_ratio=0.2, seed=7)

    assert first == second
    assert len(first.train_paths) == 8
    assert len(first.holdout_paths) == 2
    assert set(first.train_paths).isdisjoint(first.holdout_paths)


def test_write_manifest_round_trips_paths(tmp_path):
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    manifest_path = tmp_path / "manifest.jsonl"

    rlt_eval.write_manifest(paths, manifest_path)

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [row["shard_path"] for row in rows] == [str(path.resolve()) for path in paths]


def test_critic_usability_flags_bad_failure_advantage():
    decision = rlt_eval.judge_critic_usability(
        success_q_mean=0.6,
        failure_q_mean=0.2,
        auc=0.9,
        success_actor_advantage_mean=0.1,
        failure_actor_advantage_mean=0.2,
        q_gap_stability_warning=False,
    )

    assert not decision.is_critic_usable
    assert "failure_actor_advantage" in decision.warning_reason


def test_critic_usability_accepts_separated_holdout():
    decision = rlt_eval.judge_critic_usability(
        success_q_mean=0.7,
        failure_q_mean=0.1,
        auc=0.82,
        success_actor_advantage_mean=0.2,
        failure_actor_advantage_mean=0.1,
        q_gap_stability_warning=False,
    )

    assert decision.is_critic_usable
    assert decision.warning_reason == ""


def test_actor_usability_rejects_failure_advantage_above_success():
    decision = rlt_eval.judge_actor_usability(
        actor_advantage_mean=0.05,
        success_actor_advantage_mean=0.02,
        failure_actor_advantage_mean=0.07,
        actor_delta_norm=0.03,
        min_q_advantage=0.0,
        max_delta_norm=0.09,
    )

    assert not decision.is_actor_usable
    assert "failure_actor_advantage>success_actor_advantage" in decision.warning_reason


def test_best_actor_metric_prefers_safe_actor_over_high_auc_critic_metric():
    unsafe_high_auc = {
        "step": 2000,
        "auc": 0.90,
        "q_gap": 0.30,
        "holdout_bellman_loss": 0.01,
        "actor_advantage_mean": 0.08,
        "success_actor_advantage_mean": 0.01,
        "failure_actor_advantage_mean": 0.09,
        "actor_delta_norm": 0.03,
    }
    safe_lower_auc = {
        "step": 1000,
        "auc": 0.72,
        "q_gap": 0.10,
        "holdout_bellman_loss": 0.02,
        "actor_advantage_mean": 0.04,
        "success_actor_advantage_mean": 0.08,
        "failure_actor_advantage_mean": 0.01,
        "actor_delta_norm": 0.02,
    }

    best = rlt_eval.best_actor_metric([unsafe_high_auc, safe_lower_auc])

    assert best is safe_lower_auc

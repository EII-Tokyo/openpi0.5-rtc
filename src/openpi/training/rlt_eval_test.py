import json
import csv

import numpy as np
import pytest

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


def test_best_checkpoint_metric_prefers_q_propagation_over_auc():
    high_auc_no_propagation = {
        "step": 1000,
        "auc": 0.95,
        "q_gap": 0.05,
        "holdout_bellman_loss": 0.01,
        "success_actor_advantage_mean": 0.04,
        "failure_actor_advantage_mean": 0.01,
        "q_propagation_score": 0.01,
    }
    lower_auc_with_propagation = {
        "step": 2000,
        "auc": 0.70,
        "q_gap": 0.02,
        "holdout_bellman_loss": 0.03,
        "success_actor_advantage_mean": 0.03,
        "failure_actor_advantage_mean": 0.01,
        "q_propagation_score": 0.40,
    }

    best = rlt_eval.best_checkpoint_metric([high_auc_no_propagation, lower_auc_with_propagation])

    assert best is lower_auc_with_propagation


def test_attach_q_propagation_metrics_measures_success_early_lift_without_failure_lift():
    metrics = [
        {"step": 1000, "auc": 0.90},
        {"step": 2000, "auc": 0.70},
    ]
    rows = []
    for step, success_lift, failure_lift in [(1000, 0.0, 0.0), (2000, 0.5, 0.4)]:
        for transition_index, progress in enumerate([0.0, 0.33, 0.67, 1.0]):
            rows.append(
                {
                    "checkpoint_step": step,
                    "episode_id": "success",
                    "shard_path": "/tmp/success.npz",
                    "label": 1,
                    "transition_index": transition_index,
                    "progress": progress,
                    "predicted_q": success_lift if progress <= 0.5 else success_lift + 0.2,
                }
            )
            rows.append(
                {
                    "checkpoint_step": step,
                    "episode_id": "failure",
                    "shard_path": "/tmp/failure.npz",
                    "label": 0,
                    "transition_index": transition_index,
                    "progress": progress,
                    "predicted_q": failure_lift,
                }
            )

    rlt_eval.attach_q_propagation_metrics(metrics, rows)

    assert metrics[0]["q_success_early_lift"] == pytest.approx(0.0)
    assert metrics[1]["q_success_early_lift"] == pytest.approx(0.5)
    assert metrics[1]["q_failure_early_lift"] == pytest.approx(0.4)
    assert metrics[1]["q_propagation_score"] == pytest.approx(0.1)


def test_apply_z_rl_normalization_uses_checkpoint_metadata_without_touching_other_arrays():
    arrays = {
        "z_rl": np.asarray([[1.0, 3.0]], dtype=np.float32),
        "next_z_rl": np.asarray([[2.0, 5.0]], dtype=np.float32),
        "proprio": np.asarray([[7.0, 8.0]], dtype=np.float32),
    }
    metadata = {"z_rl_normalization": {"mean": [1.0, 1.0], "std": [1.0, 2.0]}}

    normalized = rlt_eval.apply_z_rl_normalization(arrays, metadata)

    np.testing.assert_allclose(normalized["z_rl"], np.asarray([[0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized["next_z_rl"], np.asarray([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized["proprio"], arrays["proprio"])
    np.testing.assert_allclose(arrays["z_rl"], np.asarray([[1.0, 3.0]], dtype=np.float32))


def test_summarize_checkpoint_reports_calql_calibration_metrics(tmp_path):
    rows = [
        {
            "label": 1,
            "predicted_q": 0.9,
            "target_q": 0.8,
            "bellman_error": 0.01,
            "actor_q": 0.95,
            "reference_q": 0.7,
            "actor_advantage": 0.25,
            "actor_delta_norm": 0.02,
            "reference_value": 0.6,
        },
        {
            "label": 0,
            "predicted_q": 0.2,
            "target_q": 0.1,
            "bellman_error": 0.01,
            "actor_q": 0.25,
            "reference_q": 0.3,
            "actor_advantage": -0.05,
            "actor_delta_norm": 0.01,
            "reference_value": 0.4,
        },
    ]

    metric = rlt_eval.summarize_checkpoint(
        checkpoint_dir=tmp_path / "00001000",
        metadata={"step": 1000},
        rows=rows,
        train_critic_loss=None,
    )

    assert metric["reference_value_mean"] == 0.5
    assert metric["calibration_margin_mean"] == pytest.approx(0.0)
    assert metric["floor_violation_rate"] == 0.5


def test_write_transition_reports_includes_per_shard_summary(tmp_path):
    rows = [
        {
            "checkpoint_step": 1000,
            "shard_path": "/tmp/a.npz",
            "episode_id": "a",
            "label": 1,
            "transition_index": 0,
            "num_transitions": 2,
            "progress": 0.0,
            "done": False,
            "predicted_q": 0.2,
            "target_q": 0.1,
            "actor_advantage": 0.01,
            "actor_delta_norm": 0.02,
            "bellman_error": 0.01,
            "floor_violation": 0.0,
        },
        {
            "checkpoint_step": 1000,
            "shard_path": "/tmp/a.npz",
            "episode_id": "a",
            "label": 1,
            "transition_index": 1,
            "num_transitions": 2,
            "progress": 1.0,
            "done": True,
            "predicted_q": 0.4,
            "target_q": 1.0,
            "actor_advantage": 0.03,
            "actor_delta_norm": 0.04,
            "bellman_error": 0.36,
            "floor_violation": 0.0,
        },
        {
            "checkpoint_step": 1000,
            "shard_path": "/tmp/b.npz",
            "episode_id": "b",
            "label": 0,
            "transition_index": 0,
            "num_transitions": 1,
            "progress": 1.0,
            "done": True,
            "predicted_q": 0.8,
            "target_q": 0.0,
            "actor_advantage": -0.01,
            "actor_delta_norm": 0.0,
            "bellman_error": 0.64,
            "floor_violation": 1.0,
        },
    ]

    rlt_eval.write_transition_reports(rows, tmp_path)

    assert (tmp_path / "critic_holdout_transitions.csv").exists()
    shard_rows = list(csv.DictReader((tmp_path / "critic_holdout_shards.csv").open()))
    assert len(shard_rows) == 2
    by_episode = {row["episode_id"]: row for row in shard_rows}
    assert float(by_episode["a"]["predicted_q_mean"]) == pytest.approx(0.3)
    assert float(by_episode["b"]["predicted_q_mean"]) == pytest.approx(0.8)

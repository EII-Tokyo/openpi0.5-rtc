import json

from scripts import summarize_rlt_eval_compare


def _write_summary(path, payload):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_eval_compare_sorts_by_selection_score_and_renders_markdown(tmp_path):
    _write_summary(
        tmp_path / "evals" / "low_run",
        {
            "actor_dir": "/checkpoints/actor_low/00001000",
            "actor_step": 1000,
            "scores_terminal_transitions": {
                "q_actual": {"auc": 0.2, "gap": 0.1},
                "q_actor": {"auc": 0.3, "gap": 0.2},
            },
            "scores_all_transitions": {"q_actor": {"auc": 0.4}},
            "actor_delta_norm": {"all": {"mean": 0.1, "p95": 0.9}},
            "actor_chunk_smoothness": {"all": {"mean": 0.2, "p95": 0.8}},
        },
    )
    _write_summary(
        tmp_path / "evals" / "nested" / "high_run",
        {
            "actor_dir": "/checkpoints/actor_high/00002000",
            "actor_step": 2000,
            "scores_terminal_transitions": {
                "q_actual": {"auc": 0.9, "mean_gap_success_minus_failure": 0.5},
                "q_actor": {"auc": 0.8, "mean_gap_success_minus_failure": 0.4},
            },
            "scores_all_transitions": {"q_actor": {"auc": 0.7}},
            "actor_delta_norm": {"all": {"mean": 0.05, "p95": 0.1}},
            "actor_chunk_smoothness": {"all": {"mean": 0.06, "p95": 0.2}},
        },
    )

    rows = summarize_rlt_eval_compare.summarize(tmp_path / "evals")
    markdown = summarize_rlt_eval_compare.render_markdown(rows)

    assert [row["目录名"] for row in rows] == ["high_run", "low_run"]
    assert "| 目录名 | actor_dir | actor_step |" in markdown
    assert "| high_run | /checkpoints/actor_high/00002000 | 2000 |" in markdown
    assert "| high_run | /checkpoints/actor_high/00002000 | 2000 | 0.9 | 0.5 | 0.8 | 0.4 |" in markdown
    assert "terminal q_actual auc" in markdown
    assert "selection_score" in markdown

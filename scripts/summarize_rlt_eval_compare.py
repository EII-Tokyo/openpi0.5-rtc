"""Summarize RLT evaluation summary.json files as a Markdown comparison table."""

import argparse
import json
import pathlib


COLUMNS = [
    ("目录名", "目录名"),
    ("actor_dir", "actor_dir"),
    ("actor_step", "actor_step"),
    ("terminal q_actual auc", "terminal_q_actual_auc"),
    ("terminal q_actual gap", "terminal_q_actual_gap"),
    ("terminal q_actor auc", "terminal_q_actor_auc"),
    ("terminal q_actor gap", "terminal_q_actor_gap"),
    ("all q_actor auc", "all_q_actor_auc"),
    ("delta mean", "delta_mean"),
    ("delta p95", "delta_p95"),
    ("smoothness mean", "smoothness_mean"),
    ("smoothness p95", "smoothness_p95"),
    ("selection_score", "selection_score"),
]


def _nested_get(data, keys):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _score_metric(data, score_name, metric):
    value = _nested_get(data, ["scores_terminal_transitions", score_name, metric])
    if value is None and metric == "gap":
        return _nested_get(data, ["scores_terminal_transitions", score_name, "mean_gap_success_minus_failure"])
    return value


def _score_value(value, missing):
    if value is None:
        return missing
    return value


def _selection_score(row):
    terminal_q_actual_auc = _score_value(row["terminal_q_actual_auc"], 0.0)
    terminal_q_actor_auc = _score_value(row["terminal_q_actor_auc"], 0.0)
    delta_p95 = _score_value(row["delta_p95"], 1.0)
    smoothness_p95 = _score_value(row["smoothness_p95"], 1.0)
    return terminal_q_actual_auc * 0.35 + terminal_q_actor_auc * 0.35 - delta_p95 * 0.15 - smoothness_p95 * 0.15


def _row_from_summary(summary_path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "目录名": summary_path.parent.name,
        "actor_dir": data.get("actor_dir"),
        "actor_step": data.get("actor_step"),
        "terminal_q_actual_auc": _score_metric(data, "q_actual", "auc"),
        "terminal_q_actual_gap": _score_metric(data, "q_actual", "gap"),
        "terminal_q_actor_auc": _score_metric(data, "q_actor", "auc"),
        "terminal_q_actor_gap": _score_metric(data, "q_actor", "gap"),
        "all_q_actor_auc": _nested_get(data, ["scores_all_transitions", "q_actor", "auc"]),
        "delta_mean": _nested_get(data, ["actor_delta_norm", "all", "mean"]),
        "delta_p95": _nested_get(data, ["actor_delta_norm", "all", "p95"]),
        "smoothness_mean": _nested_get(data, ["actor_chunk_smoothness", "all", "mean"]),
        "smoothness_p95": _nested_get(data, ["actor_chunk_smoothness", "all", "p95"]),
    }
    row["selection_score"] = _selection_score(row)
    return row


def summarize(eval_root):
    eval_root = pathlib.Path(eval_root)
    rows = [_row_from_summary(path) for path in sorted(eval_root.rglob("summary.json"))]
    return sorted(rows, key=lambda row: row["selection_score"], reverse=True)


def _format_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_markdown_cell(value):
    return _format_cell(value).replace("|", "\\|")


def render_markdown(rows):
    headers = [header for header, _key in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(row[key]) for _header, key in COLUMNS) + " |")
    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(description="汇总 RLT eval summary.json 并输出 Markdown 对比表。")
    parser.add_argument("--eval-root", required=True, type=pathlib.Path, help="递归查找 summary.json 的根目录。")
    parser.add_argument("--output-md", type=pathlib.Path, help="可选：写入 Markdown 文件路径。")
    return parser.parse_args()


def main():
    args = _parse_args()
    markdown = render_markdown(summarize(args.eval_root))
    print(markdown)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

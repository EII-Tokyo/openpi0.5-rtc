from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
from typing import Any

import tyro


@dataclasses.dataclass
class Args:
    output_dir: pathlib.Path
    title: str = "RLT Critic Holdout Comparison"
    run_spec: str | None = None
    metrics_csv: list[pathlib.Path] = dataclasses.field(default_factory=list)
    label: list[str] = dataclasses.field(default_factory=list)


def _read_rows(path: pathlib.Path, label: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["run_label"] = label
        row["step"] = int(float(row["step"]))
        for key, value in list(row.items()):
            if key in {"run_label", "checkpoint_path", "warning_reason", "actor_warning_reason"}:
                continue
            row[key] = _to_float(value)
    return rows


def _parse_run_specs(args: Args) -> list[tuple[str, pathlib.Path]]:
    if args.run_spec:
        pairs = []
        for item in args.run_spec.split(","):
            if "=" not in item:
                raise ValueError(f"run_spec item must be LABEL=CSV_PATH, got: {item!r}")
            label, path = item.split("=", 1)
            label = label.strip()
            path = path.strip()
            if not label or not path:
                raise ValueError(f"run_spec item must be LABEL=CSV_PATH, got: {item!r}")
            pairs.append((label, pathlib.Path(path)))
        return pairs

    if len(args.metrics_csv) != len(args.label):
        raise ValueError("metrics_csv and label must have the same length")
    return list(zip(args.label, args.metrics_csv, strict=True))


def _to_float(value: Any) -> Any:
    if value in (None, ""):
        return math.nan
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            _finite_or(row.get("q_propagation_score"), -1e9),
            _finite_or(row.get("q_gap"), -1e9),
            -_finite_or(row.get("floor_violation_rate"), 1e9),
            -_finite_or(row.get("holdout_bellman_loss"), 1e9),
            _finite_or(row.get("auc"), -1.0),
        ),
    )


def _finite_or(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def _write_summary(runs: dict[str, list[dict[str, Any]]], output_dir: pathlib.Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| run | best_step | q_propagation | AUC | q_gap | floor_violation | calibration_margin | Bellman loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = {}
    for label, rows in runs.items():
        best = _best_row(rows)
        summary[label] = best
        lines.append(
            "| {label} | {step} | {prop:.6f} | {auc:.4f} | {q_gap:.6f} | {floor:.4f} | {margin:.6f} | {loss:.6f} |".format(
                label=label,
                step=int(best["step"]),
                prop=float(best.get("q_propagation_score", math.nan)),
                auc=float(best["auc"]),
                q_gap=float(best["q_gap"]),
                floor=float(best.get("floor_violation_rate", math.nan)),
                margin=float(best.get("calibration_margin_mean", math.nan)),
                loss=float(best.get("holdout_bellman_loss", math.nan)),
            )
        )
    lines.extend(
        [
            "",
            "Selection uses q_propagation_score first, then q_gap, lower floor violation, lower Bellman loss, and AUC only as a final tie-breaker.",
            "",
            "For Cal-QL style comparison, the useful pattern is not only higher AUC. Prefer curves where q_gap stays positive, floor violation decreases, and calibration margin does not collapse below zero.",
            "",
        ]
    )
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _plot(runs: dict[str, list[dict[str, Any]]], output_dir: pathlib.Path, title: str) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("auc", "Holdout AUC"),
        ("q_gap", "Success Q - Failure Q"),
        ("floor_violation_rate", "Reference floor violation rate"),
        ("calibration_margin_mean", "Mean reference Q - reference value"),
    ]
    for axis, (field, ylabel) in zip(axes.ravel(), plots, strict=True):
        for label, rows in runs.items():
            ordered = sorted(rows, key=lambda row: int(row["step"]))
            axis.plot([row["step"] for row in ordered], [row.get(field, math.nan) for row in ordered], marker="o", label=label)
        if field == "calibration_margin_mean":
            axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_xlabel("checkpoint step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / "cql_style_comparison_curves.png", dpi=180)
    plt.close(figure)


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {label: _read_rows(path, label) for label, path in _parse_run_specs(args)}
    _write_summary(runs, args.output_dir, args.title)
    _plot(runs, args.output_dir, args.title)


if __name__ == "__main__":
    main(tyro.cli(Args))

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import shutil
from typing import Any

import tyro

from openpi.training import rlt_data_diagnosis as diagnosis


@dataclasses.dataclass
class Args:
    manifest_path: pathlib.Path = pathlib.Path("local_rlt_manifests/trainable_clean_committed_20260623_time_sorted.jsonl")
    output_dir: pathlib.Path = pathlib.Path("local_rlt_runs/rlt_data_diagnosis_0619_0622_20260624")
    sources: tuple[str, ...] = ("2026-06-19", "2026-06-22")
    min_transitions: int = 3
    hard_negative_max_distance: float = 0.20
    overwrite: bool = False


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: pathlib.Path,
    *,
    args: Args,
    stats: dict[str, dict[str, Any]],
    label_rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
) -> None:
    label_issues = [row for row in label_rows if row["suspected_issue"]]
    slice_issues = [row for row in slice_rows if row["suspected_issue"]]
    lines = [
        "# RLT Data Diagnosis 06-19 / 06-22",
        "",
        "## Setup",
        "",
        f"- manifest: `{args.manifest_path}`",
        f"- sources: `{', '.join(args.sources)}`",
        f"- min_transitions: `{args.min_transitions}`",
        f"- hard_negative_max_distance: `{args.hard_negative_max_distance}`",
        "",
        "## Source Summary",
        "",
        "| source | episodes | success | failure | success rate | length mean | length min | length p10 | length p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, row in stats.items():
        lines.append(
            f"| {source} | {row['episodes']} | {row['success_episodes']} | {row['failure_episodes']} | "
            f"{row['success_rate']:.3f} | {row['length_mean']:.2f} | {row['length_min']} | "
            f"{row['length_p10']:.2f} | {row['length_p90']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Audit Summary",
            "",
            f"- label metadata issues: `{len(label_issues)}`",
            f"- slice length issues: `{len(slice_issues)}`",
            f"- nearest-success rows: `{len(similarity_rows)}`",
            f"- hard-negative candidates: `{len(hard_negative_rows)}`",
            "",
            "## Top Hard-Negative Candidates",
            "",
            "| source | failure | nearest success | distance | terminal | trajectory | z traj | proprio traj | action traj |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hard_negative_rows[:30]:
        lines.append(
            f"| {row['source']} | {row['failure_episode_id']} | {row['nearest_success_episode_id']} | "
            f"{float(row['combined_distance']):.4f} | {float(row['terminal_distance']):.4f} | "
            f"{float(row['trajectory_distance']):.4f} | {float(row['z_trajectory_distance']):.4f} | "
            f"{float(row['proprio_trajectory_distance']):.4f} | {float(row['action_trajectory_distance']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Closest Failure To Success By Source",
            "",
            "| source | failure | nearest success | distance | recommended |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    seen_sources: set[str] = set()
    for row in sorted(similarity_rows, key=lambda item: (item["source"], item["combined_distance"])):
        if row["source"] in seen_sources:
            continue
        seen_sources.add(row["source"])
        recommended = "yes" if float(row["combined_distance"]) <= args.hard_negative_max_distance else "no"
        lines.append(
            f"| {row['source']} | {row['failure_episode_id']} | {row['nearest_success_episode_id']} | "
            f"{float(row['combined_distance']):.4f} | {recommended} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `label_audit.csv` only checks metadata consistency; visual success/failure still needs human review for flagged candidates.",
            "- `slice_audit.csv` flags clearly suspicious shard lengths, not semantic crop quality.",
            "- `hard_negative_candidates.csv` ranks failures close to successes in resampled RL embedding/proprio/action trajectories, with terminal distance as a secondary signal.",
            "- If few or no candidates appear, the current failures are probably too far from successes to teach action-sensitive critic behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: Args) -> None:
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = diagnosis.load_episode_summaries(
        args.manifest_path,
        sources=set(args.sources),
        min_transitions=args.min_transitions,
    )
    stats = diagnosis.source_stats(episodes)
    label_rows = diagnosis.label_audit_rows(episodes)
    slice_rows = diagnosis.slice_audit_rows(episodes, min_transitions=args.min_transitions)
    similarity_rows = diagnosis.nearest_success_rows(episodes)
    hard_negative_rows = diagnosis.hard_negative_rows(
        similarity_rows,
        max_distance=args.hard_negative_max_distance,
    )

    _write_csv(args.output_dir / "label_audit.csv", label_rows)
    _write_csv(args.output_dir / "slice_audit.csv", slice_rows)
    _write_csv(args.output_dir / "episode_similarity.csv", similarity_rows)
    _write_csv(args.output_dir / "hard_negative_candidates.csv", hard_negative_rows)
    (args.output_dir / "source_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(
        args.output_dir / "report.md",
        args=args,
        stats=stats,
        label_rows=label_rows,
        slice_rows=slice_rows,
        similarity_rows=similarity_rows,
        hard_negative_rows=hard_negative_rows,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episodes": len(episodes),
                "hard_negative_candidates": len(hard_negative_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

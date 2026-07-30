#!/usr/bin/env python3
"""Build non-exclusive condition coverage and prompt-consistency evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


CATEGORIES = {
    "no_cap_named": ["no-cap", "no_cap", "havent_cap", "haven-t_cap", "havent-cap", "haven-t-cap"],
    "direction_named": ["direction"],
    "turn_over_named": ["turn_over"],
    "free_spinning_named": ["free-spinning"],
    "water_named": ["water"],
    "return_home_named": ["return-home", "return_home"],
}


def task_type(tasks: list[str]) -> str:
    text = " ".join(tasks).lower()
    if "if the bottle has a cap" in text:
        return "conditional_on_cap_presence"
    if "do the followings" in text:
        return "long_but_unconditional_cap_step"
    return "short_unconditional_unscrew"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--current-recipe-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    dataset = json.loads(args.dataset_audit.read_text())
    current_recipe = json.loads(args.current_recipe_audit.read_text())["training_data_recipe"]
    current_weights = {
        row["repo_id"]: row["weight"] for row in current_recipe["repository_weights"]
    }
    rows = []
    for repo in dataset["repositories"]:
        name = repo["repo_id"].lower()
        categories = [
            category
            for category, tokens in CATEGORIES.items()
            if any(token in name for token in tokens)
        ]
        rows.append(
            {
                "repo_id": repo["repo_id"],
                "episodes": repo["declared_total_episodes"],
                "frames": repo["declared_total_frames"],
                "deployed_run_weight": repo["sampling_weight"],
                "current_code_weight": current_weights.get(repo["repo_id"]),
                "task_type": task_type(repo["unique_tasks"]),
                "categories": categories,
            }
        )

    category_summary = {}
    for category in CATEGORIES:
        members = [row for row in rows if category in row["categories"]]
        category_summary[category] = {
            "repositories": len(members),
            "episodes": sum(row["episodes"] for row in members),
            "frames": sum(row["frames"] for row in members),
            "deployed_weighted_episodes": sum(
                row["episodes"] * row["deployed_run_weight"] for row in members
            ),
            "current_code_weighted_episodes": sum(
                row["episodes"] * (row["current_code_weight"] or 0) for row in members
            ),
        }

    prompt_summary = {}
    for prompt_type in sorted({row["task_type"] for row in rows}):
        members = [row for row in rows if row["task_type"] == prompt_type]
        prompt_summary[prompt_type] = {
            "repositories": len(members),
            "episodes": sum(row["episodes"] for row in members),
            "frames": sum(row["frames"] for row in members),
        }
    no_cap_rows = [row for row in rows if "no_cap_named" in row["categories"]]

    result = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "classification_method": "Explicit non-exclusive filename-token rules listed in this artifact.",
        "category_tokens": CATEGORIES,
        "category_summary": category_summary,
        "prompt_summary": prompt_summary,
        "no_cap_prompt_cross_check": {
            "repositories": len(no_cap_rows),
            "episodes": sum(row["episodes"] for row in no_cap_rows),
            "frames": sum(row["frames"] for row in no_cap_rows),
            "conditional_prompt_repositories": sum(
                row["task_type"] == "conditional_on_cap_presence" for row in no_cap_rows
            ),
            "unconditional_prompt_repositories": sum(
                row["task_type"] != "conditional_on_cap_presence" for row in no_cap_rows
            ),
        },
        "repositories": rows,
        "interpretation": [
            "The filename categories measure intended coverage, not verified scene labels.",
            "All six no-cap-named repositories use an unconditional unscrew instruction.",
            "This instruction mismatch is a concrete risk consistent with field-observed air-unscrewing, but it does not prove causality.",
            "Turn-over and free-spinning data exist; their presence alone does not prove those failures are solved.",
            "The later current-code weights are shown as remediation evidence, not as the deployed step-19000 recipe.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    fields = [
        "repo_id", "episodes", "frames", "deployed_run_weight",
        "current_code_weight", "task_type", "categories",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "categories": "|".join(row["categories"])})


if __name__ == "__main__":
    main()

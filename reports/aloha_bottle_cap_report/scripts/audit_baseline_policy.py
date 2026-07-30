#!/usr/bin/env python3
"""Read-only audit of the deployed bottle-sorting policy and its data recipe.

The script reads small text metadata files over SSH. It never loads checkpoint
arrays and never writes to the remote machine.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Any


CONFIG_NAME = "eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo"
CHECKPOINT = (
    "/data/openpi0.5-rtc/checkpoints/"
    "eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/"
    "no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000"
)


def ssh_read(host: str, path: str) -> str:
    remote = " ".join(
        [
            "python3",
            "-c",
            shlex.quote("import pathlib,sys;sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())"),
            shlex.quote(path),
        ]
    )
    result = subprocess.run(
        ["ssh", host, remote],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def ssh_command(host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", host, command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


class SafeAssignmentEvaluator:
    """Evaluate only the small literal/list expressions used by the data recipe."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def eval(self, node: ast.AST, local: dict[str, Any] | None = None) -> Any:
        local = local or {}
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            return self.values[node.id]
        if isinstance(node, ast.List):
            values: list[Any] = []
            for item in node.elts:
                value = self.eval(item.value, local) if isinstance(item, ast.Starred) else self.eval(item, local)
                values.extend(value if isinstance(item, ast.Starred) else [value])
            return values
        if isinstance(node, ast.Tuple):
            return tuple(self.eval(item, local) for item in node.elts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self.eval(node.left, local) * self.eval(node.right, local)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 1
        ):
            return range(self.eval(node.args[0], local))
        if isinstance(node, ast.ListComp):
            return self.eval_list_comp(node, local)
        raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")

    def eval_list_comp(self, node: ast.ListComp, local: dict[str, Any]) -> list[Any]:
        rows = [local]
        for generator in node.generators:
            if generator.ifs or generator.is_async or not isinstance(generator.target, ast.Name):
                raise ValueError("unsupported comprehension")
            expanded: list[dict[str, Any]] = []
            for row in rows:
                iterable = self.eval(generator.iter, row)
                for value in iterable:
                    child = dict(row)
                    child[generator.target.id] = value
                    expanded.append(child)
            rows = expanded
        return [self.eval(node.elt, row) for row in rows]


def parse_recipe(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    evaluator = SafeAssignmentEvaluator()
    wanted_prefixes = (
        "_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS",
        "_EII_TURN_OVER_WITHOUT_RINSE_REPO_IDS",
        "_EII_FREE_SPINNING_MERGED_ADJUST_PICKUP_REPO_ID",
        "_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_REPO_IDS",
        "_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS",
    )
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        value = node.value
        if isinstance(target, ast.Name) and target.id.startswith(wanted_prefixes):
            evaluator.values[target.id] = evaluator.eval(value)

    effective = evaluator.values[
        "_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS"
    ]
    base = evaluator.values["_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS"]
    counts = Counter(effective)

    config_call: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "_make_twist_train_config":
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != CONFIG_NAME:
            continue
        config_call["name"] = CONFIG_NAME
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                config_call[kw.arg] = kw.value.value
            elif kw.arg == "repo_ids" and isinstance(kw.value, ast.Name):
                config_call[kw.arg] = kw.value.id
        break

    return {
        "config": config_call,
        "base_entries": len(base),
        "base_unique_repositories": len(set(base)),
        "effective_entries": len(effective),
        "effective_unique_repositories": len(counts),
        "repository_weights": [
            {"repo_id": repo, "weight": count} for repo, count in counts.most_common()
        ],
        "duplicate_note": (
            "effective_entries are sampler entries, not independent datasets; "
            "repetition changes sampling weight"
        ),
    }


def parse_checkpoint(checkpoint_metadata: str, params_metadata: str, norm_stats: str) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_metadata)
    params = json.loads(params_metadata)["tree_metadata"]
    norm = json.loads(norm_stats)
    leaves = []
    total = 0
    top_groups: Counter[str] = Counter()
    for encoded_path, entry in params.items():
        keys = [part["key"] for part in entry["key_metadata"]]
        shape = entry["value_metadata"]["write_shape"]
        count = math.prod(shape)
        total += count
        group = keys[2] if len(keys) > 2 else keys[-1]
        top_groups[group] += count
        leaves.append({"path": "/".join(keys), "shape": shape, "parameter_count": count})

    ns = norm["norm_stats"]
    norm_summary = {}
    for key, value in ns.items():
        dimensions = {}
        for field, numbers in value.items():
            if isinstance(numbers, list):
                dimensions[field] = len(numbers)
        norm_summary[key] = dimensions

    init_ns = checkpoint.get("init_timestamp_nsecs")
    commit_ns = checkpoint.get("commit_timestamp_nsecs")
    return {
        "format": "Orbax OCDBT params-only checkpoint",
        "checkpoint_path": CHECKPOINT,
        "directory_step": 19000,
        "metadata_metrics": checkpoint.get("metrics"),
        "custom_metadata": checkpoint.get("custom_metadata"),
        "init_time_utc": datetime.fromtimestamp(init_ns / 1e9, timezone.utc).isoformat() if init_ns else None,
        "commit_time_utc": datetime.fromtimestamp(commit_ns / 1e9, timezone.utc).isoformat() if commit_ns else None,
        "parameter_leaf_count": len(leaves),
        "total_parameter_count": total,
        "trainable_parameter_count": None,
        "trainable_parameter_count_reason": (
            "The params-only checkpoint has no optimizer or trainability mask; "
            "full-finetune config is supporting evidence but not a per-leaf proof."
        ),
        "optimizer_state_present": False,
        "ema_present": False,
        "top_parameter_groups": dict(top_groups),
        "normalization_fields": norm_summary,
        "parameter_leaves": leaves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="aloha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = "/home/eii/openpi0.5-rlt/src/openpi/training/config.py"
    source = ssh_read(args.host, config_path)
    recipe = parse_recipe(source)
    checkpoint = parse_checkpoint(
        ssh_read(args.host, f"{CHECKPOINT}/_CHECKPOINT_METADATA"),
        ssh_read(args.host, f"{CHECKPOINT}/params/_METADATA"),
        ssh_read(args.host, f"{CHECKPOINT}/assets/trossen/norm_stats.json"),
    )
    checkpoint["remote_size_bytes"] = int(ssh_command(args.host, f"du -sb {CHECKPOINT} | cut -f1"))
    checkpoint["can_read_metadata"] = True
    checkpoint["full_array_load_tested"] = False

    result = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "remote_host_alias": args.host,
        "remote_repository": {
            "path": "/home/eii/openpi0.5-rlt",
            "branch": ssh_command(args.host, "git -C /home/eii/openpi0.5-rlt branch --show-current"),
            "commit": ssh_command(args.host, "git -C /home/eii/openpi0.5-rlt rev-parse HEAD"),
        },
        "deployed_baseline_checkpoint": checkpoint,
        "training_data_recipe": {
            **recipe,
            "evidence_kind": "current remote worktree configuration at audit time",
            "historical_run_warning": (
                "This is not substituted for the W&B run configuration. "
                "The deployed step-19000 checkpoint belongs to an earlier run recipe."
            ),
        },
        "evidence_scope": {
            "supports": [
                "checkpoint existence and directory step",
                "parameter structure and total count",
                "normalization dimensions",
                "effective data-repository sampling weights",
            ],
            "does_not_support": [
                "the historical sampling weights used before this current code revision",
                "formal robot success rate",
                "training duration",
                "best checkpoint ranking",
                "per-dataset episode/frame counts",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

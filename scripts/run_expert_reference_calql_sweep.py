from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import pathlib
import random
import subprocess
import sys

import tyro


@dataclasses.dataclass(frozen=True)
class SweepSpec:
    label: str
    critic_loss_mode: str
    conservative_alpha: float
    include_expert: bool = False
    expert_train_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.expert_train_fraction is None:
            object.__setattr__(self, "expert_train_fraction", 1.0 if self.include_expert else 0.0)
        elif self.expert_train_fraction > 0.0:
            object.__setattr__(self, "include_expert", True)


@dataclasses.dataclass(frozen=True)
class PreparedManifests:
    train_manifest: pathlib.Path
    holdout_manifest: pathlib.Path
    current_holdout_manifest: pathlib.Path | None = None


@dataclasses.dataclass
class Args:
    output_root: pathlib.Path
    current_train_manifest_path: pathlib.Path
    current_holdout_manifest_path: pathlib.Path
    expert_manifest_path: pathlib.Path
    replay_dir: pathlib.Path = pathlib.Path("/app/replay/rlt_key_regions")
    calql_alpha: float = 0.1
    sweep_mode: str = "basic"
    expert_train_fraction: tuple[float, ...] = (0.25, 0.5, 1.0)
    calql_alpha_grid: tuple[float, ...] = (0.03, 0.1)
    expert_holdout_ratio: float = 0.2
    expert_split_seed: int = 42
    manifest_rewrite_from: str = "/data/openpi0.5-rtc-reward-learning/replay"
    manifest_rewrite_to: str = "/app/replay"
    legacy_manifest_rewrite_from: str = "/home/eii/data/openpi0.5-rtc-reward-learning/replay"
    num_train_steps: int = 10_000
    batch_size: int = 64
    seed: int = 11
    critic_lr: float = 3e-4
    log_interval: int = 100
    save_interval: int = 1_000
    score_batch_size: int = 512
    expected_replay_action_horizon: int = 10
    train_action_horizon: int = 10
    title: str = "Expert reference CalQL critic sweep"
    drop_missing_shards: bool = True
    dry_run: bool = False
    skip_existing: bool = False
    eval_current_holdout: bool = False


def build_sweep_specs(args: Args) -> list[SweepSpec]:
    if args.sweep_mode == "fractional_expert":
        specs = [
            SweepSpec(
                label="TD3-current",
                critic_loss_mode="td3",
                conservative_alpha=0.0,
                include_expert=False,
                expert_train_fraction=0.0,
            )
        ]
        specs.extend(
            [
                SweepSpec(
                    label=f"TD3-expert{_fraction_label(fraction)}",
                    critic_loss_mode="td3",
                    conservative_alpha=0.0,
                    include_expert=True,
                    expert_train_fraction=float(fraction),
                )
                for fraction in args.expert_train_fraction
            ]
        )
        for alpha in args.calql_alpha_grid:
            specs.extend(
                [
                    SweepSpec(
                        label=f"CalQL{_alpha_label(alpha)}-expert{_fraction_label(fraction)}",
                        critic_loss_mode="calql",
                        conservative_alpha=float(alpha),
                        include_expert=True,
                        expert_train_fraction=float(fraction),
                    )
                    for fraction in args.expert_train_fraction
                ]
            )
        return specs
    if args.sweep_mode != "basic":
        raise ValueError(f"unknown sweep_mode: {args.sweep_mode}")
    return [
        SweepSpec(label="TD3-current", critic_loss_mode="td3", conservative_alpha=0.0, include_expert=False),
        SweepSpec(label="TD3-current+expert", critic_loss_mode="td3", conservative_alpha=0.0, include_expert=True),
        SweepSpec(
            label="CalQL-current",
            critic_loss_mode="calql",
            conservative_alpha=float(args.calql_alpha),
            include_expert=False,
        ),
        SweepSpec(
            label="CalQL-current+expert",
            critic_loss_mode="calql",
            conservative_alpha=float(args.calql_alpha),
            include_expert=True,
        ),
    ]


def _fraction_label(value: float) -> str:
    return str(round(float(value) * 100))


def _alpha_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _slug(label: str) -> str:
    return label.lower().replace("+", "_plus_").replace("-", "_").replace(".", "p")


def _rewrite_shard_path(row: dict, args: Args) -> dict:
    shard_path = str(row.get("shard_path", ""))
    rewrite_pairs = (
        (args.manifest_rewrite_from, args.manifest_rewrite_to),
        (args.legacy_manifest_rewrite_from, args.manifest_rewrite_to),
    )
    for source_prefix, target_prefix in rewrite_pairs:
        if source_prefix and shard_path.startswith(source_prefix.rstrip("/") + "/"):
            row = dict(row)
            row["shard_path"] = target_prefix.rstrip("/") + shard_path[len(source_prefix.rstrip("/")) :]
            break
    return row


def _filter_existing_rows(rows: list[dict], args: Args) -> list[dict]:
    if not args.drop_missing_shards:
        return rows
    return [row for row in rows if pathlib.Path(str(row.get("shard_path", ""))).exists()]


def _split_expert_rows(args: Args) -> tuple[list[dict], list[dict]]:
    rows = _filter_existing_rows([_rewrite_shard_path(row, args) for row in read_jsonl(args.expert_manifest_path)], args)
    if not rows:
        raise ValueError(f"expert manifest is empty: {args.expert_manifest_path}")
    shuffled = list(rows)
    random.Random(args.expert_split_seed).shuffle(shuffled)
    holdout_count = max(1, round(len(shuffled) * args.expert_holdout_ratio))
    holdout_count = min(holdout_count, len(shuffled) - 1) if len(shuffled) > 1 else 1
    return shuffled[holdout_count:], shuffled[:holdout_count]


def _sample_rows(rows: list[dict], fraction: float) -> list[dict]:
    if fraction <= 0.0:
        return []
    if fraction >= 1.0:
        return list(rows)
    count = max(1, round(len(rows) * fraction))
    return list(rows[:count])


def prepare_manifests(args: Args, specs: list[SweepSpec]) -> dict[SweepSpec, PreparedManifests]:
    prepared: dict[SweepSpec, PreparedManifests] = {}
    current_train_rows = _filter_existing_rows(
        [_rewrite_shard_path(row, args) for row in read_jsonl(args.current_train_manifest_path)],
        args,
    )
    current_holdout_rows = _filter_existing_rows(
        [_rewrite_shard_path(row, args) for row in read_jsonl(args.current_holdout_manifest_path)],
        args,
    )
    expert_train_rows, expert_holdout_rows = _split_expert_rows(args)
    manifest_dir = args.output_root / "prepared_manifests"
    current_train = write_jsonl(manifest_dir / "current_train_manifest.jsonl", current_train_rows)
    current_holdout = write_jsonl(manifest_dir / "current_holdout_manifest.jsonl", current_holdout_rows)

    for spec in specs:
        if not spec.include_expert:
            prepared[spec] = PreparedManifests(
                train_manifest=current_train,
                holdout_manifest=current_holdout,
                current_holdout_manifest=current_holdout,
            )
            continue
        selected_expert_train_rows = _sample_rows(expert_train_rows, spec.expert_train_fraction)
        train_manifest = write_jsonl(
            manifest_dir / f"{_slug(spec.label)}_train_manifest.jsonl",
            [*current_train_rows, *selected_expert_train_rows],
        )
        holdout_manifest = write_jsonl(
            manifest_dir / f"{_slug(spec.label)}_holdout_manifest.jsonl",
            [*current_holdout_rows, *expert_holdout_rows],
        )
        prepared[spec] = PreparedManifests(
            train_manifest=train_manifest,
            holdout_manifest=holdout_manifest,
            current_holdout_manifest=current_holdout,
        )

    # Keep rewritten current manifests as an audit artifact even when current-only runs use caller paths.
    _ = (current_train, current_holdout)
    return prepared


def build_train_command(args: Args, spec: SweepSpec, manifests: PreparedManifests) -> list[str]:
    output_dir = args.output_root / _slug(spec.label)
    return [
        sys.executable,
        "scripts/train_rlt_online.py",
        "--output-dir",
        str(output_dir),
        "--replay-dir",
        str(args.replay_dir),
        "--manifest-path",
        str(manifests.train_manifest),
        "--holdout-manifest-path",
        str(manifests.holdout_manifest),
        "--recursive-scan",
        "--expected-replay-action-horizon",
        str(args.expected_replay_action_horizon),
        "--train-action-horizon",
        str(args.train_action_horizon),
        "--min-replay-samples",
        "1",
        "--min-success-episodes",
        "1",
        "--min-failure-episodes",
        "1",
        "--actor-min-success-episodes",
        "1",
        "--actor-min-failure-episodes",
        "1",
        "--num-train-steps",
        str(args.num_train_steps),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--critic-lr",
        str(args.critic_lr),
        "--critic-loss-mode",
        spec.critic_loss_mode,
        "--conservative-alpha",
        str(spec.conservative_alpha),
        "--critic-burn-in-steps",
        "999999999",
        "--policy-delay",
        "100000000",
        "--actor-publish-interval",
        "0",
        "--actor-lr",
        "0.0",
        "--no-online-safety-enabled",
        "--start-trainer-enabled",
        "--no-wandb-enabled",
    ]


def build_eval_command(
    args: Args,
    spec: SweepSpec,
    manifests: PreparedManifests,
    *,
    eval_name: str = "holdout",
) -> list[str]:
    output_dir = args.output_root / _slug(spec.label)
    holdout_manifest = manifests.holdout_manifest
    eval_output_name = "holdout_eval"
    if eval_name == "current_holdout":
        if manifests.current_holdout_manifest is None:
            raise ValueError(f"current_holdout_manifest is not prepared for {spec.label}")
        holdout_manifest = manifests.current_holdout_manifest
        eval_output_name = "current_holdout_eval"
    elif eval_name != "holdout":
        raise ValueError(f"unknown eval_name: {eval_name}")
    return [
        sys.executable,
        "scripts/evaluate_rlt_holdout.py",
        "--checkpoint-dir",
        str(output_dir / "snapshots"),
        "--replay-dir",
        str(args.replay_dir),
        "--holdout-manifest-path",
        str(holdout_manifest),
        "--output-dir",
        str(output_dir / eval_output_name),
        "--score-batch-size",
        str(args.score_batch_size),
    ]


def build_compare_command(args: Args, specs: list[SweepSpec], *, eval_name: str = "holdout") -> list[str]:
    eval_output_name = "holdout_eval"
    comparison_name = "comparison"
    if eval_name == "current_holdout":
        eval_output_name = "current_holdout_eval"
        comparison_name = "comparison_current_holdout"
    elif eval_name != "holdout":
        raise ValueError(f"unknown eval_name: {eval_name}")
    run_spec = ",".join(
        f"{spec.label}={args.output_root / _slug(spec.label) / eval_output_name / 'critic_holdout_metrics.csv'}"
        for spec in specs
    )
    return [
        sys.executable,
        "scripts/compare_rlt_holdout_runs.py",
        "--output-dir",
        str(args.output_root / comparison_name),
        "--title",
        args.title if eval_name == "holdout" else f"{args.title} current holdout only",
        "--run-spec",
        run_spec,
    ]


def _run(command: list[str], *, cwd: pathlib.Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main(args: Args) -> None:
    repo_root = pathlib.Path.cwd()
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = build_sweep_specs(args)
    manifests_by_spec = prepare_manifests(args, specs)

    for spec in specs:
        metrics_csv = args.output_root / _slug(spec.label) / "holdout_eval" / "critic_holdout_metrics.csv"
        if args.skip_existing and metrics_csv.exists():
            print(f"skip existing {spec.label}: {metrics_csv}", flush=True)
            continue
        train_command = build_train_command(args, spec, manifests_by_spec[spec])
        eval_command = build_eval_command(args, spec, manifests_by_spec[spec])
        if args.dry_run:
            print("+ " + " ".join(train_command), flush=True)
            print("+ " + " ".join(eval_command), flush=True)
            if args.eval_current_holdout:
                print(
                    "+ " + " ".join(build_eval_command(args, spec, manifests_by_spec[spec], eval_name="current_holdout")),
                    flush=True,
                )
        else:
            _run(train_command, cwd=repo_root)
            _run(eval_command, cwd=repo_root)
            if args.eval_current_holdout:
                _run(build_eval_command(args, spec, manifests_by_spec[spec], eval_name="current_holdout"), cwd=repo_root)

    compare_command = build_compare_command(args, specs)
    if args.dry_run:
        print("+ " + " ".join(compare_command), flush=True)
        if args.eval_current_holdout:
            print("+ " + " ".join(build_compare_command(args, specs, eval_name="current_holdout")), flush=True)
    else:
        _run(compare_command, cwd=repo_root)
        if args.eval_current_holdout:
            _run(build_compare_command(args, specs, eval_name="current_holdout"), cwd=repo_root)


if __name__ == "__main__":
    main(tyro.cli(Args))

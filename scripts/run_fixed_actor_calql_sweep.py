from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import tyro


@dataclasses.dataclass(frozen=True)
class SweepSpec:
    label: str
    critic_loss_mode: str
    conservative_alpha: float


@dataclasses.dataclass
class Args:
    output_root: pathlib.Path
    train_manifest_path: pathlib.Path
    holdout_manifest_path: pathlib.Path
    replay_dir: pathlib.Path = pathlib.Path("/app/replay/rlt_key_regions")
    calql_alphas: tuple[float, ...] = (0.03, 0.1, 0.3)
    num_train_steps: int = 10_000
    batch_size: int = 64
    seed: int = 7
    critic_lr: float = 3e-4
    log_interval: int = 100
    save_interval: int = 1_000
    score_batch_size: int = 512
    expected_replay_action_horizon: int = 10
    train_action_horizon: int = 10
    title: str = "Fixed actor Cal-QL critic comparison"
    dry_run: bool = False
    skip_existing: bool = False


def build_sweep_specs(args: Args) -> list[SweepSpec]:
    specs = [SweepSpec(label="TD3", critic_loss_mode="td3", conservative_alpha=0.0)]
    specs.extend(
        [
            SweepSpec(
                label=f"CalQL-alpha{alpha:g}",
                critic_loss_mode="calql",
                conservative_alpha=float(alpha),
            )
            for alpha in args.calql_alphas
        ]
    )
    return specs


def _slug(label: str) -> str:
    return label.lower().replace("-", "_").replace(".", "p")


def build_train_command(args: Args, spec: SweepSpec) -> list[str]:
    output_dir = args.output_root / _slug(spec.label)
    return [
        sys.executable,
        "scripts/train_rlt_online.py",
        "--output-dir",
        str(output_dir),
        "--replay-dir",
        str(args.replay_dir),
        "--manifest-path",
        str(args.train_manifest_path),
        "--holdout-manifest-path",
        str(args.holdout_manifest_path),
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


def build_eval_command(args: Args, spec: SweepSpec) -> list[str]:
    output_dir = args.output_root / _slug(spec.label)
    return [
        sys.executable,
        "scripts/evaluate_rlt_holdout.py",
        "--checkpoint-dir",
        str(output_dir / "snapshots"),
        "--replay-dir",
        str(args.replay_dir),
        "--holdout-manifest-path",
        str(args.holdout_manifest_path),
        "--output-dir",
        str(output_dir / "holdout_eval"),
        "--score-batch-size",
        str(args.score_batch_size),
    ]


def build_compare_command(args: Args, specs: list[SweepSpec]) -> list[str]:
    run_spec = ",".join(
        f"{spec.label}={args.output_root / _slug(spec.label) / 'holdout_eval' / 'critic_holdout_metrics.csv'}"
        for spec in specs
    )
    return [
        sys.executable,
        "scripts/compare_rlt_holdout_runs.py",
        "--output-dir",
        str(args.output_root / "comparison"),
        "--title",
        args.title,
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

    for spec in specs:
        metrics_csv = args.output_root / _slug(spec.label) / "holdout_eval" / "critic_holdout_metrics.csv"
        if args.skip_existing and metrics_csv.exists():
            print(f"skip existing {spec.label}: {metrics_csv}", flush=True)
            continue
        train_command = build_train_command(args, spec)
        eval_command = build_eval_command(args, spec)
        if args.dry_run:
            print("+ " + " ".join(train_command), flush=True)
            print("+ " + " ".join(eval_command), flush=True)
        else:
            _run(train_command, cwd=repo_root)
            _run(eval_command, cwd=repo_root)

    compare_command = build_compare_command(args, specs)
    if args.dry_run:
        print("+ " + " ".join(compare_command), flush=True)
    else:
        _run(compare_command, cwd=repo_root)


if __name__ == "__main__":
    main(tyro.cli(Args))

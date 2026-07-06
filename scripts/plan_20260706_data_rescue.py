#!/usr/bin/env python3
"""Generate the 2026-07-06 RLT data rescue commands.

The raw online replay from 2026-07-06 must not be trained directly because its
saved z/proprio came from runtime cache blocks. This planner keeps the two
collection groups separate:

* base142_legacy_unmarked: 09:06-10:27, used to train the first actor.
* actor93_runtime_cache_block: 13:21-14:40, collected by that actor.

The commands rebuild both groups into formal paper_subsampled_anchor replay,
then run A-only and A+B critic/actor comparisons with the existing offline RLT
trainer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shlex
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class RescuePlanConfig:
    python: str = "python"
    source_replay_root: Path = Path(
        "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
        "rlt_key_regions/twist_off_the_bottle_cap/2026-07-06"
    )
    rollout_root: Path = Path(
        "/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/"
        "twist_off_the_bottle_cap/2026-07-06/rl"
    )
    paper_anchor_root: Path = Path(
        "/home/eii/data/openpi0.5-rtc-reward-learning/replay/"
        "paper_anchor_2048/twist_off_the_bottle_cap/2026-07-06"
    )
    manifest_dir: Path = Path("local_rlt_manifests/paper_anchor_20260706_rescue")
    work_dir: Path = Path("local_rlt_reencoded/paper_anchor_20260706_rescue")
    original_train_manifest: Path = Path(
        "local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/train_manifest.jsonl"
    )
    original_holdout_manifest: Path = Path(
        "local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/holdout_manifest.jsonl"
    )
    original_replay_dir: Path = Path("local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z")
    prior_clean_actor_checkpoint: Path = Path(
        "local_rlt_runs/strict_td3_z_ablation_20260704/"
        "actor_from_vla_token_critic6000_actor5000_td3_20260706/inference_actor/00005000"
    )
    bootstrap_train_manifest: Path = Path(
        "local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/train_bootstrap117_expert59.jsonl"
    )
    bootstrap_holdout_manifest: Path = Path(
        "local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/holdout_bootstrap29.jsonl"
    )
    strict_vla_all_manifest: Path = Path(
        "local_rlt_runs/strict_td3_z_ablation_20260704/replay/vla_token_z/all_manifest.jsonl"
    )
    bootstrap_exact_manifest_dir: Path = Path("local_rlt_manifests/bootstrap146_vla_same_forward_exact_split_20260706")
    output_run_root: Path = Path("local_rlt_runs/data_rescue_20260706")
    batch_size: int = 128
    critic_steps: int = 10_000
    actor_steps: int = 6_000
    save_interval: int = 1_000
    eval_every_steps: int = 1_000
    score_batch_size: int = 512

    @classmethod
    def for_profile(cls, profile: str) -> "RescuePlanConfig":
        if profile == "local":
            return cls()
        if profile == "103":
            data_root = Path("/data/openpi0.5-rtc-reward-learning")
            return cls(
                source_replay_root=data_root / "replay/rlt_key_regions/twist_off_the_bottle_cap/2026-07-06",
                rollout_root=data_root / "rollouts/key_regions/twist_off_the_bottle_cap/2026-07-06/rl",
                paper_anchor_root=data_root / "replay/paper_anchor_2048/twist_off_the_bottle_cap/2026-07-06",
            )
        raise ValueError(f"Unsupported profile={profile!r}; expected 'local' or '103'")

    @property
    def base_label(self) -> str:
        return "20260706_base142"

    @property
    def actor_label(self) -> str:
        return "20260706_actor93"

    @property
    def base_output_root(self) -> Path:
        return self.paper_anchor_root / "base142"

    @property
    def actor_output_root(self) -> Path:
        return self.paper_anchor_root / "actor93"

    @property
    def base_manifest(self) -> Path:
        return self.manifest_dir / f"{self.base_label}_paper_anchor_manifest.jsonl"

    @property
    def actor_manifest(self) -> Path:
        return self.manifest_dir / f"{self.actor_label}_paper_anchor_manifest.jsonl"

    @property
    def original_plus_base_manifest(self) -> Path:
        return self.manifest_dir / f"train_original_plus_{self.base_label}.jsonl"

    @property
    def original_plus_base_plus_actor_manifest(self) -> Path:
        return self.manifest_dir / f"train_original_plus_{self.base_label}_plus_actor93.jsonl"


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _cmd(parts: Iterable[object]) -> str:
    return " ".join(_q(part) for part in parts)


def _rebuild_command(cfg: RescuePlanConfig, *, group: str, label: str, output_root: Path, work_subdir: str) -> str:
    return _cmd(
        [
            cfg.python,
            "scripts/rebuild_online_rollout_paper_anchor_replay.py",
            "--phase",
            "all",
            "--source-replay-root",
            cfg.source_replay_root,
            "--rollout-root",
            cfg.rollout_root,
            "--output-root",
            output_root,
            "--work-dir",
            cfg.work_dir / work_subdir,
            "--manifest-dir",
            cfg.manifest_dir,
            "--dataset-label",
            label,
            "--collection-group",
            group,
            "--original-train-manifest",
            cfg.original_train_manifest,
        ]
    )


def _train_command(
    cfg: RescuePlanConfig,
    *,
    output_dir: Path,
    replay_dir: Path,
    manifest_path: Path,
    training_stage: str,
    num_steps: int,
    init_critic_checkpoint: Path | None = None,
    init_actor_checkpoint: Path | None = None,
) -> str:
    parts: list[object] = [
        cfg.python,
        "scripts/train_rlt_offline.py",
        "--replay-dir",
        replay_dir,
        "--output-dir",
        output_dir,
        "--num-train-steps",
        num_steps,
        "--batch-size",
        cfg.batch_size,
        "--training-stage",
        training_stage,
        "--manifest-path",
        manifest_path,
        "--min-replay-samples",
        512,
        "--min-replay-shards",
        1,
        "--min-success-episodes",
        1,
        "--min-failure-episodes",
        1,
        "--save-interval",
        cfg.save_interval,
        "--log-interval",
        100,
        "--no-target-actor-noise",
        "--critic-target-action-mode",
        "reference_action",
        "--actor-loss-mode",
        "td3",
        "--critic-loss-mode",
        "td3",
        "--train-action-horizon",
        10,
        "--expected-replay-action-horizon",
        10,
        "--expected-replay-z-dim",
        2048,
        "--eval-holdout-critic",
        "--eval-holdout-every-steps",
        cfg.eval_every_steps,
        "--holdout-score-batch-size",
        cfg.score_batch_size,
        "--no-wandb-enabled",
        "--overwrite",
    ]
    if init_critic_checkpoint is not None:
        parts.extend(["--init-critic-checkpoint", init_critic_checkpoint])
    if init_actor_checkpoint is not None:
        parts.extend(["--init-inference-actor-checkpoint", init_actor_checkpoint])
    return _cmd(parts)


def _eval_command(
    cfg: RescuePlanConfig,
    *,
    checkpoint_dir: Path,
    replay_dir: Path,
    holdout_manifest: Path,
    output_dir: Path,
) -> str:
    return _cmd(
        [
            cfg.python,
            "scripts/evaluate_rlt_holdout.py",
            "--checkpoint-dir",
            checkpoint_dir,
            "--replay-dir",
            replay_dir,
            "--holdout-manifest-path",
            holdout_manifest,
            "--output-dir",
            output_dir,
            "--score-batch-size",
            cfg.score_batch_size,
        ]
    )


def build_rescue_commands(cfg: RescuePlanConfig) -> list[str]:
    base_critic = cfg.output_run_root / "critic_base142_only_10000"
    original_base_critic = cfg.output_run_root / "critic_original_plus_base142_10000"
    original_base_actor_critic = cfg.output_run_root / "critic_original_plus_base142_plus_actor93_10000"
    actor_from_original_base = cfg.output_run_root / "actor_from_original_plus_base142_6000"
    actor_from_original_base_actor = cfg.output_run_root / "actor_from_original_plus_base142_plus_actor93_6000"

    return [
        "# 1. Rebuild A group: 09:06-10:27 base142 legacy/unmarked raw replay",
        _rebuild_command(cfg, group="base142", label=cfg.base_label, output_root=cfg.base_output_root, work_subdir="base142"),
        "# 2. Rebuild B group: 13:21-14:40 actor93 runtime-cache-block raw replay",
        _rebuild_command(cfg, group="actor93", label=cfg.actor_label, output_root=cfg.actor_output_root, work_subdir="actor93"),
        "# 3. Combine original, A, and B manifests after both rebuilds finish",
        _cmd([cfg.python, "scripts/plan_20260706_data_rescue.py", "--write-combined-manifests"]),
        "# 3b. Materialize bootstrap146 same-forward exact train/holdout split for earliest-actor audits",
        _cmd([cfg.python, "scripts/plan_20260706_data_rescue.py", "--write-bootstrap-exact-split-manifests"]),
        "# 4. Train A-only critic to isolate whether the first 142 shards are usable after correct re-encoding",
        _train_command(
            cfg,
            output_dir=base_critic,
            replay_dir=cfg.base_output_root,
            manifest_path=cfg.base_manifest,
            training_stage="critic_only",
            num_steps=cfg.critic_steps,
        ),
        "# 5. Train original+A critic, the likely safer baseline for actor continuation",
        _train_command(
            cfg,
            output_dir=original_base_critic,
            replay_dir=cfg.paper_anchor_root,
            manifest_path=cfg.original_plus_base_manifest,
            training_stage="critic_only",
            num_steps=cfg.critic_steps,
        ),
        "# 6. Evaluate original+A critic on B-only holdout before allowing B into training",
        _eval_command(
            cfg,
            checkpoint_dir=original_base_critic / "snapshots",
            replay_dir=cfg.actor_output_root,
            holdout_manifest=cfg.actor_manifest,
            output_dir=original_base_critic / "post_eval" / "actor93_rebuilt",
        ),
        "# 7. Train original+A+B critic only if step 6 does not show failure Q inflation",
        _train_command(
            cfg,
            output_dir=original_base_actor_critic,
            replay_dir=cfg.paper_anchor_root,
            manifest_path=cfg.original_plus_base_plus_actor_manifest,
            training_stage="critic_only",
            num_steps=cfg.critic_steps,
        ),
        "# 8. Continue actor from the clean prior actor with original+A critic",
        _train_command(
            cfg,
            output_dir=actor_from_original_base,
            replay_dir=cfg.paper_anchor_root,
            manifest_path=cfg.original_plus_base_manifest,
            training_stage="actor_only",
            num_steps=cfg.actor_steps,
            init_critic_checkpoint=original_base_critic / "checkpoints" / f"{cfg.critic_steps:08d}",
            init_actor_checkpoint=cfg.prior_clean_actor_checkpoint,
        ),
        "# 9. Continue actor from the clean prior actor with original+A+B critic for comparison",
        _train_command(
            cfg,
            output_dir=actor_from_original_base_actor,
            replay_dir=cfg.paper_anchor_root,
            manifest_path=cfg.original_plus_base_plus_actor_manifest,
            training_stage="actor_only",
            num_steps=cfg.actor_steps,
            init_critic_checkpoint=original_base_actor_critic / "checkpoints" / f"{cfg.critic_steps:08d}",
            init_actor_checkpoint=cfg.prior_clean_actor_checkpoint,
        ),
        "# 10. Evaluate both actor runs on original holdout and actor93 rebuilt holdout",
        _eval_command(
            cfg,
            checkpoint_dir=actor_from_original_base,
            replay_dir=cfg.original_replay_dir,
            holdout_manifest=cfg.original_holdout_manifest,
            output_dir=actor_from_original_base / "post_eval" / "original_holdout",
        ),
        _eval_command(
            cfg,
            checkpoint_dir=actor_from_original_base_actor,
            replay_dir=cfg.actor_output_root,
            holdout_manifest=cfg.actor_manifest,
            output_dir=actor_from_original_base_actor / "post_eval" / "actor93_rebuilt",
        ),
    ]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_combined_manifests(cfg: RescuePlanConfig) -> dict[str, int]:
    original_rows = read_jsonl(cfg.original_train_manifest)
    base_rows = read_jsonl(cfg.base_manifest)
    actor_rows = read_jsonl(cfg.actor_manifest)
    write_jsonl(cfg.original_plus_base_manifest, original_rows + base_rows)
    write_jsonl(cfg.original_plus_base_plus_actor_manifest, original_rows + base_rows + actor_rows)
    return {
        "original": len(original_rows),
        "base142": len(base_rows),
        "actor93": len(actor_rows),
        "original_plus_base": len(original_rows) + len(base_rows),
        "original_plus_base_plus_actor93": len(original_rows) + len(base_rows) + len(actor_rows),
    }


def _bootstrap_ids(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        if row.get("kind") not in (None, "bootstrap"):
            continue
        key_region_id = row.get("key_region_id")
        if key_region_id:
            ids.append(str(key_region_id).removeprefix("key_region_"))
    return ids


def _rows_for_ids(ids: list[str], rows_by_id: dict[str, dict]) -> tuple[list[dict], list[str]]:
    selected: list[dict] = []
    missing: list[str] = []
    for key_region_id in ids:
        row = rows_by_id.get(key_region_id)
        if row is None:
            missing.append(key_region_id)
            continue
        selected.append(
            {
                **row,
                "bootstrap_exact_split": True,
                "z_rl_source": "vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder",
            }
        )
    return selected, missing


def write_bootstrap_exact_split_manifests(cfg: RescuePlanConfig) -> dict[str, int]:
    """Reuse existing same-forward shards but preserve the original 117/29 split."""

    train_ids = _bootstrap_ids(read_jsonl(cfg.bootstrap_train_manifest))
    holdout_ids = _bootstrap_ids(read_jsonl(cfg.bootstrap_holdout_manifest))
    same_forward_rows = read_jsonl(cfg.strict_vla_all_manifest)
    rows_by_id = {str(row["key_region_id"]).removeprefix("key_region_"): row for row in same_forward_rows}
    train_rows, train_missing = _rows_for_ids(train_ids, rows_by_id)
    holdout_rows, holdout_missing = _rows_for_ids(holdout_ids, rows_by_id)
    missing = train_missing + holdout_missing
    if missing:
        missing_path = cfg.bootstrap_exact_manifest_dir / "missing_key_region_ids.json"
        cfg.bootstrap_exact_manifest_dir.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(json.dumps(missing, indent=2, sort_keys=True), encoding="utf-8")
        raise ValueError(f"Missing {len(missing)} bootstrap ids in {cfg.strict_vla_all_manifest}; wrote {missing_path}")
    write_jsonl(cfg.bootstrap_exact_manifest_dir / "train_bootstrap117_vla_same_forward_exact_split.jsonl", train_rows)
    write_jsonl(cfg.bootstrap_exact_manifest_dir / "holdout_bootstrap29_vla_same_forward_exact_split.jsonl", holdout_rows)
    return {"train": len(train_rows), "holdout": len(holdout_rows), "missing": 0}


def render_markdown_plan(cfg: RescuePlanConfig) -> str:
    commands = "\n\n".join(f"```bash\n{command}\n```" if not command.startswith("#") else command for command in build_rescue_commands(cfg))
    return f"""# 2026-07-06 RLT 数据挽救执行计划

## 数据分组

- `base142`: 09:06-10:27 的 142 条 legacy/unmarked raw replay。它们是训练今天第一个 actor 的基础数据，必须先重建成 formal `paper_subsampled_anchor`。
- `actor93`: 13:21-14:40 的 93 条由第一个 actor 采集的数据。它们是高风险 off-policy 数据，只能在正确重建并通过 holdout 检查后进入 critic 训练。

## 强约束

- 不能直接训练旧 z_rl/proprio。
- 不能把 `runtime_action_cache_block` shard 当 formal replay。
- `actor93` 不能直接当高质量示范模仿；优先用于 critic 学习失败边界。
- A+B critic 必须同时看 original holdout 和 actor93 rebuilt holdout，不能只看训练集指标。

## 执行命令

{commands}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("local", "103"), default="local")
    parser.add_argument("--output-markdown", type=Path, default=Path("local_rlt_manifests/paper_anchor_20260706_rescue/data_rescue_plan.md"))
    parser.add_argument("--output-shell", type=Path, default=Path("local_rlt_manifests/paper_anchor_20260706_rescue/data_rescue_commands.sh"))
    parser.add_argument("--write-combined-manifests", action="store_true")
    parser.add_argument("--write-bootstrap-exact-split-manifests", action="store_true")
    parser.add_argument("--print", action="store_true", dest="print_plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RescuePlanConfig.for_profile(args.profile)
    if args.write_combined_manifests:
        summary = write_combined_manifests(cfg)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.write_bootstrap_exact_split_manifests:
        summary = write_bootstrap_exact_split_manifests(cfg)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    markdown = render_markdown_plan(cfg)
    shell = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(build_rescue_commands(cfg)) + "\n"
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    args.output_shell.parent.mkdir(parents=True, exist_ok=True)
    args.output_shell.write_text(shell, encoding="utf-8")
    if args.print_plan:
        print(markdown)
    else:
        print(json.dumps({"markdown": str(args.output_markdown), "shell": str(args.output_shell)}, indent=2))


if __name__ == "__main__":
    main()

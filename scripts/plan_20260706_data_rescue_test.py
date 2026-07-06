from __future__ import annotations

import json

from scripts import plan_20260706_data_rescue as plan


def test_build_rescue_plan_separates_base_and_actor_groups() -> None:
    cfg = plan.RescuePlanConfig()

    commands = plan.build_rescue_commands(cfg)

    joined = "\n".join(commands)
    assert "--collection-group base142" in joined
    assert "--collection-group actor93" in joined
    assert "20260706_base142_paper_anchor_manifest.jsonl" in joined
    assert "20260706_actor93_paper_anchor_manifest.jsonl" in joined
    assert "train_original_plus_20260706_base142.jsonl" in joined
    assert "train_original_plus_20260706_base142_plus_actor93.jsonl" in joined
    assert "--training-stage critic_only" in joined
    assert "--training-stage actor_only" in joined


def test_markdown_plan_contains_risk_policy_for_actor93() -> None:
    md = plan.render_markdown_plan(plan.RescuePlanConfig())

    assert "actor93" in md
    assert "高风险 off-policy 数据" in md
    assert "不能直接训练旧 z_rl/proprio" in md


def test_remote_103_profile_uses_data_mount() -> None:
    cfg = plan.RescuePlanConfig.for_profile("103")

    commands = "\n".join(plan.build_rescue_commands(cfg))

    assert "/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions" in commands
    assert "/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions" not in commands


def test_write_bootstrap_exact_split_manifests_reuses_existing_same_forward_rows(tmp_path) -> None:
    bootstrap_train = tmp_path / "bootstrap_train.jsonl"
    bootstrap_holdout = tmp_path / "bootstrap_holdout.jsonl"
    same_forward_all = tmp_path / "same_forward_all.jsonl"
    out_dir = tmp_path / "out"
    bootstrap_train.write_text(
        json.dumps({"key_region_id": "a", "kind": "bootstrap"}) + "\n"
        + json.dumps({"key_region_id": "expert", "kind": "expert"}) + "\n",
        encoding="utf-8",
    )
    bootstrap_holdout.write_text(json.dumps({"key_region_id": "b", "kind": "bootstrap"}) + "\n", encoding="utf-8")
    same_forward_all.write_text(
        json.dumps({"key_region_id": "a", "shard_path": "/same/a.npz", "reward": 1}) + "\n"
        + json.dumps({"key_region_id": "b", "shard_path": "/same/b.npz", "reward": 0}) + "\n",
        encoding="utf-8",
    )
    cfg = plan.RescuePlanConfig(
        bootstrap_train_manifest=bootstrap_train,
        bootstrap_holdout_manifest=bootstrap_holdout,
        strict_vla_all_manifest=same_forward_all,
        bootstrap_exact_manifest_dir=out_dir,
    )

    summary = plan.write_bootstrap_exact_split_manifests(cfg)

    assert summary == {"train": 1, "holdout": 1, "missing": 0}
    assert "same/a.npz" in (out_dir / "train_bootstrap117_vla_same_forward_exact_split.jsonl").read_text()
    assert "same/b.npz" in (out_dir / "holdout_bootstrap29_vla_same_forward_exact_split.jsonl").read_text()

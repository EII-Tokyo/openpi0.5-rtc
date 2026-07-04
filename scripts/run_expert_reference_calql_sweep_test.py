import json
import pathlib

from scripts import run_expert_reference_calql_sweep


def _write_manifest(path: pathlib.Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for label in labels:
            file.write(json.dumps({"shard_path": f"/data/{label}.npz", "label": label}) + "\n")


def _write_path_manifest(path: pathlib.Path, shard_paths: list[pathlib.Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for shard_path in shard_paths:
            file.write(json.dumps({"shard_path": str(shard_path), "label": shard_path.stem}) + "\n")


def test_sweep_specs_compare_current_and_expert_for_td3_and_calql(tmp_path):
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=tmp_path / "current_train.jsonl",
        current_holdout_manifest_path=tmp_path / "current_holdout.jsonl",
        expert_manifest_path=tmp_path / "expert.jsonl",
        calql_alpha=0.1,
    )

    specs = run_expert_reference_calql_sweep.build_sweep_specs(args)

    assert [spec.label for spec in specs] == [
        "TD3-current",
        "TD3-current+expert",
        "CalQL-current",
        "CalQL-current+expert",
    ]
    assert [spec.critic_loss_mode for spec in specs] == ["td3", "td3", "calql", "calql"]
    assert [spec.include_expert for spec in specs] == [False, True, False, True]
    assert specs[-1].conservative_alpha == 0.1


def test_prepare_manifests_keeps_current_only_and_combines_expert_runs(tmp_path):
    current_train = tmp_path / "current_train.jsonl"
    current_holdout = tmp_path / "current_holdout.jsonl"
    expert = tmp_path / "expert.jsonl"
    _write_manifest(current_train, ["current_train_a", "current_train_b"])
    _write_manifest(current_holdout, ["current_holdout_a"])
    _write_manifest(expert, ["expert_a", "expert_b", "expert_c", "expert_d", "expert_e"])
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=current_train,
        current_holdout_manifest_path=current_holdout,
        expert_manifest_path=expert,
        expert_holdout_ratio=0.4,
        expert_split_seed=3,
        drop_missing_shards=False,
    )
    specs = run_expert_reference_calql_sweep.build_sweep_specs(args)

    manifests = run_expert_reference_calql_sweep.prepare_manifests(args, specs)

    current_spec = specs[0]
    expert_spec = specs[1]
    assert manifests[current_spec].train_manifest != current_train
    assert manifests[current_spec].holdout_manifest != current_holdout
    expert_train_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[expert_spec].train_manifest)
    expert_holdout_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[expert_spec].holdout_manifest)
    assert len(expert_train_rows) == 2 + 3
    assert len(expert_holdout_rows) == 1 + 2
    assert {row["label"] for row in expert_train_rows}.issuperset({"current_train_a", "current_train_b"})
    assert {row["label"] for row in expert_holdout_rows}.issuperset({"current_holdout_a"})


def test_prepare_manifests_drops_missing_shards_for_all_runs(tmp_path):
    existing_current = tmp_path / "existing_current.npz"
    missing_current = tmp_path / "missing_current.npz"
    existing_holdout = tmp_path / "existing_holdout.npz"
    existing_expert = tmp_path / "existing_expert.npz"
    missing_expert = tmp_path / "missing_expert.npz"
    for path in [existing_current, existing_holdout, existing_expert]:
        path.write_text("placeholder", encoding="utf-8")
    current_train = tmp_path / "current_train.jsonl"
    current_holdout = tmp_path / "current_holdout.jsonl"
    expert = tmp_path / "expert.jsonl"
    _write_path_manifest(current_train, [existing_current, missing_current])
    _write_path_manifest(current_holdout, [existing_holdout])
    _write_path_manifest(expert, [existing_expert, missing_expert])
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=current_train,
        current_holdout_manifest_path=current_holdout,
        expert_manifest_path=expert,
        expert_holdout_ratio=0.5,
    )
    specs = run_expert_reference_calql_sweep.build_sweep_specs(args)

    manifests = run_expert_reference_calql_sweep.prepare_manifests(args, specs)

    for spec in specs:
        train_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[spec].train_manifest)
        holdout_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[spec].holdout_manifest)
        all_paths = [pathlib.Path(row["shard_path"]) for row in [*train_rows, *holdout_rows]]
        assert all(path.exists() for path in all_paths)


def test_train_command_disables_actor_and_trains_critic_for_10000_steps(tmp_path):
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=tmp_path / "current_train.jsonl",
        current_holdout_manifest_path=tmp_path / "current_holdout.jsonl",
        expert_manifest_path=tmp_path / "expert.jsonl",
    )
    spec = run_expert_reference_calql_sweep.SweepSpec(
        label="CalQL-current+expert",
        critic_loss_mode="calql",
        conservative_alpha=0.1,
        include_expert=True,
    )
    manifests = run_expert_reference_calql_sweep.PreparedManifests(
        train_manifest=tmp_path / "prepared_train.jsonl",
        holdout_manifest=tmp_path / "prepared_holdout.jsonl",
    )

    command = run_expert_reference_calql_sweep.build_train_command(args, spec, manifests)

    assert "--num-train-steps" in command
    assert command[command.index("--num-train-steps") + 1] == "10000"
    assert "--critic-loss-mode" in command
    assert command[command.index("--critic-loss-mode") + 1] == "calql"
    assert "--conservative-alpha" in command
    assert command[command.index("--conservative-alpha") + 1] == "0.1"
    assert "--critic-burn-in-steps" in command
    assert command[command.index("--critic-burn-in-steps") + 1] == "999999999"
    assert "--policy-delay" in command
    assert command[command.index("--policy-delay") + 1] == "100000000"
    assert "--actor-publish-interval" in command
    assert command[command.index("--actor-publish-interval") + 1] == "0"
    assert "--actor-lr" in command
    assert command[command.index("--actor-lr") + 1] == "0.0"


def test_fractional_sweep_specs_cover_expert_ratios_and_calql_alphas(tmp_path):
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=tmp_path / "current_train.jsonl",
        current_holdout_manifest_path=tmp_path / "current_holdout.jsonl",
        expert_manifest_path=tmp_path / "expert.jsonl",
        sweep_mode="fractional_expert",
        expert_train_fraction=(0.25, 0.5),
        calql_alpha_grid=(0.03, 0.1),
    )

    specs = run_expert_reference_calql_sweep.build_sweep_specs(args)

    assert [spec.label for spec in specs] == [
        "TD3-current",
        "TD3-expert25",
        "TD3-expert50",
        "CalQL0p03-expert25",
        "CalQL0p03-expert50",
        "CalQL0p1-expert25",
        "CalQL0p1-expert50",
    ]
    assert [spec.expert_train_fraction for spec in specs] == [0.0, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5]
    assert [spec.conservative_alpha for spec in specs[-4:]] == [0.03, 0.03, 0.1, 0.1]


def test_prepare_manifests_samples_expert_train_fraction_but_keeps_expert_holdout(tmp_path):
    current_train = tmp_path / "current_train.jsonl"
    current_holdout = tmp_path / "current_holdout.jsonl"
    expert = tmp_path / "expert.jsonl"
    _write_manifest(current_train, ["current_train_a", "current_train_b"])
    _write_manifest(current_holdout, ["current_holdout_a"])
    _write_manifest(expert, [f"expert_{idx}" for idx in range(10)])
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=current_train,
        current_holdout_manifest_path=current_holdout,
        expert_manifest_path=expert,
        expert_holdout_ratio=0.2,
        expert_split_seed=7,
        drop_missing_shards=False,
    )
    spec = run_expert_reference_calql_sweep.SweepSpec(
        label="CalQL0p03-expert25",
        critic_loss_mode="calql",
        conservative_alpha=0.03,
        expert_train_fraction=0.25,
    )

    manifests = run_expert_reference_calql_sweep.prepare_manifests(args, [spec])

    train_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[spec].train_manifest)
    holdout_rows = run_expert_reference_calql_sweep.read_jsonl(manifests[spec].holdout_manifest)
    expert_train_rows = [row for row in train_rows if str(row["label"]).startswith("expert_")]
    expert_holdout_rows = [row for row in holdout_rows if str(row["label"]).startswith("expert_")]
    assert len(expert_train_rows) == 2
    assert len(expert_holdout_rows) == 2
    assert {row["label"] for row in train_rows}.issuperset({"current_train_a", "current_train_b"})
    assert {row["label"] for row in holdout_rows}.issuperset({"current_holdout_a"})


def test_current_holdout_eval_command_uses_current_only_manifest_for_fractional_runs(tmp_path):
    args = run_expert_reference_calql_sweep.Args(
        output_root=tmp_path / "out",
        current_train_manifest_path=tmp_path / "current_train.jsonl",
        current_holdout_manifest_path=tmp_path / "current_holdout.jsonl",
        expert_manifest_path=tmp_path / "expert.jsonl",
        eval_current_holdout=True,
    )
    spec = run_expert_reference_calql_sweep.SweepSpec(
        label="CalQL0p03-expert25",
        critic_loss_mode="calql",
        conservative_alpha=0.03,
        expert_train_fraction=0.25,
    )
    manifests = run_expert_reference_calql_sweep.PreparedManifests(
        train_manifest=tmp_path / "prepared_train.jsonl",
        holdout_manifest=tmp_path / "combined_holdout.jsonl",
        current_holdout_manifest=tmp_path / "current_holdout_only.jsonl",
    )

    command = run_expert_reference_calql_sweep.build_eval_command(args, spec, manifests, eval_name="current_holdout")

    assert "--holdout-manifest-path" in command
    assert command[command.index("--holdout-manifest-path") + 1] == str(tmp_path / "current_holdout_only.jsonl")
    assert command[command.index("--output-dir") + 1].endswith("current_holdout_eval")

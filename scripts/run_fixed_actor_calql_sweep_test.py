import pathlib

from scripts import run_fixed_actor_calql_sweep


def test_sweep_specs_exclude_plain_cql():
    args = run_fixed_actor_calql_sweep.Args(
        output_root=pathlib.Path("/tmp/out"),
        train_manifest_path=pathlib.Path("/tmp/train.jsonl"),
        holdout_manifest_path=pathlib.Path("/tmp/holdout.jsonl"),
        calql_alphas=(0.03, 0.1),
    )

    specs = run_fixed_actor_calql_sweep.build_sweep_specs(args)

    assert [spec.label for spec in specs] == ["TD3", "CalQL-alpha0.03", "CalQL-alpha0.1"]
    assert all(spec.critic_loss_mode != "cql" for spec in specs)


def test_train_command_disables_actor_updates():
    args = run_fixed_actor_calql_sweep.Args(
        output_root=pathlib.Path("/tmp/out"),
        train_manifest_path=pathlib.Path("/tmp/train.jsonl"),
        holdout_manifest_path=pathlib.Path("/tmp/holdout.jsonl"),
    )
    spec = run_fixed_actor_calql_sweep.SweepSpec(label="CalQL-alpha0.1", critic_loss_mode="calql", conservative_alpha=0.1)

    command = run_fixed_actor_calql_sweep.build_train_command(args, spec)

    assert "--critic-burn-in-steps" in command
    assert command[command.index("--critic-burn-in-steps") + 1] == "999999999"
    assert "--policy-delay" in command
    assert command[command.index("--policy-delay") + 1] == "100000000"
    assert "--actor-publish-interval" in command
    assert command[command.index("--actor-publish-interval") + 1] == "0"
    assert "--actor-lr" in command
    assert command[command.index("--actor-lr") + 1] == "0.0"
    assert "--critic-loss-mode" in command
    assert command[command.index("--critic-loss-mode") + 1] == "calql"
    assert "--num-train-steps" in command
    assert command[command.index("--num-train-steps") + 1] == "10000"
    assert "--min-replay-size" not in command
    assert "--min-replay-samples" in command
    assert command[command.index("--min-replay-samples") + 1] == "1"

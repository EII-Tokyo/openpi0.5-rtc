from scripts import rlt_offline_transfer


def test_build_pull_commands_copy_rollouts_replay_and_ledger():
    args = rlt_offline_transfer.TransferArgs(
        remote="eii@192.168.1.103",
        local_root="/tmp/local",
        remote_project="~/openpi0.5-rtc-reward-learning",
        remote_data_root="/data/openpi0.5-rtc-reward-learning",
    )

    commands = rlt_offline_transfer.build_pull_commands(args)

    assert commands[0] == [
        "mkdir",
        "-p",
        "/tmp/local/raw_from_103/rollouts/key_regions",
        "/tmp/local/raw_from_103/replay/rlt_key_regions",
        "/tmp/local/raw_from_103/state",
    ]
    assert commands[1][:3] == ["rsync", "-a", "--info=progress2"]
    assert "eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/" in commands[1]
    assert commands[2][-1] == "/tmp/local/raw_from_103/replay/rlt_key_regions/"
    assert commands[3][-1] == "/tmp/local/raw_from_103/state/"


def test_build_deploy_commands_copy_checkpoint_without_starting_containers():
    args = rlt_offline_transfer.TransferArgs(
        remote="eii@192.168.1.103",
        local_checkpoint="/tmp/run/inference_actor/00010000",
        remote_checkpoint_dir="/data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/run_a",
    )

    commands = rlt_offline_transfer.build_deploy_commands(args)

    assert commands[0] == [
        "ssh",
        "eii@192.168.1.103",
        "mkdir -p /data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/run_a",
    ]
    assert commands[1][:3] == ["rsync", "-a", "--info=progress2"]
    assert commands[1][-2] == "/tmp/run/inference_actor/00010000/"
    assert commands[1][-1] == "eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/run_a/"

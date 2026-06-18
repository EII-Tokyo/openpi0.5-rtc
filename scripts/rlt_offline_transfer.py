from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shlex
import subprocess


@dataclasses.dataclass
class TransferArgs:
    remote: str = "eii@192.168.1.103"
    local_root: str | pathlib.Path = "local_rlt_data"
    remote_project: str = "~/openpi0.5-rtc-reward-learning"
    remote_data_root: str = "/data/openpi0.5-rtc-reward-learning"
    local_checkpoint: str | pathlib.Path | None = None
    remote_checkpoint_dir: str = "/data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/latest"


def _local_raw_root(args: TransferArgs) -> pathlib.Path:
    return pathlib.Path(args.local_root) / "raw_from_103"


def _dir_arg(path: pathlib.Path) -> str:
    return str(path) + "/"


def build_pull_commands(args: TransferArgs) -> list[list[str]]:
    raw_root = _local_raw_root(args)
    return [
        ["mkdir", "-p", str(raw_root)],
        [
            "rsync",
            "-a",
            "--info=progress2",
            f"{args.remote}:{args.remote_data_root}/rollouts/key_regions/",
            _dir_arg(raw_root / "rollouts" / "key_regions"),
        ],
        [
            "rsync",
            "-a",
            "--info=progress2",
            f"{args.remote}:{args.remote_data_root}/replay/rlt_key_regions/",
            _dir_arg(raw_root / "replay" / "rlt_key_regions"),
        ],
        [
            "rsync",
            "-a",
            "--info=progress2",
            f"{args.remote}:{args.remote_project}/voice_assistant_web/backend/state/",
            _dir_arg(raw_root / "state"),
        ],
    ]


def build_deploy_commands(args: TransferArgs) -> list[list[str]]:
    if args.local_checkpoint is None:
        raise ValueError("--local-checkpoint is required for deploy")
    local_checkpoint = pathlib.Path(args.local_checkpoint)
    return [
        ["ssh", args.remote, f"mkdir -p {shlex.quote(args.remote_checkpoint_dir)}"],
        [
            "rsync",
            "-a",
            "--info=progress2",
            str(local_checkpoint) + "/",
            f"{args.remote}:{args.remote_checkpoint_dir}/",
        ],
    ]


def _run(commands: list[list[str]], *, execute: bool) -> None:
    for command in commands:
        print(shlex.join(command))
        if execute:
            subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull RLT key-region data locally or deploy offline checkpoints to 103.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull = subparsers.add_parser("pull")
    pull.add_argument("--remote", default="eii@192.168.1.103")
    pull.add_argument("--local-root", default="local_rlt_data")
    pull.add_argument("--remote-project", default="~/openpi0.5-rtc-reward-learning")
    pull.add_argument("--remote-data-root", default="/data/openpi0.5-rtc-reward-learning")
    pull.add_argument("--execute", action="store_true")

    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("--remote", default="eii@192.168.1.103")
    deploy.add_argument("--local-checkpoint", required=True)
    deploy.add_argument(
        "--remote-checkpoint-dir",
        default="/data/openpi0.5-rtc-reward-learning/rlt_offline_checkpoints/latest",
    )
    deploy.add_argument("--execute", action="store_true")

    namespace = parser.parse_args()
    args = TransferArgs(**{key: value for key, value in vars(namespace).items() if key not in {"command", "execute"}})
    commands = build_pull_commands(args) if namespace.command == "pull" else build_deploy_commands(args)
    if not namespace.execute:
        print(json.dumps({"dry_run": True, "command_count": len(commands)}, sort_keys=True))
    _run(commands, execute=namespace.execute)


if __name__ == "__main__":
    main()

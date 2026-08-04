#!/usr/bin/env python3
import os
from pathlib import Path
import shlex
import sys


args = sys.argv[1:]
log_path = Path(os.environ["FAKE_DOCKER_LOG"])
with log_path.open("a", encoding="utf-8") as stream:
    one_line_args = [
        " ".join(argument.splitlines())
        for argument in args
    ]
    stream.write(shlex.join(one_line_args) + "\n")


def env(name, default=""):
    return os.environ.get(name, default)


if not args:
    raise SystemExit(2)

command = args[0]
joined = " ".join(args)

if command in {"info", "version"}:
    raise SystemExit(0)

if command == "image" and args[1:2] == ["inspect"]:
    raise SystemExit(0)

if command == "inspect":
    if env("FAKE_CONTAINER", "running") == "absent":
        raise SystemExit(1)
    if "--format" not in args:
        print("{}")
        raise SystemExit(0)
    template = args[args.index("--format") + 1]
    values = {
        "{{.State.Status}}": env("FAKE_CONTAINER", "running"),
        "{{.Config.Image}}": env(
            "FAKE_IMAGE",
            "lyl472324464/robot:aloha-2.0",
        ),
        "{{.HostConfig.Memory}}": env(
            "FAKE_MEMORY",
            "51539607552",
        ),
        "{{.HostConfig.NetworkMode}}": env(
            "FAKE_NETWORK",
            "host",
        ),
        "{{.HostConfig.Privileged}}": env(
            "FAKE_PRIVILEGED",
            "true",
        ),
        "{{.HostConfig.Runtime}}": env(
            "FAKE_RUNTIME",
            "nvidia",
        ),
    }
    if template in values:
        print(values[template])
    elif ".Config.Env" in template:
        print(
            "NVIDIA_VISIBLE_DEVICES="
            + env("FAKE_VISIBLE_DEVICES", "all")
        )
        print(
            "NVIDIA_DRIVER_CAPABILITIES="
            + env(
                "FAKE_DRIVER_CAPABILITIES",
                "compute,utility,video",
            )
        )
    elif ".Mounts" in template:
        repo = env("FAKE_REPO", "/home/eii/aloha-2.0")
        print(f"{repo}|/root/interbotix_ws/src/aloha")
        print(env("FAKE_DEV_SOURCE", "/dev") + "|/dev")
    raise SystemExit(0)

if command == "run":
    print("fake-container-id")
    raise SystemExit(0)

if command == "exec":
    if "record_episodes_copy.py" in joined and "pgrep" not in joined:
        raise SystemExit(int(env("FAKE_RECORDER_EXIT", "0")))
    if "pgrep" in joined and "ecord_episodes_copy.py" in joined:
        recorder = env("FAKE_RECORDER")
        if recorder:
            print(recorder)
            raise SystemExit(0)
        raise SystemExit(1)
    if "pgrep" in joined and "loha_bringup.launch.py" in joined:
        count = int(env("FAKE_BRINGUP_COUNT", "1"))
        for index in range(count):
            print(
                f"{100 + index} ros2 launch aloha "
                "aloha_bringup.launch.py robot:=aloha_stationary"
            )
        raise SystemExit(0 if count else 1)
    if "--classify-graph" in joined:
        print(env("FAKE_GRAPH", "complete"))
        raise SystemExit(0)
    if "check_collect_ready.py" in joined:
        raise SystemExit(int(env("FAKE_READY_EXIT", "0")))
    if "nvidia-smi" in joined or "h264_nvenc" in joined:
        raise SystemExit(int(env("FAKE_NVENC_EXIT", "0")))
    if "test -e" in joined:
        raise SystemExit(int(env("FAKE_PEDAL_EXIT", "0")))
    raise SystemExit(0)

raise SystemExit(2)

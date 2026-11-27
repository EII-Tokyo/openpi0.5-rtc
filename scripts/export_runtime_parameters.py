"""Publish a read-only snapshot of the current ALOHA launch for the web drawer."""

import argparse
import ast
import datetime
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = (
    "OPENPI_DEFER_INITIAL_HOME", "OPENPI_RINSE_RUNTIME",
    "ALOHA_HOME_POSITION_TOLERANCE", "ALOHA_HOME_DEPARTURE_TOLERANCE",
    "ALOHA_HOME_VELOCITY_TOLERANCE", "ALOHA_HOME_STABLE_SECONDS",
    "ALOHA_ROLLOUT_DIR", "ALOHA_FOOT_SWITCH", "ALOHA_TASK_1_PROMPT", "ALOHA_TASK_2_PROMPT",
)
PARSE_ARGS = """
from dataclasses import asdict
from enum import Enum
import importlib.util, json, sys
import tyro

script, *arguments = sys.argv[1:]
spec = importlib.util.spec_from_file_location('runtime_parameter_inspection', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
args = tyro.cli(module.Args, args=arguments)
print(json.dumps(asdict(args), default=lambda obj: obj.value if isinstance(obj, Enum) else str(obj)))
"""


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE)


def inspect(name):
    container, = json.loads(command("docker", "inspect", name))
    if not container["State"]["Running"]:
        raise RuntimeError(f"{name} is not running")
    return container


def parsed_args(name, executable, config):
    script = config["Cmd"] if config["Cmd"][0].endswith(".py") else config["Cmd"][1:]
    raw = command("docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", name,
                  executable, "-c", PARSE_ARGS, *script)
    return json.loads(raw.splitlines()[-1])


def checkout(container):
    source = next(m["Source"] for m in container["Mounts"] if m["Destination"] == "/app")
    return {
        "branch": command("git", "-C", source, "branch", "--show-current").strip() or None,
        "checkout_commit": command("git", "-C", source, "rev-parse", "HEAD").strip(),
    }


def startup_metadata(runtime_name, started):
    start = datetime.datetime.fromisoformat(started.replace("Z", "+00:00"))
    end = start + datetime.timedelta(minutes=2)
    logs = subprocess.run(
        ["docker", "logs", "--since", started, "--until", end.isoformat(), runtime_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
    ).stdout
    marker = "Server metadata: "
    line = next((line for line in logs.splitlines() if marker in line), None)
    if line is None:
        raise RuntimeError("Runtime startup policy metadata is unavailable")
    return ast.literal_eval(line.split(marker, 1)[1])["runtime"]


def collect(project):
    runtime_name = f"{project}-runtime-1"
    server_name = f"{project}-openpi_server-1"
    runtime = inspect(runtime_name)
    server = inspect(server_name)
    runtime_args = parsed_args(runtime_name, "python3", runtime["Config"])
    server_args = parsed_args(server_name, "/.venv/bin/python3", server["Config"])
    metadata = startup_metadata(runtime_name, runtime["State"]["StartedAt"])
    video = metadata["video_input"]
    runtime_args.update(video_memory_num_frames=video["num_frames"],
                        video_memory_stride_seconds=video["stride_seconds"])
    env = dict(item.split("=", 1) for item in runtime["Config"]["Env"] if "=" in item)
    server_env = dict(item.split("=", 1) for item in server["Config"]["Env"] if "=" in item)
    high_level = metadata.get("high_level", {})
    for name, old in ((runtime_name, runtime), (server_name, server)):
        new = inspect(name)
        if (old["Id"], old["State"]["StartedAt"]) != (new["Id"], new["State"]["StartedAt"]):
            raise RuntimeError(f"{name} changed while collecting parameters")
    return {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_started_at": runtime["State"]["StartedAt"],
        "server_started_at": server["State"]["StartedAt"],
        "runtime": runtime_args,
        "policy_server": server_args,
        "checkpoint": {**video, **{key: metadata[key] for key in (
            "adapt_to_pi", "action_horizon", "action_dim", "joint_position_history", "task_catalog_version",
        )}},
        "sol": {**{key: high_level[key] for key in (
            "enabled", "model", "camera_keys", "image_history_seconds", "wire_protocol",
            "prompt_version", "archive_required", "reset_protocol",
            "request_interval_seconds", "reply_expiry_seconds",
        ) if key in high_level},
            **{key: server_env[key] for key in (
                "OPENPI_SOL_REQUEST_INTERVAL_SECONDS", "OPENPI_SOL_PERIODIC_1HZ",
                "OPENPI_SOL_SERVICE_TIER",
            ) if key in server_env and server_env[key]}},
        "runtime_environment": {key: env[key] for key in ENV_KEYS if key in env},
        "code": {"runtime": checkout(runtime), "policy_server": checkout(server)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="openpi05-rtc")
    parser.add_argument("--output", type=Path, default=ROOT / "voice_assistant_web/frontend/public/runtime-parameters.json")
    args = parser.parse_args()
    snapshot = collect(args.project)
    # Replace atomically so a browser never sees a partially written snapshot.
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as output:
        json.dump(snapshot, output, indent=2)
        output.write("\n")
    path = Path(output.name)
    path.chmod(0o644)
    path.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

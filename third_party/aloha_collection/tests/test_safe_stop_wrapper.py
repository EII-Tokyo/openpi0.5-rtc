import json
import os
from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SAFE_STOP = ROOT / "scripts" / "safe_stop_container.sh"


def _run_wrapper(
    tmp_path,
    observations,
    *,
    owner_alive=True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            log_path = Path(os.environ["FAKE_DOCKER_LOG"])
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(args) + "\\n")

            if args[:2] == ["ps", "-a"]:
                print("container-103")
            elif args[:2] == ["inspect", "--format"]:
                print("true")
            elif args[:3] == ["exec", "container-103", "pgrep"]:
                print("4242")
            elif args[:4] == ["exec", "container-103", "kill", "-INT"]:
                pass
            elif args[:3] == ["exec", "container-103", "python3"]:
                observations = json.loads(
                    os.environ["FAKE_SAFETY_OBSERVATIONS"]
                )
                counter_path = Path(os.environ["FAKE_DOCKER_COUNTER"])
                index = (
                    int(counter_path.read_text(encoding="utf-8"))
                    if counter_path.exists()
                    else 0
                )
                counter_path.write_text(str(index + 1), encoding="utf-8")
                item = observations[min(index, len(observations) - 1)]
                output = item["output"]
                expected_recovery_id = args[-1]
                fields = output.split("|")
                if (
                    item["status"] == 0
                    and expected_recovery_id != "-"
                    and len(fields) >= 2
                    and fields[1] != expected_recovery_id
                ):
                    print("recovery_id mismatch")
                    raise SystemExit(1)
                print(output)
                raise SystemExit(item["status"])
            elif args[:4] == ["exec", "container-103", "kill", "-0"]:
                raise SystemExit(
                    0 if os.environ["FAKE_OWNER_ALIVE"] == "1" else 1
                )
            elif args[:3] == ["stop", "--time", "120"]:
                print("container-103")
            else:
                print(f"unexpected fake docker call: {args}", file=sys.stderr)
                raise SystemExit(97)
            """
        ),
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    log_path = tmp_path / "docker.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_DOCKER_LOG": str(log_path),
            "FAKE_DOCKER_COUNTER": str(tmp_path / "counter"),
            "FAKE_OWNER_ALIVE": "1" if owner_alive else "0",
            "FAKE_SAFETY_OBSERVATIONS": json.dumps(observations),
            "ALOHA_SAFE_STOP_TIMEOUT_SECONDS": "2",
            "ALOHA_SAFE_STOP_POLL_SECONDS": "0",
        }
    )
    completed = subprocess.run(
        [str(SAFE_STOP), "aloha2-collect"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=2,
    )
    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    return completed, calls


def _observation(state, recovery_id, owner_pid, source, safe):
    return {
        "status": 0,
        "output": (
            f"{state}|{recovery_id}|{owner_pid}|{source}|"
            f"{str(safe).lower()}"
        ),
    }


def _stop_calls(calls):
    return [call for call in calls if call[:1] == ["stop"]]


def test_wrapper_stops_only_after_live_owner_safe_proof(tmp_path):
    completed, calls = _run_wrapper(
        tmp_path,
        [_observation("SAFE_TO_STOP", "rid-1", "4242", "recorder", True)],
    )

    assert completed.returncode == 0
    assert _stop_calls(calls) == [
        ["stop", "--time", "120", "container-103"]
    ]


def test_wrapper_refuses_unsafe_hold_and_dead_owner(tmp_path):
    unsafe, unsafe_calls = _run_wrapper(
        tmp_path / "unsafe",
        [_observation("UNSAFE_HOLD", "rid-1", "7001", "recorder", False)],
    )
    dead, dead_calls = _run_wrapper(
        tmp_path / "dead",
        [_observation("SAFE_TO_STOP", "rid-1", "7001", "recorder", True)],
        owner_alive=False,
    )

    assert unsafe.returncode == 3
    assert dead.returncode == 3
    assert not _stop_calls(unsafe_calls)
    assert not _stop_calls(dead_calls)


def test_wrapper_times_out_on_invalid_or_stale_schema(tmp_path):
    completed, calls = _run_wrapper(
        tmp_path,
        [{"status": 1, "output": "schema_version must be 2"}],
    )

    assert completed.returncode == 4
    assert not _stop_calls(calls)


def test_wrapper_rejects_recovery_id_change(tmp_path):
    completed, calls = _run_wrapper(
        tmp_path,
        [
            _observation(
                "RECOVERY_IN_PROGRESS",
                "rid-1",
                "4242",
                "recorder",
                False,
            ),
            _observation(
                "SAFE_TO_STOP",
                "rid-2",
                "7001",
                "standalone",
                True,
            ),
        ],
    )

    assert completed.returncode == 4
    assert not _stop_calls(calls)


def test_wrapper_allows_recorder_to_standalone_owner_transfer(tmp_path):
    completed, calls = _run_wrapper(
        tmp_path,
        [
            _observation(
                "EXTERNAL_RECOVERY_REQUIRED",
                "rid-1",
                "4242",
                "recorder",
                False,
            ),
            _observation(
                "SAFE_TO_STOP",
                "rid-1",
                "7001",
                "standalone",
                True,
            ),
        ],
    )

    assert completed.returncode == 0
    assert len(_stop_calls(calls)) == 1

#!/usr/bin/env python3
"""Coordinate synchronized ALOHA workers; the default transport is fake-only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from tools.aloha1_mapping.home_sleep_sync import build_run_identity
from tools.aloha1_mapping.home_sleep_sync import deadline_ns
from tools.aloha1_mapping.home_sleep_sync import validate_ready_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/aloha1_home_sleep_synchronized_real_sim.yaml"
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_sync_fake_run.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FakeWorker:
    """Deterministic worker used to prove coordinator safety without I/O."""

    def __init__(
        self,
        worker: str,
        *,
        ready: bool = True,
        manifest_sha256: str | None = None,
        late_at_index: int | None = None,
        operator_stop_at_index: int | None = None,
    ) -> None:
        self.worker = worker
        self.ready = ready
        self.manifest_sha256 = manifest_sha256
        self.late_at_index = late_at_index
        self.operator_stop_at_index = operator_stop_at_index
        self.arm_calls = 0
        self.abort_calls = 0
        self.publish_count = 0
        self.sample_indices: list[int] = []
        self._identity: Mapping[str, object] | None = None

    def prepare(self, identity: Mapping[str, object]) -> dict[str, object]:
        self._identity = identity
        return {
            "run_id": identity["run_id"],
            "manifest_sha256": self.manifest_sha256
            or identity["manifest_sha256"],
            "command_signature": identity["command_signature"],
            "worker": self.worker,
            "status": "READY" if self.ready else "PREPARED",
        }

    def arm(self, start_wall_time_ns: int) -> dict[str, object]:
        self.arm_calls += 1
        return {
            "worker": self.worker,
            "status": "ARMED",
            "start_wall_time_ns": int(start_wall_time_ns),
        }

    def step(
        self, sample: Mapping[str, object], *, sample_deadline_ns: int
    ) -> dict[str, object]:
        index = int(sample["index"])
        if self.operator_stop_at_index == index:
            return {
                "worker": self.worker,
                "status": "REAL_EXECUTION_ABORTED",
                "reason": "operator_stop",
            }
        if self.late_at_index == index:
            return {
                "worker": self.worker,
                "status": "ABORTED_DEADLINE_MISS",
                "reason": "deadline_miss_no_burst_catchup",
            }
        self.sample_indices.append(index)
        self.publish_count += 1
        return {
            "worker": self.worker,
            "status": "APPLIED",
            "sample_index": index,
            "sample_deadline_ns": int(sample_deadline_ns),
        }

    def abort(self, reason: str) -> dict[str, object]:
        self.abort_calls += 1
        return {"worker": self.worker, "status": "ABORTED", "reason": reason}

    def report(self) -> dict[str, object]:
        return {
            "worker": self.worker,
            "sample_indices": list(self.sample_indices),
            "applied_sample_count": len(self.sample_indices),
            "arm_calls": self.arm_calls,
            "abort_calls": self.abort_calls,
        }


def run_coordinator(
    *,
    identity: Mapping[str, object],
    workers: Mapping[str, FakeWorker],
    samples: Sequence[Mapping[str, object]],
    start_wall_time_ns: int = 1_000_000_000,
) -> dict[str, Any]:
    """Execute the deterministic prepare/ready/arm/run protocol."""

    expected_workers = tuple(str(item) for item in identity["workers"])
    if set(workers) != set(expected_workers):
        return _coordinator_report(
            "BLOCKED_WORKER_SET", identity, workers, commands_published=0
        )
    ready_records = {
        name: workers[name].prepare(identity) for name in expected_workers
    }
    if any(record["status"] != "READY" for record in ready_records.values()):
        return _coordinator_report(
            "BLOCKED_NOT_ALL_READY",
            identity,
            workers,
            commands_published=0,
            ready_records=ready_records,
        )
    identity_failures = {
        name: validate_ready_record(record, identity)
        for name, record in ready_records.items()
    }
    identity_failures = {
        name: failures for name, failures in identity_failures.items() if failures
    }
    if identity_failures:
        return _coordinator_report(
            "BLOCKED_IDENTITY_MISMATCH",
            identity,
            workers,
            commands_published=0,
            ready_records=ready_records,
            identity_failures=identity_failures,
        )

    for name in expected_workers:
        workers[name].arm(start_wall_time_ns)

    status = "PASS_FAKE_TRANSPORT"
    sample_period_ns = int(identity["sample_period_ns"])
    safety_first_order = ("real", "isaac", "cam_high")
    for sample in samples:
        sample_index = int(sample["index"])
        expected_deadline = deadline_ns(
            start_wall_time_ns, sample_index, sample_period_ns
        )
        for name in safety_first_order:
            result = workers[name].step(
                sample, sample_deadline_ns=expected_deadline
            )
            if result["status"] != "APPLIED":
                status = str(result["status"])
                for other_name, other in workers.items():
                    if other_name != name:
                        other.abort(status)
                break
        if status != "PASS_FAKE_TRANSPORT":
            break
    return _coordinator_report(
        status,
        identity,
        workers,
        commands_published=0,
        ready_records=ready_records,
    )


def _coordinator_report(
    status: str,
    identity: Mapping[str, object],
    workers: Mapping[str, FakeWorker],
    *,
    commands_published: int,
    ready_records: Mapping[str, Mapping[str, object]] | None = None,
    identity_failures: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "transport": "fake",
        "identity": dict(identity),
        "ready_records": dict(ready_records or {}),
        "identity_failures": dict(identity_failures or {}),
        "workers": {name: worker.report() for name, worker in workers.items()},
        "network_access_performed": False,
        "ros_transport_instantiated": False,
        "ssh_connection_opened": False,
        "serial_device_opened": False,
        "torque_changed": False,
        "commands_published_to_real_hardware": int(commands_published),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transport", choices=("fake",), default="fake")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["manifest"]["path"]
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != config["manifest"]["sha256"]:
        raise RuntimeError("configured manifest SHA-256 does not match the file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = build_run_identity(
        run_id=f"fake-{manifest_hash[:12]}",
        manifest_sha256=manifest_hash,
        command_signature=manifest["command_signature"],
        command_rate_hz=int(manifest["command_rate_hz"]),
    )
    workers = {
        name: FakeWorker(name) for name in ("isaac", "real", "cam_high")
    }
    report = run_coordinator(
        identity=identity, workers=workers, samples=manifest["samples"]
    )
    report["config"] = {
        "absolute_path": str(config_path),
        "sha256": _sha256(config_path),
    }
    report["manifest"] = {
        "absolute_path": str(manifest_path.resolve()),
        "sha256": manifest_hash,
        "sample_count": len(manifest["samples"]),
    }
    report["authorization"] = dict(config["authorization"])
    report["live_transport_available"] = False
    report["real_execution"] = "NOT_RUN_AUTHORIZATION_REQUIRED"
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "commands_published_to_real_hardware": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_FAKE_TRANSPORT" else 2


if __name__ == "__main__":
    raise SystemExit(main())

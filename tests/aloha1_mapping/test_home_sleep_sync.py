from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.home_sleep_sync import build_run_identity
from tools.aloha1_mapping.home_sleep_sync import classify_start_skew
from tools.aloha1_mapping.home_sleep_sync import deadline_ns
from tools.aloha1_mapping.home_sleep_sync import validate_ready_record
from tools.run_aloha1_home_sleep_sync import FakeWorker
from tools.run_aloha1_home_sleep_sync import run_coordinator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_home_sleep_synchronized_real_sim.yaml"


def test_build_run_identity_binds_manifest_and_workers() -> None:
    identity = build_run_identity(
        run_id="run-001",
        manifest_sha256="a" * 64,
        command_signature="b" * 64,
        command_rate_hz=50,
    )

    assert identity == {
        "schema_version": 1,
        "run_id": "run-001",
        "manifest_sha256": "a" * 64,
        "command_signature": "b" * 64,
        "command_rate_hz": 50,
        "sample_period_ns": 20_000_000,
        "workers": ["isaac", "real", "cam_high"],
    }


def test_run_identity_rejects_non_sha256_inputs() -> None:
    with pytest.raises(ValueError, match="manifest_sha256"):
        build_run_identity(
            run_id="run-001",
            manifest_sha256="short",
            command_signature="b" * 64,
            command_rate_hz=50,
        )


def test_deadline_uses_absolute_index_without_accumulated_sleep() -> None:
    assert deadline_ns(1_000_000_000, 1849, 20_000_000) == 37_980_000_000


def test_start_classification_uses_one_command_period() -> None:
    assert (
        classify_start_skew(20_000_000, sample_period_ns=20_000_000)
        == "SYNCHRONIZED_START_PASS"
    )
    assert (
        classify_start_skew(-20_000_000, sample_period_ns=20_000_000)
        == "SYNCHRONIZED_START_PASS"
    )
    assert (
        classify_start_skew(20_000_001, sample_period_ns=20_000_000)
        == "POST_ALIGNED_ONLY"
    )


def test_ready_record_must_match_frozen_identity() -> None:
    identity = build_run_identity(
        run_id="run-001",
        manifest_sha256="a" * 64,
        command_signature="b" * 64,
        command_rate_hz=50,
    )
    ready = {
        **identity,
        "worker": "real",
        "status": "READY",
    }
    assert validate_ready_record(ready, identity) == []

    ready["manifest_sha256"] = "c" * 64
    assert validate_ready_record(ready, identity) == ["manifest_sha256"]


def test_synchronized_config_is_fail_closed_and_bound_to_selected_sleep() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 1
    assert config["active_robot"] == "follower_left"
    assert config["manifest"]["command_signature"] == (
        "d481b71bc8d6160fae0bdc1b264715e782712565064bb18099f8a9a4883f499e"
    )
    assert config["manifest"]["sample_count"] == 1850
    assert config["timing"]["command_rate_hz"] == 50
    assert config["timing"]["synchronized_start_gate_ns"] == 20_000_000
    assert config["real"]["joint_states_topic"] == "/puppet_left/joint_states"
    assert config["real"]["command_topic_candidate"] == (
        "/puppet_left/commands/joint_group"
    )
    assert config["camera"]["topic"] == "/cam_high"
    assert config["camera"]["covers_complete_follower_left"] is True
    assert config["authorization"] == {
        "real_access_authorized": False,
        "real_motion_authorized": False,
    }


def _identity() -> dict[str, object]:
    return build_run_identity(
        run_id="run-001",
        manifest_sha256="a" * 64,
        command_signature="b" * 64,
        command_rate_hz=50,
    )


def _samples(count: int) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "cycle": 0 if index == 0 else 1,
            "segment": "initial_home_hold" if index == 0 else "cycle_01_home_to_sleep",
            "q_rad": [0.0] * 6,
        }
        for index in range(count)
    ]


def _ready_workers(*, real: FakeWorker | None = None) -> dict[str, FakeWorker]:
    return {
        "isaac": FakeWorker("isaac"),
        "real": real or FakeWorker("real"),
        "cam_high": FakeWorker("cam_high"),
    }


def test_coordinator_never_arms_before_all_workers_ready() -> None:
    workers = {
        "isaac": FakeWorker("isaac", ready=True),
        "real": FakeWorker("real", ready=False),
        "cam_high": FakeWorker("cam_high", ready=True),
    }

    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(3))

    assert report["status"] == "BLOCKED_NOT_ALL_READY"
    assert all(worker.arm_calls == 0 for worker in workers.values())


def test_manifest_mismatch_aborts_without_transport_publish() -> None:
    real = FakeWorker("real", manifest_sha256="c" * 64)

    report = run_coordinator(
        identity=_identity(), workers=_ready_workers(real=real), samples=_samples(3)
    )

    assert report["status"] == "BLOCKED_IDENTITY_MISMATCH"
    assert real.publish_count == 0


def test_fake_workers_execute_all_indices_once() -> None:
    workers = _ready_workers()

    report = run_coordinator(
        identity=_identity(), workers=workers, samples=_samples(1850)
    )

    assert report["status"] == "PASS_FAKE_TRANSPORT"
    assert report["workers"]["real"]["sample_indices"] == list(range(1850))
    assert workers["real"].publish_count == 1850
    assert report["network_access_performed"] is False
    assert report["commands_published_to_real_hardware"] == 0


def test_late_real_worker_never_bursts_missed_commands() -> None:
    real = FakeWorker("real", late_at_index=2)

    report = run_coordinator(
        identity=_identity(), workers=_ready_workers(real=real), samples=_samples(5)
    )

    assert report["status"] == "ABORTED_DEADLINE_MISS"
    assert real.sample_indices == [0, 1]
    assert real.publish_count == 2


def test_operator_stop_aborts_other_workers() -> None:
    workers = _ready_workers(real=FakeWorker("real", operator_stop_at_index=2))

    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(5))

    assert report["status"] == "REAL_EXECUTION_ABORTED"
    assert workers["isaac"].abort_calls == 1
    assert workers["cam_high"].abort_calls == 1

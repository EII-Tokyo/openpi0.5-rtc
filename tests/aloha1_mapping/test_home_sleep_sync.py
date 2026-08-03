from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.home_sleep_sync import build_run_identity
from tools.aloha1_mapping.home_sleep_sync import classify_start_skew
from tools.aloha1_mapping.home_sleep_sync import deadline_ns
from tools.aloha1_mapping.home_sleep_sync import validate_ready_record
from tools.build_aloha1_home_sleep_sync_preflight_report import classify_offline_readiness
from tools.run_aloha1_home_sleep_isaac_worker import build_isaac_worker_plan
from tools.run_aloha1_home_sleep_isaac_worker import build_validator_argv
from tools.run_aloha1_home_sleep_isaac_worker import frame_deadline_ns
from tools.run_aloha1_home_sleep_isaac_worker import frame_lateness_status
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
    assert classify_start_skew(20_000_000, sample_period_ns=20_000_000) == "SYNCHRONIZED_START_PASS"
    assert classify_start_skew(-20_000_000, sample_period_ns=20_000_000) == "SYNCHRONIZED_START_PASS"
    assert classify_start_skew(20_000_001, sample_period_ns=20_000_000) == "POST_ALIGNED_ONLY"


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


def test_synchronized_config_is_fail_closed_and_bound_to_runtime_sleep() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 1
    assert config["active_robot"] == "follower_left"
    assert config["manifest"]["command_signature"] == (
        "6e454405843d0a231bef75f35b471e2f07c60f277f76e5c9690a4f89d745ed47"
    )
    assert config["manifest"]["sample_count"] == 1850
    assert config["manifest"]["sequence_kind"] == "SLEEP_HOME_SLEEP"
    assert config["manifest"]["initial_pose_label"] == "runtime_measured_sleep"
    assert config["manifest"]["terminal_pose_label"] == "runtime_measured_sleep"
    assert config["timing"]["command_rate_hz"] == 50
    assert config["timing"]["synchronized_start_gate_ns"] == 20_000_000
    assert config["real"]["joint_states_topic"] == "/puppet_left/joint_states"
    assert config["real"]["command_topic_candidate"] == ("/puppet_left/commands/joint_group")
    assert config["camera"]["topic"] == "/cam_high"
    assert config["camera"]["covers_complete_follower_left"] is True
    assert config["authorization"] == {
        "real_access_authorized": False,
        "real_motion_authorized": False,
    }


def test_isaac_worker_plan_hash_pins_all_frozen_inputs(tmp_path: Path) -> None:
    stage = tmp_path / "stage.usda"
    manifest = tmp_path / "manifest.json"
    finger = tmp_path / "finger.usda"
    stage.write_text("stage", encoding="utf-8")
    manifest.write_text(
        '{"command_signature":"' + "b" * 64 + '","sample_count":1850}',
        encoding="utf-8",
    )
    finger.write_text("finger", encoding="utf-8")
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plan = build_isaac_worker_plan(
        run_id="run-isaac-001",
        stage=stage,
        stage_sha256=sha(stage),
        manifest=manifest,
        manifest_sha256=sha(manifest),
        finger_limit_layer=finger,
        finger_limit_sha256=sha(finger),
        command_signature="b" * 64,
        start_monotonic_ns=10_000_000_000,
        headless=True,
        gui_workspace=2,
    )

    assert plan["status"] == "READY"
    assert plan["worker"] == "isaac"
    assert plan["sample_count"] == 1850
    assert plan["gui_workspace"] == 2
    assert plan["stage"]["sha256"] == sha(stage)


def test_isaac_worker_plan_rejects_changed_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage.usda"
    manifest = tmp_path / "manifest.json"
    finger = tmp_path / "finger.usda"
    stage.write_text("changed", encoding="utf-8")
    manifest.write_text(
        '{"command_signature":"' + "b" * 64 + '","sample_count":1850}',
        encoding="utf-8",
    )
    finger.write_text("finger", encoding="utf-8")

    with pytest.raises(ValueError, match="stage SHA-256 mismatch"):
        build_isaac_worker_plan(
            run_id="run-isaac-001",
            stage=stage,
            stage_sha256="0" * 64,
            manifest=manifest,
            manifest_sha256=(__import__("hashlib").sha256(manifest.read_bytes()).hexdigest()),
            finger_limit_layer=finger,
            finger_limit_sha256=(__import__("hashlib").sha256(finger.read_bytes()).hexdigest()),
            command_signature="b" * 64,
            start_monotonic_ns=10_000_000_000,
            headless=True,
            gui_workspace=2,
        )


def test_isaac_worker_builds_validator_command_with_future_start(tmp_path: Path) -> None:
    args = build_validator_argv(
        python_executable=Path("/project/.venv_issac/bin/python"),
        validator=Path("/project/tools/validate_aloha1_home_sleep_digital.py"),
        stage=Path("/project/stage.usda"),
        stage_sha256="a" * 64,
        manifest=Path("/project/manifest.json"),
        manifest_sha256="b" * 64,
        finger_limit_layer=Path("/project/finger.usda"),
        finger_limit_sha256="c" * 64,
        output=tmp_path / "report.json",
        telemetry=tmp_path / "telemetry.csv",
        repeat_index=1,
        run_id="run-isaac-001",
        start_monotonic_ns=10_000_000_000,
        headless=True,
    )

    assert "--realtime-pacing" in args
    assert args[args.index("--start-monotonic-ns") + 1] == "10000000000"
    assert args[args.index("--run-id") + 1] == "run-isaac-001"
    assert "--headless" in args


def test_isaac_frame_deadlines_do_not_accumulate_rounding_error() -> None:
    start = 10_000_000_000

    assert frame_deadline_ns(start, frame_index=0, physics_rate_hz=60) == start
    assert frame_deadline_ns(start, frame_index=60, physics_rate_hz=60) == (start + 1_000_000_000)


def test_isaac_worker_aborts_instead_of_bursting_late_frames() -> None:
    assert frame_lateness_status(16_666_666, physics_rate_hz=60) == "ON_TIME"
    assert frame_lateness_status(16_666_667, physics_rate_hz=60) == "ABORTED_DEADLINE_MISS"


def test_offline_readiness_never_claims_real_execution() -> None:
    report = classify_offline_readiness(
        fake_status="PASS_FAKE_TRANSPORT",
        isaac_statuses=["PASS", "PASS", "PASS"],
        isaac_signatures=["a" * 64, "a" * 64, "a" * 64],
        ros_source_audit_status="NOT_RUN_AUTHORIZATION_REQUIRED",
        prohibited_side_effects_detected=False,
    )

    assert report["status"] == "READY_FOR_SUPERVISED_REAL_EXECUTION"
    assert report["real_execution"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert "real_motion_authorized" in report["remaining_live_gates"]


def test_offline_readiness_blocks_on_isaac_mismatch_or_side_effect() -> None:
    report = classify_offline_readiness(
        fake_status="PASS_FAKE_TRANSPORT",
        isaac_statuses=["PASS", "FAIL"],
        isaac_signatures=["a" * 64, "b" * 64],
        ros_source_audit_status="NOT_RUN_AUTHORIZATION_REQUIRED",
        prohibited_side_effects_detected=True,
    )

    assert report["status"] == "BLOCKED_OFFLINE_PREFLIGHT"
    assert set(report["failed_gates"]) == {
        "isaac_process_statuses",
        "isaac_deterministic_signature",
        "prohibited_side_effects_absent",
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

    report = run_coordinator(identity=_identity(), workers=_ready_workers(real=real), samples=_samples(3))

    assert report["status"] == "BLOCKED_IDENTITY_MISMATCH"
    assert real.publish_count == 0


def test_fake_workers_execute_all_indices_once() -> None:
    workers = _ready_workers()

    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(1850))

    assert report["status"] == "PASS_FAKE_TRANSPORT"
    assert report["workers"]["real"]["sample_indices"] == list(range(1850))
    assert workers["real"].publish_count == 1850
    assert report["network_access_performed"] is False
    assert report["commands_published_to_real_hardware"] == 0


def test_late_real_worker_never_bursts_missed_commands() -> None:
    real = FakeWorker("real", late_at_index=2)

    report = run_coordinator(identity=_identity(), workers=_ready_workers(real=real), samples=_samples(5))

    assert report["status"] == "ABORTED_DEADLINE_MISS"
    assert real.sample_indices == [0, 1]
    assert real.publish_count == 2


def test_operator_stop_aborts_other_workers() -> None:
    workers = _ready_workers(real=FakeWorker("real", operator_stop_at_index=2))

    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(5))

    assert report["status"] == "REAL_EXECUTION_ABORTED"
    assert workers["isaac"].abort_calls == 1
    assert workers["cam_high"].abort_calls == 1

from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.home_sleep_sync import build_run_identity
from tools.aloha1_mapping.home_sleep_sync import classify_start_skew
from tools.aloha1_mapping.home_sleep_sync import deadline_ns
from tools.aloha1_mapping.home_sleep_sync import validate_ready_record

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

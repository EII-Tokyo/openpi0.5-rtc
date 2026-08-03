from tools.aloha1_mapping.home_sleep_real_safety import build_dry_run_plan
from tools.aloha1_mapping.home_sleep_real_safety import validate_real_execution_gate
from tools.preflight_aloha1_home_sleep_real import build_preflight_report
from tools.run_aloha1_home_sleep_real import build_runner_report


def _passing_gates() -> dict:
    return {
        "execute_real": True,
        "robot": "follower_left",
        "manifest_sha_matches": True,
        "digital_report_status": "PASS",
        "preflight_report_status": "PASS",
        "preflight_manifest_sha_matches": True,
        "real_motion_authorized": True,
        "operator_workspace_clear": True,
        "stop_control_ready": True,
    }


def test_real_execution_gate_requires_every_explicit_safety_condition() -> None:
    gates = _passing_gates()
    assert validate_real_execution_gate(gates)["status"] == "PASS"

    for key in gates:
        rejected = dict(gates)
        rejected[key] = False if key != "robot" else "follower_right"
        result = validate_real_execution_gate(rejected)
        assert result["status"] == "BLOCKED"
        assert key in result["failed_gates"]


def test_dry_run_plan_has_no_live_transport_or_device_side_effects() -> None:
    plan = build_dry_run_plan(
        manifest_sha256="manifest",
        digital_status="FAIL",
        sample_count=1850,
    )

    assert plan["mode"] == "DRY_RUN"
    assert plan["status"] == "NOT_RUN_DIGITAL_GATE_FAILED"
    assert plan["network_access_performed"] is False
    assert plan["ros_transport_instantiated"] is False
    assert plan["serial_device_opened"] is False
    assert plan["torque_changed"] is False
    assert plan["commands_published"] == 0
    assert plan["planned_samples"] == 1850


def test_preflight_is_not_run_when_digital_gate_failed() -> None:
    report = build_preflight_report(
        digital_report={"status": "FAIL", "classification": "limit conflict"},
        manifest={"sample_count": 1850, "command_signature": "command"},
        digital_report_sha256="digital",
        manifest_sha256="manifest",
    )

    assert report["status"] == "NOT_RUN_DIGITAL_GATE_FAILED"
    assert report["read_only_remote_checks_performed"] is False
    assert report["real_execution_authorized"] is False


def test_offline_preflight_after_digital_pass_still_requires_live_authorization() -> None:
    report = build_preflight_report(
        digital_report={"status": "PASS", "classification": "DIGITAL_HOME_SLEEP_VERIFIED"},
        manifest={"sample_count": 1850, "command_signature": "command"},
        digital_report_sha256="digital",
        manifest_sha256="manifest",
    )

    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert report["boundary"] == (
        "Digital qualification passed; real access and motion remain blocked until a separate "
        "explicit authorization and all live operator safety checks pass."
    )
    assert report["network_access_performed"] is False
    assert report["commands_published"] == 0


def test_real_runner_default_is_literal_dry_run_even_with_pass_reports() -> None:
    report = build_runner_report(
        execute_real=False,
        robot="follower_left",
        digital_status="PASS",
        preflight_status="PASS",
        manifest_sha_matches=True,
        preflight_manifest_sha_matches=True,
        authorization={},
        manifest_sha256="manifest",
        sample_count=1850,
    )

    assert report["mode"] == "DRY_RUN"
    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert report["commands_published"] == 0
    assert report["network_access_performed"] is False

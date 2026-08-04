import json

import pytest

from tools.aloha1_mapping.dual_real_publisher import build_dual_command
from tools.aloha1_mapping.dual_real_publisher import build_dual_dry_run_report
from tools.aloha1_mapping.dual_real_publisher import validate_dual_manifest


def _manifest() -> dict[str, object]:
    samples = [
        {"index": 0, "time_ns": 0, "cycle": 0, "segment": "sleep_hold", "q_rad": [0.0] * 6},
        {"index": 1, "time_ns": 20_000_000, "cycle": 1, "segment": "sleep_to_home", "q_rad": [0.1] * 6},
    ]
    return {
        "sample_count": 2,
        "samples": samples,
        "command_rate_hz": 50,
        "joint_order": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        "sleep_rad": [0.0] * 6,
        "home_rad": [0.1] * 6,
        "command_signature": "sig",
    }


def test_validate_dual_manifest_requires_same_sample_clock_and_targets() -> None:
    manifest = _manifest()
    validated = validate_dual_manifest(manifest, manifest)
    assert validated["sample_count"] == 2
    assert validated["command_rate_hz"] == 50
    assert validated["left_command_signature"] == "sig"


def test_validate_dual_manifest_rejects_different_sample_times() -> None:
    left = _manifest()
    right = _manifest()
    right["samples"] = [*right["samples"]]  # type: ignore[index]
    right["samples"][1] = {**right["samples"][1], "time_ns": 40_000_000}  # type: ignore[index]
    with pytest.raises(ValueError, match="sample clock"):
        validate_dual_manifest(left, right)


def test_build_dual_command_mentions_both_roles_and_no_implicit_publish() -> None:
    command = build_dual_command(
        left_manifest="/app/left.json",
        right_manifest="/app/right.json",
        output="/app/out.json",
    )
    text = json.dumps(command)
    assert "puppet_left" in text
    assert "puppet_right" in text
    assert "--execute-real" not in text
    assert "--allow-dual-real-motion" not in text


def test_dual_dry_run_report_is_fail_closed() -> None:
    report = build_dual_dry_run_report(left_sha256="left", right_sha256="right", sample_count=2)
    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert report["commands_published"] == {"puppet_left": 0, "puppet_right": 0}
    assert report["ros_transport_instantiated"] is False
    assert report["real_motion_authorized"] is False

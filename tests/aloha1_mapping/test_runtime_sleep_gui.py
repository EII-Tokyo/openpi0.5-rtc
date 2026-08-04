from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tools.open_aloha1_runtime_sleep_gui import build_ready_report
from tools.open_aloha1_runtime_sleep_gui import load_verified_inputs
from tools.open_aloha1_runtime_sleep_gui import main
from tools.open_aloha1_runtime_sleep_gui import resolve_full_experience


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    finger = tmp_path / "finger.usda"
    finger.write_text("#usda 1.0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sequence_kind": "SLEEP_HOME_SLEEP",
                "initial_pose_label": "runtime_measured_sleep",
                "terminal_pose_label": "runtime_measured_sleep",
                "initial_arm_rad": [0.0, -1.8, 1.6, 0.0, -1.8, 0.0],
                "physics_rate_hz": 60,
                "sample_count": 1,
                "command_signature": "fixture-command-signature",
                "samples": [{"index": 0, "q_rad": [0.0] * 6}],
                "diagnostic_limit_override": {
                    "classification": "DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET",
                    "changes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return stage, manifest, finger


def test_load_verified_inputs_accepts_only_runtime_sleep_contract(tmp_path: Path) -> None:
    stage, manifest, finger = _inputs(tmp_path)

    result = load_verified_inputs(
        stage=stage,
        stage_sha256=_sha256(stage),
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        finger_limit_layer=finger,
        finger_limit_sha256=_sha256(finger),
    )

    assert result["manifest"]["sequence_kind"] == "SLEEP_HOME_SLEEP"
    assert result["manifest"]["initial_pose_label"] == "runtime_measured_sleep"
    assert result["hashes"]["stage"] == _sha256(stage)


def test_load_verified_inputs_rejects_hash_mismatch(tmp_path: Path) -> None:
    stage, manifest, finger = _inputs(tmp_path)

    with pytest.raises(ValueError, match="Stage SHA-256 mismatch"):
        load_verified_inputs(
            stage=stage,
            stage_sha256="0" * 64,
            manifest=manifest,
            manifest_sha256=_sha256(manifest),
            finger_limit_layer=finger,
            finger_limit_sha256=_sha256(finger),
        )


def test_ready_report_requires_workspace_two_paused_and_sleep_readback(tmp_path: Path) -> None:
    stage, manifest, finger = _inputs(tmp_path)
    inputs = load_verified_inputs(
        stage=stage,
        stage_sha256=_sha256(stage),
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        finger_limit_layer=finger,
        finger_limit_sha256=_sha256(finger),
    )
    target = inputs["manifest"]["initial_arm_rad"]
    readback = [value + 0.001 for value in target]

    report = build_ready_report(
        inputs=inputs,
        runtime={"isaac_sim": "5.1.0.0", "kit": "107.3.3", "physx": "107.3.26"},
        runtime_pid=123,
        window_id="456",
        workspace_number=2,
        workspace_move_passed=True,
        active_workspace_before=1,
        active_workspace_after=1,
        timeline_paused=True,
        target_arm_rad=target,
        readback_arm_rad=readback,
        dof_order=[
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
            "left_finger",
            "right_finger",
        ],
        stage_hash_after=_sha256(stage),
        session_layers=["anon:runtime_sleep", str(finger.resolve())],
    )

    assert report["status"] == "READY_FOR_USER_REVIEW"
    assert report["gates"]["active_workspace_unchanged"] is True
    assert report["maximum_sleep_error_rad"] == pytest.approx(0.001)
    assert report["real_motion_commands"] == 0
    assert report["source_or_final_asset_modified"] is False
    assert "samples" not in report["inputs"]["manifest"]
    assert report["inputs"]["manifest"]["sequence_kind"] == "SLEEP_HOME_SLEEP"


def test_ready_report_fails_closed_when_timeline_is_playing(tmp_path: Path) -> None:
    stage, manifest, finger = _inputs(tmp_path)
    inputs = load_verified_inputs(
        stage=stage,
        stage_sha256=_sha256(stage),
        manifest=manifest,
        manifest_sha256=_sha256(manifest),
        finger_limit_layer=finger,
        finger_limit_sha256=_sha256(finger),
    )
    target = inputs["manifest"]["initial_arm_rad"]

    report = build_ready_report(
        inputs=inputs,
        runtime={"isaac_sim": "5.1.0.0", "kit": "107.3.3", "physx": "107.3.26"},
        runtime_pid=123,
        window_id="456",
        workspace_number=2,
        workspace_move_passed=True,
        active_workspace_before=1,
        active_workspace_after=1,
        timeline_paused=False,
        target_arm_rad=target,
        readback_arm_rad=target,
        dof_order=[
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
            "left_finger",
            "right_finger",
        ],
        stage_hash_after=_sha256(stage),
        session_layers=[],
    )

    assert report["status"] == "FAIL_NOT_READY"
    assert report["gates"]["timeline_paused"] is False


def test_gui_runtime_version_readback_uses_kit_application() -> None:
    source = inspect.getsource(main)

    assert "runtime=_runtime_versions(kit_app)" in source


def test_gui_uses_installed_full_kit_experience() -> None:
    experience = resolve_full_experience()

    assert experience.name == "isaacsim.exp.full.kit"
    assert experience.is_file()


def test_gui_right_arm_uses_legal_runtime_sleep_candidate() -> None:
    source = inspect.getsource(main)

    assert "RIGHT_RUNTIME_LEGAL_SLEEP_CANDIDATE" in inspect.getsource(__import__(
        "tools.open_aloha1_runtime_sleep_gui", fromlist=["RIGHT_RUNTIME_SLEEP_SOURCE"]
    ))
    assert "right_button_samples" in source
    assert "right_target_arm_rad" in source

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_correct_finger_task5_entrypoint_reuses_frozen_trial_and_captures() -> None:
    source = (
        PROJECT_ROOT / "tools/validate_aloha1_correct_finger_task5.py"
    ).read_text(encoding="utf-8")
    runtime = (
        PROJECT_ROOT / "tools/validate_aloha1_gripper_collider_ab.py"
    ).read_text(encoding="utf-8")

    assert "_run_trial(" in source
    assert "repeats_per_robot" in source
    assert "verify_correct_finger_sources" in source
    assert "gripper_correct_finger_task5_trials" in source
    assert "screenshot_context=" in source
    assert "set_solve_articulation_contact_last(True)" in runtime
    assert "world.render()" in runtime
    assert "save_camera_rgba_png" in runtime
    assert 'screenshot_context["fixed_camera_target_world_m"]' in runtime
    assert "open_with_bottle_isometric" in runtime
    assert "SurfaceGripper" not in source


def test_correct_finger_task5_report_is_complete_and_keeps_default_asset() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_correct_finger_task5.json"
        ).read_text(encoding="utf-8")
    )

    assert report["experiment_execution_status"] == "PASS"
    assert report["repeats_per_robot"] >= 20
    assert set(report["groups"]) == {
        "hull_current",
        "decomposition_current",
    }
    assert report["fresh_reset_per_trial"] is True
    assert report["default_asset_collider_modified"] is False
    assert report["task8"] == "NOT_RUN"
    for group in report["groups"].values():
        assert group["combined"]["trial_count"] >= 40
        assert group["combined"]["complete"] is True
        assert group["screenshots"]["status"] == "PASS"


def test_task5_separates_contact_and_hold_gates() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/gripper_correct_finger_task5.json"
        ).read_text(encoding="utf-8")
    )
    gates = report["task5_gates"]

    assert set(gates) == {
        "finger_motion_direction",
        "aperture_monotonicity",
        "mimic_accuracy",
        "bilateral_contact_establishment",
        "contact_persistence",
        "penetration",
        "unexpected_internal_collision",
        "static_bottle_hold",
        "determinism",
        "screenshots",
    }
    assert gates["bilateral_contact_establishment"]["status"] in {
        "PASS",
        "FAIL",
    }
    assert gates["static_bottle_hold"]["status"] in {"PASS", "FAIL"}
    assert (
        gates["bilateral_contact_establishment"]["status"]
        != "STATIC_HOLD_PASS"
    )


def test_task5_screenshot_manifest_has_absolute_contact_and_hold_images() -> None:
    manifest = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_screenshot_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "PASS"
    phases = {capture["phase"] for capture in manifest["captures"]}
    assert {"asset_preflight", "bilateral_contact", "release_hold"} <= phases
    for capture in manifest["captures"]:
        assert Path(capture["absolute_path"]).is_absolute()
        assert Path(capture["absolute_path"]).is_file()
        assert len(capture["file_sha256"]) == 64
        assert len(capture["decoded_pixel_sha256"]) == 64


def test_correct_finger_mimic_failure_is_classified_without_greenwashing() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/"
            "gripper_correct_finger_mimic_classification.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "FAIL"
    assert (
        report["classification"]
        == "RUNTIME_MIMIC_READBACK_AND_LIMIT_OVERSHOOT"
    )
    assert report["gate"]["maximum_sampled_residual_m"] > report["gate"][
        "tolerance_m"
    ]
    assert report["interpretation"]["static_hold_cause"] is False
    assert report["interpretation"]["parameter_tuning_performed"] is False
    assert report["task8"] == "NOT_RUN"

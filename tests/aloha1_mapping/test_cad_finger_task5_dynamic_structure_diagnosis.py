from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tools/finalize_aloha_viper_cad_finger_task5_dynamic_structure.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("task5_dynamic_diagnosis", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_extracts_numeric_gate_evidence() -> None:
    module = _load_module()
    report = {
        "status": "PASS",
        "gates": {"no_bottle": True},
        "stage": {
            "absolute_path": "/tmp/diagnostic.usda",
            "sha256_before": "a" * 64,
            "sha256_after": "a" * 64,
        },
        "drive_readback": {
            "left": {"max_force": 5.0},
            "right": {"max_force": 5.0},
        },
        "arm_drive_readback": {
            "waist": {"max_force": 10.0},
        },
        "trajectories": [
            {
                "base_translation_drift_m": 0.00002,
                "maximum_arm_dof_drift": 0.0001,
                "intended_joint_results": [
                    {
                        "direction_correct": True,
                        "final_error_m": 0.00001,
                    }
                ],
                "non_target_finger_results": [{"drift_m": 0.00002}],
            }
        ],
    }

    metrics = module.metrics_from_report(report)

    assert metrics["status"] == "PASS"
    assert metrics["maximum_base_translation_drift_m"] == 0.00002
    assert metrics["maximum_arm_dof_drift"] == 0.0001
    assert metrics["maximum_intended_final_error_m"] == 0.00001
    assert metrics["maximum_non_target_finger_drift_m"] == 0.00002
    assert metrics["stage_immutable"] is True


def test_main_opens_bottle_gate_after_viewport_replay_review_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    output = tmp_path / "diagnosis.json"
    output_md = tmp_path / "diagnosis.md"
    blocker = tmp_path / "blocker.json"
    blocker_md = tmp_path / "blocker.md"
    monkeypatch.setattr(module, "OUTPUT", output)
    monkeypatch.setattr(module, "OUTPUT_MD", output_md)
    monkeypatch.setattr(module, "BLOCKER_OUTPUT", blocker)
    monkeypatch.setattr(module, "BLOCKER_OUTPUT_MD", blocker_md)

    assert module.main() == 0

    diagnosis = json.loads(output.read_text(encoding="utf-8"))
    screenshot_blocker = json.loads(blocker.read_text(encoding="utf-8"))
    assert diagnosis["status"] == "PASS"
    assert diagnosis["numeric_structure_gate"] == "PASS"
    assert diagnosis["visual_runtime_gate"] == (
        "PASS_AUXILIARY_RUNTIME_READBACK_REPLAY"
    )
    assert diagnosis["scope"]["bottle_contact_grasp"] == "NOT_RUN"
    assert diagnosis["scope"]["task8"] == "NOT_RUN"
    assert diagnosis["scope"]["default_or_final_asset_modified"] is False
    assert screenshot_blocker["status"] == (
        "RESOLVED_WITH_ALTERNATE_VIEWPORT_BACKEND"
    )
    assert screenshot_blocker["blocker_code"] == (
        "HARD_BLOCKER_RUNTIME_CAMERA_EMPTY_BUFFER_ON_ROOT_FRAME_DIAGNOSTIC"
    )
    assert len(screenshot_blocker["attempts"]) == 3
    assert all(not item["accepted_capture"] for item in screenshot_blocker["attempts"])
    assert screenshot_blocker["resolution"]["status"] == "PASS"
    assert screenshot_blocker["consequences"]["bottle_test_allowed"] is True

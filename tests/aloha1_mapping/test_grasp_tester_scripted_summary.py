from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path("tools/summarize_aloha1_grasp_tester_scripted_equivalent.py")
RUN_NAMES = (
    "A_run1",
    "A_run2",
    "A_run3",
    "B_run10",
    "B_run11",
    "B_run12",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "grasp_tester_scripted_summary",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(
    root: Path,
    run_name: str,
    *,
    signature: str,
    physics_steps: int,
    active_joints: list[str],
    export_bytes: bytes,
    gate_rerun: bool,
) -> None:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    contacts = [
        {
            "event_type": "ContactEventType.CONTACT_FOUND",
            "physics_step": 1,
            "body0_path": "/World/robot/left_finger_link",
            "body1_path": "/World/Bottle500",
            "impulse_ns": 0.0,
        },
        {
            "event_type": "ContactEventType.CONTACT_PERSIST",
            "physics_step": physics_steps,
            "body0_path": "/World/robot/right_finger_link",
            "body1_path": "/World/Bottle500",
            "impulse_ns": 0.25,
        },
    ]
    export_path = run_dir / "grasp_export.yaml"
    export_path.write_bytes(export_bytes)
    export_sha = hashlib.sha256(export_bytes).hexdigest()
    report = {
        "status": "PARTIAL",
        "classification": "DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI",
        "trial_classification": "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS",
        "deterministic_trial_signature": signature,
        "trial": {
            "successful_yields": physics_steps - 1,
            "hold_command_count": physics_steps,
            "telemetry": [
                {"physics_step": step}
                for step in range(1, physics_steps + 1)
            ],
            "tester_status_messages": ["Closing", "Passed"],
            "tester_terminal_callbacks": 1,
            "contacts": contacts,
        },
        "cleanup": {
            "errors": [],
            "no_persistent_stage_write": True,
            "post_cleanup_hash_errors": [],
        },
        "root_layer_unchanged": True,
        "native_export_status": "WRITTEN_FROM_GRASP_TESTER",
        "native_export_path": str(export_path),
        "native_export_active_joints": active_joints,
        "intended_exit_code": 0,
        "shell_exit_code_is_not_authoritative": True,
        "gui_evidence": "GUI_PENDING",
        "ik": "NOT_RUN",
    }
    if gate_rerun:
        report["deterministic_run_signature"] = (
            ("c" if run_name.startswith("A_") else "d") * 64
        )
        report["native_export_validation"] = {
            "active_joints": active_joints,
            "finite": True,
            "format": "isaac_grasp",
            "format_version": 1.0,
            "grasp_count": 1,
            "sha256": export_sha,
            "size_bytes": len(export_bytes),
        }
    (run_dir / "report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (run_dir / "exit_code.txt").write_text("139\n", encoding="utf-8")


@pytest.fixture
def evidence_root(tmp_path: Path) -> Path:
    signatures = {
        "A_run1": "a" * 64,
        "A_run2": "a" * 64,
        "A_run3": "a" * 64,
        "B_run10": "b" * 64,
        "B_run11": "b" * 64,
        "B_run12": "b" * 64,
    }
    for run_name in RUN_NAMES:
        group = run_name[0]
        _write_run(
            tmp_path,
            run_name,
            signature=signatures[run_name],
            physics_steps=127 if group == "A" else 125,
            active_joints=(
                ["left_finger", "right_finger"]
                if group == "A"
                else ["left_finger"]
            ),
            export_bytes=(
                b"format: isaac_grasp\nvariant: A\n"
                if group == "A"
                else b"format: isaac_grasp\nvariant: B\n"
            ),
            gate_rerun=run_name in {"A_run3", "B_run12"},
        )
    return tmp_path


def test_build_summary_preserves_required_group_evidence(
    evidence_root: Path,
) -> None:
    module = _load_module()

    summary = module.build_summary(evidence_root)

    assert summary["fixed_inputs"] == list(RUN_NAMES)
    assert set(summary["runs"]) == set(RUN_NAMES)
    assert set(summary["groups"]) == {"A", "B"}

    group_a = summary["groups"]["A"]
    assert group_a["classification"] == {
        "script": "DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI",
        "trial": "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS",
        "status": "PARTIAL",
    }
    assert group_a["steps"] == {
        "successful_yields": 126,
        "hold_command_count": 127,
        "telemetry_samples": 127,
        "tester_status_messages": 2,
        "terminal_callbacks": 1,
        "max_physics_step": 127,
    }
    assert group_a["contacts"] == {
        "event_count": 2,
        "physics_step_count": 2,
        "min_physics_step": 1,
        "max_physics_step": 127,
        "left_finger_events": 1,
        "right_finger_events": 1,
        "positive_impulse_events": 1,
        "event_types": {
            "ContactEventType.CONTACT_FOUND": 1,
            "ContactEventType.CONTACT_PERSIST": 1,
        },
    }
    assert group_a["runs"] == ["A_run1", "A_run2", "A_run3"]
    assert group_a["trial_signature"] == {
        "value": "a" * 64,
        "identical_across_runs": True,
        "repeat_count": 3,
    }
    assert group_a["cleanup"]["all_clean"] is True
    assert group_a["export"]["status"] == "WRITTEN_FROM_GRASP_TESTER"
    assert group_a["export"]["active_joints"] == [
        "left_finger",
        "right_finger",
    ]
    export_bytes = b"format: isaac_grasp\nvariant: A\n"
    assert group_a["export"]["sha256"] == hashlib.sha256(
        export_bytes
    ).hexdigest()
    assert group_a["export"]["size_bytes"] == len(export_bytes)
    assert group_a["export"]["identical_across_runs"] is True
    assert group_a["exit_code"] == {
        "intended": 0,
        "shell": 139,
        "shell_authoritative": False,
        "assessment": "SHELL_139_NON_AUTHORITATIVE",
    }

    group_b = summary["groups"]["B"]
    assert group_b["steps"]["max_physics_step"] == 125
    assert group_b["contacts"]["max_physics_step"] == 125
    assert group_b["trial_signature"]["value"] == "b" * 64
    assert group_b["export"]["active_joints"] == ["left_finger"]


def test_summary_enforces_scope_ceiling_and_ik_hard_blocker(
    evidence_root: Path,
) -> None:
    module = _load_module()

    summary = module.build_summary(evidence_root)

    assert (
        summary["highest_conclusion"]
        == "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
    )
    assert summary["gui_evidence"] == "GUI_PENDING"
    assert summary["ik"] == {
        "source_status": "NOT_RUN",
        "status": "IK_NOT_RUN",
        "decision": "DO_NOT_START_IK",
    }
    assert summary["visual_tutor_bridge"] == {
        "available": False,
        "status": "HARD_BLOCKER",
        "reason": "VISUAL_TUTOR_BRIDGE_UNAVAILABLE",
    }
    assert summary["trial_repeat_evidence"] == {
        "A": {
            "runs": ["A_run1", "A_run2", "A_run3"],
            "repeat_count": 3,
            "status": "PASS_IDENTICAL",
        },
        "B": {
            "runs": ["B_run10", "B_run11", "B_run12"],
            "repeat_count": 3,
            "status": "PASS_IDENTICAL",
        },
    }
    assert summary["new_gate_reruns"]["A"]["run"] == "A_run3"
    assert summary["new_gate_reruns"]["A"]["status"] == "PASS"
    assert (
        summary["new_gate_reruns"]["A"]["deterministic_run_signature"]
        == "c" * 64
    )
    assert summary["new_gate_reruns"]["A"]["native_export_validation"][
        "finite"
    ] is True
    assert summary["new_gate_reruns"]["B"]["run"] == "B_run12"
    assert summary["new_gate_reruns"]["B"]["status"] == "PASS"


def test_old_runs_are_preserved_as_historical_gate_evidence(
    evidence_root: Path,
) -> None:
    module = _load_module()

    summary = module.build_summary(evidence_root)

    assert summary["runs"]["A_run1"]["gate_evidence_status"] == (
        "HISTORICAL_PRE_GATE_FIELDS"
    )
    assert summary["runs"]["A_run1"]["deterministic_run_signature"] is None
    assert summary["runs"]["A_run1"]["native_export_validation"] is None
    assert summary["runs"]["B_run11"]["gate_evidence_status"] == (
        "HISTORICAL_PRE_GATE_FIELDS"
    )
    assert summary["runs"]["A_run3"]["gate_evidence_status"] == (
        "NEW_GATE_RERUN_PASS"
    )
    assert summary["runs"]["B_run12"]["gate_evidence_status"] == (
        "NEW_GATE_RERUN_PASS"
    )


def test_markdown_makes_non_authoritative_exit_and_gate_explicit(
    evidence_root: Path,
) -> None:
    module = _load_module()
    markdown = module.render_markdown(module.build_summary(evidence_root))

    assert "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS" in markdown
    assert "shell `139` is non-authoritative" in markdown
    assert "`GUI_PENDING`" in markdown
    assert "`IK_NOT_RUN`" in markdown
    assert "`HARD_BLOCKER`" in markdown
    assert "Visual Tutor bridge unavailable" in markdown
    assert "Do not start IK" in markdown
    assert "A_run1" in markdown
    assert "A_run3" in markdown
    assert "B_run11" in markdown
    assert "B_run12" in markdown
    assert "three trial repeats are identical" in markdown
    assert "new gate rerun passed once" in markdown
    assert "c" * 64 in markdown
    assert hashlib.sha256(
        b"format: isaac_grasp\nvariant: A\n"
    ).hexdigest() in markdown
    assert "native export validation: PASS" in markdown
    assert "HISTORICAL_PRE_GATE_FIELDS" in markdown


def test_missing_fixed_input_fails_closed(evidence_root: Path) -> None:
    module = _load_module()
    (evidence_root / "B_run11" / "report.json").unlink()

    with pytest.raises(FileNotFoundError, match="B_run11"):
        module.build_summary(evidence_root)


def test_group_signature_disagreement_fails_closed(
    evidence_root: Path,
) -> None:
    module = _load_module()
    report_path = evidence_root / "A_run2" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["deterministic_trial_signature"] = "c" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="A.*trial signature"):
        module.build_summary(evidence_root)


def test_group_export_bytes_disagreement_fails_closed(
    evidence_root: Path,
) -> None:
    module = _load_module()
    (evidence_root / "B_run10" / "grasp_export.yaml").write_bytes(
        b"different export\n"
    )

    with pytest.raises(ValueError, match="B.*export.*SHA/size"):
        module.build_summary(evidence_root)

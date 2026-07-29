#!/usr/bin/env python3
"""Summarize the six fixed scripted Grasp Tester evidence runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / ".codex"
    / "artifacts"
    / "20260730-aloha1-grasp-editor-ik-evidence"
    / "grasp_tester_scripted"
)
DEFAULT_JSON_OUTPUT = (
    REPO_ROOT
    / "reports"
    / "aloha1_mapping"
    / "aloha1_grasp_tester_scripted_equivalent.json"
)
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_JSON_OUTPUT.with_suffix(".md")

FIXED_RUNS = (
    "A_run1",
    "A_run2",
    "A_run3",
    "B_run10",
    "B_run11",
    "B_run12",
)
GROUP_RUNS = {
    "A": ("A_run1", "A_run2", "A_run3"),
    "B": ("B_run10", "B_run11", "B_run12"),
}
NEW_GATE_RUNS = {"A": "A_run3", "B": "B_run12"}
EXPECTED_SCRIPT_CLASSIFICATION = "DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI"
HIGHEST_CONCLUSION = "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_shell_exit_code(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed input: {path}")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"invalid shell exit code: {path}") from exc


def _require_dict(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(
    value: object,
    *,
    label: str,
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _steps(trial: dict[str, Any]) -> dict[str, int | None]:
    telemetry = _require_list(
        trial.get("telemetry"),
        label="trial.telemetry",
    )
    statuses = _require_list(
        trial.get("tester_status_messages"),
        label="trial.tester_status_messages",
    )
    physics_steps = [
        row.get("physics_step")
        for row in telemetry
        if isinstance(row, dict) and isinstance(row.get("physics_step"), int)
    ]
    return {
        "successful_yields": int(trial["successful_yields"]),
        "hold_command_count": int(trial["hold_command_count"]),
        "telemetry_samples": len(telemetry),
        "tester_status_messages": len(statuses),
        "terminal_callbacks": int(trial["tester_terminal_callbacks"]),
        "max_physics_step": max(physics_steps, default=None),
    }


def _contacts(trial: dict[str, Any]) -> dict[str, Any]:
    contacts = _require_list(
        trial.get("contacts"),
        label="trial.contacts",
    )
    physics_steps: list[int] = []
    event_types: Counter[str] = Counter()
    left_finger_events = 0
    right_finger_events = 0
    positive_impulse_events = 0

    for index, event in enumerate(contacts):
        if not isinstance(event, dict):
            raise ValueError(f"trial.contacts[{index}] must be an object")
        physics_step = event.get("physics_step")
        if isinstance(physics_step, int):
            physics_steps.append(physics_step)
        event_type = event.get("event_type")
        if isinstance(event_type, str):
            event_types[event_type] += 1
        bodies = " ".join(
            str(event.get(key, ""))
            for key in ("body0_path", "body1_path")
        ).lower()
        if "left_finger" in bodies:
            left_finger_events += 1
        if "right_finger" in bodies:
            right_finger_events += 1
        impulse = event.get("impulse_ns")
        if isinstance(impulse, int | float) and impulse > 0:
            positive_impulse_events += 1

    unique_steps = set(physics_steps)
    return {
        "event_count": len(contacts),
        "physics_step_count": len(unique_steps),
        "min_physics_step": min(physics_steps, default=None),
        "max_physics_step": max(physics_steps, default=None),
        "left_finger_events": left_finger_events,
        "right_finger_events": right_finger_events,
        "positive_impulse_events": positive_impulse_events,
        "event_types": dict(sorted(event_types.items())),
    }


def _cleanup(report: dict[str, Any]) -> dict[str, Any]:
    cleanup = _require_dict(
        report.get("cleanup"),
        label="cleanup",
    )
    errors = _require_list(cleanup.get("errors"), label="cleanup.errors")
    hash_errors = _require_list(
        cleanup.get("post_cleanup_hash_errors"),
        label="cleanup.post_cleanup_hash_errors",
    )
    no_stage_write = cleanup.get("no_persistent_stage_write") is True
    root_unchanged = report.get("root_layer_unchanged") is True
    return {
        "errors": errors,
        "post_cleanup_hash_errors": hash_errors,
        "no_persistent_stage_write": no_stage_write,
        "root_layer_unchanged": root_unchanged,
        "clean": (
            not errors
            and not hash_errors
            and no_stage_write
            and root_unchanged
        ),
    }


def _file_integrity(path: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed input: {path}")
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _export(
    report: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    active_joints = _require_list(
        report.get("native_export_active_joints"),
        label="native_export_active_joints",
    )
    export_path = run_dir / "grasp_export.yaml"
    integrity = _file_integrity(export_path)
    return {
        "status": report.get("native_export_status"),
        "path": str(export_path),
        "reported_path": report.get("native_export_path"),
        "active_joints": active_joints,
        **integrity,
    }


def _gate_evidence(
    report: dict[str, Any],
    export: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    run_signature = report.get("deterministic_run_signature")
    validation = report.get("native_export_validation")
    if run_signature is None and validation is None:
        return {
            "gate_evidence_status": "HISTORICAL_PRE_GATE_FIELDS",
            "deterministic_run_signature": None,
            "native_export_validation": None,
        }
    if not isinstance(run_signature, str) or len(run_signature) != 64:
        raise ValueError(f"{run_name} invalid deterministic run signature")
    validation = _require_dict(
        validation,
        label=f"{run_name}.native_export_validation",
    )
    matches_export = (
        validation.get("sha256") == export["sha256"]
        and validation.get("size_bytes") == export["size_bytes"]
        and validation.get("active_joints") == export["active_joints"]
    )
    if validation.get("finite") is not True or not matches_export:
        raise ValueError(
            f"{run_name} native export validation does not match export"
        )
    return {
        "gate_evidence_status": "NEW_GATE_RERUN_PASS",
        "deterministic_run_signature": run_signature,
        "native_export_validation": validation,
    }


def _exit_code(report: dict[str, Any], shell_code: int) -> dict[str, Any]:
    shell_authoritative = (
        report.get("shell_exit_code_is_not_authoritative") is not True
    )
    assessment = (
        "SHELL_139_NON_AUTHORITATIVE"
        if shell_code == 139 and not shell_authoritative
        else "SHELL_EXIT_REQUIRES_REVIEW"
    )
    return {
        "intended": report.get("intended_exit_code"),
        "shell": shell_code,
        "shell_authoritative": shell_authoritative,
        "assessment": assessment,
    }


def _summarize_run(
    input_root: Path,
    run_name: str,
) -> dict[str, Any]:
    run_dir = input_root / run_name
    report = _read_json(run_dir / "report.json")
    trial = _require_dict(report.get("trial"), label=f"{run_name}.trial")
    shell_code = _read_shell_exit_code(run_dir / "exit_code.txt")
    signature = report.get("deterministic_trial_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(f"{run_name} has invalid trial signature")

    export = _export(report, run_dir)
    return {
        "source_report": str(run_dir / "report.json"),
        "classification": {
            "script": report.get("classification"),
            "trial": report.get("trial_classification"),
            "status": report.get("status"),
        },
        "steps": _steps(trial),
        "contacts": _contacts(trial),
        "trial_signature": signature,
        "cleanup": _cleanup(report),
        "export": export,
        **_gate_evidence(report, export, run_name),
        "exit_code": _exit_code(report, shell_code),
        "gui_evidence": report.get("gui_evidence"),
        "ik": report.get("ik"),
    }


def _common_group_value(
    group: str,
    runs: list[dict[str, Any]],
    key: str,
    *,
    label: str | None = None,
) -> Any:
    values = [run[key] for run in runs]
    if any(value != values[0] for value in values[1:]):
        description = label or key
        raise ValueError(f"{group} group {description} disagreement")
    return values[0]


def _summarize_group(
    group: str,
    run_names: tuple[str, ...],
    runs_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    runs = [runs_by_name[name] for name in run_names]
    classification = _common_group_value(
        group,
        runs,
        "classification",
    )
    steps = _common_group_value(group, runs, "steps")
    contacts = _common_group_value(group, runs, "contacts")
    signature = _common_group_value(
        group,
        runs,
        "trial_signature",
        label="trial signature",
    )

    cleanups = [run["cleanup"] for run in runs]
    export_statuses = [run["export"]["status"] for run in runs]
    active_joints = [run["export"]["active_joints"] for run in runs]
    export_integrity = [
        (run["export"]["sha256"], run["export"]["size_bytes"])
        for run in runs
    ]
    exit_codes = [run["exit_code"] for run in runs]

    if any(value != export_statuses[0] for value in export_statuses[1:]):
        raise ValueError(f"{group} group export status disagreement")
    if any(value != active_joints[0] for value in active_joints[1:]):
        raise ValueError(f"{group} group export active joints disagreement")
    if any(
        value != export_integrity[0] for value in export_integrity[1:]
    ):
        raise ValueError(f"{group} group export SHA/size disagreement")
    if any(value != exit_codes[0] for value in exit_codes[1:]):
        raise ValueError(f"{group} group exit code disagreement")

    return {
        "runs": list(run_names),
        "classification": classification,
        "steps": steps,
        "contacts": contacts,
        "trial_signature": {
            "value": signature,
            "identical_across_runs": True,
            "repeat_count": len(run_names),
        },
        "cleanup": {
            "all_clean": all(item["clean"] for item in cleanups),
            "per_run": {
                name: runs_by_name[name]["cleanup"] for name in run_names
            },
        },
        "export": {
            "status": export_statuses[0],
            "active_joints": active_joints[0],
            "sha256": export_integrity[0][0],
            "size_bytes": export_integrity[0][1],
            "identical_across_runs": True,
            "paths": {
                name: runs_by_name[name]["export"]["path"]
                for name in run_names
            },
        },
        "exit_code": exit_codes[0],
    }


def _validate_scope(runs: dict[str, dict[str, Any]]) -> None:
    for run_name, run in runs.items():
        classification = run["classification"]
        if classification["script"] != EXPECTED_SCRIPT_CLASSIFICATION:
            raise ValueError(
                f"{run_name} unexpected script classification: "
                f"{classification['script']}"
            )
        if classification["trial"] != HIGHEST_CONCLUSION:
            raise ValueError(
                f"{run_name} exceeds or misses conclusion ceiling: "
                f"{classification['trial']}"
            )
        if run["gui_evidence"] != "GUI_PENDING":
            raise ValueError(f"{run_name} GUI evidence is not pending")
        if run["ik"] != "NOT_RUN":
            raise ValueError(f"{run_name} IK must remain NOT_RUN")
        if run["exit_code"]["assessment"] != (
            "SHELL_139_NON_AUTHORITATIVE"
        ):
            raise ValueError(
                f"{run_name} shell exit does not match fixed evidence"
            )


def build_summary(input_root: Path) -> dict[str, Any]:
    """Build the fail-closed summary from the six fixed run directories."""
    input_root = input_root.resolve()
    runs = {
        run_name: _summarize_run(input_root, run_name)
        for run_name in FIXED_RUNS
    }
    _validate_scope(runs)
    groups = {
        group: _summarize_group(group, run_names, runs)
        for group, run_names in GROUP_RUNS.items()
    }
    if not all(group["cleanup"]["all_clean"] for group in groups.values()):
        raise ValueError("one or more fixed runs did not clean up")

    new_gate_reruns: dict[str, dict[str, Any]] = {}
    for group, run_name in NEW_GATE_RUNS.items():
        run = runs[run_name]
        if run["gate_evidence_status"] != "NEW_GATE_RERUN_PASS":
            raise ValueError(f"{group} new gate rerun did not pass")
        new_gate_reruns[group] = {
            "run": run_name,
            "status": "PASS",
            "deterministic_run_signature": (
                run["deterministic_run_signature"]
            ),
            "native_export_validation": run[
                "native_export_validation"
            ],
        }

    return {
        "schema_version": 1,
        "title": "ALOHA1 scripted Grasp Tester equivalent summary",
        "input_root": str(input_root),
        "fixed_inputs": list(FIXED_RUNS),
        "classification": EXPECTED_SCRIPT_CLASSIFICATION,
        "highest_conclusion": HIGHEST_CONCLUSION,
        "gui_evidence": "GUI_PENDING",
        "ik": {
            "source_status": "NOT_RUN",
            "status": "IK_NOT_RUN",
            "decision": "DO_NOT_START_IK",
        },
        "visual_tutor_bridge": {
            "available": False,
            "status": "HARD_BLOCKER",
            "reason": "VISUAL_TUTOR_BRIDGE_UNAVAILABLE",
        },
        "shell_exit_code_authority": "NON_AUTHORITATIVE",
        "trial_repeat_evidence": {
            group: {
                "runs": list(run_names),
                "repeat_count": len(run_names),
                "status": "PASS_IDENTICAL",
            }
            for group, run_names in GROUP_RUNS.items()
        },
        "new_gate_reruns": new_gate_reruns,
        "runs": runs,
        "groups": groups,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact human-readable statement of the evidence ceiling."""
    lines = [
        "# ALOHA1 scripted Grasp Tester equivalent summary",
        "",
        "## Decision boundary",
        "",
        f"- Highest conclusion: `{summary['highest_conclusion']}`.",
        "- This is scripted-equivalent evidence, not GUI task-pass evidence.",
        f"- GUI evidence remains `{summary['gui_evidence']}`.",
        f"- IK remains `{summary['ik']['status']}`. **Do not start IK.**",
        (
            "- Visual Tutor bridge unavailable: "
            f"`{summary['visual_tutor_bridge']['status']}` "
            "(`VISUAL_TUTOR_BRIDGE_UNAVAILABLE`)."
        ),
        "- For all six fixed runs, shell `139` is non-authoritative.",
        (
            "- In each A/B group, three trial repeats are identical by "
            "deterministic trial signature and compact evidence."
        ),
        (
            "- For each group, one new gate rerun passed once: "
            "`A_run3` and `B_run12`."
        ),
        "",
        "## Group summary",
        "",
        (
            "| Group | Runs | Trial classification | Steps | Contacts | "
            "Cleanup | Export | Shell exit |"
        ),
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for group_name in ("A", "B"):
        group = summary["groups"][group_name]
        lines.append(
            "| {group} | {runs} | `{trial}` | {steps} | {contacts} | "
            "{cleanup} | `{export}` | "
            "`{shell}` (non-authoritative) |".format(
                group=group_name,
                runs=", ".join(group["runs"]),
                trial=group["classification"]["trial"],
                steps=group["steps"]["max_physics_step"],
                contacts=group["contacts"]["event_count"],
                cleanup="clean" if group["cleanup"]["all_clean"] else "FAIL",
                export=group["export"]["status"],
                shell=group["exit_code"]["shell"],
            )
        )

    lines.extend(
        [
            "",
            "## Deterministic trial signatures",
            "",
        ]
    )
    for group_name in ("A", "B"):
        signature = summary["groups"][group_name]["trial_signature"]
        lines.append(
            f"- Group {group_name}: `{signature['value']}` "
            "(identical across all three runs)."
        )

    lines.extend(
        [
            "",
            "## Export byte identity",
            "",
        ]
    )
    for group_name in ("A", "B"):
        export = summary["groups"][group_name]["export"]
        lines.append(
            f"- Group {group_name}: `{export['sha256']}`, "
            f"{export['size_bytes']} bytes; identical across all three "
            "exports."
        )

    lines.extend(
        [
            "",
            "## New gate reruns",
            "",
        ]
    )
    for group_name in ("A", "B"):
        rerun = summary["new_gate_reruns"][group_name]
        validation = rerun["native_export_validation"]
        lines.append(
            f"- {rerun['run']}: `{rerun['status']}`; "
            "deterministic run signature "
            f"`{rerun['deterministic_run_signature']}`; "
            "native export validation: PASS "
            f"(finite={str(validation['finite']).lower()}, "
            f"SHA-256 `{validation['sha256']}`, "
            f"{validation['size_bytes']} bytes)."
        )

    historical_runs = [
        run_name
        for run_name in FIXED_RUNS
        if summary["runs"][run_name]["gate_evidence_status"]
        == "HISTORICAL_PRE_GATE_FIELDS"
    ]
    lines.extend(
        [
            "",
            "## Historical gate-field compatibility",
            "",
            (
                f"- {', '.join(historical_runs)} are preserved as "
                "`HISTORICAL_PRE_GATE_FIELDS`; their missing new fields "
                "are expected historical evidence, not failures."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Fixed run evidence",
            "",
            (
                "| Run | Script classification | Steps | Contacts | "
                "Intended exit | Shell exit |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for run_name in FIXED_RUNS:
        run = summary["runs"][run_name]
        lines.append(
            "| {run_name} | `{classification}` | {steps} | {contacts} | "
            "{intended} | {shell} |".format(
                run_name=run_name,
                classification=run["classification"]["script"],
                steps=run["steps"]["max_physics_step"],
                contacts=run["contacts"]["event_count"],
                intended=run["exit_code"]["intended"],
                shell=run["exit_code"]["shell"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_outputs(
    summary: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(
        render_markdown(summary),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Directory containing A_run1/A_run2/A_run3 and "
            "B_run10/B_run11/B_run12."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build_summary(args.input_root)
    _write_outputs(summary, args.json_output, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

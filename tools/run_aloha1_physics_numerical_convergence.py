#!/usr/bin/env python3
"""Run the predeclared ALOHA1 Isaac 5.1 numerical-convergence matrix."""

from __future__ import annotations

import argparse
import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from tools.aloha1_mapping.physics_numerical_convergence import build_runtime_cell_command
from tools.aloha1_mapping.physics_numerical_convergence import compare_runtime_cells
from tools.aloha1_mapping.physics_numerical_convergence import extract_runtime_cell_metrics
from tools.aloha1_mapping.physics_numerical_convergence import select_coarsest_converged_value
from tools.aloha1_mapping.physics_numerical_convergence import should_continue_solver_sweeps

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv_issac/bin/python"
GRASP_LAUNCHER = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"
FREE_MOTION_LAUNCHER = ROOT / "tools/diagnose_aloha1_bottle_com_velocity.py"
DEFAULT_CONFIG = ROOT / "configs/aloha1_physics_numerical_convergence.yaml"
DEFAULT_ARTIFACT_ROOT = (
    ROOT / ".codex/artifacts/20260803-aloha1-model-first/convergence/matrix"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/aloha1_mapping/aloha1_physics_numerical_convergence.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _stable_report_signature(payload: dict[str, Any]) -> str:
    """Hash numerical evidence while excluding volatile process metadata."""

    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: stable(item)
                for key, item in value.items()
                if key
                not in {
                    "process",
                    "deterministic_signature",
                    "extra_solver_cells_from_superseded_driver",
                }
            }
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return hashlib.sha256(
        json.dumps(
            stable(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _resolve_frozen(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    actual = _sha256(path)
    if actual != str(record["sha256"]):
        raise RuntimeError(f"frozen input hash mismatch: {path}")
    return path


def _superseded_solver_cell_paths(artifact_root: Path) -> list[str]:
    runtime_root = artifact_root / "runtime_cells"
    if not runtime_root.is_dir():
        return []
    prefixes = ("position_", "velocity_", "final_repeat_")
    return [
        str(path.resolve())
        for path in sorted(runtime_root.iterdir())
        if path.is_dir() and path.name.startswith(prefixes)
    ]


def _run_logged(
    *,
    command: list[str],
    log_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = str(ROOT)
    started = time.perf_counter()
    with log_path.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        timed_out = False
        try:
            exit_code = int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                exit_code = int(process.wait(timeout=20.0))
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = int(process.wait(timeout=20.0))
    return {
        "command": command,
        "process_id": int(process.pid),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "runtime_seconds": time.perf_counter() - started,
        "log_absolute_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
        "environment": {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": str(ROOT),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_selected_sample(preflight_path: Path, sample_id: str) -> dict[str, Any]:
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in payload.get("selected_samples", [])
        if str(record.get("sample_id")) == sample_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one frozen sample: {sample_id}")
    sample = matches[0]
    if sample.get("preflight_status") != "PASS":
        raise RuntimeError(f"frozen sample is not PASS: {sample_id}")
    return sample


def _free_motion_cell(
    *,
    frequency_hz: int,
    artifact_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    cell_root = artifact_root / "free_motion" / f"v1_f{frequency_hz:04d}"
    report_path = cell_root / "report.json"
    samples_path = cell_root / "samples.jsonl"
    log_path = cell_root / "process.log"
    command = [
        str(ISAAC_PYTHON),
        str(FREE_MOTION_LAUNCHER),
        "--variant",
        "V1",
        "--output",
        str(report_path),
        "--samples-output",
        str(samples_path),
        "--steps",
        str(2 * frequency_hz + 1),
        "--physics-frequency-hz",
        str(frequency_hz),
    ]
    if report_path.is_file() and samples_path.is_file():
        process = {
            "resume_status": "REUSED_COMPLETE_CELL",
            "command": command,
            "log_absolute_path": str(log_path.resolve()),
            "log_sha256": _sha256(log_path) if log_path.is_file() else None,
        }
    else:
        cell_root.mkdir(parents=True, exist_ok=False)
        process = _run_logged(
            command=command,
            log_path=log_path,
            timeout_s=timeout_s,
        )
    if not report_path.is_file() or not samples_path.is_file():
        raise RuntimeError(f"free-motion output missing: {cell_root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"free-motion cell failed: {frequency_hz}")
    metrics = report["metrics"]
    return {
        "frequency_hz": frequency_hz,
        "status": report["status"],
        "sample_count": int(metrics["sample_count"]),
        "dt_s": metrics["dt_s"],
        "analytic_position_error_norm_m": float(
            report["analytic_command_check"]["error_norm_m"]
        ),
        "com_velocity_max_error_m_s": float(
            metrics["com_backward_fd_vs_velocity"]["max_error_m_s"]
        ),
        "signed_velocity_integral_vector_m": metrics[
            "signed_velocity_integral_vector_m"
        ],
        "report_absolute_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "samples_absolute_path": str(samples_path.resolve()),
        "samples_sha256": _sha256(samples_path),
        "process": process,
    }


def _runtime_cell(
    *,
    family: str,
    frequency_hz: int,
    position_iterations: int,
    velocity_iterations: int,
    runtime_config: Path,
    pose_path: Path,
    initial_q: list[float],
    stage_path: Path,
    stage_hash: str,
    artifact_root: Path,
    timeout_s: float,
) -> dict[str, Any]:
    name = (
        f"{family}_f{frequency_hz:04d}_p{position_iterations:03d}_"
        f"v{velocity_iterations:03d}"
    )
    cell_root = artifact_root / "runtime_cells" / name
    report_path = cell_root / "aloha1_grasp_20cm_runtime.json"
    telemetry_path = cell_root / "aloha1_grasp_20cm_telemetry.jsonl"
    requested = {
        "frequency_hz": frequency_hz,
        "position_iterations": position_iterations,
        "velocity_iterations": velocity_iterations,
    }
    command = build_runtime_cell_command(
        isaac_python=ISAAC_PYTHON,
        launcher=GRASP_LAUNCHER,
        runtime_config=runtime_config,
        artifact_root=cell_root,
        bottle_transform_path=pose_path,
        initial_arm_q_rad=initial_q,
        frequency_hz=frequency_hz,
        position_iterations=position_iterations,
        velocity_iterations=velocity_iterations,
        capture_failure_evidence=False,
    )
    log_path = artifact_root / "logs" / f"{name}.log"
    if report_path.is_file() and telemetry_path.is_file():
        process = {
            "resume_status": "REUSED_COMPLETE_CELL",
            "command": command,
            "log_absolute_path": str(log_path.resolve()),
            "log_sha256": _sha256(log_path) if log_path.is_file() else None,
        }
    else:
        cell_root.mkdir(parents=True, exist_ok=False)
        before = _sha256(stage_path)
        process = _run_logged(
            command=command,
            log_path=log_path,
            timeout_s=timeout_s,
        )
        after = _sha256(stage_path)
        process["stage_sha256_before"] = before
        process["stage_sha256_after"] = after
    if not report_path.is_file() or not telemetry_path.is_file():
        raise RuntimeError(f"runtime cell output missing: {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    telemetry = _read_jsonl(telemetry_path)
    readback = report.get("runtime", {}).get(
        "numerical_convergence", {}
    ).get("readback", {})
    effective = {
        "frequency_hz": round(1.0 / float(readback["effective_physics_dt_s"])),
        "position_iterations": int(readback["effective_position_iterations"]),
        "velocity_iterations": int(readback["effective_velocity_iterations"]),
    }
    if effective != requested:
        raise RuntimeError(
            f"numerical readback mismatch for {name}: {effective} != {requested}"
        )
    if report.get("stage", {}).get("sha256_before") != stage_hash or (
        report.get("stage", {}).get("sha256_after") != stage_hash
    ):
        raise RuntimeError(f"frozen Stage hash drift in {name}")
    metrics = extract_runtime_cell_metrics(report=report, telemetry=telemetry)
    failure_evidence: dict[str, Any] | None = None
    if report.get("status") != "PASS":
        evidence_root = artifact_root / "failure_evidence" / name
        evidence_report = evidence_root / "aloha1_grasp_20cm_runtime.json"
        evidence_candidate = (
            evidence_root
            / "video_attempt_001/video/candidate_manifest.json"
        )
        if evidence_report.is_file() and evidence_candidate.is_file():
            evidence_process: dict[str, Any] = {
                "resume_status": "REUSED_COMPLETE_FAILURE_EVIDENCE"
            }
        else:
            evidence_root.mkdir(parents=True, exist_ok=False)
            evidence_command = build_runtime_cell_command(
                isaac_python=ISAAC_PYTHON,
                launcher=GRASP_LAUNCHER,
                runtime_config=runtime_config,
                artifact_root=evidence_root,
                bottle_transform_path=pose_path,
                initial_arm_q_rad=initial_q,
                frequency_hz=frequency_hz,
                position_iterations=position_iterations,
                velocity_iterations=velocity_iterations,
                capture_failure_evidence=True,
            )
            evidence_process = _run_logged(
                command=evidence_command,
                log_path=artifact_root / "logs" / f"{name}_failure_evidence.log",
                timeout_s=timeout_s,
            )
        failure_evidence = {
            "status": (
                "CAPTURED"
                if evidence_report.is_file() and evidence_candidate.is_file()
                else "FAILED"
            ),
            "runtime_report_absolute_path": str(evidence_report.resolve()),
            "runtime_report_sha256": (
                _sha256(evidence_report) if evidence_report.is_file() else None
            ),
            "candidate_manifest_absolute_path": str(
                evidence_candidate.resolve()
            ),
            "candidate_manifest_sha256": (
                _sha256(evidence_candidate)
                if evidence_candidate.is_file()
                else None
            ),
            "visual_model_review": "PENDING_AFTER_MATRIX",
            "process": evidence_process,
        }
    return {
        "name": name,
        "family": family,
        "requested": requested,
        "effective": effective,
        "machine_status": report.get("status"),
        "machine_reason": report.get("reason"),
        "deterministic_signature": report.get("deterministic_signature"),
        "metrics": metrics,
        "report_absolute_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "telemetry_absolute_path": str(telemetry_path.resolve()),
        "telemetry_sha256": _sha256(telemetry_path),
        "process": process,
        "failure_evidence": failure_evidence,
        "_telemetry": telemetry,
    }


def _pair_comparison(
    coarse: dict[str, Any],
    fine: dict[str, Any],
    *,
    bottle_position_bound_m: float,
    joint_position_bound_rad: float,
) -> dict[str, Any]:
    trace = compare_runtime_cells(
        coarse=coarse["_telemetry"],
        fine=fine["_telemetry"],
    )
    coarse_metrics = coarse["metrics"]
    fine_metrics = fine["metrics"]
    coarse_contact = coarse_metrics["first_bilateral_solver_contact_time_s"]
    fine_contact = fine_metrics["first_bilateral_solver_contact_time_s"]
    onset_difference = (
        None
        if coarse_contact is None or fine_contact is None
        else abs(float(coarse_contact) - float(fine_contact))
    )
    dt_bound = max(
        float(coarse_metrics["numerical_readback"]["effective_physics_dt_s"]),
        float(fine_metrics["numerical_readback"]["effective_physics_dt_s"]),
    )
    coarse_impulse = sum(
        abs(float(coarse_metrics["contact"][name]))
        for name in (
            "left_signed_normal_impulse_ns",
            "right_signed_normal_impulse_ns",
        )
    )
    fine_impulse = sum(
        abs(float(fine_metrics["contact"][name]))
        for name in (
            "left_signed_normal_impulse_ns",
            "right_signed_normal_impulse_ns",
        )
    )
    impulse_scale = max(abs(coarse_impulse), abs(fine_impulse), 1e-15)
    return {
        "coarse": coarse["requested"],
        "fine": fine["requested"],
        "trace": trace,
        "contact_onset_difference_s": onset_difference,
        "contact_onset_bound_s": dt_bound,
        "signed_normal_impulse_absolute_sum_coarse_ns": coarse_impulse,
        "signed_normal_impulse_absolute_sum_fine_ns": fine_impulse,
        "signed_normal_impulse_relative_difference": abs(
            coarse_impulse - fine_impulse
        ) / impulse_scale,
        "signed_drive_work_difference_j": abs(
            float(coarse_metrics["drive"]["signed_work_j"])
            - float(fine_metrics["drive"]["signed_work_j"])
        ),
        "maximum_penetration_difference_m": abs(
            float(coarse_metrics["contact"]["maximum_penetration_m"] or 0.0)
            - float(fine_metrics["contact"]["maximum_penetration_m"] or 0.0)
        ),
        "gates": {
            "physical_model_signature_equal": (
                coarse_metrics["physical_model_signature"]
                == fine_metrics["physical_model_signature"]
            ),
            "joint_position_within_official_tick": (
                trace["joint_position_max_abs_difference"]
                <= joint_position_bound_rad
            ),
            "bottle_position_within_geometry_numeric_budget": (
                trace["bottle_position_max_norm_difference_m"]
                <= bottle_position_bound_m
            ),
            "contact_onset_within_one_coarse_step": (
                onset_difference is not None and onset_difference <= dt_bound
            ),
            "terminal_classification_equal": (
                coarse["machine_status"] == fine["machine_status"]
                and coarse["machine_reason"] == fine["machine_reason"]
            ),
        },
    }


def _selection_records(
    cells: list[dict[str, Any]],
    *,
    axis: str,
    bottle_position_bound_m: float,
    joint_position_bound_rad: float,
) -> tuple[list[dict[str, Any]], int | None]:
    ordered = sorted(cells, key=lambda item: int(item["requested"][axis]))
    comparisons = [
        _pair_comparison(
            coarse,
            fine,
            bottle_position_bound_m=bottle_position_bound_m,
            joint_position_bound_rad=joint_position_bound_rad,
        )
        for coarse, fine in pairwise(ordered)
    ]
    flattened = [
        {
            "coarse": int(record["coarse"][axis]),
            "fine": int(record["fine"][axis]),
            "gates": record["gates"],
        }
        for record in comparisons
    ]
    selected = select_coarsest_converged_value(
        ordered_values=[int(item["requested"][axis]) for item in ordered],
        comparisons=flattened,
    )
    return comparisons, selected


def _public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cell.items() if key != "_telemetry"}


def _markdown(report: dict[str, Any]) -> str:
    selected = report["selection"]
    lines = [
        "# ALOHA1 Isaac 5.1 numerical convergence",
        "",
        f"- Status: `{report['status']}`",
        f"- Frequency: `{selected['frequency_hz']}`",
        f"- Position iterations: `{selected['position_iterations']}`",
        f"- Velocity iterations: `{selected['velocity_iterations']}`",
        f"- Final repeat deterministic: `{selected['fresh_repeat_signature_equal']}`",
        "- Physical model parameters were frozen; only dt/solver iterations changed.",
        "- Grasp PASS/FAIL was not used to create a tolerance.",
        "",
        "## Free-motion baseline",
        "",
        "| Hz | position error (m) | COM velocity error (m/s) |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {cell['frequency_hz']} | {cell['analytic_position_error_norm_m']:.9g} "
            f"| {cell['com_velocity_max_error_m_s']:.9g} |"
        )
        for cell in report["free_motion"]
    )
    lines.extend(
        [
            "",
            "## Frequency-pair convergence",
            "",
            "| pair (Hz) | joint position max (rad) | bottle position max (m) "
            "| contact onset delta (s) | all gates |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    for comparison in report["frequency_sweep"]["comparisons"]:
        gates = comparison["gates"]
        lines.append(
            f"| {comparison['coarse']['frequency_hz']}→"
            f"{comparison['fine']['frequency_hz']} "
            f"| {comparison['trace']['joint_position_max_abs_difference']:.9g} "
            f"| {comparison['trace']['bottle_position_max_norm_difference_m']:.9g} "
            f"| {comparison['contact_onset_difference_s']!s} "
            f"| {all(bool(value) for value in gates.values())} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This validates numerical sensitivity of the current isolated diagnostic model. "
            "It does not promote the rejected contact-band collider, calibrate friction, "
            "or establish continuous-duty actuator limits.",
            "A machine `PARTIAL` caused only by cross-step numerical disagreement is not "
            "a visible grasp failure. When every cell still physically passes, a failure "
            "video is `NOT_REQUIRED`; the signed telemetry and pairwise metrics are the "
            "authoritative evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("classification") != (
        "DIAGNOSTIC_NUMERICAL_ONLY_NOT_PHYSICAL_CALIBRATION"
    ):
        raise RuntimeError("unexpected convergence config classification")
    frozen = config["frozen_inputs"]
    runtime_config = _resolve_frozen(frozen["runtime_config"])
    stage_path = _resolve_frozen(frozen["approved_stage"])
    preflight_path = _resolve_frozen(frozen["five_pose_preflight"])
    _resolve_frozen(frozen["bottle_source"])
    stage_hash = str(frozen["approved_stage"]["sha256"])
    sample_id = str(config["runtime"]["sample_id"])
    sample = _load_selected_sample(preflight_path, sample_id)
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    pose_path = artifact_root / "frozen_world_from_object.json"
    _atomic_json(
        pose_path,
        {
            "schema_version": 1,
            "sample_id": sample_id,
            "world_from_object": sample["world_from_object"],
        },
    )
    initial_q = [float(value) for value in sample["initial_arm_q_rad"]]
    matrix = config["matrix"]
    frequencies = [int(value) for value in matrix["frequency_hz"]]
    free_motion = [
        _free_motion_cell(
            frequency_hz=frequency,
            artifact_root=artifact_root,
            timeout_s=float(args.timeout_s),
        )
        for frequency in frequencies
    ]
    bottle_bound = float(
        config["independent_bounds"]["bottle_position_difference_m"]["value"]
    )
    joint_bound = float(
        config["independent_bounds"]["joint_position_difference_rad"]["value"]
    )
    frequency_cells = [
        _runtime_cell(
            family="frequency",
            frequency_hz=frequency,
            position_iterations=int(
                matrix["frequency_sweep_position_iterations"]
            ),
            velocity_iterations=int(
                matrix["frequency_sweep_velocity_iterations"]
            ),
            runtime_config=runtime_config,
            pose_path=pose_path,
            initial_q=initial_q,
            stage_path=stage_path,
            stage_hash=stage_hash,
            artifact_root=artifact_root,
            timeout_s=float(args.timeout_s),
        )
        for frequency in frequencies
    ]
    frequency_comparisons, selected_frequency = _selection_records(
        frequency_cells,
        axis="frequency_hz",
        bottle_position_bound_m=bottle_bound,
        joint_position_bound_rad=joint_bound,
    )
    optional_used = False
    if selected_frequency is None:
        optional_used = True
        optional_frequency = int(matrix["optional_frequency_hz"])
        free_motion.append(
            _free_motion_cell(
                frequency_hz=optional_frequency,
                artifact_root=artifact_root,
                timeout_s=float(args.timeout_s),
            )
        )
        frequency_cells.append(
            _runtime_cell(
                family="frequency",
                frequency_hz=optional_frequency,
                position_iterations=int(
                    matrix["frequency_sweep_position_iterations"]
                ),
                velocity_iterations=int(
                    matrix["frequency_sweep_velocity_iterations"]
                ),
                runtime_config=runtime_config,
                pose_path=pose_path,
                initial_q=initial_q,
                stage_path=stage_path,
                stage_hash=stage_hash,
                artifact_root=artifact_root,
                timeout_s=float(args.timeout_s),
            )
        )
        frequencies.append(optional_frequency)
        frequency_comparisons, selected_frequency = _selection_records(
            frequency_cells,
            axis="frequency_hz",
            bottle_position_bound_m=bottle_bound,
            joint_position_bound_rad=joint_bound,
        )
    if not should_continue_solver_sweeps(
        selected_frequency_hz=selected_frequency
    ):
        blocked_report = {
            "schema_version": 1,
            "status": "PARTIAL",
            "classification": "DIAGNOSTIC_NUMERICAL_CONVERGENCE_ONLY",
            "reason": "TIMESTEP_NOT_CONVERGED_SOLVER_SWEEPS_NOT_RUN",
            "runtime": config["runtime"],
            "config": {
                "absolute_path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "frozen_stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_hash,
                "sha256_after": _sha256(stage_path),
            },
            "sample": {
                "sample_id": sample_id,
                "preflight_absolute_path": str(preflight_path),
                "preflight_sha256": _sha256(preflight_path),
                "bottle_transform_absolute_path": str(pose_path),
                "bottle_transform_sha256": _sha256(pose_path),
                "initial_arm_q_rad": initial_q,
            },
            "independent_bounds": config["independent_bounds"],
            "free_motion": free_motion,
            "free_motion_floor": {
                "maximum_analytic_position_error_m": max(
                    cell["analytic_position_error_norm_m"]
                    for cell in free_motion
                ),
                "maximum_com_velocity_error_m_s": max(
                    cell["com_velocity_max_error_m_s"]
                    for cell in free_motion
                ),
            },
            "frequency_sweep": {
                "cells": [_public_cell(cell) for cell in frequency_cells],
                "comparisons": frequency_comparisons,
                "optional_960_used": optional_used,
                "selected_frequency_hz": None,
            },
            "position_iteration_sweep": {
                "status": "NOT_RUN_TIMESTEP_NOT_CONVERGED",
                "cells": [],
                "comparisons": [],
                "selected_position_iterations": None,
            },
            "velocity_iteration_sweep": {
                "status": "NOT_RUN_TIMESTEP_NOT_CONVERGED",
                "cells": [],
                "comparisons": [],
                "selected_velocity_iterations": None,
            },
            "selection": {
                "frequency_hz": None,
                "position_iterations": None,
                "velocity_iterations": None,
                "fresh_repeat_signature_equal": None,
            },
            "boundaries": {
                **config["boundaries"],
                "grasp_pass_used_to_set_tolerance": False,
                "failure_video_required": any(
                    cell["machine_status"] != "PASS"
                    for cell in frequency_cells
                ),
                "failure_video_capture_status": (
                    "NOT_REQUIRED"
                    if all(
                        cell["machine_status"] == "PASS"
                        for cell in frequency_cells
                    )
                    else "SEE_PER_CELL_FAILURE_EVIDENCE"
                ),
                "extra_solver_cells_from_superseded_driver": {
                    "status": "EXCLUDED_FROM_DECISION_IF_PRESENT",
                    "absolute_paths": _superseded_solver_cell_paths(
                        artifact_root
                    ),
                },
            },
            "task8": "AUTHORIZED_PAUSED_AT_MODEL_PROOF",
        }
        blocked_report["deterministic_signature"] = _stable_report_signature(
            blocked_report
        )
        output = args.output.resolve()
        _atomic_json(output, blocked_report)
        _atomic_text(output.with_suffix(".md"), _markdown(blocked_report))
        return 2
    analysis_frequency = selected_frequency or max(frequencies)
    position_cells: list[dict[str, Any]] = []
    for position_value in matrix["position_iterations"]:
        position = int(position_value)
        reusable = next(
            (
                cell
                for cell in frequency_cells
                if cell["requested"]
                == {
                    "frequency_hz": analysis_frequency,
                    "position_iterations": position,
                    "velocity_iterations": int(
                        matrix["frequency_sweep_velocity_iterations"]
                    ),
                }
            ),
            None,
        )
        position_cells.append(
            reusable
            or _runtime_cell(
                family="position",
                frequency_hz=analysis_frequency,
                position_iterations=position,
                velocity_iterations=int(
                    matrix["frequency_sweep_velocity_iterations"]
                ),
                runtime_config=runtime_config,
                pose_path=pose_path,
                initial_q=initial_q,
                stage_path=stage_path,
                stage_hash=stage_hash,
                artifact_root=artifact_root,
                timeout_s=float(args.timeout_s),
            )
        )
    position_comparisons, selected_position = _selection_records(
        position_cells,
        axis="position_iterations",
        bottle_position_bound_m=bottle_bound,
        joint_position_bound_rad=joint_bound,
    )
    analysis_position = selected_position or max(matrix["position_iterations"])
    velocity_cells: list[dict[str, Any]] = []
    for velocity_value in matrix["velocity_iterations"]:
        velocity = int(velocity_value)
        reusable = next(
            (
                cell
                for cell in position_cells
                if cell["requested"]
                == {
                    "frequency_hz": analysis_frequency,
                    "position_iterations": int(analysis_position),
                    "velocity_iterations": velocity,
                }
            ),
            None,
        )
        velocity_cells.append(
            reusable
            or _runtime_cell(
                family="velocity",
                frequency_hz=analysis_frequency,
                position_iterations=int(analysis_position),
                velocity_iterations=velocity,
                runtime_config=runtime_config,
                pose_path=pose_path,
                initial_q=initial_q,
                stage_path=stage_path,
                stage_hash=stage_hash,
                artifact_root=artifact_root,
                timeout_s=float(args.timeout_s),
            )
        )
    velocity_comparisons, selected_velocity = _selection_records(
        velocity_cells,
        axis="velocity_iterations",
        bottle_position_bound_m=bottle_bound,
        joint_position_bound_rad=joint_bound,
    )
    analysis_velocity = selected_velocity or max(matrix["velocity_iterations"])
    final_cell = next(
        cell
        for cell in velocity_cells
        if int(cell["requested"]["velocity_iterations"]) == analysis_velocity
    )
    repeat = _runtime_cell(
        family="final_repeat",
        frequency_hz=analysis_frequency,
        position_iterations=int(analysis_position),
        velocity_iterations=int(analysis_velocity),
        runtime_config=runtime_config,
        pose_path=pose_path,
        initial_q=initial_q,
        stage_path=stage_path,
        stage_hash=stage_hash,
        artifact_root=artifact_root,
        timeout_s=float(args.timeout_s),
    )
    repeat_equal = (
        final_cell["deterministic_signature"]
        == repeat["deterministic_signature"]
        and final_cell["deterministic_signature"] is not None
    )
    fully_selected = all(
        value is not None
        for value in (selected_frequency, selected_position, selected_velocity)
    )
    status = "PASS" if fully_selected and repeat_equal else "PARTIAL"
    all_runtime_cells = [
        *frequency_cells,
        *position_cells,
        *velocity_cells,
        repeat,
    ]
    failure_cells = [
        cell for cell in all_runtime_cells if cell["machine_status"] != "PASS"
    ]
    if not failure_cells:
        failure_capture_status = "NOT_REQUIRED"
    elif all(
        (cell.get("failure_evidence") or {}).get("status") == "CAPTURED"
        for cell in failure_cells
    ):
        failure_capture_status = "CAPTURED_PENDING_VISUAL_MODEL_REVIEW"
    else:
        failure_capture_status = "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "classification": "DIAGNOSTIC_NUMERICAL_CONVERGENCE_ONLY",
        "runtime": config["runtime"],
        "config": {
            "absolute_path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "frozen_stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash,
            "sha256_after": _sha256(stage_path),
        },
        "sample": {
            "sample_id": sample_id,
            "preflight_absolute_path": str(preflight_path),
            "preflight_sha256": _sha256(preflight_path),
            "bottle_transform_absolute_path": str(pose_path),
            "bottle_transform_sha256": _sha256(pose_path),
            "initial_arm_q_rad": initial_q,
        },
        "independent_bounds": config["independent_bounds"],
        "free_motion": free_motion,
        "free_motion_floor": {
            "maximum_analytic_position_error_m": max(
                cell["analytic_position_error_norm_m"] for cell in free_motion
            ),
            "maximum_com_velocity_error_m_s": max(
                cell["com_velocity_max_error_m_s"] for cell in free_motion
            ),
        },
        "frequency_sweep": {
            "cells": [_public_cell(cell) for cell in frequency_cells],
            "comparisons": frequency_comparisons,
            "optional_960_used": optional_used,
            "selected_frequency_hz": selected_frequency,
        },
        "position_iteration_sweep": {
            "cells": [_public_cell(cell) for cell in position_cells],
            "comparisons": position_comparisons,
            "selected_position_iterations": selected_position,
        },
        "velocity_iteration_sweep": {
            "cells": [_public_cell(cell) for cell in velocity_cells],
            "comparisons": velocity_comparisons,
            "selected_velocity_iterations": selected_velocity,
        },
        "selection": {
            "frequency_hz": selected_frequency,
            "position_iterations": selected_position,
            "velocity_iterations": selected_velocity,
            "analysis_fallback": {
                "frequency_hz": analysis_frequency,
                "position_iterations": analysis_position,
                "velocity_iterations": analysis_velocity,
            },
            "fresh_repeat_signature_equal": repeat_equal,
            "final_cell_signature": final_cell["deterministic_signature"],
            "repeat_signature": repeat["deterministic_signature"],
            "repeat": _public_cell(repeat),
        },
        "boundaries": {
            **config["boundaries"],
            "grasp_pass_used_to_set_tolerance": False,
            "failure_video_required": any(
                cell["machine_status"] != "PASS" for cell in all_runtime_cells
            ),
            "failure_video_capture_status": failure_capture_status,
        },
        "task8": "AUTHORIZED_PAUSED_AT_MODEL_PROOF",
    }
    if report["frozen_stage"]["sha256_after"] != stage_hash:
        report["status"] = "FAIL"
    report["deterministic_signature"] = _stable_report_signature(report)
    output = args.output.resolve()
    _atomic_json(output, report)
    _atomic_text(output.with_suffix(".md"), _markdown(report))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

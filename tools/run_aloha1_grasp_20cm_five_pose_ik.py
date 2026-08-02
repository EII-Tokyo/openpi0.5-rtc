#!/usr/bin/env python3
"""Execute five frozen ALOHA Bottle500 grasps in fresh Isaac processes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv_issac/bin/python"
ISAAC_LAUNCHER = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"
EXPECTED_SAMPLE_IDS = [f"sample_{index:02d}" for index in range(1, 6)]
CLASSIFICATION = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reuse_accepted_runtime_records(
    source: dict[str, Any],
    *,
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    """Reuse only complete deterministic successes that the user accepted."""

    requested = [str(value) for value in sample_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("reused sample_ids must be unique")
    by_id = {
        str(record.get("sample_id")): record
        for record in source.get("samples", [])
    }
    reused: list[dict[str, Any]] = []
    for sample_id in requested:
        if sample_id not in by_id:
            raise ValueError(f"reused runtime sample missing: {sample_id}")
        source_record = by_id[sample_id]
        primary = source_record.get("primary", {})
        repeat = source_record.get("collider_repeat", {})
        gates = {
            "primary_machine_pass": primary.get("machine_status") == "PASS",
            "primary_evidence_pass": primary.get("evidence_status") == "PASS",
            "repeat_machine_pass": repeat.get("machine_status") == "PASS",
            "repeat_evidence_pass": repeat.get("evidence_status") == "PASS",
            "deterministic_signature_equal": (
                primary.get("deterministic_signature")
                == repeat.get("deterministic_signature")
                and primary.get("deterministic_signature") is not None
            ),
            "primary_video_pair_present": primary.get("video_count") == 2,
            "repeat_collision_evidence_complete": (
                repeat.get("collision_record_count") == 24
            ),
            "primary_initialization_contract_pass": (
                primary.get("initialization_contract_status") == "PASS"
            ),
            "repeat_initialization_contract_pass": (
                repeat.get("initialization_contract_status") == "PASS"
            ),
            "initialization_signature_equal": (
                primary.get("initialization_signature")
                == repeat.get("initialization_signature")
                and primary.get("initialization_signature") is not None
            ),
            "primary_finger_safety_pass": (
                primary.get("finger_safety_status") == "PASS"
                and primary.get("finger_safety_violation_count") == 0
            ),
            "repeat_finger_safety_pass": (
                repeat.get("finger_safety_status") == "PASS"
                and repeat.get("finger_safety_violation_count") == 0
            ),
        }
        if not all(gates.values()):
            failed = [name for name, passed in gates.items() if not passed]
            raise ValueError(
                f"reused runtime sample is not a complete success: "
                f"{sample_id}: {failed}"
            )
        record = copy.deepcopy(source_record)
        record["execution_policy"] = (
            "REUSED_USER_ACCEPTED_SUCCESS_NO_RERECORDING"
        )
        record["initial_orientation_policy"] = record.get(
            "initial_orientation_policy",
            "USER_ACCEPTED_LEGACY_INITIAL_ORIENTATION_EXCEPTION",
        )
        record["reuse_validation_gates"] = gates
        reused.append(record)
    return reused


def resume_verified_runtime_records(
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resume an interrupted batch without rerunning complete machine evidence.

    This is deliberately distinct from user-accepted reuse: it only preserves
    records whose clean-video primary and collision-evidence repeat both pass
    and whose deterministic signatures are identical.
    """

    sample_ids = [str(record.get("sample_id")) for record in source.get("samples", [])]
    if not sample_ids:
        raise ValueError("interrupted runtime contains no completed samples")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("interrupted runtime sample_ids must be unique")
    unexpected = sorted(set(sample_ids) - set(EXPECTED_SAMPLE_IDS))
    if unexpected:
        raise ValueError(f"interrupted runtime has unexpected samples: {unexpected}")

    resumed = reuse_accepted_runtime_records(source, sample_ids=sample_ids)
    for record in resumed:
        record["execution_policy"] = (
            "RESUMED_INTERRUPTED_MACHINE_SUCCESS_NO_RERECORDING"
        )
    return resumed


def build_five_pose_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the fixed five-sample machine gate before visual review."""

    sample_ids = [str(record.get("sample_id")) for record in records]
    process_ids = [
        int(run.get("process_id", -1))
        for record in records
        for run in (
            record.get("primary", {}),
            record.get("collider_repeat", {}),
        )
        if int(run.get("process_id", -1)) > 0
    ]
    per_sample_gates: list[dict[str, Any]] = []
    failed_sample_ids: list[str] = []
    evidence_failed_sample_ids: list[str] = []
    for record in records:
        sample_id = str(record.get("sample_id"))
        primary = record.get("primary", {})
        repeat = record.get("collider_repeat", {})
        machine_gates = {
            "primary_machine_pass": (
                primary.get("machine_status") == "PASS"
            ),
            "repeat_machine_pass": (
                repeat.get("machine_status") == "PASS"
            ),
            "paired_signature_equal": (
                primary.get("deterministic_signature")
                == repeat.get("deterministic_signature")
                and primary.get("deterministic_signature") is not None
            ),
            "primary_initialization_contract_pass": (
                primary.get("initialization_contract_status") == "PASS"
            ),
            "repeat_initialization_contract_pass": (
                repeat.get("initialization_contract_status") == "PASS"
            ),
            "paired_initialization_signature_equal": (
                primary.get("initialization_signature")
                == repeat.get("initialization_signature")
                and primary.get("initialization_signature") is not None
            ),
            "primary_finger_safety_pass": (
                primary.get("finger_safety_status") == "PASS"
                and primary.get("finger_safety_violation_count") == 0
            ),
            "repeat_finger_safety_pass": (
                repeat.get("finger_safety_status") == "PASS"
                and repeat.get("finger_safety_violation_count") == 0
            ),
        }
        evidence_gates = {
            "primary_process_exit_zero": primary.get("exit_code") == 0,
            "repeat_process_exit_zero": repeat.get("exit_code") == 0,
            "primary_evidence_complete": (
                primary.get("evidence_status") == "PASS"
            ),
            "repeat_evidence_complete": (
                repeat.get("evidence_status") == "PASS"
            ),
            "raw_and_annotated_video_present": (
                primary.get("video_count") == 2
            ),
            "collision_screenshot_records_complete": (
                repeat.get("collision_record_count") == 24
            ),
        }
        machine_status = (
            "PASS" if all(machine_gates.values()) else "FAIL"
        )
        evidence_status = (
            "PASS" if all(evidence_gates.values()) else "FAIL"
        )
        status = (
            "PASS"
            if machine_status == "PASS" and evidence_status == "PASS"
            else "FAIL"
        )
        if machine_status == "FAIL":
            failed_sample_ids.append(sample_id)
        if evidence_status == "FAIL":
            evidence_failed_sample_ids.append(sample_id)
        per_sample_gates.append(
            {
                "sample_id": sample_id,
                "machine_gates": machine_gates,
                "evidence_gates": evidence_gates,
                "machine_status": machine_status,
                "evidence_status": evidence_status,
                "status": status,
            }
        )
    machine_pass_count = sum(
        record["machine_status"] == "PASS"
        for record in per_sample_gates
    )
    evidence_pass_count = sum(
        record["evidence_status"] == "PASS"
        for record in per_sample_gates
    )
    machine_global_gates = {
        "exact_five_frozen_sample_ids": (
            sample_ids == EXPECTED_SAMPLE_IDS
        ),
        "ten_fresh_unique_processes": (
            len(process_ids) == 10 and len(set(process_ids)) == 10
        ),
        "all_five_machine_pass": machine_pass_count == 5,
    }
    evidence_global_gates = {
        "all_five_evidence_complete": evidence_pass_count == 5,
    }
    machine_pass = all(machine_global_gates.values())
    evidence_pass = all(evidence_global_gates.values())
    return {
        "status": (
            "PARTIAL" if machine_pass and evidence_pass else "FAIL"
        ),
        "machine_status": "PASS" if machine_pass else "FAIL",
        "machine_pass_count": machine_pass_count,
        "evidence_pass_count": evidence_pass_count,
        "required_sample_count": 5,
        "primary_video_count": sum(
            record.get("primary", {}).get("video_count") == 2
            for record in records
        ),
        "fresh_process_count": len(set(process_ids)),
        "failed_sample_ids": failed_sample_ids,
        "evidence_failed_sample_ids": evidence_failed_sample_ids,
        "global_gates": {
            **machine_global_gates,
            **evidence_global_gates,
        },
        "per_sample_gates": per_sample_gates,
        "visual_model_review": "NOT_RUN",
        "user_confirmation": "NOT_RUN",
        "promotion_status": (
            "AWAITING_VISUAL_MODEL_REVIEW"
            if machine_pass
            else "BLOCKED_BY_MACHINE_FAILURE"
        ),
        "task8": "NOT_RUN",
    }


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
        process_id = int(process.pid)
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
        "process_id": process_id,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "environment": {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": str(ROOT),
        },
        "log_absolute_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
    }


def _validated_video_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    kinds: set[str] = set()
    for record in records:
        path = Path(str(record.get("absolute_path", "")))
        kind = str(record.get("kind"))
        if not path.is_file():
            errors.append(f"missing_video:{kind}")
            continue
        actual_hash = _sha256(path)
        expected_hash = str(record.get("sha256"))
        probe = record.get("probe", {})
        if actual_hash != expected_hash:
            errors.append(f"video_hash:{kind}")
        if int(probe.get("frame_count", 0)) <= 0:
            errors.append(f"video_frames:{kind}")
        if float(probe.get("fps", 0.0)) <= 0.0:
            errors.append(f"video_fps:{kind}")
        resolution = probe.get("resolution", [])
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or min(int(value) for value in resolution) <= 0
        ):
            errors.append(f"video_resolution:{kind}")
        kinds.add(kind)
        validated.append(
            {
                **record,
                "actual_sha256": actual_hash,
                "absolute_path": str(path.resolve()),
            }
        )
    if kinds != {"raw", "annotated"}:
        errors.append("video_kinds")
    return validated, errors


def _read_run_evidence(
    *,
    artifact_root: Path,
    process: dict[str, Any],
    collision_repeat: bool,
    selected: dict[str, Any],
    stage_sha256: str,
    readback_tolerance_rad: float,
    first_frame_jump_tolerance_rad: float,
    hold_frames: int,
) -> dict[str, Any]:
    result = dict(process)
    runtime_path = artifact_root / "aloha1_grasp_20cm_runtime.json"
    telemetry_path = artifact_root / "aloha1_grasp_20cm_telemetry.jsonl"
    candidate_path = (
        artifact_root
        / "video_attempt_001/video/candidate_manifest.json"
    )
    missing_mandatory = [
        str(path)
        for path in (runtime_path, telemetry_path)
        if not path.is_file()
    ]
    if missing_mandatory:
        result.update(
            {
                "machine_status": "MISSING_REPORT",
                "evidence_status": "FAIL",
                "evidence_errors": [
                    f"missing:{path}" for path in missing_mandatory
                ],
                "video_count": 0,
                "collision_record_count": 0,
            }
        )
        return result

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    candidate = (
        json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate_path.is_file()
        else {}
    )
    expected_transform = np.asarray(
        selected["world_from_object"],
        dtype=np.float64,
    )
    actual_transform = np.asarray(
        runtime.get("bottle_random_position", {}).get(
            "world_from_object",
            [],
        ),
        dtype=np.float64,
    )
    expected_q = np.asarray(selected["initial_arm_q_rad"], dtype=np.float64)
    initial_pose = runtime.get("runtime", {}).get("initial_pose", {})
    initialization_contract = runtime.get("runtime", {}).get(
        "initialization_contract",
        {},
    )
    finger_safety = runtime.get("runtime", {}).get("finger_safety", {})
    actual_q = np.asarray(
        initial_pose.get("initial_arm_q_target_rad", []),
        dtype=np.float64,
    )
    stage = runtime.get("stage", {})
    metrics = runtime.get("metrics", {})
    boundaries = runtime.get("boundaries", {})
    evidence_errors: list[str] = []
    gates = {
        "process_exit_zero": process.get("exit_code") == 0,
        "process_not_timed_out": process.get("timed_out") is False,
        "runtime_machine_pass": runtime.get("status") == "PASS",
        "stage_hash_unchanged": (
            stage.get("sha256_before") == stage_sha256
            and stage.get("sha256_after") == stage_sha256
        ),
        "frozen_bottle_transform_applied": (
            actual_transform.shape == (4, 4)
            and np.allclose(
                actual_transform,
                expected_transform,
                rtol=0.0,
                atol=1.0e-12,
            )
            and runtime.get("bottle_random_position", {}).get("pose_mode")
            == "FROZEN_CENTER_AND_YAW_TRANSFORM"
        ),
        "frozen_initial_arm_q_applied": (
            actual_q.shape == (6,)
            and np.allclose(
                actual_q,
                expected_q,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "initial_pose_hold_complete": (
            initial_pose.get("initial_pose_hold_frames_required")
            == hold_frames
            and initial_pose.get("initial_pose_hold_frames_observed")
            == hold_frames
        ),
        "initial_readback_within_gate": (
            math.isfinite(
                float(
                    initial_pose.get(
                        "initial_arm_max_readback_error_rad",
                        math.nan,
                    )
                )
            )
            and float(
                initial_pose["initial_arm_max_readback_error_rad"]
            )
            <= readback_tolerance_rad
        ),
        "first_frame_jump_within_gate": (
            math.isfinite(
                float(initial_pose.get("first_frame_jump_rad", math.nan))
            )
            and float(initial_pose["first_frame_jump_rad"])
            <= first_frame_jump_tolerance_rad
        ),
        "formal_bottle_dynamic": (
            metrics.get("dynamic_during_formal_phases") is True
        ),
        "no_forbidden_constraint": (
            metrics.get("forbidden_constraint") is False
            and boundaries.get("surface_gripper") is False
            and boundaries.get("fixed_joint") is False
            and boundaries.get("parent_attachment") is False
        ),
        "finite_state": metrics.get("finite_state") is True,
        "task8_not_run": boundaries.get("task8") == "NOT_RUN",
        "telemetry_nonempty": telemetry_path.stat().st_size > 0,
        "initialization_contract_pass": (
            initialization_contract.get("status") == "PASS"
            and initialization_contract.get("signature") is not None
        ),
        "finger_safety_pass": (
            finger_safety.get("status") == "PASS"
            and finger_safety.get("violation_count") == 0
        ),
    }
    raw_videos = list(candidate.get("videos", []))
    if collision_repeat:
        validated_videos, video_errors = [], []
    else:
        validated_videos, video_errors = _validated_video_records(raw_videos)
    collision = candidate.get("collision_evidence", {})
    collision_records = list(collision.get("records", []))
    if collision_repeat:
        gates["collision_evidence_complete"] = (
            collision.get("status")
            in {"PASS", "AWAITING_VISUAL_MODEL_REVIEW"}
            and len(collision_records) == 24
        )
    else:
        gates["clean_primary_video"] = (
            collision.get("status") == "NOT_RUN_PRIMARY_CLEAN_VIDEO"
            and len(collision_records) == 0
            and len(validated_videos) == 2
            and not video_errors
        )
    evidence_errors.extend(
        name for name, passed in gates.items() if not passed
    )
    evidence_errors.extend(video_errors)
    if not candidate_path.is_file():
        evidence_errors.append(f"missing:{candidate_path}")
    result.update(
        {
            "machine_status": runtime.get("status"),
            "machine_reason": runtime.get("reason"),
            "evidence_status": (
                "PASS" if not evidence_errors else "FAIL"
            ),
            "evidence_errors": evidence_errors,
            "evidence_gates": gates,
            "deterministic_signature": runtime.get(
                "deterministic_signature"
            ),
            "initialization_contract_status": (
                initialization_contract.get("status")
            ),
            "initialization_signature": initialization_contract.get(
                "signature"
            ),
            "finger_safety_status": finger_safety.get("status"),
            "finger_safety_violation_count": finger_safety.get(
                "violation_count"
            ),
            "runtime_report_absolute_path": str(runtime_path.resolve()),
            "runtime_report_sha256": _sha256(runtime_path),
            "telemetry_absolute_path": str(telemetry_path.resolve()),
            "telemetry_sha256": _sha256(telemetry_path),
            "telemetry_line_count": sum(
                1
                for line in telemetry_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ),
            "candidate_status": candidate.get("status"),
            "candidate_manifest_absolute_path": str(
                candidate_path.resolve()
            ),
            "candidate_manifest_sha256": (
                _sha256(candidate_path) if candidate_path.is_file() else None
            ),
            "videos": validated_videos if not collision_repeat else [],
            "video_count": (
                len(validated_videos) if not collision_repeat else 0
            ),
            "collision_status": collision.get("status"),
            "collision_records": (
                collision_records if collision_repeat else []
            ),
            "collision_record_count": (
                len(collision_records) if collision_repeat else 0
            ),
            "metrics": metrics,
            "stage": stage,
            "bottle_random_position": runtime.get(
                "bottle_random_position"
            ),
            "initial_pose": initial_pose,
            "runtime_ik": runtime.get("runtime", {}).get("ik"),
            "boundaries": boundaries,
        }
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-results", type=Path)
    parser.add_argument("--resume-results", type=Path)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve(strict=True)
    preflight_path = args.preflight.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise RuntimeError("five-pose config schema mismatch")
    if config.get("classification") != CLASSIFICATION:
        raise RuntimeError("five-pose classification mismatch")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("selected_sample_count") != 5
    ):
        raise RuntimeError("five-pose preflight is not PASS")
    selected_samples = list(preflight["selected_samples"])
    if [record["sample_id"] for record in selected_samples] != (
        EXPECTED_SAMPLE_IDS
    ):
        raise RuntimeError("frozen sample IDs/order changed")
    if preflight.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("preflight does not bind current config")
    stage_path = Path(
        config["frozen_inputs"]["approved_stage"]["path"]
    )
    if not stage_path.is_absolute():
        stage_path = ROOT / stage_path
    stage_path = stage_path.resolve(strict=True)
    stage_sha256 = str(
        config["frozen_inputs"]["approved_stage"]["sha256"]
    )
    if _sha256(stage_path) != stage_sha256:
        raise RuntimeError("approved Stage hash changed before runtime")
    runtime_config_record = config["frozen_inputs"]["runtime_config"]
    runtime_config_path = Path(str(runtime_config_record["path"]))
    if not runtime_config_path.is_absolute():
        runtime_config_path = ROOT / runtime_config_path
    runtime_config_path = runtime_config_path.resolve(strict=True)
    runtime_config_sha256 = str(runtime_config_record["sha256"])
    if _sha256(runtime_config_path) != runtime_config_sha256:
        raise RuntimeError("frozen runtime config hash changed before runtime")
    if not ISAAC_PYTHON.is_file():
        raise RuntimeError(f"missing project Isaac Python: {ISAAC_PYTHON}")
    timeout_s = float(args.timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout must be finite and positive")
    artifact_root = args.artifact_root.resolve()
    if artifact_root.exists():
        raise FileExistsError(
            f"fresh runtime artifact root already exists: {artifact_root}"
        )
    artifact_root.mkdir(parents=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"runtime output already exists: {output_path}"
        )
    hold_frames = int(config["runtime"]["initial_pose_hold_frames"])
    readback_gate = float(
        config["gates"]["initial_arm_readback_tolerance_rad"]
    )
    jump_gate = float(
        config["gates"]["first_frame_jump_tolerance_rad"]
    )
    phase_readback_gate = float(
        config["gates"]["arm_phase_readback_tolerance_rad"]
    )
    trajectory_config = config["arm_trajectory"]
    arm_trajectory_mode = str(trajectory_config["mode"])
    acceleration_limits = [
        float(value)
        for value in trajectory_config["acceleration_limits_rad_s2"]
    ]
    trajectory_source = ROOT / str(
        trajectory_config["source"]["local_path"]
    )
    if (
        not trajectory_source.is_file()
        or _sha256(trajectory_source)
        != str(trajectory_config["source"]["sha256"])
    ):
        raise RuntimeError(
            "frozen official ViperX-300 trajectory source changed"
        )
    preserved_ids = [
        str(value)
        for value in config["sampling"].get(
            "preserved_success_sample_ids",
            [],
        )
    ]
    reuse_manifest: dict[str, Any] = {"status": "NOT_USED"}
    records: list[dict[str, Any]] = []
    if args.reuse_results is not None and args.resume_results is not None:
        raise RuntimeError(
            "--reuse-results and --resume-results are mutually exclusive"
        )
    if args.resume_results is not None:
        resume_path = args.resume_results.resolve(strict=True)
        resume_source = json.loads(resume_path.read_text(encoding="utf-8"))
        expected_bindings = {
            "config": _sha256(config_path),
            "preflight": _sha256(preflight_path),
            "runtime_config": runtime_config_sha256,
            "stage_after": stage_sha256,
        }
        observed_bindings = {
            "config": resume_source.get("config", {}).get("sha256"),
            "preflight": resume_source.get("preflight", {}).get("sha256"),
            "runtime_config": resume_source.get("runtime_config", {}).get(
                "sha256"
            ),
            "stage_after": resume_source.get("stage", {}).get("sha256_after"),
        }
        if observed_bindings != expected_bindings:
            raise RuntimeError(
                "interrupted runtime bindings changed: "
                f"expected={expected_bindings}, observed={observed_bindings}"
            )
        records = resume_verified_runtime_records(resume_source)
        reuse_manifest = {
            "status": "PASS",
            "absolute_path": str(resume_path),
            "sha256": _sha256(resume_path),
            "sample_ids": [str(record["sample_id"]) for record in records],
            "policy": "RESUMED_INTERRUPTED_MACHINE_SUCCESS_NO_RERECORDING",
        }
    elif args.reuse_results is not None:
        reuse_path = args.reuse_results.resolve(strict=True)
        reuse_source = json.loads(reuse_path.read_text(encoding="utf-8"))
        records = reuse_accepted_runtime_records(
            reuse_source,
            sample_ids=preserved_ids,
        )
        reuse_manifest = {
            "status": "PASS",
            "absolute_path": str(reuse_path),
            "sha256": _sha256(reuse_path),
            "sample_ids": preserved_ids,
            "policy": "REUSED_USER_ACCEPTED_SUCCESS_NO_RERECORDING",
        }
    elif preserved_ids:
        raise RuntimeError(
            "config preserves prior successes but --reuse-results was omitted"
        )
    reused_id_set = {str(record["sample_id"]) for record in records}
    selected_to_run = [
        record
        for record in selected_samples
        if str(record["sample_id"]) not in reused_id_set
    ]
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "machine_status": "NOT_RUN",
        "classification": CLASSIFICATION,
        "config": {
            "absolute_path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "preflight": {
            "absolute_path": str(preflight_path),
            "sha256": _sha256(preflight_path),
            "deterministic_signature": preflight[
                "deterministic_signature"
            ],
        },
        "runtime_config": {
            "absolute_path": str(runtime_config_path),
            "sha256": runtime_config_sha256,
        },
        "samples": records,
        "reused_successes": reuse_manifest,
        "task8": "NOT_RUN",
    }
    _atomic_json(output_path, aggregate)
    for selected in selected_to_run:
        sample_id = str(selected["sample_id"])
        sample_root = artifact_root / sample_id
        sample_root.mkdir()
        pose_path = sample_root / "frozen_world_from_object.json"
        _atomic_json(
            pose_path,
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "world_from_object": selected["world_from_object"],
            },
        )
        q = [float(value) for value in selected["initial_arm_q_rad"]]
        common = [
            str(ISAAC_PYTHON),
            str(ISAAC_LAUNCHER),
            "--config",
            str(runtime_config_path),
            "--autorun",
            "--close-after-terminal",
            "--bottle-world-from-object-json",
            str(pose_path),
            "--initial-arm-q-rad",
            *(repr(value) for value in q),
            "--initial-pose-hold-frames",
            str(hold_frames),
            "--arm-phase-readback-tolerance-rad",
            repr(phase_readback_gate),
            "--arm-trajectory-mode",
            arm_trajectory_mode,
            "--arm-acceleration-limits-rad-s2",
            *(repr(value) for value in acceleration_limits),
        ]
        stage_hash_before_primary = _sha256(stage_path)
        primary_root = sample_root / "primary"
        primary_process = _run_logged(
            command=[
                *common,
                "--artifact-root",
                str(primary_root),
                "--skip-collider-evidence",
            ],
            log_path=sample_root / "primary.log",
            timeout_s=timeout_s,
        )
        stage_hash_after_primary = _sha256(stage_path)
        primary = _read_run_evidence(
            artifact_root=primary_root,
            process=primary_process,
            collision_repeat=False,
            selected=selected,
            stage_sha256=stage_sha256,
            readback_tolerance_rad=readback_gate,
            first_frame_jump_tolerance_rad=jump_gate,
            hold_frames=hold_frames,
        )
        primary["source_stage_sha256_before_process"] = (
            stage_hash_before_primary
        )
        primary["source_stage_sha256_after_process"] = (
            stage_hash_after_primary
        )
        repeat_root = sample_root / "collider_repeat"
        stage_hash_before_repeat = _sha256(stage_path)
        repeat_process = _run_logged(
            command=[
                *common,
                "--artifact-root",
                str(repeat_root),
                "--collision-evidence-only",
            ],
            log_path=sample_root / "collider_repeat.log",
            timeout_s=timeout_s,
        )
        stage_hash_after_repeat = _sha256(stage_path)
        repeat = _read_run_evidence(
            artifact_root=repeat_root,
            process=repeat_process,
            collision_repeat=True,
            selected=selected,
            stage_sha256=stage_sha256,
            readback_tolerance_rad=readback_gate,
            first_frame_jump_tolerance_rad=jump_gate,
            hold_frames=hold_frames,
        )
        repeat["source_stage_sha256_before_process"] = (
            stage_hash_before_repeat
        )
        repeat["source_stage_sha256_after_process"] = (
            stage_hash_after_repeat
        )
        records.append(
            {
                "sample_id": sample_id,
                "candidate_index": int(selected["candidate_index"]),
                "seed": int(selected["seed"]),
                "frozen_pose_manifest": {
                    "absolute_path": str(pose_path.resolve()),
                    "sha256": _sha256(pose_path),
                },
                "bottle_geometric_center_world_m": selected[
                    "bottle_geometric_center_world_m"
                ],
                "bottle_line_yaw_deg": selected[
                    "bottle_line_yaw_deg"
                ],
                "a_world_m": selected["a_world_m"],
                "b_world_m": selected["b_world_m"],
                "axis_unit_world": selected["axis_unit_world"],
                "axis_to_world_z_deg": selected[
                    "axis_to_world_z_deg"
                ],
                "lowest_point_to_table_gap_m": selected[
                    "lowest_point_to_table_gap_m"
                ],
                "world_from_object": selected["world_from_object"],
                "initial_arm_q_rad": q,
                "initial_ee_position_world_m": selected[
                    "initial_ee_position_world_m"
                ],
                "initial_ee_orientation_world_wxyz": selected[
                    "initial_ee_orientation_world_wxyz"
                ],
                "preflight_ik": selected["ik"],
                "preflight_initial_collision": selected[
                    "initial_collision"
                ],
                "primary": primary,
                "collider_repeat": repeat,
                "visual_review_status": "NOT_REVIEWED",
                "user_confirmation": "NOT_RUN",
            }
        )
        records.sort(
            key=lambda record: EXPECTED_SAMPLE_IDS.index(
                str(record["sample_id"])
            )
        )
        aggregate = {
            "schema_version": 1,
            **build_five_pose_summary(records),
            "classification": CLASSIFICATION,
            "config": {
                "absolute_path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "preflight": {
                "absolute_path": str(preflight_path),
                "sha256": _sha256(preflight_path),
                "deterministic_signature": preflight[
                    "deterministic_signature"
                ],
            },
            "runtime_config": {
                "absolute_path": str(runtime_config_path),
                "sha256": runtime_config_sha256,
            },
            "stage": {
                "absolute_path": str(stage_path),
                "sha256_before": stage_sha256,
                "sha256_after": _sha256(stage_path),
            },
            "reused_successes": reuse_manifest,
            "samples": records,
            "boundaries": {
                **config["boundaries"],
                "task8": "NOT_RUN",
            },
        }
        _atomic_json(output_path, aggregate)
    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "machine_status": aggregate["machine_status"],
                "failed_sample_ids": aggregate["failed_sample_ids"],
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if aggregate["machine_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

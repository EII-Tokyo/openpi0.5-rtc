#!/usr/bin/env python3
"""Run one fail-closed GraspTester diagnostic without claiming GUI evidence.

This is a fresh-process, scripted equivalent of the Grasp Editor's tester.  It
does not run a table task and it deliberately provides no inverse-kinematics
path.  Its strongest possible outcome is tester-only diagnostic evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE_PATH = REPO_ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/"
    "aloha1_signal_correspondence_workcell.usda"
)
BOTTLE_PATH = (
    REPO_ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
)
CANDIDATE_PATH = REPO_ROOT / (
    "configs/aloha1_grasps/"
    "bottle500_horizontal_body_grasp.isaac_grasp.yaml"
)
TRANSFORM_REPORT_PATH = REPO_ROOT / (
    "reports/aloha1_mapping/aloha1_grasp_transform_validation.json"
)

FROZEN_INPUTS = {
    "stage": {
        "path": STAGE_PATH,
        "sha256": (
            "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
        ),
    },
    "bottle": {
        "path": BOTTLE_PATH,
        "sha256": (
            "16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e"
        ),
    },
    "candidate": {
        "path": CANDIDATE_PATH,
        "sha256": (
            "b3307c86a44101eadd6ed2151722e7668bb7d644422378765d98eac906835cca"
        ),
    },
    "transform_report": {
        "path": TRANSFORM_REPORT_PATH,
        "sha256": (
            "37d36dcbb4bfd7a9fdc39f96565c796bdc0d9b8d571172bf4639251a23b3f329"
        ),
    },
}

EXPECTED_VERSIONS = {
    "isaac_sim": "5.1.0.0",
    "kit": "107.3.3",
    "physx": "107.3.26",
    "grasp_editor": "2.0.20",
}

EXPECTED_DOF_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
)
INITIAL_ARM_Q_RAD = (
    -0.16720470786094666,
    0.5324101448059082,
    -0.017540352419018745,
    -0.3624092638492584,
    0.9591664671897888,
    -0.11042828112840652,
)

ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
GRIPPER_FRAME_PATH = (
    "/World/follower_left/vx300s_left/follower_left_ee_gripper_link"
)
LEFT_FINGER_PATH = (
    "/World/follower_left/vx300s_left/"
    "follower_left_left_finger_link"
)
RIGHT_FINGER_PATH = (
    "/World/follower_left/vx300s_left/"
    "follower_left_right_finger_link"
)
TABLE_PATH = "/World/environment/worldBody/user_confirmed_table"
REQUIRED_STAGE_PRIM_PATHS = (
    ARTICULATION_PATH,
    GRIPPER_FRAME_PATH,
    LEFT_FINGER_PATH,
    RIGHT_FINGER_PATH,
    TABLE_PATH,
)
SESSION_ROOT_PATH = "/World/ALOHA1GraspEditorSession"
BOTTLE_SESSION_PATH = f"{SESSION_ROOT_PATH}/Bottle500"

OPEN_LEFT_M = 0.057
OPEN_RIGHT_M = -0.057
CLOSE_LEFT_M = 0.021
CLOSE_RIGHT_M = -0.021
CLOSE_SPEED_M_S = 0.02
PHYSICS_DT_S = 1.0 / 60.0
BOTTLE_MASS_KG = 0.020
FRICTION = 0.7
RESTITUTION = 0.0
EXPECTED_BOTTLE_COLLISIONS = 41
MAX_STEPS = 3600
MAX_SIM_TIME_S = 60.0
MAX_WALL_TIME_S = 180.0
MAX_STAGE_LOAD_WALL_TIME_S = 60.0
MAX_IMPORT_ERROR_MESSAGE_CHARS = 1000
MAX_IMPORT_TRACEBACK_CHARS = 8000

TOP_LEVEL_EVIDENCE = {
    "status": "PARTIAL",
    "gui_evidence": "GUI_PENDING",
    "ik": "NOT_RUN",
    "classification": "DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI",
}
PLACEMENT_SCOPE = "NOT_TABLE_TASK/NOT_IK"
ARM_HOLD_STATUS = "INCONCLUSIVE_NO_APPROVED_ARM_HOLD_TOLERANCE"
MIMIC_STATUS = "INCONCLUSIVE_NO_APPROVED_MIMIC_TOLERANCE"

_VARIANTS = {
    "dual_active_exact_candidate": {
        "name": "dual_active_exact_candidate",
        "active_joints": ("left_finger", "right_finger"),
        "native_export_joints": ("left_finger", "right_finger"),
        "observer_joints": (),
        "mimic_commandability_risk": True,
        "recommended": False,
    },
    "left_active_mimic_observed": {
        "name": "left_active_mimic_observed",
        "active_joints": ("left_finger",),
        "native_export_joints": ("left_finger",),
        "observer_joints": ("right_finger",),
        "mimic_commandability_risk": False,
        "recommended": True,
    },
}
_VARIANT_ALIASES = {
    "A": "dual_active_exact_candidate",
    "B": "left_active_mimic_observed",
    **{name: name for name in _VARIANTS},
}


def resolve_variant(name: str) -> dict[str, Any]:
    """Return the only two approved active-joint configurations."""
    canonical = _VARIANT_ALIASES.get(name)
    if canonical is None:
        raise ValueError(
            "invalid variant; choose A/dual_active_exact_candidate or "
            "B/left_active_mimic_observed"
        )
    return copy.deepcopy(_VARIANTS[canonical])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )
    return actual


def verify_all_frozen_inputs() -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for label, item in FROZEN_INPUTS.items():
        path = Path(item["path"])
        verified[label] = {
            "path": str(path),
            "sha256": verify_sha256(
                path,
                str(item["sha256"]),
                label=label,
            ),
        }
    return verified


def validate_dof_order(actual: Sequence[str]) -> None:
    actual_tuple = tuple(str(item) for item in actual)
    if actual_tuple != EXPECTED_DOF_ORDER:
        raise RuntimeError(
            "DOF order mismatch: "
            f"expected {list(EXPECTED_DOF_ORDER)}, got {list(actual_tuple)}"
        )


def resolve_full_experience(package_file: str | Path) -> Path:
    """Resolve the full Kit experience from the installed isaacsim package."""
    package_init = Path(package_file).expanduser().resolve()
    if not package_init.is_file():
        raise FileNotFoundError(
            f"installed isaacsim package file is missing: {package_init}"
        )
    experience = (
        package_init.parent / "apps" / "isaacsim.exp.full.kit"
    ).resolve()
    if not experience.is_absolute() or not experience.is_file():
        raise FileNotFoundError(
            f"installed Isaac Sim full experience is missing: {experience}"
        )
    return experience


def compute_world_from_object(
    report_or_path: Mapping[str, Any] | str | Path,
) -> list[list[float]]:
    """Compute T_W_O = T_W_G @ inverse(T_O_G) from the frozen report."""
    import numpy as np

    if isinstance(report_or_path, Mapping):
        report = report_or_path
    else:
        report = json.loads(
            Path(report_or_path).read_text(encoding="utf-8")
        )
    matrices = report["matrices"]
    world_from_gripper = np.asarray(
        matrices["world_from_gripper_reference"],
        dtype=float,
    )
    object_from_gripper = np.asarray(
        matrices["object_from_gripper"],
        dtype=float,
    )
    if world_from_gripper.shape != (4, 4) or object_from_gripper.shape != (
        4,
        4,
    ):
        raise ValueError("transform matrices must both be 4x4")
    result = world_from_gripper @ np.linalg.inv(object_from_gripper)
    if not np.isfinite(result).all():
        raise ValueError("computed world-from-object transform is nonfinite")
    return result.tolist()


def canonical_signature(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_signature", None)
    encoded = json.dumps(
        json_safe(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    """Preserve nonfinite evidence in strict JSON using explicit tags."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "NaN"
        elif value > 0.0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {"__nonfinite_float__": label}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return [json_safe(item) for item in value]
    return value


_VOLATILE_SIGNATURE_KEYS = {
    "artifact_dir",
    "canonical_signature",
    "export_path",
    "output_path",
    "report_path",
    "telemetry_path",
    "traceback",
    "wall_time_s",
}


def _without_volatile_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_volatile_fields(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_SIGNATURE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return [_without_volatile_fields(item) for item in value]
    return value


def deterministic_trial_signature(record: Mapping[str, Any]) -> str:
    return canonical_signature(_without_volatile_fields(record))


def deterministic_run_signature(report: Mapping[str, Any]) -> str:
    basis = {
        "variant": report.get("variant", {}).get("name"),
        "trial_classification": report.get("trial_classification"),
        "frozen_inputs": report.get("frozen_inputs"),
        "cleanup": report.get("cleanup"),
        "native_export_validation": report.get(
            "native_export_validation"
        ),
        "trial": report.get("trial"),
    }
    return canonical_signature(_without_volatile_fields(basis))


def validate_native_export(
    export_path: Path,
    active_joints: Sequence[str],
) -> dict[str, Any]:
    expected_active = tuple(str(name) for name in active_joints)
    if not export_path.is_file():
        raise RuntimeError(f"native export is not a file: {export_path}")
    size_bytes = export_path.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError("native export is empty")
    try:
        payload = yaml.safe_load(export_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"native export YAML parse failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("native export root is not a mapping")
    if payload.get("format") != "isaac_grasp":
        raise RuntimeError("native export format mismatch")
    if float(payload.get("format_version", -1.0)) != 1.0:
        raise RuntimeError("native export format_version mismatch")
    if payload.get("object_frame") != BOTTLE_SESSION_PATH:
        raise RuntimeError("native export object_frame mismatch")
    if payload.get("gripper_frame") != GRIPPER_FRAME_PATH:
        raise RuntimeError("native export gripper_frame mismatch")
    grasps = payload.get("grasps")
    if not isinstance(grasps, Mapping) or len(grasps) != 1:
        raise RuntimeError("native export must contain exactly one grasp")
    grasp = next(iter(grasps.values()))
    if not isinstance(grasp, Mapping):
        raise RuntimeError("native export grasp is not a mapping")
    cspace = grasp.get("cspace_position")
    pregrasp = grasp.get("pregrasp_cspace_position")
    if not isinstance(cspace, Mapping) or tuple(cspace) != expected_active:
        raise RuntimeError("native export cspace active-joint mismatch")
    if (
        not isinstance(pregrasp, Mapping)
        or tuple(pregrasp) != expected_active
    ):
        raise RuntimeError(
            "native export pregrasp active-joint mismatch"
        )
    if _contains_nonfinite(payload):
        raise RuntimeError("native export contains nonfinite values")
    position = grasp.get("position")
    orientation = grasp.get("orientation")
    if not isinstance(position, Sequence) or len(position) != 3:
        raise RuntimeError("native export position is not length 3")
    if not isinstance(orientation, Mapping):
        raise RuntimeError("native export orientation is not a mapping")
    orientation_xyz = orientation.get("xyz")
    if (
        "w" not in orientation
        or not isinstance(orientation_xyz, Sequence)
        or len(orientation_xyz) != 3
    ):
        raise RuntimeError("native export orientation shape mismatch")
    return {
        "sha256": sha256_file(export_path),
        "size_bytes": size_bytes,
        "format": payload["format"],
        "format_version": float(payload["format_version"]),
        "object_frame": payload["object_frame"],
        "gripper_frame": payload["gripper_frame"],
        "active_joints": list(expected_active),
        "grasp_count": len(grasps),
        "finite": True,
    }


def native_export_status(classification: str) -> str:
    if classification == "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS":
        return "WRITE_NATIVE_EXPORT"
    return f"NOT_WRITTEN_{classification}"


def trial_exit_code(classification: str) -> int:
    return (
        0
        if classification == "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
        else 1
    )


def validate_output_paths(
    report: Path,
    export: Path,
    telemetry: Path,
    artifact_dir: Path,
    *,
    frozen_paths: Sequence[Path] | None = None,
) -> dict[str, Path]:
    """Resolve every output and reject path, inode, or directory aliasing."""
    resolved_artifact_dir = artifact_dir.expanduser().resolve()
    if resolved_artifact_dir.exists() and not resolved_artifact_dir.is_dir():
        raise ValueError(
            f"artifact-dir must be a directory or absent: "
            f"{resolved_artifact_dir}"
        )
    outputs = {
        "report": report.expanduser().resolve(),
        "export": export.expanduser().resolve(),
        "telemetry": telemetry.expanduser().resolve(),
        "runtime_contract_snapshot": (
            resolved_artifact_dir / "runtime_contract_snapshot.json"
        ).resolve(),
    }
    for label, path in outputs.items():
        if path == resolved_artifact_dir:
            raise ValueError(f"{label} cannot alias artifact-dir")
        if path.exists() and not path.is_file():
            raise ValueError(f"{label} output must be a file or absent: {path}")

    output_items = list(outputs.items())
    for index, (left_label, left_path) in enumerate(output_items):
        for right_label, right_path in output_items[index + 1 :]:
            if left_path == right_path:
                raise ValueError(
                    "output paths must be unique: "
                    f"{left_label} and {right_label}"
                )
            if (
                left_path.exists()
                and right_path.exists()
                and os.path.samefile(left_path, right_path)
            ):
                raise ValueError(
                    "output paths must be unique files: "
                    f"{left_label} and {right_label}"
                )

    frozen = (
        tuple(Path(item["path"]) for item in FROZEN_INPUTS.values())
        if frozen_paths is None
        else tuple(frozen_paths)
    )
    for output_label, output_path in output_items:
        for frozen_path in frozen:
            resolved_frozen = frozen_path.expanduser().resolve()
            if output_path == resolved_frozen:
                raise ValueError(
                    f"{output_label} aliases frozen input: {resolved_frozen}"
                )
            if (
                output_path.exists()
                and resolved_frozen.exists()
                and os.path.samefile(output_path, resolved_frozen)
            ):
                raise ValueError(
                    f"{output_label} is same file as frozen input: "
                    f"{resolved_frozen}"
                )
    return {**outputs, "artifact_dir": resolved_artifact_dir}


def wait_for_required_stage_prims(
    context: Any,
    simulation_app: Any,
    required_paths: Sequence[str],
    *,
    timeout_s: float = MAX_STAGE_LOAD_WALL_TIME_S,
    monotonic: Any = time.monotonic,
) -> Any:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("Stage readiness timeout must be finite and positive")
    started = float(monotonic())
    if not math.isfinite(started):
        raise RuntimeError("Stage readiness clock is nonfinite")
    update_count = 0
    while True:
        stage = context.get_stage()
        stage_was_none = stage is None
        missing_required_prim_paths = sorted(
            str(path)
            for path in required_paths
            if stage is None or not stage.GetPrimAtPath(path).IsValid()
        )
        if not missing_required_prim_paths:
            return stage
        if not simulation_app.is_running():
            raise RuntimeError("SimulationApp stopped during Stage loading")
        elapsed = float(monotonic()) - started
        if not math.isfinite(elapsed):
            raise RuntimeError("Stage readiness elapsed time is nonfinite")
        if elapsed >= timeout_s:
            raise RuntimeError(
                "stage_load_timeout: "
                f"stage_was_none={stage_was_none}; "
                "missing_required_prim_paths="
                f"{json.dumps(missing_required_prim_paths)}; "
                f"elapsed_s={elapsed:.6f}; "
                f"update_count={update_count}"
            )
        simulation_app.update()
        update_count += 1


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int | float):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _contact_paths(contact: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(contact.get(key, ""))
        for key in (
            "body0_path",
            "body1_path",
            "collider0_path",
            "collider1_path",
        )
    )


def _contact_involves_bottle(contact: Mapping[str, Any]) -> bool:
    return any(BOTTLE_SESSION_PATH in path for path in _contact_paths(contact))


def _is_penetrating_contact(contact: Mapping[str, Any]) -> bool:
    separation = float(contact.get("separation_m", math.inf))
    impulse = float(contact.get("impulse_ns", math.nan))
    return math.isfinite(separation) and math.isfinite(impulse) and (
        separation <= 0.0 and impulse > 0.0
    )


def evaluate_trial(record: Mapping[str, Any]) -> str:
    """Classify diagnostic evidence without elevating it to a task pass."""
    if not bool(record.get("frozen_hashes_verified", False)):
        return "FAIL_FROZEN_INPUT"
    if tuple(record.get("actual_dof_order", ())) != EXPECTED_DOF_ORDER:
        return "FAIL_DOF_ORDER"
    if record.get("timeout_reasons"):
        return "FAIL_TIMEOUT"
    if int(record.get("hold_command_count", 0)) <= 0:
        return "FAIL_MISSING_ARM_HOLD_COMMAND"
    if _contains_nonfinite(record.get("telemetry", {})) or _contains_nonfinite(
        record.get("contacts", [])
    ):
        return "FAIL_NONFINITE_TELEMETRY"
    if int(record.get("tester_terminal_callbacks", 0)) != 1:
        return "FAIL_NO_TERMINAL_RESULT"
    if not bool(record.get("tester_success", False)):
        return "FAIL_GRASP_TESTER"

    bottle_contacts = [
        contact
        for contact in record.get("contacts", [])
        if _contact_involves_bottle(contact)
    ]
    forbidden_tokens = ("gripper", "bar", "base")
    for contact in bottle_contacts:
        non_bottle_paths = [
            path.lower()
            for path in _contact_paths(contact)
            if BOTTLE_SESSION_PATH not in path
        ]
        if any(
            token in path
            for path in non_bottle_paths
            for token in forbidden_tokens
        ):
            return "FAIL_FORBIDDEN_BOTTLE_CONTACT"

    left_steps = {
        int(contact["physics_step"])
        for contact in bottle_contacts
        if "physics_step" in contact
        and _is_penetrating_contact(contact)
        and any("left_finger" in path for path in _contact_paths(contact))
    }
    right_steps = {
        int(contact["physics_step"])
        for contact in bottle_contacts
        if "physics_step" in contact
        and _is_penetrating_contact(contact)
        and any("right_finger" in path for path in _contact_paths(contact))
    }
    if not left_steps.intersection(right_steps):
        return "INCONCLUSIVE_NO_BILATERAL_CONTACT"
    return "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(
            json_safe(payload),
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _publish_pre_shutdown(
    report: dict[str, Any],
    *,
    report_path: Path,
    telemetry_path: Path,
    export_path: Path,
    simulation_app: Any,
    verified_frozen_manifest: Mapping[str, Any],
    frozen_verifier: Any = verify_all_frozen_inputs,
    json_writer: Any = _atomic_write_json,
) -> dict[str, Any]:
    publication_error: Exception | None = None
    try:
        report_path.unlink(missing_ok=True)
        telemetry_path.unlink(missing_ok=True)
        frozen_verifier()
        classification = str(report["trial_classification"])
        trial = report.get("trial", {})
        report["publication_phase"] = (
            "PRE_KIT_SHUTDOWN_AFTER_PHYSICS_CLEANUP"
        )
        report["simulation_app_close_status"] = (
            "SCHEDULED_AS_FINAL_ACTION_NOT_POST_READABLE"
        )
        report["shell_exit_code_is_not_authoritative"] = True
        report["intended_exit_code"] = trial_exit_code(classification)
        report["frozen_inputs"] = dict(verified_frozen_manifest)
        signature_basis = (
            trial
            if isinstance(trial, Mapping) and trial
            else {"trial_classification": classification}
        )
        report["deterministic_trial_signature"] = (
            deterministic_trial_signature(signature_basis)
        )
        report["deterministic_run_signature"] = (
            deterministic_run_signature(report)
        )
        report["canonical_signature"] = canonical_signature(report)
        telemetry_payload = {
            **TOP_LEVEL_EVIDENCE,
            "publication_phase": report["publication_phase"],
            "simulation_app_close_status": report[
                "simulation_app_close_status"
            ],
            "shell_exit_code_is_not_authoritative": True,
            "intended_exit_code": report["intended_exit_code"],
            "variant": report.get("variant", {}).get("name", "UNKNOWN"),
            "trial_classification": classification,
            "deterministic_trial_signature": report[
                "deterministic_trial_signature"
            ],
            "deterministic_run_signature": report[
                "deterministic_run_signature"
            ],
            "telemetry": (
                trial.get("telemetry", [])
                if isinstance(trial, Mapping)
                else []
            ),
            "contacts": (
                trial.get("contacts", [])
                if isinstance(trial, Mapping)
                else []
            ),
            "tester_status_messages": (
                trial.get("tester_status_messages", [])
                if isinstance(trial, Mapping)
                else []
            ),
        }
        json_writer(telemetry_path, telemetry_payload)
        frozen_verifier()
        json_writer(report_path, report)
    except Exception as exc:
        publication_error = exc
        for partial_path in (report_path, telemetry_path, export_path):
            try:
                partial_path.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                _emit_bounded_error(
                    "PUBLICATION_PARTIAL_REMOVAL_FAILURE",
                    cleanup_exc,
                )
        _emit_bounded_error("PUBLICATION_FAILURE_BEFORE_CLOSE", exc)
    finally:
        try:
            simulation_app.close()
        except Exception as close_exc:
            _emit_bounded_error(
                "SIMULATION_APP_CLOSE_RETURNED_ERROR",
                close_exc,
            )
            if publication_error is None:
                raise
    if publication_error is not None:
        raise publication_error
    return report


def _emit_bounded_error(event: str, exc: Exception) -> None:
    payload = {
        "event": event,
        "exception_type": type(exc).__name__,
        "message": str(exc)[:MAX_IMPORT_ERROR_MESSAGE_CHARS],
        "traceback": "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
                limit=8,
            )
        )[-MAX_IMPORT_TRACEBACK_CHARS:],
    }
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _emit_bounded_import_error(exc: Exception) -> None:
    _emit_bounded_error("ISAAC_POST_SIMULATION_APP_IMPORT_FAILURE", exc)


def _version_prefix(extension_manager: Any, extension_name: str) -> str:
    extension_id = extension_manager.get_enabled_extension_id(extension_name)
    if not extension_id:
        raise RuntimeError(f"required extension is not enabled: {extension_name}")
    details = extension_manager.get_extension_dict(extension_id)
    version = str(details.get("package", {}).get("version", ""))
    return version.split("+", maxsplit=1)[0]


def _assert_versions(app: Any, carb: Any) -> dict[str, str]:
    isaac_version = importlib.metadata.version("isaacsim")
    kit_version = carb.tokens.get_tokens_interface().resolve(
        "${kit_version}"
    ).split("+", maxsplit=1)[0]
    manager = app.get_extension_manager()
    manager.set_extension_enabled_immediate(
        "isaacsim.robot_setup.grasp_editor",
        True,  # noqa: FBT003 - Kit's API requires a positional boolean.
    )
    app.update()
    grasp_version = _version_prefix(
        manager,
        "isaacsim.robot_setup.grasp_editor",
    )
    physx_version = _version_prefix(manager, "omni.physx")
    actual = {
        "isaac_sim": isaac_version,
        "kit": kit_version,
        "physx": physx_version,
        "grasp_editor": grasp_version,
    }
    if actual != EXPECTED_VERSIONS:
        raise RuntimeError(
            f"runtime version mismatch: expected {EXPECTED_VERSIONS}, "
            f"got {actual}"
        )
    return actual


def _snapshot_root(stage: Any) -> dict[str, Any]:
    root = stage.GetRootLayer()
    serialized = root.ExportToString()
    return {
        "serialized": serialized,
        "serialized_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "sublayers": list(root.subLayerPaths),
        "references": sorted(
            line.strip()
            for line in serialized.splitlines()
            if "references" in line
        ),
        "dirty": bool(root.dirty),
        "identifier": str(root.identifier),
        "real_path": str(root.realPath),
        "resolved_path": str(root.resolvedPath),
        "runtime_metadata": {
            "time_codes_per_second": {
                "authored": bool(root.HasTimeCodesPerSecond()),
                "value": (
                    float(root.timeCodesPerSecond)
                    if root.HasTimeCodesPerSecond()
                    else None
                ),
            },
            "start_time_code": {
                "authored": bool(root.HasStartTimeCode()),
                "value": (
                    float(root.startTimeCode)
                    if root.HasStartTimeCode()
                    else None
                ),
            },
            "end_time_code": {
                "authored": bool(root.HasEndTimeCode()),
                "value": (
                    float(root.endTimeCode)
                    if root.HasEndTimeCode()
                    else None
                ),
            },
            "custom_layer_data": {
                "authored": bool(root.HasCustomLayerData()),
                "value": (
                    copy.deepcopy(root.customLayerData)
                    if root.HasCustomLayerData()
                    else None
                ),
            },
        },
    }


def restore_root_runtime_metadata(
    root: Any,
    root_before: Mapping[str, Any],
) -> None:
    metadata = root_before["runtime_metadata"]

    time_codes = metadata["time_codes_per_second"]
    if time_codes["authored"]:
        root.timeCodesPerSecond = time_codes["value"]
    else:
        root.ClearTimeCodesPerSecond()

    start_time = metadata["start_time_code"]
    if start_time["authored"]:
        root.startTimeCode = start_time["value"]
    else:
        root.ClearStartTimeCode()

    end_time = metadata["end_time_code"]
    if end_time["authored"]:
        root.endTimeCode = end_time["value"]
    else:
        root.ClearEndTimeCode()

    custom_data = metadata["custom_layer_data"]
    if custom_data["authored"]:
        root.customLayerData = copy.deepcopy(custom_data["value"])
    else:
        root.ClearCustomLayerData()

    expected_dirty = bool(root_before["dirty"])
    if not expected_dirty:
        verify_sha256(
            STAGE_PATH,
            FROZEN_INPUTS["stage"]["sha256"],
            label="stage before clean-root reload",
        )
        if not root.Reload(force=True):
            raise RuntimeError("clean root layer reload failed")
    if bool(root.dirty) != expected_dirty:
        raise RuntimeError(
            "root layer dirty state changed: "
            f"expected {expected_dirty}, got {bool(root.dirty)}"
        )


def _validate_bottle(
    stage: Any,
    bottle_prim: Any,
    usd: Any,
    usd_geom: Any,
    usd_physics: Any,
) -> dict[str, Any]:
    if not bottle_prim.IsValid():
        raise RuntimeError("Bottle500 reference did not compose")
    if not bottle_prim.HasAPI(usd_physics.RigidBodyAPI):
        raise RuntimeError("Bottle500 root is missing RigidBodyAPI")

    meshes = [
        prim
        for prim in usd.PrimRange(bottle_prim)
        if prim.IsA(usd_geom.Mesh)
    ]
    if not meshes:
        raise RuntimeError("Bottle500 has no real composed mesh")
    nested_rigid = [
        str(prim.GetPath())
        for prim in usd.PrimRange(bottle_prim)
        if prim != bottle_prim and prim.HasAPI(usd_physics.RigidBodyAPI)
    ]
    if nested_rigid:
        raise RuntimeError(f"Bottle500 has nested rigid bodies: {nested_rigid}")
    collisions = [
        str(prim.GetPath())
        for prim in usd.PrimRange(bottle_prim)
        if prim.HasAPI(usd_physics.CollisionAPI)
    ]
    if len(collisions) != EXPECTED_BOTTLE_COLLISIONS:
        raise RuntimeError(
            "Bottle500 collision count mismatch: "
            f"expected {EXPECTED_BOTTLE_COLLISIONS}, got {len(collisions)}"
        )

    bbox_cache = usd_geom.BBoxCache(
        usd.TimeCode.Default(),
        [usd_geom.Tokens.default_],
    )
    aligned = bbox_cache.ComputeWorldBound(bottle_prim).ComputeAlignedBox()
    minimum = list(aligned.GetMin())
    maximum = list(aligned.GetMax())
    if not all(math.isfinite(float(v)) for v in minimum + maximum):
        raise RuntimeError("Bottle500 bounding box is nonfinite")
    if any(
        float(hi) <= float(lo)
        for lo, hi in zip(minimum, maximum, strict=True)
    ):
        raise RuntimeError("Bottle500 bounding box is degenerate")
    return {
        "mesh_count": len(meshes),
        "collision_count": len(collisions),
        "nested_rigid_body_paths": nested_rigid,
        "world_bbox_min": minimum,
        "world_bbox_max": maximum,
    }


def _validate_bottle_physics_material_binding(
    bottle_prim: Any,
    material: Any,
    usd: Any,
    usd_physics: Any,
    usd_shade: Any,
) -> dict[str, Any]:
    expected_material_path = str(material.GetPath())
    binding_api = usd_shade.MaterialBindingAPI(bottle_prim)
    direct_binding = binding_api.GetDirectBinding(
        materialPurpose="physics",
    )
    direct_material_path = str(direct_binding.GetMaterialPath())
    if direct_material_path != expected_material_path:
        raise RuntimeError(
            "Bottle500 physics material direct-binding mismatch: "
            f"expected {expected_material_path}, got {direct_material_path}"
        )

    material_api = usd_physics.MaterialAPI(material.GetPrim())
    material_values = {
        "static_friction": float(
            material_api.GetStaticFrictionAttr().Get(),
        ),
        "dynamic_friction": float(
            material_api.GetDynamicFrictionAttr().Get(),
        ),
        "restitution": float(
            material_api.GetRestitutionAttr().Get(),
        ),
    }
    expected_values = {
        "static_friction": FRICTION,
        "dynamic_friction": FRICTION,
        "restitution": RESTITUTION,
    }
    if any(
        not math.isclose(
            material_values[name],
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        for name, expected in expected_values.items()
    ):
        raise RuntimeError(
            "Bottle500 physics material value mismatch: "
            f"expected {expected_values}, got {material_values}"
        )

    collision_paths: list[str] = []
    mismatched_collision_bindings: list[dict[str, str]] = []
    for prim in usd.PrimRange(bottle_prim):
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        collision_path = str(prim.GetPath())
        collision_paths.append(collision_path)
        bound_material, _ = usd_shade.MaterialBindingAPI(
            prim,
        ).ComputeBoundMaterial(materialPurpose="physics")
        bound_material_path = (
            str(bound_material.GetPath()) if bound_material else ""
        )
        if bound_material_path != expected_material_path:
            mismatched_collision_bindings.append(
                {
                    "collision_path": collision_path,
                    "bound_material_path": bound_material_path,
                },
            )
    if len(collision_paths) != EXPECTED_BOTTLE_COLLISIONS:
        raise RuntimeError(
            "Bottle500 physics material collision count mismatch: "
            f"expected {EXPECTED_BOTTLE_COLLISIONS}, "
            f"got {len(collision_paths)}"
        )
    if mismatched_collision_bindings:
        raise RuntimeError(
            "Bottle500 collision physics material mismatch: "
            f"{mismatched_collision_bindings}"
        )
    return {
        "purpose": "physics",
        "direct_material_path": direct_material_path,
        "material_values": material_values,
        "collision_count": len(collision_paths),
        "all_collision_bindings_match": True,
    }


def _matrix_pose(
    matrix: Sequence[Sequence[float]],
    np: Any,
) -> tuple[Any, Any]:
    homogeneous = np.asarray(matrix, dtype=float)
    translation = homogeneous[:3, 3].copy()
    rotation = homogeneous[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        orientation = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        final_index = (index + 2) % 3
        scale = math.sqrt(
            1.0
            + float(rotation[index, index])
            - float(rotation[next_index, next_index])
            - float(rotation[final_index, final_index])
        ) * 2.0
        orientation = np.zeros(4, dtype=float)
        orientation[index + 1] = 0.25 * scale
        orientation[0] = (
            rotation[final_index, next_index]
            - rotation[next_index, final_index]
        ) / scale
        orientation[next_index + 1] = (
            rotation[next_index, index]
            + rotation[index, next_index]
        ) / scale
        orientation[final_index + 1] = (
            rotation[final_index, index]
            + rotation[index, final_index]
        ) / scale
    norm = float(np.linalg.norm(orientation))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("world-from-object rotation produced bad quaternion")
    orientation /= norm
    return translation, orientation


def _path_from_id(value: int, schema_tools: Any) -> str:
    return str(schema_tools.intToSdfPath(int(value)))


def _serialize_contacts(
    headers: Sequence[Any],
    data: Sequence[Any],
    schema_tools: Any,
    *,
    physics_step: int,
    sim_time_s: float,
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for header in headers:
        paths = {
            "body0_path": _path_from_id(header.actor0, schema_tools),
            "body1_path": _path_from_id(header.actor1, schema_tools),
            "collider0_path": _path_from_id(
                header.collider0,
                schema_tools,
            ),
            "collider1_path": _path_from_id(
                header.collider1,
                schema_tools,
            ),
        }
        offset = int(header.contact_data_offset)
        contacts.extend(
            [
                {
                    **paths,
                    "event_type": str(header.type),
                    "physics_step": physics_step,
                    "sim_time_s": sim_time_s,
                    "position_m": [float(v) for v in item.position],
                    "normal": [float(v) for v in item.normal],
                    "impulse_ns": math.sqrt(
                        sum(float(v) ** 2 for v in item.impulse)
                    ),
                    "separation_m": float(item.separation),
                }
                for item in data[
                    offset : offset + int(header.num_contact_data)
                ]
            ]
        )
    return contacts


def _run_isaac(
    variant: Mapping[str, Any],
    export_path: Path,
    artifact_dir: Path,
    report_path: Path,
    telemetry_path: Path,
    verified_frozen_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    # This must remain the first Isaac/Kit import in the process.
    import isaacsim as isaacsim_package
    from isaacsim import SimulationApp

    full_experience_path = resolve_full_experience(
        isaacsim_package.__file__
    )
    simulation_app = SimulationApp(
        {
            "fast_shutdown": False,
            "headless": True,
            "sync_loads": True,
        },
        experience=str(full_experience_path),
    )

    # No Kit or USD module is imported until SimulationApp exists. Import
    # failure is protected so app closure and frozen-hash checks still run.
    try:
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_setup.grasp_editor.data_writer import DataWriter
        from isaacsim.robot_setup.grasp_editor.grasp_tester import GraspTester
        from isaacsim.robot_setup.grasp_editor.grasp_tester import GraspTestResults
        from isaacsim.robot_setup.grasp_editor.grasp_tester import GraspTestSettings
        import numpy as np
        from omni import physx as omni_physx
        import omni.kit.app
        import omni.usd
        from pxr import PhysicsSchemaTools
        from pxr import PhysxSchema
        from pxr import Sdf
        from pxr import Usd
        from pxr import UsdGeom
        from pxr import UsdPhysics
        from pxr import UsdShade
    except Exception as exc:
        _emit_bounded_import_error(exc)
        import_failure_report = {
            **TOP_LEVEL_EVIDENCE,
            "placement_scope": PLACEMENT_SCOPE,
            "variant": dict(variant),
            "fresh_process_intended": True,
            "arm_hold_status": ARM_HOLD_STATUS,
            "mimic_status": MIMIC_STATUS,
            "native_export_status": "NOT_WRITTEN_FAIL_RUNTIME_IMPORT",
            "trial_classification": "FAIL_RUNTIME_IMPORT",
            "runtime_error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "cleanup": {
                "errors": [],
                "post_cleanup_hash_errors": [],
                "no_persistent_stage_write": True,
            },
        }
        return _publish_pre_shutdown(
            import_failure_report,
            report_path=report_path,
            telemetry_path=telemetry_path,
            export_path=export_path,
            simulation_app=simulation_app,
            verified_frozen_manifest=verified_frozen_manifest,
        )

    report: dict[str, Any] = {
        **TOP_LEVEL_EVIDENCE,
        "placement_scope": PLACEMENT_SCOPE,
        "variant": dict(variant),
        "fresh_process_intended": True,
        "arm_hold_status": ARM_HOLD_STATUS,
        "mimic_status": MIMIC_STATUS,
        "native_export_status": "NOT_WRITTEN",
        "cleanup": {},
    }
    stage = None
    world = None
    articulation = None
    bottle = None
    contact_subscription = None
    callback_registered = False
    diagnostic_layer = None
    previous_edit_target = None
    session_sublayers_before = None
    root_before = None
    contacts: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "successful_yields": 0,
        "tester_terminal_callbacks": 0,
        "tester_success": False,
        "tester_status_messages": [],
        "hold_command_count": 0,
        "timeout_reasons": [],
        "telemetry": [],
        "physics_step": 0,
    }
    cleanup_errors: list[str] = []

    try:
        app = omni.kit.app.get_app()
        report["versions"] = _assert_versions(app, carb)
        context = omni.usd.get_context()
        if not context.open_stage(str(STAGE_PATH)):
            raise RuntimeError(f"failed to open frozen Stage: {STAGE_PATH}")
        stage = wait_for_required_stage_prims(
            context,
            simulation_app,
            REQUIRED_STAGE_PRIM_PATHS,
        )

        root_before = _snapshot_root(stage)
        previous_edit_target = stage.GetEditTarget()
        session_layer = stage.GetSessionLayer()
        session_sublayers_before = tuple(session_layer.subLayerPaths)
        diagnostic_layer = Sdf.Layer.CreateAnonymous(
            "aloha1_grasp_tester_scripted_equivalent.usda"
        )
        session_layer.subLayerPaths.append(diagnostic_layer.identifier)
        stage.SetEditTarget(Usd.EditTarget(diagnostic_layer))
        if stage.GetEditTarget().GetLayer() is not diagnostic_layer:
            raise RuntimeError("anonymous diagnostic layer is not edit target")

        UsdGeom.Xform.Define(stage, SESSION_ROOT_PATH)
        bottle_prim = UsdGeom.Xform.Define(
            stage,
            BOTTLE_SESSION_PATH,
        ).GetPrim()
        bottle_prim.GetReferences().AddReference(
            str(BOTTLE_PATH),
            Sdf.Path("/Bottle500"),
        )
        simulation_app.update()
        report["bottle_validation"] = _validate_bottle(
            stage,
            bottle_prim,
            Usd,
            UsdGeom,
            UsdPhysics,
        )

        mass_api = UsdPhysics.MassAPI.Apply(bottle_prim)
        mass_api.CreateMassAttr(BOTTLE_MASS_KG)
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            bottle_prim
        ).CreateDisableGravityAttr(
            True,  # noqa: FBT003 - USD attribute creation is positional.
        )

        material = UsdShade.Material.Define(
            stage,
            f"{SESSION_ROOT_PATH}/BottlePhysicsMaterial",
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(FRICTION)
        material_api.CreateDynamicFrictionAttr(FRICTION)
        material_api.CreateRestitutionAttr(RESTITUTION)
        UsdShade.MaterialBindingAPI.Apply(bottle_prim).Bind(
            material,
            materialPurpose="physics",
        )
        report["physics_material_validation"] = (
            _validate_bottle_physics_material_binding(
                bottle_prim,
                material,
                Usd,
                UsdPhysics,
                UsdShade,
            )
        )

        for path in (
            BOTTLE_SESSION_PATH,
            LEFT_FINGER_PATH,
            RIGHT_FINGER_PATH,
        ):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"missing contact-report body: {path}")
            PhysxSchema.PhysxContactReportAPI.Apply(
                prim
            ).CreateThresholdAttr(0.0)

        world_from_object = compute_world_from_object(TRANSFORM_REPORT_PATH)
        position, orientation = _matrix_pose(world_from_object, np)
        report["placement"] = {
            "scope": PLACEMENT_SCOPE,
            "formula": "T_W_O = T_W_G @ inverse(T_O_G)",
            "world_from_object": world_from_object,
        }

        world = World(
            physics_dt=PHYSICS_DT_S,
            rendering_dt=PHYSICS_DT_S,
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
        )
        physics_context = world.get_physics_context()
        physics_context.set_solve_articulation_contact_last(True)
        solve_contact_last = (
            physics_context.get_solve_articulation_contact_last()
        )
        if solve_contact_last is not True:
            raise RuntimeError(
                "solve_articulation_contact_last readback was not True"
            )
        report["solve_articulation_contact_last"] = solve_contact_last
        articulation = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH,
                name="complete_follower_left",
                reset_xform_properties=False,
            )
        )
        bottle = world.scene.add(
            RigidPrim(
                prim_paths_expr=BOTTLE_SESSION_PATH,
                name="session_bottle500_batch",
                positions=np.asarray([position]),
                orientations=np.asarray([orientation]),
                masses=np.asarray([BOTTLE_MASS_KG]),
                reset_xform_properties=True,
            )
        )

        # The reset is intentionally complete before callback registration.
        world.reset()
        validate_dof_order(articulation.dof_names)
        bottle.disable_gravities()
        bottle.set_velocities(np.zeros((1, 6), dtype=float))

        initial_q = np.asarray(
            articulation.get_joint_positions(),
            dtype=float,
        )
        if initial_q.shape != (len(EXPECTED_DOF_ORDER),):
            raise RuntimeError(
                f"unexpected joint-position shape: {initial_q.shape}"
            )
        initial_q[:6] = np.asarray(INITIAL_ARM_Q_RAD)
        initial_q[7] = OPEN_LEFT_M
        initial_q[8] = OPEN_RIGHT_M
        articulation.set_joint_positions(initial_q)
        hold_indices = np.arange(7, dtype=np.int32)
        hold_targets = initial_q[:7].copy()

        active = tuple(variant["active_joints"])
        joint_values = {
            "left_finger": (
                OPEN_LEFT_M,
                CLOSE_LEFT_M,
                CLOSE_SPEED_M_S,
            ),
            "right_finger": (
                OPEN_RIGHT_M,
                CLOSE_RIGHT_M,
                CLOSE_SPEED_M_S,
            ),
        }
        settings = GraspTestSettings(
            articulation_path=ARTICULATION_PATH,
            articulation_pose_frame=GRIPPER_FRAME_PATH,
            active_joints=list(active),
            active_joint_open_positions=[
                joint_values[name][0] for name in active
            ],
            active_joint_closed_positions=[
                joint_values[name][1] for name in active
            ],
            active_joint_close_speeds=[
                joint_values[name][2] for name in active
            ],
            inactive_joint_fixed_positions=[
                float(initial_q[index])
                for index, name in enumerate(EXPECTED_DOF_ORDER)
                if name not in active
            ],
            rigid_body_path=BOTTLE_SESSION_PATH,
            rigid_body_pose_frame=BOTTLE_SESSION_PATH,
            external_force_magnitude=0.0,
            external_torque_magnitude=0.0,
        )
        tester = GraspTester()
        tester.initialize_test_grasp_script(
            articulation,
            bottle,
            settings,
        )

        def on_contacts(
            contact_headers: Sequence[Any],
            contact_data: Sequence[Any],
        ) -> None:
            contacts.extend(
                _serialize_contacts(
                    contact_headers,
                    contact_data,
                    PhysicsSchemaTools,
                    physics_step=int(state["physics_step"]),
                    sim_time_s=(
                        int(state["physics_step"]) * PHYSICS_DT_S
                    ),
                )
            )

        contact_subscription = (
            omni_physx.get_physx_simulation_interface()
            .subscribe_contact_report_events(on_contacts)
        )

        def on_physics_step(dt: float) -> None:
            if state["tester_terminal_callbacks"]:
                return
            if not math.isfinite(float(dt)) or float(dt) <= 0.0:
                state["timeout_reasons"].append("nonfinite_step_dt")
                return
            articulation.apply_action(
                ArticulationAction(
                    joint_positions=hold_targets,
                    joint_indices=hold_indices,
                )
            )
            state["hold_command_count"] += 1
            result = tester.update_grasp_test(dt)
            if isinstance(result, GraspTestResults):
                state["tester_terminal_callbacks"] += 1
                state["tester_success"] = bool(result.success)
                state["terminal_result"] = result
            else:
                state["successful_yields"] += 1
                if result not in (None, (), ""):
                    state["tester_status_messages"].append(str(result))

        world.add_physics_callback(
            "aloha1_grasp_tester_scripted_equivalent",
            on_physics_step,
        )
        callback_registered = True

        started = time.monotonic()
        steps = 0
        while (
            state["tester_terminal_callbacks"] == 0
            and not state["timeout_reasons"]
        ):
            elapsed = time.monotonic() - started
            sim_time = steps * PHYSICS_DT_S
            if elapsed >= MAX_WALL_TIME_S:
                state["timeout_reasons"].append("wall_timeout")
                break
            if sim_time >= MAX_SIM_TIME_S:
                state["timeout_reasons"].append("sim_timeout")
                break
            if steps >= MAX_STEPS:
                state["timeout_reasons"].append("step_timeout")
                break
            state["physics_step"] = steps + 1
            world.step(render=False)
            steps += 1
            q = np.asarray(articulation.get_joint_positions(), dtype=float)
            velocities = np.asarray(
                articulation.get_joint_velocities(),
                dtype=float,
            )
            bottle_velocity = np.asarray(
                bottle.get_velocities(),
                dtype=float,
            )
            state["telemetry"].append(
                {
                    "physics_step": steps,
                    "sim_time_s": steps * PHYSICS_DT_S,
                    "wall_time_s": time.monotonic() - started,
                    "joint_positions": q.tolist(),
                    "joint_velocities": velocities.tolist(),
                    "arm_hold_error_rad": (
                        q[:6] - np.asarray(INITIAL_ARM_Q_RAD)
                    ).tolist(),
                    "aux_gripper_hold_error": float(q[6] - hold_targets[6]),
                    "mimic_observation": {
                        "left_position_m": float(q[7]),
                        "right_position_m": float(q[8]),
                        "bilateral_sum_m": float(q[7] + q[8]),
                    },
                    "bottle_velocity": bottle_velocity.tolist(),
                }
            )

        if callback_registered:
            world.remove_physics_callback(
                "aloha1_grasp_tester_scripted_equivalent"
            )
            callback_registered = False

        trial_record = {
            "frozen_hashes_verified": True,
            "actual_dof_order": list(articulation.dof_names),
            "timeout_reasons": list(state["timeout_reasons"]),
            "tester_terminal_callbacks": state[
                "tester_terminal_callbacks"
            ],
            "tester_success": state["tester_success"],
            "successful_yields": state["successful_yields"],
            "tester_status_messages": list(
                state["tester_status_messages"]
            ),
            "hold_command_count": state["hold_command_count"],
            "telemetry": state["telemetry"],
            "contacts": contacts,
        }
        report["trial"] = trial_record
        trial_classification = evaluate_trial(trial_record)
        report["trial_classification"] = trial_classification
        report["deterministic_trial_signature"] = (
            deterministic_trial_signature(trial_record)
        )
        report["arm_hold_status"] = (
            "FAIL_NONFINITE_OR_MISSING_ARM_HOLD"
            if state["hold_command_count"] <= 0
            or _contains_nonfinite(state["telemetry"])
            else ARM_HOLD_STATUS
        )
        report["mimic_status"] = (
            "FAIL_NONFINITE_MIMIC_OBSERVATION"
            if _contains_nonfinite(state["telemetry"])
            else MIMIC_STATUS
        )

        result = state.get("terminal_result")
        export_decision = native_export_status(trial_classification)
        report["native_export_status"] = export_decision
        if (
            result is not None
            and trial_classification
            == "GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS"
        ):
            export_path.parent.mkdir(parents=True, exist_ok=True)
            writer = DataWriter(
                GRIPPER_FRAME_PATH,
                BOTTLE_SESSION_PATH,
            )
            writer.write_grasp_to_file(
                result,
                float(result.suggested_confidence),
                str(export_path),
            )
            report["native_export_validation"] = validate_native_export(
                export_path,
                active,
            )
            report["native_export_status"] = "WRITTEN_FROM_GRASP_TESTER"
            report["native_export_path"] = str(export_path)
            report["native_export_active_joints"] = list(active)
            exported_grasp = next(iter(writer.data["grasps"].values()))
            if tuple(exported_grasp["cspace_position"]) != active:
                raise RuntimeError(
                    "native DataWriter active-joint semantics changed"
                )
        elif export_path.exists():
            export_path.unlink()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            artifact_dir / "runtime_contract_snapshot.json",
            {
                "versions": report["versions"],
                "bottle_validation": report["bottle_validation"],
                "placement": report["placement"],
                "variant": dict(variant),
            },
        )
    except Exception as exc:
        _emit_bounded_error("RUNTIME_FAILURE_BEFORE_CLOSE", exc)
        if export_path.exists():
            try:
                export_path.unlink()
            except Exception as unlink_exc:
                cleanup_errors.append(
                    f"remove invalid native export: {unlink_exc}"
                )
        report["runtime_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["trial_classification"] = "FAIL_RUNTIME"
        report["native_export_status"] = "NOT_WRITTEN_FAIL_RUNTIME"
    finally:
        if world is not None and callback_registered:
            try:
                world.remove_physics_callback(
                    "aloha1_grasp_tester_scripted_equivalent"
                )
            except Exception as exc:
                cleanup_errors.append(f"remove callback: {exc}")
        del contact_subscription
        if world is not None:
            try:
                world.stop()
            except Exception as exc:
                cleanup_errors.append(f"stop world: {exc}")
        articulation = None
        bottle = None
        world = None
        try:
            World.clear_instance()
        except Exception as exc:
            cleanup_errors.append(f"clear World singleton: {exc}")

        if stage is not None and previous_edit_target is not None:
            try:
                stage.SetEditTarget(previous_edit_target)
                if (
                    stage.GetEditTarget().GetLayer()
                    is not previous_edit_target.GetLayer()
                ):
                    raise RuntimeError("previous edit target was not restored")
            except Exception as exc:
                cleanup_errors.append(f"restore edit target: {exc}")
        if stage is not None and diagnostic_layer is not None:
            try:
                paths = stage.GetSessionLayer().subLayerPaths
                matches = [
                    index
                    for index, value in enumerate(paths)
                    if value == diagnostic_layer.identifier
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one anonymous sublayer, found {len(matches)}"
                    )
                del paths[matches[0]]
                if diagnostic_layer.identifier in paths:
                    raise RuntimeError("anonymous sublayer removal was not exact")
                if (
                    session_sublayers_before is None
                    or tuple(paths) != session_sublayers_before
                ):
                    raise RuntimeError(
                        "session sublayers were not restored exactly"
                    )
            except Exception as exc:
                cleanup_errors.append(f"remove anonymous sublayer: {exc}")
        if stage is not None and root_before is not None:
            try:
                restore_root_runtime_metadata(
                    stage.GetRootLayer(),
                    root_before,
                )
                root_after = _snapshot_root(stage)
                report["root_layer_unchanged"] = root_after == root_before
                report["root_before"] = root_before
                report["root_after"] = root_after
                if root_after != root_before:
                    raise RuntimeError("frozen root layer changed")
            except Exception as exc:
                cleanup_errors.append(f"verify root layer: {exc}")

        post_hash_errors: list[str] = []
        for label, item in FROZEN_INPUTS.items():
            try:
                verify_sha256(
                    Path(item["path"]),
                    str(item["sha256"]),
                    label=label,
                )
            except Exception as exc:
                post_hash_errors.append(str(exc))
        if cleanup_errors or post_hash_errors:
            report["trial_classification"] = "FAIL_CLEANUP"
            report["native_export_status"] = "NOT_WRITTEN_FAIL_CLEANUP"
            if export_path.exists():
                try:
                    export_path.unlink()
                except Exception as exc:
                    cleanup_errors.append(
                        f"remove invalid native export: {exc}"
                    )
        report["cleanup"] = {
            "errors": cleanup_errors,
            "post_cleanup_hash_errors": post_hash_errors,
            "no_persistent_stage_write": not cleanup_errors
            and not post_hash_errors,
        }

    return _publish_pre_shutdown(
        report,
        report_path=report_path,
        telemetry_path=telemetry_path,
        export_path=export_path,
        simulation_app=simulation_app,
        verified_frozen_manifest=verified_frozen_manifest,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local Isaac Sim 5.1 GraspTester scripted-equivalent "
            "diagnostic. This is not GUI or task-pass evidence."
        )
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        variant = resolve_variant(args.variant)
        outputs = validate_output_paths(
            args.report,
            args.export,
            args.telemetry,
            args.artifact_dir,
        )
        verified = verify_all_frozen_inputs()
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    outputs["export"].unlink(missing_ok=True)
    report = _run_isaac(
        variant,
        outputs["export"],
        outputs["artifact_dir"],
        outputs["report"],
        outputs["telemetry"],
        verified,
    )
    return int(report["intended_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

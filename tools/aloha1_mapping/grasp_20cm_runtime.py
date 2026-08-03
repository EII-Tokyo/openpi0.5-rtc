"""Frozen-input helpers for the Isaac Sim 5.1 ALOHA 20 cm grasp runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import yaml

from tools.aloha1_mapping.grasp_20cm_controller import ACTIVE_PHASES
from tools.aloha1_mapping.grasp_20cm_controller import Grasp20cmController
from tools.aloha1_mapping.grasp_20cm_controller import Grasp20cmThresholds
from tools.aloha1_mapping.grasp_20cm_controller import Phase
from tools.aloha1_mapping.grasp_20cm_controller import RunObservation
from tools.aloha1_mapping.grasp_20cm_controller import TransitionRecord

EXPECTED_DOF_ORDER = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
]


class FrozenInputError(RuntimeError):
    """Raised before runtime mutation when a frozen input contract fails."""


class RuntimeBindings(Protocol):
    """Isaac-facing operations consumed by the one-step adapter."""

    def prepare_run(self) -> None:
        """Validate and prepare session-owned runtime state."""

    def read_observation(
        self,
        *,
        frame: int,
        time_s: float,
    ) -> RunObservation:
        """Read one post-physics-step observation."""

    def apply_phase_target(self, phase: Phase) -> None:
        """Apply only the target associated with the current phase."""

    def set_bottle_kinematic(self, *, enabled: bool) -> None:
        """Set up or release the session-owned Bottle500."""

    def finalize_run(self, phase: Phase, reason: str) -> None:
        """Persist terminal runtime evidence."""

    def reset_session(self) -> None:
        """Remove and recreate only session-owned state."""


class Grasp20cmRuntimeAdapter:
    """Advance the pure controller exactly once per physics callback."""

    def __init__(
        self,
        *,
        bindings: RuntimeBindings,
        thresholds: Grasp20cmThresholds | None = None,
    ) -> None:
        self.bindings = bindings
        self.controller = Grasp20cmController(thresholds)
        self.physics_step_count = 0
        self._elapsed_s = 0.0
        self._running = False

    @property
    def phase(self) -> Phase:
        return self.controller.phase

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> TransitionRecord:
        self.bindings.prepare_run()
        transition = self.controller.start()
        self._running = True
        return transition

    def on_physics_step(
        self,
        step_s: float,
    ) -> TransitionRecord | None:
        if not self._running:
            return None
        step_s = float(step_s)
        if not math.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("physics step must be finite and positive")
        self.physics_step_count += 1
        self._elapsed_s += step_s
        observation = self.bindings.read_observation(
            frame=self.physics_step_count,
            time_s=self._elapsed_s,
        )
        transition = self.controller.observe(observation)
        if transition.current is Phase.RELEASE_DYNAMIC:
            self.bindings.set_bottle_kinematic(enabled=False)
        if transition.current in ACTIVE_PHASES:
            self.bindings.apply_phase_target(transition.current)
        else:
            self._running = False
            self.bindings.finalize_run(
                transition.current,
                transition.reason,
            )
        return transition

    def abort(self) -> TransitionRecord:
        if not self._running:
            raise RuntimeError("cannot abort an inactive run")
        transition = self.controller.request_abort()
        self._running = False
        self.bindings.finalize_run(
            transition.current,
            transition.reason,
        )
        return transition

    def fail_due_to_exception(self, reason: str) -> TransitionRecord:
        """Stop target writes while a caller persists the exception report."""

        if not self._running:
            raise RuntimeError("cannot fail an inactive run")
        transition = self.controller.request_failure(reason)
        self._running = False
        return transition

    def reset(self) -> TransitionRecord:
        if self._running:
            raise RuntimeError("cannot reset an active run")
        self.bindings.reset_session()
        transition = self.controller.reset()
        self.physics_step_count = 0
        self._elapsed_s = 0.0
        return transition


def sha256_file(path: Path) -> str:
    """Hash one file without following any guessed alternative path."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    """Resolve and verify one exact frozen file."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FrozenInputError(f"missing frozen input: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise FrozenInputError(
            f"sha256 mismatch for {resolved}: "
            f"{actual} != {expected_sha256}"
        )
    return {"absolute_path": str(resolved), "sha256": actual}


def apply_verified_session_sublayers(
    *,
    stage: Any,
    records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Compose verified diagnostic layers only in the anonymous session layer."""

    session_layer = stage.GetSessionLayer()
    if session_layer is None:
        raise FrozenInputError("opened Stage has no session layer")
    before = [str(path) for path in session_layer.subLayerPaths]
    inserted: list[str] = []
    already_present: list[str] = []
    for record in reversed(list(records)):
        path = str(record["absolute_path"])
        if path in session_layer.subLayerPaths:
            already_present.append(path)
            continue
        session_layer.subLayerPaths.insert(0, path)
        inserted.append(path)
    return {
        "status": "PASS",
        "session_layer_identifier": str(
            getattr(session_layer, "identifier", "ANONYMOUS_SESSION_LAYER")
        ),
        "before": before,
        "after": [str(path) for path in session_layer.subLayerPaths],
        "inserted_paths": inserted,
        "already_present_paths": already_present,
        "root_layer_saved": False,
    }


def load_and_verify_config(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Load the diagnostic profile and verify every referenced source."""
    config_record = verify_frozen_file(
        config_path,
        sha256_file(config_path.resolve()),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise FrozenInputError("config root must be a mapping")
    if config.get("schema_version") != 1:
        raise FrozenInputError("unsupported config schema_version")
    if config.get("classification") != (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    ):
        raise FrozenInputError("unexpected diagnostic classification")

    records: dict[str, dict[str, str]] = {}
    records["stage"] = _verify_record(
        config.get("stage"),
        project_root=project_root,
        label="stage",
    )
    records["bottle"] = _verify_record(
        config.get("bottle"),
        project_root=project_root,
        label="bottle",
    )
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise FrozenInputError("frozen_inputs must be a mapping")
    for name, record in frozen.items():
        records[str(name)] = _verify_record(
            record,
            project_root=project_root,
            label=str(name),
        )

    session_sublayers: list[dict[str, str]] = []
    raw_session_sublayers = config.get("diagnostic_session_sublayers", [])
    if not isinstance(raw_session_sublayers, Sequence) or isinstance(
        raw_session_sublayers, str | bytes
    ):
        raise FrozenInputError(
            "diagnostic_session_sublayers must be a sequence"
        )
    for index, record in enumerate(raw_session_sublayers):
        session_sublayers.append(
            _verify_record(
                record,
                project_root=project_root,
                label=f"diagnostic_session_sublayers[{index}]",
            )
        )

    dof_order = config.get("robot", {}).get("dof_order")
    if dof_order != EXPECTED_DOF_ORDER:
        raise FrozenInputError(
            f"unexpected DOF order: {dof_order!r}"
        )
    if config.get("boundaries", {}).get("task8") != "NOT_RUN":
        raise FrozenInputError("Task 8 boundary must remain NOT_RUN")
    return {
        "config": dict(config),
        "config_path": config_record["absolute_path"],
        "config_sha256": config_record["sha256"],
        "frozen_inputs": records,
        "session_sublayers": session_sublayers,
    }


def apply_task8_collider_profile(
    profile: Mapping[str, Any],
    *,
    candidate_report_path: Path,
    profile_name: str,
) -> dict[str, Any]:
    """Select one frozen, non-promoted Task 8 collider profile.

    The returned mapping is a deep copy.  Only the Stage frozen-input record
    and the explicit Task 8 boundary are changed; the grasp configuration,
    session layers, physical parameters, and source files remain untouched.
    """

    if profile_name not in {"fidelity_profile", "throughput_profile"}:
        raise FrozenInputError(f"unsupported Task 8 collider profile: {profile_name}")
    report_record = verify_frozen_file(
        candidate_report_path,
        sha256_file(candidate_report_path.resolve()),
    )
    report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if report.get("classification") != "DIAGNOSTIC_ONLY_NOT_PROMOTED":
        raise FrozenInputError("unexpected Task 8 candidate classification")
    if report.get("candidate_promoted") is not False:
        raise FrozenInputError("Task 8 collider candidate must not be promoted")

    result = copy.deepcopy(dict(profile))
    source_record = result.get("frozen_inputs", {}).get("stage")
    manifest_source = report.get("source_stage")
    if not isinstance(source_record, Mapping) or not isinstance(
        manifest_source, Mapping
    ):
        raise FrozenInputError("Task 8 source Stage records are missing")
    if (
        str(Path(str(source_record.get("absolute_path"))).resolve())
        != str(Path(str(manifest_source.get("absolute_path"))).resolve())
        or source_record.get("sha256") != manifest_source.get("sha256")
    ):
        raise FrozenInputError("Task 8 candidate source Stage does not match runtime")

    layer_record = report.get("layers", {}).get(profile_name)
    if not isinstance(layer_record, Mapping):
        raise FrozenInputError(f"missing Task 8 layer record: {profile_name}")
    candidate = verify_frozen_file(
        Path(str(layer_record.get("absolute_path"))),
        str(layer_record.get("sha256")),
    )
    config = result.get("config")
    frozen_inputs = result.get("frozen_inputs")
    if not isinstance(config, dict) or not isinstance(frozen_inputs, dict):
        raise FrozenInputError("runtime profile is not mutable after deep copy")
    stage_config = config.get("stage")
    if not isinstance(stage_config, dict):
        raise FrozenInputError("runtime Stage config is missing")
    stage_config["path"] = candidate["absolute_path"]
    stage_config["sha256"] = candidate["sha256"]
    frozen_inputs["stage"] = candidate
    boundaries = config.get("boundaries")
    if not isinstance(boundaries, dict):
        raise FrozenInputError("runtime boundary record is missing")
    boundaries["task8"] = "AUTHORIZED_IN_PROGRESS"
    result["task8_diagnostic"] = {
        "profile_name": profile_name,
        "classification": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "candidate_promoted": False,
        "candidate_report": report_record,
        "source_stage": dict(manifest_source),
        "runtime_stage": candidate,
        "physical_parameters_changed": False,
        "final_or_default_asset_modified": False,
    }
    return result


def _verify_record(
    record: Any,
    *,
    project_root: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise FrozenInputError(f"{label} record must be a mapping")
    path_value = record.get("path")
    hash_value = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise FrozenInputError(
            f"{label} requires string path and sha256"
        )
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return verify_frozen_file(path, hash_value)


def validate_composed_stage(
    *,
    stage: Any,
    expected_root_prim: str,
    required_prims: Sequence[str],
) -> dict[str, Any]:
    """Validate required composed prims without editing the USD Stage."""
    root_prim = stage.GetPrimAtPath(expected_root_prim)
    if not root_prim.IsValid():
        raise FrozenInputError(
            f"missing expected root prim: {expected_root_prim}"
        )
    missing = [
        path
        for path in required_prims
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise FrozenInputError(f"missing required prims: {missing}")
    return {
        "root_prim": expected_root_prim,
        "sublayers": list(stage.GetRootLayer().subLayerPaths),
        "required_prims": list(required_prims),
    }

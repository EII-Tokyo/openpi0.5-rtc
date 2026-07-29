#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Probe ALOHA's embedded gripper against the local Isaac Sim 5.1 Grasp Editor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0"
    / "aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
EXTENSION_ID = "isaacsim.robot_setup.grasp_editor"
EXPECTED_EXTENSION_VERSION = "2.0.20"
OFFICIAL_MCP_INDEX_VERSION = "2.2.0"
ARTICULATION_PATH = "/World/follower_left/vx300s_left/root_joint"
GRIPPER_FRAME_PATH = (
    "/World/follower_left/vx300s_left/follower_left_gripper_link"
)
LEFT_FINGER_PATH = (
    "/World/follower_left/vx300s_left/follower_left_left_finger_link"
)
RIGHT_FINGER_PATH = (
    "/World/follower_left/vx300s_left/follower_left_right_finger_link"
)
BOTTLE_SESSION_PATH = "/World/ALOHA1GraspEditorSession/Bottle500"
EXPECTED_DOF_NAMES = [
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
ARM_JOINT_NAMES = EXPECTED_DOF_NAMES[:6]
ACTIVE_JOINT_NAMES = ["left_finger", "right_finger"]
OPEN_POSITIONS_M = [0.057, -0.057]
CLOSED_POSITIONS_M = [0.021, -0.021]
CLOSE_SPEEDS_M_S = [0.02, 0.02]
DEFAULT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json"
)
VALID_CLASSIFICATIONS = {
    "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED",
    "REQUIRES_DIAGNOSTIC_GRIPPER_ONLY",
    "INCOMPATIBLE",
    "INCONCLUSIVE",
}
_LIVE_APPS: list[Any] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def classify_structural_api_compatibility(
    *,
    extension_version: str,
    dof_names: list[str],
    active_joint_names: list[str],
    structural_setup_arm_joint_mutation: bool,
    synthetic_serializer_parse_pass: bool,
    stage_immutable: bool,
) -> str:
    """Classify only the exact local structural API contract."""
    if (
        extension_version != EXPECTED_EXTENSION_VERSION
        or dof_names != EXPECTED_DOF_NAMES
        or active_joint_names != ACTIVE_JOINT_NAMES
        or not synthetic_serializer_parse_pass
        or not stage_immutable
    ):
        return "INCOMPATIBLE"
    if structural_setup_arm_joint_mutation:
        return "REQUIRES_DIAGNOSTIC_GRIPPER_ONLY"
    return "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"


def _extension_version(manager: Any) -> tuple[str | None, str | None]:
    enabled_id = manager.get_enabled_extension_id(EXTENSION_ID)
    if not enabled_id:
        return None, None
    extension = manager.get_extension_dict(enabled_id)
    version = None
    if extension:
        version = extension.get("package", {}).get("version")
    return str(enabled_id), str(version) if version is not None else None


def _dof_properties(articulation: Any) -> list[dict[str, Any]]:
    properties = articulation.dof_properties
    names = list(properties.dtype.names or ())
    result: list[dict[str, Any]] = []
    for index, dof_name in enumerate(articulation.dof_names):
        record = {
            "index": index,
            "name": str(dof_name),
            "properties": {
                field: _json_value(properties[index][field])
                for field in names
            },
        }
        result.append(record)
    return result


def _probe_synthetic_serializer(
    *,
    grasp_test_results_type: Any,
    data_writer_type: Any,
    settings: Any,
) -> dict[str, Any]:
    import numpy as np

    result = grasp_test_results_type(
        settings,
        np.asarray([0.069, 0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray(CLOSED_POSITIONS_M, dtype=np.float64),
        1.0,
        False,  # noqa: FBT003 - synthetic data is not a passing grasp.
    )
    with tempfile.TemporaryDirectory(
        prefix="aloha1-grasp-editor-roundtrip-"
    ) as directory:
        first_path = Path(directory) / "first.yaml"
        second_path = Path(directory) / "second.yaml"
        first = data_writer_type(GRIPPER_FRAME_PATH, BOTTLE_SESSION_PATH)
        second = data_writer_type(GRIPPER_FRAME_PATH, BOTTLE_SESSION_PATH)
        first.write_grasp_to_file(result, 0.0, str(first_path))
        second.write_grasp_to_file(result, 0.0, str(second_path))
        importer_a = data_writer_type(
            GRIPPER_FRAME_PATH,
            BOTTLE_SESSION_PATH,
        )
        importer_b = data_writer_type(
            GRIPPER_FRAME_PATH,
            BOTTLE_SESSION_PATH,
        )
        error_a = importer_a.import_grasps_from_file(str(first_path))
        error_b = importer_b.import_grasps_from_file(str(second_path))
        bytes_equal = first_path.read_bytes() == second_path.read_bytes()
        semantic_equal = (
            error_a == ""
            and error_b == ""
            and _json_value(importer_a.data["grasps"])
            == _json_value(importer_b.data["grasps"])
        )
        return {
            "synthetic": True,
            "uses_grasp_tester_output": False,
            "exercises_gui_import_remap": False,
            "confidence": 0.0,
            "format": "isaac_grasp",
            "format_version": 1.0,
            "first_sha256": _sha256(first_path),
            "second_sha256": _sha256(second_path),
            "bytes_equal": bytes_equal,
            "semantic_equal": semantic_equal,
            "import_error_first": error_a,
            "import_error_second": error_b,
            "pass": bytes_equal and semantic_equal,
        }


def _runtime_probe(stage_path: Path) -> dict[str, Any]:
    import isaacsim
    import numpy as np

    app = isaacsim.SimulationApp({"headless": True})
    _LIVE_APPS.append(app)
    try:
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.api.articulations import ArticulationSubset
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import open_stage
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(
            EXTENSION_ID,
            True,  # noqa: FBT003 - local Kit binding is positional-only.
        )
        app.update()
        enabled_id, extension_version = _extension_version(manager)

        from isaacsim.robot_setup.grasp_editor.data_writer import DataWriter
        from isaacsim.robot_setup.grasp_editor.grasp_tester import GraspTestResults
        from isaacsim.robot_setup.grasp_editor.grasp_tester import GraspTestSettings

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open frozen Stage: {stage_path}")
        app.update()

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        required_prim_paths = [
            "/World",
            ARTICULATION_PATH,
            GRIPPER_FRAME_PATH,
            LEFT_FINGER_PATH,
            RIGHT_FINGER_PATH,
        ]
        prim_readback = {
            path: bool(stage.GetPrimAtPath(path).IsValid())
            for path in required_prim_paths
        }
        if not all(prim_readback.values()):
            raise RuntimeError(f"required prim check failed: {prim_readback}")

        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="aloha1_grasp_editor_compatibility_follower_left",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        world.reset()
        app.update()

        dof_names = list(articulation.dof_names)
        before = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        ).copy()
        subset = ArticulationSubset(articulation, ACTIVE_JOINT_NAMES)
        subset_indices = [
            int(value) for value in subset.get_joint_subset_indices()
        ]
        inactive_positions = [
            float(before[index])
            for index, name in enumerate(dof_names)
            if name not in ACTIVE_JOINT_NAMES
        ]
        settings = GraspTestSettings(
            ARTICULATION_PATH,
            GRIPPER_FRAME_PATH,
            list(ACTIVE_JOINT_NAMES),
            list(OPEN_POSITIONS_M),
            list(CLOSED_POSITIONS_M),
            list(CLOSE_SPEEDS_M_S),
            inactive_positions,
            BOTTLE_SESSION_PATH,
            BOTTLE_SESSION_PATH,
            0.0,
            0.0,
        )
        serializer_probe = _probe_synthetic_serializer(
            grasp_test_results_type=GraspTestResults,
            data_writer_type=DataWriter,
            settings=settings,
        )
        after = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        ).copy()
        arm_indices = [dof_names.index(name) for name in ARM_JOINT_NAMES]
        arm_delta = after[arm_indices] - before[arm_indices]
        structural_setup_arm_joint_mutation = not np.array_equal(
            after[arm_indices],
            before[arm_indices],
        )
        app.update()
        after_passive_physics_update = np.asarray(
            articulation.get_joint_positions(),
            dtype=np.float64,
        ).copy()
        passive_physics_arm_delta = (
            after_passive_physics_update[arm_indices] - after[arm_indices]
        )
        source_readback = {
            "GraspTestSettings_active_joints": list(settings.active_joints),
            "GraspTestSettings_open_positions": [
                float(value)
                for value in settings.active_joint_open_positions
            ],
            "GraspTestSettings_closed_positions": [
                float(value)
                for value in settings.active_joint_closed_positions
            ],
            "GraspTestSettings_close_velocities": [
                float(value)
                for value in settings.active_joint_close_velocities
            ],
            "DataWriter_format": "isaac_grasp",
            "DataWriter_format_version": 1.0,
        }
        return {
            "extension": {
                "requested_id": EXTENSION_ID,
                "enabled_id": enabled_id,
                "runtime_version": extension_version,
                "expected_local_version": EXPECTED_EXTENSION_VERSION,
                "official_mcp_index_version": OFFICIAL_MCP_INDEX_VERSION,
                "implementation_authority": (
                    "LOCAL_ISAAC_SIM_5_1_SOURCE_AND_RUNTIME"
                ),
                "mcp_version_mismatch": (
                    OFFICIAL_MCP_INDEX_VERSION != EXPECTED_EXTENSION_VERSION
                ),
            },
            "runtime": {
                "isaac_sim": importlib.metadata.version("isaacsim"),
                "kit": str(
                    carb.tokens.get_tokens_interface().resolve(
                        "${kit_version}"
                    )
                ).split("+", maxsplit=1)[0],
            },
            "stage": {
                "required_prims": prim_readback,
                "root_prim": "/World",
                "sublayers": list(stage.GetRootLayer().subLayerPaths),
            },
            "articulation": {
                "path": ARTICULATION_PATH,
                "dof_names": dof_names,
                "dof_count": len(dof_names),
                "dof_properties": _dof_properties(articulation),
                "active_joint_names": list(subset.joint_names),
                "active_joint_indices": subset_indices,
                "arm_joint_names": list(ARM_JOINT_NAMES),
                "arm_joint_indices": arm_indices,
                "structural_setup_arm_joint_mutation": (
                    structural_setup_arm_joint_mutation
                ),
                "arm_joint_delta": [
                    float(value) for value in arm_delta
                ],
                "mutation_measurement_boundary": (
                    "IMMEDIATE_BEFORE_AFTER_SUBSET_SETTINGS_AND_YAML"
                ),
                "passive_physics_update_arm_delta": [
                    float(value) for value in passive_physics_arm_delta
                ],
                "passive_physics_update_excluded_from_editor_mutation": True,
            },
            "local_source_contract": source_readback,
            "synthetic_serializer_parse_probe": serializer_probe,
            "structural_api_runtime_pass": (
                extension_version == EXPECTED_EXTENSION_VERSION
                and dof_names == EXPECTED_DOF_NAMES
                and list(subset.joint_names) == ACTIVE_JOINT_NAMES
                and not structural_setup_arm_joint_mutation
                and serializer_probe["pass"]
            ),
        }
    finally:
        # Persist the report and frozen-Stage hash before shutting Kit down.
        # SimulationApp.close() may terminate this local process immediately.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_path = args.stage.resolve()
    report_path = args.report.resolve()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "scope": (
            "ALOHA1 follower-left Grasp Editor structural API and "
            "synthetic serializer probe"
        ),
        "evidence_scope": (
            "LOCAL_SOURCE_STRUCTURAL_API_AND_SYNTHETIC_SERIALIZER_ONLY"
        ),
        "status": "PARTIAL",
        "classification": "INCONCLUSIVE",
        "structural_api_classification": "INCONCLUSIVE",
        "structural_api_probe_status": "NOT_RUN",
        "grasp_tester_execution_status": "NOT_RUN",
        "timeline_physics_execution_status": "NOT_RUN",
        "session_bottle500_composition_status": "NOT_RUN",
        "arm_hold_during_grasp_test_status": "NOT_RUN",
        "actual_isaac_grasp_export_status": "NOT_RUN",
        "mimic_commandability_status": "GUI_PENDING",
        "ik_execution_status": "NOT_RUN",
        "gui_evidence_status": "GUI_PENDING",
        "must_not_claim": [
            "GUI_COMPLETED",
            "GRASP_TESTER_PASSED",
            "BOTTLE500_PHYSICS_TESTED",
            "ARM_HOLD_DURING_GRASP_TEST_PASSED",
            "ACTUAL_ISAAC_GRASP_EXPORTED",
        ],
        "task8": "NOT_RUN",
        "source_stage": {
            "path": str(stage_path),
            "expected_sha256": EXPECTED_STAGE_SHA256,
            "sha256_before": None,
            "sha256_after": None,
            "immutable": False,
        },
        "error": None,
    }
    exit_code = 1
    try:
        before_hash = _sha256(stage_path)
        payload["source_stage"]["sha256_before"] = before_hash
        if before_hash != EXPECTED_STAGE_SHA256:
            raise RuntimeError(
                "frozen Stage hash mismatch: "
                f"expected={EXPECTED_STAGE_SHA256}, actual={before_hash}"
            )
        runtime = _runtime_probe(stage_path)
        payload["probe"] = runtime
        after_hash = _sha256(stage_path)
        payload["source_stage"]["sha256_after"] = after_hash
        immutable = before_hash == after_hash
        payload["source_stage"]["immutable"] = immutable
        classification = classify_structural_api_compatibility(
            extension_version=runtime["extension"]["runtime_version"],
            dof_names=runtime["articulation"]["dof_names"],
            active_joint_names=runtime["articulation"][
                "active_joint_names"
            ],
            structural_setup_arm_joint_mutation=runtime["articulation"][
                "structural_setup_arm_joint_mutation"
            ],
            synthetic_serializer_parse_pass=runtime[
                "synthetic_serializer_parse_probe"
            ]["pass"],
            stage_immutable=immutable,
        )
        if classification not in VALID_CLASSIFICATIONS:
            raise RuntimeError(f"invalid classification: {classification}")
        payload["structural_api_classification"] = classification
        # Full GraspTester physics execution and GUI selection have not run.
        # Keep the overall behavioral classification fail-closed.
        payload["classification"] = "INCONCLUSIVE"
        payload["structural_api_probe_status"] = (
            "PASS"
            if classification
            == "FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED"
            else "PARTIAL"
        )
        payload["status"] = "PARTIAL"
        exit_code = (
            0
            if payload["structural_api_probe_status"] == "PASS"
            else 2
        )
    except Exception as error:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        payload["structural_api_probe_status"] = "FAIL"
        if stage_path.is_file():
            payload["source_stage"]["sha256_after"] = _sha256(stage_path)
            payload["source_stage"]["immutable"] = (
                payload["source_stage"]["sha256_before"]
                == payload["source_stage"]["sha256_after"]
            )
    _atomic_json(report_path, payload)
    print(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if _LIVE_APPS:
        _LIVE_APPS[-1].close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

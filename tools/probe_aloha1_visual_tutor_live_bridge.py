#!/usr/bin/env python3
"""Run the queued, read-only ALOHA Visual Tutor live-bridge hard gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/aloha1_grasp_editor_live_manifest.yaml"
APPROVED_MANIFEST_SHA256 = (
    "fecacb461c43e299e0ec1209ffde5bd8e9826ac93fd3defcdc80bfb4405e93ba"
)
VISUAL_TUTOR_EXTENSION_ID = "my.isaac.visual_tutor"
VISUAL_TUTOR_EXTENSION_PARENT = ROOT / "visual_tutor/isaac_extensions"
VISUAL_TUTOR_EXTENSION_DIR = (
    VISUAL_TUTOR_EXTENSION_PARENT / VISUAL_TUTOR_EXTENSION_ID
)
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence"
    / "live_bridge"
)
REQUIRED_CAPTURE_PRIMS = (
    "/World",
    "/World/follower_left/vx300s_left/root_joint",
    "/World/follower_left/vx300s_left/follower_left_gripper_link",
)
MAX_ACK_UPDATE_POLLS = 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=(1, 2), required=True)
    return parser.parse_args()


def _load_manifest() -> Any:
    visual_tutor_root = ROOT / "visual_tutor"
    sys.path.insert(0, str(visual_tutor_root))
    from my_visual_tutor.grasp_editor_manifest import load_approved_manifest

    return load_approved_manifest(MANIFEST_PATH)


def _preflight_report_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite live bridge report: {path}")


def _write_report_before_close(
    path: Path,
    payload: dict[str, Any],
) -> None:
    _preflight_report_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _enabled_extension_version(
    extension_manager: Any,
    enabled_id: str,
) -> str:
    details = extension_manager.get_extension_dict(enabled_id)
    return str(details.get("package", {}).get("version", "")).split(
        "+",
        maxsplit=1,
    )[0]


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _normalize_stage_path(identifier: str) -> Path:
    parsed = urlparse(identifier)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        raise RuntimeError(f"Stage identifier is not a local file: {identifier}")
    return Path(identifier).resolve()


def _serialized_sha256(layer: Any) -> str:
    serialized = layer.ExportToString()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _layer_state(layer: Any) -> dict[str, Any]:
    return {
        "identifier": str(layer.identifier),
        "real_path": str(layer.realPath),
        "resolved_path": str(layer.resolvedPath),
        "dirty": bool(layer.dirty),
        "sublayers": list(layer.subLayerPaths),
        "serialized_sha256": _serialized_sha256(layer),
    }


def _used_layer_closure(stage: Any) -> list[dict[str, Any]]:
    layers = sorted(
        stage.GetUsedLayers(),
        key=lambda layer: str(layer.identifier),
    )
    return [_layer_state(layer) for layer in layers]


def _composed_reference_inventory(stage: Any) -> list[dict[str, str]]:
    inventory = [
        {
            "prim_path": str(prim.GetPath()),
            "references": str(prim.GetMetadata("references")),
        }
        for prim in stage.Traverse()
        if prim.HasMetadata("references")
    ]
    return sorted(inventory, key=lambda item: item["prim_path"])


def _capture_runtime_baseline(
    *,
    context: Any,
    stage: Any,
) -> dict[str, Any]:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    root_layer = stage.GetRootLayer()
    session_layer = stage.GetSessionLayer()
    edit_target = stage.GetEditTarget().GetLayer()
    default_prim = stage.GetDefaultPrim()
    composed_reference_inventory = _composed_reference_inventory(stage)
    used_layer_closure = _used_layer_closure(stage)
    return {
        "stage_identifier": str(context.get_stage_url()),
        "root_layer": _layer_state(root_layer),
        "session_layer": _layer_state(session_layer),
        "edit_target_identifier": str(edit_target.identifier),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_current_time": float(timeline.get_current_time()),
        "default_prim_path": (
            str(default_prim.GetPath()) if default_prim.IsValid() else None
        ),
        "required_prims": {
            path: bool(stage.GetPrimAtPath(path).IsValid())
            for path in REQUIRED_CAPTURE_PRIMS
        },
        "used_layer_closure": used_layer_closure,
        "composed_reference_inventory": composed_reference_inventory,
    }


def _await_exact_ack(
    *,
    simulation_app: Any,
    bridge: Any,
    run_id: str,
    request_sequence: int,
) -> dict[str, Any]:
    for _ in range(MAX_ACK_UPDATE_POLLS):
        simulation_app.update()
        ack = bridge.get_ack(run_id, request_sequence)
        if ack is not None:
            _assert_equal("ack run_id", ack["run_id"], run_id)
            _assert_equal(
                "ack request sequence",
                ack["request_sequence"],
                request_sequence,
            )
            return ack
    raise RuntimeError(
        "Timed out waiting for exact queued capture acknowledgement: "
        f"run_id={run_id!r}, request_sequence={request_sequence}"
    )


def _disable_visual_tutor_extension() -> dict[str, Any]:
    import omni.kit.app

    extension_manager = (
        omni.kit.app.get_app().get_extension_manager()
    )
    was_enabled = bool(
        extension_manager.is_extension_enabled(VISUAL_TUTOR_EXTENSION_ID)
    )
    if not was_enabled:
        return {
            "status": "PASS_ALREADY_DISABLED",
            "was_enabled": False,
            "disable_result": None,
            "extension_disabled": True,
        }
    disable_result = extension_manager.set_extension_enabled_immediate(
        VISUAL_TUTOR_EXTENSION_ID,
        False,  # noqa: FBT003 - local Kit binding is positional-only.
    )
    extension_disabled = not extension_manager.is_extension_enabled(
        VISUAL_TUTOR_EXTENSION_ID
    )
    status = (
        "PASS"
        if disable_result is True and extension_disabled
        else "FAIL_DISABLE"
    )
    return {
        "status": status,
        "was_enabled": True,
        "disable_result": disable_result,
        "extension_disabled": extension_disabled,
    }


def _probe_live_bridge(
    *,
    simulation_app: Any,
    manifest: Any,
    manifest_sha256: str,
    stage_sha256_before: str,
    bottle_sha256_before: str,
    attempt: int,
) -> dict[str, Any]:
    import carb
    from isaacsim.core.utils.stage import open_stage
    import omni.kit.app
    import omni.usd

    if not open_stage(str(manifest.stage_path)):
        raise RuntimeError(f"Failed to open approved Stage: {manifest.stage_path}")
    for _ in range(3):
        simulation_app.update()

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("USD context returned no Stage after open_stage")
    runtime_baseline_before_enable = _capture_runtime_baseline(
        context=context,
        stage=stage,
    )
    if not all(runtime_baseline_before_enable["required_prims"].values()):
        raise RuntimeError(
            "Required composed prim check failed: "
            f"{runtime_baseline_before_enable['required_prims']}"
        )
    if runtime_baseline_before_enable["timeline_playing"] is not False:
        raise RuntimeError("Timeline is playing before extension enable")
    _assert_equal(
        "Stage identifier before enable",
        _normalize_stage_path(
            str(runtime_baseline_before_enable["stage_identifier"])
        ),
        manifest.stage_path.resolve(),
    )
    root_identifier = runtime_baseline_before_enable["root_layer"]["identifier"]
    _assert_equal(
        "root layer identifier before enable",
        _normalize_stage_path(str(root_identifier)),
        manifest.stage_path.resolve(),
    )

    extension_manager = (
        omni.kit.app.get_app().get_extension_manager()
    )
    if extension_manager.is_extension_enabled(VISUAL_TUTOR_EXTENSION_ID):
        raise RuntimeError("Visual Tutor extension was enabled before the gate")
    if extension_manager.is_extension_enabled(
        manifest.isaac.grasp_editor_extension
    ):
        raise RuntimeError("Grasp Editor must remain disabled during Task 2")

    extension_parent = VISUAL_TUTOR_EXTENSION_PARENT
    extension_manager.add_path(str(extension_parent))
    enable_result = extension_manager.set_extension_enabled_immediate(
        VISUAL_TUTOR_EXTENSION_ID,
        True,  # noqa: FBT003 - local Kit binding is positional-only.
    )
    if enable_result is not True:
        raise RuntimeError(
            f"Visual Tutor enable did not return True: {enable_result!r}"
        )
    if not extension_manager.is_extension_enabled(VISUAL_TUTOR_EXTENSION_ID):
        raise RuntimeError("Visual Tutor is not enabled after enable request")
    enabled_id = extension_manager.get_enabled_extension_id(
        VISUAL_TUTOR_EXTENSION_ID
    )
    if not enabled_id:
        raise RuntimeError("Visual Tutor enabled extension ID is empty")
    approved_extension_dir = VISUAL_TUTOR_EXTENSION_DIR.resolve()
    enabled_extension_dir = Path(
        extension_manager.get_extension_path(enabled_id)
    ).resolve()
    _assert_equal(
        "Visual Tutor extension path",
        enabled_extension_dir,
        approved_extension_dir,
    )

    physx_enabled_id = extension_manager.get_enabled_extension_id("omni.physx")
    if not physx_enabled_id:
        raise RuntimeError("PhysX enabled extension ID is empty")
    versions = {
        "isaac_sim": importlib.metadata.version("isaacsim"),
        "kit": carb.tokens.get_tokens_interface()
        .resolve("${kit_version}")
        .split("+", maxsplit=1)[0],
        "physx": _enabled_extension_version(
            extension_manager,
            physx_enabled_id,
        ),
        "visual_tutor": _enabled_extension_version(
            extension_manager,
            enabled_id,
        ),
        "grasp_editor_static_pin": manifest.isaac.grasp_editor_version,
        "grasp_editor_runtime": "NOT_RUN_TASK2_EXTENSION_DISABLED",
    }
    for name, expected in {
        "isaac_sim": manifest.isaac.version,
        "kit": manifest.isaac.kit,
        "physx": manifest.isaac.physx,
    }.items():
        _assert_equal(f"{name} version", versions[name], expected)

    bridge_module = importlib.import_module("my.isaac.visual_tutor")
    bridge = bridge_module.get_live_bridge()
    if bridge is None:
        raise RuntimeError("Live Visual Tutor bridge singleton is unavailable")

    run_id = f"attempt-{attempt}-capture"
    first_request = bridge.request_capture_state(run_id, manifest_sha256)
    first_sequence = int(first_request["request_sequence"])
    if bridge.get_ack(run_id, first_sequence) is not None:
        raise RuntimeError("Stale acknowledgement matched the first request")
    first_ack = _await_exact_ack(
        simulation_app=simulation_app,
        bridge=bridge,
        run_id=run_id,
        request_sequence=first_sequence,
    )

    second_request = bridge.request_capture_state(run_id, manifest_sha256)
    second_sequence = int(second_request["request_sequence"])
    if second_sequence <= first_sequence:
        raise RuntimeError("Request sequence did not advance")
    if bridge.get_ack(run_id, second_sequence) is not None:
        raise RuntimeError("Stale acknowledgement matched the second request")
    second_ack = _await_exact_ack(
        simulation_app=simulation_app,
        bridge=bridge,
        run_id=run_id,
        request_sequence=second_sequence,
    )

    main_thread_ident = threading.main_thread().ident
    for label, ack in (("first", first_ack), ("second", second_ack)):
        _assert_equal(f"{label} ack status", ack["status"], "PASS")
        _assert_equal(
            f"{label} ack manifest SHA",
            ack["expected_manifest_sha"],
            manifest_sha256,
        )
        _assert_equal(
            f"{label} callback thread",
            ack["callback_thread_ident"],
            main_thread_ident,
        )
        if ack["fingerprints_unchanged"] is not True:
            raise RuntimeError(f"{label} ack changed a Stage fingerprint")
        if float(ack["completed_monotonic"]) <= float(
            ack["requested_monotonic"]
        ):
            raise RuntimeError(f"{label} ack completion is not fresh")
    if int(second_ack["update_number"]) <= int(first_ack["update_number"]):
        raise RuntimeError("Acknowledgement update number did not advance")
    if float(second_ack["heartbeat_monotonic"]) <= float(
        first_ack["heartbeat_monotonic"]
    ):
        raise RuntimeError("Acknowledgement heartbeat did not advance")

    runtime_baseline_after_ack = _capture_runtime_baseline(
        context=context,
        stage=stage,
    )
    _assert_equal(
        "complete runtime baseline after acknowledgements",
        runtime_baseline_after_ack,
        runtime_baseline_before_enable,
    )

    cleanup = _disable_visual_tutor_extension()
    _assert_equal("Visual Tutor cleanup", cleanup["status"], "PASS")
    if bridge_module.get_live_bridge() is not None:
        raise RuntimeError("Visual Tutor singleton remains after disable")
    runtime_baseline_after_cleanup = _capture_runtime_baseline(
        context=context,
        stage=stage,
    )
    _assert_equal(
        "complete runtime baseline after cleanup",
        runtime_baseline_after_cleanup,
        runtime_baseline_before_enable,
    )
    if extension_manager.is_extension_enabled(
        manifest.isaac.grasp_editor_extension
    ):
        raise RuntimeError("Task 2 unexpectedly enabled Grasp Editor")

    stage_path_after = manifest.verify_stage()
    bottle_path_after = manifest.verify_bottle()
    _assert_equal("Stage path after capture", stage_path_after, manifest.stage_path)
    _assert_equal(
        "Bottle path after capture",
        bottle_path_after,
        manifest.bottle_usd_path,
    )
    from my_visual_tutor.grasp_editor_manifest import sha256_file

    stage_sha256_after = sha256_file(manifest.stage_path)
    bottle_sha256_after = sha256_file(manifest.bottle_usd_path)
    _assert_equal(
        "Stage SHA-256 after capture",
        stage_sha256_after,
        stage_sha256_before,
    )
    _assert_equal(
        "Bottle SHA-256 after capture",
        bottle_sha256_after,
        bottle_sha256_before,
    )
    return {
        "status": "PASS",
        "classification": "LOCAL_VISUAL_TUTOR_LIVE_BRIDGE_PASS",
        "versions": versions,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": manifest_sha256,
        "stage_path": str(manifest.stage_path),
        "stage_sha256_before": stage_sha256_before,
        "stage_sha256_after": stage_sha256_after,
        "bottle_path": str(manifest.bottle_usd_path),
        "bottle_sha256_before": bottle_sha256_before,
        "bottle_sha256_after": bottle_sha256_after,
        "extension_identity": {
            "parent_registered": str(extension_parent),
            "enabled_id": enabled_id,
            "approved_extension_dir": str(approved_extension_dir),
            "enabled_extension_dir": str(enabled_extension_dir),
        },
        "runtime_baseline_before_enable": runtime_baseline_before_enable,
        "runtime_baseline_after_ack": runtime_baseline_after_ack,
        "runtime_baseline_after_cleanup": runtime_baseline_after_cleanup,
        "first_request": first_request,
        "first_ack": first_ack,
        "second_request": second_request,
        "second_ack": second_ack,
        "cleanup": cleanup,
        "main_thread_ident": main_thread_ident,
        "report_written_before_close": True,
        "grasp_editor": "NOT_RUN_TASK2_EXTENSION_DISABLED",
        "ik": "NOT_RUN",
        "task8": "NOT_RUN",
    }


def main() -> int:
    args = _parse_args()
    manifest = _load_manifest()
    stage_path = manifest.verify_stage()
    bottle_path = manifest.verify_bottle()
    from my_visual_tutor.grasp_editor_manifest import sha256_file

    manifest_sha256 = sha256_file(MANIFEST_PATH)
    _assert_equal(
        "approved manifest SHA-256",
        manifest_sha256,
        APPROVED_MANIFEST_SHA256,
    )
    stage_sha256_before = sha256_file(stage_path)
    bottle_sha256_before = sha256_file(bottle_path)
    attempt_root = ARTIFACT_ROOT / f"attempt_{args.attempt}"
    report_path = attempt_root / "live_bridge_report.json"
    _preflight_report_path(report_path)

    simulation_app = None
    report_written = False
    exit_code = 1
    try:
        import isaacsim

        simulation_app = isaacsim.SimulationApp({"headless": False})
        payload = _probe_live_bridge(
            simulation_app=simulation_app,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            stage_sha256_before=stage_sha256_before,
            bottle_sha256_before=bottle_sha256_before,
            attempt=args.attempt,
        )
        payload["attempt"] = args.attempt
        _write_report_before_close(report_path, payload)
        report_written = True
        exit_code = 0
    except Exception as error:
        cleanup: dict[str, Any]
        if simulation_app is None:
            cleanup = {
                "status": "NOT_REQUIRED_APP_NOT_CREATED",
                "extension_disabled": True,
            }
        else:
            try:
                cleanup = _disable_visual_tutor_extension()
            except Exception as cleanup_error:
                cleanup = {
                    "status": "FAIL_CLEANUP_EXCEPTION",
                    "exception_type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "extension_disabled": False,
                }
        payload = {
            "status": "FAIL",
            "classification": "LOCAL_VISUAL_TUTOR_LIVE_BRIDGE_FAIL",
            "attempt": args.attempt,
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": manifest_sha256,
            "stage_path": str(manifest.stage_path),
            "stage_sha256_before": stage_sha256_before,
            "bottle_path": str(manifest.bottle_usd_path),
            "bottle_sha256_before": bottle_sha256_before,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=12),
            "cleanup": cleanup,
            "report_written_before_close": True,
            "grasp_editor": "NOT_RUN_TASK2_EXTENSION_DISABLED",
            "ik": "NOT_RUN",
            "task8": "NOT_RUN",
        }
        _write_report_before_close(report_path, payload)
        report_written = True
    finally:
        if simulation_app is not None:
            simulation_app.close()

    if not report_written:
        raise RuntimeError("Live bridge report was not written before close")
    print(
        json.dumps(
            {
                "status": "PASS" if exit_code == 0 else "FAIL",
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

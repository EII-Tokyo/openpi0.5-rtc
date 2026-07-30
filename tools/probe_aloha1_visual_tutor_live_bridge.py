#!/usr/bin/env python3
"""Run the read-only ALOHA Grasp Editor live-bridge hard gate."""

from __future__ import annotations

import argparse
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
VISUAL_TUTOR_EXTENSION_ID = "my.isaac.visual_tutor"
VISUAL_TUTOR_EXTENSION_PARENT = ROOT / "visual_tutor/isaac_extensions"
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=(1, 2), required=True)
    return parser.parse_args()


def _load_manifest() -> Any:
    visual_tutor_root = ROOT / "visual_tutor"
    sys.path.insert(0, str(visual_tutor_root))
    from my_visual_tutor.grasp_editor_manifest import load_approved_manifest

    return load_approved_manifest(MANIFEST_PATH)


def _write_report_before_close(
    path: Path,
    payload: dict[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite live bridge report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _extension_version(
    extension_manager: Any,
    module_name: str,
) -> str:
    extension_id = extension_manager.get_extension_id_by_module(module_name)
    if not extension_id:
        raise RuntimeError(f"Extension is not registered: {module_name}")
    details = extension_manager.get_extension_dict(extension_id)
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


def _probe_live_bridge(
    *,
    simulation_app: Any,
    manifest: Any,
    stage_sha256_before: str,
    bottle_sha256_before: str,
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
    prim_readback = {
        path: bool(stage.GetPrimAtPath(path).IsValid())
        for path in REQUIRED_CAPTURE_PRIMS
    }
    if not all(prim_readback.values()):
        raise RuntimeError(f"Required composed prim check failed: {prim_readback}")

    extension_manager = (
        omni.kit.app.get_app().get_extension_manager()
    )
    extension_parent = VISUAL_TUTOR_EXTENSION_PARENT
    extension_manager.add_path(str(extension_parent))
    extension_manager.set_extension_enabled_immediate(
        VISUAL_TUTOR_EXTENSION_ID,
        True,  # noqa: FBT003 - local Kit binding is positional-only.
    )
    simulation_app.update()

    bridge_module = importlib.import_module("my.isaac.visual_tutor")
    bridge = bridge_module.get_live_bridge()
    if bridge is None:
        raise RuntimeError("Live Visual Tutor bridge singleton is unavailable")

    first_capture = bridge.capture_state()
    first_heartbeat_number = int(first_capture["heartbeat_update_number"])
    first_heartbeat_monotonic = float(first_capture["heartbeat_monotonic"])
    for _ in range(3):
        simulation_app.update()
    capture = bridge.capture_state()

    main_thread_ident = threading.main_thread().ident
    _assert_equal(
        "capture status",
        capture["status"],
        "PASS",
    )
    _assert_equal(
        "capture callback thread",
        capture["capture_thread_ident"],
        main_thread_ident,
    )
    _assert_equal(
        "heartbeat callback thread",
        capture["heartbeat_thread_ident"],
        main_thread_ident,
    )
    if int(capture["heartbeat_update_number"]) <= first_heartbeat_number:
        raise RuntimeError("Kit heartbeat update number did not advance")
    if float(capture["heartbeat_monotonic"]) <= first_heartbeat_monotonic:
        raise RuntimeError("Kit heartbeat monotonic timestamp did not advance")
    _assert_equal(
        "Stage identifier",
        _normalize_stage_path(str(capture["stage_identifier"])),
        manifest.stage_path.resolve(),
    )
    _assert_equal(
        "root layer identifier",
        _normalize_stage_path(str(capture["root_layer_identifier"])),
        manifest.stage_path.resolve(),
    )
    session_identifier = str(capture["session_layer_identifier"])
    if not session_identifier.startswith("anon:"):
        raise RuntimeError(
            f"Session layer is not anonymous: {session_identifier!r}"
        )
    _assert_equal(
        "edit target identifier",
        capture["edit_target_identifier"],
        capture["root_layer_identifier"],
    )
    _assert_equal("default prim", capture["default_prim_path"], "/World")
    if capture["timeline_playing"] is not False:
        raise RuntimeError("Timeline unexpectedly entered playing state")
    if capture["timeline_stopped"] is not True:
        raise RuntimeError("Timeline is not stopped")
    if capture["fingerprints_unchanged"] is not True:
        raise RuntimeError("Read-only capture changed a Stage fingerprint")

    stage_path_after = manifest.verify_stage()
    bottle_path_after = manifest.verify_bottle()
    _assert_equal(
        "Stage path after capture",
        stage_path_after,
        manifest.stage_path,
    )
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

    kit_version = carb.tokens.get_tokens_interface().resolve(
        "${kit_version}"
    ).split("+", maxsplit=1)[0]
    versions = {
        "isaac_sim": importlib.metadata.version("isaacsim"),
        "kit": kit_version,
        "physx": _extension_version(extension_manager, "omni.physx"),
        "grasp_editor": _extension_version(
            extension_manager,
            manifest.isaac.grasp_editor_extension,
        ),
        "visual_tutor": _extension_version(
            extension_manager,
            VISUAL_TUTOR_EXTENSION_ID,
        ),
    }
    expected_versions = {
        "isaac_sim": manifest.isaac.version,
        "kit": manifest.isaac.kit,
        "physx": manifest.isaac.physx,
        "grasp_editor": manifest.isaac.grasp_editor_version,
    }
    for name, expected in expected_versions.items():
        _assert_equal(f"{name} version", versions[name], expected)

    object_prim_valid = bool(
        stage.GetPrimAtPath(manifest.prims.object).IsValid()
    )
    return {
        "status": "PASS",
        "classification": "LOCAL_VISUAL_TUTOR_LIVE_BRIDGE_PASS",
        "versions": versions,
        "manifest_path": str(MANIFEST_PATH),
        "stage_path": str(manifest.stage_path),
        "stage_sha256_before": stage_sha256_before,
        "stage_sha256_after": stage_sha256_after,
        "bottle_path": str(manifest.bottle_usd_path),
        "bottle_sha256_before": bottle_sha256_before,
        "bottle_sha256_after": bottle_sha256_after,
        "extension_parent_registered": str(extension_parent),
        "required_capture_prims": prim_readback,
        "session_object_readback": {
            "path": manifest.prims.object,
            "valid_before_prepare_session": object_prim_valid,
            "expected_creation_phase": "Task 3 prepare_approved_session",
        },
        "first_capture": first_capture,
        "capture": capture,
        "root_prim": capture["default_prim_path"],
        "root_sublayers": capture["root_sublayers"],
        "root_authored_reference_lines": (
            capture["root_authored_reference_lines"]
        ),
        "session_layer_identifier": session_identifier,
        "edit_target_identifier": capture["edit_target_identifier"],
        "timeline_readback": {
            "playing": capture["timeline_playing"],
            "stopped": capture["timeline_stopped"],
        },
        "main_thread_ident": main_thread_ident,
        "report_written_before_close": True,
        "ik": "NOT_RUN",
        "task8": "NOT_RUN",
    }


def main() -> int:
    args = _parse_args()
    manifest = _load_manifest()
    stage_path = manifest.verify_stage()
    bottle_path = manifest.verify_bottle()
    from my_visual_tutor.grasp_editor_manifest import sha256_file

    stage_sha256_before = sha256_file(stage_path)
    bottle_sha256_before = sha256_file(bottle_path)
    attempt_root = ARTIFACT_ROOT / f"attempt_{args.attempt}"
    report_path = attempt_root / "live_bridge_report.json"
    simulation_app = None
    report_written = False
    exit_code = 1
    try:
        import isaacsim

        simulation_app = isaacsim.SimulationApp({"headless": False})
        payload = _probe_live_bridge(
            simulation_app=simulation_app,
            manifest=manifest,
            stage_sha256_before=stage_sha256_before,
            bottle_sha256_before=bottle_sha256_before,
        )
        payload["attempt"] = args.attempt
        _write_report_before_close(report_path, payload)
        report_written = True
        exit_code = 0
    except Exception as error:
        payload = {
            "status": "FAIL",
            "classification": "LOCAL_VISUAL_TUTOR_LIVE_BRIDGE_FAIL",
            "attempt": args.attempt,
            "manifest_path": str(MANIFEST_PATH),
            "stage_path": str(manifest.stage_path),
            "stage_sha256_before": stage_sha256_before,
            "bottle_path": str(manifest.bottle_usd_path),
            "bottle_sha256_before": bottle_sha256_before,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=12),
            "report_written_before_close": True,
            "ik": "NOT_RUN",
            "task8": "NOT_RUN",
        }
        if not report_path.exists():
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

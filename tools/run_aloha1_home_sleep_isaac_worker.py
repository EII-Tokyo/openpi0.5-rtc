#!/usr/bin/env python3
"""Hash-pinned launcher for synchronized ALOHA Home/Sleep Isaac replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
)
DEFAULT_MANIFEST = (
    ROOT / "reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json"
)
DEFAULT_FINGER_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/"
    "configuration/finger_source_limits.usda"
)
DEFAULT_VALIDATOR = ROOT / "tools/validate_aloha1_home_sleep_digital.py"


def frame_deadline_ns(
    start_monotonic_ns: int, *, frame_index: int, physics_rate_hz: int
) -> int:
    """Return an absolute physics-frame deadline without accumulated rounding."""

    if start_monotonic_ns <= 0:
        raise ValueError("start_monotonic_ns must be positive")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if physics_rate_hz <= 0:
        raise ValueError("physics_rate_hz must be positive")
    return int(start_monotonic_ns) + (
        int(frame_index) * 1_000_000_000 // int(physics_rate_hz)
    )


def frame_lateness_status(lateness_ns: int, *, physics_rate_hz: int) -> str:
    """Reject a frame delayed by more than one nominal physics period."""

    if physics_rate_hz <= 0:
        raise ValueError("physics_rate_hz must be positive")
    maximum_lateness_ns = 1_000_000_000 // int(physics_rate_hz)
    return (
        "ON_TIME"
        if int(lateness_ns) <= maximum_lateness_ns
        else "ABORTED_DEADLINE_MISS"
    )


def wait_until_monotonic_ns(deadline_ns: int) -> int:
    """Wait for an absolute monotonic deadline and return the observed time."""

    while True:
        now = time.monotonic_ns()
        remaining_ns = int(deadline_ns) - now
        if remaining_ns <= 0:
            return now
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_000_000) / 1.0e9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(label: str, path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def build_isaac_worker_plan(
    *,
    run_id: str,
    stage: Path,
    stage_sha256: str,
    manifest: Path,
    manifest_sha256: str,
    finger_limit_layer: Path,
    finger_limit_sha256: str,
    command_signature: str,
    start_monotonic_ns: int,
    headless: bool,
    gui_workspace: int,
) -> dict[str, Any]:
    """Validate frozen inputs before any Isaac import or Stage open occurs."""

    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if start_monotonic_ns <= 0:
        raise ValueError("start_monotonic_ns must be positive")
    if gui_workspace < 1:
        raise ValueError("gui_workspace must be one-indexed and positive")
    stage_path = stage.resolve(strict=True)
    manifest_path = manifest.resolve(strict=True)
    finger_path = finger_limit_layer.resolve(strict=True)
    stage_hash = _verify_hash("stage", stage_path, stage_sha256)
    manifest_hash = _verify_hash("manifest", manifest_path, manifest_sha256)
    finger_hash = _verify_hash(
        "finger-limit layer", finger_path, finger_limit_sha256
    )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_payload.get("command_signature") != command_signature:
        raise ValueError("manifest command_signature mismatch")
    sample_count = int(manifest_payload.get("sample_count", -1))
    if sample_count != 1850:
        raise ValueError(f"expected 1850 frozen samples, observed {sample_count}")
    return {
        "schema_version": 1,
        "status": "READY",
        "worker": "isaac",
        "run_id": run_id,
        "manifest_sha256": manifest_hash,
        "command_signature": command_signature,
        "sample_count": sample_count,
        "start_monotonic_ns": int(start_monotonic_ns),
        "headless": bool(headless),
        "gui_workspace": int(gui_workspace),
        "stage": {"absolute_path": str(stage_path), "sha256": stage_hash},
        "manifest": {
            "absolute_path": str(manifest_path),
            "sha256": manifest_hash,
        },
        "finger_limit_layer": {
            "absolute_path": str(finger_path),
            "sha256": finger_hash,
        },
        "isaac_modules_imported_during_preflight": False,
        "stage_opened_during_preflight": False,
        "source_or_final_asset_modified": False,
    }


def build_validator_argv(
    *,
    python_executable: Path,
    validator: Path,
    stage: Path,
    stage_sha256: str,
    manifest: Path,
    manifest_sha256: str,
    finger_limit_layer: Path,
    finger_limit_sha256: str,
    output: Path,
    telemetry: Path,
    repeat_index: int,
    run_id: str,
    start_monotonic_ns: int,
    headless: bool,
) -> list[str]:
    """Build the exact argv for the already qualified digital validator."""

    return [
        str(python_executable),
        str(validator),
        "--stage",
        str(stage),
        "--stage-sha256",
        stage_sha256,
        "--manifest",
        str(manifest),
        "--manifest-sha256",
        manifest_sha256,
        "--finger-limit-layer",
        str(finger_limit_layer),
        "--finger-limit-sha256",
        finger_limit_sha256,
        "--output",
        str(output),
        "--telemetry",
        str(telemetry),
        "--repeat-index",
        str(repeat_index),
        "--run-id",
        run_id,
        "--start-monotonic-ns",
        str(start_monotonic_ns),
        "--realtime-pacing",
        "--headless" if headless else "--no-headless",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--stage-sha256", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--finger-limit-layer", type=Path, default=DEFAULT_FINGER_LAYER)
    parser.add_argument("--finger-limit-sha256", required=True)
    parser.add_argument("--command-signature", required=True)
    parser.add_argument("--start-monotonic-ns", required=True, type=int)
    parser.add_argument("--ready-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--gui-workspace", type=int, default=2)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--execute-isaac", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = build_isaac_worker_plan(
        run_id=args.run_id,
        stage=args.stage,
        stage_sha256=args.stage_sha256,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        finger_limit_layer=args.finger_limit_layer,
        finger_limit_sha256=args.finger_limit_sha256,
        command_signature=args.command_signature,
        start_monotonic_ns=args.start_monotonic_ns,
        headless=args.headless,
        gui_workspace=args.gui_workspace,
    )
    ready_output = args.ready_output.resolve()
    ready_output.parent.mkdir(parents=True, exist_ok=True)
    ready_output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.execute_isaac:
        print(
            json.dumps(
                {
                    "status": "READY_DRY_RUN",
                    "ready_output": str(ready_output),
                    "stage_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0

    argv = build_validator_argv(
        python_executable=Path(sys.executable),
        validator=DEFAULT_VALIDATOR.resolve(strict=True),
        stage=Path(plan["stage"]["absolute_path"]),
        stage_sha256=args.stage_sha256,
        manifest=Path(plan["manifest"]["absolute_path"]),
        manifest_sha256=args.manifest_sha256,
        finger_limit_layer=Path(plan["finger_limit_layer"]["absolute_path"]),
        finger_limit_sha256=args.finger_limit_sha256,
        output=args.output.resolve(),
        telemetry=args.telemetry.resolve(),
        repeat_index=args.repeat_index,
        run_id=args.run_id,
        start_monotonic_ns=args.start_monotonic_ns,
        headless=args.headless,
    )
    os.execv(sys.executable, argv)
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())

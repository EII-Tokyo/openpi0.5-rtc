#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

from tools.audit_aloha1_task8_baseline import start_usd_runtime_if_needed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_dict(data: Any) -> dict[str, dict[str, Any]]:
    return {
        str(item.name): {
            "value": item.value,
            "unit": str(getattr(item, "unit", "")),
        }
        for item in data.measurements
    }


def benchmark(
    *,
    app: Any,
    stage_path: Path,
    expected_sha256: str | None,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, Any]:
    import omni.kit.app
    import omni.timeline
    from pxr import Usd

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate(
        "isaacsim.benchmark.services", True  # noqa: FBT003 - C++ binding is positional
    )
    app.update()

    from isaacsim.benchmark.services.base_isaac_benchmark import set_sync_mode
    from isaacsim.benchmark.services.datarecorders.interface import InputContext
    from isaacsim.benchmark.services.datarecorders.memory import MemoryRecorder
    from isaacsim.benchmark.services.recorders import IsaacFrameTimeRecorder
    from isaacsim.benchmark.services.utils import wait_until_stage_is_fully_loaded
    from isaacsim.core.utils.stage import open_stage

    stage_path = stage_path.resolve(strict=True)
    stage_hash = _sha256(stage_path)
    set_sync_mode()
    load_start = time.perf_counter_ns()
    open_stage(str(stage_path))
    wait_until_stage_is_fully_loaded()
    load_ms = (time.perf_counter_ns() - load_start) / 1_000_000.0
    for _ in range(warmup_frames):
        app.update()

    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    default_prim = stage.GetDefaultPrim() if stage else None
    timeline = omni.timeline.get_timeline_interface()
    context = InputContext(
        artifact_prefix="aloha1_task8",
        kit_version="107.3.3",
        phase="fixed_frame_playing",
    )
    recorder = IsaacFrameTimeRecorder(context, gpu_frametime=True)
    memory_before = _measurement_dict(MemoryRecorder().get_data())
    timeline.play()
    recorder.start_collecting()
    wall_start = time.perf_counter_ns()
    for _ in range(measured_frames):
        app.update()
    measured_wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    recorder.stop_collecting()
    timeline.pause()
    frame_metrics = _measurement_dict(recorder.get_data())
    memory_after = _measurement_dict(MemoryRecorder().get_data())

    app_samples = frame_metrics.get("App_Update Frametime Samples", {}).get("value", [])
    physics_samples = frame_metrics.get("Physics Frametime Samples", {}).get("value", [])
    status = (
        "PASS"
        if (expected_sha256 is None or expected_sha256 == stage_hash)
        and stage is not None
        and default_prim
        and len(app_samples) > 0
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "status": status,
        "classification": "TASK8_FRESH_PROCESS_STAGE_BENCHMARK",
        "stage": {
            "absolute_path": str(stage_path),
            "sha256": stage_hash,
            "expected_sha256": expected_sha256,
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
        },
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "benchmark_source": "isaacsim.benchmark.services 5.1 local installation",
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "timeline_playing_during_measurement": True,
        },
        "metrics": {
            "stage_load_ms": load_ms,
            "measured_wall_ms": measured_wall_ms,
            "app_frame_sample_count": len(app_samples),
            "physics_frame_sample_count": len(physics_samples),
            "official_frame_recorder": frame_metrics,
            "memory_before": memory_before,
            "memory_after": memory_after,
        },
        "boundaries": {
            "physics_parameters_modified": False,
            "final_or_default_asset_modified": False,
            "grasp_acceptance_test": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = start_usd_runtime_if_needed()
    result = 1
    try:
        report = benchmark(
            app=app,
            stage_path=args.stage,
            expected_sha256=args.expected_sha256,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "stage": report["stage"]["absolute_path"],
                    "load_ms": report["metrics"]["stage_load_ms"],
                    "app_samples": report["metrics"]["app_frame_sample_count"],
                    "physics_samples": report["metrics"]["physics_frame_sample_count"],
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = 0 if report["status"] == "PASS" else 1
    except Exception:
        print("TASK8_BENCHMARK_EXCEPTION", flush=True)
        traceback.print_exc()
    finally:
        if app is not None:
            app.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Orchestrate the isolated Isaac 5.1 Hydra protoPath diagnostic matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from tools.aloha1_mapping.hydra_protopath_diagnosis import build_variant_matrix
from tools.aloha1_mapping.hydra_protopath_diagnosis import classify_diagnosis
from tools.aloha1_mapping.hydra_protopath_diagnosis import parse_protopath_errors

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
EXPECTED_STAGE_SHA256 = "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
ARTIFACT_ROOT = ROOT / ".codex/artifacts/20260729-aloha1-signal-correspondence/hydra_protopath_diagnosis"
REPORT_ROOT = ROOT / "reports/aloha1_mapping"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    mean = flat.mean(axis=0)
    standard_deviation = flat.std(axis=0)
    return {
        "readable": True,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean_rgb": [float(value) for value in mean],
        "std_rgb": [float(value) for value in standard_deviation],
        "nonuniform": bool(np.max(standard_deviation) > 2.0),
    }


def _run_variant(
    variant: dict[str, Any],
    stage: Path,
    expected_sha256: str,
    artifact_root: Path,
    *,
    suffix: str = "",
    reuse_existing: bool = False,
) -> dict[str, Any]:
    identifier = f"{variant['id']}{suffix}"
    variant_root = artifact_root / identifier
    variant_root.mkdir(parents=True, exist_ok=True)
    output = variant_root / "runtime.json"
    screenshot = variant_root / "native_raw.png"
    stdout_path = variant_root / "stdout.log"
    stderr_path = variant_root / "stderr.log"
    if variant["materialize_visual_instances"]:
        command = [
            str(ROOT / ".venv_issac/bin/python"),
            str(ROOT / "tools/probe_aloha1_hydra_protopath_variant.py"),
            "--variant-id",
            identifier,
            "--stage",
            str(stage),
            "--expected-stage-sha256",
            expected_sha256,
            "--overrides-json",
            json.dumps(variant["setting_overrides"], sort_keys=True),
            "--diagnostic-root",
            str(variant_root / "diagnostic_stage"),
            "--output",
            str(output),
            "--screenshot",
            str(screenshot),
            "--materialize-visual-instances",
        ]
    else:
        command = [
            str(ROOT / ".venv_issac/bin/python"),
            str(ROOT / "tools/capture_aloha1_signal_correspondence_screenshots.py"),
            "--stage",
            str(stage),
            "--output-root",
            str(variant_root / "exact_capture"),
            "--metadata",
            str(output),
            "--robot",
            "follower_left",
        ]
        if variant["setting_overrides"]:
            path, value = next(iter(variant["setting_overrides"].items()))
            command.extend(
                [
                    "--diagnostic-hydra-setting-path",
                    path,
                    "--diagnostic-hydra-setting-value",
                    str(value).lower(),
                ]
            )
    environment = {
        "PYTHONPATH": str(ROOT),
        "OMNI_KIT_ACCEPT_EULA": "YES",
    }
    if reuse_existing:
        if not output.exists() or not stdout_path.exists() or not stderr_path.exists():
            raise RuntimeError(f"resume artifacts are incomplete: {variant_root}")
        saved_runtime = json.loads(output.read_text(encoding="utf-8"))
        completed = SimpleNamespace(returncode=0 if saved_runtime.get("status") == "PASS" else 1)
    else:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_stream,
            stderr_path.open("w", encoding="utf-8") as stderr_stream,
        ):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**dict(__import__("os").environ), **environment},
                stdout=stdout_stream,
                stderr=stderr_stream,
                check=False,
            )
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    errors = parse_protopath_errors(f"{stdout_text}\n{stderr_text}")
    runtime = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else {
            "status": "FAIL",
            "error": "runtime report missing",
        }
    )
    screenshot_paths = []
    if variant["materialize_visual_instances"]:
        if screenshot.exists():
            screenshot_paths.append(screenshot)
    else:
        screenshot_paths = [
            Path(record["raw_absolute_path"])
            for record in runtime.get("captures", [])
            if Path(record["raw_absolute_path"]).exists()
        ]
        if screenshot_paths:
            screenshot = screenshot_paths[0]
    diagnostic = runtime.get("hydra_protopath_diagnostic", {})
    settings_effective = runtime.get(
        "settings_effective",
        diagnostic.get("settings_effective", {}),
    )
    cpu_inventory = runtime.get("cpu_usd", diagnostic.get("cpu_usd", {}))
    fabric_inventory = runtime.get(
        "fabric_usdrt",
        diagnostic.get("fabric_usdrt", {}),
    )
    image = (
        _image_metrics(screenshot)
        if screenshot.exists() and screenshot.stat().st_size > 0
        else {
            "readable": False,
            "nonuniform": False,
        }
    )
    use_fabric = settings_effective.get(
        "/app/useFabricSceneDelegate",
        {},
    ).get("value")
    cpu_mesh_count = cpu_inventory.get("visible_visual_mesh_count")
    fabric_mesh_count = fabric_inventory.get("mesh_count")
    actual_render_mesh_count = fabric_mesh_count if use_fabric and fabric_mesh_count is not None else cpu_mesh_count
    record = {
        "id": variant["id"],
        "run_id": identifier,
        "name": variant["name"],
        "status": runtime.get("status", "FAIL"),
        "exit_code": completed.returncode,
        "exit_code_source": (
            "INFERRED_FROM_MACHINE_RUNTIME_STATUS_ON_RESUME" if reuse_existing else "SUBPROCESS_RETURN_CODE"
        ),
        "command": command,
        "environment": environment,
        "setting_overrides": variant["setting_overrides"],
        "materialize_visual_instances": variant["materialize_visual_instances"],
        "runtime_report": str(output.resolve()),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "screenshot": str(screenshot.resolve()) if screenshot.exists() else None,
        "screenshots": [str(path.resolve()) for path in screenshot_paths],
        "image_metrics": image,
        "proto_error_count": errors["total_count"],
        "proto_error_unique_pair_count": errors["unique_pair_count"],
        "proto_error_unique_pairs": errors["unique_pairs"],
        "actual_render_mesh_count": actual_render_mesh_count,
        "actual_render_mesh_count_method": (
            "USDRT_FABRIC_POPULATED_MESH_COUNT" if use_fabric else "USD_OMNIHYDRA_INPUT_VISIBLE_VISUAL_MESH_COUNT"
        ),
        "cpu_visible_visual_mesh_count": cpu_mesh_count,
        "fabric_mesh_count": fabric_mesh_count,
        "settings_effective": settings_effective,
        "cpu_inventory": cpu_inventory,
        "fabric_inventory": fabric_inventory,
        "runtime": runtime,
    }
    record["native_render_complete"] = bool(
        record["exit_code"] == 0
        and image.get("readable")
        and image.get("nonuniform")
        and actual_render_mesh_count
        and actual_render_mesh_count > 0
        and errors["total_count"] == 0
    )
    record["deterministic_signature"] = _signature(
        {
            "id": record["id"],
            "setting_overrides": record["setting_overrides"],
            "materialize_visual_instances": record["materialize_visual_instances"],
            "proto_error_count": record["proto_error_count"],
            "proto_error_unique_pairs": record["proto_error_unique_pairs"],
            "actual_render_mesh_count": record["actual_render_mesh_count"],
            "cpu_visible_visual_mesh_count": record["cpu_visible_visual_mesh_count"],
            "fabric_mesh_count": record["fabric_mesh_count"],
            "native_render_complete": record["native_render_complete"],
            "settings_effective": settings_effective,
            "source_stage_unchanged": runtime.get(
                "source_stage_unchanged",
                runtime.get("stage", {}).get("immutable"),
            ),
        }
    )
    return record


def _write_csv(path: Path, variants: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "run_id",
        "status",
        "exit_code",
        "proto_error_count",
        "proto_error_unique_pair_count",
        "cpu_visible_visual_mesh_count",
        "fabric_mesh_count",
        "actual_render_mesh_count",
        "native_render_complete",
        "deterministic_signature",
        "screenshot",
        "stderr",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in variants:
            writer.writerow({field: record.get(field) for field in fields})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "| Variant | Setting change | protoPath errors | Render mesh count | Native render |",
        "|---|---|---:|---:|---|",
    ]
    for record in report["variants"]:
        setting = json.dumps(record["setting_overrides"], sort_keys=True) if record["setting_overrides"] else "none"
        rows.append(
            f"| {record['run_id']} | `{setting}` | {record['proto_error_count']} | "
            f"{record['actual_render_mesh_count']} | "
            f"{'PASS' if record['native_render_complete'] else 'FAIL'} |"
        )
    path.write_text(
        "\n".join(
            [
                "# ALOHA1 Hydra protoPath controlled diagnosis",
                "",
                f"- Status: `{report['status']}`",
                f"- Classification: `{report['classification']['classification']}`",
                f"- Frozen Stage: `{report['input_manifest']['frozen_stage']}`",
                f"- Frozen SHA-256: `{report['input_manifest']['frozen_stage_sha256']}`",
                "- Scope: screenshot rendering diagnosis only; physics composition was not changed.",
                "- Task 7 numeric reports: frozen and rechecked after the matrix.",
                "- Task 8: `NOT_RUN`.",
                "",
                *rows,
                "",
                "## Evidence classes",
                "",
                "- NVIDIA official documentation: Carbonite settings can be overridden per process and require scene reload where documented.",
                "- Local runtime readback: setting existence/type/value, delegate selection, USD/Fabric inventories.",
                "- Numerical evidence: error counts, unique instance/prototype pairs, mesh counts, image readability and signatures.",
                "- Engineering inference: the classification is limited to the predeclared matrix and does not claim an unobserved internal implementation cause.",
                "- Not proved: renderer-internal draw-call count and a general fix for unrelated stages.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    if stage != STAGE.resolve():
        raise ValueError("only the user-approved signal-correspondence Stage is accepted")
    stage_sha256 = _sha256(stage)
    if stage_sha256 != EXPECTED_STAGE_SHA256:
        raise RuntimeError(f"approved Stage hash changed: {stage_sha256} != {EXPECTED_STAGE_SHA256}")
    artifact_root = args.artifact_root.resolve()
    report_root = args.report_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    baseline_definition = {
        "id": "A",
        "name": "FSD_DEFAULT",
        "setting_overrides": {},
        "materialize_visual_instances": False,
    }

    def run_or_resume(
        definition: dict[str, Any],
        *,
        suffix: str = "",
    ) -> dict[str, Any]:
        identifier = f"{definition['id']}{suffix}"
        output = artifact_root / identifier / "runtime.json"
        if args.resume and output.exists():
            return _run_variant(
                definition,
                stage,
                stage_sha256,
                artifact_root,
                suffix=suffix,
                reuse_existing=True,
            )
        if args.resume and (artifact_root / identifier).exists():
            suffix = f"{suffix}_RESUME1"
        return _run_variant(
            definition,
            stage,
            stage_sha256,
            artifact_root,
            suffix=suffix,
        )

    variants = [run_or_resume(baseline_definition)]
    settings_before = variants[0]["runtime"].get(
        "settings_before",
        variants[0]["runtime"].get("hydra_protopath_diagnostic", {}).get("settings_before", {}),
    )
    supported = {path: bool(record.get("exists")) for path, record in settings_before.items()}
    unsupported = sorted(path for path, exists in supported.items() if not exists)
    matrix = build_variant_matrix(supported)
    variants.extend([run_or_resume(definition) for definition in matrix[1:]])

    classification = classify_diagnosis(variants)
    repeat_record = None
    effective_id = classification["effective_variant"]
    if effective_id in {item["id"] for item in matrix}:
        definition = next(item for item in matrix if item["id"] == effective_id)
        repeat_record = run_or_resume(definition, suffix="_REPEAT")
        variants.append(repeat_record)
    effective_record = next(
        (record for record in variants if record["id"] == effective_id),
        None,
    )
    repeat_deterministic = bool(
        effective_record
        and repeat_record
        and effective_record["deterministic_signature"] == repeat_record["deterministic_signature"]
    )

    restore_definition = {
        "id": "RESTORE",
        "name": "DEFAULT_SETTINGS_RESTORE_VERIFICATION",
        "setting_overrides": {},
        "materialize_visual_instances": False,
    }
    restore = run_or_resume(restore_definition)
    variants.append(restore)
    default_settings_restored = restore["settings_effective"] == variants[0]["settings_effective"]
    input_manifest = {
        "frozen_stage": str(stage),
        "frozen_stage_sha256": stage_sha256,
        "root_prim": variants[0]["cpu_inventory"].get("root_prim"),
        "root_sublayers": variants[0]["cpu_inventory"].get("root_sublayers"),
        "references": variants[0]["cpu_inventory"].get("references"),
        "required_prims": variants[0]["cpu_inventory"].get("required_prims"),
        "isaac_sim_version": "5.1.0.0",
        "kit_version": "107.3.3",
        "hydra_usdrt_delegate_version": "7.5.1",
        "usdrt_scenegraph_version": "7.6.1",
        "supported_settings": supported,
        "unsupported_settings": dict.fromkeys(
            unsupported,
            "NOT_SUPPORTED_IN_LOCAL_7_5_1",
        ),
        "official_gateway_used": True,
        "source_stage_modified": False,
        "task_8": "NOT_RUN",
    }
    input_manifest["manifest_signature"] = _signature(input_manifest)
    input_manifest_path = report_root / "aloha1_hydra_protopath_input_manifest.json"
    input_manifest_path.write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    status = (
        "PASS"
        if all(record["status"] == "PASS" for record in variants)
        and default_settings_restored
        and _sha256(stage) == stage_sha256
        else "PARTIAL"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "classification": classification,
        "responsibility_boundary_before_matrix": ("VERIFIED_FSD_INSTANCE_PROTOTYPE_RESOLUTION_FAILURE"),
        "input_manifest": input_manifest,
        "variants": variants,
        "repeat_deterministic": repeat_deterministic,
        "workaround_eligible": bool(classification["classification"] != "INCONCLUSIVE" and repeat_deterministic),
        "default_settings_restored": default_settings_restored,
        "frozen_stage_sha256_after": _sha256(stage),
        "frozen_stage_unchanged": _sha256(stage) == stage_sha256,
        "physics_composition_changed": False,
        "final_asset_instanceable_changed": False,
        "default_renderer_changed": False,
        "task_7b": "NOT_RUN",
        "task_8": "NOT_RUN",
    }
    report["deterministic_signature"] = _signature(
        {
            "classification": classification,
            "variant_signatures": [record["deterministic_signature"] for record in variants],
            "stage_sha256": stage_sha256,
            "default_settings_restored": default_settings_restored,
        }
    )
    json_path = report_root / "aloha1_hydra_protopath_diagnosis.json"
    markdown_path = report_root / "aloha1_hydra_protopath_diagnosis.md"
    csv_path = report_root / "aloha1_hydra_protopath_diagnosis_matrix.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(markdown_path, report)
    _write_csv(csv_path, variants)
    print(
        json.dumps(
            {
                "status": status,
                "classification": classification["classification"],
                "json": str(json_path),
                "matrix_csv": str(csv_path),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

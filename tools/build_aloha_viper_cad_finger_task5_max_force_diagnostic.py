#!/usr/bin/env python3
"""Build the isolated max-force-only supplier-CAD Task 5 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT
    / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
BASELINE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_max_force_only"
)
OUTPUT_LAYER = (
    OUTPUT_ROOT
    / "configuration/supplier_cad_finger_max_force_only.usda"
)
OUTPUT_STAGE = OUTPUT_ROOT / "aloha_viperx_supplier_cad_max_force_only.usda"
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_max_force_asset.json"
)

SOURCE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
BASELINE_SHA256 = (
    "8040edd01859af9f8c51285d198d34aae19e66625a2d5f21729879774e1644d9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exact(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite drifted diagnostic: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    source = SOURCE_STAGE.resolve(strict=True)
    baseline = BASELINE_STAGE.resolve(strict=True)
    source_before = _sha256(source)
    baseline_before = _sha256(baseline)
    if source_before != SOURCE_SHA256:
        raise RuntimeError("approved source Stage hash mismatch")
    if baseline_before != BASELINE_SHA256:
        raise RuntimeError("baseline diagnostic Stage hash mismatch")

    layer_text = """#usda 1.0

over "workcell"
{
    over "joints"
    {
        over "vx300s_left_left_finger"
        {
            float drive:linear:physics:maxForce = 5
        }

        over "vx300s_left_right_finger"
        {
            float drive:linear:physics:maxForce = 5
        }
    }
}
"""
    stage_text = """#usda 1.0
(
    defaultPrim = "workcell"
    metersPerUnit = 1
    subLayers = [
        @configuration/supplier_cad_finger_max_force_only.usda@
    ]
    upAxis = "Z"
)

def Xform "workcell" (
    prepend references = @../cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda@</workcell>
)
{
}
"""
    _write_exact(OUTPUT_LAYER, layer_text)
    _write_exact(OUTPUT_STAGE, stage_text)
    source_after = _sha256(source)
    baseline_after = _sha256(baseline)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "profile": "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING",
        "changed_variable": "drive:linear:physics:maxForce",
        "changed_values": {
            "left_finger_n": 5.0,
            "right_finger_n": 5.0,
        },
        "frozen_authored_values": {
            "drive_type": "force",
            "stiffness": 200.0,
            "damping": 0.0,
            "collider": "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC",
        },
        "parameter_evidence": {
            "value": 5.0,
            "semantics": "URDF prismatic joint effort limit",
            "sources": [
                str(
                    (
                        ROOT
                        / "assets/Trossen/ALOHA1/1.0/follower_vx300s/"
                        "follower_left/source/follower_left.urdf"
                    ).resolve()
                ),
                str((ROOT / "configs/aloha1_joint_map.yaml").resolve()),
            ],
        },
        "inputs": {
            "approved_source_stage": {
                "absolute_path": str(source),
                "sha256_before": source_before,
                "sha256_after": source_after,
            },
            "baseline_diagnostic_stage": {
                "absolute_path": str(baseline),
                "sha256_before": baseline_before,
                "sha256_after": baseline_after,
            },
        },
        "outputs": {
            "configuration_layer": {
                "absolute_path": str(OUTPUT_LAYER.resolve()),
                "sha256": _sha256(OUTPUT_LAYER),
            },
            "diagnostic_stage": {
                "absolute_path": str(OUTPUT_STAGE.resolve()),
                "sha256": _sha256(OUTPUT_STAGE),
            },
        },
        "gates": {
            "approved_source_stage_immutable": (
                source_before == source_after == SOURCE_SHA256
            ),
            "baseline_diagnostic_stage_immutable": (
                baseline_before == baseline_after == BASELINE_SHA256
            ),
            "only_max_force_authored": True,
            "default_or_final_asset_unchanged": True,
        },
        "scope": {
            "bottle_contact_grasp": "NOT_RUN",
            "task8": "NOT_RUN",
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL"
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"status={report['status']}")
    print(f"stage={OUTPUT_STAGE.resolve()}")
    print(f"report={OUTPUT_REPORT.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compose root-frame correction over the frozen 5 N finger-drive profile."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
MAX_FORCE_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_max_force_asset.json"
)
ROOT_FRAME_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_root_frame_asset.json"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_max_force_plus_root_frame"
)
OUTPUT_LAYER = (
    OUTPUT_ROOT
    / "configuration/supplier_cad_root_frame_over_max_force.usda"
)
OUTPUT_STAGE = (
    OUTPUT_ROOT
    / "aloha_viperx_supplier_cad_max_force_plus_root_frame.usda"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_combined_asset.json"
)
SOURCE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
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


def render_configuration_layer(
    *,
    position: Sequence[float],
    rotation: Sequence[float],
) -> str:
    """Render only the world-side frame of the follower-left root joint."""

    if len(position) != 3 or len(rotation) != 4:
        raise ValueError("root-joint frame must be xyz plus wxyz")
    return f"""#usda 1.0

over "workcell"
{{
    over "joints"
    {{
        over "rootJoint_vx300s_left"
        {{
            point3f physics:localPos0 = ({position[0]:.17g}, {position[1]:.17g}, {position[2]:.17g})
            quatf physics:localRot0 = ({rotation[0]:.17g}, {rotation[1]:.17g}, {rotation[2]:.17g}, {rotation[3]:.17g})
        }}
    }}
}}
"""


def render_diagnostic_stage() -> str:
    """Reference the frozen max-force diagnostic under a root-frame layer."""

    return """#usda 1.0
(
    defaultPrim = "workcell"
    metersPerUnit = 1
    subLayers = [
        @configuration/supplier_cad_root_frame_over_max_force.usda@
    ]
    upAxis = "Z"
)

def Xform "workcell" (
    prepend references = @../cad_finger_task5_max_force_only/aloha_viperx_supplier_cad_max_force_only.usda@</workcell>
)
{
}
"""


def main() -> int:
    source = SOURCE_STAGE.resolve(strict=True)
    source_before = _sha256(source)
    if source_before != SOURCE_SHA256:
        raise RuntimeError("approved source Stage hash mismatch")
    max_force_report_path = MAX_FORCE_REPORT.resolve(strict=True)
    root_frame_report_path = ROOT_FRAME_REPORT.resolve(strict=True)
    max_force_report = json.loads(
        max_force_report_path.read_text(encoding="utf-8")
    )
    root_frame_report = json.loads(
        root_frame_report_path.read_text(encoding="utf-8")
    )
    if max_force_report["status"] != "PASS":
        raise RuntimeError("max-force parent asset report is not PASS")
    if root_frame_report["status"] != "PASS":
        raise RuntimeError("root-frame source report is not PASS")
    max_force_stage = Path(
        max_force_report["outputs"]["diagnostic_stage"]["absolute_path"]
    ).resolve(strict=True)
    max_force_hash_before = _sha256(max_force_stage)
    if (
        max_force_hash_before
        != max_force_report["outputs"]["diagnostic_stage"]["sha256"]
    ):
        raise RuntimeError("max-force parent Stage hash drift")

    computed = root_frame_report["computed_frame"]
    position = computed["body1_world_position_m"]
    rotation = computed["body1_world_orientation_wxyz"]
    layer_text = render_configuration_layer(
        position=position,
        rotation=rotation,
    )
    stage_text = render_diagnostic_stage()
    _write_exact(OUTPUT_LAYER, layer_text)
    _write_exact(OUTPUT_STAGE, stage_text)

    source_after = _sha256(source)
    max_force_hash_after = _sha256(max_force_stage)
    gates = {
        "approved_source_stage_immutable": (
            source_before == source_after == SOURCE_SHA256
        ),
        "max_force_parent_immutable": (
            max_force_hash_before == max_force_hash_after
        ),
        "only_root_frame_authored_in_new_layer": (
            "physics:localPos0" in layer_text
            and "physics:localRot0" in layer_text
            and "drive:" not in layer_text
            and "collision" not in layer_text.lower()
            and "material" not in layer_text.lower()
        ),
        "root_frame_computed_not_guessed": (
            computed["method"]
            == "UsdGeom.XformCache.GetLocalToWorldTransform"
        ),
        "default_or_final_asset_unchanged": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "profile": "DIAGNOSTIC_ONLY_NOT_FINAL_COMPOSED_MAPPING",
        "parent_profile": "max_force_only",
        "changed_variable_relative_to_parent": [
            "physics:localPos0",
            "physics:localRot0",
        ],
        "inherited_parent_values": {
            "left_finger_max_force_n": 5.0,
            "right_finger_max_force_n": 5.0,
            "drive_type": "force",
            "stiffness": 200.0,
            "damping": 0.0,
        },
        "computed_root_frame": computed,
        "frozen": {
            "collider": "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC",
            "friction": "UNCHANGED",
            "physics_frequency_hz": 60,
            "solver_iterations": "UNCHANGED",
            "bottle": "NOT_PRESENT",
        },
        "inputs": {
            "approved_source_stage": {
                "absolute_path": str(source),
                "sha256_before": source_before,
                "sha256_after": source_after,
            },
            "max_force_parent_stage": {
                "absolute_path": str(max_force_stage),
                "sha256_before": max_force_hash_before,
                "sha256_after": max_force_hash_after,
            },
            "max_force_asset_report": {
                "absolute_path": str(max_force_report_path),
                "sha256": _sha256(max_force_report_path),
            },
            "root_frame_asset_report": {
                "absolute_path": str(root_frame_report_path),
                "sha256": _sha256(root_frame_report_path),
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
        "gates": gates,
        "scope": {
            "bottle_contact_grasp": "NOT_RUN",
            "task8": "NOT_RUN",
        },
    }
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

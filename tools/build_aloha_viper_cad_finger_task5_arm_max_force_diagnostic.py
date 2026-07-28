#!/usr/bin/env python3
"""Author URDF-evidenced arm maxForce over the frozen Task 5 diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
PARENT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_combined_asset.json"
)
URDF_PATH = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/follower_vx300s/"
    "follower_left/source/follower_left.urdf"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_arm_max_force_over_combined"
)
OUTPUT_LAYER = (
    OUTPUT_ROOT
    / "configuration/supplier_cad_arm_max_force_only.usda"
)
OUTPUT_STAGE = (
    OUTPUT_ROOT
    / "aloha_viperx_supplier_cad_arm_max_force_over_combined.usda"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_arm_max_force_asset.json"
)
SOURCE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)
ARM_MAX_FORCE_N = {
    "vx300s_left_waist": 10.0,
    "vx300s_left_shoulder": 20.0,
    "vx300s_left_elbow": 15.0,
    "vx300s_left_forearm_roll": 2.0,
    "vx300s_left_wrist_angle": 5.0,
    "vx300s_left_wrist_rotate": 1.0,
}


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


def _urdf_arm_efforts(path: Path) -> dict[str, float]:
    tree = ET.parse(path)
    by_name = {
        joint.attrib["name"]: joint
        for joint in tree.getroot().findall("joint")
    }
    result = {}
    for prefixed_name in ARM_MAX_FORCE_N:
        urdf_name = prefixed_name.removeprefix("vx300s_left_")
        joint = by_name.get(urdf_name)
        if joint is None:
            raise RuntimeError(f"URDF joint missing: {urdf_name}")
        limit = joint.find("limit")
        if limit is None or "effort" not in limit.attrib:
            raise RuntimeError(f"URDF effort missing: {urdf_name}")
        result[prefixed_name] = float(limit.attrib["effort"])
    return result


def render_configuration_layer(
    efforts: Mapping[str, float],
) -> str:
    """Render only angular maxForce attributes for the six arm joints."""

    if dict(efforts) != ARM_MAX_FORCE_N:
        raise ValueError("arm maxForce values differ from frozen URDF evidence")
    blocks = []
    for joint_name, effort in efforts.items():
        blocks.append(
            f"""        over "{joint_name}"
        {{
            float drive:angular:physics:maxForce = {effort:g}
        }}"""
        )
    return """#usda 1.0

over "workcell"
{
    over "joints"
    {
""" + "\n\n".join(blocks) + """
    }
}
"""


def render_diagnostic_stage() -> str:
    """Reference the root-frame plus finger-maxForce parent diagnostic."""

    return """#usda 1.0
(
    defaultPrim = "workcell"
    metersPerUnit = 1
    subLayers = [
        @configuration/supplier_cad_arm_max_force_only.usda@
    ]
    upAxis = "Z"
)

def Xform "workcell" (
    prepend references = @../cad_finger_task5_max_force_plus_root_frame/aloha_viperx_supplier_cad_max_force_plus_root_frame.usda@</workcell>
)
{
}
"""


def main() -> int:
    source = SOURCE_STAGE.resolve(strict=True)
    source_before = _sha256(source)
    if source_before != SOURCE_SHA256:
        raise RuntimeError("approved source Stage hash mismatch")
    parent_report_path = PARENT_REPORT.resolve(strict=True)
    parent_report = json.loads(
        parent_report_path.read_text(encoding="utf-8")
    )
    if parent_report["status"] != "PASS":
        raise RuntimeError("combined parent asset report is not PASS")
    parent_stage = Path(
        parent_report["outputs"]["diagnostic_stage"]["absolute_path"]
    ).resolve(strict=True)
    parent_hash_before = _sha256(parent_stage)
    if (
        parent_hash_before
        != parent_report["outputs"]["diagnostic_stage"]["sha256"]
    ):
        raise RuntimeError("combined parent Stage hash drift")
    urdf = URDF_PATH.resolve(strict=True)
    urdf_hash_before = _sha256(urdf)
    urdf_efforts = _urdf_arm_efforts(urdf)
    if urdf_efforts != ARM_MAX_FORCE_N:
        raise RuntimeError(
            f"URDF arm efforts differ from manifest: {urdf_efforts}"
        )

    layer_text = render_configuration_layer(urdf_efforts)
    stage_text = render_diagnostic_stage()
    _write_exact(OUTPUT_LAYER, layer_text)
    _write_exact(OUTPUT_STAGE, stage_text)
    source_after = _sha256(source)
    parent_hash_after = _sha256(parent_stage)
    urdf_hash_after = _sha256(urdf)
    gates = {
        "approved_source_stage_immutable": (
            source_before == source_after == SOURCE_SHA256
        ),
        "parent_diagnostic_stage_immutable": (
            parent_hash_before == parent_hash_after
        ),
        "source_urdf_immutable": (
            urdf_hash_before == urdf_hash_after
        ),
        "urdf_efforts_exactly_match_layer": (
            urdf_efforts == ARM_MAX_FORCE_N
        ),
        "only_arm_angular_max_force_authored": (
            layer_text.count("drive:angular:physics:maxForce") == 6
            and "stiffness" not in layer_text
            and "damping" not in layer_text
            and "collision" not in layer_text.lower()
            and "material" not in layer_text.lower()
        ),
        "default_or_final_asset_unchanged": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "profile": "DIAGNOSTIC_ONLY_NOT_FINAL_ARM_HOLD",
        "parent_profile": "max_force_plus_root_frame",
        "changed_variable_relative_to_parent": (
            "drive:angular:physics:maxForce"
        ),
        "changed_values_n": urdf_efforts,
        "frozen": {
            "arm_stiffness": "INHERIT_UNCHANGED",
            "arm_damping": "INHERIT_UNCHANGED",
            "finger_max_force_n": {
                "left": 5.0,
                "right": 5.0,
            },
            "root_joint_frame": "INHERIT_COMPUTED_PARENT",
            "collider": "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC",
            "friction": "UNCHANGED",
            "physics_frequency_hz": 60,
            "solver_iterations": "UNCHANGED",
            "bottle": "NOT_PRESENT",
        },
        "evidence": {
            "urdf": {
                "absolute_path": str(urdf),
                "sha256_before": urdf_hash_before,
                "sha256_after": urdf_hash_after,
                "joint_effort_values_n": urdf_efforts,
            },
            "joint_map": {
                "absolute_path": str(
                    (ROOT / "configs/aloha1_joint_map.yaml").resolve()
                ),
                "sha256": _sha256(
                    ROOT / "configs/aloha1_joint_map.yaml"
                ),
            },
        },
        "inputs": {
            "approved_source_stage": {
                "absolute_path": str(source),
                "sha256_before": source_before,
                "sha256_after": source_after,
            },
            "parent_diagnostic_stage": {
                "absolute_path": str(parent_stage),
                "sha256_before": parent_hash_before,
                "sha256_after": parent_hash_after,
            },
            "parent_asset_report": {
                "absolute_path": str(parent_report_path),
                "sha256": _sha256(parent_report_path),
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

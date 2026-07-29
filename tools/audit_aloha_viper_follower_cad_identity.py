#!/usr/bin/env python3
"""Generate the bounded follower left/right CAD identity report."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools.aloha1_mapping.follower_cad_identity_report import build_identity_report
from tools.aloha1_mapping.follower_cad_identity_report import render_markdown

ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIT = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "follower_left_right_cad_identity/freecad_identity_audit.json"
)
XACRO_CONFIG = ROOT / "configs/aloha1_xacro_args.yaml"
LEFT_URDF = ROOT / "generated/urdf/follower_left.urdf"
RIGHT_URDF = ROOT / "generated/urdf/follower_right.urdf"
PURCHASE_REPORT = (
    ROOT / "reports/aloha1_mapping/aloha_purchased_model_identification.json"
)
TOOLCHAIN_MANIFEST = ROOT / "local_tools/freecad-tessellation/manifest.json"
TESSELLATION_MANIFEST = (
    ROOT
    / "local_tools/freecad-tessellation/validation/"
    "final_fresh_tessellation/manifest.json"
)
OUTPUT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    inputs = [
        RAW_AUDIT,
        XACRO_CONFIG,
        LEFT_URDF,
        RIGHT_URDF,
        PURCHASE_REPORT,
        TOOLCHAIN_MANIFEST,
        TESSELLATION_MANIFEST,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    report = build_identity_report(
        raw_cad_audit=_json(RAW_AUDIT),
        xacro_config=yaml.safe_load(
            XACRO_CONFIG.read_text(encoding="utf-8")
        ),
        purchase_report=_json(PURCHASE_REPORT),
        toolchain_manifest=_json(TOOLCHAIN_MANIFEST),
        tessellation_manifest=_json(TESSELLATION_MANIFEST),
        left_urdf_path=LEFT_URDF,
        right_urdf_path=RIGHT_URDF,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        f"ALOHA follower CAD identity: {report['status']} "
        f"{report['classification']}"
    )
    return 0 if report["robot_local_identity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

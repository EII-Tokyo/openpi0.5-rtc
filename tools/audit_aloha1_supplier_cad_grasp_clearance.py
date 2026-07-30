#!/usr/bin/env python3
"""Finalize two fresh FreeCAD complete-gripper clearance audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def _markdown(report: dict[str, Any]) -> str:
    selection = report["station_selection"]
    contact = report["contact_solution"]
    frame = report["grasp_frame"]
    shell = report["forbidden_envelopes"]["supplier_gripper_shell"]
    bar = report["forbidden_envelopes"]["runtime_urdf_gripper_bar"]
    return "\n".join(
        [
            "# ALOHA1 Complete Gripper CAD Clearance",
            "",
            f"- Status: `{report['status']}`",
            f"- Classification: `{report['classification']}`",
            "- Supplier CAD: complete Simple Aloha Viper gripper assembly",
            "- Bottle: project-authored Bottle500 B-Rep",
            "- Task 8: `NOT_RUN`",
            "",
            "## Result",
            "",
            (
                "- Rejected run13 bottle-axis center: "
                f"`{selection['rejected_station']['station_m']:.9f} m`"
            ),
            (
                "- Corrected bottle-axis center: "
                f"`{selection['selected_station_m']:.9f} m`"
            ),
            (
                "- Corrected pad-contact midpoint: "
                f"`{selection['selected_pad_contact_station_m']:.9f} m`"
            ),
            (
                "- Max-min minimum hard margin: "
                f"`{selection['selected_minimum_margin_m']:.9f} m`"
            ),
            (
                "- Contact finger q: "
                f"`left={contact['left_finger_q_m']:.9f} m`, "
                f"`right={contact['right_finger_q_m']:.9f} m`"
            ),
            (
                "- Bottle-axis center offset from pad frame: "
                f"`{frame['bottle_axis_center_from_grasp_m']}`"
            ),
            "",
            "## Evidence Boundary",
            "",
            (
                "- Supplier shell maximum approach extent: "
                f"`{shell['maximum_approach_x_m']:.9f} m`."
            ),
            (
                "- Runtime URDF gripper-bar conservative maximum extent: "
                f"`{bar['maximum_approach_x_m']:.9f} m`."
            ),
            (
                "- The runtime bar is the controlling forbidden envelope; "
                "the supplier shell and runtime mesh are retained separately."
            ),
            (
                "- The two fresh FreeCAD semantic signatures match: "
                f"`{report['determinism']['semantic_signature']}`."
            ),
            "",
            "This is a static geometry gate. It does not prove contact, hold, "
            "IK reachability, or dynamic pickup.",
            "",
        ]
    )


def main() -> int:
    args = _parse_args()
    run_a_path = args.run_a.resolve(strict=True)
    run_b_path = args.run_b.resolve(strict=True)
    run_a = json.loads(run_a_path.read_text(encoding="utf-8"))
    run_b = json.loads(run_b_path.read_text(encoding="utf-8"))
    signature_a = str(run_a.get("semantic_signature"))
    signature_b = str(run_b.get("semantic_signature"))
    match = (
        run_a.get("status") == "PASS"
        and run_b.get("status") == "PASS"
        and signature_a == signature_b
    )
    if not match:
        raise RuntimeError(
            "fresh FreeCAD clearance semantic signatures do not match"
        )
    report = dict(run_a)
    report["determinism"] = {
        "status": "PASS",
        "fresh_run_count": 2,
        "semantic_signature_match": True,
        "semantic_signature": signature_a,
        "run_a": {
            "absolute_path": str(run_a_path),
            "sha256": _sha256(run_a_path),
        },
        "run_b": {
            "absolute_path": str(run_b_path),
            "sha256": _sha256(run_b_path),
        },
    }
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_json": str(output_json),
                "output_md": str(output_md),
                "semantic_signature": signature_a,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

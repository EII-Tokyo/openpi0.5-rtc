#!/usr/bin/env python3
"""Build the corrective Task 7 finger-pair screenshot re-audit report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
SCREENSHOT_REVIEW = REPORT_ROOT / "aloha1_task7_virtual_helper_failure_screenshot_review_right.json"
JOINT_GEOMETRY = REPORT_ROOT / "aloha1_task7_joint_state_geometry_audit.json"
STATIC_COLLISION = REPORT_ROOT / "aloha1_cad_derived_collision_replan_static.json"
GEOMETRY_AUDIT = (
    ROOT
    / ".codex/artifacts/20260802-aloha-task7-finger-orientation-reaudit/"
    "intact_candidate_q_geometry.json"
)
BROKEN_RUNTIME_LOG = (
    ROOT
    / ".codex/artifacts/20260802-aloha-task7-finger-orientation-reaudit/"
    "current_candidate_q_geometry.log"
)
OUTPUT = REPORT_ROOT / "aloha1_task7_finger_pair_reaudit.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finger_pair(state: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        pair
        for pair in state["pairs"]
        if pair["scope"] == "FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP"
    ]
    if len(pairs) != 1:
        raise RuntimeError(f"expected one finger pair in {state['state']}: {len(pairs)}")
    pair = pairs[0]
    return {
        "target_m": state["target_m"],
        "readback_m": state["readback_m"],
        "relation": pair["relation"],
        "signed_chebyshev_margin_m": pair["signed_chebyshev_margin_m"],
        "overlap_volume_m3": pair["overlap_volume_m3"],
        "intersection_vertex_count": pair["intersection_vertex_count"],
        "left_collider": pair["collider_a"],
        "right_collider": pair["collider_b"],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    zero = report["geometry_states"]["illegal_static_q_zero"]
    closed = report["geometry_states"]["legal_closed_limit"]
    return "\n".join(
        [
            "# ALOHA1 Task 7 finger-pair corrective re-audit",
            "",
            f"- Status: `{report['status']}` (the screenshot geometry gate, not the final asset)",
            f"- Classification: `{report['classification']}`",
            f"- Task 7: `{report['task7']}`",
            f"- Task 8: `{report['task8']}`",
            "",
            "## What the disputed screenshot actually shows",
            "",
            "The image came from a deliberately rejected helper-body candidate. The capture "
            "script loaded the USD and rendered authored transforms without a physics reset, "
            "joint-limit solve, or articulation readback. Its finger geometry therefore remained "
            "at static `q=(0, 0)`, outside the authored legal intervals.",
            "",
            f"At static zero, the two independently authored supplier-CAD colliders are "
            f"`{zero['relation']}` with `{zero['overlap_volume_m3']:.12g} m^3` overlap and "
            f"`{zero['signed_chebyshev_margin_m']:.12g} m` signed margin.",
            "",
            f"At the legal closed limits `(+0.021, -0.021) m`, they are "
            f"`{closed['relation']}` with `{closed['overlap_volume_m3']:.12g} m^3` overlap.",
            "",
            "## Corrected interpretation",
            "",
            "- The left and right finger collision meshes are separate prims under separate finger links; they were not merged at the base.",
            "- Articulation self-collision is disabled in the frozen diagnostic configuration, so finger-finger contact is not the closing stop.",
            "- The authored prismatic limits are the closing stop. A static viewport capture that bypasses them is invalid finger-installation evidence.",
            "- The previous screenshot `PASS` is revoked. Visual legibility remains PASS, but supplier-CAD orientation, legal runtime qpos, and pair response were NOT_RUN in that capture.",
            "- The image alone cannot distinguish a reversed installation from an illegal unsolved state. The numeric q-state experiment identifies the latter for this capture.",
            "",
        ]
    )


def build(*, write: bool = True) -> dict[str, Any]:
    review = _load(SCREENSHOT_REVIEW)
    joints = _load(JOINT_GEOMETRY)
    collision = _load(STATIC_COLLISION)
    geometry = _load(GEOMETRY_AUDIT)
    states = {state["state"]: _finger_pair(state) for state in geometry["states"]}
    finger_joints = [
        item
        for item in joints["joints"]
        if item["follower"] == "follower_right" and "finger" in item["prim_path"]
    ]
    collider_records = [
        item
        for item in review["overlay"]["records"]
        if "finger_link" in item["source_prim"]
    ]
    collider_paths = [item["source_prim"] for item in collider_records]
    distinct_links = (
        len(collider_paths) == 2
        and any("left_finger_link" in path for path in collider_paths)
        and any("right_finger_link" in path for path in collider_paths)
    )
    self_collision = collision["session_diagnostics"]["self_collision"]["follower_right"]
    report = {
        "schema_version": 1,
        "status": "FAIL",
        "classification": "ILLEGAL_STATIC_Q_ZERO_BYPASSED_RUNTIME_LIMITS",
        "screenshot_evidence_status": "REJECTED_FOR_FINGER_GEOMETRY",
        "visual_evidence_legibility": "PASS",
        "finger_installation_orientation_from_this_image": "INCONCLUSIVE",
        "task7": "PARTIAL",
        "task8": "NOT_RUN",
        "disputed_screenshot": {
            "absolute_path": review["captures"][0]["raw_absolute_path"],
            "sha256": review["captures"][0]["raw_sha256"],
            "stage": review["stage"],
            "capture_semantics": "STATIC_USD_RENDER_NO_PHYSICS_RESET_OR_JOINT_READBACK",
            "corrected_review_report": str(SCREENSHOT_REVIEW.resolve()),
            "corrected_review_report_sha256": _sha256(SCREENSHOT_REVIEW),
        },
        "collider_authoring": {
            "left_right_meshes_merged": False,
            "distinct_rigid_link_paths": distinct_links,
            "collider_paths": collider_paths,
            "source": "runtime-composed USD inventory from capture report",
        },
        "joint_authoring": {
            "records": finger_joints,
            "legal_limits_m": {
                "left_finger": [0.021, 0.057],
                "right_finger": [-0.057, -0.021],
            },
            "static_geometry_state_m": [0.0, 0.0],
        },
        "runtime_policy": {
            "enabled_self_collisions": bool(self_collision["before"]),
            "diagnostic_session_value": bool(self_collision["diagnostic_session_value"]),
            "meaning": (
                "Finger-pair contact is not an active articulation closing stop in this "
                "configuration; authored joint limits must prevent illegal closure."
            ),
        },
        "geometry_states": states,
        "geometry_audit": {
            "absolute_path": str(GEOMETRY_AUDIT.resolve()),
            "sha256": _sha256(GEOMETRY_AUDIT),
            "method": geometry["method"],
        },
        "broken_candidate_runtime_probe": {
            "absolute_log_path": str(BROKEN_RUNTIME_LOG.resolve()),
            "sha256": _sha256(BROKEN_RUNTIME_LOG),
            "status": "FAIL",
            "reason": (
                "The deliberately removed helper bodies prevent ee_gripper/finger joints "
                "and mimic from forming a valid articulation, so this candidate cannot "
                "serve as runtime gripper evidence."
            ),
        },
        "impact_boundary": {
            "previous_failure_screenshot_pass_revoked": True,
            "physicsrules_machine_findings_invalidated": False,
            "previous_user_confirmed_dynamic_grasp_videos_invalidated": False,
            "final_or_default_asset_modified": False,
        },
        "source_reports": [
            {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (SCREENSHOT_REVIEW, JOINT_GEOMETRY, STATIC_COLLISION)
        ],
    }
    if write:
        OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUTPUT_MD.write_text(_render_markdown(report).rstrip() + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

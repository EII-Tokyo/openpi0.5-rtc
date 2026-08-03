#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
OUTPUT = REPORT_ROOT / "aloha1_task8_current_summary.json"
MARKDOWN = REPORT_ROOT / "aloha1_task8_current_summary.md"


def _load(name: str) -> tuple[dict[str, Any], dict[str, str]]:
    path = REPORT_ROOT / name
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_report() -> dict[str, Any]:
    authorization, authorization_input = _load("aloha1_task8_progression_authorization.json")
    baseline, baseline_input = _load("aloha1_task8_baseline_inventory.json")
    candidate, candidate_input = _load("aloha1_task8_visual_material_candidate.json")
    benchmark, benchmark_input = _load("aloha1_task8_benchmark_comparison.json")
    visual_meshes = [mesh for mesh in baseline["meshes"] if not mesh["is_collision"]]
    visual_points = sum(int(mesh["point_count"]) for mesh in visual_meshes)
    visual_faces = sum(int(mesh["face_count"]) for mesh in visual_meshes)
    finger_savings_points = 1662
    finger_savings_faces = 3324
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS_FIRST_CANDIDATE_NO_MEASURABLE_IMPROVEMENT",
        "task7": "PARTIAL_ACCEPTED_FOR_TASK8",
        "task8": "AUTHORIZED_IN_PROGRESS",
        "inputs": {
            "authorization": authorization_input,
            "baseline": baseline_input,
            "candidate": candidate_input,
            "benchmark": benchmark_input,
        },
        "gates": {
            "approximate_simulation_allowed": authorization["progression_gate"][
                "approximate_simulation_allowed"
            ],
            "baseline_inventory": baseline["status"],
            "candidate_static": candidate["status"],
            "candidate_performance": benchmark["status"],
            "candidate_promoted": False,
            "final_or_default_asset_modified": False,
        },
        "first_candidate": {
            "type": "VISUAL_MATERIAL_BINDING_DEDUP",
            "effective_visual_materials_before": candidate["comparison"][
                "baseline_bound_visual_material_count"
            ],
            "effective_visual_materials_after": candidate["comparison"][
                "candidate_bound_visual_material_count"
            ],
            "protected_physics_signature_unchanged": candidate["comparison"][
                "protected_physics_signature_unchanged"
            ],
            "performance_result": benchmark["status"],
            "nonoverlapping_regressions": benchmark["nonoverlapping_regressions"],
            "grasp_smoke": "NOT_RUN_CANDIDATE_REJECTED_BEFORE_RUNTIME_ACCEPTANCE",
            "visual_failure_evidence": benchmark["visual_evidence"],
        },
        "next_optimization_boundary": {
            "baseline_instanceable_prim_count": baseline["summary"][
                "instanceable_prim_count"
            ],
            "baseline_payload_prim_count": baseline["summary"]["payload_prim_count"],
            "visual_mesh_count": baseline["summary"]["visual_mesh_count"],
            "visual_instance_proxy_mesh_count": sum(
                bool(mesh["is_instance_proxy"]) for mesh in visual_meshes
            ),
            "nonproxy_supplier_finger_visual_mesh_count": 4,
            "maximum_finger_visual_dedup_points": finger_savings_points,
            "maximum_finger_visual_dedup_faces": finger_savings_faces,
            "maximum_finger_visual_dedup_point_percent": 100.0
            * finger_savings_points
            / visual_points,
            "maximum_finger_visual_dedup_face_percent": 100.0
            * finger_savings_faces
            / visual_faces,
            "recommendation": "DEFER_TINY_FINGER_VISUAL_INSTANCING; NEXT_MEANINGFUL_SCOPE_IS_ISOLATED_COLLIDER_COMPLEXITY_CANDIDATE",
        },
        "known_issue_policy": "NONBLOCKING_REMINDERS_RECALLED_ONLY_ON_MATCHING_FAILURE_OR_PROMOTION_REVIEW",
    }
    report["deterministic_signature"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["first_candidate"]
    boundary = report["next_optimization_boundary"]
    return "\n".join(
        [
            "# ALOHA1 Task 8 current summary",
            "",
            f"Status: `{report['status']}`",
            "",
            "Task 8 is active under the user-authorized approximation boundary. The first "
            "isolated visual-material candidate reduces effective visual material paths from "
            f"{candidate['effective_visual_materials_before']} to "
            f"{candidate['effective_visual_materials_after']} without changing the protected "
            "physics signature, but three fresh-process benchmark pairs show no measurable "
            "improvement and a nonoverlapping physics-frame regression. It is not promoted.",
            "",
            f"The baseline already contains {boundary['baseline_instanceable_prim_count']} "
            f"instanceable prims and {boundary['baseline_payload_prim_count']} payloads. Only "
            "four non-proxy supplier-finger visual meshes remain; their theoretical dedup upper "
            f"bound is {boundary['maximum_finger_visual_dedup_point_percent']:.3f}% of visual "
            f"points and {boundary['maximum_finger_visual_dedup_face_percent']:.3f}% of visual "
            "faces, so that path is deferred.",
            "",
            "The next meaningful Task 8 scope is an isolated collider-complexity candidate. "
            "Previously recorded physical/calibration gaps remain non-blocking reminders and "
            "are recalled only if a matching failure appears or during final/default promotion.",
            "",
        ]
    )


def main() -> int:
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

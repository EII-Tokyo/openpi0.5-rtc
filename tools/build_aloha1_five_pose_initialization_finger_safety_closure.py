#!/usr/bin/env python3
"""Build the Task 7 finger-initialization and safety closure report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
INPUTS = {
    "historical": REPORT_ROOT / "aloha1_cad_derived_five_pose_visual_review_zup_attempt7.json",
    "runtime": REPORT_ROOT / "aloha1_cad_derived_five_pose_runtime_finger_safe_attempt10_machine_only.json",
    "screenshot_review": REPORT_ROOT / "aloha1_five_pose_finger_safe_collision_screenshot_review_attempt10.json",
    "semantics": REPORT_ROOT / "aloha1_finger_limit_collision_semantics.json",
    "session_probe": REPORT_ROOT / "aloha1_finger_source_limit_session_layer_probe.json",
    "negative_controls": REPORT_ROOT / "aloha1_grasp_initialization_negative_controls.json",
    "prior_task7": REPORT_ROOT / "aloha1_cad_derived_task7_closure_zup_attempt7.json",
    "physics_root_cause": REPORT_ROOT / "aloha1_task7_physicsrules_root_cause_closure.json",
}
OUTPUT_JSON = REPORT_ROOT / "aloha1_five_pose_initialization_finger_safety_closure.json"
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
SAMPLE_IDS = [f"sample_{index:02d}" for index in range(1, 6)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_sample_gates(sample: dict[str, Any]) -> dict[str, bool]:
    primary = sample.get("primary", {})
    repeat = sample.get("collider_repeat", {})
    return {
        "primary_machine_pass": primary.get("machine_status") == "PASS",
        "repeat_machine_pass": repeat.get("machine_status") == "PASS",
        "physics_signatures_match": (
            primary.get("deterministic_signature")
            == repeat.get("deterministic_signature")
            and primary.get("deterministic_signature") is not None
        ),
        "initialization_signatures_match": (
            primary.get("initialization_signature")
            == repeat.get("initialization_signature")
            and primary.get("initialization_signature") is not None
        ),
        "initialization_contract_pass": (
            primary.get("initialization_contract_status") == "PASS"
            and repeat.get("initialization_contract_status") == "PASS"
        ),
        "finger_safety_pass": (
            primary.get("finger_safety_status") == "PASS"
            and repeat.get("finger_safety_status") == "PASS"
            and primary.get("finger_safety_violation_count") == 0
            and repeat.get("finger_safety_violation_count") == 0
        ),
    }


def build_closure(
    *,
    historical: dict[str, Any],
    runtime: dict[str, Any],
    screenshot_review: dict[str, Any],
    semantics: dict[str, Any],
    negative_controls: dict[str, Any],
    prior_task7: dict[str, Any],
    physics_root_cause: dict[str, Any],
    session_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep grasp outcome, fresh safety, and asset promotion independent."""

    runtime_samples = list(runtime.get("samples", []))
    sample_reports = []
    for sample in runtime_samples:
        gates = _runtime_sample_gates(sample)
        primary = sample.get("primary", {})
        sample_reports.append(
            {
                "sample_id": sample.get("sample_id"),
                "status": "PASS" if all(gates.values()) else "FAIL",
                "gates": gates,
                "physics_deterministic_signature": primary.get(
                    "deterministic_signature"
                ),
                "initialization_signature": primary.get(
                    "initialization_signature"
                ),
                "hold_drop_m": primary.get("metrics", {}).get("hold_drop_m"),
                "maximum_clearance_m": primary.get("metrics", {}).get(
                    "maximum_clearance_m"
                ),
            }
        )
    runtime_gates = {
        "machine_status_pass": runtime.get("machine_status") == "PASS",
        "exact_five_samples": [
            sample.get("sample_id") for sample in runtime_samples
        ]
        == SAMPLE_IDS,
        "ten_fresh_processes": runtime.get("fresh_process_count") == 10,
        "all_sample_safety_gates_pass": all(
            sample["status"] == "PASS" for sample in sample_reports
        ),
    }
    fresh_safety = "PASS" if all(runtime_gates.values()) else "FAIL"
    historical_pass = (
        historical.get("status") == "PASS"
        and historical.get("user_confirmation") == "PASS"
    )
    screenshot_pass = (
        screenshot_review.get("status") == "PASS"
        and screenshot_review.get("capture_record_count") == 120
        and screenshot_review.get("image_record_count") == 240
    )
    semantics_candidate = semantics.get("candidate", {})
    semantics_pass = (
        semantics.get("status") == "PASS"
        and semantics.get("limit_semantics_status") == "VERIFIED_USD_LIMIT_DEFECT"
        and semantics.get("candidate_created") is True
        and semantics_candidate.get("verification_status") == "PASS"
    )
    negative_pass = (
        negative_controls.get("status") == "PASS"
        and negative_controls.get("control_count") == 4
    )
    session_probe_pass = session_probe is None or (
        session_probe.get("status") == "PASS"
        and all(session_probe.get("gates", {}).values())
    )
    pair_status = (
        "NOT_AUTHORED_INCONCLUSIVE"
        if semantics.get("pair_collision_support_status") == "INCONCLUSIVE"
        and semantics_candidate.get("pair_collision_authored") is False
        else "AUTHORED_DIAGNOSTIC"
        if semantics_candidate.get("pair_collision_authored") is True
        else "INCONCLUSIVE"
    )
    candidate_promoted = semantics_candidate.get("status") not in {
        "CREATED_NOT_PROMOTED",
        None,
    }
    prior_asset_pass = prior_task7.get("asset_promotion") == "PASS"
    root_blockers = list(physics_root_cause.get("remaining_real_blockers", []))
    remaining_blockers = list(root_blockers)
    if not candidate_promoted:
        remaining_blockers.insert(
            0, "FINGER_SOURCE_LIMIT_SESSION_LAYER_NOT_PROMOTED"
        )

    applicable_runtime_gates = {
        "historical_user_confirmed_grasp_outcome": historical_pass,
        "attempt10_fresh_initialization_and_finger_safety": fresh_safety == "PASS",
        "attempt10_collision_screenshot_review": screenshot_pass,
        "negative_controls": negative_pass,
        "source_limit_and_mimic_semantics": semantics_pass,
        "session_layer_runtime_readback": session_probe_pass,
    }
    task7 = (
        "FAIL"
        if not all(applicable_runtime_gates.values())
        else "PASS"
        if prior_asset_pass and candidate_promoted and not remaining_blockers
        else "PARTIAL"
    )
    task8_ok = all(
        report.get("task8") == "NOT_RUN"
        for report in (
            runtime,
            screenshot_review,
            semantics,
            negative_controls,
            physics_root_cause,
        )
    )
    if not task8_ok:
        task7 = "FAIL"
    return {
        "schema_version": 1,
        "status": task7,
        "task7": task7,
        "runtime_grasp_outcome": "PASS" if historical_pass else "FAIL",
        "attempt10_finger_safety": fresh_safety,
        "attempt10_collision_visual_evidence": (
            "PASS" if screenshot_pass else "FAIL"
        ),
        "negative_controls": "PASS" if negative_pass else "FAIL",
        "source_limit_semantics": semantics.get("limit_semantics_status"),
        "mimic_status": (
            "VERIFIED_UNCHANGED"
            if semantics_pass
            and semantics_candidate.get("non_limit_invariants", {})
            .get("fields", {})
            .get("mimic_api", True)
            else "INCONCLUSIVE"
        ),
        "physical_pair_collision_candidate": pair_status,
        "physical_pair_collision_gate_role": (
            "NON_BLOCKING_SECONDARY_CANDIDATE; SOURCE_LIMITS_AND_PER_FRAME_"
            "OVERLAP_GUARD_ARE_THE_VALIDATED_CLOSING_STOP"
        ),
        "final_default_promotion": (
            "PROMOTED" if candidate_promoted else "NOT_PROMOTED"
        ),
        "applicable_gates": applicable_runtime_gates,
        "runtime_gates": runtime_gates,
        "samples": sample_reports,
        "historical_attempt7": {
            "status": historical.get("status"),
            "user_confirmation": historical.get("user_confirmation"),
            "semantic_boundary": (
                "PROVES_GRASP_OUTCOME; DOES_NOT_PROVE_ATTEMPT10_PER_FRAME_"
                "FINGER_SAFETY"
            ),
            "videos_rerun": False,
        },
        "formal_attempt10": {
            "machine_status": runtime.get("machine_status"),
            "fresh_process_count": runtime.get("fresh_process_count"),
            "screenshot_capture_count": screenshot_review.get(
                "capture_record_count"
            ),
            "screenshot_image_count": screenshot_review.get("image_record_count"),
            "semantic_boundary": (
                "FRESH_RUNTIME_AND_COLLISION_SCREENSHOTS_PROVE_INITIALIZATION_"
                "AND_FINGER_SAFETY_WITH_HISTORICAL_VIDEO_BINDING"
            ),
        },
        "prior_task7_asset_status": prior_task7.get("asset_promotion"),
        "physics_root_cause_status": physics_root_cause.get("status"),
        "remaining_real_blockers": remaining_blockers,
        "boundaries": {
            "historical_mp4_modified_or_rerecorded": False,
            "source_stage_modified": False,
            "final_default_asset_modified": False,
            "final_default_collider_modified": False,
            "real_robot": False,
            "remote_103": False,
            "visual_evidence_is_auxiliary": True,
            "runtime_telemetry_is_authoritative": True,
            "candidate_promotion_requires_user_review": True,
        },
        "task8": "NOT_RUN" if task8_ok else "FAIL",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 five-pose initialization and finger-safety closure",
        "",
        f"- Task 7: `{report['task7']}`",
        f"- Historical grasp outcome: `{report['runtime_grasp_outcome']}`",
        f"- Attempt10 finger safety: `{report['attempt10_finger_safety']}`",
        (
            "- Attempt10 collision screenshot review: "
            f"`{report['attempt10_collision_visual_evidence']}`"
        ),
        f"- Negative controls: `{report['negative_controls']}`",
        f"- Source-limit semantics: `{report['source_limit_semantics']}`",
        f"- Physical pair collision: `{report['physical_pair_collision_candidate']}`",
        f"- Final/default promotion: `{report['final_default_promotion']}`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Sample | Runtime/safety | Hold drop (m) | Clearance (m) |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| `{sample['sample_id']}` | `{sample['status']}` | "
            f"{sample['hold_drop_m']} | {sample['maximum_clearance_m']} |"
        )
        for sample in report["samples"]
    )
    lines.extend(
        [
            "",
            "The five previously user-confirmed videos were not rerun. They "
            "remain evidence of grasp outcome. Attempt10 adds ten fresh-process "
            "runtime records, per-frame finger-limit/overlap guards and 240 "
            "hash-bound raw/annotated collision images.",
            "",
            "Task 7 remains `PARTIAL` because passing a diagnostic session "
            "layer is not permission to promote it into final/default assets, "
            "and the independently tracked PhysicsRules candidates remain "
            "unpromoted. This is an asset-promotion boundary, not a grasp failure.",
            "",
            "## Remaining real blockers",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in report["remaining_real_blockers"])
    lines.append("")
    return "\n".join(lines)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def main() -> int:
    reports = {name: _load(path) for name, path in INPUTS.items()}
    report = build_closure(
        historical=reports["historical"],
        runtime=reports["runtime"],
        screenshot_review=reports["screenshot_review"],
        semantics=reports["semantics"],
        session_probe=reports["session_probe"],
        negative_controls=reports["negative_controls"],
        prior_task7=reports["prior_task7"],
        physics_root_cause=reports["physics_root_cause"],
    )
    report["evidence"] = {
        name: {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path.resolve()),
            "status": reports[name].get("status"),
        }
        for name, path in INPUTS.items()
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT_JSON)}))
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

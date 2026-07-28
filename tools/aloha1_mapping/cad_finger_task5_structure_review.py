"""Finalize the supplier-CAD Task 5 no-bottle screenshot review."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

REQUIRED_STATES = ("closed", "partial", "maximum_legal_aperture")
REQUIRED_VIEWS = ("true_top", "true_bottom", "tip_end", "base_oblique")
REQUIRED_CHECKS = (
    "both_handed_fingers_fully_visible",
    "blue_left_orange_right_mapping_correct",
    "inward_surfaces_opposed",
    "no_critical_crop",
    "no_critical_occlusion",
    "state_visually_distinct",
    "paired_camera_pose_exact",
    "labels_do_not_overlap",
    "annotations_do_not_cover_key_geometry",
    "visual_gate_only_wording",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(
    *,
    path_text: str,
    expected_sha256: str,
    expected_resolution: list[int],
) -> dict[str, Any]:
    path = Path(path_text).resolve(strict=True)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"image hash drift: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        resolution = [image.width, image.height]
        mode = image.mode
    if resolution != expected_resolution:
        raise RuntimeError(f"image resolution drift: {path}")
    return {
        "absolute_path": str(path),
        "sha256": actual_sha256,
        "resolution": resolution,
        "mode": mode,
        "readable": True,
    }


def _camera_signature(camera: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(camera["actual_position_world_m"]),
        tuple(camera["actual_orientation_wxyz"]),
        tuple(camera["target_world_m"]),
        tuple(camera["resolution"]),
    )


def build_review_report(
    *,
    raw_report: dict[str, Any],
    annotation_metadata: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    retake_history: list[dict[str, Any]],
    approved_source_stage: dict[str, Any],
) -> dict[str, Any]:
    """Build a visual PASS without hiding the preserved dynamics FAIL."""

    if raw_report["status"] != "FAIL":
        raise RuntimeError("expected preserved Task 5 dynamics FAIL")
    if raw_report["screenshot_manifest"]["status"] != "PASS":
        raise RuntimeError("raw screenshot manifest must pass")
    if raw_report["gates"]["post_step_drive_tracking"]:
        raise RuntimeError("dynamic drive tracking failure was not preserved")
    if raw_report["gates"]["physx_mimic_or_controller_coupling"]:
        raise RuntimeError("dynamic mimic/coupling failure was not preserved")
    if annotation_metadata["physics_report_status"] != "FAIL":
        raise RuntimeError("annotation metadata hid the physics failure")

    required = {
        f"{state}_{view}"
        for state in REQUIRED_STATES
        for view in REQUIRED_VIEWS
    }
    raw_by_name = {
        record["capture_name"]: record for record in raw_report["captures"]
    }
    annotated_by_name = {
        record["capture_name"]: record
        for record in annotation_metadata["captures"]
    }
    if set(raw_by_name) != required or set(annotated_by_name) != required:
        raise RuntimeError("expected exactly twelve named raw/annotated captures")
    if set(decisions) != required:
        raise RuntimeError("visual decisions must cover exactly twelve captures")

    records: list[dict[str, Any]] = []
    for name in sorted(required):
        raw = raw_by_name[name]
        annotated = annotated_by_name[name]
        decision = decisions[name]
        if decision.get("raw") != "PASS":
            raise RuntimeError(f"raw visual review is not PASS: {name}")
        if decision.get("annotated") != "PASS":
            raise RuntimeError(f"annotated visual review is not PASS: {name}")
        if decision.get("conclusion") != "PASS":
            raise RuntimeError(f"visual conclusion is not PASS: {name}")
        checks = decision.get("checks", {})
        missing_checks = [
            check for check in REQUIRED_CHECKS if checks.get(check) is not True
        ]
        if missing_checks:
            raise RuntimeError(
                f"visual checks did not pass for {name}: {missing_checks}"
            )

        raw_file = _verify_image(
            path_text=raw["absolute_path"],
            expected_sha256=raw["file_sha256"],
            expected_resolution=raw["resolution"],
        )
        annotated_file = _verify_image(
            path_text=annotated["annotated_absolute_path"],
            expected_sha256=annotated["annotated_sha256"],
            expected_resolution=annotated["annotated_resolution"],
        )
        if annotated["raw_sha256"] != raw["file_sha256"]:
            raise RuntimeError(f"annotation raw-source hash mismatch: {name}")
        if annotated["camera"] != raw["camera"]:
            raise RuntimeError(f"annotation camera metadata drift: {name}")
        if annotated["simulation"] != raw["simulation"]:
            raise RuntimeError(f"annotation simulation metadata drift: {name}")

        records.append(
            {
                "capture_name": name,
                "state": raw["simulation"]["state"],
                "view": raw["camera"]["view"],
                "target": "supplier-CAD handed finger installation",
                "part": "left/right finger and inward gripping surfaces",
                "phase": (
                    "fresh World reset; full legal pose reinjected; "
                    "no physics step"
                ),
                "acceptance_criteria": list(REQUIRED_CHECKS),
                "raw": raw_file,
                "annotated": annotated_file,
                "camera": raw["camera"],
                "simulation": raw["simulation"],
                "visual_model_review": {
                    "status": "PASS",
                    "raw": "PASS",
                    "annotated": "PASS",
                    "checks": checks,
                    "conclusion": decision["notes"],
                    "retake_reason": None,
                },
            }
        )

    camera_checks: dict[str, bool] = {}
    gap_checks: dict[str, bool] = {}
    for view in REQUIRED_VIEWS:
        view_records = [
            raw_by_name[f"{state}_{view}"] for state in REQUIRED_STATES
        ]
        camera_checks[view] = len(
            {
                _camera_signature(record["camera"])
                for record in view_records
            }
        ) == 1
        gaps = [
            float(record["simulation"]["surface_gap_m"])
            for record in view_records
        ]
        gap_checks[view] = gaps[0] < gaps[1] < gaps[2]

    protected_immutable = (
        raw_report["protected_hashes_before"]
        == raw_report["protected_hashes_after"]
    )
    source_immutable = (
        approved_source_stage["sha256_before"]
        == approved_source_stage["sha256_after"]
        == raw_report["protected_hashes_after"]["approved_source_stage"]
    )
    gates = {
        "capture_count": len(records) == 12,
        "raw_visual_model_review": True,
        "annotated_visual_model_review": True,
        "paired_camera_pose_exact": all(camera_checks.values()),
        "three_state_aperture_monotonic": all(gap_checks.values()),
        "approved_source_stage_immutable": source_immutable,
        "all_protected_assets_immutable": protected_immutable,
        "visual_gate_scope_explicit": True,
        "dynamic_failure_preserved": raw_report["status"] == "FAIL",
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "gate": "TASK5_NO_BOTTLE_STRUCTURE_SCREENSHOT_EVIDENCE_ONLY",
        "approved_source_stage": approved_source_stage,
        "diagnostic_stage": {
            "absolute_path": raw_report["stage_absolute_path"],
            "sha256": raw_report["protected_hashes_after"]["root_usd"],
        },
        "capture_count": len(records),
        "captures": records,
        "paired_state_checks": {
            view: {
                "camera_pose_exact_across_three_states": camera_checks[view],
                "surface_gap_strictly_increasing": gap_checks[view],
            }
            for view in REQUIRED_VIEWS
        },
        "retake_history": retake_history,
        "gates": gates,
        "separate_gate_status": {
            "screenshot_visual_evidence": status,
            "dynamic_drive_tracking": "FAIL",
            "mimic_or_controller_coupling": "FAIL",
            "bottle_contact_grasp": "NOT_RUN",
            "task8": "NOT_RUN",
        },
        "scope_boundaries": {
            "screenshot_pass_unblocks_bottle_test": False,
            "collision_geometry_acceptance": "NOT_RUN",
            "contact_acceptance": "NOT_RUN",
            "grasp_acceptance": "NOT_RUN",
            "source_stage_modified": False,
            "default_configuration_modified": False,
            "final_default_collider_modified": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the bounded human-readable screenshot review."""

    lines = [
        "# ALOHA ViperX CAD finger Task 5 screenshot review",
        "",
        f"- Screenshot evidence status: `{report['status']}`",
        "- Dynamic drive tracking: `FAIL`",
        "- Mimic/controller coupling: `FAIL`",
        "- Bottle/contact/grasp: `NOT_RUN`",
        "- Task 8: `NOT_RUN`",
        "- This PASS applies only to no-bottle structure screenshots.",
        "- It does not unblock bottle testing while the dynamics gate fails.",
        (
            "- Approved source Stage: "
            f"`{report['approved_source_stage']['absolute_path']}`"
        ),
        (
            "- Approved source SHA-256: "
            f"`{report['approved_source_stage']['sha256_after']}`"
        ),
        (
            "- Diagnostic Stage: "
            f"`{report['diagnostic_stage']['absolute_path']}`"
        ),
        (
            "- Diagnostic Stage SHA-256: "
            f"`{report['diagnostic_stage']['sha256']}`"
        ),
        "",
        "## Per-image visual-model review",
        "",
        "| Capture | Result | Raw | Annotated |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        (
            f"| `{record['capture_name']}` | "
            f"`{record['visual_model_review']['status']}` | "
            f"`{record['raw']['absolute_path']}` | "
            f"`{record['annotated']['absolute_path']}` |"
        )
        for record in report["captures"]
    )
    lines.extend(
        [
            "",
            "## Same-view state checks",
            "",
            "| View | Exact camera | Gap monotonic |",
            "|---|---:|---:|",
        ]
    )
    for view, checks in report["paired_state_checks"].items():
        lines.append(
            f"| `{view}` | "
            f"`{checks['camera_pose_exact_across_three_states']}` | "
            f"`{checks['surface_gap_strictly_increasing']}` |"
        )
    lines.extend(["", "## Retake history", ""])
    lines.extend(
        (
            f"- `{item['attempt']}`: `{item['status']}` — "
            f"{item['reason']}"
        )
        for item in report["retake_history"]
    )
    lines.extend(
        [
            "",
            "## Acceptance boundary",
            "",
            "Each raw and annotated image was reviewed individually with a "
            "vision model. The accepted images prove visibility, handed-color "
            "mapping, opposing inward surfaces, state distinction, exact "
            "same-view camera pairing, and annotation legibility. They do not "
            "prove collision clearance, drive tracking, mimic behavior, "
            "contact, force delivery, or bottle hold.",
            "",
        ]
    )
    return "\n".join(lines)

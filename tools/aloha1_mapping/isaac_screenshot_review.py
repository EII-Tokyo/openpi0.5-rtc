"""Finalize explicit visual-model review of Isaac CAD finger screenshots."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

REQUIRED_STATES = ("closed", "open")
REQUIRED_VIEWS = ("true_top", "true_bottom", "tip_end", "base_oblique")
REQUIRED_CHECKS = (
    "both_fingers_fully_visible",
    "blue_orange_mapping_correct",
    "inward_surfaces_opposed",
    "no_critical_crop",
    "no_critical_occlusion",
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
        tuple(camera["position_world_m"]),
        tuple(camera["orientation_wxyz"]),
        tuple(camera["target_world_m"]),
    )


def build_review_report(
    *,
    raw_report: dict[str, Any],
    annotation_metadata: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    retake_history: list[dict[str, Any]],
    approved_source_stage: dict[str, Any],
) -> dict[str, Any]:
    """Build a PASS report only from eight explicit visual PASS decisions."""
    if raw_report["status"] != "PASS":
        raise RuntimeError("raw machine report must pass")
    required = {
        f"{state}_{view}"
        for state in REQUIRED_STATES
        for view in REQUIRED_VIEWS
    }
    raw_by_name = {
        record["capture_name"]: record for record in raw_report["captures"]
    }
    ann_by_name = {
        record["capture_name"]
        for record in annotation_metadata["captures"]
    }
    if set(raw_by_name) != required or ann_by_name != required:
        raise RuntimeError("expected exactly eight named raw/annotated captures")
    if set(decisions) != required:
        raise RuntimeError("visual decisions must cover exactly eight captures")

    records: list[dict[str, Any]] = []
    annotated_by_name = {
        record["capture_name"]: record
        for record in annotation_metadata["captures"]
    }
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
        records.append(
            {
                "capture_name": name,
                "state": raw["simulation"]["state"],
                "view": raw["camera"]["view"],
                "target": "supplier CAD handed finger installation",
                "part": "left_finger and right_finger inward surfaces",
                "phase": "Isaac isolated diagnostic visual review",
                "acceptance_criteria": list(REQUIRED_CHECKS),
                "raw": raw_file,
                "annotated": annotated_file,
                "camera": raw["camera"],
                "simulation": raw["simulation"],
                "visual_self_review": {
                    "status": "PASS",
                    "raw": "PASS",
                    "annotated": "PASS",
                    "checks": checks,
                    "conclusion": decision.get("notes", "PASS"),
                    "retake_reason": decision.get("retake_reason"),
                },
            }
        )

    paired_camera = {}
    distinct_states = {}
    for view in REQUIRED_VIEWS:
        closed = raw_by_name[f"closed_{view}"]
        opened = raw_by_name[f"open_{view}"]
        same_camera = _camera_signature(
            closed["camera"]
        ) == _camera_signature(opened["camera"])
        closed_gap = float(closed["simulation"]["surface_gap_m"])
        open_gap = float(opened["simulation"]["surface_gap_m"])
        paired_camera[view] = same_camera
        distinct_states[view] = open_gap > closed_gap

    source_immutable = (
        approved_source_stage["sha256_before"]
        == approved_source_stage["sha256_after"]
    )
    diagnostic_immutable = (
        raw_report["stage_sha256_before"]
        == raw_report["stage_sha256_after"]
    )
    gates = {
        "capture_count": len(records) == 8,
        "raw_visual_review": True,
        "annotated_visual_review": True,
        "paired_camera_pose_exact": all(paired_camera.values()),
        "open_closed_visually_distinct": all(distinct_states.values()),
        "approved_source_stage_immutable": source_immutable,
        "diagnostic_stage_immutable": diagnostic_immutable,
        "visual_gate_scope_explicit": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "gate": "ISAAC_CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY",
        "approved_source_stage": approved_source_stage,
        "diagnostic_stage": {
            "absolute_path": raw_report["stage_absolute_path"],
            "sha256_before": raw_report["stage_sha256_before"],
            "sha256_after": raw_report["stage_sha256_after"],
        },
        "capture_count": len(records),
        "captures": records,
        "open_closed_pair_checks": {
            view: {
                "camera_pose_exact": paired_camera[view],
                "open_surface_gap_exceeds_closed": distinct_states[view],
            }
            for view in REQUIRED_VIEWS
        },
        "retake_history": retake_history,
        "gates": gates,
        "scope_boundaries": {
            "physics_acceptance": "NOT_RUN",
            "collision_acceptance": "NOT_RUN",
            "contact_acceptance": "NOT_RUN",
            "grasp_acceptance": "NOT_RUN",
            "final_default_asset_modified": False,
            "final_default_collider_modified": False,
            "task_8": "NOT_RUN",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a bounded human-readable companion for a review report."""
    lines = [
        "# ALOHA ViperX supplier-CAD finger Isaac screenshot review",
        "",
        f"- Status: `{report['status']}`",
        f"- Gate: `{report['gate']}`",
        "- Scope: CAD installation visual evidence only.",
        "- Boundary: **NO collision/contact/grasp acceptance**.",
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
            f"`{report['diagnostic_stage']['sha256_after']}`"
        ),
        "",
        "## Capture review",
        "",
        "| Capture | Verdict | Raw | Annotated |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        (
            "| "
            f"`{record['capture_name']}` | "
            f"`{record['visual_self_review']['status']}` | "
            f"`{record['raw']['absolute_path']}` | "
            f"`{record['annotated']['absolute_path']}` |"
        )
        for record in report["captures"]
    )
    lines.extend(
        [
            "",
            "## Open/closed paired-camera gates",
            "",
            "| View | Camera exact | Open gap > closed gap |",
            "|---|---:|---:|",
        ]
    )
    for view, checks in report["open_closed_pair_checks"].items():
        lines.append(
            f"| `{view}` | `{checks['camera_pose_exact']}` | "
            f"`{checks['open_surface_gap_exceeds_closed']}` |"
        )
    lines.extend(
        [
            "",
            "## Retake history",
            "",
        ]
    )
    lines.extend(
        (
            f"- `{record['attempt']}`: `{record['status']}` — "
            f"{record.get('reason', record.get('disposition', ''))}"
        )
        for record in report["retake_history"]
    )
    lines.extend(
        [
            "",
            "## Acceptance boundary",
            "",
            "This PASS proves only that the isolated supplier-CAD visual "
            "installation is consistently presented from four paired views. "
            "It does not validate collider geometry, contact, dynamics, or "
            "bottle grasping. Task 8 remains `NOT_RUN`.",
            "",
        ]
    )
    return "\n".join(lines)

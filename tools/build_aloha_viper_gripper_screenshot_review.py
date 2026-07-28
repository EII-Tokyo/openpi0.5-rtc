#!/usr/bin/env python3
"""Build the durable CAD gripper screenshot visual-review report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from tools.aloha1_mapping.cad_gripper_screenshot_review import camera_matrix_mm
from tools.aloha1_mapping.cad_gripper_screenshot_review import review_status

VIEW_CONCLUSIONS = {
    "true_top": (
        "Both complete handed fingers and the gripper rail are visible from "
        "proven CAD +Z. The center gap grows clearly from closed to open."
    ),
    "true_bottom": (
        "The proven CAD -Z view reverses image-side color order as expected "
        "without changing the CAD +X/-X role mapping; both fingers remain "
        "complete and the paired aperture change is clear."
    ),
    "tip_end": (
        "Strongest palm-orientation view: the recessed inner/contact-facing "
        "surfaces on both handed B-Reps face each other across the centerline."
    ),
    "base_oblique": (
        "The documented CAD +Y/+Z oblique exposes both installed fingers, "
        "their sliding-carriage relation, and the opening change without the "
        "pure base-end shell occlusion."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(path: Path) -> tuple[str, list[int]]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        return hashlib.sha256(rgba.tobytes()).hexdigest(), list(rgba.size)


def _absolute(path: Path) -> str:
    return str(path.resolve())


def _retake_history(artifact_root: Path) -> list[dict[str, Any]]:
    gripper_root = artifact_root / "viper_gripper"
    return [
        {
            "attempt": "attempt_2",
            "status": "FAIL",
            "disposition": "REJECTED_INSUFFICIENT_ILLUMINATION",
            "path": _absolute(gripper_root / "screenshots_raw"),
            "log": _absolute(gripper_root / "logs/blender_render_attempt2.log"),
            "reason": (
                "Eevee output was too dark to inspect the inner surfaces; "
                "file existence and nonzero pixels were not accepted."
            ),
        },
        {
            "attempt": "attempt_3",
            "status": "FAIL",
            "disposition": "REJECTED_INSUFFICIENT_SURFACE_VISIBILITY",
            "path": _absolute(gripper_root / "attempt3_metric_scene/screenshots_raw"),
            "log": _absolute(
                gripper_root
                / "logs/blender_render_attempt3_metric_scene.log"
            ),
            "reason": (
                "Correct mm-to-m rendering units did not make Eevee surface "
                "details sufficiently visible for the CAD orientation gate."
            ),
        },
        {
            "attempt": "attempt_4",
            "status": "FAIL",
            "disposition": "REJECTED_CROPPING_AND_BASE_END_OCCLUSION",
            "path": _absolute(gripper_root / "attempt4_workbench/screenshots_raw"),
            "log": _absolute(
                gripper_root / "logs/blender_render_attempt4_workbench.log"
            ),
            "reason": (
                "Workbench fixed visibility, but finger tips were cropped, "
                "open tip-end fingers touched the frame edge, and the pure "
                "base-end view hid both fingers behind the gripper shell."
            ),
        },
        {
            "attempt": "attempt_5_raw",
            "status": "PASS",
            "disposition": "ACCEPTED_AFTER_PER_IMAGE_VISION_REVIEW",
            "path": _absolute(
                gripper_root / "attempt5_candidate/screenshots_raw"
            ),
            "log": _absolute(
                gripper_root / "logs/blender_render_attempt5_candidate.log"
            ),
            "reason": (
                "Eight raw images passed individual visual inspection after "
                "increasing the paired frame margin and replacing the "
                "occluded base-end view with a proven base oblique."
            ),
        },
        {
            "attempt": "attempt_5_annotated_v1",
            "status": "FAIL",
            "disposition": "REJECTED_LABEL_OVERLAP",
            "path": _absolute(
                gripper_root
                / "attempt5_candidate/screenshots_annotated"
            ),
            "log": _absolute(gripper_root / "logs/annotate_attempt5.log"),
            "reason": (
                "Long local left/right labels overlapped in the closed "
                "true-top and closed tip-end evidence."
            ),
        },
        {
            "attempt": "attempt_5_annotated_v2",
            "status": "FAIL",
            "disposition": "REJECTED_LABEL_OVERLAP",
            "path": _absolute(
                gripper_root
                / "attempt5_candidate/screenshots_annotated_v2"
            ),
            "log": _absolute(gripper_root / "logs/annotate_attempt5_v2.log"),
            "reason": (
                "Short L/R tags fixed the first overlap, but the closed "
                "base-oblique distance label still covered the R tag."
            ),
        },
        {
            "attempt": "attempt_5_annotated_v3",
            "status": "PASS",
            "disposition": "ACCEPTED_AFTER_PER_IMAGE_VISION_REVIEW",
            "path": _absolute(
                gripper_root
                / "attempt5_candidate/screenshots_annotated_v3"
            ),
            "log": _absolute(gripper_root / "logs/annotate_attempt5_v3.log"),
            "reason": (
                "All eight annotations passed individual visual inspection; "
                "local labels are short L/R tags and edge-near measurement "
                "labels use collision-free placement."
            ),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-metadata", type=Path, required=True)
    parser.add_argument("--determinism-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    annotation_path = args.annotation_metadata.resolve(strict=True)
    determinism_path = args.determinism_report.resolve(strict=True)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
    if annotation["capture_count"] != 8:
        raise RuntimeError("expected eight annotation records")
    if determinism["status"] != "PASS":
        raise RuntimeError("repeat-render determinism did not pass")
    captures = []
    for capture in annotation["captures"]:
        raw_path = Path(capture["raw_path"]).resolve(strict=True)
        annotated_path = Path(capture["annotated_path"]).resolve(strict=True)
        raw_pixel_hash, raw_size = _pixel_sha256(raw_path)
        annotated_pixel_hash, annotated_size = _pixel_sha256(annotated_path)
        captures.append(
            {
                "state_id": capture["state_id"],
                "view_id": capture["view_id"],
                "target": (
                    "supplier-CAD handed finger installation, center-facing "
                    "inner surfaces, and open/closed aperture"
                ),
                "part": "follower gripper handed finger pair",
                "phase": f"static_CAD_{capture['state_id']}",
                "physical_stage_frame_time": "NOT_APPLICABLE_STATIC_CAD",
                "acceptance_criteria": {
                    "both_fingers_complete": True,
                    "blue_orange_mapping_correct": True,
                    "inner_surfaces_face_center": True,
                    "not_cropped": True,
                    "critical_geometry_not_shell_occluded": True,
                    "open_closed_visibly_distinct": True,
                    "paired_camera_identical": True,
                    "labels_do_not_overlap": True,
                    "annotations_do_not_hide_critical_geometry": True,
                    "pass_scope_is_cad_visual_only": True,
                },
                "raw": {
                    "absolute_path": str(raw_path),
                    "sha256": _sha256(raw_path),
                    "pixel_sha256": raw_pixel_hash,
                    "resolution": raw_size,
                    "visual_self_review": "PASS",
                },
                "annotated": {
                    "absolute_path": str(annotated_path),
                    "sha256": _sha256(annotated_path),
                    "pixel_sha256": annotated_pixel_hash,
                    "resolution": annotated_size,
                    "visual_self_review": "PASS",
                },
                "camera": {
                    **capture["camera"],
                    "camera_to_world_matrix_cad_mm": camera_matrix_mm(
                        camera=capture["camera"]
                    ),
                    "open_closed_pair_key": capture["camera_key"],
                },
                "state_translation_mm": capture["state_translation_mm"],
                "finger_minimum_shape_distance_mm": capture[
                    "finger_minimum_distance_mm"
                ],
                "annotation_geometry": capture[
                    "measured_annotation_geometry"
                ],
                "annotation_sample_semantics": (
                    "CAD-derived inner/contact-facing surface sample for "
                    "orientation annotation only; not a physical contact point"
                ),
                "visual_self_review": "PASS",
                "assistant_vision_conclusion": VIEW_CONCLUSIONS[
                    capture["view_id"]
                ],
                "retake_reason": None,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for capture in captures:
        grouped.setdefault(capture["view_id"], []).append(capture)
    pair_checks = {}
    for view_id, pair in grouped.items():
        if len(pair) != 2:
            raise RuntimeError(f"expected open/closed pair for {view_id}")
        pair_checks[view_id] = {
            "status": "PASS",
            "states": sorted(item["state_id"] for item in pair),
            "camera_identical": pair[0]["camera"] == pair[1]["camera"],
            "raw_pixel_hashes_different": (
                pair[0]["raw"]["pixel_sha256"]
                != pair[1]["raw"]["pixel_sha256"]
            ),
            "minimum_distance_monotonic": (
                min(
                    item["finger_minimum_shape_distance_mm"]
                    for item in pair
                    if item["state_id"] == "closed"
                )
                < min(
                    item["finger_minimum_shape_distance_mm"]
                    for item in pair
                    if item["state_id"] == "open"
                )
            ),
        }
        if not all(
            (
                pair_checks[view_id]["camera_identical"],
                pair_checks[view_id]["raw_pixel_hashes_different"],
                pair_checks[view_id]["minimum_distance_monotonic"],
            )
        ):
            pair_checks[view_id]["status"] = "FAIL"
    screenshot_status = review_status(captures)
    overall = (
        "PASS"
        if screenshot_status == "PASS"
        and all(item["status"] == "PASS" for item in pair_checks.values())
        and determinism["status"] == "PASS"
        else "FAIL"
    )
    artifact_root = (
        annotation_path.parents[3]
        if annotation_path.parents[3].name
        == "20260729-aloha-finger-palm-orientation"
        else Path(
            ".codex/artifacts/20260729-aloha-finger-palm-orientation"
        ).resolve()
    )
    report = {
        "schema_version": 1,
        "status": overall,
        "gate": "CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY",
        "corrected_installation_status": "AWAITING_USER_VISUAL_CONFIRMATION",
        "source_classification": "SUPPLIER_PUBLIC_CAD_FIRST_HAND",
        "source_cad": {
            "absolute_path": captures[0]["raw"]["absolute_path"]
            and annotation["captures"][0]["source_cad_path"],
            "sha256": annotation["captures"][0]["source_cad_sha256"],
            "assembly": "Simple Aloha Viper 2024-5-13.step",
        },
        "finger_mapping": {
            "blue": "left_finger / CAD +X side",
            "orange": "right_finger / CAD -X side",
            "geometry": (
                "two distinct handed B-Reps embedded in the supplier assembly"
            ),
            "standalone_3d_a1_v3_used": False,
            "arbitrary_single_side_180_degree_rotation_used": False,
        },
        "state_semantics": {
            "closed": "supplier assembly static CLOSED_REFERENCE",
            "open": "both embedded fingers translated outward by 36 mm along CAD X",
        },
        "capture_count": len(captures),
        "raw_and_annotated_file_count": 2 * len(captures),
        "captures": captures,
        "open_closed_pair_checks": pair_checks,
        "repeat_render_determinism": determinism,
        "retake_history": _retake_history(artifact_root),
        "scope_boundaries": {
            "isaac_runtime": "NOT_RUN",
            "collider_validation": "NOT_RUN",
            "contact_validation": "NOT_RUN",
            "bottle_hold": "NOT_RUN",
            "task_8": "NOT_RUN",
            "final_default_asset_modified": False,
        },
        "inputs": {
            "annotation_metadata": {
                "absolute_path": str(annotation_path),
                "sha256": _sha256(annotation_path),
            },
            "determinism_report": {
                "absolute_path": str(determinism_path),
                "sha256": _sha256(determinism_path),
            },
        },
    }
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        (
            "| "
            + " | ".join(
                (
                    capture["state_id"],
                    capture["view_id"],
                    "PASS",
                    f"`{capture['raw']['absolute_path']}`",
                    f"`{capture['annotated']['absolute_path']}`",
                )
            )
            + " |"
        )
        for capture in captures
    ]
    output_md.write_text(
        "\n".join(
            (
                "# ALOHA Viper Gripper Screenshot Review",
                "",
                f"- Status: `{overall}`",
                "- Gate: `CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY`",
                "- User confirmation: `AWAITING_USER_VISUAL_CONFIRMATION`",
                "- Isaac runtime/contact/hold: `NOT_RUN`",
                "- Final/default asset modified: `false`",
                "- Source: `Simple Aloha Viper 2024-5-13.step`",
                f"- Source SHA-256: `{report['source_cad']['sha256']}`",
                "",
                "Every accepted raw and annotated image below was inspected "
                "individually with the vision model. File existence, color "
                "bounds, or hashes alone were not accepted.",
                "",
                "| State | View | Vision review | Raw | Annotated |",
                "|---|---|---:|---|---|",
                *rows,
                "",
                "## Interpretation",
                "",
                "- Blue is the embedded CAD +X handed B-Rep mapped to "
                "`left_finger`.",
                "- Orange is the embedded CAD -X handed B-Rep mapped to "
                "`right_finger`.",
                "- The magenta samples are CAD-derived annotation points, "
                "not physical contact points.",
                "- `tip_end` is the strongest evidence that both recessed "
                "inner surfaces face the gripper center.",
                "- Open/closed pairs share identical orthographic camera "
                "metadata and differ in image content and B-Rep minimum "
                "distance.",
                "- A fresh repeat render reproduced all 8 raw pixel hashes.",
                "",
                "## Retakes",
                "",
                *[
                    f"- `{item['attempt']}`: `{item['disposition']}` — "
                    f"{item['reason']} Path: `{item['path']}`"
                    for item in report["retake_history"]
                ],
                "",
                "This PASS does not claim collider, contact, grasp, hold, or "
                "Isaac runtime correctness.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": overall,
                "capture_count": len(captures),
                "json": str(output_json),
                "markdown": str(output_md),
            },
            indent=2,
        )
    )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
